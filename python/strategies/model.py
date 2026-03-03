import torch
import torch.nn as nn
import numpy as np
import chimera_core
from python.strategies.base import BaseStrategy

class ChimeraNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2):
        super(ChimeraNet, self).__init__()
        # LSTM for temporal series prediction
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

class AIStrategy(BaseStrategy):
    """
    An interface for the PyTorch Neural Network (using ROCm 7.1.0) architecture
    capable of performing temporal predictions.
    """
    def __init__(self, name: str, params: dict = None):
        super().__init__(name, params)
        # Verify ROCm availability
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[{name}] Initializing AI Strategy on device: {self.device}")
        
        # Initialize model (Assuming input features: Returns, Volume, RSI, MACD Hist)
        self.model = ChimeraNet(input_size=4).to(self.device)
        if hasattr(torch, 'compile'):
            self.model = torch.compile(self.model)
        self.model.eval()
        
        # Local state
        self.window_size = self.params.get('window_size', 60) # 60 minutes lookback

    def train(self, epochs: int = 5, batch_size: int = 64, learning_rate: float = 0.001):
        """
        Trains the PyTorch model natively on the structured C++ zero-copy buffer.
        """
        view = self.buffer.get_buffer_view()
        if len(view) < self.window_size + 1:
            print(f"[{self.name}] Not enough data to train. Need {self.window_size + 1}, got {len(view)}")
            return
            
        print(f"[{self.name}] Preparing training data from {len(view)} ticks...")
        from python.strategies.indicators import Indicators
        
        closes = view['close']
        returns = np.zeros_like(closes, dtype=np.float32)
        with np.errstate(divide='ignore', invalid='ignore'):
            returns[1:] = np.where(closes[:-1] < 1e-8, 0.0, np.diff(closes) / closes[:-1])
            
        rsi = Indicators.rsi(self.buffer, timeperiod=14)
        _, _, macdhist = Indicators.macd(self.buffer)
        
        # Extract features natively
        features = np.column_stack((
            returns,
            view['volume'],
            rsi,
            macdhist
        )).astype(np.float32)

        # Standardize features (Z-Score Normalization)
        if not hasattr(self, 'feature_mean') or self.feature_mean is None or not hasattr(self, 'feature_std') or self.feature_std is None:
            self.feature_mean = np.nanmean(features, axis=0, dtype=np.float64)
            self.feature_std = np.nanstd(features, axis=0, dtype=np.float64)
            self.feature_std[self.feature_std == 0] = 1.0 # Prevent div by zero
        features = ((features - self.feature_mean) / self.feature_std).astype(np.float32)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        # 1. Vectorized sliding window for X
        X = np.lib.stride_tricks.sliding_window_view(features[:-1], (self.window_size, 4)).squeeze(axis=1)

        # 2. Vectorized target for y (Percentage Return)
        closes = view['close'].astype(np.float32)
        curr_closes = closes[self.window_size - 1 : -1]
        next_closes = closes[self.window_size :]
        
        with np.errstate(divide='ignore', invalid='ignore'):
            pct_returns = np.where(curr_closes < 1e-8, 0.0, ((next_closes - curr_closes) / curr_closes) * 100.0)
            
        y = np.clip(pct_returns, -10.0, 10.0).reshape(-1, 1).astype(np.float32)

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            # VRAM Pre-Loading: Directly cast and move full tensors to GPU to bypass CPU-bound DataLoaders
            X_tensor = torch.from_numpy(X).to(self.device)
            y_tensor = torch.from_numpy(y).to(self.device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # AMP Scaler
        scaler = torch.amp.GradScaler('cuda') if self.device.type == 'cuda' else None
        
        self.model.train()
        print(f"[{self.name}] Starting Training (Device: {self.device})")
        
        num_samples = X_tensor.size(0)
        total_batches = (num_samples + batch_size - 1) // batch_size
        
        best_loss = float('inf')
        patience = 5
        patience_counter = 0

        import time
        import os
        weights_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "weights")
        os.makedirs(weights_dir, exist_ok=True)
        checkpoint_path = os.path.join(weights_dir, f"chimeranet_checkpoint.pt")

        for epoch in range(epochs):
            epoch_start_time = time.time()
            total_loss = 0
            
            # Manual batch slicing executing directly natively on VRAM
            for i in range(0, num_samples, batch_size):
                batch_X = X_tensor[i : i + batch_size]
                batch_y = y_tensor[i : i + batch_size]
                
                optimizer.zero_grad()
                
                if self.device.type == 'cuda':
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        outputs = self.model(batch_X)
                        loss = criterion(outputs, batch_y)
                    
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    
                total_loss += loss.item()
                
            epoch_elapsed = time.time() - epoch_start_time
            avg_loss = total_loss / total_batches
            print(f"[{self.name}] Epoch {epoch+1}/{epochs} - Average Loss: {avg_loss:.4f} ({epoch_elapsed:.2f}s)")
            
            # Early Stopping Verification
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), checkpoint_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"[{self.name}] Early Stopping Triggered at Epoch {epoch+1}")
                    break

        self.model.eval()
        print(f"[{self.name}] Training Complete. Best Loss: {best_loss:.4f}")

    def evaluate(self):
        # We need at least window_size ticks to make a prediction
        view = self.buffer.get_buffer_view()
        if len(view) < self.window_size:
            return
        
        from python.strategies.indicators import Indicators
        
        closes = view['close']
        returns = np.zeros_like(closes, dtype=np.float32)
        with np.errstate(divide='ignore', invalid='ignore'):
            returns[1:] = np.where(closes[:-1] < 1e-8, 0.0, np.diff(closes) / closes[:-1])
            
        rsi = Indicators.rsi(self.buffer, timeperiod=14)
        _, _, macdhist = Indicators.macd(self.buffer)
        
        features = np.column_stack((
            returns,
            view['volume'],
            rsi,
            macdhist
        )).astype(np.float32)

        if hasattr(self, 'feature_mean') and hasattr(self, 'feature_std'):
            features = (features - self.feature_mean) / self.feature_std

        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        recent_features = features[-self.window_size:]

        # Reshape for LSTM: (batch_size, sequence_length, input_size)
        x_tensor = torch.tensor(recent_features, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            prediction = self.model(x_tensor)
            
        pred_return = prediction.item()
        
        # Basic logic: Returns predicted natively
        if pred_return > 0.05: # Expected > 0.05% gain
            return 1 # BUY
        elif pred_return < -0.05: # Expected < -0.05% loss
            return -1 # SELL
        return 0 # HOLD
