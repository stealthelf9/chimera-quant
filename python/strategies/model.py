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
        
        # Initialize model (Assuming input features: Open, High, Low, Close, Volume)
        self.model = ChimeraNet(input_size=5).to(self.device)
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
        # Extract features natively
        features = np.column_stack((
            view['open'],
            view['high'],
            view['low'],
            view['close'],
            view['volume']
        )).astype(np.float32)

        # Standardize features (Z-Score Normalization)
        self.feature_mean = np.mean(features, axis=0)
        self.feature_std = np.std(features, axis=0)
        self.feature_std[self.feature_std == 0] = 1.0 # Prevent div by zero
        features = (features - self.feature_mean) / self.feature_std

        X, y = [], []
        for i in range(len(features) - self.window_size):
            X.append(features[i:i+self.window_size])
            # Target prediction is the percentage return of the next tick close vs current
            curr_close = view['close'][i + self.window_size - 1]
            next_close = view['close'][i + self.window_size]
            pct_return = (next_close - curr_close) / curr_close if curr_close != 0 else 0
            y.append([pct_return])
            
        X = torch.tensor(np.array(X)).to(self.device)
        y = torch.tensor(np.array(y)).to(self.device)
        
        dataset = torch.utils.data.TensorDataset(X, y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        self.model.train()
        print(f"[{self.name}] Starting Training (Device: {self.device})")
        for epoch in range(epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"[{self.name}] Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")

        self.model.eval()
        print(f"[{self.name}] Training Complete.")

    def evaluate(self):
        # We need at least window_size ticks to make a prediction
        view = self.buffer.get_buffer_view()
        if len(view) < self.window_size:
            return
        
        # Extract features (Zero-copy into NumPy, then onto PyTorch Tensor)
        # Note: PyTorch requires a copy when moving from CPU to GPU
        recent_data = view[-self.window_size:]
        
        # Shape: (window_size, 5)
        features = np.column_stack((
            recent_data['open'],
            recent_data['high'],
            recent_data['low'],
            recent_data['close'],
            recent_data['volume']
        )).astype(np.float32)

        if hasattr(self, 'feature_mean') and hasattr(self, 'feature_std'):
            features = (features - self.feature_mean) / self.feature_std

        # Reshape for LSTM: (batch_size, sequence_length, input_size)
        x_tensor = torch.tensor(features).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            prediction = self.model(x_tensor)
            
        pred_return = prediction.item()
        
        # Basic logic: Returns predicted natively
        if pred_return > 0.001: # Expected > 0.1% gain
            return 1 # BUY
        elif pred_return < -0.001: # Expected < -0.1% loss
            return -1 # SELL
        return 0 # HOLD
