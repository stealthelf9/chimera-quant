import torch
import torch.nn as nn
import numpy as np
import chimera_core
from python.strategies.base import BaseStrategy

class ChimeraNet(nn.Module):
    """
    Lightweight Transformer Encoder + DQN Head for Reinforcement Learning.
    Input shape: (batch_size, sequence_length, features)
    Output shape: (batch_size, action_space) -> Q-values for (Hold, Buy, Sell)
    """
    def __init__(self, input_size: int = 5, d_model: int = 64, nhead: int = 4, num_layers: int = 2, action_space: int = 3):
        super(ChimeraNet, self).__init__()

        # Linear layer to project input features to d_model size
        self.input_projection = nn.Linear(input_size, d_model)

        # Positional Encoding to inject sequence order info
        self.max_len = 1000
        pe = torch.zeros(self.max_len, d_model)
        position = torch.arange(0, self.max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # DQN Action Head (Hold=0, Buy=1, Sell=2)
        self.fc = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, action_space)
        )

    def forward(self, x):
        # x is (batch_size, seq_len, features)
        x = self.input_projection(x)  # -> (batch_size, seq_len, d_model)
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        out = self.transformer(x)     # -> (batch_size, seq_len, d_model)
        # Take the output of the last time step for DQN
        out = self.fc(out[:, -1, :])  # -> (batch_size, action_space)
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
        
        # Initialize model (Assuming input features: Returns, Volume, RSI, MACD Hist, Sentiment)
        self.model = ChimeraNet(input_size=5).to(self.device)
        self.target_model = ChimeraNet(input_size=5).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

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
        
        # Integrate Sentiment logic. In a real system we fetch from cache.
        # For simplicity and backtest speed here, we pass a neutral sentiment (0)
        # However the network architecture is now built to accept it (5 features).
        sentiment = np.zeros_like(returns)

        # Extract features natively
        features = np.column_stack((
            returns,
            view['volume'],
            rsi,
            macdhist,
            sentiment
        )).astype(np.float32)

        # Standardize features (Z-Score Normalization)
        if not hasattr(self, 'feature_mean') or self.feature_mean is None or not hasattr(self, 'feature_std') or self.feature_std is None:
            self.feature_mean = np.nanmean(features, axis=0, dtype=np.float64)
            self.feature_std = np.nanstd(features, axis=0, dtype=np.float64)
            self.feature_std[self.feature_std == 0] = 1.0 # Prevent div by zero
        features = ((features - self.feature_mean) / self.feature_std).astype(np.float32)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        # 1. Vectorized sliding window for X
        X = np.lib.stride_tricks.sliding_window_view(features[:-1], (self.window_size, 5)).squeeze(axis=1)

        # 2. Extract forward returns to compute rewards for RL (Hold, Buy, Sell)
        closes = view['close'].astype(np.float32)
        curr_closes = closes[self.window_size - 1 : -1]
        next_closes = closes[self.window_size :]
        
        with np.errstate(divide='ignore', invalid='ignore'):
            pct_returns = np.where(curr_closes < 1e-8, 0.0, ((next_closes - curr_closes) / curr_closes) * 100.0)
            
        returns_fwd = np.clip(pct_returns, -10.0, 10.0)

        # For Offline DQN, we will generate a mock replay buffer from historical states.
        # Action space: 0=Hold, 1=Buy, 2=Sell
        # Reward function:
        # Action Buy (1): +return
        # Action Sell (2): -return
        # Action Hold (0): 0

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            states = torch.from_numpy(X).to(self.device)
            next_states = torch.cat((states[1:], states[-1:])) # approximate next state
            rewards_fwd = torch.from_numpy(returns_fwd).to(self.device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # AMP Scaler
        scaler = torch.amp.GradScaler('cuda') if self.device.type == 'cuda' else None
        
        self.model.train()
        print(f"[{self.name}] Starting Training Offline DQN (Device: {self.device})")
        
        num_samples = states.size(0)
        total_batches = (num_samples + batch_size - 1) // batch_size
        
        best_loss = float('inf')
        patience = 5
        patience_counter = 0
        gamma = 0.99

        import time
        import os
        weights_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "weights")
        os.makedirs(weights_dir, exist_ok=True)
        checkpoint_path = os.path.join(weights_dir, f"chimeranet_checkpoint.pt")

        for epoch in range(epochs):
            epoch_start_time = time.time()
            total_loss = 0
            
            # Shuffle data indices for offline RL
            indices = torch.randperm(num_samples)
            states = states[indices]
            next_states = next_states[indices]
            rewards_fwd = rewards_fwd[indices]

            for i in range(0, num_samples, batch_size):
                b_states = states[i : i + batch_size]
                b_next_states = next_states[i : i + batch_size]
                b_returns = rewards_fwd[i : i + batch_size]
                
                optimizer.zero_grad()
                
                if self.device.type == 'cuda':
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        # Current Q Values
                        q_values = self.model(b_states)

                        # Simulate max reward logic for offline DQN using known future
                        # For each state, the best action retrospectively:
                        # If b_returns > 0: Best is Buy (1). Reward is b_returns.
                        # If b_returns < 0: Best is Sell (2). Reward is -b_returns.
                        # If abs(b_returns) is small: Best is Hold (0). Reward is 0.

                        # Compute Target Q Values across ALL actions to properly train the full DQN layer
                        with torch.no_grad():
                            next_q_values = self.target_model(b_next_states)
                            max_next_q, _ = next_q_values.max(1)

                            # Expand target to shape (batch_size, 3)
                            # Action 0 (Hold): reward = 0
                            # Action 1 (Buy): reward = b_returns
                            # Action 2 (Sell): reward = -b_returns
                            target_q_all = torch.zeros_like(q_values)
                            target_q_all[:, 0] = 0.0 + gamma * max_next_q
                            target_q_all[:, 1] = b_returns + gamma * max_next_q
                            target_q_all[:, 2] = -b_returns + gamma * max_next_q

                        loss = criterion(q_values, target_q_all)
                    
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    q_values = self.model(b_states)
                    with torch.no_grad():
                        next_q_values = self.target_model(b_next_states)
                        max_next_q, _ = next_q_values.max(1)
                        target_q_all = torch.zeros_like(q_values)
                        target_q_all[:, 0] = 0.0 + gamma * max_next_q
                        target_q_all[:, 1] = b_returns + gamma * max_next_q
                        target_q_all[:, 2] = -b_returns + gamma * max_next_q

                    loss = criterion(q_values, target_q_all)
                    loss.backward()
                    optimizer.step()
                    
                total_loss += loss.item()
                
            # Update target model
            if epoch % 5 == 0:
                self.target_model.load_state_dict(self.model.state_dict())

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
        
        sentiment = np.zeros_like(returns)

        features = np.column_stack((
            returns,
            view['volume'],
            rsi,
            macdhist,
            sentiment
        )).astype(np.float32)

        if hasattr(self, 'feature_mean') and hasattr(self, 'feature_std'):
            features = (features - self.feature_mean) / self.feature_std

        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        recent_features = features[-self.window_size:]

        # Reshape for Transformer: (batch_size, sequence_length, input_size)
        x_tensor = torch.tensor(recent_features, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q_values = self.model(x_tensor)
            action = torch.argmax(q_values, dim=1).item()
            
        # Action space: 0=Hold, 1=Buy, 2=Sell
        if action == 1:
            return 1 # BUY
        elif action == 2:
            return -1 # SELL
        return 0 # HOLD
