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

        # Reshape for LSTM: (batch_size, sequence_length, input_size)
        x_tensor = torch.tensor(features).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            prediction = self.model(x_tensor)
            
        pred_value = prediction.item()
        current_price = recent_data[-1]['close']
        
        # Basic logic (Replace with real logic)
        if pred_value > current_price * 1.001:
            print(f"[{self.name}] AI Signal: BUY at {current_price:.2f} (Target: {pred_value:.2f})")
        elif pred_value < current_price * 0.999:
            print(f"[{self.name}] AI Signal: SELL at {current_price:.2f} (Target: {pred_value:.2f})")
