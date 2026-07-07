import torch
import torch.nn as nn

# =========================================================================
# ===== Main LSTM Model Architecture =====
# =========================================================================
class LSTM(nn.Module):
    def __init__(self, input_dim, time_steps, n_classes):
        super().__init__()

        hidden_dim = 64

        # LSTM Layer (Dynamically accepts input_dim from standard script)
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )

        # Classifier Block
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, n_classes)

        # Temporal dimension pooling
        self.avg_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # Expected input shape from DataLoader: (B, input_dim, time_steps)

        # Permute for Recurrent Network: (B, input_dim, time_steps) -> (B, time_steps, input_dim)
        x = x.permute(0, 2, 1)

        # Optimize RNN parameters for fast execution
        self.lstm.flatten_parameters()
        hidden, _ = self.lstm(x)  # Output shape: (B, time_steps, hidden_dim)

        # Permute for Pooling: (B, time_steps, hidden_dim) -> (B, hidden_dim, time_steps)
        hidden = hidden.permute(0, 2, 1)

        # Global average pooling across time axis
        x = self.avg_pool(hidden)  # Shape: (B, hidden_dim, 1)
        x = x.squeeze(-1)          # Shape: (B, hidden_dim)

        # Feed-forward dense mapping
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)            # Raw Logits Output (B, n_classes)

        return x