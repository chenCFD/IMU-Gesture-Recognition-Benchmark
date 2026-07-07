import torch
import torch.nn as nn

# =========================================================================
# ===== GRU Sub-Block =====
# =========================================================================
class GRUBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )

    def forward(self, x):
        # Input shape expected by GRU: (B, T, input_dim)
        out, _ = self.gru(x)
        return out  # Output shape: (B, T, hidden_dim)


# =========================================================================
# ===== Conv1D Sub-Block =====
# =========================================================================
class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, k):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_c, out_c, kernel_size=k, padding=k//2, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(out_c)
        )

    def forward(self, x):
        return self.block(x)


# =========================================================================
# ===== Main RCNN Model Architecture =====
# =========================================================================
class RCNN(nn.Module):
    def __init__(self, input_dim, time_steps, n_classes):
        super().__init__()
        
        hidden_dim = 64
        
        # Dynamic allocation based on standard training script arguments
        self.gru = GRUBlock(input_dim=input_dim, hidden_dim=hidden_dim)

        # Main Conv1D sequential branch
        self.conv1 = ConvBlock(in_c=hidden_dim, out_c=32, k=5)
        self.conv2 = ConvBlock(in_c=32, out_c=16, k=3)

        # Residual skip-connection branch
        self.conv_res = ConvBlock(in_c=hidden_dim, out_c=16, k=1)

        # Temporal dimension pooling
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Final Classification Fully-Connected network
        self.fc = nn.Sequential(
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, n_classes)
        )

    def forward(self, x):
        # Input standard shape from DataLoader: (B, input_dim, time_steps)
        
        # Permute for GRU: (B, input_dim, time_steps) -> (B, time_steps, input_dim)
        x = x.permute(0, 2, 1)   
        x = self.gru(x)          # Output shape: (B, time_steps, hidden_dim)

        # Permute for Conv1D: (B, time_steps, hidden_dim) -> (B, hidden_dim, time_steps)
        x = x.permute(0, 2, 1)   

        # Execute parallel branches (Main vs Residual)
        out = self.conv1(x)
        out = self.conv2(out)    # Main feature map out: (B, 16, time_steps)

        res = self.conv_res(x)   # Residual map out: (B, 16, time_steps)

        # Residual Addition
        out = out + res

        # Global average pooling across time dimension
        out = self.pool(out)     # Shape: (B, 16, 1)
        out = out.squeeze(-1)    # Shape: (B, 16)

        # Classification Dense Layers
        out = self.fc(out)       # Final logits: (B, n_classes)

        return out