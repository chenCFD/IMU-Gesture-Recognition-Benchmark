import torch
import torch.nn as nn

# =========================================================================
# ===== Main CRNN Model Architecture =====
# =========================================================================
class CRNN(nn.Module):
    def __init__(self, input_dim, time_steps, n_classes):
        super().__init__()

        # CNN Feature Extractor Branch
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            nn.ReLU()
        )

        # Recurrent GRU Layer
        self.gru = nn.GRU(
            input_size=64,
            hidden_size=32,
            num_layers=1,
            batch_first=True
        )

        # Temporal dimension pooling
        self.avg_pool = nn.AdaptiveAvgPool1d(1)

        # Final Classifier Layer
        self.classifier = nn.Linear(32, n_classes)

    def forward(self, x):
        # Expected input shape from DataLoader: (B, input_dim, time_steps)

        # 1. 1D Convolutional Layers Processing
        x = self.conv(x)         # Output shape: (B, 64, time_steps_prime)

        # 2. Reshape for Recurrent Network: (B, C, T) -> (B, T, C)
        x = x.permute(0, 2, 1)

        # 3. Gated Recurrent Unit Execution
        hidden, _ = self.gru(x)  # Output shape: (B, time_steps_prime, 32)

        # 4. Reshape for Pooling: (B, T, C) -> (B, C, T)
        hidden = hidden.permute(0, 2, 1)

        # 5. Global average pooling across temporal reduction axis
        x = self.avg_pool(hidden) # Shape: (B, 32, 1)
        x = x.squeeze(-1)         # Shape: (B, 32)

        # 6. Dense Classification Layer
        x = self.classifier(x)    # Raw Logits Output: (B, n_classes)

        return x