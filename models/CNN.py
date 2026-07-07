import torch
import torch.nn as nn

# =========================================================================
# ===== Main CNN Model Architecture =====
# =========================================================================
class CNN(nn.Module):
    def __init__(self, input_dim, time_steps, n_classes):
        super().__init__()

        # Local hyperparameter definition
        dropout_rate = 0.3

        # First Convolutional Block (Dynamically bound to input_dim)
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)
        self.drop1 = nn.Dropout(dropout_rate)

        # Second Convolutional Block
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)
        self.drop2 = nn.Dropout(dropout_rate)

        # Third Convolutional Block
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.drop3 = nn.Dropout(dropout_rate)

        # Global Average Pooling (GAP) across temporal dimension
        self.gap = nn.AdaptiveAvgPool1d(1)

        # Fully-Connected Neural Network layers
        self.fc1 = nn.Linear(64, 64)
        self.drop4 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(64, n_classes)

    def forward(self, x):
        # Expected input shape from DataLoader: (B, input_dim, time_steps)

        # Block 1 Execution
        x = torch.relu(self.conv1(x))
        x = self.bn1(x)
        x = self.pool1(x)
        x = self.drop1(x)

        # Block 2 Execution
        x = torch.relu(self.conv2(x))
        x = self.bn2(x)
        x = self.pool2(x)
        x = self.drop2(x)

        # Block 3 Execution
        x = torch.relu(self.conv3(x))
        x = self.bn3(x)
        x = self.drop3(x)

        # Spatial / Temporal reduction via GAP
        x = self.gap(x)          # Shape: (B, 64, 1)
        x = x.squeeze(-1)        # Shape: (B, 64)

        # Dense Classifier Layers
        x = torch.relu(self.fc1(x))
        x = self.drop4(x)
        x = self.fc2(x)          # Raw Logits Output (B, n_classes)

        return x