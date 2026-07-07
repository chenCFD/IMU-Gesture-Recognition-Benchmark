import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================================================================
# ===== Component 1: GADF Encoder (Time Domain) =====
# =========================================================================
class GADF_Encoder(nn.Module):
    """
    Transforms 1D time-series sequences into 2D GADF images.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # Input shape: [batch, time, channels] -> Transpose to [batch, channels, time]
        x = x.transpose(1, 2)
        
        # 1. Normalize data array into [-1, 1] scale boundaries
        x_min = x.min(dim=-1, keepdim=True)[0]
        x_max = x.max(dim=-1, keepdim=True)[0]
        x_norm = 2.0 * (x - x_min) / (x_max - x_min + 1e-8) - 1.0
        
        # 2. Map normalized values into polar coordinate angles (phi)
        phi = torch.acos(torch.clamp(x_norm, -1.0, 1.0))
        
        # 3. Formulate GADF matrix fields via trigonometric computation
        phi_i = phi.unsqueeze(-1)  # [batch, channels, time, 1]
        phi_j = phi.unsqueeze(-2)  # [batch, channels, 1, time]
        gadf = torch.sin(phi_i - phi_j)  # Matrix cross calculation: [batch, channels, time, time]
        
        return gadf

# =========================================================================
# ===== Component 2: FFT Feature Extractor (Frequency Domain) =====
# =========================================================================
class FFT_Feature_Extractor(nn.Module):
    """
    Extracts 4 distinct frequency-domain metrics via 1D Real FFT calculation.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # Input shape: [batch, time, channels] -> Transpose to [batch, channels, time]
        x = x.transpose(1, 2)
        
        fft_vals = torch.fft.rfft(x, dim=-1)
        amp = torch.abs(fft_vals)
        
        freqs = torch.linspace(0, 1, amp.shape[-1], device=x.device).view(1, 1, -1)
        sum_amp = amp.sum(dim=-1) + 1e-8
        
        max_mag = amp.max(dim=-1)[0]
        mean_freq = (amp * freqs).sum(dim=-1) / sum_amp
        freq_var = (amp * (freqs - mean_freq.unsqueeze(-1))**2).sum(dim=-1) / sum_amp
        center_freq = amp.argmax(dim=-1).float() / amp.shape[-1]
        
        # Stack all 4 metrics together
        features = torch.stack([center_freq, mean_freq, freq_var, max_mag], dim=-1)
        
        batch_size = x.shape[0]
        return features.view(batch_size, 1, -1)

# =========================================================================
# ===== Component 3: Self-Attention Fusion Unit =====
# =========================================================================
class AttentionFusion(nn.Module):
    """
    Applies scaled dot-product cross attention to fuse time and frequency vectors.
    """
    def __init__(self, token_dim):
        super().__init__()
        self.dk = token_dim
        self.W_Q = nn.Linear(token_dim, token_dim)
        self.W_K = nn.Linear(token_dim, token_dim)
        self.W_V = nn.Linear(token_dim, token_dim)

    def forward(self, f1, f2):
        # Stack feature tokens along dimension 1
        E = torch.stack([f1, f2], dim=1) 
        
        Q = self.W_Q(E)
        K = self.W_K(E)
        V = self.W_V(E)
        
        scores = torch.bmm(Q, K.transpose(1, 2)) / (self.dk ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        
        F_att = torch.bmm(attn_weights, V)
        F_att1 = F_att[:, 0, :]
        F_att2 = F_att[:, 1, :]
        
        # Concatenate native features alongside context-aware attention vectors
        F_final = torch.cat([f1, f2, F_att1, F_att2], dim=-1)
        return F_final

# =========================================================================
# ===== Main Network Architecture: TF_Fusion_2025 =====
# =========================================================================
class TF_Fusion_2025(nn.Module):
    def __init__(self, input_dim, time_steps, n_classes):
        super().__init__()
        
        token_dim = 64
        dropout_rate = 0.5  # Encapsulated dropout within local constructor scope
        
        self.gadf_encoder = GADF_Encoder()
        self.fft_extractor = FFT_Feature_Extractor()
        
        # --- 2D-CNN Branch (Time-domain pattern processing) ---
        self.branch_2d = nn.Sequential(
            nn.Conv2d(in_channels=input_dim, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.AdaptiveMaxPool2d((1, 1)), 
            nn.Flatten(),
            nn.Linear(128, token_dim)
        )
        
        # --- 1D-CNN Branch (Frequency-domain pattern processing) ---
        self.branch_1d = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.MaxPool1d(2),
            
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.MaxPool1d(2),
            
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten(),
            nn.Linear(128, token_dim)
        )
        
        self.attention_fusion = AttentionFusion(token_dim)
        self.classifier = nn.Linear(4 * token_dim, n_classes)

    def forward(self, x):
        # Expected native shape from standard DataLoader: [B, input_dim, time_steps]
            
        # 2. Transpose into processing format: [B, input_dim, truncated_steps] -> [B, truncated_steps, input_dim]
        x = x.transpose(1, 2)
        
        # Encoding source signals into discrete operational representation spaces
        gadf_image = self.gadf_encoder(x)
        fft_features = self.fft_extractor(x)
        
        # Feature vector extraction pipeline
        F2_D = self.branch_2d(gadf_image)
        F1_D = self.branch_1d(fft_features)
        
        # Symmetric fusion mapping stage
        F_final = self.attention_fusion(F1_D, F2_D)
        logits = self.classifier(F_final)
        
        return logits