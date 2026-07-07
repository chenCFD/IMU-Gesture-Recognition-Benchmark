import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiBandFilterbank(nn.Module):
    """ Multi-band Filterbank """
    def __init__(self, in_channels, num_bands=3):
        super(MultiBandFilterbank, self).__init__()
        self.in_channels = in_channels
        self.num_bands = num_bands
        self.filterbank = nn.Conv1d(
            in_channels=in_channels,
            out_channels=in_channels * num_bands,
            kernel_size=3,
            padding=1,
            groups=in_channels
        )
        
    def forward(self, x):
        return self.filterbank(x)


class ResidualBlock1D(nn.Module):
    """ 1D Residual Convolutional Block (Lightweight Design) """
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class MultiBandConvBranch(nn.Module):
    """ 1. Multi-band Convolutional Branch """
    def __init__(self, in_channels, num_bands=3, cnn_dim=64):
        super(MultiBandConvBranch, self).__init__()
        self.filterbank = MultiBandFilterbank(in_channels, num_bands)
        
        mid_channels = in_channels * num_bands
        self.res_block1 = ResidualBlock1D(mid_channels, 32, stride=1)
        self.res_block2 = ResidualBlock1D(32, cnn_dim, stride=2)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x_filtered = self.filterbank(x)
        out = self.res_block1(x_filtered)
        out = self.res_block2(out)
        out = self.global_pool(out)
        out = torch.flatten(out, 1)
        return out


class StatisticalFeatureExtractor(nn.Module):
    """ Statistical Feature Extractor """
    def __init__(self, window_size=5, stride=5):
        super(StatisticalFeatureExtractor, self).__init__()
        self.window_size = window_size
        self.stride = stride

    def forward(self, x):
        windows = x.unfold(1, self.window_size, self.stride)
        
        mean = windows.mean(dim=-1)       
        std = windows.std(dim=-1)         
        max_val = windows.max(dim=-1)[0]   
        min_val = windows.min(dim=-1)[0]   
        rms = torch.sqrt((windows ** 2).mean(dim=-1) + 1e-8) 
        
        windows_4_fft = windows.permute(0, 1, 2, 3) 
        fft_vals = torch.fft.rfft(windows_4_fft, dim=-1)
        fft_amp = torch.abs(fft_vals).mean(dim=-1) 
        
        tokens = torch.cat([mean, std, max_val, min_val, rms, fft_amp], dim=-1)
        return tokens


class StatisticalTransformerBranch(nn.Module):
    """ 2. Statistical Transformer Branch """
    def __init__(self, num_windows=3, stat_feat_dim=72, token_dim=32, num_heads=2, num_layers=1):
        super(StatisticalTransformerBranch, self).__init__()
        self.feature_extractor = StatisticalFeatureExtractor(window_size=5, stride=5)
        self.linear_encoder = nn.Linear(stat_feat_dim, token_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(1, num_windows, token_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=token_dim, nhead=num_heads, batch_first=True, dim_feedforward=64)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.query_vector = nn.Parameter(torch.randn(1, 1, token_dim))

    def forward(self, x):
        tokens = self.feature_extractor(x)
        tokens_encoded = self.linear_encoder(tokens)
        tokens_encoded = tokens_encoded + self.pos_embedding
        trans_out = self.transformer_encoder(tokens_encoded)
        
        batch_size = x.shape[0]
        Q = self.query_vector.repeat(batch_size, 1, 1) 
        scores = torch.bmm(Q, trans_out.transpose(1, 2)) / math.sqrt(Q.shape[-1]) 
        attn_weights = F.softmax(scores, dim=-1)
        pooled_out = torch.bmm(attn_weights, trans_out).squeeze(1) 
        return pooled_out


class Huawei_Transformer_2026(nn.Module):
    """ Complete Mix-Token Network Architecture """
    def __init__(self, input_dim=12, time_steps=15, n_classes=9, cnn_dim=64, token_dim=32):
        super(Huawei_Transformer_2026, self).__init__()
        
        self.time_steps = time_steps
        self.cnn_branch = MultiBandConvBranch(in_channels=input_dim, num_bands=3, cnn_dim=cnn_dim)
        
        stat_feat_dim = input_dim * 6
        num_windows = time_steps // 5 
        
        self.transformer_branch = StatisticalTransformerBranch(
            num_windows=num_windows, 
            stat_feat_dim=stat_feat_dim, 
            token_dim=token_dim,
            num_heads=2,
            num_layers=1
        )
        
        self.mlp_cnn = nn.Sequential(
            nn.Linear(cnn_dim, 32),
            nn.ReLU(),
            nn.Linear(32, n_classes)
        )
        
        self.mlp_attn = nn.Sequential(
            nn.Linear(token_dim, 16),
            nn.ReLU(),
            nn.Linear(16, n_classes)
        )
        
        self.mlp_fused = nn.Sequential(
            nn.Linear(cnn_dim + token_dim, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes)
        )
        
        self.w = nn.Parameter(torch.zeros(3))

    def forward(self, x):
        # Dynamically slice based on configured time_steps
        # x = x[:, :, :self.time_steps]
        # print(x.shape)
        
        x_permuted = x.transpose(1, 2) # Reshape for Transformer branch
        emb_cnn = self.cnn_branch(x)
        emb_attn = self.transformer_branch(x_permuted) 
        
        y_cnn = self.mlp_cnn(emb_cnn)
        y_attn = self.mlp_attn(emb_attn)
        
        emb_fused = torch.cat([emb_cnn, emb_attn], dim=-1)
        y_fused = self.mlp_fused(emb_fused)
        
        pi = F.softmax(self.w, dim=0)
        y_final = pi[0] * y_cnn + pi[1] * y_attn + pi[2] * y_fused
        return y_final