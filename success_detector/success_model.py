import torch
import torch.nn as nn


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(x).view(x.size(0), -1, 1, 1)


class CNNEncoder(nn.Module):
    def __init__(self, channels: int = 4, output_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            SEBlock(64),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        self.lin_proj = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(64 * 4 * 4, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        return self.lin_proj(x)


class SuccessPredictor(nn.Module):
    def __init__(self, num_cameras: int = 3, encoder_dim: int = 128,
                 num_heads: int = 4, dropout: float = 0.3):
        super().__init__()
        self.num_cameras = num_cameras
        self.encoders = nn.ModuleList([
            CNNEncoder(output_dim=encoder_dim, dropout=dropout)
            for _ in range(num_cameras)
        ])
        self.attn = nn.MultiheadAttention(encoder_dim, num_heads, batch_first=True)
        self.attn_norm = nn.LayerNorm(encoder_dim)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(encoder_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, latent_views):
        features = torch.stack(
            [self.encoders[i](v) for i, v in enumerate(latent_views)], dim=1
        )
        attended, _ = self.attn(features, features, features)
        fused = self.attn_norm(attended.mean(dim=1))
        return self.classifier(fused)
