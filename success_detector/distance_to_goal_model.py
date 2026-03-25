import torch
import torch.nn as nn

from success_model import SEBlock, CNNEncoder


class DistanceToGoalPredictor(nn.Module):
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
        self.regressor = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(encoder_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.ReLU(),
        )

    def forward(self, latent_views):
        features = torch.stack(
            [self.encoders[i](v) for i, v in enumerate(latent_views)], dim=1
        )
        attended, _ = self.attn(features, features, features)
        fused = self.attn_norm(attended.mean(dim=1))
        return self.regressor(fused)
