import torch
import torch.nn as nn


class SuccessPredictor(nn.Module):
    # Coverts (1280, 9, 5) and pools/averages to a (1280) vector which is fed to the MLP
    # to produce a logit

    def __init__(self, latent_dim: int = 1280, hidden_dims: tuple = (512, 128)):
        super().__init__()

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        layers = []
        in_dim = latent_dim
        for h_dim in hidden_dims:
            layers.extend([nn.Linear(in_dim, h_dim), nn.ReLU()])
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, 1))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x).flatten(1)
        return self.mlp(x)

