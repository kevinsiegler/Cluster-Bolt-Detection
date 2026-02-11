import torch
import torch.nn as nn


def chamfer_distance(x, y):
    x = x.unsqueeze(2)
    y = y.unsqueeze(1)
    dist = torch.sum((x - y) ** 2, dim=3)
    min_x = torch.min(dist, dim=2)[0]
    min_y = torch.min(dist, dim=1)[0]
    return torch.mean(min_x) + torch.mean(min_y)


class PointNetAE(nn.Module):
    def __init__(self, latent_dim=128, max_points=150):
        super().__init__()
        self.max_points = max_points

        self.encoder = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, max_points * 2)
        )

    def forward(self, x):
        B, N, _ = x.shape
        feat = self.encoder(x)
        global_feat = torch.max(feat, dim=1)[0]
        out = self.decoder(global_feat)
        out = out.view(B, self.max_points, 2)
        return out