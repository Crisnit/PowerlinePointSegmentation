# ==================== Улучшенный RandLA-Net ====================
import torch
import torch.nn as nn
import torch.nn.functional as F

class LocalFeatureAggregation(nn.Module):
    def __init__(self, in_channels, out_channels, k=16):
        super().__init__()
        self.k = k
        self.mlp = nn.Sequential(
            nn.Linear(in_channels * 2, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )
        self.shortcut = nn.Linear(in_channels, out_channels) if in_channels != out_channels else nn.Identity()

    def forward(self, x, xyz):
        B, N, C = x.shape
        # Упрощённая локальная агрегация (для скорости)
        x = x.view(-1, C)
        x = self.mlp(torch.cat([x, x], dim=1))  # заглушка, в полной версии — KNN
        x = x.view(B, N, -1)
        return x + self.shortcut(x)


class RandLANet(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.fc_start = nn.Linear(3, 32)
        self.bn_start = nn.BatchNorm1d(32)

        self.encoder = nn.ModuleList([
            LocalFeatureAggregation(32, 64),
            LocalFeatureAggregation(64, 128),
            LocalFeatureAggregation(128, 256),
            LocalFeatureAggregation(256, 512),
        ])

        self.decoder = nn.ModuleList([
            nn.Sequential(nn.Linear(512 + 256, 256), nn.BatchNorm1d(256), nn.ReLU()),
            nn.Sequential(nn.Linear(256 + 128, 128), nn.BatchNorm1d(128), nn.ReLU()),
            nn.Sequential(nn.Linear(128 + 64, 64), nn.BatchNorm1d(64), nn.ReLU()),
            nn.Sequential(nn.Linear(64 + 32, 32), nn.BatchNorm1d(32), nn.ReLU()),
        ])

        self.fc_end = nn.Linear(32, num_classes)
        self.drop = nn.Dropout(0.2)

    def forward(self, xyz):
        B, N, _ = xyz.shape
        x = xyz.view(-1, 3)
        x = F.relu(self.bn_start(self.fc_start(x)))
        x = x.view(B, N, -1)

        # Encoder
        features = [x]
        for layer in self.encoder:
            x = layer(x, xyz)
            features.append(x)

        # Decoder
        for i, layer in enumerate(self.decoder):
            x = torch.cat([x, features[-(i + 2)]], dim=-1)
            x = x.view(-1, x.shape[-1])
            x = layer(x)
            x = x.view(B, N, -1)

        x = self.drop(x)
        x = self.fc_end(x.view(-1, x.shape[-1]))
        return x.view(B, N, -1)