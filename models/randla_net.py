import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------
# 1. Вспомогательный блок: настоящее локальное агрегирование
# -----------------------------------------------------------
class LocalFeatureAggregation(nn.Module):
    def __init__(self, d_in, d_out, n_neighbors=16, d_pe=10): # d_pe - positional encoding dims
        super().__init__()
        self.n_neighbors = n_neighbors
        self.d_pe = d_pe
        mlp_in = d_in * 2 + d_pe  # конкат: центральная точка + разность фич + позиционный код расстояния

        self.mlp = nn.Sequential(
            nn.Linear(mlp_in, d_out),
            nn.BatchNorm1d(d_out),
            nn.ReLU(),
            nn.Linear(d_out, d_out),
            nn.BatchNorm1d(d_out),
            nn.ReLU()
        )
        self.shortcut = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()

    def forward(self, xyz, features):
        """
        xyz: (B, N, 3)
        features: (B, N, d_in)
        """
        B, N, _ = xyz.shape
        dist, idx = self.knn(xyz, self.n_neighbors)  # idx: (B, N, K)

        # Собираем соседей в пространстве признаков
        idx_base = torch.arange(0, B, device=xyz.device).view(-1, 1, 1) * N
        idx = idx + idx_base
        idx = idx.view(-1)  # (B*N*K,)

        # Центральная точка (повторенная K раз)
        central_feat = features.unsqueeze(2).expand(-1, -1, self.n_neighbors, -1) # (B, N, K, d_in)
        central_feat = central_feat.reshape(B * N * self.n_neighbors, -1)

        # Соседи
        neighbor_feat = features.view(B * N, -1)[idx, :] # (B*N*K, d_in)

        # Разность фич
        delta_feat = neighbor_feat - central_feat

        # Позиционное кодирование расстояния
        dist = dist.unsqueeze(-1) # (B, N, K, 1)
        pe = self.positional_encoding(dist) # (B, N, K, d_pe)
        pe = pe.reshape(B * N * self.n_neighbors, -1)

        # Сборка
        combined = torch.cat([central_feat, delta_feat, pe], dim=1) # (B*N*K, mlp_in)
        out = self.mlp(combined) # (B*N*K, d_out)

        # Max-pooling соседей обратно в центральные точки
        out = out.view(B, N, self.n_neighbors, -1).max(dim=2)[0] # (B, N, d_out)

        # Residual
        shortcut = self.shortcut(features.view(B * N, -1)).view(B, N, -1)
        return out + shortcut

    @staticmethod
    def knn(xyz, k):
        """Простой KNN для батча"""
        inner = -2 * torch.matmul(xyz, xyz.transpose(2, 1))
        xx = torch.sum(xyz ** 2, dim=-1, keepdim=True)
        pairwise_distance = -xx - inner - xx.transpose(2, 1)
        # Исключаем саму точку (k+1 и сдвиг)
        dist, idx = pairwise_distance.topk(k=k + 1, dim=-1, largest=True)
        return dist[..., 1:], idx[..., 1:]  # убираем саму точку

    def positional_encoding(self, dist):
        """
        Синус-косинусное кодирование, как в оригинальном RandLA-Net.
        dist: (B, N, K, 1)
        """
        # dist может быть отрицательным из-за численной неточности, берем abs
        dist = torch.abs(dist.clamp(min=1e-10))
        
        # Вычисляем максимальное расстояние для нормализации (можно сделать на весь датасет, но тут упрощаем)
        # В идеале использовать max_s = xyz.max(), но для разнообразия облаков лучше оставить learnable bias или просто log
        # Упрощенный вариант: используем логарифмическую шкалу
        scales = 2.0 ** torch.arange(self.d_pe // 2, device=dist.device).float() 
        # scales: [1, 2, 4, 8, ...]
        
        dist = dist.unsqueeze(-1) / (scales.view(1, 1, 1, -1) + 1e-8)
        sin_term = torch.sin(dist)
        cos_term = torch.cos(dist)
        return torch.cat([sin_term, cos_term], dim=-1)  # (B, N, K, d_pe)



# -----------------------------------------------------------
# 2. Сама сеть RandLA-Net (с корректным декодером)
# -----------------------------------------------------------
class RandLANet(nn.Module):
    def __init__(self, num_classes=4, d_in=3, n_neighbors=16):
        super().__init__()
        # Увеличим размерности для реальной задачи
        self.fc_start = nn.Sequential(
            nn.Linear(d_in, 32),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        
        # Энкодер (не забываем добавить сэмплинг на реальной сцене для ускорения, но оставим пока без FPS чтобы сравнивать честно)
        self.encoder = nn.ModuleList([
            LocalFeatureAggregation(32, 64, n_neighbors),
            LocalFeatureAggregation(64, 128, n_neighbors),
            LocalFeatureAggregation(128, 256, n_neighbors),
            LocalFeatureAggregation(256, 512, n_neighbors),
            LocalFeatureAggregation(512, 512, n_neighbors),
        ])
        
        # Декодер
        self.decoder = nn.ModuleList([
            nn.Sequential(nn.Linear(512 + 512, 512), nn.BatchNorm1d(512), nn.ReLU()),
            nn.Sequential(nn.Linear(512 + 256, 256), nn.BatchNorm1d(256), nn.ReLU()),
            nn.Sequential(nn.Linear(256 + 128, 128), nn.BatchNorm1d(128), nn.ReLU()),
            nn.Sequential(nn.Linear(128 + 64, 64), nn.BatchNorm1d(64), nn.ReLU()),
            nn.Sequential(nn.Linear(64 + 32, 32), nn.BatchNorm1d(32), nn.ReLU()),
        ])
        
        self.fc_end = nn.Linear(32, num_classes)
        self.drop = nn.Dropout(0.2)

    def forward(self, xyz):
        B, N, _ = xyz.shape
        
        # Начальные фичи
        x = xyz.view(-1, 3)
        x = self.fc_start(x)
        x = x.view(B, N, -1)
        
        # Энкодер
        encoder_features = [x]
        for layer in self.encoder:
            x = layer(xyz, x)  # <-- ПЕРЕДАЕМ XYZ ВМЕСТЕ С ФИЧАМИ!
            encoder_features.append(x)
        
        # Декодер со skip-connections
        for i, layer in enumerate(self.decoder):
            skip = encoder_features[-(i + 2)]
            # Важно: тут конкатенируем x (апсемпленный) и skip (с энкодера)
            x_cat = torch.cat([x, skip], dim=-1)
            x_flat = x_cat.view(-1, x_cat.shape[-1])
            x_flat = layer(x_flat)
            x = x_flat.view(B, N, -1)
        
        x = self.drop(x)
        x = self.fc_end(x.view(-1, x.shape[-1]))
        return x.view(B, N, -1)
