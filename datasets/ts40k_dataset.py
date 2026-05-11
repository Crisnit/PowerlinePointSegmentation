"""
TS40K Dataset для семантической сегментации (4 класса)

Пользовательские классы:
0 - Ground      (земля)
1 - Support     (опоры / towers)
2 - Wire        (провода / power lines)
3 - Vegetation  (растительность)

Оригинальные классы TS40K (после маппинга в датасете):
0: Noise
1: Ground
2: Low Vegetation
3: Medium Vegetation
4: Power line support tower
5: Power lines
"""

import os
import glob
import torch
import numpy as np
from torch.utils.data import Dataset

# ==================== МАППИНГ КЛАССОВ (ОБНОВЛЁННЫЙ) ====================
LABEL_MAP = {
    1: 0,      # Ground                    → 0 Ground
    2: 3,      # Low Vegetation            → 3 Vegetation
    3: 3,      # Medium Vegetation         → 3 Vegetation
    4: 1,      # Power line support tower  → 1 Support
    5: 2,      # Power lines               → 2 Wire
}

CLASS_NAMES = ["Ground", "Support", "Wire", "Vegetation"]
NUM_CLASSES = 4
IGNORE_LABEL = -1

# =====================================================================

def farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    if xyz.dim() == 2:
        xyz = xyz.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    B, N, _ = xyz.shape
    device = xyz.device

    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].unsqueeze(1)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, dim=-1)[1]

    return centroids[0] if squeeze else centroids


class TS40KSegDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        split: str = "fit",
        sample_types: list = None,
        n_points: int = 8192,
        augment: bool = True,
        normalize: bool = True,
        ignore_unmapped: bool = True,
        use_fps: bool = False,                 # ← НОВЫЙ ПАРАМЕТР
    ):
        super().__init__()
        self.root_dir = root_dir
        self.split = split
        self.n_points = n_points
        self.augment = augment
        self.normalize = normalize
        self.ignore_unmapped = ignore_unmapped
        self.use_fps = use_fps

        if sample_types is None:
            sample_types = ["tower_radius", "2_towers", "no_tower"]

        self.files = []
        for stype in sample_types:
            pattern = os.path.join(root_dir, stype, split, "*.pt")
            self.files.extend(sorted(glob.glob(pattern)))

        if len(self.files) == 0:
            raise FileNotFoundError(f"Не найдено .pt файлов в {root_dir} для split={split}")

        print(f"TS40KSegDataset: найдено {len(self.files)} сэмплов (split={split})")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        data = torch.load(path, map_location="cpu")

        xyz = data["input_pcd"].float()
        labels_raw = data["semantic_labels"].long().squeeze()

        # Remapping
        labels = torch.full_like(labels_raw, IGNORE_LABEL if self.ignore_unmapped else 0)
        for src, dst in LABEL_MAP.items():
            labels[labels_raw == src] = dst

        # Subsampling
        N = xyz.shape[0]
        if N > self.n_points:
            if self.use_fps:
                indices = farthest_point_sample(xyz, self.n_points)
            else:
                indices = torch.randperm(N)[:self.n_points]   # ← БЫСТРЫЙ ВАРИАНТ
            xyz = xyz[indices]
            labels = labels[indices]
        elif N < self.n_points:
            extra = torch.randint(0, N, (self.n_points - N,))
            xyz = torch.cat([xyz, xyz[extra]], dim=0)
            labels = torch.cat([labels, labels[extra]], dim=0)

        # Нормализация
        if self.normalize:
            centroid = xyz.mean(dim=0, keepdim=True)
            xyz = xyz - centroid
            max_dist = torch.max(torch.norm(xyz, dim=1, keepdim=True))
            if max_dist > 0:
                xyz = xyz / max_dist

        # Аугментации
        if self.augment:
            theta = np.random.uniform(0, 2 * np.pi)
            cos, sin = np.cos(theta), np.sin(theta)
            rot_z = torch.tensor([[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]], dtype=torch.float32)
            xyz = xyz @ rot_z
            xyz = xyz + torch.randn_like(xyz) * 0.005
            xyz = xyz * np.random.uniform(0.9, 1.1)

        return {
            "xyz": xyz,
            "labels": labels,
            "path": path,
            "original_n": N
        }


def compute_class_weights(dataset, num_classes=4):
    print("Вычисляем веса классов...")
    counts = torch.zeros(num_classes)
    total = 0
    for i in range(len(dataset)):
        item = dataset[i]
        lbl = item["labels"]
        for c in range(num_classes):
            counts[c] += (lbl == c).sum().item()
        total += len(lbl)
        if i % 300 == 0:
            print(f"  Обработано {i}/{len(dataset)}...")
    weights = 1.0 / (counts / total + 1e-6)
    weights = weights / weights.sum() * num_classes
    print("Class weights:", [round(w, 4) for w in weights.tolist()])
    return weights


if __name__ == "__main__":
    ROOT = "/path/to/TS40K-FULL"
    ds = TS40KSegDataset(ROOT, split="fit", n_points=4096, augment=True, use_fps=False)
    sample = ds[0]
    print("xyz shape:", sample["xyz"].shape)
    print("labels unique:", sample["labels"].unique().tolist())