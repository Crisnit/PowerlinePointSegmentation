import torch
sample = torch.load("/home/artem/work/Dataset/TS40K/tower_radius/fit/sample_0.pt")
print(sample['semantic_labels'].unique())

LABEL_MAP = {
    # Земля / ground + road
    1: 0,  # Ground, Road surface и т.п.
    # Растительность
    2: 3, 3: 3,   # Low vegetation, Medium vegetation
    # Опоры
    4: 1,        # Power line support tower
    # Провода
    5: 2  # Main power line, Other power line, Fiber optic
    # Всё остальное (шум, нерелевантное) → можно игнорировать или в "другое"
}
NUM_CLASSES = 4
CLASS_NAMES = ["ground", "vegetation", "tower", "wire"]