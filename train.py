"""
Полный train.py для семантической сегментации TS40K с PyTorch Lightning + PointNet++

Запуск:
    python train.py
"""

import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import torch.nn.functional as F
import numpy as np

# Импортируем наш датасет
from datasets.ts40k_dataset import TS40KSegDataset, NUM_CLASSES, CLASS_NAMES, compute_class_weights

# ==================== PointNet++ ====================

# ==================== ПРОСТАЯ И СТАБИЛЬНАЯ МОДЕЛЬ (PointNet) ====================
class PointNetSeg(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, 512, 1)
        self.conv5 = nn.Conv1d(512, 1024, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(1024)

        self.drop = nn.Dropout(0.3)
        self.conv_out = nn.Conv1d(1088, num_classes, 1)  # 1024 + 64

    def forward(self, xyz):
        xyz = xyz.permute(0, 2, 1)                    # (B, 3, N)
        
        x1 = F.relu(self.bn1(self.conv1(xyz)))
        x2 = F.relu(self.bn2(self.conv2(x1)))
        x3 = F.relu(self.bn3(self.conv3(x2)))
        x4 = F.relu(self.bn4(self.conv4(x3)))
        x5 = F.relu(self.bn5(self.conv5(x4)))

        global_feat = torch.max(x5, 2, keepdim=True)[0]   # (B, 1024, 1)
        global_feat = global_feat.repeat(1, 1, xyz.shape[2])  # (B, 1024, N)

        concat = torch.cat([x1, global_feat], dim=1)      # (B, 1088, N)
        out = self.drop(F.relu(self.conv_out(concat)))
        return out.permute(0, 2, 1)                       # (B, N, num_classes)

# ==================== Lightning Module (обновлённый) ====================
class LitPointNet2(pl.LightningModule):
    def __init__(self, num_classes=NUM_CLASSES, class_weights=None, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = PointNetSeg(num_classes)          # ← правильная строка
        self.criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-1)
        self.lr = lr

    def forward(self, xyz):
        return self.model(xyz)

    def training_step(self, batch, batch_idx):
        xyz = batch["xyz"]
        labels = batch["labels"]
        logits = self(xyz)
        loss = self.criterion(logits.reshape(-1, NUM_CLASSES), labels.reshape(-1))
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        xyz = batch["xyz"]
        labels = batch["labels"]
        logits = self(xyz)
        loss = self.criterion(logits.reshape(-1, NUM_CLASSES), labels.reshape(-1))
        preds = torch.argmax(logits, dim=-1)
        iou = self._compute_iou(preds, labels)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_mIoU", iou, prog_bar=True)
        return loss

    def _compute_iou(self, preds, targets):
        ious = []
        for c in range(NUM_CLASSES):
            pred_c = (preds == c)
            target_c = (targets == c)
            intersection = (pred_c & target_c).sum().float()
            union = (pred_c | target_c).sum().float()
            if union > 0:
                ious.append(intersection / union)
        return torch.stack(ious).mean() if ious else torch.tensor(0.0, device=preds.device)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
        return [optimizer], [scheduler]


# ==================== MAIN ====================
def main():
    ROOT = "/home/artem/Рабочий стол/Diploma/lidar_defect_detection/data"          # ← ИЗМЕНИ НА СВОЙ ПУТЬ!
    BATCH_SIZE = 4
    N_POINTS = 8192
    MAX_EPOCHS = 10
    LR = 1e-3

    print("=== Подготовка данных ===")
    train_ds = TS40KSegDataset(ROOT, split="fit", n_points=N_POINTS, augment=True)
    val_ds = TS40KSegDataset(ROOT, split="test", n_points=N_POINTS, augment=False)

    if os.path.exists("class_weights.pt"):
        class_weights = torch.load("class_weights.pt")
        print("Загружены сохранённые веса классов")
    else:
        class_weights = compute_class_weights(train_ds)
        torch.save(class_weights, "class_weights.pt")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=4, pin_memory=True)

    print("=== Создаём модель ===")
    model = LitPointNet2(class_weights=class_weights, lr=LR)

    checkpoint_cb = ModelCheckpoint(
        monitor="val_mIoU",
        mode="max",
        save_top_k=3,
        filename="pointnet2-{epoch:02d}-{val_mIoU:.4f}",
        dirpath="checkpoints/"
    )
    early_stop = EarlyStopping(monitor="val_mIoU", patience=15, mode="max")

    logger = TensorBoardLogger("logs", name="TS40K_PointNet2")

    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="gpu",
        devices=1,
        precision="16-mixed",
        callbacks=[checkpoint_cb, early_stop],
        logger=logger,
        log_every_n_steps=50,
        gradient_clip_val=1.0
    )

    print("=== Старт обучения ===")
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()