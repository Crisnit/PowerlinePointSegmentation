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

# ==================== RandLA-Net (полная исправленная версия) ====================

class RandLANet(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.fc1 = nn.Linear(3, 32)
        self.bn1 = nn.BatchNorm1d(32)
        
        self.encoder_layers = nn.ModuleList([
            self._make_layer(32, 64),
            self._make_layer(64, 128),
            self._make_layer(128, 256),
            self._make_layer(256, 512),
        ])
        
        self.decoder_layers = nn.ModuleList([
            nn.Linear(512 + 256, 256),
            nn.Linear(256 + 128, 128),
            nn.Linear(128 + 64, 64),
            nn.Linear(64 + 32, 32),
        ])
        
        self.bn_decoder = nn.ModuleList([
            nn.BatchNorm1d(256),
            nn.BatchNorm1d(128),
            nn.BatchNorm1d(64),
            nn.BatchNorm1d(32),
        ])
        
        self.fc_out = nn.Linear(32, num_classes)

    def _make_layer(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Linear(in_ch, out_ch),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
            nn.Linear(out_ch, out_ch),
            nn.BatchNorm1d(out_ch),
            nn.ReLU()
        )

    def forward(self, xyz):
        B, N, _ = xyz.shape
        x = xyz.view(-1, 3)
        x = F.relu(self.bn1(self.fc1(x)))
        x = x.view(B, N, -1)

        # Encoder
        features = [x]
        for layer in self.encoder_layers:
            x = x.view(-1, x.shape[-1])
            x = layer(x)
            x = x.view(B, N, -1)
            features.append(x)

        # Decoder
        for i, layer in enumerate(self.decoder_layers):
            x = torch.cat([x, features[-(i+2)]], dim=-1)
            x = x.view(-1, x.shape[-1])
            x = F.relu(self.bn_decoder[i](layer(x)))
            x = x.view(B, N, -1)

        x = self.fc_out(x.view(-1, x.shape[-1]))
        return x.view(B, N, -1)

# ==================== RandLA-Net (сильная модель) ====================

class LitRandLA(pl.LightningModule):
    def __init__(self, num_classes=NUM_CLASSES, class_weights=None, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = RandLANet(num_classes)
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
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60)
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
    model = LitRandLA(class_weights=class_weights, lr=LR)

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