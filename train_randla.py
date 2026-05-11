import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from datasets.ts40k_dataset import TS40KSegDataset, NUM_CLASSES, compute_class_weights
from models.randla_net import RandLANet          # ← положи выше в models/randla_net.py

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
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
        return [optimizer], [scheduler]


def main():
    ROOT = "/data/TS40K-FULL"          # ← измени под свой путь на сервере
    BATCH_SIZE = 16                    # A100 позволяет
    N_POINTS = 16384
    MAX_EPOCHS = 80
    LR = 1e-3

    train_ds = TS40KSegDataset(ROOT, split="fit", n_points=N_POINTS, augment=True, use_fps=False)
    val_ds = TS40KSegDataset(ROOT, split="test", n_points=N_POINTS, augment=False, use_fps=False)

    if os.path.exists("class_weights.pt"):
        class_weights = torch.load("class_weights.pt")
    else:
        class_weights = compute_class_weights(train_ds)
        torch.save(class_weights, "class_weights.pt")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)

    model = LitRandLA(class_weights=class_weights, lr=LR)

    checkpoint_cb = ModelCheckpoint(monitor="val_mIoU", mode="max", save_top_k=3,
                                    filename="randla-{epoch:02d}-{val_mIoU:.4f}")
    early_stop = EarlyStopping(monitor="val_mIoU", patience=15, mode="max")

    logger = TensorBoardLogger("logs", name="RandLA_Net_A100")
    # logger = WandbLogger(project="TS40K-RandLA", name="A100-run")  # опционально

    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="gpu",
        devices=1,
        precision="16-mixed",
        callbacks=[checkpoint_cb, early_stop],
        logger=logger,
        log_every_n_steps=50,
        gradient_clip_val=1.0,
    )

    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()