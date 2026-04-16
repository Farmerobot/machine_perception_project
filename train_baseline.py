"""Tiny U-Net baseline for inverse-depth estimation. CPU-friendly, trains in seconds."""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import time

# Config
DATA_ROOT = Path("data/cityscapes-depth-and-segmentation/data")
MEAN = torch.tensor([0.2869, 0.3254, 0.2839]).view(3, 1, 1)
STD = torch.tensor([0.1701, 0.1748, 0.1722]).view(3, 1, 1)

TRAIN_SUBSET = 1000
VAL_SUBSET = 200
BATCH_SIZE = 16
EPOCHS = 10
LR = 1e-3


# Dataset
class DepthDataset(Dataset):
    def __init__(self, split, max_samples=None):
        self.split = split
        folder = DATA_ROOT / split / "image"
        self.ids = sorted(int(p.stem) for p in folder.glob("*.npy"))
        if max_samples:
            self.ids = self.ids[:max_samples]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        sid = self.ids[idx]
        img = np.load(DATA_ROOT / self.split / "image" / f"{sid}.npy").squeeze()
        dep = np.load(DATA_ROOT / self.split / "depth" / f"{sid}.npy").squeeze()

        img = torch.from_numpy(img).float().permute(2, 0, 1)
        img = (img - MEAN) / STD
        dep = torch.from_numpy(dep).float().unsqueeze(0)
        mask = (dep > 0).float()
        return img, dep, mask


# Tiny U-Net
class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)

class TinyUNet(nn.Module):
    def __init__(self, ch=16):
        super().__init__()
        self.enc1 = ConvBlock(3, ch)
        self.enc2 = ConvBlock(ch, ch * 2)
        self.enc3 = ConvBlock(ch * 2, ch * 4)

        self.pool = nn.MaxPool2d(2)

        self.dec2 = ConvBlock(ch * 4 + ch * 2, ch * 2)
        self.dec1 = ConvBlock(ch * 2 + ch, ch)

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.head = nn.Conv2d(ch, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        d2 = self.dec2(torch.cat([self.up(e3), e2], 1))
        d1 = self.dec1(torch.cat([self.up(d2), e1], 1))

        return torch.sigmoid(self.head(d1))


# Masked L1 loss
def masked_l1(pred, target, mask):
    loss = (torch.abs(pred - target) * mask).sum()
    n_valid = mask.sum().clamp(min=1)
    return loss / n_valid


# Training
def main():
    train_ds = DepthDataset("train", TRAIN_SUBSET)
    val_ds = DepthDataset("val", VAL_SUBSET)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = TinyUNet(ch=16)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: TinyUNet, {n_params:,} params")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    t_start = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        t0 = time.time()
        for imgs, depths, masks in train_dl:
            pred = model(imgs)
            loss = masked_l1(pred, depths, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, depths, masks in val_dl:
                pred = model(imgs)
                val_loss += masked_l1(pred, depths, masks).item() * imgs.size(0)
        val_loss /= len(val_ds)

        scheduler.step()
        dt = time.time() - t0
        print(f"Epoch {epoch}/{EPOCHS}  train_L1={train_loss:.4f}  val_L1={val_loss:.4f}  lr={scheduler.get_last_lr()[0]:.6f}  ({dt:.1f}s)")

    total = time.time() - t_start
    print(f"\nTotal training time: {total:.1f}s")
    print("Saving model to baseline_unet.pth")
    torch.save(model.state_dict(), "baseline_unet.pth")

    evaluate(model, val_dl)


@torch.no_grad()
def evaluate(model, dataloader):
    model.eval()
    total_abs_rel, total_squared_error, total_pixels = 0.0, 0.0, 0
    accuracy_within = {1.25: 0, 1.25**2: 0, 1.25**3: 0}

    for images, ground_truth, masks in dataloader:
        predictions = model(images)
        valid_mask = masks.bool().squeeze(1)

        pred_valid = predictions.squeeze(1)[valid_mask]
        gt_valid = ground_truth.squeeze(1)[valid_mask]

        # skip near-zero ground truth to avoid division blow-up in relative metrics
        nonzero = gt_valid > 1e-3
        pred_valid, gt_valid = pred_valid[nonzero], gt_valid[nonzero]

        total_abs_rel += (torch.abs(pred_valid - gt_valid) / gt_valid).sum().item()
        total_squared_error += ((pred_valid - gt_valid) ** 2).sum().item()
        total_pixels += len(pred_valid)

        pred_gt_ratio = torch.max(pred_valid / gt_valid, gt_valid / pred_valid)
        for threshold in accuracy_within:
            accuracy_within[threshold] += (pred_gt_ratio < threshold).sum().item()

    mean_abs_rel = total_abs_rel / total_pixels
    rmse = (total_squared_error / total_pixels) ** 0.5

    print(f"\nVal metrics ({total_pixels:,} valid pixels):")
    print(f"  Abs Rel Error:  {mean_abs_rel:.4f}  (predictions are {mean_abs_rel*100:.1f}% off ground truth on average)")
    print(f"  RMSE:           {rmse:.4f}  (in inverse-depth units, range [0, 0.49])")
    for threshold, count in accuracy_within.items():
        pct = count / total_pixels * 100
        print(f"  delta < {threshold:.2f}:    {pct:.1f}% of pixels where pred is within {(threshold-1)*100:.0f}% of ground truth")


if __name__ == "__main__":
    main()
