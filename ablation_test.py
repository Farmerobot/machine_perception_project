"""Full-scale ablation: 4 variants, same config (1000/200, 6 epochs, BS=16, cosine LR)."""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import time

DATA_ROOT = Path("data/cityscapes-depth-and-segmentation/data")
MEAN = torch.tensor([0.2869, 0.3254, 0.2839]).view(3, 1, 1)
STD = torch.tensor([0.1701, 0.1748, 0.1722]).view(3, 1, 1)

TRAIN_N = 1000
VAL_N = 200
BS = 16
EPOCHS = 6
LR = 1e-3


class DepthDataset(Dataset):
    def __init__(self, split, max_n, use_log):
        self.split = split
        self.use_log = use_log
        folder = DATA_ROOT / split / "image"
        self.ids = sorted(int(p.stem) for p in folder.glob("*.npy"))[:max_n]

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
        if self.use_log:
            dep = torch.where(mask.bool(), torch.log1p(dep), torch.zeros_like(dep))
        return img, dep, mask


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
    def __init__(self, ch=16, use_sigmoid=True):
        super().__init__()
        self.use_sigmoid = use_sigmoid
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
        out = self.head(d1)
        return torch.sigmoid(out) if self.use_sigmoid else out


def masked_l1(pred, target, mask):
    return ((torch.abs(pred - target) * mask).sum()) / mask.sum().clamp(min=1)


@torch.no_grad()
def evaluate(model, dl, use_log):
    model.eval()
    total_abs_rel, total_sq, total_px = 0.0, 0.0, 0
    acc = {1.25: 0, 1.25**2: 0, 1.25**3: 0}

    for images, gt, masks in dl:
        pred = model(images)
        valid = masks.bool().squeeze(1)
        p = pred.squeeze(1)[valid]
        g = gt.squeeze(1)[valid]

        if use_log:
            p = torch.expm1(p)
            g = torch.expm1(g)

        keep = g > 1e-3
        p, g = p[keep], g[keep]

        total_abs_rel += (torch.abs(p - g) / g).sum().item()
        total_sq += ((p - g) ** 2).sum().item()
        total_px += len(p)

        ratio = torch.max(p / g, g / p)
        for t in acc:
            acc[t] += (ratio < t).sum().item()

    rel = total_abs_rel / total_px
    rmse = (total_sq / total_px) ** 0.5
    d1 = acc[1.25] / total_px * 100
    d2 = acc[1.25**2] / total_px * 100
    d3 = acc[1.25**3] / total_px * 100
    return rel, rmse, d1, d2, d3


def run_variant(name, use_sigmoid, use_log):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"  sigmoid={use_sigmoid}  log_target={use_log}")
    print(f"{'='*60}")

    train_ds = DepthDataset("train", TRAIN_N, use_log)
    val_ds = DepthDataset("val", VAL_N, use_log)
    train_dl = DataLoader(train_ds, batch_size=BS, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=BS, shuffle=False, num_workers=0)

    model = TinyUNet(ch=16, use_sigmoid=use_sigmoid)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for imgs, depths, masks in train_dl:
            loss = masked_l1(model(imgs), depths, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
        train_loss /= len(train_ds)
        scheduler.step()
        print(f"    Epoch {epoch}/{EPOCHS}  train_L1={train_loss:.4f}")
    dt = time.time() - t0

    rel, rmse, d1, d2, d3 = evaluate(model, val_dl, use_log)
    print(f"  Time: {dt:.0f}s  AbsRel: {rel*100:.1f}%  RMSE: {rmse:.4f}  d1: {d1:.1f}%  d2: {d2:.1f}%  d3: {d3:.1f}%")
    return name, rel, rmse, d1, d2, d3, dt


results = []
results.append(run_variant("A: sigmoid + raw targets",       use_sigmoid=True,  use_log=False))
results.append(run_variant("B: sigmoid + log targets",        use_sigmoid=True,  use_log=True))
results.append(run_variant("C: linear head + raw targets",    use_sigmoid=False, use_log=False))
results.append(run_variant("D: linear head + log targets",    use_sigmoid=False, use_log=True))

print(f"\n{'='*70}")
print(f"  FINAL COMPARISON (1000 train, 200 val, 6 epochs, BS=16)")
print(f"{'='*70}")
print(f"  {'Variant':<35s} AbsRel%  RMSE    d<1.25  d<1.56  d<1.95  Time")
print(f"  {'-'*35} -------  ------  ------  ------  ------  ----")
for name, rel, rmse, d1, d2, d3, dt in results:
    print(f"  {name:<35s} {rel*100:6.1f}   {rmse:.4f}  {d1:5.1f}%  {d2:5.1f}%  {d3:5.1f}%  {dt:.0f}s")

best = min(results, key=lambda r: r[1])
print(f"\n  BEST: {best[0]}")
