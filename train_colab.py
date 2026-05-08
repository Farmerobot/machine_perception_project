"""
Fine-tune Depth Anything V2 Small on Cityscapes inverse-depth data (Colab GPU).
Freeze DINOv2 backbone, train DPT head only, push result to HF Hub.

Usage: paste into Google Colab. Run the pip install cell first.
"""

# ============================================================
# 0. Install dependencies  (run this as a separate Colab cell)
# ============================================================
# !pip install -q kagglehub transformers accelerate

# ============================================================
# 1. Download dataset
# ============================================================
import kagglehub
from pathlib import Path

dataset_path = kagglehub.dataset_download(
    "sakshaymahna/cityscapes-depth-and-segmentation"
)
print(f"Dataset path: {dataset_path}")

# ============================================================
# 2. Imports and config
# ============================================================
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from PIL import Image
import time
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

# --- Tuning config (adjust as needed) ---
TRAIN_SAMPLES = None  # None = use all
VAL_SAMPLES = None
BATCH_SIZE = 32
NUM_WORKERS = 4  # increase to reduce data-loading bottleneck
EPOCHS = 10
LR = 5e-5
HF_REPO_NAME = "da2-small-cityscapes-depth"

# ============================================================
# 3. Locate the data root
# ============================================================
data_root = Path(dataset_path)
for candidate in [data_root / "data", data_root]:
    if (candidate / "train" / "image").exists():
        data_root = candidate
        break
print(f"Data root: {data_root}")
print(f"  Train images: {len(list((data_root / 'train' / 'image').glob('*.npy')))}")
print(f"  Val images:   {len(list((data_root / 'val' / 'image').glob('*.npy')))}")


# ============================================================
# 4. Dataset
# ============================================================
class CityscapesDepthDataset(Dataset):
    def __init__(self, root, split, processor, max_samples=None):
        self.root = Path(root)
        self.split = split
        self.processor = processor
        folder = self.root / split / "image"
        self.ids = sorted(int(p.stem) for p in folder.glob("*.npy"))
        if max_samples:
            self.ids = self.ids[:max_samples]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        sid = self.ids[idx]
        img_np = np.load(self.root / self.split / "image" / f"{sid}.npy").squeeze()
        dep_np = np.load(self.root / self.split / "depth" / f"{sid}.npy").squeeze()

        img_uint8 = (img_np * 255).clip(0, 255).astype(np.uint8)
        pil_img = Image.fromarray(img_uint8)
        inputs = self.processor(images=pil_img, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)

        depth = torch.from_numpy(dep_np).float()
        mask = (depth > 0).float()
        return pixel_values, depth, mask


# ============================================================
# 5. Loss: scale-invariant masked L1
# ============================================================
def scale_invariant_loss(pred, gt, mask):
    B = pred.shape[0]
    total_loss = 0.0
    for b in range(B):
        p, g, m = pred[b], gt[b], mask[b].bool()
        if m.sum() < 10:
            continue
        pv, gv = p[m], g[m]
        p_mean, p_std = pv.mean(), pv.std().clamp(min=1e-6)
        g_mean, g_std = gv.mean(), gv.std().clamp(min=1e-6)
        pv_aligned = (pv - p_mean) / p_std * g_std + g_mean
        total_loss += torch.abs(pv_aligned - gv).mean()
    return total_loss / max(B, 1)


# ============================================================
# 6. HF token helper
# ============================================================
def get_hf_token():
    try:
        from google.colab import userdata

        return userdata.get("HF_TOKEN")
    except Exception:
        return os.environ.get("HF_TOKEN")


# ============================================================
# 7. Training
# ============================================================
def train():
    model_name = "depth-anything/Depth-Anything-V2-Small-hf"
    print(f"\nLoading {model_name}...")
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForDepthEstimation.from_pretrained(model_name).to(DEVICE)

    # Freeze backbone (DINOv2), train only neck + head (DPT)
    for name, param in model.named_parameters():
        if "backbone" in name:
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} params ({trainable/total*100:.1f}%)")

    train_ds = CityscapesDepthDataset(data_root, "train", processor, TRAIN_SAMPLES)
    val_ds = CityscapesDepthDataset(data_root, "val", processor, VAL_SAMPLES)
    train_dl = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
    )

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LR
    )

    print(
        f"\nTraining: {len(train_ds)} samples, {EPOCHS} epochs, bs={BATCH_SIZE}, workers={NUM_WORKERS}"
    )
    t_start = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        t0 = time.time()

        for pixel_values, depth_gt, mask in train_dl:
            pixel_values = pixel_values.to(DEVICE)
            depth_gt = depth_gt.to(DEVICE)
            mask = mask.to(DEVICE)

            pred = model(pixel_values).predicted_depth
            pred_resized = F.interpolate(
                pred.unsqueeze(1),
                size=depth_gt.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)

            loss = scale_invariant_loss(pred_resized, depth_gt, mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * pixel_values.size(0)

        train_loss /= len(train_ds)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for pixel_values, depth_gt, mask in val_dl:
                pixel_values = pixel_values.to(DEVICE)
                depth_gt = depth_gt.to(DEVICE)
                mask = mask.to(DEVICE)
                pred = model(pixel_values).predicted_depth
                pred_resized = F.interpolate(
                    pred.unsqueeze(1),
                    size=depth_gt.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)
                val_loss += scale_invariant_loss(
                    pred_resized, depth_gt, mask
                ).item() * pixel_values.size(0)
        val_loss /= len(val_ds)

        dt = time.time() - t0
        print(
            f"Epoch {epoch}/{EPOCHS}  train_L1={train_loss:.4f}  val_L1={val_loss:.4f}  ({dt:.1f}s)"
        )

    print(f"\nTotal training time: {time.time() - t_start:.1f}s")
    return model, processor


# ============================================================
# 8. Push to Hugging Face Hub
# ============================================================
def push_to_hub(model, processor, repo_name=HF_REPO_NAME):
    from huggingface_hub import HfApi

    token = get_hf_token()
    if not token:
        print("HF_TOKEN not found — skipping push to Hub.")
        return

    api = HfApi(token=token)
    user = api.whoami()["name"]
    repo_id = f"{user}/{repo_name}"

    api.create_repo(repo_id=repo_id, exist_ok=True, repo_type="model")
    print(f"Repo ready: https://huggingface.co/{repo_id}")

    model.push_to_hub(repo_id, token=token)
    processor.push_to_hub(repo_id, token=token)
    print(f"Model pushed to https://huggingface.co/{repo_id}")


# ============================================================
# 9. Main
# ============================================================
def main():
    model, processor = train()
    push_to_hub(model, processor)


if __name__ == "__main__":
    main()
