"""Compare depth predictions: Baseline vs Depth Anything V2 (S/B/L) vs Marigold.

Loads a single validation image and displays in a 2-row grid:
  Row 1: Original image, Ground Truth, Baseline, DA2-Small
  Row 2: DA2-Base, DA2-Large, Marigold

All models run on CPU.
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import time

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_ROOT = SCRIPT_DIR / "data"
SAMPLE_ID = 0  # which validation sample to visualize
BASELINE_WEIGHTS = SCRIPT_DIR / "baseline_unet.pth"

MEAN = torch.tensor([0.2869, 0.3254, 0.2839]).view(3, 1, 1)
STD = torch.tensor([0.1701, 0.1748, 0.1722]).view(3, 1, 1)


# ---------------------------------------------------------------------------
# Baseline model (same architecture as train_baseline.py)
# ---------------------------------------------------------------------------
class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
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


# ---------------------------------------------------------------------------
# Load sample image and ground truth
# ---------------------------------------------------------------------------
def load_sample(sample_id: int):
    """Load a validation image and its ground truth depth map."""
    img_path = DATA_ROOT / "val" / "image" / f"{sample_id}.npy"
    dep_path = DATA_ROOT / "val" / "depth" / f"{sample_id}.npy"

    img = np.load(img_path).squeeze()  # (H, W, 3) float in [0, 1]
    dep = np.load(dep_path).squeeze()  # (H, W)
    return img, dep


# ---------------------------------------------------------------------------
# Baseline prediction
# ---------------------------------------------------------------------------
@torch.no_grad()
def predict_baseline(img_np: np.ndarray) -> np.ndarray:
    """Run TinyUNet baseline on the image."""
    model = TinyUNet(ch=16)
    model.load_state_dict(torch.load(BASELINE_WEIGHTS, map_location="cpu"))
    model.eval()

    img_t = torch.from_numpy(img_np).float().permute(2, 0, 1)
    img_t = (img_t - MEAN) / STD
    img_t = img_t.unsqueeze(0)

    pred = model(img_t)
    return pred.squeeze().numpy()


# ---------------------------------------------------------------------------
# Depth Anything V2 prediction (via HuggingFace Transformers)
# ---------------------------------------------------------------------------
DA2_MODELS = {
    "Small": "depth-anything/Depth-Anything-V2-Small-hf",
    "Base": "depth-anything/Depth-Anything-V2-Base-hf",
    "Large": "depth-anything/Depth-Anything-V2-Large-hf",
}


def predict_depth_anything_v2(img_np: np.ndarray, variant: str = "Small") -> np.ndarray:
    """Run Depth Anything V2 on CPU via transformers pipeline."""
    from transformers import pipeline

    model_id = DA2_MODELS[variant]
    print(f"  Loading Depth Anything V2 {variant} ({model_id})...")
    pipe = pipeline(task="depth-estimation", model=model_id, device="cpu")

    # Convert numpy image [0,1] float to PIL
    img_uint8 = (img_np * 255).clip(0, 255).astype(np.uint8)
    pil_img = Image.fromarray(img_uint8)

    result = pipe(pil_img)
    depth_pil = result["depth"]  # PIL Image
    depth_arr = np.array(depth_pil).astype(np.float32)

    # Normalize to [0, 1] for visualization
    depth_arr = (depth_arr - depth_arr.min()) / (
        depth_arr.max() - depth_arr.min() + 1e-8
    )
    return depth_arr


# ---------------------------------------------------------------------------
# Marigold prediction (via diffusers)
# ---------------------------------------------------------------------------
def predict_marigold(img_np: np.ndarray) -> np.ndarray:
    """Run Marigold depth estimation on CPU."""
    import diffusers

    print("  Loading Marigold (this downloads ~1.7GB on first run)...")
    pipe = diffusers.MarigoldDepthPipeline.from_pretrained(
        "prs-eth/marigold-depth-v1-1",
        torch_dtype=torch.float32,
    ).to("cpu")

    # Convert numpy image [0,1] float to PIL
    img_uint8 = (img_np * 255).clip(0, 255).astype(np.uint8)
    pil_img = Image.fromarray(img_uint8)

    # Use minimal steps for CPU feasibility
    result = pipe(pil_img, num_inference_steps=2, ensemble_size=1)
    depth_arr = result.prediction.squeeze()  # (H, W) in [0, 1]

    if isinstance(depth_arr, torch.Tensor):
        depth_arr = depth_arr.numpy()

    # Marigold outputs depth (higher = farther). Invert to match
    # inverse-depth convention used by GT and other models (higher = closer).
    depth_arr = 1.0 - depth_arr
    return depth_arr


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def visualize(img_np, gt_depth, predictions: dict):
    """Plot all results in a 2-row grid."""
    panels = (
        [("Original Image", img_np, False)]
        + [("Ground Truth", gt_depth, True)]
        + [(name, pred, True) for name, pred in predictions.items()]
    )

    n = len(panels)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
    axes = axes.flatten()

    for i, (title, data, use_cmap) in enumerate(panels):
        if use_cmap:
            axes[i].imshow(data, cmap="magma")
        else:
            axes[i].imshow(data)
        axes[i].set_title(title, fontsize=12, fontweight="bold")
        axes[i].axis("off")

    # Hide unused axes
    for j in range(n, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(SCRIPT_DIR / "comparison_output.png", dpi=150, bbox_inches="tight")
    print(f"\nSaved visualization to comparison_output.png")
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def resize_to(arr, h, w):
    """Resize a [0,1] float array to (h, w) via PIL."""
    if arr.shape == (h, w):
        return arr
    pil = Image.fromarray((arr * 255).astype(np.uint8)).resize((w, h))
    return np.array(pil).astype(np.float32) / 255.0


def main():
    print(f"Loading sample {SAMPLE_ID} from validation set...")
    img_np, gt_depth = load_sample(SAMPLE_ID)
    h, w = gt_depth.shape
    print(f"  Image shape: {img_np.shape}, Depth shape: {gt_depth.shape}")

    predictions = {}
    step = 0
    total = 5  # baseline + 3x DA2 + marigold

    # Baseline
    step += 1
    print(f"\n[{step}/{total}] Running Baseline (TinyUNet)...")
    t0 = time.time()
    predictions["Baseline (TinyUNet)"] = predict_baseline(img_np)
    print(f"  Done in {time.time() - t0:.1f}s")

    # Depth Anything V2 — Small, Base, Large
    for variant in ["Small", "Base", "Large"]:
        step += 1
        print(f"\n[{step}/{total}] Running Depth Anything V2 {variant}...")
        t0 = time.time()
        pred = predict_depth_anything_v2(img_np, variant=variant)
        predictions[f"DA2 {variant}"] = resize_to(pred, h, w)
        print(f"  Done in {time.time() - t0:.1f}s")

    # Marigold
    step += 1
    print(f"\n[{step}/{total}] Running Marigold (CPU, ~30-60s)...")
    t0 = time.time()
    pred = predict_marigold(img_np)
    predictions["Marigold"] = resize_to(pred, h, w)
    print(f"  Done in {time.time() - t0:.1f}s")

    # Visualize
    print("\nGenerating comparison plot...")
    visualize(img_np, gt_depth, predictions)


if __name__ == "__main__":
    main()
