"""
Evaluate all depth estimation models on Cityscapes val set (Colab GPU).
Models: Baseline, DA2 Small/Base/Large (pretrained), Marigold, DA2 Small (fine-tuned).
Saves metrics to JSON + generates 4x2 image comparison for 3 val samples.

Usage: paste into Google Colab. Run the pip install cell first.
"""

# ============================================================
# 0. Install dependencies  (run this as a separate Colab cell)
# ============================================================
# !pip install -q kagglehub transformers accelerate diffusers matplotlib

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
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from PIL import Image
import matplotlib.pyplot as plt
import json
import time
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

# --- Config ---
VAL_SAMPLES = None  # None = use all val samples
EVAL_BATCH_SIZE = 32  # batch size for DA2 / baseline inference
SAMPLE_IDS = [0, 10, 20]  # val file IDs for visual comparison
METRICS_FILE = "evaluation_results.json"
COMPARISON_FILE = "comparison_all_models.png"

TUNED_REPO = None  # auto-detected below

# ============================================================
# 3. Locate data root
# ============================================================
data_root = Path(dataset_path)
for candidate in [data_root / "data", data_root]:
    if (candidate / "train" / "image").exists():
        data_root = candidate
        break
print(f"Data root: {data_root}")


# ============================================================
# 4. HF token helper + auto-detect tuned repo
# ============================================================
def get_hf_token():
    try:
        from google.colab import userdata

        return userdata.get("HF_TOKEN")
    except Exception:
        return os.environ.get("HF_TOKEN")


def detect_tuned_repo():
    global TUNED_REPO
    if TUNED_REPO:
        return TUNED_REPO
    token = get_hf_token()
    if token:
        try:
            from huggingface_hub import HfApi

            api = HfApi(token=token)
            user = api.whoami()["name"]
            TUNED_REPO = f"{user}/da2-small-cityscapes-depth"
            print(f"Auto-detected tuned repo: {TUNED_REPO}")
            return TUNED_REPO
        except Exception:
            pass
    print("Could not detect tuned repo — fine-tuned model will be skipped.")
    return None


# ============================================================
# 5. Baseline model (same architecture as train_baseline.py)
# ============================================================
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


MEAN = torch.tensor([0.2869, 0.3254, 0.2839]).view(3, 1, 1)
STD = torch.tensor([0.1701, 0.1748, 0.1722]).view(3, 1, 1)


# ============================================================
# 6. Load all validation data once
# ============================================================
def load_val_data(max_samples=None):
    folder = data_root / "val" / "image"
    ids = sorted(int(p.stem) for p in folder.glob("*.npy"))
    if max_samples:
        ids = ids[:max_samples]
    images, gts = [], []
    for sid in ids:
        images.append(np.load(data_root / "val" / "image" / f"{sid}.npy").squeeze())
        gts.append(np.load(data_root / "val" / "depth" / f"{sid}.npy").squeeze())
    print(f"  Loaded {len(images)} val images")
    return ids, images, gts


# ============================================================
# 7. Batched prediction functions
#    All follow the same pattern: (model, images) → list of (H,W) arrays
# ============================================================
@torch.no_grad()
def predict_all_baseline(images, weights_path="baseline_unet.pth"):
    """Batched baseline TinyUNet prediction. Returns list of (H,W) arrays or None."""
    if not os.path.exists(weights_path):
        return None
    model = TinyUNet(ch=16)
    model.load_state_dict(
        torch.load(weights_path, map_location=DEVICE, weights_only=True)
    )
    model.to(DEVICE).eval()

    all_preds = []
    for i in range(0, len(images), EVAL_BATCH_SIZE):
        batch = images[i : i + EVAL_BATCH_SIZE]
        tensors = [
            (torch.from_numpy(img).float().permute(2, 0, 1) - MEAN) / STD
            for img in batch
        ]
        batch_t = torch.stack(tensors).to(DEVICE)
        preds = model(batch_t).squeeze(1).cpu().numpy()  # (B, H, W)
        for k in range(preds.shape[0]):
            all_preds.append(preds[k])
    return all_preds


@torch.no_grad()
def predict_all_da2(model, processor, images):
    """Batched DA2 prediction. Returns list of normalized [0,1] (H,W) arrays."""
    model.eval()
    all_preds = []
    h, w = images[0].shape[:2]

    for i in range(0, len(images), EVAL_BATCH_SIZE):
        batch = images[i : i + EVAL_BATCH_SIZE]
        pil_imgs = [
            Image.fromarray((img * 255).clip(0, 255).astype(np.uint8)) for img in batch
        ]
        inputs = processor(images=pil_imgs, return_tensors="pt").to(DEVICE)
        preds = model(**inputs).predicted_depth  # (B, H_out, W_out)

        preds = (
            F.interpolate(
                preds.unsqueeze(1), size=(h, w), mode="bilinear", align_corners=False
            )
            .squeeze(1)
            .cpu()
            .numpy()
        )  # (B, H, W)

        for k in range(preds.shape[0]):
            p = preds[k]
            p = (p - p.min()) / (p.max() - p.min() + 1e-8)
            all_preds.append(p)
    return all_preds


@torch.no_grad()
def predict_all_marigold(pipe, images):
    """Marigold prediction (sequential — diffusion model). Returns list of [0,1] inverse-depth."""
    all_preds = []
    for img_np in images:
        img_uint8 = (img_np * 255).clip(0, 255).astype(np.uint8)
        pil_img = Image.fromarray(img_uint8)
        result = pipe(pil_img, num_inference_steps=2, ensemble_size=1)
        depth_arr = result.prediction.squeeze()
        if isinstance(depth_arr, torch.Tensor):
            depth_arr = depth_arr.cpu().numpy()
        h, w = img_np.shape[:2]
        if depth_arr.shape != (h, w):
            depth_arr = np.array(
                Image.fromarray(depth_arr).resize((w, h), Image.BILINEAR)
            )
        # Marigold outputs depth (higher=farther), invert to inverse-depth
        depth_arr = 1.0 - depth_arr
        depth_arr = (depth_arr - depth_arr.min()) / (
            depth_arr.max() - depth_arr.min() + 1e-8
        )
        all_preds.append(depth_arr)
    return all_preds


# ============================================================
# 8. Metric computation (shared by all models)
# ============================================================
def compute_metrics(preds, gts, name, elapsed=None):
    """Compute depth metrics from parallel lists of predictions and GTs."""
    total_abs_rel, total_sq_err, total_px = 0.0, 0.0, 0
    accuracy_within = {1.25: 0, 1.25**2: 0, 1.25**3: 0}

    for pred, gt in zip(preds, gts):
        mask = gt > 1e-3
        if mask.sum() < 10:
            continue
        pv, gv = pred[mask], gt[mask]

        # Affine-align pred to GT scale
        p_mean, p_std = pv.mean(), pv.std() + 1e-8
        g_mean, g_std = gv.mean(), gv.std() + 1e-8
        pv_aligned = (pv - p_mean) / p_std * g_std + g_mean
        pv_aligned = np.clip(pv_aligned, 1e-6, None)

        total_abs_rel += (np.abs(pv_aligned - gv) / gv).sum()
        total_sq_err += ((pv_aligned - gv) ** 2).sum()
        total_px += len(pv)

        ratio = np.maximum(pv_aligned / gv, gv / pv_aligned)
        for thr in accuracy_within:
            accuracy_within[thr] += (ratio < thr).sum()

    abs_rel = float(total_abs_rel / total_px)
    rmse = float((total_sq_err / total_px) ** 0.5)
    deltas = {
        f"d<{thr:.2f}": float(cnt / total_px * 100)
        for thr, cnt in accuracy_within.items()
    }

    metrics = {
        "abs_rel": abs_rel,
        "rmse": rmse,
        **deltas,
        "valid_pixels": int(total_px),
    }
    if elapsed is not None:
        metrics["time_s"] = round(elapsed, 1)

    time_str = f", {elapsed:.1f}s" if elapsed else ""
    print(f"\n  {name}  ({total_px:,} px{time_str})")
    print(
        f"    AbsRel={abs_rel:.4f}  RMSE={rmse:.4f}  "
        + "  ".join(f"{k}={v:.1f}%" for k, v in deltas.items())
    )
    return metrics


# ============================================================
# 9. Visualization: 3 val images × 4×2 grid per image
#    Row 1: Original, GT, Baseline, DA2-Small
#    Row 2: DA2-Base, DA2-Large, Marigold, DA2-Tuned
# ============================================================
COL_LABELS_R1 = ["Original", "Ground Truth", "Baseline", "DA2-Small"]
COL_LABELS_R2 = ["DA2-Base", "DA2-Large", "Marigold", "DA2-Small (tuned)"]


def visualize(samples, preds_dict):
    """samples: list of (img_np, gt_np). preds_dict: {model_name: [pred per sample]}."""
    n_samples = len(samples)
    ncols = 4
    nrows = n_samples * 2

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))

    def safe_get(model_key, sample_idx):
        arr = preds_dict.get(model_key, [None] * n_samples)
        p = arr[sample_idx] if sample_idx < len(arr) else None
        return p if p is not None else np.zeros_like(samples[0][1])

    for s in range(n_samples):
        img, gt = samples[s]
        r1, r2 = s * 2, s * 2 + 1

        row1_data = [
            (img, False),
            (gt, True),
            (safe_get("Baseline", s), True),
            (safe_get("DA2-Small", s), True),
        ]
        row2_data = [
            (safe_get("DA2-Base", s), True),
            (safe_get("DA2-Large", s), True),
            (safe_get("Marigold", s), True),
            (safe_get("DA2-Small (tuned)", s), True),
        ]

        for c, (data, use_cmap) in enumerate(row1_data):
            ax = axes[r1, c]
            ax.imshow(data, cmap="magma") if use_cmap else ax.imshow(data)
            if s == 0:
                ax.set_title(COL_LABELS_R1[c], fontsize=13, fontweight="bold")
            ax.axis("off")

        for c, (data, use_cmap) in enumerate(row2_data):
            ax = axes[r2, c]
            ax.imshow(data, cmap="magma") if use_cmap else ax.imshow(data)
            if s == 0:
                ax.set_title(COL_LABELS_R2[c], fontsize=13, fontweight="bold")
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(COMPARISON_FILE, dpi=150, bbox_inches="tight")
    print(f"\nSaved {COMPARISON_FILE}")
    plt.show()


# ============================================================
# 10. Main
# ============================================================
def main():
    detect_tuned_repo()
    all_metrics = {}

    # --- Load all val data once ---
    print("\nLoading validation data...")
    val_ids, val_images, val_gts = load_val_data(VAL_SAMPLES)

    # Map SAMPLE_IDS to indices for visualization
    sample_indices = [val_ids.index(sid) for sid in SAMPLE_IDS if sid in val_ids]
    preds_dict = {}

    def extract_samples(all_preds):
        """Pick visualization samples from full prediction list."""
        return [all_preds[i] for i in sample_indices]

    # ---- 1. Baseline ----
    print("\n" + "=" * 50)
    print("=== Baseline (TinyUNet) ===")
    t0 = time.time()
    baseline_preds = predict_all_baseline(val_images)
    if baseline_preds is not None:
        all_metrics["Baseline"] = compute_metrics(
            baseline_preds, val_gts, "Baseline", time.time() - t0
        )
        preds_dict["Baseline"] = extract_samples(baseline_preds)
    else:
        print("  baseline_unet.pth not found — skipping.")

    # ---- 2–4. DA2 Small / Base / Large (pretrained) ----
    da2_variants = {
        "DA2-Small": "depth-anything/Depth-Anything-V2-Small-hf",
        "DA2-Base": "depth-anything/Depth-Anything-V2-Base-hf",
        "DA2-Large": "depth-anything/Depth-Anything-V2-Large-hf",
    }
    for label, model_id in da2_variants.items():
        print("\n" + "=" * 50)
        print(f"=== {label} (pretrained) ===")
        proc = AutoImageProcessor.from_pretrained(model_id)
        mdl = AutoModelForDepthEstimation.from_pretrained(model_id).to(DEVICE)

        t0 = time.time()
        preds = predict_all_da2(mdl, proc, val_images)
        all_metrics[label] = compute_metrics(preds, val_gts, label, time.time() - t0)
        preds_dict[label] = extract_samples(preds)

        del mdl
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ---- 5. Marigold (sequential — diffusion model) ----
    print("\n" + "=" * 50)
    print("=== Marigold ===")
    from diffusers import MarigoldDepthPipeline

    marigold_pipe = MarigoldDepthPipeline.from_pretrained(
        "prs-eth/marigold-depth-v1-1",
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    ).to(DEVICE)

    t0 = time.time()
    preds = predict_all_marigold(marigold_pipe, val_images)
    all_metrics["Marigold"] = compute_metrics(
        preds, val_gts, "Marigold", time.time() - t0
    )
    preds_dict["Marigold"] = extract_samples(preds)

    del marigold_pipe
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ---- 6. DA2-Small fine-tuned (from HF) ----
    if TUNED_REPO:
        print("\n" + "=" * 50)
        print(f"=== DA2-Small (tuned) from {TUNED_REPO} ===")
        try:
            token = get_hf_token()
            proc_t = AutoImageProcessor.from_pretrained(TUNED_REPO, token=token)
            mdl_t = AutoModelForDepthEstimation.from_pretrained(
                TUNED_REPO, token=token
            ).to(DEVICE)

            t0 = time.time()
            preds = predict_all_da2(mdl_t, proc_t, val_images)
            all_metrics["DA2-Small (tuned)"] = compute_metrics(
                preds, val_gts, "DA2-Small (tuned)", time.time() - t0
            )
            preds_dict["DA2-Small (tuned)"] = extract_samples(preds)

            del mdl_t
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        except Exception as e:
            print(f"  Failed to load tuned model: {e}")

    # ---- Save metrics ----
    with open(METRICS_FILE, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nAll metrics saved to {METRICS_FILE}")
    print(json.dumps(all_metrics, indent=2))

    # ---- Visualization ----
    vis_samples = [(val_images[i], val_gts[i]) for i in sample_indices]
    visualize(vis_samples, preds_dict)


if __name__ == "__main__":
    main()
