# Machine Perception Project

Monocular inverse-depth estimation on the Cityscapes dataset using a lightweight U-Net.

## Dataset

This project uses the [Cityscapes Depth and Segmentation](https://www.kaggle.com/datasets/sakshaymahna/cityscapes-depth-and-segmentation/data) dataset from Kaggle.

### Download

1. Go to https://www.kaggle.com/datasets/sakshaymahna/cityscapes-depth-and-segmentation/data
2. Download and extract so the folder structure is:

```
data/
  cityscapes-depth-and-segmentation/
    data/
      train/
        image/    # .npy files, shape (128, 256, 3), float32, range [0, 1]
        depth/    # .npy files, shape (128, 256, 1), float32, inverse depth
        label/    # .npy files, shape (128, 256), int, 19 semantic classes
      val/
        image/
        depth/
        label/
```

- **2975** training samples, **500** validation samples
- Images are pre-normalized to [0, 1]
- Depth maps are inverse depth in range [0, ~0.49]; zero means invalid (sky/infinite distance)
- Semantic labels use 19 Cityscapes classes; void = -1

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

## Usage

### Exploratory Data Analysis
Open `exploratory_analysis.ipynb` in Jupyter.

### Train baseline U-Net
```bash
python train_baseline.py
```
Trains a TinyUNet (118K params) with masked L1 loss. Saves weights to `baseline_unet.pth`.

### Ablation tests
```bash
python ablation_test.py
```
Compares 4 variants (sigmoid/linear head × raw/log targets) on the same config.

## Results

Best baseline (sigmoid + raw targets, 1000 train, 10 epochs):

| Metric | Value |
|---|---|
| Abs Rel Error | 61.6% |
| RMSE | 0.0581 |
| δ < 1.25 | 42.7% |
| δ < 1.56 | 65.7% |
| δ < 1.95 | 78.3% |
