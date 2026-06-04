# Galaxy Morphology Classifier (Swin Transformer)

Swin Transformer (Swin-T) fine-tuned to classify galaxy images into 10 morphological categories, trained on the [Galaxy10 DECals Dataset](https://www.kaggle.com/datasets/aryansanjeevarora/galaxy-morphology) using PyTorch Distributed Data Parallel (DDP) across two GPUs.

## Classes

Classes are assigned alphabetically by folder name — this is the order used by both `train_ddp.py` and `predict.py`.

## Technical Overview

**Model:** Swin-T pretrained on ImageNet, with the classifier head replaced by `Linear(768 → 512) → BatchNorm1d → GELU → Dropout(0.2) → Linear(512 → 10)`.

**Data:** All images are loaded into RAM as `uint8` arrays before training, eliminating disk I/O across epochs. A guard halts caching if RAM exceeds 95%. The dataset is split 80/10/10 (train/val/test) with stratification to preserve class proportions.

**Training:** AdamW optimizer with differential learning rates — the backbone is fine-tuned at `1.5e-5` (backbone LR scale `0.05 × 3e-4`) and the head at `3e-4`. The scheduler is `ReduceLROnPlateau` (`mode='max'`, `factor=0.5`, `patience=4`). Mixed precision (`torch.amp`) is used throughout. Class imbalance is handled by a custom `DistributedWeightedSampler`. Early stopping triggers after 16 epochs without val accuracy improvement (up to 60 total epochs).

**Distributed Training:** Uses PyTorch DDP via `mp.spawn` across 2 GPUs. Train and validation metrics are aggregated with `dist.all_reduce`. The `ReduceLROnPlateau` scheduler receives the same val accuracy on both ranks via `dist.broadcast`. Only rank-0 saves checkpoints and runs evaluation.

**Augmentation:** Training applies random horizontal/vertical flips, random 90° rotations (0°/90°/180°/270°), and MixUp (`α=0.2`, 20% of batches). CutMix is configured but disabled by default (`CUTMIX_PROB=0.0`). Label smoothing (`0.01`) is applied during training; a separate criterion without smoothing is used for validation.

**Inference:** Test-Time Augmentation (TTA) runs 16 stochastic forward passes per image on the GPU with random flips and rotations, averaging the softmax probabilities for the final prediction.

## Requirements

```
torch
torchvision
numpy
scikit-learn
tqdm
psutil
Pillow
```

All of the above are pre-installed on Kaggle.

## Files

| File | Description |
|------|-------------|
| `convert.py` | Converts `Galaxy10_DECals.h5` to an `ImageFolder`-compatible directory |
| `train_ddp.py` | DDP training and evaluation script (requires 2 GPUs) |
| `predict.py` | Classifies a single image using the trained model |
| `best_galaxy_swint.pth` | Best model weights — saved automatically during training |

## Step 1 — Download the Dataset

Download `Galaxy10_DECals.h5` from the link above.

## Step 2 — Convert to Image Folders

Open `convert.py` and set the paths:

```python
H5_PATH = r'C:\path\to\Galaxy10_DECals.h5'
OUT_DIR = r'C:\path\to\data'
```

Then run:

```bash
python convert.py
```

This produces a `data/` folder organised by class:

```
data/
  Barred_Spiral/
  Cigar_Smooth/
  Disturbed/
  Edge_On_Bulge/
  Edge_On_No_Bulge/
  Inbetween_Smooth/
  Merging/
  Round_Smooth/
  Unbarred_Loose_Spiral/
  Unbarred_Tight_Spiral/
```

## Step 3 — Zip the Data Folder

Use 7-Zip or another compression utility.

## Step 4 — Upload to Kaggle and Train

1. Go to kaggle.com and create a new notebook.
2. Upload `data.zip` and add it as a dataset. **Name it exactly `galaxy-morphology`** and update the `DATA_DIR` path in `train_ddp.py` if needed.
3. Set the accelerator to **GPU T4 x2** — the script will raise an error if fewer than 2 GPUs are detected.
4. Upload `train_ddp.py` and run:

```bash
python train_ddp.py
```

Training runs for up to 60 epochs with early stopping (patience 16). The best checkpoint is saved to `best_galaxy_swint.pth` in the notebook's working directory. Final test accuracy (with and without TTA) is printed at the end by rank-0.

## Step 5 — Download the Model

In the Kaggle notebook sidebar, go to **Output** and download `best_galaxy_swint.pth` to your local machine.

## Step 6 — Run Inference

Place `best_galaxy_swint.pth` in the same directory as `predict.py`, then run:

```bash
# Pass image path directly
python predict.py path/to/galaxy.jpg

# Or run and enter path when prompted
python predict.py
```

## Example Output

<img width="432" height="357" alt="image" src="https://github.com/user-attachments/assets/818c489f-8aaf-4815-81e7-a56606ee19a7" />

### Test Image: NGC 4414 (Spiral Galaxy)
### Top Prediction: Unbarred_Tight_Spiral
### Confidence: 54.6%

### Class Probabilities

| Classification | Probability | Visual |
| :--- | :--- | :--- |
| *Unbarred_Tight_Spiral* | *54.6%* | █████████████████████ |
| Unbarred_Loose_Spiral | 17.4% | ██████ |
| Disturbed | 7.3% | ██ |
| Round_Smooth | 5.5% | ██ |
| Barred_Spiral | 4.3% | █ |
| Inbetween_Smooth | 3.6% | █ |
| Merging | 3.0% | █ |
| Edge_On_No_Bulge | 2.1% | |
| Edge_On_Bulge | 2.0% | |
| Cigar_Smooth | 0.2% | |

Note: In astronomy, NGC 4414 is classified as an unbarred flocculent spiral, which aligns with the model's top prediction.
