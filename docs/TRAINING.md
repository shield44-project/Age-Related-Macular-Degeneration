# Training the AMD Classifier

This guide walks through training a production-grade Age-related Macular
Degeneration (AMD) classifier for this project. The trained checkpoint drops
into `backend/models/ViT_base/best_vit_model_improved.pth` and the GUI / Flask
backend will pick it up automatically.

> **Where to train.** Replit's free containers have no GPU and most of the
> dataset providers below require an account. Run training on **Google Colab
> (free T4)**, **Kaggle Notebooks (free P100/T4)**, or any local machine with
> an NVIDIA GPU ≥ 8 GB. Keep this repo as the source of truth for code, and
> only push the resulting `.pth` checkpoint back into the project.

---

## 1. Architecture choice

The training script (`backend/train_and_evaluate.py`) defaults to
**ConvNeXt-Tiny pretrained on ImageNet-22k** (`convnext_tiny.fb_in22k_ft_in1k`
in `timm`).

| Backbone | Params | Typical val F1 (AMD) | Train time / epoch (T4, bs=32) |
|---|---|---|---|
| ResNet-50 | 25 M | 0.86 – 0.89 | ~2 min |
| EfficientNetV2-S | 22 M | 0.89 – 0.91 | ~2 min |
| **ConvNeXt-Tiny (default)** | **28 M** | **0.92 – 0.94** | **~2.5 min** |
| Swin-V2-Tiny | 28 M | 0.92 – 0.95 | ~3 min |
| ViT-B/16 (the legacy model in this repo) | 86 M | 0.90 – 0.93 | ~5 min |

**Why ConvNeXt-Tiny?** It hits the same accuracy as Swin-V2 / ViT-B on retinal
benchmarks (iChallenge-AMD, ADAM, ODIR-5K) while being ~3× smaller and ~2×
faster to fine-tune. It's also more forgiving with imbalanced data, which
matters because every public AMD dataset is heavily skewed toward Normal eyes.

If you have a stronger GPU, swap the backbone with a single flag — see
"Other architectures" below.

---

## 2. Datasets

You can train on any single dataset, or concatenate several with
`--dataset-roots`. The recommended mix for a clinically-useful classifier is:

| Dataset | What's in it | Used for |
|---|---|---|
| **ADAM Challenge** | 1 200 colour fundus, AMD vs non-AMD labels | primary training |
| **ODIR-5K** | 5 000 patients, 8 disease labels (filter for "A" = AMD) | extra positives |
| **iChallenge-AMD** | 400 fundus, expert AMD annotations | additional eval |
| Kaggle DR (EyePACS) | DR-graded, but normal eyes are reusable | extra negatives |
| IDRiD / MESSIDOR-2 | retinopathy + AMD subset | sanity check |

> **DRIVE is *not* useful here** — it's a vessel-segmentation dataset (40
> images), not an AMD classification dataset. Skip it.

### 2.1 Folder layout the trainer expects

For each dataset root you pass in, the trainer expects:

```
<root>/
├── train/
│   ├── Normal/   *.jpg / *.png
│   └── AMD/
├── val/
│   ├── Normal/
│   └── AMD/
└── test/                # optional
    ├── Normal/
    └── AMD/
```

If a dataset comes with a CSV instead of folders, write a tiny script to copy
images into the right `Normal/` or `AMD/` folder. There's a starter at the
bottom of this file.

### 2.2 Dataset-by-dataset prep notes

* **ADAM** — already split into Training / Validation. Map labels:
  `non-AMD → Normal`, `AMD → AMD`. Use a 80/20 patient-level split inside
  the official train set for `train` / `val`, and keep the official validation
  set as `test`.
* **ODIR-5K** — labels are per-eye. Treat any image whose label string
  contains `"A"` (Age-related Macular Degeneration) as AMD; everything that is
  *only* `"N"` (Normal) is Normal. Discard ambiguous multi-label rows.
* **iChallenge-AMD** — almost identical to ADAM. Keep it as a held-out
  `test/` split if you've already trained on ADAM.
* **EyePACS / Kaggle DR** — only borrow the level-0 (no DR, healthy) images
  to bulk up `Normal/`. Don't use DR-positive images as AMD positives.
* **IDRiD / MESSIDOR-2** — small; great as additional `test/` data for
  generalisation reporting.

---

## 3. Training in 4 commands (Google Colab)

```bash
# 0. clone your repo and install deps
!git clone https://github.com/<you>/amd-detection-system.git
%cd amd-detection-system
!pip install -q torch torchvision timm scikit-learn opencv-python tqdm

# 1. download + prep your datasets into ./data/{adam,odir5k_amd}
#    (use the dataset-prep snippet at the bottom of this file)

# 2. train (ConvNeXt-Tiny, mixed precision, 25 epochs, ~1h on a T4)
!python -m backend.train_and_evaluate \
    --dataset-roots data/adam data/odir5k_amd \
    --arch convnext_tiny.fb_in22k_ft_in1k \
    --image-size 224 \
    --batch-size 32 \
    --epochs 25 \
    --lr 3e-4 \
    --weight-decay 5e-2 \
    --label-smoothing 0.05 \
    --warmup-epochs 2 \
    --output backend/models/ViT_base/best_vit_model_improved.pth \
    --notes "convnext-tiny on adam+odir5k, run 1"

# 3. download the checkpoint + JSON metrics report back to your machine
from google.colab import files
files.download('backend/models/ViT_base/best_vit_model_improved.pth')
files.download('backend/models/ViT_base/best_vit_model_improved.metrics.json')
```

Drop the two downloaded files into `backend/models/ViT_base/` in this repo
and the GUI will show the new metrics on its next launch.

---

## 4. Training pipeline details

The script automatically applies the techniques that matter most for AMD:

* **Class-balanced sampling** — weighted random sampler so each batch has
  roughly equal Normal / AMD ratio.
* **Class-weighted loss + label smoothing (0.05)** — handles residual
  imbalance and discourages over-confident wrong predictions.
* **Retinal-tuned augmentations** — modest rotation, mild colour jitter
  (fundus images have a narrow colour distribution), conservative random
  erasing, low-probability vertical flip.
* **Mixed-precision training (AMP)** — ~1.7× faster on any modern NVIDIA GPU.
* **AdamW + linear warm-up + cosine decay** — the standard recipe that wins
  most medical imaging fine-tuning shootouts.
* **Gradient clipping at 1.0** — keeps fine-tuning stable when class weights
  are large.
* **Early stopping on validation F1** with patience 6.

Outputs:

* `best_vit_model_improved.pth` — the checkpoint loaded by `backend/dl_model.py`.
  It contains `state_dict`, `arch`, `image_size`, `class_names`, plus all
  validation metrics (accuracy, precision, recall, F1, ROC-AUC, confusion
  matrix). The GUI displays these directly.
* `best_vit_model_improved.metrics.json` — the full report including the
  test-set numbers, training config, and per-class metrics.

---

## 5. Other architectures

```bash
# Swin-V2-Tiny — slightly higher F1, needs ~10 GB of GPU RAM
--arch swinv2_tiny_window8_256.ms_in1k --image-size 256

# EfficientNetV2-S — fastest inference, great for CPU deployment
--arch efficientnetv2_s.in21k_ft_in1k --image-size 300

# Original ViT-B/16 (the model the project was started with)
--arch vit_base_patch16_224.augreg2_in21k_ft_in1k --image-size 224
```

Any backbone available in `timm.list_models()` works; `--arch` is passed
straight through.

---

## 6. Quick dataset-prep script

Save this as `tools/prep_odir.py` and adapt it for other datasets:

```python
import csv, shutil
from pathlib import Path

src_dir   = Path("ODIR-5K/Training Images")
labels    = Path("ODIR-5K/data.csv")          # patient-level labels
out_root  = Path("data/odir5k_amd")
splits    = {"train": 0.8, "val": 0.1, "test": 0.1}

import random; random.seed(42)
rows = list(csv.DictReader(labels.open()))
random.shuffle(rows)

def label(row):
    diagnosis = (row["Left-Diagnostic Keywords"] + " " +
                 row["Right-Diagnostic Keywords"]).lower()
    if "macular degeneration" in diagnosis or row["A"] == "1":
        return "AMD"
    if row["N"] == "1":
        return "Normal"
    return None  # skip ambiguous

n = len(rows); n_train = int(n * splits["train"]); n_val = int(n * splits["val"])
buckets = {"train": rows[:n_train],
           "val":   rows[n_train:n_train + n_val],
           "test":  rows[n_train + n_val:]}

for split, items in buckets.items():
    for row in items:
        cls = label(row)
        if cls is None: continue
        for side, col in [("left", "Left-Fundus"), ("right", "Right-Fundus")]:
            src = src_dir / row[col]
            if not src.exists(): continue
            dst = out_root / split / cls / f"{row['ID']}_{side}_{src.name}"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, dst)
print("Done.")
```

---

## 7. After training

1. Copy `best_vit_model_improved.pth` and the `.metrics.json` into
   `backend/models/ViT_base/`.
2. Restart the Flask backend (`python -m backend`). The GUI's "Model" /
   "Accuracy" / "Precision" / "Recall" / "F1" labels will switch from the
   heuristic fallback values to your trained model's real numbers.
3. Upload a test fundus image — the prediction badge and saliency map are
   driven by the new checkpoint.
