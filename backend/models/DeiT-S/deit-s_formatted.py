# ===== CELL 1 =====
# ============================================================
#  CELL 1 — Dataset Loading + Preprocessing + Augmentation
#  Project : RVCE EL — AMD vs Normal Fundus Classifier (DeiT-S)
#  Framework: PyTorch + timm
# ============================================================

import os
import pathlib
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import cv2
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

print(f"PyTorch : {torch.__version__}")
print(f"CUDA    : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU     : {torch.cuda.get_device_name(0)}")


# ── 1. Config ─────────────────────────────────────────────────
BASE = "/kaggle/input/datasets/tejaswibteggihalli/master-dataset-2/Master_Dataset_2/Master_Dataset"

CFG = dict(
    TRAIN_AMD    = f"{BASE}/training/AMD_training",
    TRAIN_NORMAL = f"{BASE}/training/Normal_training",
    VAL_AMD      = f"{BASE}/validation/AMD_validation",
    VAL_NORMAL   = f"{BASE}/validation/Normal_validation",

    CLASS_NAMES  = ["Normal", "AMD"],
    IMG_SIZE     = 224,         # DeiT-S native size
    CLAHE_CLIP   = 2.0,
    CLAHE_TILE   = (8, 8),

    BATCH_SIZE   = 32,
    NUM_WORKERS  = 0,           # ✅ Kaggle T4 fix
    SEED         = 42,
)

torch.manual_seed(CFG["SEED"])


# ── 2. Sanity check ───────────────────────────────────────────
print("\n[Sanity] Checking dataset mount …")
for key in ("TRAIN_AMD", "TRAIN_NORMAL", "VAL_AMD", "VAL_NORMAL"):
    d = pathlib.Path(CFG[key])
    if not d.exists():
        raise FileNotFoundError(f"Directory not found: {d}")
    files = list(d.iterdir())
    print(f"  {key:15s} → {len(files):4d} files  |  sample: {files[0].name}")


# ── 3. Preprocessing helpers ──────────────────────────────────
def apply_clahe(bgr_img: np.ndarray) -> np.ndarray:
    """
    ✅ FIXED: LAB-space CLAHE — preserves colour information.
    AMD drusen have yellowish colour signatures lost in grayscale.
    """
    lab      = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB)
    l, a, b  = cv2.split(lab)
    clahe    = cv2.createCLAHE(
                   clipLimit=CFG["CLAHE_CLIP"],
                   tileGridSize=CFG["CLAHE_TILE"]
               )
    l        = clahe.apply(l)
    lab      = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)   # returns full colour BGR


def center_crop(img: np.ndarray, crop_size: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h < crop_size or w < crop_size:
        pad_h = max(0, crop_size - h)
        pad_w = max(0, crop_size - w)
        img = cv2.copyMakeBorder(
            img,
            pad_h // 2, pad_h - pad_h // 2,
            pad_w // 2, pad_w - pad_w // 2,
            cv2.BORDER_REFLECT_101
        )
        h, w = img.shape[:2]
    top  = (h - crop_size) // 2
    left = (w - crop_size) // 2
    return img[top : top + crop_size, left : left + crop_size]


def preprocess_image(path: str) -> np.ndarray:
    """
    LAB CLAHE + center crop + resize.
    Returns (224, 224, 3) uint8 RGB.
    """
    img = cv2.imread(path)
    img = apply_clahe(img)
    img = center_crop(img, CFG["IMG_SIZE"])
    img = cv2.resize(
              img,
              (CFG["IMG_SIZE"], CFG["IMG_SIZE"]),
              interpolation=cv2.INTER_AREA
          )
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)    # albumentations expects RGB
    return img


# ── 4. Albumentations pipelines ───────────────────────────────
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.Rotate(limit=15, border_mode=cv2.BORDER_REFLECT_101, p=0.5),
    A.ShiftScaleRotate(
        shift_limit=0.05,
        scale_limit=0.10,
        rotate_limit=0,
        border_mode=cv2.BORDER_REFLECT_101,
        p=0.4
    ),
    A.RandomBrightnessContrast(
        brightness_limit=0.15,
        contrast_limit=0.15,
        p=0.5
    ),
    A.GaussianBlur(blur_limit=(3, 5), p=0.2),
    A.GaussNoise(std_range=(0.02, 0.09), p=0.2),           # ✅ fixed API
    A.CoarseDropout(                                        # ✅ fixed API
        num_holes_range=(1, 4),
        hole_height_range=(10, 20),
        hole_width_range=(10, 20),
        p=0.15
    ),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2(),
])


# ── 5. Dataset class ──────────────────────────────────────────
class FundusDataset(Dataset):
    def __init__(self, amd_dir: str, normal_dir: str, transform=None):
        self.transform = transform
        self.paths, self.labels = [], []

        for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
            for p in pathlib.Path(normal_dir).glob(ext):
                self.paths.append(str(p))
                self.labels.append(0)
            for p in pathlib.Path(amd_dir).glob(ext):
                self.paths.append(str(p))
                self.labels.append(1)

        print(f"  Normal : {self.labels.count(0):4d}")
        print(f"  AMD    : {self.labels.count(1):4d}")
        print(f"  Total  : {len(self.paths):4d}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img   = preprocess_image(self.paths[idx])
        label = self.labels[idx]
        if self.transform:
            img = self.transform(image=img)["image"]
        return img, torch.tensor(label, dtype=torch.float32)


# ── 6. Build datasets + dataloaders ──────────────────────────
print("\n[Train]")
train_dataset = FundusDataset(CFG["TRAIN_AMD"], CFG["TRAIN_NORMAL"], train_transform)

print("\n[Val]")
val_dataset = FundusDataset(CFG["VAL_AMD"], CFG["VAL_NORMAL"], val_transform)

# ✅ WeightedRandomSampler — fixes class imbalance (Normal:AMD = 3.6:1)
labels       = train_dataset.labels
class_counts = [labels.count(0), labels.count(1)]
weights      = [1.0 / class_counts[l] for l in labels]
sampler      = WeightedRandomSampler(
                   weights,
                   num_samples=len(weights),
                   replacement=True
               )

train_loader = DataLoader(
    train_dataset,
    batch_size=CFG["BATCH_SIZE"],
    sampler=sampler,            # ✅ replaces shuffle=True
    num_workers=CFG["NUM_WORKERS"],
    pin_memory=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=CFG["BATCH_SIZE"],
    shuffle=False,
    num_workers=CFG["NUM_WORKERS"],
    pin_memory=True,
)

print(f"\nTrain batches : {len(train_loader)}")
print(f"Val   batches : {len(val_loader)}")

# ── 7. Sanity check ───────────────────────────────────────────
imgs, lbls = next(iter(train_loader))
print(f"\n[Sanity] Batch shape  : {imgs.shape}")
print(f"[Sanity] dtype        : {imgs.dtype}")
print(f"[Sanity] value range  : [{imgs.min():.3f}, {imgs.max():.3f}]")
print(f"[Sanity] Labels       : {lbls.numpy()}")
print(f"[Sanity] AMD in batch : {int(lbls.sum())}")
print(f"[Sanity] Norm in batch: {int((lbls==0).sum())}")

# ===== CELL 2 =====
# ============================================================
#  CELL 2 — Model: DeiT-S + Two-Phase Training
#  Fixes applied: LAB CLAHE, WeightedSampler, Focal alpha=0.65,
#                 simplified head, fixed run_epoch grad context
# ============================================================

import timm, torch, torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import f1_score, classification_report
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {DEVICE}")


# ── 1. Focal Loss ─────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.65, gamma=2.0):
        # ✅ alpha=0.65 (was 0.5) — penalises AMD misses harder
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce   = nn.functional.binary_cross_entropy_with_logits(
                    logits, targets, reduction="none"
                )
        probs = torch.sigmoid(logits)
        pt    = torch.where(targets == 1, probs, 1 - probs)
        alpha = torch.where(
                    targets == 1,
                    torch.tensor(self.alpha, device=logits.device),
                    torch.tensor(1 - self.alpha, device=logits.device)
                )
        focal = alpha * (1 - pt) ** self.gamma * bce
        return focal.mean()


# ── 2. DeiT-S Model ───────────────────────────────────────────
# DeiT-S: 22M params, 12 transformer blocks, embed_dim=384
# Much better than ResNet-50 for small medical datasets because
# self-attention captures global retinal structure (drusen patterns
# spread across the fundus) that CNNs miss with local receptive fields.

class DeiTBinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        # ✅ num_classes=0 removes timm's default head → gives (B, 384) features
        self.backbone = timm.create_model(
            "deit_small_patch16_224",
            pretrained=True,
            num_classes=0
        )
        # ✅ Simplified head — 333 AMD samples can't support 3-layer MLP
        self.head = nn.Sequential(
            nn.LayerNorm(384),       # stabilises transformer output variance
            nn.Dropout(0.4),
            nn.Linear(384, 256),
            nn.GELU(),               # GELU matches DeiT's internal activation
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.head(self.backbone(x))

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True

    def unfreeze_last_n_blocks(self, n: int = 4):
        """
        Gradual unfreeze — only unfreeze last n transformer blocks.
        Safer than unfreezing all at once for small datasets.
        """
        # Freeze everything first
        for p in self.backbone.parameters():
            p.requires_grad = False
        # Unfreeze last n blocks + norm layer
        blocks = list(self.backbone.blocks)
        for block in blocks[-n:]:
            for p in block.parameters():
                p.requires_grad = True
        for p in self.backbone.norm.parameters():
            p.requires_grad = True


model = DeiTBinaryClassifier().to(DEVICE)
criterion = FocalLoss(alpha=0.65, gamma=2.0)

total     = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total params     : {total:,}")
print(f"Trainable params : {trainable:,}")


# ── 3. run_epoch ──────────────────────────────────────────────
THRESHOLD = 0.5
SAVE_PATH = "/kaggle/working/best_deit_model.pth"


def run_epoch(loader, train=True):
    """✅ Fixed: no broken enable_grad context manager."""
    model.train() if train else model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    if train:
        for imgs, labels in loader:
            imgs   = imgs.to(DEVICE)
            labels = labels.to(DEVICE).unsqueeze(1)

            optimizer.zero_grad()
            logits = model(imgs)
            loss   = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            preds = (torch.sigmoid(logits) >= THRESHOLD).float()
            all_preds.extend(preds.detach().cpu().numpy().flatten())
            all_labels.extend(labels.detach().cpu().numpy().flatten())
    else:
        with torch.no_grad():
            for imgs, labels in loader:
                imgs   = imgs.to(DEVICE)
                labels = labels.to(DEVICE).unsqueeze(1)

                logits = model(imgs)
                loss   = criterion(logits, labels)

                total_loss += loss.item() * imgs.size(0)
                preds = (torch.sigmoid(logits) >= THRESHOLD).float()
                all_preds.extend(preds.cpu().numpy().flatten())
                all_labels.extend(labels.cpu().numpy().flatten())

    n   = len(all_labels)
    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / n
    f1  = f1_score(all_labels, all_preds, zero_division=0)
    return total_loss / n, acc, f1, all_preds, all_labels


# ── 4. Phase 1: head only (10 epochs) ────────────────────────
# DeiT transformers are more sensitive than CNNs to large LR early on.
# Always warm up the head first or the attention weights destabilise.

print("\n" + "="*60)
print("PHASE 1 — Head only, backbone frozen")
print("="*60)

model.freeze_backbone()
optimizer = AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-3,
    weight_decay=1e-4
)
scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-5)

PHASE1_EPOCHS = 10
best_val_f1   = 0.0

for epoch in range(1, PHASE1_EPOCHS + 1):
    tr_loss, tr_acc, tr_f1, _, _ = run_epoch(train_loader, train=True)
    vl_loss, vl_acc, vl_f1, _, _ = run_epoch(val_loader, train=False)
    scheduler.step()

    saved = ""
    if vl_f1 > best_val_f1:
        best_val_f1 = vl_f1
        torch.save(model.state_dict(), SAVE_PATH)
        saved = "✓ saved"

    print(
        f"[P1] Epoch {epoch:02d}/{PHASE1_EPOCHS} | "
        f"Train loss: {tr_loss:.4f}  acc: {tr_acc:.4f}  f1: {tr_f1:.4f} | "
        f"Val loss: {vl_loss:.4f}  acc: {vl_acc:.4f}  f1: {vl_f1:.4f}  {saved}"
    )


# ── 5. Phase 2: unfreeze last 4 blocks only (20 epochs) ──────
# ✅ DeiT-specific: unfreeze last 4 blocks instead of full backbone.
# Full unfreeze of 12 transformer blocks on 333 AMD images = guaranteed overfit.
# Last 4 blocks hold task-specific high-level features; earlier blocks
# hold low-level texture/edge features that don't need to change.

print("\n" + "="*60)
print("PHASE 2 — Last 4 transformer blocks + head unfrozen")
print("="*60)

model.unfreeze_last_n_blocks(n=4)

# Separate LR groups: backbone blocks get 10x lower LR than head
backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
head_params     = list(model.head.parameters())

optimizer = AdamW(
    [
        {"params": backbone_params, "lr": 1e-5},   # very low for transformer blocks
        {"params": head_params,     "lr": 1e-4},
    ],
    weight_decay=1e-4
)

PHASE2_EPOCHS = 20
scheduler = CosineAnnealingLR(optimizer, T_max=PHASE2_EPOCHS, eta_min=1e-6)

for epoch in range(1, PHASE2_EPOCHS + 1):
    tr_loss, tr_acc, tr_f1, _, _ = run_epoch(train_loader, train=True)
    vl_loss, vl_acc, vl_f1, _, _ = run_epoch(val_loader, train=False)
    scheduler.step()

    saved = ""
    if vl_f1 > best_val_f1:
        best_val_f1 = vl_f1
        torch.save(model.state_dict(), SAVE_PATH)
        saved = "✓ saved"

    print(
        f"[P2] Epoch {epoch:02d}/{PHASE2_EPOCHS} | "
        f"Train loss: {tr_loss:.4f}  acc: {tr_acc:.4f}  f1: {tr_f1:.4f} | "
        f"Val loss: {vl_loss:.4f}  acc: {vl_acc:.4f}  f1: {vl_f1:.4f}  {saved}"
    )


# ── 6. Final evaluation with threshold sweep ──────────────────
print("\n" + "="*60)
print("FINAL EVALUATION — threshold sweep")
print("="*60)

model.load_state_dict(torch.load(SAVE_PATH, map_location=DEVICE))
model.eval()

all_probs, all_labels = [], []
with torch.no_grad():
    for imgs, labels in val_loader:
        logits = model(imgs.to(DEVICE))
        probs  = torch.sigmoid(logits).cpu().numpy().flatten()
        all_probs.extend(probs)
        all_labels.extend(labels.numpy().flatten())

all_probs  = np.array(all_probs)
all_labels = np.array(all_labels)

print("\nThreshold sweep (F1 / recall / precision):")
best_thresh, best_f1 = 0.5, 0.0
for t in np.arange(0.3, 0.7, 0.05):
    preds = (all_probs >= t).astype(float)
    f1    = f1_score(all_labels, preds, zero_division=0)
    tp    = ((preds == 1) & (all_labels == 1)).sum()
    fn    = ((preds == 0) & (all_labels == 1)).sum()
    rec   = tp / (tp + fn + 1e-8)
    print(f"  t={t:.2f}  f1={f1:.4f}  recall={rec:.4f}")
    if f1 > best_f1:
        best_f1, best_thresh = f1, t

print(f"\nBest threshold : {best_thresh:.2f}  (F1={best_f1:.4f})")
final_preds = (all_probs >= best_thresh).astype(float)
print("\n" + classification_report(
    all_labels, final_preds,
    target_names=CFG["CLASS_NAMES"]
))
print(f"Model saved to : {SAVE_PATH}")

