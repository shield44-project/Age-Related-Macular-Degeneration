"""
AMD classifier training pipeline.

Recommended architecture
------------------------
The default backbone is ``convnext_tiny.fb_in22k_ft_in1k``. ConvNeXt-Tiny is the
sweet spot for AMD fundus classification on a single mid-range GPU:

* It matches Swin-V2-Tiny / ViT-B accuracy on retinal benchmarks
  (iChallenge-AMD, ADAM, ODIR-5K) while being ~3× smaller and ~2× faster to
  fine-tune.
* It works very well with imbalanced retinal datasets (class-weighted
  cross-entropy + label smoothing converge cleanly in <20 epochs).
* Pretrained on ImageNet-22k → strong low-level filters that transfer well to
  fundus imagery.

If you have a beefier GPU (A100, 3090, 4090, T4 with bigger batch), you can
swap to ``--arch swinv2_tiny_window8_256.ms_in1k`` (a couple of % more F1) or
``--arch efficientnetv2_s.in21k_ft_in1k`` for very fast inference.

Dataset layout
--------------
Each ``--dataset-roots`` entry must follow ImageFolder layout::

    <root>/train/Normal/*.jpg
    <root>/train/AMD/*.jpg
    <root>/val/Normal/*.jpg
    <root>/val/AMD/*.jpg
    <root>/test/Normal/*.jpg   (optional)
    <root>/test/AMD/*.jpg      (optional)

You can pass several roots and they'll be concatenated, e.g.::

    python -m backend.train_and_evaluate \
        --dataset-roots data/adam data/odir5k_amd \
        --arch convnext_tiny.fb_in22k_ft_in1k \
        --epochs 25 --batch-size 32 --image-size 224
"""
from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import timm
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from tqdm import tqdm

CLASS_NAMES = ["Normal", "AMD"]
NUM_CLASSES = 2

# Default backbone — ConvNeXt-Tiny is the recommended starting point for AMD.
DEFAULT_ARCH = "convnext_tiny.fb_in22k_ft_in1k"
DEFAULT_IMAGE_SIZE = 224

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class TrainConfig:
    dataset_roots: list[Path]
    output_path: Path
    arch: str = DEFAULT_ARCH
    image_size: int = DEFAULT_IMAGE_SIZE
    batch_size: int = 32
    epochs: int = 25
    lr: float = 3e-4
    weight_decay: float = 5e-2
    label_smoothing: float = 0.05
    warmup_epochs: int = 2
    train_workers: int = 4
    eval_workers: int = 2
    seed: int = 42
    early_stop_patience: int = 6
    use_amp: bool = True
    use_weighted_sampler: bool = True
    grad_clip: float = 1.0
    drop_path_rate: float = 0.1
    notes: str = field(default_factory=str)


# ---------------------------------------------------------------------------
# Reproducibility & data plumbing
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _imagefolder(path: Path, transform):
    if not path.exists() or not path.is_dir():
        return None
    ds = datasets.ImageFolder(str(path), transform=transform)
    if len(ds) == 0:
        return None
    return ds


def _build_split_dataset(dataset_roots: list[Path], split: str, transform):
    all_sets = []
    for root in dataset_roots:
        split_path = root / split
        ds = _imagefolder(split_path, transform)
        if ds is None:
            continue
        if set(ds.class_to_idx.keys()) != {"AMD", "Normal"}:
            raise ValueError(
                f"Unexpected class folders in {split_path}: {list(ds.class_to_idx.keys())}. "
                "Expected exactly Normal and AMD."
            )
        # Guarantee the same class ordering across datasets so labels concat correctly.
        ds.class_to_idx = {"Normal": 0, "AMD": 1}
        ds.classes = ["Normal", "AMD"]
        all_sets.append(ds)

    if not all_sets:
        raise FileNotFoundError(
            f"No datasets found for split '{split}'. "
            f"Expected folders like <root>/{split}/Normal and <root>/{split}/AMD"
        )

    return all_sets[0] if len(all_sets) == 1 else ConcatDataset(all_sets)


def build_transforms(image_size: int):
    """Augmentations tuned for fundus photographs.

    We avoid colour-space inversions and large hue shifts (real fundus images
    are warm-toned). Vertical flips are kept rare because the optic disc has
    natural left/right asymmetry."""
    train_tf = transforms.Compose([
        transforms.Resize((image_size + 16, image_size + 16)),
        transforms.RandomResizedCrop(image_size, scale=(0.85, 1.0), ratio=(0.95, 1.05)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.10),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.15, contrast=0.20, saturation=0.10, hue=0.02),
        transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.20
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        transforms.RandomErasing(p=0.10, scale=(0.02, 0.08), ratio=(0.5, 2.0)),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return train_tf, eval_tf


def _dataset_targets(dataset) -> list[int]:
    if isinstance(dataset, datasets.ImageFolder):
        return list(dataset.targets)
    if isinstance(dataset, ConcatDataset):
        targets: list[int] = []
        for sub in dataset.datasets:
            targets.extend(_dataset_targets(sub))
        return targets
    raise TypeError(f"Unsupported dataset type: {type(dataset)}")


def class_weights_from_dataset(dataset, device: torch.device) -> torch.Tensor:
    targets = _dataset_targets(dataset)
    counts = np.bincount(np.array(targets), minlength=NUM_CLASSES).astype(np.float64)
    counts[counts == 0] = 1.0
    inv = 1.0 / counts
    weights = inv / inv.sum() * NUM_CLASSES
    return torch.tensor(weights, dtype=torch.float32, device=device)


def make_weighted_sampler(dataset) -> WeightedRandomSampler:
    targets = np.array(_dataset_targets(dataset))
    counts = np.bincount(targets, minlength=NUM_CLASSES).astype(np.float64)
    counts[counts == 0] = 1.0
    sample_w = (1.0 / counts)[targets]
    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_w, dtype=torch.double),
        num_samples=len(targets),
        replacement=True,
    )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def create_model(arch: str, device: torch.device, drop_path_rate: float) -> nn.Module:
    extra: dict = {}
    if "convnext" in arch or "swin" in arch or "vit" in arch:
        extra["drop_path_rate"] = drop_path_rate
    model = timm.create_model(arch, pretrained=True, num_classes=NUM_CLASSES, **extra)
    return model.to(device)


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------
def _metric_block(y_true, y_pred, y_prob_amd):
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if len(set(y_true)) == 2:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob_amd))
        except ValueError:
            metrics["roc_auc"] = None
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    metrics["confusion_matrix"] = cm  # rows = true (Normal/AMD), cols = pred
    return metrics


def run_epoch(model, loader, criterion, optimizer, device, scaler, grad_clip):
    model.train()
    running_loss = 0.0
    y_true, y_pred, y_prob = [], [], []

    for images, labels in tqdm(loader, desc="Train", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            logits = model(images)
            loss = criterion(logits, labels)

        if scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        running_loss += float(loss.item()) * labels.size(0)
        probs = torch.softmax(logits.detach().float(), dim=1)
        preds = probs.argmax(dim=1)
        y_true.extend(labels.detach().cpu().tolist())
        y_pred.extend(preds.cpu().tolist())
        y_prob.extend(probs[:, 1].cpu().tolist())

    metrics = _metric_block(y_true, y_pred, y_prob)
    metrics["loss"] = running_loss / max(len(loader.dataset), 1)
    return metrics


@torch.no_grad()
def evaluate(model, loader, criterion, device, desc: str = "Eval"):
    model.eval()
    running_loss = 0.0
    y_true, y_pred, y_prob = [], [], []

    for images, labels in tqdm(loader, desc=desc, leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)

        running_loss += float(loss.item()) * labels.size(0)
        probs = torch.softmax(logits.float(), dim=1)
        preds = probs.argmax(dim=1)
        y_true.extend(labels.detach().cpu().tolist())
        y_pred.extend(preds.cpu().tolist())
        y_prob.extend(probs[:, 1].cpu().tolist())

    metrics = _metric_block(y_true, y_pred, y_prob)
    metrics["loss"] = running_loss / max(len(loader.dataset), 1)
    return metrics


def _build_scheduler(optimizer, cfg: TrainConfig, steps_per_epoch: int):
    """Linear warm-up followed by cosine decay over the remaining epochs."""
    total_steps = max(1, cfg.epochs * steps_per_epoch)
    warmup_steps = max(0, cfg.warmup_epochs * steps_per_epoch)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_and_evaluate(cfg: TrainConfig):
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | arch: {cfg.arch} | image size: {cfg.image_size}")

    train_tf, eval_tf = build_transforms(cfg.image_size)
    train_ds = _build_split_dataset(cfg.dataset_roots, "train", train_tf)
    val_ds = _build_split_dataset(cfg.dataset_roots, "val", eval_tf)

    test_ds = None
    try:
        test_ds = _build_split_dataset(cfg.dataset_roots, "test", eval_tf)
    except FileNotFoundError:
        print("No test split found. Final testing will be skipped.")

    pin = torch.cuda.is_available()
    if cfg.use_weighted_sampler:
        sampler = make_weighted_sampler(train_ds)
        train_loader = DataLoader(
            train_ds, batch_size=cfg.batch_size, sampler=sampler,
            num_workers=cfg.train_workers, pin_memory=pin, drop_last=True,
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=cfg.batch_size, shuffle=True,
            num_workers=cfg.train_workers, pin_memory=pin, drop_last=True,
        )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.eval_workers, pin_memory=pin,
    )
    test_loader = (
        DataLoader(
            test_ds, batch_size=cfg.batch_size, shuffle=False,
            num_workers=cfg.eval_workers, pin_memory=pin,
        )
        if test_ds is not None
        else None
    )

    model = create_model(cfg.arch, device, cfg.drop_path_rate)

    class_weights = class_weights_from_dataset(train_ds, device=device)
    print(f"Class weights (Normal, AMD): {class_weights.tolist()}")
    criterion = nn.CrossEntropyLoss(
        weight=class_weights, label_smoothing=cfg.label_smoothing
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = _build_scheduler(optimizer, cfg, steps_per_epoch=max(1, len(train_loader)))
    scaler = torch.cuda.amp.GradScaler() if (cfg.use_amp and torch.cuda.is_available()) else None

    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)

    best_state = None
    best_val_f1 = -1.0
    best_val_metrics: Optional[dict] = None
    no_improve = 0

    for epoch in range(1, cfg.epochs + 1):
        print(f"\nEpoch {epoch}/{cfg.epochs}")
        train_metrics = run_epoch(
            model, train_loader, criterion, optimizer, device, scaler, cfg.grad_clip
        )
        for _ in range(len(train_loader)):
            scheduler.step()

        val_metrics = evaluate(model, val_loader, criterion, device, desc="Val")

        print(
            "train: "
            f"loss={train_metrics['loss']:.4f} "
            f"acc={train_metrics['accuracy']:.4f} "
            f"f1={train_metrics['f1_score']:.4f}"
        )
        print(
            "val:   "
            f"loss={val_metrics['loss']:.4f} "
            f"acc={val_metrics['accuracy']:.4f} "
            f"prec={val_metrics['precision']:.4f} "
            f"rec={val_metrics['recall']:.4f} "
            f"f1={val_metrics['f1_score']:.4f} "
            f"auc={val_metrics.get('roc_auc')}"
        )

        if val_metrics["f1_score"] > best_val_f1:
            best_val_f1 = val_metrics["f1_score"]
            best_val_metrics = val_metrics
            no_improve = 0
            best_state = {
                "state_dict": model.state_dict(),
                "model_name": f"{cfg.arch} AMD Classifier",
                "arch": cfg.arch,
                "image_size": cfg.image_size,
                "class_names": CLASS_NAMES,
                "accuracy": val_metrics["accuracy"],
                "precision": val_metrics["precision"],
                "recall": val_metrics["recall"],
                "f1_score": val_metrics["f1_score"],
                "roc_auc": val_metrics.get("roc_auc"),
                "confusion_matrix": val_metrics["confusion_matrix"],
                "notes": cfg.notes,
            }
            torch.save(best_state, cfg.output_path)
            print(f"Saved improved checkpoint: {cfg.output_path}")
        else:
            no_improve += 1
            if no_improve >= cfg.early_stop_patience:
                print("Early stopping triggered.")
                break

    if best_state is None:
        raise RuntimeError("Training completed without a valid checkpoint.")

    report = {
        "config": {
            "arch": cfg.arch,
            "image_size": cfg.image_size,
            "batch_size": cfg.batch_size,
            "epochs": cfg.epochs,
            "lr": cfg.lr,
            "weight_decay": cfg.weight_decay,
            "label_smoothing": cfg.label_smoothing,
            "warmup_epochs": cfg.warmup_epochs,
            "use_amp": bool(scaler is not None),
            "use_weighted_sampler": cfg.use_weighted_sampler,
        },
        "best_validation": best_val_metrics,
        "checkpoint": str(cfg.output_path.resolve()),
    }

    if test_loader is not None:
        ckpt = torch.load(cfg.output_path, map_location=device)
        model.load_state_dict(ckpt["state_dict"], strict=True)
        test_metrics = evaluate(model, test_loader, criterion, device, desc="Test")
        report["test"] = test_metrics
        print(
            "test:  "
            f"loss={test_metrics['loss']:.4f} "
            f"acc={test_metrics['accuracy']:.4f} "
            f"prec={test_metrics['precision']:.4f} "
            f"rec={test_metrics['recall']:.4f} "
            f"f1={test_metrics['f1_score']:.4f} "
            f"auc={test_metrics.get('roc_auc')}"
        )

    report_path = cfg.output_path.with_suffix(".metrics.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Metrics report written to: {report_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train an AMD classifier and evaluate it. Each --dataset-roots entry "
            "must contain train/val/(optional)test/{Normal,AMD} folders."
        )
    )
    parser.add_argument("--dataset-roots", nargs="+", required=True)
    parser.add_argument(
        "--output",
        default="backend/models/ViT_base/best_vit_model_improved.pth",
        help="Path to save the best checkpoint (kept under ViT_base for "
             "drop-in compatibility with the existing GUI loader).",
    )
    parser.add_argument("--arch", default=DEFAULT_ARCH,
                        help="timm architecture name. Default: convnext_tiny.fb_in22k_ft_in1k")
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-2)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early-stop-patience", type=int, default=6)
    parser.add_argument("--train-workers", type=int, default=4)
    parser.add_argument("--eval-workers", type=int, default=2)
    parser.add_argument("--no-amp", action="store_true",
                        help="Disable mixed-precision (use this on CPU-only training).")
    parser.add_argument("--no-weighted-sampler", action="store_true")
    parser.add_argument("--notes", default="",
                        help="Free-form note saved into the checkpoint metadata.")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = TrainConfig(
        dataset_roots=[Path(p).expanduser().resolve() for p in args.dataset_roots],
        output_path=Path(args.output).expanduser().resolve(),
        arch=args.arch,
        image_size=args.image_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        warmup_epochs=args.warmup_epochs,
        seed=args.seed,
        early_stop_patience=args.early_stop_patience,
        train_workers=args.train_workers,
        eval_workers=args.eval_workers,
        use_amp=not args.no_amp,
        use_weighted_sampler=not args.no_weighted_sampler,
        notes=args.notes,
    )
    train_and_evaluate(cfg)


if __name__ == "__main__":
    main()
