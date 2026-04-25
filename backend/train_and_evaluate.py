import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import timm
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

CLASS_NAMES = ["Normal", "AMD"]
NUM_CLASSES = 2
IMAGE_SIZE = 224


@dataclass
class TrainConfig:
    dataset_roots: list[Path]
    output_path: Path
    batch_size: int = 16
    epochs: int = 20
    lr: float = 3e-4
    weight_decay: float = 1e-4
    train_workers: int = 4
    eval_workers: int = 2
    seed: int = 42
    early_stop_patience: int = 5


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
        if ds.class_to_idx.keys() != {"AMD", "Normal"} and ds.class_to_idx.keys() != {"Normal", "AMD"}:
            raise ValueError(
                f"Unexpected class folders in {split_path}: {list(ds.class_to_idx.keys())}. "
                "Expected exactly Normal and AMD."
            )
        all_sets.append(ds)

    if not all_sets:
        raise FileNotFoundError(
            f"No datasets found for split '{split}'. Expected folders like <root>/{split}/Normal and <root>/{split}/AMD"
        )

    if len(all_sets) == 1:
        return all_sets[0]
    return ConcatDataset(all_sets)


def build_transforms():
    train_tf = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.1),
            transforms.RandomRotation(degrees=12),
            transforms.ColorJitter(brightness=0.15, contrast=0.2, saturation=0.15, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return train_tf, eval_tf


def _dataset_targets(dataset) -> list[int]:
    if isinstance(dataset, datasets.ImageFolder):
        return dataset.targets
    if isinstance(dataset, ConcatDataset):
        targets = []
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


def create_model(device: torch.device) -> nn.Module:
    model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=NUM_CLASSES)
    return model.to(device)


def run_epoch(model, loader, criterion, optimizer, device: torch.device):
    model.train()
    running_loss = 0.0
    y_true = []
    y_pred = []

    for images, labels in tqdm(loader, desc="Train", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += float(loss.item()) * labels.size(0)
        preds = logits.argmax(dim=1)
        y_true.extend(labels.detach().cpu().tolist())
        y_pred.extend(preds.detach().cpu().tolist())

    metrics = {
        "loss": running_loss / max(len(loader.dataset), 1),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
    }
    return metrics


@torch.no_grad()
def evaluate(model, loader, criterion, device: torch.device, desc: str = "Eval"):
    model.eval()
    running_loss = 0.0
    y_true = []
    y_pred = []

    for images, labels in tqdm(loader, desc=desc, leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)

        running_loss += float(loss.item()) * labels.size(0)
        preds = logits.argmax(dim=1)
        y_true.extend(labels.detach().cpu().tolist())
        y_pred.extend(preds.detach().cpu().tolist())

    metrics = {
        "loss": running_loss / max(len(loader.dataset), 1),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
    }
    return metrics


def train_and_evaluate(cfg: TrainConfig):
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_tf, eval_tf = build_transforms()
    train_ds = _build_split_dataset(cfg.dataset_roots, "train", train_tf)
    val_ds = _build_split_dataset(cfg.dataset_roots, "val", eval_tf)

    test_ds = None
    try:
        test_ds = _build_split_dataset(cfg.dataset_roots, "test", eval_tf)
    except FileNotFoundError:
        print("No test split found. Final testing will be skipped.")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.train_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.eval_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = (
        DataLoader(
            test_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.eval_workers,
            pin_memory=torch.cuda.is_available(),
        )
        if test_ds is not None
        else None
    )

    model = create_model(device)
    class_weights = class_weights_from_dataset(train_ds, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)

    best_state = None
    best_val_f1 = -1.0
    best_val_metrics = None
    no_improve = 0

    for epoch in range(1, cfg.epochs + 1):
        print(f"\nEpoch {epoch}/{cfg.epochs}")
        train_metrics = run_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device, desc="Val")
        scheduler.step()

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
            f"f1={val_metrics['f1_score']:.4f}"
        )

        if val_metrics["f1_score"] > best_val_f1:
            best_val_f1 = val_metrics["f1_score"]
            best_val_metrics = val_metrics
            no_improve = 0
            best_state = {
                "state_dict": model.state_dict(),
                "model_name": "ViT-B16 AMD Classifier (improved)",
                "accuracy": val_metrics["accuracy"],
                "precision": val_metrics["precision"],
                "recall": val_metrics["recall"],
                "f1_score": val_metrics["f1_score"],
                "class_names": CLASS_NAMES,
                "image_size": IMAGE_SIZE,
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
            f"precision={test_metrics['precision']:.4f} "
            f"recall={test_metrics['recall']:.4f} "
            f"f1={test_metrics['f1_score']:.4f}"
        )

    report_path = cfg.output_path.with_suffix(".metrics.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Metrics report written to: {report_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train an improved AMD classifier on one or more datasets and evaluate its accuracy. "
            "Each dataset root should contain train/val/(optional)test folders with class folders Normal and AMD."
        )
    )
    parser.add_argument(
        "--dataset-roots",
        nargs="+",
        required=True,
        help="One or more dataset root directories.",
    )
    parser.add_argument(
        "--output",
        default="backend/models/ViT_base/best_vit_model_improved.pth",
        help="Path to save the best improved checkpoint.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early-stop-patience", type=int, default=5)
    parser.add_argument("--train-workers", type=int, default=4)
    parser.add_argument("--eval-workers", type=int, default=2)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = TrainConfig(
        dataset_roots=[Path(p).expanduser().resolve() for p in args.dataset_roots],
        output_path=Path(args.output).expanduser().resolve(),
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        early_stop_patience=args.early_stop_patience,
        train_workers=args.train_workers,
        eval_workers=args.eval_workers,
    )
    train_and_evaluate(cfg)


if __name__ == "__main__":
    main()
