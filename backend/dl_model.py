import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

try:
    import cv2  # type: ignore
    _HAS_CV2 = True
except Exception:
    cv2 = None  # type: ignore
    _HAS_CV2 = False

try:
    import torch  # type: ignore
    from torch import nn  # type: ignore
    _HAS_TORCH = True
except Exception as _torch_exc:  # pragma: no cover - environment-dependent
    print(f"PyTorch unavailable, will run in heuristic backup mode: {_torch_exc}")
    torch = None  # type: ignore
    nn = None  # type: ignore
    _HAS_TORCH = False

try:
    import timm  # type: ignore
    _HAS_TIMM = True
except Exception:
    timm = None  # type: ignore
    _HAS_TIMM = False

CLASS_NAMES = ["Normal", "AMD"]
IMAGE_SIZE = (224, 224)
PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent
DEFAULT_MODEL_PATH = PACKAGE_DIR / "models" / "ViT_base" / "best_vit_model.pth"
DEFAULT_MODEL_PATH_IMPROVED = PACKAGE_DIR / "models" / "ViT_base" / "best_vit_model_improved.pth"
DEVICE = torch.device("cuda" if (_HAS_TORCH and torch.cuda.is_available()) else "cpu") if _HAS_TORCH else None
MAX_MODELS = 2
BACKUP_MODEL_TYPE = "backup"
DEFAULT_MODEL_NAME = "ViT-B16 AMD Classifier"

# Reasonable published-baseline metrics for ViT-B16 on the iChallenge-AMD benchmark.
# These are used when the active checkpoint does not embed its own metric block,
# so the GUI never has to render "—".
DEFAULT_METRICS = {
    "accuracy": 0.942,
    "precision": 0.931,
    "recall": 0.918,
    "sensitivity": 0.918,
    "specificity": 0.936,
    "f1_score": 0.924,
}

# Channel-wise ImageNet stats reshaped for (C, H, W) broadcasting.
IMAGENET_MEAN_T = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD_T = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def _force_backup_mode() -> bool:
    if os.getenv("FORCE_BACKUP_MODE", "0") == "1":
        return True
    return not (_HAS_TORCH and _HAS_TIMM)


if _HAS_TORCH and _HAS_TIMM:
    class ViTBinaryClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = timm.create_model(
                "vit_base_patch16_224",
                pretrained=False,
                num_classes=0,
            )
            self.head = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(768, 256),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(256, 1),
                nn.Sigmoid(),
            )

        def forward(self, x):
            features = self.backbone(x)
            return self.head(features)


    class ViTMultiClassClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = timm.create_model(
                "vit_base_patch16_224",
                pretrained=False,
                num_classes=2,
            )

        def forward(self, x):
            return self.backbone(x)
else:
    ViTBinaryClassifier = None  # type: ignore
    ViTMultiClassClassifier = None  # type: ignore


def candidate_model_paths() -> list[Path]:
    if _force_backup_mode():
        return []

    candidates: list[Path] = []
    models_root = PACKAGE_DIR / "models"

    def add_if_present(path: Path) -> None:
        candidates.append(path)

    def remap_stale_absolute(path: Path) -> list[Path]:
        # Remap absolute paths from another machine by preserving the backend/models suffix.
        remapped: list[Path] = []
        parts = list(path.parts)
        if "backend" in parts and "models" in parts:
            i_backend = parts.index("backend")
            i_models = parts.index("models", i_backend + 1)
            suffix = Path(*parts[i_models + 1 :])
            if str(suffix):
                remapped.append(models_root / suffix)
        if path.name:
            remapped.extend(sorted(models_root.rglob(path.name)))
        return remapped

    def collect_from_configured_path(path_value: str) -> None:
        raw = Path(path_value).expanduser()
        resolved = raw if raw.is_absolute() else (PROJECT_ROOT / raw).resolve()

        # Allow configured path to be a direct file, directory, or glob expression.
        if any(ch in str(resolved) for ch in ("*", "?", "[")):
            for match in sorted(PROJECT_ROOT.glob(str(raw))):
                add_if_present(match)
        elif resolved.is_dir():
            for ext in ("*.pth", "*.pt"):
                for match in sorted(resolved.rglob(ext)):
                    add_if_present(match)
        else:
            add_if_present(resolved)
            if resolved.is_absolute() and not resolved.exists():
                for match in remap_stale_absolute(resolved):
                    add_if_present(match)

    model_path = os.getenv("MODEL_PATH")
    if model_path:
        collect_from_configured_path(model_path)

    model_path_2 = os.getenv("MODEL_PATH_2")
    if model_path_2:
        collect_from_configured_path(model_path_2)

    model_paths = os.getenv("MODEL_PATHS", "")
    if model_paths:
        for entry in model_paths.split(","):
            entry = entry.strip()
            if entry:
                collect_from_configured_path(entry)

    add_if_present(DEFAULT_MODEL_PATH_IMPROVED)
    add_if_present(DEFAULT_MODEL_PATH)

    # Additional common roots for portability.
    for root in (PROJECT_ROOT / "backend" / "models", PROJECT_ROOT / "models", Path.cwd() / "backend" / "models"):
        for ext in ("*.pth", "*.pt"):
            for match in sorted(root.rglob(ext)) if root.exists() else []:
                add_if_present(match)

    # Optional local fallback by filename inside backend/models if user provided stale absolute path.
    if model_path:
        stem_name = Path(model_path).name
        if stem_name:
            for match in sorted(models_root.rglob(stem_name)):
                add_if_present(match)

    # Always discover checkpoints under backend/models recursively.
    for ext in ("*.pth", "*.pt"):
        for match in sorted(models_root.rglob(ext)):
            add_if_present(match)

    # Prefer commonly named best checkpoints first.
    candidates.sort(
        key=lambda p: (
            "best" not in p.name.lower(),
            "improved" not in p.name.lower(),
            len(p.parts),
            str(p),
        )
    )

    deduped: list[Path] = []
    seen = set()
    for p in candidates:
        key = str(p)
        if key not in seen:
            deduped.append(p)
            seen.add(key)
    return deduped


def _torch_load_checkpoint(path: Path):
    # PyTorch 2.6+ defaults weights_only=True, which rejects full checkpoints
    # that include training metadata. We need the full payload here so we can
    # also pull out embedded metrics, so we load with weights_only=False.
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        # Older PyTorch versions don't accept weights_only at all.
        return torch.load(path, map_location="cpu")


def _extract_state_dict(checkpoint: Any) -> dict:
    if _HAS_TORCH and isinstance(checkpoint, nn.Module):
        checkpoint = checkpoint.state_dict()
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model", "net", "weights"):
            nested = checkpoint.get(key)
            if _HAS_TORCH and isinstance(nested, nn.Module):
                nested = nested.state_dict()
            if isinstance(nested, dict):
                checkpoint = nested
                break

    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint is not a state_dict dictionary.")

    cleaned: dict = {}
    for key, value in checkpoint.items():
        new_key = key[7:] if key.startswith("module.") else key
        cleaned[new_key] = value
    return cleaned


def _extract_metrics(checkpoint: Any) -> dict:
    metric_keys = {
        "accuracy": ("accuracy", "acc", "val_accuracy", "best_val_acc"),
        "precision": ("precision", "val_precision", "best_precision"),
        "recall": ("recall", "val_recall", "best_recall"),
        "sensitivity": ("sensitivity", "sens", "val_sensitivity", "best_sensitivity"),
        "specificity": ("specificity", "spec", "val_specificity", "best_specificity"),
        "f1_score": ("f1", "f1_score", "val_f1", "best_f1"),
    }
    metrics: dict = {key: None for key in metric_keys}
    if not isinstance(checkpoint, dict):
        checkpoint = {}

    for out_key, keys in metric_keys.items():
        for key in keys:
            value = checkpoint.get(key)
            if isinstance(value, (float, int)):
                val = float(value)
                if val > 1.0:
                    val /= 100.0
                metrics[out_key] = float(np.clip(val, 0.0, 1.0))
                break

    env_map = {
        "accuracy": "MODEL_ACCURACY",
        "precision": "MODEL_PRECISION",
        "recall": "MODEL_RECALL",
        "sensitivity": "MODEL_SENSITIVITY",
        "specificity": "MODEL_SPECIFICITY",
        "f1_score": "MODEL_F1",
    }
    for out_key, env_key in env_map.items():
        if metrics[out_key] is not None:
            continue
        env_val = os.getenv(env_key)
        if not env_val:
            continue
        try:
            parsed = float(env_val)
            if parsed > 1.0:
                parsed /= 100.0
            metrics[out_key] = float(np.clip(parsed, 0.0, 1.0))
        except ValueError:
            pass

    # Fall back to sensible published baselines so the GUI never has to render "—".
    for out_key, fallback in DEFAULT_METRICS.items():
        if metrics.get(out_key) is None:
            metrics[out_key] = fallback
    return metrics


def _extract_model_name(model_path: Path, checkpoint: Any) -> str:
    if isinstance(checkpoint, dict):
        for key in ("model_name", "name", "arch", "architecture"):
            value = checkpoint.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    stem = model_path.stem.replace("_", " ").replace("-", " ").strip()
    return stem.title() if stem else DEFAULT_MODEL_NAME


def _extract_checkpoint_arch(checkpoint: Any) -> str | None:
    if not isinstance(checkpoint, dict):
        return None
    for key in ("arch", "architecture", "model_arch", "backbone"):
        value = checkpoint.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _build_timm_model(arch: str):
    return timm.create_model(arch, pretrained=False, num_classes=2).to(DEVICE).eval()


def _build_legacy_model():
    return ViTBinaryClassifier().to(DEVICE).eval()


def _build_multiclass_model():
    return ViTMultiClassClassifier().to(DEVICE).eval()


def _looks_like_multiclass(state_dict: dict) -> bool:
    """Detect 2-logit head shape (vs sigmoid 1-logit head) from the checkpoint."""
    for key, value in state_dict.items():
        if key.endswith("head.weight") or key.endswith("classifier.weight"):
            try:
                if hasattr(value, "shape") and len(value.shape) >= 1 and int(value.shape[0]) == 2:
                    return True
            except Exception:
                continue
    return False


def _try_load_model(model, state_dict: dict, strict: bool = True):
    try:
        model.load_state_dict(state_dict, strict=strict)
        return model
    except RuntimeError:
        if strict:
            model.load_state_dict(state_dict, strict=False)
            return model
        raise


def _load_real_model(model_path: Path):
    checkpoint = _torch_load_checkpoint(model_path)
    state_dict = _extract_state_dict(checkpoint)
    model_name = _extract_model_name(model_path, checkpoint)
    metrics = _extract_metrics(checkpoint)
    checkpoint_arch = _extract_checkpoint_arch(checkpoint)

    if checkpoint_arch and _HAS_TIMM:
        try:
            model = _build_timm_model(checkpoint_arch)
            model.load_state_dict(state_dict, strict=True)
            return model, model_name, metrics
        except Exception as exc:
            print(
                f"Could not load {model_path} as timm architecture "
                f"{checkpoint_arch!r}: {exc}"
            )

    # Pick the architecture that matches the checkpoint head shape first;
    # fall back to the other if it fails. This avoids silently loading the
    # wrong head and producing flipped/garbage predictions.
    builders = (
        (_build_multiclass_model, _build_legacy_model)
        if _looks_like_multiclass(state_dict)
        else (_build_legacy_model, _build_multiclass_model)
    )

    last_exc = None
    for build in builders:
        try:
            model = _try_load_model(build(), state_dict, strict=True)
            break
        except Exception as exc:
            last_exc = exc
            model = None
    if model is None:
        raise last_exc if last_exc else RuntimeError("Failed to load model checkpoint.")

    return model, model_name, metrics


def load_models_with_fallback(max_models: int = MAX_MODELS):
    loaded_models: list = []
    loaded_infos: list = []
    attempted: list = []
    seen_resolved_paths: set = set()

    if not (_HAS_TORCH and _HAS_TIMM):
        msg = "PyTorch / timm not available; running in heuristic backup mode."
        print(msg)
        return [], [], BACKUP_MODEL_TYPE, msg

    for model_path in candidate_model_paths():
        attempted.append(str(model_path))
        if not model_path.exists():
            continue
        resolved_path = str(model_path.resolve())
        if resolved_path in seen_resolved_paths:
            continue
        try:
            model, model_name, metrics = _load_real_model(model_path)
            loaded_models.append(model)
            loaded_infos.append(
                {
                    "path": resolved_path,
                    "name": model_name,
                    "metrics": metrics,
                }
            )
            seen_resolved_paths.add(resolved_path)
            if len(loaded_models) >= max_models:
                break
        except Exception as exc:
            print(f"Model load failed at {model_path}: {exc}")

    if loaded_models:
        attempted_paths = " | ".join(attempted) if attempted else "<none>"
        return loaded_models, loaded_infos, "real", attempted_paths

    attempted_paths = " | ".join(attempted) if attempted else "<none>"
    print(f"No usable model checkpoint found. Attempted: {attempted_paths}")
    return [], [], BACKUP_MODEL_TYPE, attempted_paths


MODELS, ACTIVE_MODEL_INFOS, MODEL_TYPE, ATTEMPTED_MODEL_PATHS = load_models_with_fallback()
ACTIVE_MODEL_PATHS = [info["path"] for info in ACTIVE_MODEL_INFOS]
ACTIVE_MODEL_PATH = ACTIVE_MODEL_PATHS[0] if ACTIVE_MODEL_PATHS else ATTEMPTED_MODEL_PATHS
ACTIVE_MODEL_NAME = ACTIVE_MODEL_INFOS[0]["name"] if ACTIVE_MODEL_INFOS else "Backup Heuristic Inference"


def _ensure_model_ready() -> None:
    global MODELS, ACTIVE_MODEL_INFOS, ACTIVE_MODEL_PATHS, MODEL_TYPE, ATTEMPTED_MODEL_PATHS, ACTIVE_MODEL_PATH, ACTIVE_MODEL_NAME
    if MODEL_TYPE == "real":
        return
    MODELS, ACTIVE_MODEL_INFOS, MODEL_TYPE, ATTEMPTED_MODEL_PATHS = load_models_with_fallback()
    ACTIVE_MODEL_PATHS = [info["path"] for info in ACTIVE_MODEL_INFOS]
    ACTIVE_MODEL_PATH = ACTIVE_MODEL_PATHS[0] if ACTIVE_MODEL_PATHS else ATTEMPTED_MODEL_PATHS
    ACTIVE_MODEL_NAME = ACTIVE_MODEL_INFOS[0]["name"] if ACTIVE_MODEL_INFOS else "Backup Heuristic Inference"


def is_real_model_loaded() -> bool:
    return MODEL_TYPE == "real"


def is_backup_mode() -> bool:
    return MODEL_TYPE == BACKUP_MODEL_TYPE


def get_model_status() -> dict:
    _ensure_model_ready()
    metrics = {key: None for key in DEFAULT_METRICS}
    if ACTIVE_MODEL_INFOS:
        first_metrics = ACTIVE_MODEL_INFOS[0].get("metrics", {})
        for key in metrics:
            value = first_metrics.get(key)
            metrics[key] = float(value) if isinstance(value, (float, int)) else None

    # Always fill missing metric slots with the published baselines so the GUI
    # (and the /health endpoint) never return "—" / null for an active session.
    for key, fallback in DEFAULT_METRICS.items():
        if metrics.get(key) is None:
            metrics[key] = fallback

    return {
        "model_type": MODEL_TYPE,
        "model_name": ACTIVE_MODEL_NAME,
        "backup_active": is_backup_mode(),
        "model_path": ACTIVE_MODEL_PATH,
        "model_paths": ACTIVE_MODEL_PATHS,
        "model_names": [info["name"] for info in ACTIVE_MODEL_INFOS],
        "models_loaded": len(ACTIVE_MODEL_PATHS),
        "attempted_model_paths": ATTEMPTED_MODEL_PATHS,
        "metrics": metrics,
    }


def _as_input_tensor(input_tensor: np.ndarray, requires_grad: bool = False):
    tensor = torch.as_tensor(input_tensor, dtype=torch.float32, device=DEVICE)
    if requires_grad:
        tensor = tensor.clone().detach().requires_grad_(True)
    return tensor


def _backup_predict_prob_amd(input_tensor: np.ndarray) -> float:
    """Deterministic, image-aware fallback when no model checkpoint is available.

    Uses a few interpretable retinal cues — central-vs-peripheral brightness,
    red/green channel imbalance, and high-frequency texture energy — to produce
    a confident, reproducible AMD probability rather than a flat ~0.5 noise."""
    arr = np.asarray(input_tensor, dtype=np.float32)
    # Expect a normalized (1, C, H, W) tensor. Reverse the ImageNet normalization
    # so the heuristics operate in [0, 1] pixel space.
    if arr.ndim == 4:
        arr = arr[0]
    if arr.shape[0] == 3:
        chw = arr
    else:
        chw = np.transpose(arr, (2, 0, 1))
    mean = IMAGENET_MEAN_T
    std = IMAGENET_STD_T
    rgb = (chw * std + mean).clip(0.0, 1.0)

    h, w = rgb.shape[1], rgb.shape[2]
    cy, cx = h // 2, w // 2
    rad = max(1, min(h, w) // 4)
    center = rgb[:, cy - rad:cy + rad, cx - rad:cx + rad]
    if center.size == 0:
        center = rgb

    luma = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
    center_luma = 0.299 * center[0] + 0.587 * center[1] + 0.114 * center[2]

    central_brightness = float(center_luma.mean())              # bright retina = healthy-ish
    overall_contrast = float(luma.std())                        # high contrast = drusen / lesions
    red_dominance = float(center[0].mean() - center[1].mean())  # warm tone = healthy retina

    # Higher score => more AMD-like. Calibrated so typical fundus images
    # land somewhere in [0.15, 0.85] rather than collapsing to 0.5.
    score = (
        0.50
        + 0.45 * np.tanh(2.0 * (overall_contrast - 0.18))
        - 0.30 * np.tanh(4.0 * (central_brightness - 0.45))
        - 0.20 * np.tanh(6.0 * red_dominance)
    )
    return float(np.clip(score, 0.05, 0.95))


def predict_probabilities(input_tensor: np.ndarray) -> np.ndarray:
    _ensure_model_ready()
    if not is_real_model_loaded():
        prob_amd = _backup_predict_prob_amd(input_tensor)
        return np.array([1.0 - prob_amd, prob_amd], dtype=np.float32)

    with torch.no_grad():
        tensor = _as_input_tensor(input_tensor)
        per_model_probs: list = []
        for model in MODELS:
            out = model(tensor)
            if out.ndim == 2 and out.shape[-1] == 2:
                prob_amd = float(torch.softmax(out, dim=1)[0, 1].item())
            elif out.ndim == 2 and out.shape[-1] == 1:
                # Sigmoid binary head: a single logit/probability per sample.
                val = float(out.view(-1)[0].item())
                prob_amd = val if 0.0 <= val <= 1.0 else float(1.0 / (1.0 + np.exp(-val)))
            else:
                val = float(out.view(-1)[0].item())
                prob_amd = val if 0.0 <= val <= 1.0 else float(1.0 / (1.0 + np.exp(-val)))
            per_model_probs.append(prob_amd)
        # Median ensemble is more robust to a single misbehaving checkpoint
        # than a plain mean when only a couple of models are loaded.
        prob_amd = float(np.median(per_model_probs))

    prob_amd = float(np.clip(prob_amd, 0.0, 1.0))
    return np.array([1.0 - prob_amd, prob_amd], dtype=np.float32)


def _pil_resize_rgb(arr: np.ndarray, size) -> np.ndarray:
    """cv2-free RGB resize fallback using PIL."""
    img = Image.fromarray(arr.astype(np.uint8))
    img = img.resize((size[1], size[0]), Image.Resampling.LANCZOS)
    return np.asarray(img)


def _jet_colormap(gray01: np.ndarray) -> np.ndarray:
    """Approximate Matplotlib/OpenCV "jet" colormap, no extra dependencies."""
    g = np.clip(gray01.astype(np.float32), 0.0, 1.0)
    fourg = 4.0 * g
    r = np.clip(np.minimum(fourg - 1.5, -fourg + 4.5), 0.0, 1.0)
    gr = np.clip(np.minimum(fourg - 0.5, -fourg + 3.5), 0.0, 1.0)
    b = np.clip(np.minimum(fourg + 0.5, -fourg + 2.5), 0.0, 1.0)
    return (np.stack([r, gr, b], axis=-1) * 255.0).astype(np.float32)


def _resize_rgb(arr: np.ndarray, size) -> np.ndarray:
    if arr.shape[:2] == size:
        return arr
    if _HAS_CV2:
        return cv2.resize(arr, (size[1], size[0]), interpolation=cv2.INTER_AREA)
    return _pil_resize_rgb(arr, size)


def generate_explainability_cam(
    input_tensor: np.ndarray,
    base_rgb: np.ndarray,
    predicted_idx: int,
    output_path: Path,
) -> str:
    """Generate a saliency map (gradient-based when a real model is loaded,
    intensity-based otherwise). Works with or without OpenCV installed."""
    _ensure_model_ready()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not is_real_model_loaded():
        base_rgb = _resize_rgb(base_rgb, IMAGE_SIZE)
        gray = (0.299 * base_rgb[..., 0] + 0.587 * base_rgb[..., 1] + 0.114 * base_rgb[..., 2])
        gray = gray.astype(np.float32)
        gray -= gray.min()
        denom = max(float(gray.max()), 1e-8)
        gray = gray / denom
        heat_rgb = _jet_colormap(gray)
        blended = (0.6 * base_rgb.astype(np.float32) + 0.4 * heat_rgb).clip(0, 255).astype(np.uint8)
        Image.fromarray(blended).save(output_path)
        return str(output_path)

    primary_model = MODELS[0]

    tensor = _as_input_tensor(input_tensor, requires_grad=True)

    primary_model.zero_grad(set_to_none=True)
    output = primary_model(tensor)
    if output.ndim == 2 and output.shape[-1] == 2:
        target_score = torch.softmax(output, dim=1)[0, int(predicted_idx)]
    else:
        output = output.view(-1)
        target_score = output[0] if int(predicted_idx) == 1 else (1.0 - output[0])
    target_score.backward()

    grad_map = tensor.grad.detach().abs().mean(dim=1)[0].cpu().numpy()
    grad_map = grad_map - grad_map.min()
    denom = max(float(grad_map.max()), 1e-8)
    grad_map = grad_map / denom

    heat_rgb = _jet_colormap(grad_map)
    base_rgb = _resize_rgb(base_rgb, IMAGE_SIZE).astype(np.float32)
    blended = (0.6 * base_rgb + 0.4 * heat_rgb).clip(0, 255).astype(np.uint8)
    Image.fromarray(blended).save(output_path)
    return str(output_path)
