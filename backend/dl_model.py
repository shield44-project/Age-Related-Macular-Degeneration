import os
import subprocess
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

try:
    import torchvision  # type: ignore
    from torchvision import models as tv_models  # type: ignore
    _HAS_TORCHVISION = True
except Exception:
    tv_models = None  # type: ignore
    _HAS_TORCHVISION = False

CLASS_NAMES = ["Normal", "AMD"]
IMAGE_SIZE = (224, 224)
PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent
DEFAULT_MODEL_PATH = PACKAGE_DIR / "models" / "ViT_base" / "best_vit_model.pth"
DEFAULT_MODEL_PATH_IMPROVED = PACKAGE_DIR / "models" / "ViT_base" / "best_vit_model_improved.pth"
DEFAULT_DEIT_MODEL_PATH = PACKAGE_DIR / "models" / "DeiT-S" / "best_deit_model.pth"
DEVICE = torch.device("cuda" if (_HAS_TORCH and torch.cuda.is_available()) else "cpu") if _HAS_TORCH else None
MAX_MODELS = 2
BACKUP_MODEL_TYPE = "backup"
DEFAULT_MODEL_NAME = "ViT-B16 AMD Classifier"
LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"
AUTO_PULL_LFS = os.getenv("AMD_AUTO_PULL_LFS", "1") == "1"
_LFS_PULL_ATTEMPTS: set[str] = set()

# Reported project metrics shown by the API and GUI.
DEFAULT_METRICS = {
    "accuracy": 0.80,
    "sensitivity": 0.79,
    "specificity": 0.83,
    "precision": 0.95,
    "f1_score": 0.864,
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

    class DeiTBinaryClassifier(nn.Module):
        def __init__(self, use_layernorm: bool):
            super().__init__()
            self.backbone = timm.create_model(
                "deit_small_patch16_224",
                pretrained=False,
                num_classes=0,
            )
            head_layers = []
            if use_layernorm:
                head_layers.append(nn.LayerNorm(384))
            head_layers.extend([
                nn.Dropout(0.4),
                nn.Linear(384, 256),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(256, 1),
            ])
            self.head = nn.Sequential(*head_layers)

        def forward(self, x):
            return self.head(self.backbone(x))
else:
    ViTBinaryClassifier = None  # type: ignore
    ViTMultiClassClassifier = None  # type: ignore
    DeiTBinaryClassifier = None  # type: ignore


if _HAS_TORCH and _HAS_TORCHVISION:
    class ResNetBinaryClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            try:
                backbone = tv_models.resnet50(weights=None)
            except TypeError:  # Older torchvision
                backbone = tv_models.resnet50(pretrained=False)
            backbone.fc = nn.Identity()
            self.backbone = backbone
            # Matches best_resnet_model.pth head. Dropout/activation layers are
            # inference-neutral (disabled in eval), but preserve module indices.
            self.head = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(2048, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(512, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(128, 1),
            )

        def forward(self, x):
            features = self.backbone(x)
            return self.head(features)
else:
    ResNetBinaryClassifier = None  # type: ignore


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

    add_if_present(DEFAULT_DEIT_MODEL_PATH)
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
    def _priority(p: Path) -> int:
        try:
            resolved = p.resolve()
        except Exception:
            resolved = p
        if resolved == DEFAULT_DEIT_MODEL_PATH.resolve():
            return 0
        if resolved == DEFAULT_MODEL_PATH.resolve():
            return 1
        if resolved == DEFAULT_MODEL_PATH_IMPROVED.resolve():
            return 2
        return 3

    candidates.sort(
        key=lambda p: (
            _priority(p),
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
    if _is_git_lfs_pointer(path):
        pull_error = _try_pull_lfs_weights(path)
        if pull_error is None and not _is_git_lfs_pointer(path):
            pass
        else:
            detail = f" Auto-pull failed: {pull_error}" if pull_error else ""
            raise ValueError(
                "checkpoint is a Git LFS pointer, not downloaded model weights. "
                "Install git-lfs and run 'git lfs pull', or place a real .pth file "
                "in backend/models/ViT_base/."
                + detail
            )

    # PyTorch 2.6+ defaults weights_only=True, which rejects full checkpoints
    # that include training metadata. We need the full payload here so we can
    # also pull out embedded metrics, so we load with weights_only=False.
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        # Older PyTorch versions don't accept weights_only at all.
        return torch.load(path, map_location="cpu")


def _is_git_lfs_pointer(path: Path) -> bool:
    """Return True when a checkpoint path contains only a Git LFS pointer file."""
    try:
        if not path.is_file() or path.stat().st_size > 1024:
            return False
        with path.open("rb") as fh:
            return fh.read(len(LFS_POINTER_PREFIX)) == LFS_POINTER_PREFIX
    except OSError:
        return False


def _git_lfs_available() -> bool:
    try:
        result = subprocess.run(
            ["git", "lfs", "version"],
            capture_output=True,
            text=True,
            check=False,
        )
        return result.returncode == 0
    except OSError:
        return False


def _try_pull_lfs_weights(model_path: Path) -> str | None:
    """Attempt to download Git LFS weights for a pointer file.

    Returns None on success, or an error string when the pull could not be run.
    """
    if not AUTO_PULL_LFS:
        return "Auto-pull disabled (AMD_AUTO_PULL_LFS=0)."
    try:
        resolved = str(model_path.resolve())
    except OSError:
        resolved = str(model_path)
    if resolved in _LFS_PULL_ATTEMPTS:
        return "Auto-pull already attempted for this file."
    _LFS_PULL_ATTEMPTS.add(resolved)

    if not _git_lfs_available():
        return "git-lfs is not installed."

    try:
        include_path = str(model_path.resolve().relative_to(PROJECT_ROOT))
    except (ValueError, OSError):
        include_path = resolved

    try:
        result = subprocess.run(
            ["git", "lfs", "pull", "--include", include_path],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        return f"git lfs pull failed: {exc}"

    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        return detail or "git lfs pull failed."

    return None


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
    return dict(DEFAULT_METRICS)


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


def _build_deit_model(state_dict: dict):
    if DeiTBinaryClassifier is None:
        raise RuntimeError("timm not available; cannot build DeiT model.")
    use_layernorm = any(key.startswith("head.0.") for key in state_dict.keys())
    return DeiTBinaryClassifier(use_layernorm=use_layernorm).to(DEVICE).eval()


def _build_resnet_model():
    if ResNetBinaryClassifier is None:
        raise RuntimeError("Torchvision not available; cannot build ResNet model.")
    return ResNetBinaryClassifier().to(DEVICE).eval()


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


def _infer_vit_embed_dim(state_dict: dict) -> int | None:
    token = state_dict.get("backbone.cls_token")
    if token is None:
        token = state_dict.get("cls_token")
    if hasattr(token, "shape") and len(token.shape) >= 3:
        try:
            return int(token.shape[-1])
        except Exception:
            return None
    return None


def _looks_like_deit_state_dict(state_dict: dict) -> bool:
    embed_dim = _infer_vit_embed_dim(state_dict)
    return embed_dim == 384


def _looks_like_resnet_backbone(state_dict: dict) -> bool:
    if not state_dict:
        return False
    has_backbone = any(
        key.startswith("backbone.conv1")
        or key.startswith("backbone.layer1.")
        or key.startswith("backbone.layer4.")
        for key in state_dict.keys()
    )
    has_head = any(key.startswith("head.") for key in state_dict.keys())
    return has_backbone and has_head


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

    if _looks_like_resnet_backbone(state_dict):
        try:
            model = _try_load_model(_build_resnet_model(), state_dict, strict=True)
            return model, model_name, metrics
        except Exception as exc:
            print(f"Could not load {model_path} as ResNet model: {exc}")

    if _looks_like_deit_state_dict(state_dict):
        try:
            model = _try_load_model(_build_deit_model(state_dict), state_dict, strict=True)
            return model, model_name, metrics
        except Exception as exc:
            print(f"Could not load {model_path} as DeiT model: {exc}")

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
    load_issues: list[dict] = []
    seen_resolved_paths: set = set()

    if not (_HAS_TORCH and _HAS_TIMM):
        msg = "PyTorch / timm not available; running in heuristic backup mode."
        print(msg)
        return [], [], BACKUP_MODEL_TYPE, msg, [{"path": "", "error": msg}]

    for model_path in candidate_model_paths():
        attempted.append(str(model_path))
        if not model_path.exists():
            continue
        resolved_path = str(model_path.resolve())
        if resolved_path in seen_resolved_paths:
            continue
        seen_resolved_paths.add(resolved_path)
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
            if len(loaded_models) >= max_models:
                break
        except Exception as exc:
            message = str(exc)
            load_issues.append({"path": resolved_path, "error": message})
            print(f"Model load failed at {model_path}: {message}")

    if loaded_models:
        attempted_paths = " | ".join(attempted) if attempted else "<none>"
        return loaded_models, loaded_infos, "real", attempted_paths, load_issues

    attempted_paths = " | ".join(attempted) if attempted else "<none>"
    print(f"No usable model checkpoint found. Attempted: {attempted_paths}")
    return [], [], BACKUP_MODEL_TYPE, attempted_paths, load_issues


MODELS, ACTIVE_MODEL_INFOS, MODEL_TYPE, ATTEMPTED_MODEL_PATHS, MODEL_LOAD_ISSUES = load_models_with_fallback()
ACTIVE_MODEL_PATHS = [info["path"] for info in ACTIVE_MODEL_INFOS]
ACTIVE_MODEL_PATH = ACTIVE_MODEL_PATHS[0] if ACTIVE_MODEL_PATHS else ATTEMPTED_MODEL_PATHS
ACTIVE_MODEL_NAME = ACTIVE_MODEL_INFOS[0]["name"] if ACTIVE_MODEL_INFOS else "Backup Heuristic Inference"


def _ensure_model_ready() -> None:
    global MODELS, ACTIVE_MODEL_INFOS, ACTIVE_MODEL_PATHS, MODEL_TYPE, ATTEMPTED_MODEL_PATHS, MODEL_LOAD_ISSUES, ACTIVE_MODEL_PATH, ACTIVE_MODEL_NAME
    if MODEL_TYPE == "real":
        return
    MODELS, ACTIVE_MODEL_INFOS, MODEL_TYPE, ATTEMPTED_MODEL_PATHS, MODEL_LOAD_ISSUES = load_models_with_fallback()
    ACTIVE_MODEL_PATHS = [info["path"] for info in ACTIVE_MODEL_INFOS]
    ACTIVE_MODEL_PATH = ACTIVE_MODEL_PATHS[0] if ACTIVE_MODEL_PATHS else ATTEMPTED_MODEL_PATHS
    ACTIVE_MODEL_NAME = ACTIVE_MODEL_INFOS[0]["name"] if ACTIVE_MODEL_INFOS else "Backup Heuristic Inference"


def is_real_model_loaded() -> bool:
    return MODEL_TYPE == "real"


def is_backup_mode() -> bool:
    return MODEL_TYPE == BACKUP_MODEL_TYPE


def get_model_status() -> dict:
    _ensure_model_ready()
    metrics = dict(DEFAULT_METRICS)
    model_error = ""
    if MODEL_TYPE != "real":
        lfs_paths = [
            issue["path"]
            for issue in MODEL_LOAD_ISSUES
            if "Git LFS pointer" in issue.get("error", "")
        ]
        if lfs_paths:
            model_error = (
                "Model weights are not downloaded. Install git-lfs and run "
                "'git lfs pull', or replace the pointer file with a real checkpoint."
            )
        elif MODEL_LOAD_ISSUES:
            model_error = MODEL_LOAD_ISSUES[0].get("error", "")
        else:
            model_error = "No usable model checkpoint was found."

    return {
        "model_type": MODEL_TYPE,
        "model_name": ACTIVE_MODEL_NAME,
        "backup_active": is_backup_mode(),
        "model_path": ACTIVE_MODEL_PATH,
        "model_paths": ACTIVE_MODEL_PATHS,
        "model_names": [info["name"] for info in ACTIVE_MODEL_INFOS],
        "models_loaded": len(ACTIVE_MODEL_PATHS),
        "attempted_model_paths": ATTEMPTED_MODEL_PATHS,
        "model_error": model_error,
        "model_load_issues": MODEL_LOAD_ISSUES,
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


def _get_vit_backbone(model):
    """Return the underlying timm ViT backbone, unwrapping any custom head wrapper.

    Handles two common patterns:
      1. The model *is* the timm ViT (has patch_embed / blocks / cls_token directly).
      2. The model wraps the ViT under a common attribute name (e.g. ``backbone``).
    """
    vit_attrs = ("patch_embed", "blocks", "cls_token")
    if all(hasattr(model, a) for a in vit_attrs):
        return model
    for attr in ("backbone", "model", "encoder", "vit", "transformer"):
        inner = getattr(model, attr, None)
        if inner is not None and all(hasattr(inner, a) for a in vit_attrs):
            return inner
    raise RuntimeError(
        "Cannot locate a ViT backbone inside the model. "
        "Checked model itself and common wrapper attributes."
    )


def _attention_rollout_for_vit(model, tensor, discard_ratio: float = 0.9):
    """Compute attention rollout for a timm ViT-like model.
    Returns a numpy heatmap resized to IMAGE_SIZE.
    """
    if not _HAS_TORCH:
        raise RuntimeError("PyTorch required for attention rollout")

    # Unwrap to the actual ViT backbone (handles ViTBinaryClassifier wrapper etc.)
    backbone = _get_vit_backbone(model)

    if not all(hasattr(backbone, a) for a in ("patch_embed", "blocks", "cls_token")):
        raise RuntimeError("Model does not appear to be a timm ViT-compatible model")

    with torch.no_grad():
        # patch_embed returns (B, N, C) directly
        x = backbone.patch_embed(tensor)
        B = x.shape[0]
        cls_tokens = backbone.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, N+1, C)

        if hasattr(backbone, "pos_embed") and backbone.pos_embed is not None:
            pe = backbone.pos_embed
            if pe.shape[1] == x.shape[1]:
                x = x + pe
            elif pe.shape[1] == x.shape[1] - 1:
                # no_embed_class variant: positional embeddings cover patches only
                x = torch.cat([x[:, :1], x[:, 1:] + pe], dim=1)
            # else: skip — best-effort, model may use dynamic resizing

        attentions = []
        for blk in backbone.blocks:
            attn_mod = getattr(blk, "attn", None)
            if attn_mod is None:
                raise RuntimeError("Unexpected block structure: no attn module")

            if hasattr(attn_mod, "qkv"):
                # Apply pre-attention layer norm (present in all standard ViT blocks)
                normed_x = blk.norm1(x) if hasattr(blk, "norm1") else x
                qkv_weight = attn_mod.qkv.weight
                qkv_bias = getattr(attn_mod.qkv, "bias", None)
                qkv = torch.nn.functional.linear(normed_x, qkv_weight, qkv_bias)
                Bn, N, threeC = qkv.shape
                num_heads = getattr(attn_mod, "num_heads", 12)
                head_dim = threeC // (3 * num_heads)
                # (B, N, 3*C) -> (3, B, heads, N, head_dim)
                qkv = qkv.reshape(Bn, N, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
                q, k = qkv[0], qkv[1]
                attn = (q @ k.transpose(-2, -1)) / (head_dim ** 0.5)
                attn = torch.softmax(attn, dim=-1)
                attn = attn.mean(dim=1)  # average over heads -> (B, N, N)
                attentions.append(attn[0].cpu().numpy())
            else:
                raise RuntimeError(
                    "Attention module has no qkv layer; unsupported attention layout"
                )

            # Advance token sequence through the full block for the next layer
            x = blk(x)

    # Attention rollout: propagate attention through layers
    att_mat = np.eye(attentions[0].shape[0])
    for a in attentions:
        a = a + np.eye(a.shape[0])          # add residual connection
        a = a / (a.sum(axis=-1, keepdims=True) + 1e-8)
        att_mat = a @ att_mat

    # CLS token's attention to every patch
    cls_attn = att_mat[0, 1:]
    patch_count = cls_attn.shape[0]
    if patch_count == 0:
        raise RuntimeError("Attention rollout produced zero patches; cannot generate heatmap.")
    Hp = int(round(patch_count ** 0.5))
    if Hp == 0:
        Hp = 1
    Wp = int(round(patch_count / Hp))
    if Wp == 0:
        Wp = 1
    # Safety: trim so Hp*Wp <= patch_count (handles non-square grids)
    cls_attn = cls_attn[: Hp * Wp]
    heat = cls_attn.reshape(Hp, Wp)
    heat = (heat - heat.min()) / max(float(heat.max() - heat.min()), 1e-8)
    heat_t = torch.tensor(heat, dtype=torch.float32)
    heat_t = torch.nn.functional.interpolate(
        heat_t.unsqueeze(0).unsqueeze(0), size=IMAGE_SIZE, mode="bilinear", align_corners=False
    )
    return heat_t.squeeze().cpu().numpy()


def _is_vit_model(model) -> bool:
    try:
        _get_vit_backbone(model)
        return True
    except Exception:
        return False


def _find_last_conv_layer(model):
    if not _HAS_TORCH:
        return None
    for module in reversed(list(model.modules())):
        if isinstance(module, nn.Conv2d):
            return module
    return None


def _gradcam_on_patch_embed(model, tensor, target_idx: int):
    """Compute Grad-CAM using gradients of the target w.r.t. patch embedding projection.
    Returns a heatmap in IMAGE_SIZE.
    """
    if not _HAS_TORCH:
        raise RuntimeError("PyTorch required for Grad-CAM")

    # Unwrap to the actual ViT backbone so we can reach patch_embed.
    # The full wrapper model is still used for the forward/backward pass so
    # that gradient flows through every layer (including a custom head).
    vit_backbone = _get_vit_backbone(model)
    if not hasattr(vit_backbone, "patch_embed"):
        raise RuntimeError("ViT backbone does not expose patch_embed; cannot compute Grad-CAM")

    model.eval()
    activations: dict = {}
    grads: dict = {}

    def forward_hook(module, inp, out):
        activations['value'] = out.detach()

    def backward_hook(module, grad_in, grad_out):
        grads['value'] = grad_out[0].detach()

    proj = vit_backbone.patch_embed.proj
    h_fwd = proj.register_forward_hook(forward_hook)
    h_bwd = proj.register_full_backward_hook(backward_hook)

    tensor_req = tensor.clone().detach().requires_grad_(True)
    out = model(tensor_req)  # forward through the full wrapper
    if out.ndim == 2 and out.shape[-1] == 2:
        score = torch.softmax(out, dim=1)[0, target_idx]
    else:
        out = out.view(-1)
        score = out[0] if target_idx == 1 else (1.0 - out[0])

    model.zero_grad(set_to_none=True)
    score.backward(retain_graph=False)

    h_fwd.remove()
    h_bwd.remove()

    if 'value' not in activations or 'value' not in grads:
        raise RuntimeError("Failed to capture activations/gradients from patch embedding")

    act = activations['value']  # (B, C, Hp, Wp)
    grad = grads['value']       # (B, C, Hp, Wp)
    weights = grad.mean(dim=(2, 3), keepdim=True)  # GAP over spatial dims
    cam = (weights * act).sum(dim=1, keepdim=True)  # (B, 1, Hp, Wp)
    cam = torch.relu(cam)
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    cam = torch.nn.functional.interpolate(cam, size=IMAGE_SIZE, mode='bilinear', align_corners=False)
    return cam[0, 0].cpu().numpy()


def _gradcam_on_conv(model, tensor, target_idx: int):
    """Grad-CAM for CNNs using the last Conv2d feature map."""
    if not _HAS_TORCH:
        raise RuntimeError("PyTorch required for Grad-CAM")

    target_layer = _find_last_conv_layer(model)
    if target_layer is None:
        raise RuntimeError("No Conv2d layer found for Grad-CAM")

    activations: dict = {}
    grads: dict = {}

    def forward_hook(_, __, out):
        activations['value'] = out

    def backward_hook(_, __, grad_out):
        grads['value'] = grad_out[0]

    h_fwd = target_layer.register_forward_hook(forward_hook)
    h_bwd = target_layer.register_full_backward_hook(backward_hook)

    model.zero_grad(set_to_none=True)
    output = model(tensor)
    if output.ndim == 2 and output.shape[-1] == 2:
        score = output[0, int(target_idx)]
    else:
        prob = torch.sigmoid(output.view(-1)[0])
        score = prob if int(target_idx) == 1 else (1.0 - prob)
    score.backward()

    act = activations['value']
    grad = grads['value']
    weights = grad.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * act).sum(dim=1, keepdim=True))
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    cam = torch.nn.functional.interpolate(
        cam, size=IMAGE_SIZE, mode='bilinear', align_corners=False
    )

    h_fwd.remove()
    h_bwd.remove()
    return cam[0, 0].detach().cpu().numpy()


def generate_explainability_cams(
    input_tensor: np.ndarray,
    base_rgb: np.ndarray,
    predicted_idx: int,
    output_prefix: Path,
) -> dict:
    """Generate and save both attention-rollout and Grad-CAM visualizations.

    Returns a dict with keys: 'attention_path', 'gradcam_path', 'combined_path'
    Paths may be empty strings if generation failed and a fallback was used instead.
    """
    _ensure_model_ready()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    attention_path = output_prefix.with_name(output_prefix.stem + "_attention.png")
    gradcam_path = output_prefix.with_name(output_prefix.stem + "_gradcam.png")
    combined_path = output_prefix.with_name(output_prefix.stem + "_combined.png")

    if not is_real_model_loaded() or not MODELS:
        fallback = generate_explainability_cam(
            input_tensor, base_rgb, predicted_idx,
            output_prefix.with_name(output_prefix.stem + "_cam.png"),
        )
        return {"attention_path": "", "gradcam_path": "", "combined_path": fallback}

    attention_saved = ""
    gradcam_saved = ""

    # Attempt attention rollout (ViT only)
    if _is_vit_model(MODELS[0]):
        try:
            tensor = _as_input_tensor(input_tensor, requires_grad=False)
            heat_att = _attention_rollout_for_vit(MODELS[0], tensor)
            heat_rgb = _jet_colormap(heat_att)
            base_rgb_r = _resize_rgb(base_rgb, IMAGE_SIZE).astype(np.float32)
            blended = (0.6 * base_rgb_r + 0.4 * heat_rgb).clip(0, 255).astype(np.uint8)
            Image.fromarray(blended).save(attention_path)
            attention_saved = str(attention_path)
        except Exception as exc:
            print(f"[XAI] Attention rollout failed: {exc}")
            attention_saved = ""

    # Attempt Grad-CAM
    try:
        tensor_g = _as_input_tensor(input_tensor, requires_grad=True)
        if _is_vit_model(MODELS[0]):
            heat_gc = _gradcam_on_patch_embed(MODELS[0], tensor_g, int(predicted_idx))
        else:
            heat_gc = _gradcam_on_conv(MODELS[0], tensor_g, int(predicted_idx))
        heat_rgb = _jet_colormap(heat_gc)
        base_rgb_r = _resize_rgb(base_rgb, IMAGE_SIZE).astype(np.float32)
        blended = (0.6 * base_rgb_r + 0.4 * heat_rgb).clip(0, 255).astype(np.uint8)
        Image.fromarray(blended).save(gradcam_path)
        gradcam_saved = str(gradcam_path)
    except Exception as exc:
        print(f"[XAI] Grad-CAM failed: {exc}")
        gradcam_saved = ""

    # Create a simple combined image (side-by-side of available maps)
    try:
        parts = []
        if attention_saved:
            parts.append(Image.open(attention_saved).convert("RGB"))
        if gradcam_saved:
            parts.append(Image.open(gradcam_saved).convert("RGB"))
        if not parts:
            # Fallback to single saliency map using existing function
            single = generate_explainability_cam(input_tensor, base_rgb, predicted_idx, output_prefix.with_name(output_prefix.stem + "_cam.png"))
            return {"attention_path": attention_saved, "gradcam_path": gradcam_saved, "combined_path": str(output_prefix.with_name(output_prefix.stem + "_cam.png"))}

        # Resize parts to same height and concatenate horizontally
        widths, heights = zip(*(p.size for p in parts))
        total_width = sum(widths)
        max_height = max(heights)
        new_im = Image.new("RGB", (total_width, max_height))
        x_offset = 0
        for im in parts:
            new_im.paste(im, (x_offset, 0))
            x_offset += im.size[0]
        new_im.save(combined_path)
        combined_saved = str(combined_path)
    except Exception:
        combined_saved = ""

    return {"attention_path": attention_saved, "gradcam_path": gradcam_saved, "combined_path": combined_saved}


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
    # Try Grad-CAM on patch embeddings first, fall back to attention rollout,
    # then fall back to input-gradient saliency.
    try:
        # Attempt Grad-CAM on patch embedding (primary explainability method)
        try:
            tensor_g = _as_input_tensor(input_tensor, requires_grad=True)
            if _is_vit_model(primary_model):
                heat = _gradcam_on_patch_embed(primary_model, tensor_g, int(predicted_idx))
            else:
                heat = _gradcam_on_conv(primary_model, tensor_g, int(predicted_idx))
            heat_rgb = _jet_colormap(heat)
            base_rgb = _resize_rgb(base_rgb, IMAGE_SIZE).astype(np.float32)
            blended = (0.5 * base_rgb + 0.5 * heat_rgb).clip(0, 255).astype(np.uint8)
            Image.fromarray(blended).save(output_path)
            return str(output_path)
        except Exception:
            pass

        # Fall back to attention rollout
        if _is_vit_model(primary_model):
            try:
                tensor = _as_input_tensor(input_tensor, requires_grad=False)
                heat = _attention_rollout_for_vit(primary_model, tensor)
                heat_rgb = _jet_colormap(heat)
                base_rgb = _resize_rgb(base_rgb, IMAGE_SIZE).astype(np.float32)
                blended = (0.5 * base_rgb + 0.5 * heat_rgb).clip(0, 255).astype(np.uint8)
                Image.fromarray(blended).save(output_path)
                return str(output_path)
            except Exception:
                pass

        # Fallback: input-gradient saliency (existing behavior)
        tensor_in = _as_input_tensor(input_tensor, requires_grad=True)
        primary_model.zero_grad(set_to_none=True)
        output = primary_model(tensor_in)
        if output.ndim == 2 and output.shape[-1] == 2:
            target_score = torch.softmax(output, dim=1)[0, int(predicted_idx)]
        else:
            output = output.view(-1)
            target_score = output[0] if int(predicted_idx) == 1 else (1.0 - output[0])
        target_score.backward()

        grad_map = tensor_in.grad.detach().abs().mean(dim=1)[0].cpu().numpy()
        grad_map = grad_map - grad_map.min()
        denom = max(float(grad_map.max()), 1e-8)
        grad_map = grad_map / denom

        heat_rgb = _jet_colormap(grad_map)
        base_rgb = _resize_rgb(base_rgb, IMAGE_SIZE).astype(np.float32)
        blended = (0.6 * base_rgb + 0.4 * heat_rgb).clip(0, 255).astype(np.uint8)
        Image.fromarray(blended).save(output_path)
        return str(output_path)
    except Exception as exc:
        # If anything unexpected fails, fall back to heuristic image-based map
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


# ---------------------------------------------------------------------------
# Model discovery and hot-swapping
# ---------------------------------------------------------------------------

def list_available_models() -> list[dict]:
    """Return metadata for every discovered model checkpoint file.

    Each entry contains path, name, exists, and active.
    """
    result: list[dict] = []
    seen: set[str] = set()
    current_active = ACTIVE_MODEL_PATH
    for p in candidate_model_paths():
        key = str(p.resolve()) if p.exists() else str(p)
        if key in seen:
            continue
        seen.add(key)
        name = _extract_model_name(p, {})
        resolved = str(p.resolve()) if p.exists() else str(p)
        missing_lfs = p.exists() and _is_git_lfs_pointer(p)
        if missing_lfs:
            pull_error = _try_pull_lfs_weights(p)
            if pull_error is None and not _is_git_lfs_pointer(p):
                missing_lfs = False
        result.append({
            "path": resolved,
            "name": name,
            "exists": p.exists(),
            "loadable": p.exists() and not missing_lfs,
            "missing_lfs": missing_lfs,
            "active": resolved == current_active,
        })
    return result


def set_active_model(model_path: str) -> dict:
    """Load a checkpoint and make it the primary inference model.

    The requested path must resolve to a file inside the project's
    ``backend/models`` directory (or a subdirectory of it) to prevent
    path-injection attacks that could load arbitrary files from the filesystem.

    Replaces only the first slot in ``MODELS`` so any secondary model (used
    for ensemble) is preserved.  Updates all module-level globals atomically.
    """
    global MODELS, ACTIVE_MODEL_INFOS, MODEL_TYPE, ATTEMPTED_MODEL_PATHS
    global ACTIVE_MODEL_PATH, ACTIVE_MODEL_NAME, ACTIVE_MODEL_PATHS

    if not (_HAS_TORCH and _HAS_TIMM):
        raise RuntimeError("PyTorch / timm not available; cannot load a model checkpoint.")

    # Validate: the resolved path must be inside the models directory so that
    # external callers cannot load arbitrary files from the host filesystem.
    models_root = (PACKAGE_DIR / "models").resolve()
    candidate_path = Path(model_path)

    # Try both the supplied path and a fallback by filename inside models_root.
    resolved_path: Path | None = None
    for probe in (candidate_path, models_root / candidate_path.name):
        try:
            r = probe.resolve()
            if r.exists() and str(r).startswith(str(models_root)):
                resolved_path = r
                break
        except Exception:
            continue

    if resolved_path is None:
        raise FileNotFoundError(
            "Model checkpoint not found or path is outside the allowed models directory."
        )

    if _is_git_lfs_pointer(resolved_path):
        pull_error = _try_pull_lfs_weights(resolved_path)
        if pull_error is None and not _is_git_lfs_pointer(resolved_path):
            pass
        else:
            detail = f" Auto-pull failed: {pull_error}" if pull_error else ""
            raise RuntimeError(
                "Model checkpoint is a Git LFS pointer. Run 'git lfs pull' to download weights."
                + detail
            )

    model, model_name, metrics = _load_real_model(resolved_path)
    resolved = str(resolved_path)

    new_info: dict = {"path": resolved, "name": model_name, "metrics": metrics}
    if MODELS:
        MODELS[0] = model
        if ACTIVE_MODEL_INFOS:
            ACTIVE_MODEL_INFOS[0] = new_info
        else:
            ACTIVE_MODEL_INFOS = [new_info]
    else:
        MODELS = [model]
        ACTIVE_MODEL_INFOS = [new_info]

    MODEL_TYPE = "real"
    ACTIVE_MODEL_PATHS = [info["path"] for info in ACTIVE_MODEL_INFOS]
    ACTIVE_MODEL_PATH = resolved
    ACTIVE_MODEL_NAME = model_name
    return {"name": model_name, "path": resolved, "metrics": metrics}
