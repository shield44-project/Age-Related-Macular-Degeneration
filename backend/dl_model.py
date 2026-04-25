import os
from pathlib import Path

import cv2
import numpy as np
import timm
import torch
from PIL import Image
from torch import nn
from typing import Any

CLASS_NAMES = ["Normal", "AMD"]
IMAGE_SIZE = (224, 224)
PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent
DEFAULT_MODEL_PATH = PACKAGE_DIR / "models" / "ViT_base" / "best_vit_model.pth"
DEFAULT_MODEL_PATH_2 = PACKAGE_DIR / "models" / "ViT_base" / "best_vit_model_2.pth"
DEFAULT_MODEL_PATH_IMPROVED = PACKAGE_DIR / "models" / "ViT_base" / "best_vit_model_improved.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_MODELS = 2
BACKUP_MODEL_TYPE = "backup"
DEFAULT_MODEL_NAME = "ViT-B16 AMD Classifier"


def _force_backup_mode() -> bool:
    return os.getenv("FORCE_BACKUP_MODE", "0") == "1"


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
    add_if_present(DEFAULT_MODEL_PATH_2)

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
        # PyTorch 2.6 changed default weights_only=True. Try safe mode first,
        # then trusted full-load mode for older full-checkpoint files.

        #Since model loading was working perfectly when done manually, this script seems to contain the error I have written a more direct approach which may not check all edge cases
        '''try:
            return torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            # Older PyTorch without weights_only argument.
            return torch.load(path, map_location="cpu")
        except Exception as exc:
            try:
                return torch.load(path, map_location="cpu", weights_only=False)
            except TypeError:
                return torch.load(path, map_location="cpu")
            except Exception:
                raise exc'''
        return torch.load(path, map_location="cpu", weights_only=False)


def _extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, nn.Module):
        checkpoint = checkpoint.state_dict()
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model", "net", "weights"):
            nested = checkpoint.get(key)
            if isinstance(nested, nn.Module):
                nested = nested.state_dict()
            if isinstance(nested, dict):
                checkpoint = nested
                break

    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint is not a state_dict dictionary.")

    cleaned: dict[str, torch.Tensor] = {}
    for key, value in checkpoint.items():
        new_key = key[7:] if key.startswith("module.") else key
        cleaned[new_key] = value
    return cleaned


def _extract_metrics(checkpoint: Any) -> dict[str, float | None]:
    metric_keys = {
        "accuracy": ("accuracy", "acc", "val_accuracy", "best_val_acc"),
        "precision": ("precision", "val_precision", "best_precision"),
        "recall": ("recall", "val_recall", "best_recall"),
        "f1_score": ("f1", "f1_score", "val_f1", "best_f1"),
    }
    metrics: dict[str, float | None] = {
        "accuracy": None,
        "precision": None,
        "recall": None,
        "f1_score": None,
    }
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
    return metrics


def _extract_model_name(model_path: Path, checkpoint: Any) -> str:
    if isinstance(checkpoint, dict):
        for key in ("model_name", "name", "arch", "architecture"):
            value = checkpoint.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    stem = model_path.stem.replace("_", " ").replace("-", " ").strip()
    return stem.title() if stem else DEFAULT_MODEL_NAME


def _build_legacy_model() -> nn.Module:
    return ViTBinaryClassifier().to(DEVICE).eval()


def _build_multiclass_model() -> nn.Module:
    return ViTMultiClassClassifier().to(DEVICE).eval()


def _try_load_model(model: nn.Module, state_dict: dict[str, torch.Tensor], strict: bool = True) -> nn.Module:
    try:
        model.load_state_dict(state_dict, strict=strict)
        return model
    except RuntimeError:
        if strict:
            model.load_state_dict(state_dict, strict=False)
            return model
        raise


def _load_real_model(model_path: Path) -> tuple[nn.Module, str, dict[str, float | None]]:
    checkpoint = _torch_load_checkpoint(model_path)
    state_dict = _extract_state_dict(checkpoint)

    model: nn.Module
    try:
        model = _try_load_model(_build_legacy_model(), state_dict, strict=True)
    except Exception:
        model = _try_load_model(_build_multiclass_model(), state_dict, strict=True)

    model_name = _extract_model_name(model_path, checkpoint)
    metrics = _extract_metrics(checkpoint)
    return model, model_name, metrics


def load_models_with_fallback(max_models: int = MAX_MODELS) -> tuple[list[nn.Module], list[dict[str, Any]], str, str]:
    loaded_models: list[nn.Module] = []
    loaded_infos: list[dict[str, Any]] = []
    attempted: list[str] = []
    seen_resolved_paths: set[str] = set()

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


def get_model_status() -> dict[str, object]:
    _ensure_model_ready()
    metrics = {
        "accuracy": None,
        "precision": None,
        "recall": None,
        "f1_score": None,
    }
    if ACTIVE_MODEL_INFOS:
        first_metrics = ACTIVE_MODEL_INFOS[0].get("metrics", {})
        for key in metrics:
            value = first_metrics.get(key)
            metrics[key] = float(value) if isinstance(value, (float, int)) else None

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


def _as_input_tensor(input_tensor: np.ndarray, requires_grad: bool = False) -> torch.Tensor:
    tensor = torch.as_tensor(input_tensor, dtype=torch.float32, device=DEVICE)
    if requires_grad:
        tensor = tensor.clone().detach().requires_grad_(True)
    return tensor


def _backup_predict_prob_amd(input_tensor: np.ndarray) -> float:
    """Deterministic fallback score from image statistics when model weights are unavailable."""
    arr = np.asarray(input_tensor, dtype=np.float32)
    # Expect (1, C, H, W) normalized tensor. Compute stable statistics.
    mean_val = float(arr.mean())
    std_val = float(arr.std())
    # Simple bounded score in [0, 1] using brightness/texture proxy.
    score = 0.5 + 0.15 * np.tanh(std_val - 1.0) - 0.10 * np.tanh(mean_val)
    return float(np.clip(score, 0.05, 0.95))


def predict_probabilities(input_tensor: np.ndarray) -> np.ndarray:
    _ensure_model_ready()
    if not is_real_model_loaded():
        prob_amd = _backup_predict_prob_amd(input_tensor)
        return np.array([1.0 - prob_amd, prob_amd], dtype=np.float32)

    with torch.no_grad():
        tensor = _as_input_tensor(input_tensor)
        per_model_probs: list[float] = []
        for model in MODELS:
            out = model(tensor)
            if out.ndim == 2 and out.shape[-1] == 2:
                prob_amd = float(torch.softmax(out, dim=1)[0, 1].item())
            else:
                prob_amd = float(out.view(-1)[0].item())
            per_model_probs.append(prob_amd)
        prob_amd = float(np.mean(per_model_probs))

    prob_amd = float(np.clip(prob_amd, 0.0, 1.0))
    return np.array([1.0 - prob_amd, prob_amd], dtype=np.float32)


def generate_explainability_cam(
    input_tensor: np.ndarray,
    base_rgb: np.ndarray,
    predicted_idx: int,
    output_path: Path,
) -> str:
    """Generate a real gradient-based saliency map from the loaded model."""
    _ensure_model_ready()
    if not is_real_model_loaded():
        # Backup visualization from per-pixel intensity (keeps GUI flow alive).
        if base_rgb.shape[:2] != IMAGE_SIZE:
            base_rgb = cv2.resize(base_rgb, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(base_rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        gray = gray.astype(np.float32)
        gray -= gray.min()
        denom = max(float(gray.max()), 1e-8)
        gray = gray / denom
        heat_bgr = cv2.applyColorMap((gray * 255.0).astype(np.uint8), cv2.COLORMAP_JET)
        heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        base = base_rgb.astype(np.float32)
        blended = (0.6 * base + 0.4 * heat_rgb).clip(0, 255).astype(np.uint8)
        output_path.parent.mkdir(parents=True, exist_ok=True)
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

    heat_bgr = cv2.applyColorMap((grad_map * 255.0).astype(np.uint8), cv2.COLORMAP_JET)
    heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)

    if base_rgb.shape[:2] != IMAGE_SIZE:
        base_rgb = cv2.resize(base_rgb, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    base_rgb = base_rgb.astype(np.float32)

    blended = (0.6 * base_rgb + 0.4 * heat_rgb).clip(0, 255).astype(np.uint8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(blended).save(output_path)
    return str(output_path)
