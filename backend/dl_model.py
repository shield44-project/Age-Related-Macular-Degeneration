import os
from pathlib import Path

import cv2
import numpy as np
import timm
import torch
from PIL import Image
from torch import nn

CLASS_NAMES = ["Normal", "AMD"]
IMAGE_SIZE = (224, 224)
PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent
DEFAULT_MODEL_PATH = PACKAGE_DIR / "models" / "ViT_base" / "best_vit_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def candidate_model_paths() -> list[Path]:
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

    model_path = os.getenv("MODEL_PATH")
    if model_path:
        raw = Path(model_path).expanduser()
        resolved = raw if raw.is_absolute() else (PROJECT_ROOT / raw).resolve()

        # Allow MODEL_PATH to be a direct file, directory, or glob expression.
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
    candidates.sort(key=lambda p: ("best" not in p.name.lower(), len(p.parts), str(p)))

    deduped: list[Path] = []
    seen = set()
    for p in candidates:
        key = str(p)
        if key not in seen:
            deduped.append(p)
            seen.add(key)
    return deduped


def _load_state_dict(model_path: Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(model_path, map_location="cpu")
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


def _build_model() -> nn.Module:
    model = ViTBinaryClassifier()
    return model.to(DEVICE).eval()


def _load_real_model(model_path: Path) -> nn.Module:
    model = _build_model()
    state_dict = _load_state_dict(model_path)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError:
        # Fallback for minor key mismatches while still loading usable weights.
        model.load_state_dict(state_dict, strict=False)
    return model


def load_model_with_fallback() -> tuple[nn.Module, str, str]:
    attempted: list[str] = []
    for model_path in candidate_model_paths():
        attempted.append(str(model_path))
        if not model_path.exists():
            continue
        try:
            model = _load_real_model(model_path)
            return model, "real", str(model_path.resolve())
        except Exception as exc:
            print(f"Model load failed at {model_path}: {exc}")

    attempted_paths = " | ".join(attempted) if attempted else "<none>"
    print(f"No usable model checkpoint found. Attempted: {attempted_paths}")
    return _build_model(), "unavailable", attempted_paths


MODEL, MODEL_TYPE, ACTIVE_MODEL_PATH = load_model_with_fallback()


def _ensure_model_ready() -> None:
    global MODEL, MODEL_TYPE, ACTIVE_MODEL_PATH
    if MODEL_TYPE == "real":
        return
    MODEL, MODEL_TYPE, ACTIVE_MODEL_PATH = load_model_with_fallback()


def is_real_model_loaded() -> bool:
    return MODEL_TYPE == "real"


def _as_input_tensor(input_tensor: np.ndarray, requires_grad: bool = False) -> torch.Tensor:
    tensor = torch.as_tensor(input_tensor, dtype=torch.float32, device=DEVICE)
    if requires_grad:
        tensor = tensor.clone().detach().requires_grad_(True)
    return tensor


def predict_probabilities(input_tensor: np.ndarray) -> np.ndarray:
    _ensure_model_ready()
    if not is_real_model_loaded():
        raise RuntimeError(
            f"Real model checkpoint could not be loaded from: {ACTIVE_MODEL_PATH}"
        )

    with torch.no_grad():
        tensor = _as_input_tensor(input_tensor)
        prob_amd = float(MODEL(tensor).item())

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
        raise RuntimeError("Cannot generate CAM because real model is unavailable.")

    tensor = _as_input_tensor(input_tensor, requires_grad=True)

    MODEL.zero_grad(set_to_none=True)
    output = MODEL(tensor).view(-1)

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
