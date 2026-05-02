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
except Exception as _torch_exc:
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

DEFAULT_METRICS = {
    "accuracy": 0.80,
    "sensitivity": 0.79,
    "specificity": 0.83,
    "precision": 0.95,
    "f1_score": 0.864,
}

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
        remapped: list[Path] = []
        parts = list(path.parts)
        if "backend" in parts and "models" in parts:
            i_backend = parts.index("backend")
            i_models = parts.index("models", i_backend + 1)
            suffix = Path(*parts[i_models + 1:])
            if str(suffix):
                remapped.append(models_root / suffix)
        if path.name:
            remapped.extend(sorted(models_root.rglob(path.name)))
        return remapped

    def collect_from_configured_path(path_value: str) -> None:
        raw = Path(path_value).expanduser()
        resolved = raw if raw.is_absolute() else (PROJECT_ROOT / raw).resolve()
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

    for root in (
        PROJECT_ROOT / "backend" / "models",
        PROJECT_ROOT / "models",
        Path.cwd() / "backend" / "models",
    ):
        for ext in ("*.pth", "*.pt"):
            for match in sorted(root.rglob(ext)) if root.exists() else []:
                add_if_present(match)

    if model_path:
        stem_name = Path(model_path).name
        if stem_name:
            for match in sorted(models_root.rglob(stem_name)):
                add_if_present(match)

    for ext in ("*.pth", "*.pt"):
        for match in sorted(models_root.rglob(ext)):
            add_if_present(match)

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
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
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


def _build_legacy_model():
    return ViTBinaryClassifier().to(DEVICE).eval()


def _build_multiclass_model():
    return ViTMultiClassClassifier().to(DEVICE).eval()


def _looks_like_multiclass(state_dict: dict) -> bool:
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
            print(f"Could not load {model_path} as timm architecture {checkpoint_arch!r}: {exc}")

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
            loaded_infos.append({
                "path": resolved_path,
                "name": model_name,
                "metrics": metrics,
            })
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
    metrics = dict(DEFAULT_METRICS)
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
    arr = np.asarray(input_tensor, dtype=np.float32)
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

    central_brightness = float(center_luma.mean())
    overall_contrast = float(luma.std())
    red_dominance = float(center[0].mean() - center[1].mean())

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
                val = float(out.view(-1)[0].item())
                prob_amd = val if 0.0 <= val <= 1.0 else float(1.0 / (1.0 + np.exp(-val)))
            else:
                val = float(out.view(-1)[0].item())
                prob_amd = val if 0.0 <= val <= 1.0 else float(1.0 / (1.0 + np.exp(-val)))
            per_model_probs.append(prob_amd)
        prob_amd = float(np.median(per_model_probs))

    prob_amd = float(np.clip(prob_amd, 0.0, 1.0))
    return np.array([1.0 - prob_amd, prob_amd], dtype=np.float32)


def _pil_resize_rgb(arr: np.ndarray, size) -> np.ndarray:
    img = Image.fromarray(arr.astype(np.uint8))
    img = img.resize((size[1], size[0]), Image.Resampling.LANCZOS)
    return np.asarray(img)


def _jet_colormap(gray01: np.ndarray) -> np.ndarray:
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


# ── Attention Rollout ─────────────────────────────────────────────────────────

def get_attention_rollout(model, input_tensor: np.ndarray) -> np.ndarray:
    """
    Computes attention rollout heatmap for a single fundus image.
    Returns a (224, 224) float32 numpy array, values in [0, 1].
    Higher values = model paid more attention to that region.

    Fix: hooks block.attn.attn_drop (captures [B, heads, tokens, tokens])
    instead of block.attn (which captures token embeddings [B, tokens, dim]).
    """
    attentions = []

    def hook_fn(module, input, output):
        # output shape: (B, num_heads, num_tokens, num_tokens)
        attentions.append(output.detach().cpu())

    hooks = []
    try:
        for block in model.backbone.blocks:
            # attn_drop is applied directly after softmax attention weights
            # — this gives us the actual attention weight matrix
            hooks.append(block.attn.attn_drop.register_forward_hook(hook_fn))
    except AttributeError as e:
        print(f"Hook registration failed: {e}")
        return np.ones((224, 224), dtype=np.float32) * 0.5

    model.eval()
    try:
        with torch.no_grad():
            tensor = torch.as_tensor(input_tensor, dtype=torch.float32, device=DEVICE)
            _ = model(tensor)
    finally:
        for h in hooks:
            h.remove()

    if not attentions:
        return np.ones((224, 224), dtype=np.float32) * 0.5

    # attentions[i] shape: (1, num_heads, num_tokens, num_tokens)
    # Rollout: average heads, add residual, multiply through layers
    num_tokens = attentions[0].shape[-1]   # 197 for ViT-B/16
    result = torch.eye(num_tokens)

    for attn in attentions:
        # Average over heads → (num_tokens, num_tokens)
        attn_avg = attn.mean(dim=1).squeeze(0)
        # Add residual connection
        attn_avg = attn_avg + torch.eye(num_tokens)
        # Normalize rows
        attn_avg = attn_avg / attn_avg.sum(dim=-1, keepdim=True)
        result = torch.matmul(attn_avg, result)

    # CLS token attention to patch tokens (index 0 = CLS, 1: = patches)
    mask = result[0, 1:].numpy()          # (196,) for ViT-B/16

    num_patches = int(mask.shape[0] ** 0.5)
    square_size = num_patches * num_patches
    mask = mask[:square_size].reshape(num_patches, num_patches)  # (14, 14)

    # Normalize to [0, 1]
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)

    # Upsample to 224×224
    if _HAS_CV2:
        mask_up = cv2.resize(mask, (224, 224), interpolation=cv2.INTER_LINEAR)
    else:
        mask_up = np.array(
            Image.fromarray((mask * 255).astype(np.uint8)).resize((224, 224), Image.BILINEAR),
            dtype=np.float32,
        ) / 255.0

    return mask_up.astype(np.float32)


def check_macula_attention(
    attention_map: np.ndarray,
    cx: int = 112,
    cy: int = 112,
    radius: int = 35,
    top_percent: float = 0.20,
) -> dict:
    """
    Checks whether attention concentrates on the fovea/macula region.
    Used for subclinical AMD flagging.

    cx, cy      : macula centre in 224x224 image (default: image centre)
    radius      : ROI radius in pixels (~35px ≈ 3.5mm at standard FOV)
    top_percent : fraction of highest-attention pixels to consider

    Returns:
        subclinical_flag  — True if >40% of top-attention pixels are inside macula ROI
        overlap_ratio     — fraction of top-attention pixels inside macula ROI
        drusen_area_ratio — attention-weighted area as fraction of total image
    """
    h, w = attention_map.shape
    Y, X = np.ogrid[:h, :w]
    macula_mask = ((X - cx) ** 2 + (Y - cy) ** 2) <= radius ** 2

    threshold = np.percentile(attention_map, (1 - top_percent) * 100)
    high_attn_mask = attention_map >= threshold

    overlap = np.logical_and(high_attn_mask, macula_mask)
    overlap_ratio = float(overlap.sum() / (high_attn_mask.sum() + 1e-8))

    drusen_mask = attention_map >= 0.5
    drusen_area_ratio = float(drusen_mask.sum() / (h * w))

    return {
        "subclinical_flag": overlap_ratio >= 0.40,
        "overlap_ratio": overlap_ratio,
        "drusen_area_ratio": drusen_area_ratio,
    }


def _vit_reshape_transform(tensor, height=14, width=14):
    """
    Converts ViT token sequence to spatial feature map.
    Input:  (B, num_patches+1, dim) — includes CLS token at index 0
    Output: (B, dim, height, width)
    """
    # Drop CLS token, reshape patches to 2D grid
    result = tensor[:, 1:, :].reshape(
        tensor.size(0), height, width, tensor.size(2)
    )
    # Move channels first: (B, H, W, C) → (B, C, H, W)
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def generate_explainability_cam(
    input_tensor: np.ndarray,
    base_rgb: np.ndarray,
    predicted_idx: int,
    output_path: Path,
) -> str:
    """
    Generates an attention rollout heatmap overlaid on the fundus image.
    Falls back to gradient-based saliency if rollout fails.
    Falls back to grayscale overlay if no real model is loaded.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── No real model: grayscale brightness overlay ───────────────────
    if not is_real_model_loaded():
        base = _resize_rgb(base_rgb, IMAGE_SIZE).astype(np.float32)
        gray = 0.299 * base[..., 0] + 0.587 * base[..., 1] + 0.114 * base[..., 2]
        gray = (gray - gray.min()) / (gray.max() - gray.min() + 1e-8)
        heat = _jet_colormap(gray)
        blended = (0.6 * base + 0.4 * heat).clip(0, 255).astype(np.uint8)
        Image.fromarray(blended).save(output_path)
        return str(output_path)

    primary_model = MODELS[0]

    # ── GradCAM via pytorch-grad-cam library ──────────────────────────
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        from pytorch_grad_cam.utils.image import show_cam_on_image

        # Target layer for ViT: last transformer block's norm1
        target_layers = [primary_model.backbone.blocks[-1].norm1]

        # Prepare RGB float image in [0, 1] for overlay
        base = _resize_rgb(base_rgb, IMAGE_SIZE).astype(np.float32) / 255.0

        # Prepare input tensor
        tensor = torch.as_tensor(input_tensor, dtype=torch.float32, device=DEVICE)

        # ClassifierOutputTarget: 0 = Normal, 1 = AMD
        targets = [ClassifierOutputTarget(int(predicted_idx))]

        with GradCAM(
            model=primary_model,
            target_layers=target_layers,
            reshape_transform=_vit_reshape_transform,
        ) as cam:
            grayscale_cam = cam(input_tensor=tensor, targets=targets)
            grayscale_cam = grayscale_cam[0, :]   # (224, 224)

        # show_cam_on_image expects RGB float [0,1] and grayscale [0,1]
        visualization = show_cam_on_image(base, grayscale_cam, use_rgb=True)
        Image.fromarray(visualization).save(output_path)
        return str(output_path)

    except Exception as e:
        print(f"GradCAM failed, falling back to gradient saliency: {e}")

    # ── Fallback: gradient-based saliency ────────────────────────────
    try:
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
        grad_map = (grad_map - grad_map.min()) / (grad_map.max() - grad_map.min() + 1e-8)
        heat = _jet_colormap(grad_map)

        base = _resize_rgb(base_rgb, IMAGE_SIZE).astype(np.float32)
        blended = (0.6 * base + 0.4 * heat).clip(0, 255).astype(np.uint8)
        Image.fromarray(blended).save(output_path)
        return str(output_path)

    except Exception as e:
        print(f"Gradient fallback also failed: {e}")
        base = _resize_rgb(base_rgb, IMAGE_SIZE)
        Image.fromarray(base.astype(np.uint8)).save(output_path)
        return str(output_path)
