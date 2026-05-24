"""
╔══════════════════════════════════════════════════════════════╗
║        XAI Heatmap Analyser — Streamlit Dashboard           ║
║  Grad-CAM · Grad-CAM++ · Score-CAM · Guided Grad-CAM        ║
║  + Attention Rollout (ViT only)                              ║
╚══════════════════════════════════════════════════════════════╝
"""

import io
import os
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="XAI Heatmap Analyser",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS — dark, clinical, precise
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

/* ── BACKGROUND ── */
.stApp {
    background: #0a0c10;
    color: #c8d0db;
}

/* ── SIDEBAR ── */
section[data-testid="stSidebar"] {
    background: #0d1017 !important;
    border-right: 1px solid #1e2530;
}
section[data-testid="stSidebar"] * {
    color: #a0aab8 !important;
}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stFileUploader label {
    color: #5a9fd4 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 11px !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

/* ── HEADERS ── */
h1, h2, h3 {
    font-family: 'IBM Plex Mono', monospace !important;
    letter-spacing: -0.02em;
}
h1 { color: #e8edf2 !important; font-size: 1.6rem !important; }
h2 { color: #5a9fd4 !important; font-size: 1.1rem !important; font-weight: 600 !important; border-bottom: 1px solid #1e2530; padding-bottom: 6px; }
h3 { color: #7cb8e0 !important; font-size: 0.9rem !important; }

/* ── METRIC CARDS ── */
div[data-testid="metric-container"] {
    background: #111620 !important;
    border: 1px solid #1e2530 !important;
    border-radius: 4px !important;
    padding: 12px 16px !important;
}
div[data-testid="metric-container"] label {
    color: #5a9fd4 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 10px !important;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #e8edf2 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 1.1rem !important;
}

/* ── HEATMAP CARDS ── */
.heatmap-card {
    background: #111620;
    border: 1px solid #1e2530;
    border-radius: 6px;
    padding: 10px;
    margin-bottom: 8px;
    transition: border-color 0.2s;
}
.heatmap-card:hover { border-color: #2d4a6e; }

.heatmap-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 4px 8px;
    border-radius: 3px;
    margin-bottom: 8px;
    display: inline-block;
}

/* label colour per method */
.label-gradcam      { background: #0e2235; color: #5a9fd4; border: 1px solid #1a3a5c; }
.label-gradcampp    { background: #0e2a1a; color: #4ec98a; border: 1px solid #1a4a30; }
.label-scorecam     { background: #2a1a0e; color: #e0974c; border: 1px solid #4a2e1a; }
.label-guided       { background: #200e2a; color: #c47de0; border: 1px solid #3a1a4a; }
.label-rollout      { background: #2a1a1a; color: #e05a5a; border: 1px solid #4a2020; }
.label-input        { background: #1a1a0e; color: #d4c84e; border: 1px solid #3a3a1a; }

/* ── CONFIDENCE BAR ── */
.conf-bar-wrap {
    background: #1a1f2a;
    border-radius: 3px;
    height: 6px;
    width: 100%;
    margin-top: 4px;
}
.conf-bar-fill {
    height: 6px;
    border-radius: 3px;
    background: linear-gradient(90deg, #2d6fa8, #5ab4e0);
}

/* ── STATUS BADGE ── */
.badge {
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    letter-spacing: 0.08em;
    padding: 2px 8px;
    border-radius: 2px;
    text-transform: uppercase;
}
.badge-vit  { background:#0e2235; color:#5a9fd4; border:1px solid #1a3a5c; }
.badge-cnn  { background:#0e2a1a; color:#4ec98a; border:1px solid #1a4a30; }
.badge-cpu  { background:#1a1a1a; color:#888; border:1px solid #333; }
.badge-gpu  { background:#2a1a0e; color:#e0974c; border:1px solid #4a2e1a; }

/* ── DIVIDER ── */
hr { border-color: #1e2530 !important; }

/* ── SELECTBOX / FILE INPUT ── */
.stSelectbox > div > div,
.stFileUploader > div {
    background: #111620 !important;
    border-color: #1e2530 !important;
    color: #c8d0db !important;
}

/* ── INFO / WARNING ── */
.stAlert { border-radius: 4px !important; }

/* ── SPINNER ── */
.stSpinner > div { border-top-color: #5a9fd4 !important; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# IMAGENET CLASSES (top-1000, abbreviated for bundle size)
# We load these lazily; if unavailable we fall back to "Class {idx}"
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_imagenet_labels() -> dict:
    try:
        import urllib.request, json
        url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
        with urllib.request.urlopen(url, timeout=3) as r:
            return {i: v for i, v in enumerate(json.loads(r.read()))}
    except Exception:
        return {}


# ──────────────────────────────────────────────────────────────────────────────
# MODEL LOADER
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model(pth_path: str, arch: str, num_classes: int, device: str):
    """Load a .pth checkpoint into the requested torchvision architecture."""
    import torchvision.models as tvm

    arch = arch.lower()

    # ── CNN backbones ──
    cnn_map = {
        "resnet18":   tvm.resnet18,
        "resnet34":   tvm.resnet34,
        "resnet50":   tvm.resnet50,
        "resnet101":  tvm.resnet101,
        "densenet121":tvm.densenet121,
        "densenet169":tvm.densenet169,
        "efficientnet_b0": tvm.efficientnet_b0,
        "efficientnet_b3": tvm.efficientnet_b3,
        "efficientnet_b4": tvm.efficientnet_b4,
        "mobilenet_v2":    tvm.mobilenet_v2,
        "mobilenet_v3_small": tvm.mobilenet_v3_small,
        "vgg16":      tvm.vgg16,
    }
    # ── ViT backbones ──
    vit_map = {
        "vit_b_16":   tvm.vit_b_16,
        "vit_b_32":   tvm.vit_b_32,
        "vit_l_16":   tvm.vit_l_16,
    }

    # ── DeiT backbones (via timm) ──
    deit_map = {
        "deit_s": "deit_small_patch16_224",
    }

    is_vit = arch in vit_map or arch in deit_map

    if arch in cnn_map:
        model = cnn_map[arch](weights=None)
        # replace final layer
        if hasattr(model, "fc"):
            in_f = model.fc.in_features
            model.fc = nn.Linear(in_f, num_classes)
        elif hasattr(model, "classifier"):
            clf = model.classifier
            if isinstance(clf, nn.Sequential):
                in_f = clf[-1].in_features
                clf[-1] = nn.Linear(in_f, num_classes)
            elif isinstance(clf, nn.Linear):
                in_f = clf.in_features
                model.classifier = nn.Linear(in_f, num_classes)
    elif arch in vit_map:
        model = vit_map[arch](weights=None)
        in_f = model.heads.head.in_features
        model.heads.head = nn.Linear(in_f, num_classes)
        is_vit = True
    elif arch in deit_map:
        try:
            import timm
        except ImportError:
            raise ImportError(
                "DeiT-S requires the 'timm' package. "
                "Install it with:  pip install timm"
            )
        model = timm.create_model(
            deit_map[arch],
            pretrained=False,
            num_classes=num_classes,
        )
        is_vit = True
    else:
        raise ValueError(f"Unknown architecture: {arch}")

    state = torch.load(pth_path, map_location=device)
    # handle common checkpoint wrappers
    if isinstance(state, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in state:
                state = state[key]
                break
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model, is_vit


# ──────────────────────────────────────────────────────────────────────────────
# PRE-PROCESSING
# ──────────────────────────────────────────────────────────────────────────────
_MEAN = torch.tensor([0.485, 0.456, 0.406])
_STD  = torch.tensor([0.229, 0.224, 0.225])

def preprocess(img_pil: Image.Image, device: str) -> torch.Tensor:
    img = img_pil.convert("RGB").resize((224, 224))
    t   = torch.from_numpy(np.array(img)).float() / 255.0  # (H,W,3)
    t   = (t - _MEAN) / _STD
    return t.permute(2, 0, 1).unsqueeze(0).to(device)      # (1,3,224,224)


# ──────────────────────────────────────────────────────────────────────────────
# COLOUR HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def jet_colormap(gray01: np.ndarray) -> np.ndarray:
    g     = np.clip(gray01.astype(np.float32), 0, 1)
    fg    = 4.0 * g
    r     = np.clip(np.minimum(fg - 1.5, -fg + 4.5), 0, 1)
    gr    = np.clip(np.minimum(fg - 0.5, -fg + 3.5), 0, 1)
    b     = np.clip(np.minimum(fg + 0.5, -fg + 2.5), 0, 1)
    return (np.stack([r, gr, b], axis=-1) * 255).astype(np.uint8)

def blend(base_rgb: np.ndarray, heat_rgb: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    base  = base_rgb.astype(np.float32)
    heat  = heat_rgb.astype(np.float32)
    out   = np.clip((1 - alpha) * base + alpha * heat, 0, 255).astype(np.uint8)
    return out

def norm01(arr: np.ndarray) -> np.ndarray:
    mn, mx = arr.min(), arr.max()
    return (arr - mn) / (mx - mn + 1e-8)

def resize_map(arr: np.ndarray, hw=(224, 224)) -> np.ndarray:
    pil = Image.fromarray((norm01(arr) * 255).astype(np.uint8))
    pil = pil.resize((hw[1], hw[0]), Image.Resampling.BILINEAR)
    return np.array(pil, dtype=np.float32) / 255.0


# ──────────────────────────────────────────────────────────────────────────────
# GRAD-CAM
# ──────────────────────────────────────────────────────────────────────────────
class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model        = model
        self.gradients    = None
        self.activations  = None
        self._fwd = target_layer.register_forward_hook(self._save_act)
        self._bwd = target_layer.register_full_backward_hook(self._save_grad)

    def _save_act(self, _, __, out):  self.activations = out.detach()
    def _save_grad(self, _, __, go):  self.gradients   = go[0].detach()

    def remove(self):
        self._fwd.remove(); self._bwd.remove()

    def __call__(self, x: torch.Tensor, class_idx: int) -> np.ndarray:
        self.model.zero_grad()
        logits = self.model(x)
        logits[0, class_idx].backward()

        w   = self.gradients.mean(dim=(2, 3), keepdim=True)      # (1,C,1,1)
        cam = F.relu((w * self.activations).sum(dim=1, keepdim=True))  # (1,1,h,w)
        return resize_map(cam[0, 0].cpu().numpy())


# ──────────────────────────────────────────────────────────────────────────────
# GRAD-CAM++
# ──────────────────────────────────────────────────────────────────────────────
class GradCAMPlusPlus:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model       = model
        self.gradients   = None
        self.activations = None
        self._fwd = target_layer.register_forward_hook(self._save_act)
        self._bwd = target_layer.register_full_backward_hook(self._save_grad)

    def _save_act(self, _, __, out): self.activations = out.detach()
    def _save_grad(self, _, __, go): self.gradients   = go[0].detach()

    def remove(self):
        self._fwd.remove(); self._bwd.remove()

    def __call__(self, x: torch.Tensor, class_idx: int) -> np.ndarray:
        self.model.zero_grad()
        logits = self.model(x)
        logits[0, class_idx].backward()

        grads = self.gradients          # (1,C,H,W)
        acts  = self.activations        # (1,C,H,W)

        grads2 = grads ** 2
        grads3 = grads ** 3
        denom  = 2 * grads2 + (acts * grads3).sum(dim=(2, 3), keepdim=True) + 1e-8
        alpha  = grads2 / denom
        w      = (alpha * F.relu(grads)).sum(dim=(2, 3), keepdim=True)
        cam    = F.relu((w * acts).sum(dim=1, keepdim=True))
        return resize_map(cam[0, 0].cpu().numpy())


# ──────────────────────────────────────────────────────────────────────────────
# SCORE-CAM  (mask-based, no backward pass)
# ──────────────────────────────────────────────────────────────────────────────
def score_cam(
    model: nn.Module,
    x: torch.Tensor,
    target_layer: nn.Module,
    class_idx: int,
    max_masks: int = 32,
) -> np.ndarray:
    activations = {}

    def hook(_, __, out):
        activations["act"] = out.detach()

    h = target_layer.register_forward_hook(hook)
    with torch.no_grad():
        baseline_logits = model(x)
    h.remove()

    act = activations["act"][0]          # (C, h, w)
    C   = min(act.shape[0], max_masks)   # limit for speed
    act = act[:C]

    scores = torch.zeros(C, device=x.device)
    for i in range(C):
        m = resize_map(act[i].cpu().numpy())                     # (224,224) in [0,1]
        mask = torch.from_numpy(m).float().to(x.device).unsqueeze(0).unsqueeze(0)
        masked_x = x * mask
        with torch.no_grad():
            logits = model(masked_x)
        scores[i] = F.softmax(logits, dim=1)[0, class_idx]

    w   = scores.view(-1, 1, 1)                                  # (C,1,1)
    cam = F.relu((w * act).sum(dim=0)).cpu().numpy()             # (h,w)
    return resize_map(cam)


# ──────────────────────────────────────────────────────────────────────────────
# GUIDED BACKPROP
# ──────────────────────────────────────────────────────────────────────────────
class GuidedBackprop:
    def __init__(self, model: nn.Module):
        self.model = model
        self.hooks: list = []
        self._register()

    def _register(self):
        for mod in self.model.modules():
            if isinstance(mod, nn.ReLU):
                h = mod.register_backward_hook(self._guide)
                self.hooks.append(h)

    @staticmethod
    def _guide(_, grad_in, grad_out):
        return (F.relu(grad_in[0]),)

    def remove(self):
        for h in self.hooks: h.remove()

    def __call__(self, x: torch.Tensor, class_idx: int) -> np.ndarray:
        x = x.requires_grad_(True)
        self.model.zero_grad()
        out = self.model(x)
        out[0, class_idx].backward()
        gb  = x.grad[0].cpu().numpy()           # (3,224,224)
        gb  = np.maximum(gb, 0).mean(axis=0)    # (224,224)  guided = only positive
        return norm01(gb)


def guided_gradcam(
    gb_map: np.ndarray, cam_map: np.ndarray
) -> np.ndarray:
    return norm01(gb_map * cam_map)


# ──────────────────────────────────────────────────────────────────────────────
# ATTENTION ROLLOUT  (ViT only)
# ──────────────────────────────────────────────────────────────────────────────
def attention_rollout(model: nn.Module, x: torch.Tensor, discard: float = 0.9) -> np.ndarray:
    attn_maps: list = []
    hooks: list     = []

    def hook(_, __, out):
        if isinstance(out, torch.Tensor) and out.ndim == 4:
            attn_maps.append(out.detach().cpu())

    for _, mod in model.named_modules():
        if "Attention" in mod.__class__.__name__ or "MultiheadAttention" in mod.__class__.__name__:
            hooks.append(mod.register_forward_hook(hook))

    with torch.no_grad():
        model(x)
    for h in hooks: h.remove()

    if not attn_maps:
        return np.zeros((224, 224), dtype=np.float32)

    B = attn_maps[0].shape[0]
    seq = attn_maps[0].shape[2]
    rollout = torch.eye(seq).unsqueeze(0)

    for am in attn_maps:
        avg  = am.mean(dim=1)        # (B, seq, seq)
        if discard > 0:
            thresh = torch.quantile(avg.reshape(B, -1), discard, dim=1).reshape(B, 1, 1)
            avg = torch.where(avg < thresh, torch.zeros_like(avg), avg)
        avg  = avg / (avg.sum(dim=2, keepdim=True) + 1e-8)
        rollout = torch.bmm(avg, rollout)

    cls_attn = rollout[0, 0, 1:].numpy()
    side     = int(np.sqrt(len(cls_attn)))
    if side * side != len(cls_attn):
        side = int(np.sqrt(len(cls_attn) + 0.5))
    spatial  = cls_attn[:side*side].reshape(side, side)
    return resize_map(spatial)


# ──────────────────────────────────────────────────────────────────────────────
# LAST CONV LAYER FINDER
# ──────────────────────────────────────────────────────────────────────────────
def find_last_conv(model: nn.Module) -> Optional[nn.Module]:
    last = None
    for mod in model.modules():
        if isinstance(mod, nn.Conv2d):
            last = mod
    return last


# ──────────────────────────────────────────────────────────────────────────────
# RUN ALL XAI METHODS
# ──────────────────────────────────────────────────────────────────────────────
def run_xai(model, x, class_idx, is_vit, device):
    results = {}

    if not is_vit:
        target_layer = find_last_conv(model)
        if target_layer is None:
            st.warning("Could not find a Conv2d layer — Grad-CAM methods unavailable.")
            return results

        # ── Grad-CAM ──
        try:
            gc = GradCAM(model, target_layer)
            results["Grad-CAM"] = gc(x.clone(), class_idx)
            gc.remove()
        except Exception as e:
            st.warning(f"Grad-CAM failed: {e}")

        # ── Grad-CAM++ ──
        try:
            gc2 = GradCAMPlusPlus(model, target_layer)
            results["Grad-CAM++"] = gc2(x.clone(), class_idx)
            gc2.remove()
        except Exception as e:
            st.warning(f"Grad-CAM++ failed: {e}")

        # ── Score-CAM ──
        try:
            results["Score-CAM"] = score_cam(model, x.clone(), target_layer, class_idx)
        except Exception as e:
            st.warning(f"Score-CAM failed: {e}")

        # ── Guided Grad-CAM ──
        try:
            gbp = GuidedBackprop(model)
            gb  = gbp(x.clone(), class_idx)
            gbp.remove()
            gc3 = GradCAM(model, target_layer)
            cam3 = gc3(x.clone(), class_idx)
            gc3.remove()
            results["Guided Grad-CAM"] = guided_gradcam(gb, cam3)
        except Exception as e:
            st.warning(f"Guided Grad-CAM failed: {e}")

    else:
        # ViT: Attention Rollout only for the 5th slot; others are blank
        results["Grad-CAM"]         = None   # not applicable
        results["Grad-CAM++"]       = None
        results["Score-CAM"]        = None
        results["Guided Grad-CAM"]  = None
        try:
            results["Attention Rollout"] = attention_rollout(model, x.clone())
        except Exception as e:
            st.warning(f"Attention Rollout failed: {e}")
            results["Attention Rollout"] = np.zeros((224, 224), dtype=np.float32)

    return results


# ──────────────────────────────────────────────────────────────────────────────
# RENDER A SINGLE HEATMAP CARD
# ──────────────────────────────────────────────────────────────────────────────
LABEL_CLASS = {
    "Grad-CAM":           ("Grad-CAM",          "label-gradcam"),
    "Grad-CAM++":         ("Grad-CAM++",         "label-gradcampp"),
    "Score-CAM":          ("Score-CAM",          "label-scorecam"),
    "Guided Grad-CAM":    ("Guided Grad-CAM",    "label-guided"),
    "Attention Rollout":  ("Attention Rollout",  "label-rollout"),
}

def render_heatmap_card(col, name: str, heat_map, base_rgb: np.ndarray):
    label_text, label_cls = LABEL_CLASS.get(name, (name, "label-input"))
    with col:
        st.markdown(
            f'<span class="heatmap-label {label_cls}">{label_text}</span>',
            unsafe_allow_html=True,
        )
        if heat_map is None:
            st.markdown(
                "<div style='background:#0d1017;border:1px dashed #1e2530;"
                "height:224px;display:flex;align-items:center;justify-content:center;"
                "color:#333;font-family:IBM Plex Mono,monospace;font-size:11px;"
                "border-radius:4px;'>N/A — CNN only</div>",
                unsafe_allow_html=True,
            )
        else:
            colored  = jet_colormap(heat_map)
            blended  = blend(base_rgb, colored, alpha=0.45)
            st.image(blended, use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔬 XAI Analyser")
    st.markdown("---")

    st.markdown("### Model")
    pth_path = st.text_input(
        "PATH TO .pth FILE",
        placeholder="/path/to/model.pth",
    )
    arch = st.selectbox(
        "ARCHITECTURE",
        [
            "resnet50", "resnet18", "resnet34", "resnet101",
            "densenet121", "densenet169",
            "efficientnet_b0", "efficientnet_b3", "efficientnet_b4",
            "mobilenet_v2", "mobilenet_v3_small",
            "vgg16",
            "vit_b_16", "vit_b_32", "vit_l_16",
            "deit_s",
        ],
    )
    num_classes = st.number_input("NUM CLASSES", min_value=2, max_value=21843, value=1000, step=1)

    st.markdown("---")
    st.markdown("### Image")
    uploaded = st.file_uploader("UPLOAD IMAGE", type=["jpg", "jpeg", "png", "bmp", "tiff"])

    st.markdown("---")
    st.markdown("### Settings")
    device_choice = st.selectbox("DEVICE", ["auto", "cpu", "cuda"])
    discard_ratio = st.slider("ATTENTION DISCARD RATIO", 0.0, 0.99, 0.9, 0.01,
                              help="ViT only — fraction of lowest attention weights zeroed")
    score_cam_masks = st.slider("SCORE-CAM MAX MASKS", 8, 128, 32, 8,
                                help="More masks = higher quality but slower")

    run_btn = st.button("▶  RUN ANALYSIS", use_container_width=True)

    st.markdown("---")
    st.caption("XAI Heatmap Analyser · xAI project")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN AREA
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("# XAI Heatmap Analyser")
st.markdown(
    "Visualise and compare **Grad-CAM · Grad-CAM++ · Score-CAM · "
    "Guided Grad-CAM** (CNN) and **Attention Rollout** (ViT) on any "
    "`.pth` checkpoint."
)
st.markdown("---")

if not run_btn:
    st.info("👈  Configure your model and upload an image in the sidebar, then click **RUN ANALYSIS**.")
    st.stop()

# ── validation ──
if not pth_path or not Path(pth_path).is_file():
    st.error("❌  `.pth` file not found. Check the path in the sidebar.")
    st.stop()
if uploaded is None:
    st.error("❌  Please upload an image.")
    st.stop()

# ── device ──
if device_choice == "auto":
    device = "cuda" if torch.cuda.is_available() else "cpu"
else:
    device = device_choice

# ── load image ──
img_pil  = Image.open(io.BytesIO(uploaded.read())).convert("RGB")
img_224  = img_pil.resize((224, 224))
base_rgb = np.array(img_224)

# ── load model ──
with st.spinner("Loading model …"):
    try:
        model, is_vit = load_model(pth_path, arch, int(num_classes), device)
    except Exception as e:
        st.error(f"❌  Model loading failed: {e}")
        st.stop()

# ── inference ──
with st.spinner("Running inference …"):
    x          = preprocess(img_pil, device)
    with torch.no_grad():
        logits     = model(x)
        probs      = F.softmax(logits, dim=1)[0]
        top5_vals, top5_idx = probs.topk(5)
        pred_idx   = int(top5_idx[0])
        pred_conf  = float(top5_vals[0]) * 100

labels     = load_imagenet_labels()
pred_label = labels.get(pred_idx, f"Class {pred_idx}")

# ── xai ──
with st.spinner("Computing heatmaps … (Score-CAM may take ~10s)"):
    xai_results = run_xai(model, x, pred_idx, is_vit, device)

# ──────────────────────────────────────────────────────────────────────────────
# RESULTS LAYOUT
# ──────────────────────────────────────────────────────────────────────────────

arch_type = "ViT" if is_vit else "CNN"
badge_arch  = f'<span class="badge badge-vit">{arch_type}</span>'
badge_dev   = f'<span class="badge badge-{"gpu" if device=="cuda" else "cpu"}">{device.upper()}</span>'

# ── Row 1: model info + top-5 ──
st.markdown("## Model & Prediction")
r1c1, r1c2, r1c3, r1c4 = st.columns([2, 2, 2, 3])
r1c1.metric("Architecture", arch.upper())
r1c2.metric("Type", arch_type)
r1c3.metric("Device", device.upper())
r1c4.metric("Top-1 Prediction", pred_label, f"{pred_conf:.1f}% confidence")

st.markdown("---")

# ── Row 2: confidence bar + top-5 table ──
st.markdown("## Classification Output")
cb1, cb2 = st.columns([3, 2])

with cb1:
    st.markdown(f"**Top-1:** `{pred_label}`")
    bar_w = int(pred_conf)
    st.markdown(
        f'<div class="conf-bar-wrap"><div class="conf-bar-fill" style="width:{bar_w}%"></div></div>',
        unsafe_allow_html=True,
    )
    st.markdown(f"<small style='color:#5a9fd4;font-family:IBM Plex Mono,monospace'>"
                f"{pred_conf:.2f}% confidence</small>", unsafe_allow_html=True)

with cb2:
    st.markdown("**Top-5 Predictions**")
    rows = []
    for v, i in zip(top5_vals, top5_idx):
        lbl = labels.get(int(i), f"Class {int(i)}")
        rows.append({"Class": lbl, "Confidence": f"{float(v)*100:.2f}%"})
    import pandas as pd
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

st.markdown("---")

# ── Row 3: input + heatmaps ──
st.markdown("## Heatmap Comparison")

if is_vit:
    st.markdown(
        f'{badge_arch} ViT detected — Grad-CAM variants are **not applicable**. '
        f'Showing **Attention Rollout** only.',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        f'{badge_arch} CNN detected — showing all four Grad-CAM variants.',
        unsafe_allow_html=True,
    )

st.markdown("")

# 5 columns: input | GC | GC++ | ScoreCAM | GuidedGC / Rollout
col_in, col1, col2, col3, col4 = st.columns(5)

# Input image card
with col_in:
    st.markdown('<span class="heatmap-label label-input">Input Image</span>', unsafe_allow_html=True)
    st.image(base_rgb, use_container_width=True)

if is_vit:
    render_heatmap_card(col1, "Grad-CAM",        None, base_rgb)
    render_heatmap_card(col2, "Grad-CAM++",      None, base_rgb)
    render_heatmap_card(col3, "Score-CAM",       None, base_rgb)
    render_heatmap_card(col4, "Attention Rollout",
                        xai_results.get("Attention Rollout"), base_rgb)
else:
    render_heatmap_card(col1, "Grad-CAM",        xai_results.get("Grad-CAM"),        base_rgb)
    render_heatmap_card(col2, "Grad-CAM++",      xai_results.get("Grad-CAM++"),      base_rgb)
    render_heatmap_card(col3, "Score-CAM",       xai_results.get("Score-CAM"),       base_rgb)
    render_heatmap_card(col4, "Guided Grad-CAM", xai_results.get("Guided Grad-CAM"), base_rgb)

st.markdown("---")

# ── Row 4: method descriptions ──
with st.expander("📖  What does each method show?"):
    st.markdown("""
| Method | What it highlights | Best for |
|---|---|---|
| **Grad-CAM** | Class-discriminative regions using gradients of the last conv layer | Quick overview, coarse localisation |
| **Grad-CAM++** | Weighted positive gradients — handles multiple object instances better | Multi-object scenes |
| **Score-CAM** | Mask-based — no gradients needed, more stable | Noisy gradient scenarios |
| **Guided Grad-CAM** | Fine-grained pixel-level detail fused with Grad-CAM | Texture / edge attribution |
| **Attention Rollout** | Propagated self-attention across all ViT layers | ViT models only |
""")
