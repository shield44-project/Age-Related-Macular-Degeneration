import io
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import streamlit as st
import torch
import torch.nn as st_nn
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="XAI Heatmap Analyser",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
.stApp { background: #0a0c10; color: #c8d0db; }
section[data-testid="stSidebar"] { background: #0d1017 !important; border-right: 1px solid #1e2530; }
section[data-testid="stSidebar"] * { color: #a0aab8 !important; }
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stFileUploader label {
    color: #5a9fd4 !important; font-family: 'IBM Plex Mono', monospace !important;
    font-size: 11px !important; letter-spacing: 0.08em; text-transform: uppercase;
}
h1, h2, h3 { font-family: 'IBM Plex Mono', monospace !important; letter-spacing: -0.02em; }
h1 { color: #e8edf2 !important; font-size: 1.6rem !important; }
h2 { color: #5a9fd4 !important; font-size: 1.1rem !important; font-weight: 600 !important;
     border-bottom: 1px solid #1e2530; padding-bottom: 6px; }
h3 { color: #7cb8e0 !important; font-size: 0.9rem !important; }
div[data-testid="metric-container"] { background: #111620 !important;
    border: 1px solid #1e2530 !important; border-radius: 4px !important;
    padding: 12px 16px !important; }
div[data-testid="metric-container"] label { color: #5a9fd4 !important;
    font-family: 'IBM Plex Mono', monospace !important; font-size: 10px !important;
    letter-spacing: 0.1em; text-transform: uppercase; }
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #e8edf2 !important; font-family: 'IBM Plex Mono', monospace !important;
    font-size: 1.1rem !important; }
.heatmap-label { font-family: 'IBM Plex Mono', monospace; font-size: 11px; font-weight: 600;
    letter-spacing: 0.12em; text-transform: uppercase; padding: 4px 8px;
    border-radius: 3px; margin-bottom: 8px; display: inline-block; }
.label-gradcam    { background:#0e2235; color:#5a9fd4; border:1px solid #1a3a5c; }
.label-gradcampp  { background:#0e2a1a; color:#4ec98a; border:1px solid #1a4a30; }
.label-scorecam   { background:#2a1a0e; color:#e0974c; border:1px solid #4a2e1a; }
.label-guided     { background:#200e2a; color:#c47de0; border:1px solid #3a1a4a; }
.label-vitgradcam { background:#0e2a2a; color:#4ec9c9; border:1px solid #1a4a4a; }
.label-rollout    { background:#2a1a1a; color:#e05a5a; border:1px solid #4a2020; }
.label-transattr  { background:#2a2a0e; color:#c9c94e; border:1px solid #4a4a1a; }
.label-input      { background:#1a1a0e; color:#d4c84e; border:1px solid #3a3a1a; }
.conf-bar-wrap { background:#1a1f2a; border-radius:3px; height:6px; width:100%; margin-top:4px; }
.conf-bar-fill { height:6px; border-radius:3px; background:linear-gradient(90deg,#2d6fa8,#5ab4e0); }
.badge { display:inline-block; font-family:'IBM Plex Mono',monospace; font-size:10px;
    letter-spacing:0.08em; padding:2px 8px; border-radius:2px; text-transform:uppercase; }
.badge-vit { background:#0e2235; color:#5a9fd4; border:1px solid #1a3a5c; }
.badge-cnn { background:#0e2a1a; color:#4ec98a; border:1px solid #1a4a30; }
hr { border-color: #1e2530 !important; }
.stSelectbox > div > div, .stFileUploader > div {
    background:#111620 !important; border-color:#1e2530 !important; color:#c8d0db !important; }
.stAlert { border-radius:4px !important; }
.stSpinner > div { border-top-color:#5a9fd4 !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# FIX 1: timm import guard at module level — clear error, not buried in loader
# ─────────────────────────────────────────────────────────────────────────────
try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADER
# FIX 2: cache_resource keyed on path+arch+classes so stale patched model
#         never persists across arch changes
# FIX 3: DeiT model string updated for timm>=1.0 API
# FIX 4: monkey-patch reset removed — we never patch forward() anymore;
#         attention is captured via hooks only (see hook section below)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model(pth_path: str, arch: str, num_classes: int, device: str):
    import torchvision.models as tvm

    arch = arch.lower()
    cnn_map = {
        "resnet18": tvm.resnet18, "resnet34": tvm.resnet34,
        "resnet50": tvm.resnet50, "resnet101": tvm.resnet101,
        "densenet121": tvm.densenet121, "densenet169": tvm.densenet169,
        "efficientnet_b0": tvm.efficientnet_b0, "efficientnet_b3": tvm.efficientnet_b3,
        "efficientnet_b4": tvm.efficientnet_b4,
        "mobilenet_v2": tvm.mobilenet_v2, "mobilenet_v3_small": tvm.mobilenet_v3_small,
        "vgg16": tvm.vgg16,
    }
    vit_map  = {"vit_b_16": tvm.vit_b_16, "vit_b_32": tvm.vit_b_32, "vit_l_16": tvm.vit_l_16}

    # FIX 3: timm>=1.0 correct model string for DeiT-S
    deit_map = {"deit_s": "deit_small_patch16_224.fb_in1k"}

    is_vit = arch in vit_map or arch in deit_map

    if arch in cnn_map:
        model = cnn_map[arch](weights=None)
        if hasattr(model, "fc"):
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif hasattr(model, "classifier"):
            clf = model.classifier
            if isinstance(clf, nn.Sequential):
                clf[-1] = nn.Linear(clf[-1].in_features, num_classes)
            elif isinstance(clf, nn.Linear):
                model.classifier = nn.Linear(clf.in_features, num_classes)
    elif arch in vit_map:
        model = vit_map[arch](weights=None)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    elif arch in deit_map:
        # FIX 1: proper guard with user-facing error in UI
        if not TIMM_AVAILABLE:
            st.error("timm not installed. Add `timm>=1.0.0` to requirements.txt and restart.")
            st.stop()
        model = timm.create_model(
            deit_map[arch],
            pretrained=False,
            num_classes=num_classes
        )
    else:
        raise ValueError(f"Unknown architecture: {arch}")

    state = torch.load(pth_path, map_location=device, weights_only=False)
    if isinstance(state, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in state:
                state = state[key]
                break
    model.load_state_dict(state, strict=False)

    # FIX 8: timm DeiT/ViT uses fused_attn=True by default which calls
    # F.scaled_dot_product_attention — a fused CUDA kernel that NEVER
    # materialises the [B,H,N,N] attention matrix in Python memory.
    # attn_drop hook therefore sees nothing. Disable it so the explicit
    # attn = softmax(q @ k.T) path runs and hooks capture real tensors.
    if is_vit:
        for mod in model.modules():
            if hasattr(mod, "fused_attn"):
                mod.fused_attn = False

    model.to(device)
    model.eval()
    return model, is_vit


# ─────────────────────────────────────────────────────────────────────────────
# PRE-PROCESSING
# ─────────────────────────────────────────────────────────────────────────────
_MEAN     = torch.tensor([0.485, 0.456, 0.406])
_STD      = torch.tensor([0.229, 0.224, 0.225])
_BASELINE = (-_MEAN / _STD).view(1, 3, 1, 1)


def preprocess(img_pil: Image.Image, device: str) -> torch.Tensor:
    img = img_pil.convert("RGB").resize((224, 224))
    t   = torch.from_numpy(np.array(img)).float() / 255.0
    t   = (t - _MEAN) / _STD
    return t.permute(2, 0, 1).unsqueeze(0).to(device)


# ─────────────────────────────────────────────────────────────────────────────
# COLOUR HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def jet_colormap(gray01: np.ndarray) -> np.ndarray:
    g  = np.clip(gray01.astype(np.float32), 0, 1)
    fg = 4.0 * g
    r  = np.clip(np.minimum(fg - 1.5, -fg + 4.5), 0, 1)
    gr = np.clip(np.minimum(fg - 0.5, -fg + 3.5), 0, 1)
    b  = np.clip(np.minimum(fg + 0.5, -fg + 2.5), 0, 1)
    return (np.stack([r, gr, b], axis=-1) * 255).astype(np.uint8)


def blend(base_rgb: np.ndarray, heat_rgb: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    return np.clip(
        (1 - alpha) * base_rgb.astype(np.float32) + alpha * heat_rgb.astype(np.float32),
        0, 255,
    ).astype(np.uint8)


def norm01(arr: np.ndarray) -> np.ndarray:
    mn, mx = arr.min(), arr.max()
    return (arr - mn) / (mx - mn + 1e-8)


def resize_map(arr: np.ndarray, hw=(224, 224)) -> np.ndarray:
    pil = Image.fromarray((norm01(arr) * 255).astype(np.uint8))
    pil = pil.resize((hw[1], hw[0]), Image.Resampling.BILINEAR)
    return np.array(pil, dtype=np.float32) / 255.0


# ─────────────────────────────────────────────────────────────────────────────
# CNN BACKPROP ENGINES
# ─────────────────────────────────────────────────────────────────────────────
def find_last_conv(model: nn.Module) -> Optional[nn.Module]:
    last_standard = None
    last_any      = None
    for mod in model.modules():
        if not isinstance(mod, nn.Conv2d):
            continue
        last_any = mod
        kh = mod.kernel_size[0] if isinstance(mod.kernel_size, tuple) else mod.kernel_size
        kw = mod.kernel_size[1] if isinstance(mod.kernel_size, tuple) else mod.kernel_size
        is_spatial   = kh > 1 or kw > 1
        is_depthwise = mod.groups == mod.in_channels and mod.in_channels > 1
        if is_spatial and not is_depthwise:
            last_standard = mod
    return last_standard if last_standard is not None else last_any


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self._activation = None
        self._fwd = target_layer.register_forward_hook(self._fwd_hook)

    def _fwd_hook(self, m, inp, out):
        self._activation = out

    def remove(self):
        self._fwd.remove()

    def __call__(self, x: torch.Tensor, class_idx: int) -> np.ndarray:
        x = x.detach().requires_grad_(True)
        with torch.enable_grad():
            self.model.zero_grad()
            out = self.model(x)
            if self._activation is not None:
                self._activation.retain_grad()
            out[0, class_idx].backward()

        act  = self._activation
        grad = act.grad if (act is not None and act.grad is not None) else None
        if act is None or grad is None:
            return np.zeros((224, 224), dtype=np.float32)

        w   = grad.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * act.detach()).sum(dim=1, keepdim=True))
        return resize_map(cam[0, 0].cpu().numpy())


class GradCAMPlusPlus:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model       = model
        self._activation = None
        self._fwd = target_layer.register_forward_hook(self._fwd_hook)

    def _fwd_hook(self, m, inp, out):
        self._activation = out

    def remove(self):
        self._fwd.remove()

    def __call__(self, x: torch.Tensor, class_idx: int) -> np.ndarray:
        x = x.detach().requires_grad_(True)
        with torch.enable_grad():
            self.model.zero_grad()
            out = self.model(x)
            if self._activation is not None:
                self._activation.retain_grad()
            out[0, class_idx].backward()

        act  = self._activation
        grad = act.grad if (act is not None and act.grad is not None) else None
        if act is None or grad is None:
            return np.zeros((224, 224), dtype=np.float32)

        act_d = act.detach()
        g     = grad.detach()
        g2    = g ** 2
        g3    = g ** 3
        sum_A = (act_d * g3).sum(dim=(2, 3), keepdim=True)
        denom = 2.0 * g2 + sum_A + 1e-7
        alpha = torch.where(denom.abs() > 1e-7, g2 / denom, torch.zeros_like(g2))
        w     = (alpha * F.relu(g)).sum(dim=(2, 3), keepdim=True)
        cam   = F.relu((w * act_d).sum(dim=1, keepdim=True))
        return resize_map(cam[0, 0].cpu().numpy())


def score_cam(model: nn.Module, x: torch.Tensor, target_layer: nn.Module,
              class_idx: int, max_masks: int = 32) -> np.ndarray:
    acts = {}
    h = target_layer.register_forward_hook(lambda m, i, o: acts.update({"a": o.detach()}))
    with torch.no_grad():
        model(x)
    h.remove()

    act = acts["a"][0]
    C   = min(act.shape[0], max_masks)
    act = act[:C]
    baseline = _BASELINE.to(x.device)
    scores   = torch.zeros(C, device=x.device)

    pbar = st.progress(0, text=f"Score-CAM: mask 0 / {C}")
    with torch.no_grad():
        for i in range(C):
            mask_np    = resize_map(act[i].cpu().numpy())
            mask       = torch.from_numpy(mask_np).float().to(x.device).unsqueeze(0).unsqueeze(0)
            masked     = baseline + mask * (x - baseline)
            logits     = model(masked)
            scores[i]  = F.softmax(logits, dim=1)[0, class_idx]
            pbar.progress(int((i + 1) / C * 100), text=f"Score-CAM: mask {i+1} / {C}")
    pbar.empty()

    cam = (scores.view(-1, 1, 1) * act).sum(dim=0).cpu().numpy()
    return resize_map(F.relu(torch.from_numpy(cam)).numpy())


class VanillaSaliency:
    def __init__(self, model: nn.Module):
        self.model = model

    def remove(self):
        pass

    def __call__(self, x: torch.Tensor, class_idx: int) -> np.ndarray:
        x = x.detach().requires_grad_(True)
        with torch.enable_grad():
            self.model.zero_grad()
            out = self.model(x)
            out[0, class_idx].backward()
        grad = x.grad[0].cpu().numpy()
        grad = np.abs(grad).mean(axis=0)
        return norm01(grad)


def guided_gradcam(saliency: np.ndarray, cam: np.ndarray) -> np.ndarray:
    return norm01(saliency * cam)


# ─────────────────────────────────────────────────────────────────────────────
# FIX 5 + FIX 6: ATTENTION HOOK COLLECTOR
# No monkey-patching of forward(). Pure register_forward_hook only.
# Covers timm DeiT/ViT (proj_drop, attn_drop) AND torchvision ViT (MHA).
# Tuple output from blocks handled safely.
# ─────────────────────────────────────────────────────────────────────────────
def _collect_attn_hooks(model: nn.Module):
    """
    Returns list of (kind, module) tuples for attention modules.
    Covers:
      - timm DeiT/ViT: hooks on attn_drop or proj_drop (input is [B,H,N,N] matrix)
      - torchvision ViT: nn.MultiheadAttention modules
    """
    targets = []

    # timm>=1.0: attention weight tensor flows through attn_drop
    # name patterns: blocks.X.attn.attn_drop  OR  blocks.X.attn.proj_drop
    for name, mod in model.named_modules():
        n = name.lower()
        if ("attn.attn_drop" in n or "attn.proj_drop" in n):
            targets.append(("timm_drop", mod))

    if not targets:
        # torchvision ViT path
        for name, mod in model.named_modules():
            if isinstance(mod, nn.MultiheadAttention):
                targets.append(("torchvision_mha", mod))

    return targets


def _register_attn_collection(targets, store: list, retain_gradients: bool = False):
    """
    Registers hooks WITHOUT monkey-patching forward().
    For MHA we use need_weights via a pre-hook that sets kwargs.
    Returns (hooks_list, {}) — empty dict, no patched forwards.
    """
    hooks = []

    for kind, mod in targets:
        if kind == "timm_drop":
            # input[0] to dropout layer is the [B, H, N, N] attention weight matrix
            def make_hook(s, rg):
                def hook(m, inp, out):
                    if inp and isinstance(inp[0], torch.Tensor) and inp[0].ndim == 4:
                        attn_mat = inp[0].detach() if not rg else inp[0]
                        if rg and attn_mat.requires_grad:
                            attn_mat.retain_grad()
                        s.append(attn_mat)
                return hook
            hooks.append(mod.register_forward_hook(make_hook(store, retain_gradients)))

        elif kind == "torchvision_mha":
            # FIX 6: use pre_hook to inject need_weights=True, average_attn_weights=False
            # without touching forward() directly
            def make_pre_hook():
                def pre_hook(m, args, kwargs):
                    kwargs["need_weights"]         = True
                    kwargs["average_attn_weights"] = False
                    return args, kwargs
                return pre_hook

            def make_post_hook(s, rg):
                def post_hook(m, inp, out):
                    if isinstance(out, tuple) and len(out) == 2:
                        w = out[1]
                        if isinstance(w, torch.Tensor) and w.ndim in (3, 4):
                            if w.ndim == 3:
                                w = w.unsqueeze(1)
                            w = w.detach() if not rg else w
                            if rg and w.requires_grad:
                                w.retain_grad()
                            s.append(w)
                return post_hook

            hooks.append(mod.register_forward_pre_hook(make_pre_hook(), with_kwargs=True))
            hooks.append(mod.register_forward_hook(make_post_hook(store, retain_gradients)))

    return hooks, {}   # empty dict — no patched forwards to restore


def _restore_forwards(_):
    pass   # no-op; nothing patched


# ─────────────────────────────────────────────────────────────────────────────
# FIX 7: vit_gradcam — handle tuple output from block forward safely
# ─────────────────────────────────────────────────────────────────────────────
def vit_gradcam(model: nn.Module, x: torch.Tensor, class_idx: int) -> Optional[np.ndarray]:
    target = None

    if hasattr(model, "blocks") and len(model.blocks) > 0:
        target = model.blocks[-1]
    elif hasattr(model, "encoder") and hasattr(model.encoder, "layers"):
        target = model.encoder.layers[-1]

    if target is None:
        return None

    saved: dict = {}

    def fwd_hook(m, inp, out):
        # FIX 7: handle tuple output (some timm versions return (x, attn))
        t = out[0] if isinstance(out, (tuple, list)) else out
        if isinstance(t, torch.Tensor) and t.ndim == 3:
            saved["act"] = t

    def bwd_hook(m, gin, gout):
        for g in gout:
            if isinstance(g, torch.Tensor) and g.ndim == 3:
                saved["grad"] = g.detach()
                return

    h1 = target.register_forward_hook(fwd_hook)
    h2 = target.register_full_backward_hook(bwd_hook)

    x = x.detach().requires_grad_(True)
    with torch.enable_grad():
        model.zero_grad()
        out = model(x)
        if "act" in saved and saved["act"].requires_grad:
            saved["act"].retain_grad()
        out[0, class_idx].backward()

    h1.remove()
    h2.remove()

    if "act" not in saved or "grad" not in saved:
        return np.zeros((224, 224), dtype=np.float32)

    act  = saved["act"][0, 1:].detach()
    grad = saved["grad"][0, 1:].detach() if saved["grad"] is not None else torch.ones_like(act)

    weights = grad.mean(dim=-1)
    cam     = F.relu((weights.unsqueeze(-1) * act).sum(dim=-1))

    n    = cam.shape[0]
    side = int(n ** 0.5)
    if side * side != n:
        side = int(np.sqrt(n))
        cam  = cam[:side * side]

    cam = cam.reshape(side, side).cpu().numpy()
    return resize_map(cam)


def attention_rollout(model: nn.Module, x: torch.Tensor, discard: float = 0.9) -> np.ndarray:
    attn_maps: list = []
    targets = _collect_attn_hooks(model)
    if not targets:
        st.warning("Attention Rollout: no attention modules found.")
        return np.zeros((224, 224), dtype=np.float32)

    hooks, _ = _register_attn_collection(targets, attn_maps, retain_gradients=False)
    with torch.no_grad():
        model(x)

    for h in hooks:
        h.remove()

    # FIX: reduce [B,H,N,N] → [N,N] immediately, discard raw tensor to free RAM
    valid_reduced = []
    for a in attn_maps:
        if a.ndim == 4 and a.shape[2] == a.shape[3]:
            valid_reduced.append(a.detach().cpu().mean(dim=1).squeeze(0))
    del attn_maps

    if not valid_reduced:
        st.warning("Attention Rollout: hooks fired but no valid [B,H,N,N] tensors captured.")
        return np.zeros((224, 224), dtype=np.float32)

    seq     = valid_reduced[0].shape[0]
    rollout = torch.eye(seq)

    for avg in valid_reduced:
        if discard > 0:
            flat   = avg.flatten()
            thresh = torch.quantile(flat, discard)
            avg    = torch.where(avg < thresh, torch.zeros_like(avg), avg)

        I       = torch.eye(seq)
        a_fused = 0.5 * avg + 0.5 * I
        a_fused = a_fused / a_fused.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        rollout = torch.mm(a_fused, rollout)
        del avg, a_fused

    cls_attn = rollout[0, 1:].numpy()
    side = int(cls_attn.shape[0] ** 0.5)
    import gc; gc.collect()
    return resize_map(cls_attn[:side * side].reshape(side, side))


def transformer_attribution(model: nn.Module, x: torch.Tensor, class_idx: int) -> np.ndarray:
    attn_tensors: list = []
    targets = _collect_attn_hooks(model)
    if not targets:
        st.warning("Transformer Attribution: no attention modules found.")
        return np.zeros((224, 224), dtype=np.float32)

    hooks, _ = _register_attn_collection(targets, attn_tensors, retain_gradients=True)

    x = x.detach().requires_grad_(True)
    with torch.enable_grad():
        model.zero_grad()
        logits  = model(x)
        one_hot = torch.zeros_like(logits)
        one_hot[0, class_idx] = 1.0
        logits.backward(gradient=one_hot, retain_graph=True)

    for h in hooks:
        h.remove()

    valid_maps = [at for at in attn_tensors if at.ndim == 4]
    del attn_tensors
    if not valid_maps:
        st.warning("Transformer Attribution: no valid attention maps with gradients captured.")
        return np.zeros((224, 224), dtype=np.float32)

    seq = valid_maps[0].shape[-1]
    R   = torch.eye(seq)

    for at in valid_maps:
        at_det = at.detach().cpu()
        grad   = at.grad.detach().cpu() if at.grad is not None else torch.zeros_like(at_det)
        cam    = (at_det * F.relu(grad)).mean(dim=1).squeeze(0)
        del at_det, grad
        I   = torch.eye(seq)
        cam = I + cam
        cam = cam / cam.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        R   = torch.mm(cam, R)
        del cam

    del valid_maps
    import gc; gc.collect()

    cls_rel = R[0, 1:].numpy()
    side    = int(cls_rel.shape[0] ** 0.5)
    return resize_map(cls_rel[:side * side].reshape(side, side))


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE ROUTER
# ─────────────────────────────────────────────────────────────────────────────
def run_xai(model, x, class_idx, is_vit, arch, device, discard_ratio, score_cam_masks, status_box=None):
    results: dict = {}

    def _status(msg):
        if status_box is not None:
            status_box.info(f"⚙️ {msg}")

    if not is_vit:
        if hasattr(model, "layer4"):
            target_layer = model.layer4[-1]
        else:
            target_layer = find_last_conv(model)

        if target_layer is None:
            st.warning("No standard features layer caught.")
            return results

        try:
            _status("Running Grad-CAM++...")
            obj = GradCAMPlusPlus(model, target_layer)
            results["Grad-CAM++"] = obj(x.clone(), class_idx)
            obj.remove()
        except Exception as e:
            st.warning(f"Grad-CAM++ error: {e}")

        try:
            _status("Running Score-CAM (slowest — progress bar below)...")
            results["Score-CAM"] = score_cam(model, x.clone(), target_layer, class_idx, score_cam_masks)
        except Exception as e:
            st.warning(f"Score-CAM error: {e}")

    else:
        try:
            _status("Running Attention Rollout...")
            results["Attention Rollout"] = attention_rollout(model, x.clone(), discard=discard_ratio)
        except Exception as e:
            st.warning(f"Attention Rollout error: {e}")
            results["Attention Rollout"] = np.zeros((224, 224), dtype=np.float32)

    _status("Done.")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# RENDER
# ─────────────────────────────────────────────────────────────────────────────
LABEL_CLASS = {
    "Grad-CAM":                ("Grad-CAM",               "label-gradcam"),
    "Grad-CAM++":              ("Grad-CAM++",              "label-gradcampp"),
    "Score-CAM":               ("Score-CAM",               "label-scorecam"),
    "Guided Grad-CAM":         ("Guided Grad-CAM",         "label-guided"),
    "ViT Grad-CAM":            ("ViT Grad-CAM",            "label-vitgradcam"),
    "Attention Rollout":       ("Attention Rollout",       "label-rollout"),
    "Transformer Attribution": ("Transformer Attribution", "label-transattr"),
}


def render_heatmap_card(col, name: str, heat_map, base_rgb: np.ndarray):
    label_text, label_cls = LABEL_CLASS.get(name, (name, "label-input"))
    with col:
        st.markdown(f'<span class="heatmap-label {label_cls}">{label_text}</span>',
                    unsafe_allow_html=True)
        if heat_map is None or np.all(heat_map == 0):
            st.markdown(
                "<div style='background:#0d1017;border:1px dashed #1e2530;height:224px;"
                "display:flex;align-items:center;justify-content:center;color:#ff5a5a;"
                "font-family:IBM Plex Mono,monospace;font-size:11px;border-radius:4px;'>"
                "Matrix Empty / Missing Weights</div>",
                unsafe_allow_html=True)
        else:
            st.image(blend(base_rgb, jet_colormap(heat_map), alpha=0.45),
                     use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    st.markdown("<h1>🔬 XAI Heatmap Analyser</h1>", unsafe_allow_html=True)
    st.markdown("---")

    st.sidebar.markdown("<h2>⚙️ Configurations</h2>", unsafe_allow_html=True)

    pth_file = st.sidebar.text_input(
        "Weights path (.pth / .pt)",
        value="model.pth",
        help="Absolute or relative path to your PyTorch checkpoint."
    )

    arch_options = [
        "resnet18", "resnet34", "resnet50", "resnet101",
        "densenet121", "densenet169",
        "efficientnet_b0", "efficientnet_b3", "efficientnet_b4",
        "mobilenet_v2", "mobilenet_v3_small", "vgg16",
        "vit_b_16", "vit_b_32", "vit_l_16", "deit_s"
    ]
    arch        = st.sidebar.selectbox("Model Architecture", options=arch_options, index=0)
    num_classes = st.sidebar.number_input("Number of Classes", min_value=1, value=2, step=1)

    labels_raw   = st.sidebar.text_input("Class Labels (comma separated)", value="Normal, AMD")
    class_names  = [label.strip() for label in labels_raw.split(",")]
    if len(class_names) < num_classes:
        class_names += [f"Class_{i}" for i in range(len(class_names), num_classes)]

    st.sidebar.markdown("<h3>🔍 XAI Hyperparameters</h3>", unsafe_allow_html=True)
    discard_ratio   = st.sidebar.slider("Rollout Head Discard Ratio", 0.0, 0.95, 0.10, 0.05)
    score_cam_masks = st.sidebar.slider("Score-CAM Max Masks", 8, 128, 32, 8)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    st.sidebar.markdown(f"**Runtime Target:** `{device.upper()}`")

    # FIX 1: surface timm availability in UI for deit_s
    if arch == "deit_s" and not TIMM_AVAILABLE:
        st.error("DeiT-S requires timm>=1.0.0. Add it to requirements.txt and restart.")
        return

    uploaded_file = st.file_uploader(
        "Upload Image For Explainable Diagnosis...", type=["jpg", "jpeg", "png"]
    )

    if not Path(pth_file).exists():
        st.info(f"💡 Awaiting valid weights file at: `{pth_file}`")
        return

    if uploaded_file is None:
        st.info("💡 Drop an image above to compute spatial attributions.")
        return

    try:
        # FIX 2: cache key includes arch+num_classes so stale cache never reused
        model, is_vit = load_model(pth_file, arch, int(num_classes), device)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return

    arch_badge = (
        '<span class="badge badge-vit">ViT Network Architecture</span>'
        if is_vit else
        '<span class="badge badge-cnn">CNN Backbone Network</span>'
    )
    st.markdown(f"**Architecture:** `{arch}` &nbsp;&nbsp; {arch_badge}", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    img_pil      = Image.open(uploaded_file)
    base_rgb     = np.array(img_pil.convert("RGB").resize((224, 224)))
    input_tensor = preprocess(img_pil, device)

    # free any lingering tensors before inference
    import gc
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    with torch.no_grad():
        logits       = model(input_tensor)
        probs        = F.softmax(logits, dim=1)[0].cpu().numpy()
        pred_idx     = int(np.argmax(probs))
        logit_margin = logits[0].max().item()
    del logits

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Predicted Class", f"#{pred_idx} : {class_names[pred_idx]}")
    with m2:
        st.metric("Confidence", f"{probs[pred_idx] * 100:.2f}%")
    with m3:
        st.metric("Logit Margin (Δ)", f"{logit_margin:.3f}")

    st.markdown("**Probability Distribution**")
    for idx, (p_val, label_name) in enumerate(zip(probs, class_names)):
        st.markdown(
            f"<div style='font-size:11px;font-family:IBM Plex Mono;margin-bottom:2px;'>"
            f"{label_name}: {p_val*100:.1f}%</div>"
            f"<div class='conf-bar-wrap'><div class='conf-bar-fill' style='width:{p_val*100}%;'></div></div>",
            unsafe_allow_html=True
        )
    st.markdown("<br><hr>", unsafe_allow_html=True)

    target_explain_class = st.selectbox(
        "Target class for feature maps:",
        options=list(range(num_classes)),
        index=pred_idx,
        format_func=lambda i: f"Class {i}: {class_names[i]} {'(Predicted)' if i == pred_idx else ''}"
    )

    st.markdown("### \u23f3 Computing Attribution Maps")
    _status = st.empty()
    xai_maps = run_xai(
        model=model,
        x=input_tensor,
        class_idx=target_explain_class,
        is_vit=is_vit,
        arch=arch,
        device=device,
        discard_ratio=discard_ratio,
        score_cam_masks=score_cam_masks,
        status_box=_status,
    )
    _status.empty()

    st.markdown("## 📊 Interpretability Attribution Heatmaps")

    if not is_vit:
        row1 = st.columns(3)
        with row1[0]:
            st.markdown('<span class="heatmap-label label-input">Input Specimen</span>',
                        unsafe_allow_html=True)
            st.image(base_rgb, use_container_width=True)
        render_heatmap_card(row1[1], "Grad-CAM++", xai_maps.get("Grad-CAM++"), base_rgb)
        render_heatmap_card(row1[2], "Score-CAM",  xai_maps.get("Score-CAM"),  base_rgb)
    else:
        row1 = st.columns(2)
        with row1[0]:
            st.markdown('<span class="heatmap-label label-input">Input Specimen</span>',
                        unsafe_allow_html=True)
            st.image(base_rgb, use_container_width=True)
        render_heatmap_card(row1[1], "Attention Rollout", xai_maps.get("Attention Rollout"), base_rgb)


if __name__ == "__main__":
    main()