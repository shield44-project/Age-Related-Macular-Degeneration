# XAI Heatmap Analyser — Dashboard

A Streamlit tool to compare **Grad-CAM**, **Grad-CAM++**, **Score-CAM**,
**Guided Grad-CAM** and **Attention Rollout** side-by-side on any `.pth`
checkpoint.


How to run:
Through terminal, get inside the heatmap_dashboard folder.

run these scripts: (copy paste)

pip install -r requirements_dashboard.txt
streamlit run gradcam_dashboard.py

---

## Quick Start

### 1 — Install dependencies

```bash
# create a virtual env (recommended)
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# install packages
pip install -r requirements_dashboard.txt
```

> **GPU users** — install the CUDA-enabled torch wheel first:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> pip install -r requirements_dashboard.txt
> ```

> **DeiT-S users** — `timm` is now included in `requirements_dashboard.txt` and installs automatically. If you skipped the requirements file, install it manually:
> ```bash
> pip install timm>=0.9.0
> ```

---

### 2 — Run the dashboard

```bash
streamlit run gradcam_dashboard.py
```

Streamlit will open your browser automatically at `http://localhost:8501`.

---

## How to use

| Step | What to do |
|------|-----------|
| **1** | Enter the **full path** to your `.pth` file in the sidebar (e.g. `/home/user/models/resnet50_retina.pth`) |
| **2** | Select the **architecture** that matches your checkpoint from the dropdown |
| **3** | Set **Num Classes** to the number of output classes your model was trained on |
| **4** | Upload a **test image** (JPG / PNG / BMP / TIFF) |
| **5** | Choose **Device** (`auto` picks CUDA if available, else CPU) |
| **6** | Click **▶ RUN ANALYSIS** |

---

## Sidebar options

| Option | Default | Notes |
|--------|---------|-------|
| Path to `.pth` | — | Absolute or relative path |
| Architecture | `resnet50` | See supported list below |
| Num Classes | `1000` | Must match your checkpoint's output head |
| Device | `auto` | `auto` / `cpu` / `cuda` |
| Attention Discard Ratio | `0.9` | ViT only — keeps top (1−ratio) attention weights |
| Score-CAM Max Masks | `32` | Higher = better quality, slower. 8 is fast; 128 is high-quality |

---

## Supported architectures

**CNN (all four Grad-CAM variants)**
- `resnet18`, `resnet34`, `resnet50`, `resnet101`
- `densenet121`, `densenet169`
- `efficientnet_b0`, `efficientnet_b3`, `efficientnet_b4`
- `mobilenet_v2`, `mobilenet_v3_small`
- `vgg16`

**ViT (Attention Rollout only; Grad-CAM slots show "N/A")**
- `vit_b_16`, `vit_b_32`, `vit_l_16`
- `deit_s` *(requires `timm`)*

---

## Checkpoint format

The loader handles the most common `.pth` formats automatically:

```python
torch.save(model.state_dict(), "model.pth")              # ✅ plain state dict
torch.save({"model_state_dict": model.state_dict()}, …)  # ✅ training checkpoint
torch.save({"state_dict": model.state_dict()}, …)        # ✅ lightning / other
torch.save({"model": model.state_dict()}, …)             # ✅ custom key
```

`strict=False` is used so minor layer name mismatches (e.g. a renamed head)
don't block loading.

---

## What each heatmap shows

| Method | Description |
|--------|-------------|
| **Grad-CAM** | Gradients of the predicted class w.r.t. the last conv layer, averaged into a coarse spatial map |
| **Grad-CAM++** | Improves on Grad-CAM by weighting positive gradient contributions — better at localising multiple objects |
| **Score-CAM** | Gradient-free: masks each activation channel onto the input, scores it via the model, then linearly combines channels |
| **Guided Grad-CAM** | Element-wise product of Guided Backpropagation (fine pixel gradients) and Grad-CAM (coarse location) |
| **Attention Rollout** | Recursively multiplies attention matrices across all transformer layers to trace which input tokens the [CLS] token attended to |

---

## Troubleshooting

**DeiT-S not loading / "Unknown architecture"**
Make sure you selected `deit_s` exactly from the dropdown and that `timm` is installed (`pip install timm`). DeiT uses timm internally — it is not available via torchvision.

**"No attention maps captured"**
Your ViT sub-module names may differ. Open `gradcam_dashboard.py` and in
`attention_rollout()` adjust the string check:
```python
if "Attention" in mod.__class__.__name__ ...
```
to match your model's attention class name.

**Score-CAM is very slow**
Reduce *Score-CAM Max Masks* in the sidebar to `8`–`16` for a quick preview.

**CUDA out of memory**
Switch Device to `cpu`, or reduce Score-CAM Max Masks.

**Wrong predictions (ImageNet labels)**
If your model was trained on a custom dataset the class names won't match.
The *Top-1 Prediction* will show `Class {idx}` — that's expected.
The heatmaps are still computed correctly for that index.
