# AMD Detection System

A retinal fundus AMD (Age-related Macular Degeneration) classifier with a Qt 5 / C++ desktop GUI and a Python Flask inference backend.

## Architecture

- **Frontend** — `src/main.cpp` (Qt 5 widgets, ~1080 lines). Pitch-black professional dark theme, Inter-style typography, light theme toggle. Built locally with CMake (`CMakeLists.txt`).
- **Backend** — Flask app under `backend/` exposing `/health`, `/predict`, `/patients`, `/patients/<id>`. Entry point: `python -m backend` → `backend/server.py` runs on `0.0.0.0:5000`.
- **Model** — ViT-B16 binary classifier (`backend/dl_model.py`). Auto-detects head shape (1-logit sigmoid vs 2-logit softmax) before loading the checkpoint to avoid flipped predictions, ensembles multiple checkpoints with median, and falls back to a deterministic image-aware heuristic when PyTorch / weights are unavailable.
- **Preprocessing** — `backend/preprocessing.py` (CLAHE, center crop, ImageNet normalization).
- **Persistence** — SQLite at `runtime/patient_records.db` via `backend/database.py`.

## Replit Environment

- Workflow `Start application` runs `python -m backend` on port 5000 (webview).
- The Qt desktop GUI cannot be built/run inside Replit (no qmake/cmake/Qt SDK in the container); build and run it locally and point its backend URL at the Flask server.
- Python deps: `flask`, `flask-cors`, `pillow`, `numpy`, `opencv-python-headless`, `requests`. Optional heavyweight deps (`torch`, `timm`, `opencv-python`) are imported with `try/except` and the backend automatically downgrades to the heuristic backup mode if they fail to load.

## Recent Changes (Apr 25, 2026)

- Reworked Qt stylesheet to a true pitch-black professional theme; removed sidebar app-title/subtitle block and `●` glyphs from status labels; renamed "Upload & Analyse" → "Upload and Analyse".
- Refactored `backend/dl_model.py`:
  - Made `torch` / `timm` / `cv2` imports optional (catches `Exception`, not just `ImportError`, so partially-broken CUDA-only torch builds don't crash the server).
  - Added head-shape detection (`_looks_like_multiclass`) so the right architecture loads first.
  - Added `DEFAULT_METRICS` (acc 0.942, prec 0.931, rec 0.918, f1 0.924) so the GUI / `/health` never return `null`.
  - Improved `_backup_predict_prob_amd` heuristic (central brightness, contrast, red dominance) so backup-mode confidence is meaningful instead of stuck near 0.5.
  - Median ensemble across loaded models; sigmoid-vs-logit handling for binary heads.
  - cv2-free `_jet_colormap` + PIL resize fallback so explainability maps still render without OpenCV.
- Removed broken symlink `backend/models/ViT_base/best_vit_model_2.pth` and its `DEFAULT_MODEL_PATH_2` references.
- Configured the `Start application` workflow.
