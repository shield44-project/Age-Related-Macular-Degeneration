# Backend API Documentation

## Overview

This backend provides a Flask API for AMD image prediction with persistent patient record storage.

Main endpoint:
- POST /predict

Additional endpoints:
- GET /
- GET /health
- GET /patients
- GET /patients/<id>

The predict endpoint supports both:
- Multipart upload input (form-data key: image)
- JSON input with a local image path (image_path)

This is useful for both web/API testing and Qt GUI integration.

## Folder Structure

- backend/server.py: App runner (entry point)
- backend/api.py: Flask app and route logic
- backend/dl_model.py: Model loading and prediction helpers
- backend/preprocessing.py: Image preprocessing and fundus validation
- backend/database.py: SQLite patient record persistence
- backend/models/: Put trained model files here

## Training a Better Model

### Option A: Kaggle Notebook

Open `backend/models/ViT_base/ViT_base_kaggle.ipynb` on Kaggle.
Make sure the AMD dataset is mounted at the path in cell 1.
Run both cells. The best checkpoint is saved to `/kaggle/working/best_vit_model.pth`.
Download it and place it at `backend/models/ViT_base/best_vit_model.pth`.

### Option B: Standalone Training Script

Run `train.py` from the project root:

```bash
pip install -r requirements.txt scikit-learn

python train.py \
  --train_amd    /path/to/train/amd \
  --train_normal /path/to/train/normal \
  --val_amd      /path/to/val/amd \
  --val_normal   /path/to/val/normal \
  --output_dir   backend/models/ViT_base \
  --output_name  best_vit_model.pth
```

Key training improvements over the original notebook:
- **Two-phase fine-tuning**: 5 epochs head-only (backbone frozen) then 30 epochs full
  fine-tuning with discriminative learning rates (backbone 5e-6, head 5e-4)
- **Focal loss** with label smoothing — handles class imbalance, regularises predictions
- **Fundus-specific augmentation**: GridDistortion, RandomGamma, HueSaturationValue,
  random CLAHE clip variation, CoarseDropout
- **AdamW + cosine LR schedule** with linear warm-up
- **Mixed precision (AMP)** + gradient clipping
- **Early stopping** (patience = 7)
- **Full metrics saved in checkpoint** — accuracy, precision, recall, F1
  (the GUI /health endpoint and Qt display will show real numbers automatically)

The checkpoint format is:
```python
{
  "state_dict": ...,
  "model_name": "ViT-B16 AMD Classifier (Fine-Tuned)",
  "accuracy":   0.XXX,
  "precision":  0.XXX,
  "recall":     0.XXX,
  "f1_score":   0.XXX,
  "epoch":      N,
}
```

## Model Architecture (v2)

`ViTBinaryClassifier` uses ViT-B/16 backbone (timm) with an improved 3-layer head:

```
ViT-B/16 backbone  →  (B, 768)
LayerNorm(768)
Dropout(0.3)
Linear(768 → 512) + GELU
Dropout(0.2)
Linear(512 → 128) + GELU
Dropout(0.1)
Linear(128 → 1)   + Sigmoid
→ (B, 1)  AMD probability
```

## How Model Selection Works

Model selection is handled in backend/dl_model.py.

Priority order:
1. Use MODEL_PATH environment variable if provided.
2. Else use the bundled checkpoint at backend/models/ViT_base/best_vit_model.pth.
3. If the model file is a wrapped checkpoint, the loader will extract the first supported state_dict key.
4. If model loading fails or file is missing, use backup heuristic inference.

Response field model_type tells which one is active:
- real
- backup

## Endpoint: POST /predict

### Input Option A: Multipart Form

Content-Type: multipart/form-data

Required field:
- image: image file (must be a retinal fundus photograph)

Optional fields:
- patient_name (string)
- patient_age  (integer, 0-120)

Non-fundus images are rejected with HTTP 400 and `"invalid_fundus": true`.

### Input Option B: JSON

Content-Type: application/json

Example:
{
  "image_path": "/absolute/or/workspace/path/to/image.jpg",
  "patient_name": "John Doe",
  "patient_age": 65
}

Required key:
- image_path

Optional keys:
- patient_name
- patient_age

### Response JSON

Successful response includes:
- model_type: backup or real
- patient_name, patient_age, patient_id (DB record ID)
- image_path: source or saved upload path
- cam_image_path: generated saliency image path
- diagnosis / prediction: predicted class label
- eye_condition: same as prediction
- confidence: top class probability
- class_probabilities: map of class -> probability
- accuracy, precision, recall, f1_score: model metrics
- model_name, model_paths, models_loaded, backup_active

Every prediction is auto-saved to the SQLite database (runtime/patient_records.db).

## Endpoint: GET /patients

Returns all patient records ordered by date descending.

Response JSON:
- success (bool)
- count (int)
- patients (list): each record has id, name, age, image_path, prediction, confidence, date

## Endpoint: GET /patients/<id>

Returns a single patient record by its integer ID.

Returns 404 with success=false if not found.

## Endpoint: GET /health

Returns service health and active model info.

## Endpoint: GET /

Returns a simple JSON message confirming the backend is running and listing available routes.

## Run the Backend

From project root:
Make sure to `pip install -r requirements.txt`

python -m backend

Alternative script mode:

python backend/server.py

Server runs on:
- http://0.0.0.0:5000

## Quick Test Commands

### Test with file upload

curl -X POST http://localhost:5000/predict \
  -F "image=@/path/to/fundus.jpg" \
  -F "patient_name=John Doe" \
  -F "patient_age=65"

### Test with JSON path

curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"image_path":"/path/to/fundus.jpg","patient_name":"John Doe","patient_age":65}'

### List all patient records

curl http://localhost:5000/patients

### Get a specific patient record

curl http://localhost:5000/patients/1

## Add a Real Model Later

Option 1:
- Place trained model at backend/models/ViT_base/best_vit_model.pth

Option 2:
- Set environment variable MODEL_PATH to your model location:

MODEL_PATH=/absolute/path/to/model.pt python -m backend

## Notes

- Backup mode predictions are heuristic-based and not medically meaningful.
- Generated CAM image is a gradient saliency overlay (real model) or intensity-based (backup).
- Patient records are stored in runtime/patient_records.db (created automatically).


Main endpoint:
- POST /predict

Additional endpoints:
- GET /
- GET /health
- GET /patients
- GET /patients/<id>

The predict endpoint supports both:
- Multipart upload input (form-data key: image)
- JSON input with a local image path (image_path)

This is useful for both web/API testing and Qt GUI integration.

## Folder Structure

- backend/server.py: App runner (entry point)
- backend/api.py: Flask app and route logic
- backend/dl_model.py: Model loading and prediction helpers
- backend/preprocessing.py: Image preprocessing and CAM generation
- backend/database.py: SQLite patient record persistence
- backend/models/: Put trained model files here

## How Model Selection Works

Model selection is handled in backend/dl_model.py.

Priority order:
1. Use MODEL_PATH environment variable if provided.
2. Else use the bundled checkpoint at backend/models/ViT_base/best_vit_model.pth.
3. If the model file is a wrapped checkpoint, the loader will extract the first supported state_dict key.
4. If model loading fails or file is missing, use backup heuristic inference.

Response field model_type tells which one is active:
- real
- backup

## Endpoint: POST /predict

### Input Option A: Multipart Form

Content-Type: multipart/form-data

Required field:
- image: image file

Optional fields:
- patient_name (string)
- patient_age  (integer, 0-120)

### Input Option B: JSON

Content-Type: application/json

Example:
{
  "image_path": "/absolute/or/workspace/path/to/image.jpg",
  "patient_name": "John Doe",
  "patient_age": 65
}

Required key:
- image_path

Optional keys:
- patient_name
- patient_age

### Response JSON

Successful response includes:
- model_type: backup or real
- patient_name, patient_age, patient_id (DB record ID)
- image_path: source or saved upload path
- cam_image_path: generated saliency image path
- diagnosis / prediction: predicted class label
- eye_condition: same as prediction
- confidence: top class probability
- class_probabilities: map of class -> probability
- accuracy, precision, recall, f1_score: model metrics
- model_name, model_paths, models_loaded, backup_active

Every prediction is auto-saved to the SQLite database (runtime/patient_records.db).

## Endpoint: GET /patients

Returns all patient records ordered by date descending.

Response JSON:
- success (bool)
- count (int)
- patients (list): each record has id, name, age, image_path, prediction, confidence, date

## Endpoint: GET /patients/<id>

Returns a single patient record by its integer ID.

Returns 404 with success=false if not found.

## Endpoint: GET /health

Returns service health and active model info.

## Endpoint: GET /

Returns a simple JSON message confirming the backend is running and listing available routes.

## Run the Backend

From project root:
Make sure to `pip install -r requirements.txt`

python -m backend

Alternative script mode:

python backend/server.py

Server runs on:
- http://0.0.0.0:5000

## Quick Test Commands

### Test with file upload

curl -X POST http://localhost:5000/predict \
  -F "image=@/path/to/fundus.jpg" \
  -F "patient_name=John Doe" \
  -F "patient_age=65"

### Test with JSON path

curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"image_path":"/path/to/fundus.jpg","patient_name":"John Doe","patient_age":65}'

### List all patient records

curl http://localhost:5000/patients

### Get a specific patient record

curl http://localhost:5000/patients/1

## Add a Real Model Later

Option 1:
- Place trained model at backend/models/ViT_base/best_vit_model.pth

Option 2:
- Set environment variable MODEL_PATH to your model location:

MODEL_PATH=/absolute/path/to/model.pt python -m backend

## Notes

- Backup mode predictions are heuristic-based and not medically meaningful.
- Generated CAM image is a gradient saliency overlay (real model) or intensity-based (backup).
- Patient records are stored in runtime/patient_records.db (created automatically).
