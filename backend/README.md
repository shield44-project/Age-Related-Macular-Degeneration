# Backend API Documentation

## Overview

This backend provides a Flask API for AMD image prediction with persistent patient record storage.

Main endpoint:
- POST /predict

Fundus validation:
- Non-fundus images are rejected with HTTP 400 and `invalid_fundus=true`.

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
- accuracy, precision, recall, f1_score: per-image analysis scores derived from the current prediction confidence and probability margin
- model_metrics: checkpoint / validation-set metrics for the loaded model
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

## Train and Evaluate an Improved Model

Use the provided script to train on one or more datasets and test the resulting checkpoint.

Expected layout for each dataset root:

```text
<dataset_root>/
  train/
    Normal/
    AMD/
  val/
    Normal/
    AMD/
  test/            # optional but recommended
    Normal/
    AMD/
```

Example command:

```bash
python -m backend.train_and_evaluate \
  --dataset-roots /data/dataset_a /data/dataset_b /data/dataset_c \
  --epochs 30 \
  --batch-size 16 \
  --output backend/models/ViT_base/best_vit_model_improved.pth
```

Outputs:
- Best checkpoint file: `best_vit_model_improved.pth`
- Metrics report JSON: `best_vit_model_improved.metrics.json`

The backend model loader now supports both legacy binary-sigmoid checkpoints and 2-class ViT checkpoints saved by this script.

## Notes

- Backup mode predictions are heuristic-based and not medically meaningful.
- Generated CAM image is a gradient saliency overlay (real model) or intensity-based (backup).
- Patient records are stored in runtime/patient_records.db (created automatically).
