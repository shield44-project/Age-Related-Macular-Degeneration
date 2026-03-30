# Backend API Documentation

## Overview

This backend provides a Flask API for AMD image prediction.
It currently supports a dummy PyTorch model for development and can automatically switch to a real trained model when available.

Main endpoint:
- POST /predict

The endpoint supports both:
- Multipart upload input (form-data key: image)
- JSON input with a local image path (image_path)

This is useful for both web/API testing and Qt GUI integration.

## Folder Structure

- backend/backend.py: App runner (entry point)
- backend/api.py: Flask app and route logic
- backend/dl_model.py: Model loading and prediction helpers
- backend/preprocessing.py: Image preprocessing and dummy CAM generation
- backend/models/: Put trained model files here

## How Model Selection Works

Model selection is handled in backend/dl_model.py.

Priority order:
1. Use MODEL_PATH environment variable if provided.
2. Else use default path: backend/models/amd_model.pt
3. If model loading fails or file is missing, use dummy model.

Response field model_type tells which one is active:
- real
- dummy

## Endpoint: POST /predict

### Input Option A: Multipart Form

Content-Type: multipart/form-data

Required field:
- image: image file

Optional field:
- patient_name (if sent in JSON mode only; multipart clients can keep this empty for now)

### Input Option B: JSON

Content-Type: application/json

Example:
{
  "image_path": "/absolute/or/workspace/path/to/image.jpg",
  "patient_name": "John Doe"
}

Required key:
- image_path

Optional key:
- patient_name

## Response JSON

Successful response includes:
- model_type: dummy or real
- patient_name: patient name if provided
- image_path: source or saved upload path
- cam_image_path: generated CAM-like image path
- diagnosis: predicted class label
- prediction: same as diagnosis
- confidence: top class probability
- class_probabilities: map of class -> probability

Example:
{
  "model_type": "dummy",
  "patient_name": "John Doe",
  "image_path": "runtime/uploads/fundus_20260322_101010.png",
  "cam_image_path": "runtime/cams/fundus_20260322_101010_cam.png",
  "diagnosis": "Normal Eye",
  "prediction": "Normal Eye",
  "confidence": 0.61,
  "class_probabilities": {
    "Normal Eye": 0.61,
    "Treatable AMD": 0.22,
    "Non-Treatable AMD": 0.17
  }
}

On failure:
- status code: 400
- JSON key: error

## Run the Backend

From project root:

python backend/backend.py

Server runs on:
- http://0.0.0.0:5000

## Quick Test Commands

### Test with file upload

curl -X POST http://localhost:5000/predict \
  -F "image=@/path/to/fundus.jpg"

### Test with JSON path

curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"image_path":"/path/to/fundus.jpg","patient_name":"John Doe"}'

## Add a Real Model Later

Option 1:
- Place trained model at backend/models/amd_model.pt

Option 2:
- Set environment variable MODEL_PATH to your model location:

MODEL_PATH=/absolute/path/to/model.pt python backend/backend.py

## Notes

- Dummy model predictions are placeholders and not medically meaningful.
- Generated CAM image is a simulated overlay for UI integration, not Grad-CAM.
