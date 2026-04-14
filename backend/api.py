from datetime import datetime
from io import BytesIO
from pathlib import Path
from uuid import uuid4

from flask import Flask, jsonify, request
from PIL import Image

if __package__:
    from .dl_model import (
        CLASS_NAMES,
        generate_explainability_cam,
        get_model_status,
        predict_probabilities,
    )
    from .preprocessing import preprocess_for_inference
else:
    from dl_model import (
        CLASS_NAMES,
        generate_explainability_cam,
        get_model_status,
        predict_probabilities,
    )
    from preprocessing import preprocess_for_inference

app = Flask(__name__)

UPLOAD_DIR = Path("runtime/uploads")
CAMS_DIR = Path("runtime/cams")

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
CAMS_DIR.mkdir(parents=True, exist_ok=True)


@app.get("/")
def index():
    return jsonify(
        {
            "message": "AMD backend is running.",
            "endpoints": {
                "health": "/health",
                "predict": "/predict",
            },
        }
    )


@app.get("/health")
def health():
    status = get_model_status()
    return jsonify(
        {
            "status": "ok",
            "model_type": status["model_type"],
            "model_name": status["model_name"],
            "backup_active": status["backup_active"],
            "model_path": status["model_path"],
            "model_paths": status["model_paths"],
            "model_names": status["model_names"],
            "models_loaded": status["models_loaded"],
            "metrics": status["metrics"],
        }
    )


def get_request_image() -> tuple[bytes, str]:
    if "image" in request.files:
        file = request.files["image"]
        if file.filename == "":
            raise ValueError("Empty filename.")

        image_bytes = file.read()
        if not image_bytes:
            raise ValueError("Uploaded file is empty.")

        stem = Path(file.filename).stem or f"upload_{uuid4().hex[:8]}"
        upload_path = UPLOAD_DIR / f"{stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        Image.open(BytesIO(image_bytes)).convert("RGB").save(upload_path)
        return image_bytes, str(upload_path)

    payload = request.get_json(silent=True) or {}
    image_path = payload.get("image_path")
    if not image_path:
        raise ValueError(
            "No image input found. Send form-data 'image' or JSON with 'image_path'."
        )

    source_path = Path(image_path)
    if not source_path.exists() or not source_path.is_file():
        raise FileNotFoundError(f"Image path not found: {image_path}")

    image_bytes = source_path.read_bytes()
    if not image_bytes:
        raise ValueError("Image file is empty.")

    return image_bytes, str(source_path)


@app.post("/predict")
def predict():
    try:
        payload = request.get_json(silent=True) or {}
        # Support both JSON clients and multipart form-data clients (Qt GUI).
        patient_name = (payload.get("patient_name") or request.form.get("patient_name", "")).strip()

        image_bytes, image_path = get_request_image()
        input_tensor, cam_base_rgb = preprocess_for_inference(image_bytes)

        probs = predict_probabilities(input_tensor)
        pred_idx = int(probs.argmax())
        prediction = CLASS_NAMES[pred_idx]
        confidence = float(probs[pred_idx])

        path_stem = Path(image_path).stem
        cam_path = generate_explainability_cam(
            input_tensor=input_tensor,
            base_rgb=cam_base_rgb,
            predicted_idx=pred_idx,
            output_path=CAMS_DIR / f"{path_stem}_cam.png",
        )

        model_status = get_model_status()

        return jsonify(
            {
                "model_type": model_status["model_type"],
                "model_name": model_status["model_name"],
                "backup_active": model_status["backup_active"],
                "model_paths": model_status["model_paths"],
                "model_names": model_status["model_names"],
                "models_loaded": model_status["models_loaded"],
                "patient_name": patient_name,
                "image_path": str(Path(image_path).resolve()),
                "cam_image_path": str(Path(cam_path).resolve()),
                "eye_condition": prediction,
                "diagnosis": prediction,
                "prediction": prediction,
                "confidence": confidence,
                "accuracy": model_status["metrics"]["accuracy"],
                "precision": model_status["metrics"]["precision"],
                "recall": model_status["metrics"]["recall"],
                "f1_score": model_status["metrics"]["f1_score"],
                "class_probabilities": {
                    name: float(prob) for name, prob in zip(CLASS_NAMES, probs)
                },
            }
        )
    except Exception as exc:
        return jsonify({"error": f"Failed to process image: {str(exc)}"}), 400
