from datetime import datetime
from io import BytesIO
from pathlib import Path
from uuid import uuid4

from flask import Flask, jsonify, request
from PIL import Image

from .dl_model import CLASS_NAMES, MODEL_TYPE, predict_probabilities
from .preprocessing import create_dummy_cam, preprocess_image

app = Flask(__name__)

UPLOAD_DIR = Path("runtime/uploads")
CAMS_DIR = Path("runtime/cams")

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
CAMS_DIR.mkdir(parents=True, exist_ok=True)


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
        patient_name = payload.get("patient_name", "")

        image_bytes, image_path = get_request_image()
        input_tensor = preprocess_image(image_bytes)

        probs = predict_probabilities(input_tensor)
        pred_idx = int(probs.argmax())
        prediction = CLASS_NAMES[pred_idx]
        confidence = float(probs[pred_idx])

        path_stem = Path(image_path).stem
        cam_path = create_dummy_cam(image_bytes, path_stem, CAMS_DIR)

        return jsonify(
            {
                "model_type": MODEL_TYPE,
                "patient_name": patient_name,
                "image_path": image_path,
                "cam_image_path": cam_path,
                "diagnosis": prediction,
                "prediction": prediction,
                "confidence": confidence,
                "class_probabilities": {
                    name: float(prob) for name, prob in zip(CLASS_NAMES, probs)
                },
            }
        )
    except Exception as exc:
        return jsonify({"error": f"Failed to process image: {str(exc)}"}), 400
