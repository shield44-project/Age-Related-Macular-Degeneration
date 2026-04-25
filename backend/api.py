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
    from .preprocessing import preprocess_for_inference, is_valid_fundus_image, decode_image_bytes
    from .database import (
        initialize_database,
        insert_patient_record,
        get_all_patients,
        get_patient_by_id,
    )
else:
    from dl_model import (
        CLASS_NAMES,
        generate_explainability_cam,
        get_model_status,
        predict_probabilities,
    )
    from preprocessing import preprocess_for_inference, is_valid_fundus_image, decode_image_bytes
    from database import (
        initialize_database,
        insert_patient_record,
        get_all_patients,
        get_patient_by_id,
    )

app = Flask(__name__)

UPLOAD_DIR = Path("runtime/uploads")
CAMS_DIR = Path("runtime/cams")

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
CAMS_DIR.mkdir(parents=True, exist_ok=True)

# Initialise persistent patient database on startup
initialize_database()


@app.get("/")
def index():
    return jsonify(
        {
            "message": "AMD backend is running.",
            "endpoints": {
                "health": "/health",
                "predict": "/predict",
                "patients": "/patients",
                "patient": "/patients/<id>",
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
        patient_age_raw = (payload.get("patient_age") or request.form.get("patient_age", "")).strip()
        try:
            patient_age = int(patient_age_raw) if patient_age_raw else 0
        except (ValueError, TypeError):
            patient_age = 0
            print(f"Warning: could not parse patient_age value {patient_age_raw!r}; defaulting to 0.")

        image_bytes, image_path = get_request_image()

        # Validate that the image is a retinal fundus photograph before any
        # heavy model inference so we can return a clear error immediately.
        raw_bgr = decode_image_bytes(image_bytes)
        if not is_valid_fundus_image(raw_bgr):
            return jsonify({
                "error": "Invalid Fundus Image. Please enter a valid eye fundus image.",
                "invalid_fundus": True,
            }), 400

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

        # Persist the scan to the patient database
        db_result = insert_patient_record(
            name=patient_name if patient_name else "Unknown",
            age=patient_age if 0 <= patient_age <= 120 else 0,
            image_path=str(Path(image_path).resolve()),
            prediction=prediction,
            confidence=confidence,
        )
        patient_id = db_result.get("record_id") if db_result.get("success") else None

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
                "patient_age": patient_age,
                "patient_id": patient_id,
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


@app.get("/patients")
def patients():
    """Return all patient records from the database."""
    result = get_all_patients()
    if not result.get("success"):
        return jsonify({"error": "Failed to retrieve patient records."}), 500
    # Explicitly return only known-safe fields to avoid taint propagation.
    return jsonify({"success": True, "count": result["count"], "patients": result["patients"]})


@app.get("/patients/<int:patient_id>")
def patient(patient_id: int):
    """Return a single patient record by ID."""
    result = get_patient_by_id(patient_id)
    if not result.get("success"):
        return jsonify({"error": f"No patient found with ID {patient_id}."}), 404
    # Explicitly return only the patient record fields.
    return jsonify({"success": True, "patient": result["patient"]})
