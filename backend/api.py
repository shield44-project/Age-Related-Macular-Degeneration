import hashlib
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from flask import Flask, jsonify, request
from PIL import Image

if __package__:
    from .dl_model import (
        CLASS_NAMES,
        generate_explainability_cam,
        get_model_status,
        list_available_models,
        predict_probabilities,
        set_active_model,
    )
    from .preprocessing import preprocess_for_inference, is_valid_fundus_image, decode_image_bytes
    from .database import (
        initialize_database,
        insert_patient_record,
        get_all_patients,
        get_patient_by_id,
        delete_patient_record,
        delete_all_patient_records,
    )
else:
    from dl_model import (
        CLASS_NAMES,
        generate_explainability_cam,
        get_model_status,
        list_available_models,
        predict_probabilities,
        set_active_model,
    )
    from preprocessing import preprocess_for_inference, is_valid_fundus_image, decode_image_bytes
    from database import (
        initialize_database,
        insert_patient_record,
        get_all_patients,
        get_patient_by_id,
        delete_patient_record,
        delete_all_patient_records,
    )

app = Flask(__name__)

UPLOAD_DIR = Path("runtime/uploads")
CAMS_DIR = Path("runtime/cams")

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
CAMS_DIR.mkdir(parents=True, exist_ok=True)

# Initialise persistent patient database on startup
initialize_database()


PROJECT_METRICS = {
    "accuracy": 0.80,
    "sensitivity": 0.79,
    "specificity": 0.83,
    "precision": 0.95,
    "f1_score": 0.864,
}


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
            "model_error": status["model_error"],
            "model_load_issues": status["model_load_issues"],
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

        decoded = decode_image_bytes(image_bytes)
        if not is_valid_fundus_image(decoded):
            raise ValueError("INVALID_FUNDUS_IMAGE")

        stem = Path(file.filename).stem or f"upload_{uuid4().hex[:8]}"
        upload_path = UPLOAD_DIR / f"{stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        Image.fromarray(decoded[:, :, ::-1]).save(upload_path)
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

    decoded = decode_image_bytes(image_bytes)
    if not is_valid_fundus_image(decoded):
        raise ValueError("INVALID_FUNDUS_IMAGE")

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
        image_hash = hashlib.sha256(image_bytes).hexdigest()
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
            output_path=CAMS_DIR / f"{path_stem}_gradcampp.png",
        )

        # Persist or update the scan in the patient database
        db_result = insert_patient_record(
            name=patient_name if patient_name else "Unknown",
            age=patient_age if 0 <= patient_age <= 120 else 0,
            image_path=str(Path(image_path).resolve()),
            prediction=prediction,
            confidence=confidence,
            image_hash=image_hash,
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
                "model_error": model_status["model_error"],
                "model_load_issues": model_status["model_load_issues"],
                "patient_name": patient_name,
                "patient_age": patient_age,
                "patient_id": patient_id,
                "image_path": str(Path(image_path).resolve()),
                "cam_image_path": str(Path(cam_path).resolve()),
                "cam_gradcampp_path": str(Path(cam_path).resolve()),
                "cam_gradcam_path": str(Path(cam_path).resolve()),
                "cam_combined_path": "",
                "eye_condition": prediction,
                "diagnosis": prediction,
                "prediction": prediction,
                "confidence": confidence,
                "accuracy": PROJECT_METRICS["accuracy"],
                "sensitivity": PROJECT_METRICS["sensitivity"],
                "specificity": PROJECT_METRICS["specificity"],
                "precision": PROJECT_METRICS["precision"],
                "f1_score": PROJECT_METRICS["f1_score"],
                "metrics_source": "project_reported_metrics",
                "model_metrics": model_status["metrics"],
                "class_probabilities": {
                    name: float(prob) for name, prob in zip(CLASS_NAMES, probs)
                },
            }
        )
    except ValueError as exc:
        if str(exc) == "INVALID_FUNDUS_IMAGE":
            return jsonify(
                {
                    "error": "Invalid fundus image. Please upload a valid retinal fundus image.",
                    "invalid_fundus": True,
                }
            ), 400
        return jsonify({"error": f"Failed to process image: {str(exc)}"}), 400
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


@app.delete("/patients/<int:patient_id>")
def patient_delete(patient_id: int):
    """Delete a single patient record by ID."""
    result = delete_patient_record(patient_id)
    if not result.get("success"):
        return jsonify({"error": result.get("message", "Delete failed.")}), 404
    return jsonify({"success": True, "deleted": result.get("deleted", 1), "id": patient_id})


@app.delete("/patients")
def patients_clear():
    """Delete every patient record. Intended for the GUI's 'Clear All' action."""
    result = delete_all_patient_records()
    if not result.get("success"):
        return jsonify({"error": result.get("message", "Clear failed.")}), 500
    return jsonify({"success": True, "deleted": result.get("deleted", 0)})


# ---------------------------------------------------------------------------
# Model management endpoints
# ---------------------------------------------------------------------------

@app.get("/models")
def models_list():
    """Return all discovered model checkpoint files with active-model marker."""
    available = list_available_models()
    return jsonify({"models": available, "count": len(available)})


@app.post("/models/active")
def models_set_active():
    """Switch the active inference model to the checkpoint at the given path.

    Expected JSON body: ``{"path": "/absolute/or/relative/path/to/model.pth"}``
    """
    payload = request.get_json(silent=True) or {}
    model_path = payload.get("path", "").strip()
    if not model_path:
        return jsonify({"error": "Missing 'path' in request body."}), 400
    try:
        result = set_active_model(model_path)
        return jsonify({
            "success": True,
            "model_name": result["name"],
            "model_path": result["path"],
        })
    except FileNotFoundError:
        return jsonify({"error": "Model checkpoint not found or not permitted."}), 404
    except Exception as exc:
        print(f"[models/active] Failed to load model: {exc}")
        return jsonify({"error": "Failed to load the requested model checkpoint."}), 500
