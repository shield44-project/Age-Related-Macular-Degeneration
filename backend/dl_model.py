import os
from pathlib import Path

import numpy as np
import tensorflow as tf

CLASS_NAMES = ["Normal Eye", "Treatable AMD", "Non-Treatable AMD"]
IMAGE_SIZE = (224, 224)
DEFAULT_MODEL_PATH = Path("backend/models/amd_model.keras")


def build_dummy_model() -> tf.keras.Model:
    tf.random.set_seed(42)
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(224, 224, 3)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(len(CLASS_NAMES), activation="softmax"),
        ]
    )
    return model


def resolve_model_path() -> Path:
    model_path = os.getenv("MODEL_PATH")
    if model_path:
        return Path(model_path)
    return DEFAULT_MODEL_PATH


def load_model_with_fallback() -> tuple[tf.keras.Model, str]:
    model_path = resolve_model_path()

    if model_path.exists() and model_path.is_file():
        try:
            model = tf.keras.models.load_model(model_path)
            return model, "real"
        except Exception:
            pass

    return build_dummy_model(), "dummy"


MODEL, MODEL_TYPE = load_model_with_fallback()


def predict_probabilities(input_tensor: np.ndarray) -> np.ndarray:
    return MODEL.predict(input_tensor, verbose=0)[0]
