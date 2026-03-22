from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

from .dl_model import IMAGE_SIZE


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image = image.resize(IMAGE_SIZE)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return np.expand_dims(array, axis=0)


def create_dummy_cam(image_bytes: bytes, stem: str, cams_dir: Path) -> str:
    source = Image.open(BytesIO(image_bytes)).convert("RGB").resize(IMAGE_SIZE)
    source_arr = np.asarray(source, dtype=np.float32)

    gray = source_arr.mean(axis=2)
    gray = gray - gray.min()
    denom = max(gray.max(), 1e-6)
    gray = gray / denom

    # Red-tinted overlay to mimic lesion attention map for UI wiring.
    heat = np.zeros_like(source_arr)
    heat[..., 0] = gray * 255.0

    blended = (0.6 * source_arr + 0.4 * heat).clip(0, 255).astype(np.uint8)
    cam_image = Image.fromarray(blended)

    cam_path = cams_dir / f"{stem}_cam.png"
    cam_image.save(cam_path)
    return str(cam_path)
