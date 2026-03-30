from io import BytesIO
from pathlib import Path
import numpy as np
import cv2
from PIL import Image

IMAGE_SIZE = (224, 224) #I changed this because import IMAGE_SIZE from dl_models would not work - dl_models is a directory

def preprocess_image(image_bytes: bytes) -> np.ndarray: #I changed the preprocessing such that it matches what I have used in the kaggle notebooks exactly. Added CLAHE
    # Decode bytes → BGR uint8
    nparr = np.frombuffer(image_bytes, np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # CLAHE grayscale → 3-channel
    gray     = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    img      = np.stack([enhanced, enhanced, enhanced], axis=-1)

    # Center crop to 224×224
    h, w = img.shape[:2]
    if h < 224 or w < 224:
        pad_h = max(0, 224 - h)
        pad_w = max(0, 224 - w)
        img = cv2.copyMakeBorder(
            img,
            pad_h // 2, pad_h - pad_h // 2,
            pad_w // 2, pad_w - pad_w // 2,
            cv2.BORDER_REFLECT_101
        )
        h, w = img.shape[:2]
    top  = (h - 224) // 2
    left = (w - 224) // 2
    img  = img[top:top+224, left:left+224]

    # Resize + ImageNet normalise
    img = cv2.resize(img, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])

    # (H,W,C) → (1,C,H,W) for PyTorch
    img = np.transpose(img, (2, 0, 1))
    return np.expand_dims(img, axis=0)   # (1,3,224,224)

def create_dummy_cam(image_bytes: bytes, stem: str, cams_dir: Path) -> str: #I have kept this function unchanged for now, will have to change this later when we have heatmap generation in our models
    source = Image.open(BytesIO(image_bytes)).convert("RGB").resize(IMAGE_SIZE)
    source_arr = np.asarray(source, dtype=np.float32)
    gray = source_arr.mean(axis=2)
    gray = gray - gray.min()
    denom = max(gray.max(), 1e-6)
    gray = gray / denom
    heat = np.zeros_like(source_arr)
    heat[..., 0] = gray * 255.0
    blended = (0.6 * source_arr + 0.4 * heat).clip(0, 255).astype(np.uint8)
    cam_image = Image.fromarray(blended)
    cam_path = cams_dir / f"{stem}_cam.png"
    cam_image.save(cam_path)
    return str(cam_path)
