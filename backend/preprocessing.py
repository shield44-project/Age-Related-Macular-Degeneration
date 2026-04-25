import io
import numpy as np
import cv2
from PIL import Image

IMAGE_SIZE = (224, 224)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def is_valid_fundus_image(bgr_img: np.ndarray) -> bool:
    """
    Heuristic check to determine whether an image looks like a retinal fundus photograph.

    Fundus images have three consistent characteristics:
      1. Very dark / black corners (the camera vignette / background).
      2. A clearly brighter circular region in the centre (the retina).
      3. Warm colour tone: the red channel dominates the blue channel inside the
         bright region (orange/red retinal tissue).

    Returns True when all criteria are satisfied, False otherwise.
    """
    h, w = bgr_img.shape[:2]
    if h < 64 or w < 64:
        return False

    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)

    # ── 1. Dark corners ──────────────────────────────────────────────────────
    ch = max(h // 8, 16)
    cw = max(w // 8, 16)
    corner_pixels = np.concatenate([
        gray[:ch, :cw].ravel(),
        gray[:ch, -cw:].ravel(),
        gray[-ch:, :cw].ravel(),
        gray[-ch:, -cw:].ravel(),
    ])
    corner_mean = float(corner_pixels.mean())
    # Fundus background is virtually black (typical mean ≤ 55)
    if corner_mean > 55:
        return False

    # ── 2. Brighter centre ───────────────────────────────────────────────────
    cy, cx = h // 2, w // 2
    r_inner = min(h, w) // 4
    center_patch = gray[
        max(0, cy - r_inner): cy + r_inner,
        max(0, cx - r_inner): cx + r_inner,
    ]
    center_mean = float(center_patch.mean())
    # Centre must be noticeably brighter than the dark corners
    if center_mean < corner_mean + 25:
        return False

    # ── 3. Sufficient bright-pixel coverage ──────────────────────────────────
    bright_fraction = float((gray > 25).mean())
    # Retina fills roughly 30–85 % of a typical fundus image
    if bright_fraction < 0.15 or bright_fraction > 0.95:
        return False

    # ── 4. Warm colour dominance in the bright region ────────────────────────
    bright_mask = gray > 25
    b_ch, g_ch, r_ch = cv2.split(bgr_img)
    bright_r = float(r_ch[bright_mask].mean()) if bright_mask.any() else 0.0
    bright_b = float(b_ch[bright_mask].mean()) if bright_mask.any() else 0.0
    # In a fundus image the red channel is always dominant over the blue channel
    if bright_r < bright_b * 0.95:
        return False

    return True


def decode_image_bytes(image_bytes: bytes) -> np.ndarray:
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        # Fallback for image variants OpenCV occasionally fails to decode.
        try:
            pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            rgb = np.asarray(pil_img, dtype=np.uint8)
            image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        except Exception as exc:
            raise ValueError(f"Failed to decode image bytes: {exc}") from exc
    return image


def apply_clahe(bgr_img: np.ndarray) -> np.ndarray:
    """BGR uint8 -> CLAHE grayscale stacked to 3-channel uint8."""
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return np.stack([enhanced, enhanced, enhanced], axis=-1)


def center_crop(img: np.ndarray, crop_size: int) -> np.ndarray:
    """Square center crop with reflection padding if needed."""
    h, w = img.shape[:2]
    if h < crop_size or w < crop_size:
        pad_h = max(0, crop_size - h)
        pad_w = max(0, crop_size - w)
        img = cv2.copyMakeBorder(
            img,
            pad_h // 2,
            pad_h - pad_h // 2,
            pad_w // 2,
            pad_w - pad_w // 2,
            cv2.BORDER_REFLECT_101,
        )
        h, w = img.shape[:2]
    top = (h - crop_size) // 2
    left = (w - crop_size) // 2
    return img[top : top + crop_size, left : left + crop_size]


def preprocess_bgr_image(bgr_img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Notebook-aligned preprocessing returning model tensor + display RGB image."""
    image = apply_clahe(bgr_img)
    image = center_crop(image, IMAGE_SIZE[0])
    image = cv2.resize(image, IMAGE_SIZE, interpolation=cv2.INTER_AREA)

    display_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = image.astype(np.float32) / 255.0
    image = (image - IMAGENET_MEAN) / IMAGENET_STD
    image = np.transpose(image, (2, 0, 1))
    tensor = np.expand_dims(image, axis=0)
    return tensor, display_rgb


def preprocess_for_inference(image_bytes: bytes) -> tuple[np.ndarray, np.ndarray]:
    bgr_img = decode_image_bytes(image_bytes)
    return preprocess_bgr_image(bgr_img)


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    tensor, _ = preprocess_for_inference(image_bytes)
    return tensor
