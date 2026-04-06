import numpy as np
import cv2

IMAGE_SIZE = (224, 224)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def decode_image_bytes(image_bytes: bytes) -> np.ndarray:
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Failed to decode image bytes.")
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
