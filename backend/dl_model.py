import os
from pathlib import Path
import numpy as np
import torch
import timm
from torch import nn

CLASS_NAMES = ["Normal", "AMD"]
IMAGE_SIZE = (224, 224)
DEFAULT_MODEL_PATH = Path("backend/models/ViT_base/best_vit_model.pth")

class ViTBinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            "vit_base_patch16_224",
            pretrained=False,
            num_classes=0,
        )
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(768, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)

def resolve_model_path() -> Path:
    model_path = os.getenv("MODEL_PATH")
    if model_path:
        return Path(model_path)
    return DEFAULT_MODEL_PATH

def load_model_with_fallback() -> tuple[nn.Module, str]:
    model_path = resolve_model_path()
    if model_path.exists():
        try:
            model = ViTBinaryClassifier()
            model.load_state_dict(
                torch.load(model_path, map_location="cpu")
            )
            return model.eval(), "real"
        except Exception as e:
            print(f"Model load failed: {e}")
    model = ViTBinaryClassifier()
    return model.eval(), "dummy"

MODEL, MODEL_TYPE = load_model_with_fallback()

def predict_probabilities(input_tensor: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        tensor   = torch.as_tensor(input_tensor, dtype=torch.float32)
        prob_amd = MODEL(tensor).item()
    return np.array([1 - prob_amd, prob_amd])
