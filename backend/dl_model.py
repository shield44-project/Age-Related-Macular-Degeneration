import os
from pathlib import Path

import numpy as np
import torch
from torch import nn


###IMAGE PREPROCESSING###


CLASS_NAMES = ["Normal Eye", "Treatable AMD", "Non-Treatable AMD"]
IMAGE_SIZE = (224, 224)
DEFAULT_MODEL_PATH = Path("backend/models/amd_model.pt")


class DummyAMDModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 224 * 224, len(CLASS_NAMES)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


def build_dummy_model() -> nn.Module:
    torch.manual_seed(42)
    return DummyAMDModel()


def resolve_model_path() -> Path:
    model_path = os.getenv("MODEL_PATH")
    if model_path:
        return Path(model_path)
    return DEFAULT_MODEL_PATH


def _load_pytorch_model(model_path: Path) -> nn.Module:
    try:
        model = torch.jit.load(str(model_path), map_location="cpu")
        return model.eval()
    except Exception:
        pass

    loaded = torch.load(model_path, map_location="cpu")
    if isinstance(loaded, nn.Module):
        return loaded.eval()

    raise ValueError(
        "Unsupported model format. Provide a TorchScript module or a saved nn.Module."
    )


def load_model_with_fallback() -> tuple[nn.Module, str]:
    model_path = resolve_model_path()

    if model_path.exists() and model_path.is_file():
        try:
            model = _load_pytorch_model(model_path)
            return model, "real"
        except Exception:
            pass

    return build_dummy_model().eval(), "dummy"


MODEL, MODEL_TYPE = load_model_with_fallback()


def predict_probabilities(input_tensor: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        tensor = torch.as_tensor(input_tensor, dtype=torch.float32)
        output = MODEL(tensor)

        if isinstance(output, (tuple, list)):
            output = output[0]

        if output.ndim == 1:
            output = output.unsqueeze(0)

        probs = torch.softmax(output, dim=1)
        return probs[0].cpu().numpy()
