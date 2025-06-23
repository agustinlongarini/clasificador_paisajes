import io
import os
from typing import Tuple

from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import models, transforms

# Lista de clases del modelo
CLASSES = [
    "buildings",
    "forest",
    "glacier",
    "mountain",
    "sea",
    "street",
]

# Transformaciones utilizadas en la fase de inferencia
_preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_model(model_path: str, device: torch.device | None = None) -> Tuple[torch.nn.Module, torch.device]:
    """Carga el modelo entrenado desde ``model_path``."""
    global CLASSES
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Crear la arquitectura base
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, len(CLASSES))

    # Cargar pesos
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        if "class_names" in checkpoint:
            CLASSES = checkpoint["class_names"]
            # Ajustar la capa final si el numero de clases difiere
            if len(CLASSES) != model.fc.out_features:
                model.fc = torch.nn.Linear(num_ftrs, len(CLASSES))
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, device


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Aplica las transformaciones necesarias a una imagen."""
    image = image.convert("RGB")
    tensor = _preprocess(image).unsqueeze(0)
    return tensor


def predict(
    model: torch.nn.Module, device: torch.device, tensor: torch.Tensor
) -> Tuple[str, float, dict[str, float]]:
    """Realiza la predicción de una imagen.

    Devuelve la clase predicha, la confianza de esa clase y un diccionario con
    las probabilidades de todas las clases.
    """
    with torch.no_grad():
        tensor = tensor.to(device)
        outputs = model(tensor)
        probs_tensor = F.softmax(outputs, dim=1).squeeze(0).cpu()

        predicted_idx = int(torch.argmax(probs_tensor))
        predicted_class = CLASSES[predicted_idx]
        confidence = float(probs_tensor[predicted_idx])
        prob_dict = {
            cls: float(probs_tensor[i]) for i, cls in enumerate(CLASSES)
        }

    return predicted_class, confidence, prob_dict
