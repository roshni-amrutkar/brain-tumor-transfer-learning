"""
Model Factory for Transfer Learning CNN Architectures.
Supports: AlexNet, ResNet18, MobileNetV3Small, and EfficientNetB0
using modern torchvision weights APIs.
"""

from pathlib import Path
from typing import Dict, Any
import os
import torch
import torch.nn as nn
import torchvision.models as models

AVAILABLE_MODELS = [
    "AlexNet",
    "ResNet18",
    "MobileNetV3Small",
    "EfficientNetB0",
]


def create_alexnet(num_classes: int) -> nn.Module:
    """Load pretrained AlexNet and replace final classifier layer."""
    weights = models.AlexNet_Weights.DEFAULT
    model = models.alexnet(weights=weights)
    in_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(in_features, num_classes)
    return model


def create_resnet18(num_classes: int) -> nn.Module:
    """Load pretrained ResNet-18 and replace final fully connected layer."""
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def create_mobilenet_v3_small(num_classes: int) -> nn.Module:
    """Load pretrained MobileNetV3-Small and replace final classifier layer."""
    weights = models.MobileNet_V3_Small_Weights.DEFAULT
    model = models.mobilenet_v3_small(weights=weights)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model


def create_efficientnet_b0(num_classes: int) -> nn.Module:
    """Load pretrained EfficientNet-B0 and replace final classifier layer."""
    weights = models.EfficientNet_B0_Weights.DEFAULT
    model = models.efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


MODEL_BUILDERS = {
    "AlexNet": create_alexnet,
    "ResNet18": create_resnet18,
    "MobileNetV3Small": create_mobilenet_v3_small,
    "EfficientNetB0": create_efficientnet_b0,
}


def get_model(model_name: str, num_classes: int) -> nn.Module:
    """
    Factory method to instantiate a CNN architecture configured for num_classes.

    Args:
        model_name: One of 'AlexNet', 'ResNet18', 'MobileNetV3Small', 'EfficientNetB0'
        num_classes: Number of output classification categories

    Returns:
        torch.nn.Module with adapted classification head
    """
    # Normalize model name for robust case-insensitive lookup
    lookup = {k.lower(): k for k in MODEL_BUILDERS.keys()}
    key = lookup.get(model_name.lower())

    if not key:
        raise ValueError(
            f"Unknown model name '{model_name}'. Supported models are: {list(MODEL_BUILDERS.keys())}"
        )

    model = MODEL_BUILDERS[key](num_classes=num_classes)
    return model


def get_model_size_mb(checkpoint_path: Path) -> float:
    """
    Calculate the file size in megabytes (MB) of the saved model weights checkpoint.

    Args:
        checkpoint_path: Path to the .pth checkpoint file

    Returns:
        Size in MB rounded to 2 decimal places
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    size_bytes = os.path.getsize(checkpoint_path)
    size_mb = size_bytes / (1024 ** 2)
    return round(size_mb, 2)
