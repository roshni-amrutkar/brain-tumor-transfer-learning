"""
Utility Functions Module for Transfer Learning Benchmark.
Includes reproducibility seeds, data transformations, metric computation,
and JSON logging helpers.
"""

import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import torch
from torchvision import transforms

import config


def set_seed(seed: int = config.RANDOM_SEED) -> None:
    """
    Set random seeds across Python, NumPy, PyTorch, and CUDA for deterministic runs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Random seed set to: {seed}")


def get_transforms(image_size: int = config.IMAGE_SIZE) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Construct torchvision data transformations for training and validation/testing.
    Training uses standard augmentation (flip, rotation), while val/test is deterministic.
    """
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
    ])

    return train_transform, eval_transform


def calculate_metrics(
    y_true: List[int],
    y_pred: List[int],
    class_names: List[str]
) -> Dict[str, Any]:
    """
    Compute comprehensive classification metrics:
    - Accuracy
    - Macro F1-score
    - Per-class Sensitivity (Recall) & Macro Sensitivity
    - Per-class Specificity & Macro Specificity
    - Confusion Matrix
    """
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    num_classes = len(class_names)

    # Core scikit-learn metrics
    accuracy = float(accuracy_score(y_true_arr, y_pred_arr))
    macro_f1 = float(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0))

    # Confusion matrix: rows = actual, columns = predicted
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=list(range(num_classes)))

    per_class_sensitivity: Dict[str, float] = {}
    per_class_specificity: Dict[str, float] = {}
    sensitivities: List[float] = []
    specificities: List[float] = []

    total_samples = np.sum(cm)

    for i, class_name in enumerate(class_names):
        tp = float(cm[i, i])
        fn = float(np.sum(cm[i, :]) - tp)
        fp = float(np.sum(cm[:, i]) - tp)
        tn = float(total_samples - (tp + fn + fp))

        # Sensitivity = TP / (TP + FN)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        # Specificity = TN / (TN + FP)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        per_class_sensitivity[class_name] = round(sensitivity, 4)
        per_class_specificity[class_name] = round(specificity, 4)

        sensitivities.append(sensitivity)
        specificities.append(specificity)

    macro_sensitivity = float(np.mean(sensitivities)) if sensitivities else 0.0
    macro_specificity = float(np.mean(specificities)) if specificities else 0.0

    return {
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "macro_sensitivity": round(macro_sensitivity, 4),
        "macro_specificity": round(macro_specificity, 4),
        "per_class_sensitivity": per_class_sensitivity,
        "per_class_specificity": per_class_specificity,
        "confusion_matrix": cm.tolist(),
        "class_names": class_names,
    }


def save_json(data: Any, file_path: Path) -> None:
    """Save serializable data to formatted JSON."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def load_json(file_path: Path) -> Any:
    """Load data from a JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def setup_logger(name: str, log_file: Path) -> logging.Logger:
    """Configure a clean standard logger outputting to file and console."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Avoid adding duplicate handlers if logger was already created
    if not logger.handlers:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_fmt = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        file_handler.setFormatter(file_fmt)

        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_fmt = logging.Formatter("%(message)s")
        stream_handler.setFormatter(stream_fmt)

        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

    return logger
