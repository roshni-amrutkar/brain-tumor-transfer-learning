"""
Model Evaluation and Metrics Module.
Evaluates model on the untouched test set to measure:
- Accuracy
- Macro F1-score
- Macro & Per-class Sensitivity
- Macro & Per-class Specificity
- Inference Latency (ms/image)
- Checkpoint Size (MB)
Generates high-resolution confusion matrix plots (300 DPI).
"""

import argparse
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm

import config
from models import get_model, get_model_size_mb
import utils


def measure_inference_latency(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    warmup_batches: int = 3,
    max_eval_images: int = 400
) -> float:
    """
    Measure pure forward-pass inference latency in milliseconds per image.
    Uses GPU synchronization if CUDA is active.
    """
    model.eval()
    total_time = 0.0
    total_samples = 0

    # 1. Warm-up forward passes
    with torch.no_grad():
        for i, (images, _) in enumerate(dataloader):
            if i >= warmup_batches:
                break
            images = images.to(device)
            _ = model(images)
            if device.type == "cuda":
                torch.cuda.synchronize()

    # 2. Timed inference forward passes
    with torch.no_grad():
        for images, _ in dataloader:
            if total_samples >= max_eval_images:
                break

            images = images.to(device)
            batch_sz = images.size(0)

            if device.type == "cuda":
                torch.cuda.synchronize()

            t0 = time.perf_counter()
            _ = model(images)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            total_time += (t1 - t0)
            total_samples += batch_sz

    latency_ms = (total_time / total_samples) * 1000.0 if total_samples > 0 else 0.0
    return round(latency_ms, 2)


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    model_name: str,
    output_path: Path
) -> None:
    """
    Generate and save a 300 DPI labeled confusion matrix plot.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 5), dpi=300)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=False,
        annot_kws={"size": 11, "weight": "bold"}
    )
    plt.title(f"Confusion Matrix — {model_name}", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Predicted Class", fontsize=11, labelpad=8)
    plt.ylabel("Actual Class", fontsize=11, labelpad=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def evaluate_model(
    model_name: str,
    checkpoint_path: Path = None,
    device: torch.device = None
) -> Dict[str, Any]:
    """
    Evaluate a trained model checkpoint on the test dataset.
    """
    config.ensure_directories()
    if device is None:
        device = config.get_device()

    if checkpoint_path is None:
        checkpoint_path = config.CHECKPOINTS_DIR / f"{model_name}_best.pth"

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint for '{model_name}' not found at {checkpoint_path}. "
            f"Please train the model first with 'python train_model.py --model {model_name}'."
        )

    # Prepare test data loader
    _, eval_transform = utils.get_transforms(config.IMAGE_SIZE)
    test_dataset = ImageFolder(root=str(config.TEST_DATA_DIR), transform=eval_transform)
    class_names = test_dataset.classes
    num_classes = len(class_names)

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=(torch.cuda.is_available())
    )

    # Load model architecture & weights
    model = get_model(model_name=model_name, num_classes=num_classes)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    # Collect predictions
    y_true: List[int] = []
    y_pred: List[int] = []

    print(f"\nEvaluating {model_name} on untouched test set ({len(test_dataset)} images)...")
    with torch.no_grad():
        for images, targets in tqdm(test_loader, desc="Testing", leave=False):
            images = images.to(device)
            outputs = model(images)
            _, preds = outputs.max(1)

            y_true.extend(targets.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())

    # Compute classification metrics
    metrics = utils.calculate_metrics(y_true=y_true, y_pred=y_pred, class_names=class_names)

    # Measure latency & model size
    print("Measuring inference latency...")
    latency_ms = measure_inference_latency(model, test_loader, device=device)
    model_size_mb = get_model_size_mb(checkpoint_path)

    metrics["model_name"] = model_name
    metrics["latency_ms_per_image"] = latency_ms
    metrics["model_size_mb"] = model_size_mb

    # Generate individual confusion matrix plot
    cm_arr = np.array(metrics["confusion_matrix"])
    cm_plot_path = config.CONFUSION_DIR / f"{model_name}.png"
    plot_confusion_matrix(cm_arr, class_names, model_name, cm_plot_path)

    # Save evaluation log
    eval_log_path = config.LOGS_DIR / f"{model_name}_eval.json"
    utils.save_json({
        "metrics": metrics,
        "y_true": y_true,
        "y_pred": y_pred,
    }, eval_log_path)

    print(f"\nEvaluation Results for {model_name}:")
    print(f"  - Accuracy         : {metrics['accuracy'] * 100:.2f}%")
    print(f"  - Macro F1-Score   : {metrics['macro_f1'] * 100:.2f}%")
    print(f"  - Macro Sensitivity: {metrics['macro_sensitivity'] * 100:.2f}%")
    print(f"  - Macro Specificity: {metrics['macro_specificity'] * 100:.2f}%")
    print(f"  - Inference Latency: {latency_ms} ms/image")
    print(f"  - Checkpoint Size  : {model_size_mb} MB")
    print(f"  - Confusion Matrix : {cm_plot_path}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained model checkpoint on the test set.")
    parser.add_argument(
        "--model",
        type=str,
        default="ResNet18",
        choices=["AlexNet", "ResNet18", "MobileNetV3Small", "EfficientNetB0"],
        help="Model architecture to evaluate"
    )
    args = parser.parse_args()
    evaluate_model(model_name=args.model)


if __name__ == "__main__":
    main()
