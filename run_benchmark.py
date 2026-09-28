"""
Master Benchmark Orchestration Script.
Executes the transfer learning benchmark across the 4 CNN architectures:
1. AlexNet
2. ResNet-18
3. MobileNetV3-Small
4. EfficientNet-B0

Trains each model for exactly 5 epochs, evaluates on test set,
generates individual and comparative plots, creates 2x2 confusion grid,
saves structured CSV/JSON results, and prints the summary benchmark table.
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import config
from create_confusion_grid import generate_confusion_grid
from evaluate_models import evaluate_model
from train_model import train_model
import utils

MODEL_DISPLAY_NAMES = {
    "AlexNet": "AlexNet",
    "ResNet18": "ResNet-18",
    "MobileNetV3Small": "MobileNetV3-Small",
    "EfficientNetB0": "EfficientNet-B0",
}


def plot_training_curves(history: Dict[str, List[float]], model_name: str, output_path: Path) -> None:
    """
    Plot and save training vs validation loss and accuracy curves for 5 epochs.
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(12, 5), dpi=300)

    # Loss subplot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], "o-", label="Train Loss", color="#1f77b4", linewidth=2)
    plt.plot(epochs, history["val_loss"], "s--", label="Val Loss", color="#ff7f0e", linewidth=2)
    plt.title(f"{MODEL_DISPLAY_NAMES.get(model_name, model_name)} — Loss", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Cross-Entropy Loss", fontsize=11)
    plt.xticks(list(epochs))
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=10)

    # Accuracy subplot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], "o-", label="Train Accuracy", color="#2ca02c", linewidth=2)
    plt.plot(epochs, history["val_acc"], "s--", label="Val Accuracy", color="#d62728", linewidth=2)
    plt.title(f"{MODEL_DISPLAY_NAMES.get(model_name, model_name)} — Accuracy", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Accuracy (%)", fontsize=11)
    plt.xticks(list(epochs))
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=10)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_comparative_charts(df_results: pd.DataFrame, output_dir: Path) -> None:
    """
    Generate the 4 comparative benchmark charts:
    1. performance_comparison.png (Accuracy and Macro F1)
    2. latency_comparison.png (Latency ms/image)
    3. model_size_comparison.png (Size in MB)
    4. sensitivity_specificity_comparison.png (Macro Sensitivity and Specificity)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    models = df_results["Model"].tolist()
    x = np.arange(len(models))

    # 1. Performance (Accuracy & Macro F1)
    plt.figure(figsize=(8, 5), dpi=300)
    width = 0.35
    plt.bar(x - width/2, df_results["Accuracy_Pct"], width, label="Test Accuracy (%)", color="#2b5c8f")
    plt.bar(x + width/2, df_results["Macro_F1_Pct"], width, label="Macro F1-Score (%)", color="#e26d5c")
    plt.xlabel("Model Architecture", fontsize=11, fontweight="bold")
    plt.ylabel("Score (%)", fontsize=11, fontweight="bold")
    plt.title("Performance Comparison: Accuracy & Macro F1", fontsize=13, fontweight="bold")
    plt.xticks(x, models, fontsize=10)
    plt.ylim(0, 105)
    plt.grid(axis="y", linestyle=":", alpha=0.7)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(output_dir / "performance_comparison.png", dpi=300)
    plt.close()

    # 2. Latency Comparison
    plt.figure(figsize=(7, 5), dpi=300)
    bars = plt.bar(models, df_results["Latency_ms_per_image"], color="#4ea8de", edgecolor="#0077b6", width=0.5)
    plt.xlabel("Model Architecture", fontsize=11, fontweight="bold")
    plt.ylabel("Latency (ms / image)", fontsize=11, fontweight="bold")
    plt.title("Inference Latency Comparison", fontsize=13, fontweight="bold")
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.1, f"{yval:.2f} ms", ha="center", va="bottom", fontsize=10, fontweight="bold")
    plt.grid(axis="y", linestyle=":", alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_dir / "latency_comparison.png", dpi=300)
    plt.close()

    # 3. Model Size Comparison
    plt.figure(figsize=(7, 5), dpi=300)
    bars = plt.bar(models, df_results["Model_Size_MB"], color="#f4a261", edgecolor="#e76f51", width=0.5)
    plt.xlabel("Model Architecture", fontsize=11, fontweight="bold")
    plt.ylabel("Checkpoint Size (MB)", fontsize=11, fontweight="bold")
    plt.title("Model Checkpoint Size Comparison", fontsize=13, fontweight="bold")
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, f"{yval:.2f} MB", ha="center", va="bottom", fontsize=10, fontweight="bold")
    plt.grid(axis="y", linestyle=":", alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_dir / "model_size_comparison.png", dpi=300)
    plt.close()

    # 4. Sensitivity & Specificity Comparison
    plt.figure(figsize=(8, 5), dpi=300)
    plt.bar(x - width/2, df_results["Macro_Sensitivity_Pct"], width, label="Macro Sensitivity (%)", color="#52b788")
    plt.bar(x + width/2, df_results["Macro_Specificity_Pct"], width, label="Macro Specificity (%)", color="#7400b8")
    plt.xlabel("Model Architecture", fontsize=11, fontweight="bold")
    plt.ylabel("Score (%)", fontsize=11, fontweight="bold")
    plt.title("Macro Sensitivity vs Specificity Comparison", fontsize=13, fontweight="bold")
    plt.xticks(x, models, fontsize=10)
    plt.ylim(0, 105)
    plt.grid(axis="y", linestyle=":", alpha=0.7)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(output_dir / "sensitivity_specificity_comparison.png", dpi=300)
    plt.close()


def print_benchmark_table(records: List[Dict[str, Any]]) -> None:
    """Print the exact formatted summary table to console."""
    print("\n" + "=" * 80)
    print("TRANSFER LEARNING BENCHMARK")
    print("=" * 80)
    print(f"{'Model':<20} {'Accuracy':<10} {'Macro F1':<10} {'Sensitivity':<13} {'Specificity':<13} {'Latency(ms)':<13} {'Size(MB)':<10}")
    print("-" * 92)
    for r in records:
        print(
            f"{r['Model']:<20} "
            f"{r['Accuracy_Pct']:>7.2f}%   "
            f"{r['Macro_F1_Pct']:>7.2f}%   "
            f"{r['Macro_Sensitivity_Pct']:>9.2f}%   "
            f"{r['Macro_Specificity_Pct']:>9.2f}%   "
            f"{r['Latency_ms_per_image']:>9.2f}    "
            f"{r['Model_Size_MB']:>7.2f}"
        )
    print("-" * 92 + "\n")


def run_benchmark(
    models_to_run: List[str],
    epochs: int = config.EPOCHS,
    batch_size: int = config.BATCH_SIZE,
    learning_rate: float = config.LEARNING_RATE,
) -> None:
    """Run end-to-end benchmark workflow."""
    config.ensure_directories()
    utils.set_seed(config.RANDOM_SEED)
    device = config.print_device_info()

    benchmark_records: List[Dict[str, Any]] = []

    for model_name in models_to_run:
        display_name = MODEL_DISPLAY_NAMES.get(model_name, model_name)
        print(f"\n>>> Running Benchmark for: {display_name} <<<")

        # 1. Train model
        train_res = train_model(
            model_name=model_name,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            device=device
        )

        # 2. Save training curves
        curve_path = config.PLOTS_DIR / f"{model_name}_training_curves.png"
        plot_training_curves(train_res["history"], model_name, curve_path)

        # 3. Evaluate on test set
        eval_metrics = evaluate_model(
            model_name=model_name,
            checkpoint_path=train_res["checkpoint_path"],
            device=device
        )

        # 4. Record results
        record = {
            "Model": display_name,
            "Internal_Key": model_name,
            "Accuracy": eval_metrics["accuracy"],
            "Accuracy_Pct": eval_metrics["accuracy"] * 100,
            "Macro_F1": eval_metrics["macro_f1"],
            "Macro_F1_Pct": eval_metrics["macro_f1"] * 100,
            "Macro_Sensitivity": eval_metrics["macro_sensitivity"],
            "Macro_Sensitivity_Pct": eval_metrics["macro_sensitivity"] * 100,
            "Macro_Specificity": eval_metrics["macro_specificity"],
            "Macro_Specificity_Pct": eval_metrics["macro_specificity"] * 100,
            "Latency_ms_per_image": eval_metrics["latency_ms_per_image"],
            "Model_Size_MB": eval_metrics["model_size_mb"],
            "Best_Validation_Accuracy": train_res["best_val_acc"],
            "Training_Time_seconds": round(train_res["total_training_time"], 2),
            "Device": device.type,
            "Epochs": epochs,
            "Batch_Size": batch_size,
            "Image_Size": config.IMAGE_SIZE,
        }
        benchmark_records.append(record)

    df_results = pd.DataFrame(benchmark_records)

    # Save CSV and JSON
    df_results.to_csv(config.BENCHMARK_CSV, index=False)
    utils.save_json(benchmark_records, config.BENCHMARK_JSON)

    # Save machine-readable reproducibility run config
    utils.save_json({
        "seed": config.RANDOM_SEED,
        "image_size": config.IMAGE_SIZE,
        "batch_size": batch_size,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "optimizer": "AdamW",
        "weight_decay": config.WEIGHT_DECAY,
        "models": models_to_run,
        "val_split": config.VAL_SPLIT,
        "device": device.type,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }, config.RUN_CONFIG_JSON)

    # Generate comparative plots and confusion matrix grid
    if len(benchmark_records) > 0:
        plot_comparative_charts(df_results, config.PLOTS_DIR)
        generate_confusion_grid(config.GRID_PLOT_PATH)

    # Print summary console table
    print_benchmark_table(benchmark_records)

    # Print required final console message
    print("=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)
    print("\nModels evaluated:")
    for m in models_to_run:
        print(f"- {MODEL_DISPLAY_NAMES.get(m, m)}")
    print(f"\nEpochs per model: {epochs}")
    print("\nResults saved to:")
    print(f"{config.BENCHMARK_CSV}")
    print(f"{config.BENCHMARK_JSON}")
    print("\nConfusion matrix grid:")
    print(f"{config.GRID_PLOT_PATH}")
    print("\nYou can now use these measured results in your survey/report.")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Run Brain Tumor MRI Transfer Learning Benchmark.")
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["AlexNet", "ResNet18", "MobileNetV3Small", "EfficientNetB0", "all"],
        help="Specific model to benchmark or 'all' (default: all)"
    )
    parser.add_argument("--epochs", type=int, default=config.EPOCHS, help="Number of training epochs (default: 5)")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE, help="Batch size (default: 16)")
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE, help="Learning rate (default: 1e-4)")

    args = parser.parse_args()

    if args.model.lower() == "all":
        models_to_run = config.MODELS_TO_BENCHMARK
    else:
        models_to_run = [m for m in config.MODELS_TO_BENCHMARK if m.lower() == args.model.lower()]

    run_benchmark(
        models_to_run=models_to_run,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )


if __name__ == "__main__":
    main()
