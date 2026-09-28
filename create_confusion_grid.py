"""
2x2 Confusion Matrix Grid Generator.
Creates a publication-quality side-by-side comparison figure:
Top Row    : AlexNet | ResNet-18
Bottom Row : MobileNetV3-Small | EfficientNet-B0
Saved to: results/plots/confusion_matrix_grid.png (300 DPI)
"""

from pathlib import Path
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import config
import utils

GRID_LAYOUT = [
    [("AlexNet", "AlexNet"), ("ResNet18", "ResNet-18")],
    [("MobileNetV3Small", "MobileNetV3-Small"), ("EfficientNetB0", "EfficientNet-B0")]
]


def load_model_evaluation(model_name: str) -> Optional[Dict]:
    """Load evaluation log JSON for a model."""
    eval_path = config.LOGS_DIR / f"{model_name}_eval.json"
    if eval_path.exists():
        return utils.load_json(eval_path)
    return None


def generate_confusion_grid(output_path: Path = config.GRID_PLOT_PATH) -> bool:
    """
    Generate and save a 2x2 grid of confusion matrices for the 4 benchmarked models.
    """
    config.ensure_directories()
    fig, axes = plt.subplots(2, 2, figsize=(13, 11), dpi=300)

    all_present = True

    for row_idx, row in enumerate(GRID_LAYOUT):
        for col_idx, (model_key, display_name) in enumerate(row):
            ax = axes[row_idx, col_idx]
            eval_data = load_model_evaluation(model_key)

            if eval_data and "metrics" in eval_data:
                metrics = eval_data["metrics"]
                cm = np.array(metrics["confusion_matrix"])
                class_names = metrics.get("class_names", ["glioma", "meningioma", "notumor", "pituitary"])
                acc = metrics.get("accuracy", 0.0) * 100
                f1 = metrics.get("macro_f1", 0.0) * 100

                sns.heatmap(
                    cm,
                    annot=True,
                    fmt="d",
                    cmap="Blues",
                    xticklabels=class_names,
                    yticklabels=class_names,
                    cbar=False,
                    ax=ax,
                    annot_kws={"size": 11, "weight": "bold"}
                )
                ax.set_title(
                    f"{display_name}\nAcc: {acc:.2f}% | Macro F1: {f1:.2f}%",
                    fontsize=12,
                    fontweight="bold",
                    pad=10
                )
                ax.set_xlabel("Predicted Label", fontsize=10, labelpad=6)
                ax.set_ylabel("True Label", fontsize=10, labelpad=6)
            else:
                all_present = False
                ax.text(
                    0.5, 0.5,
                    f"{display_name}\n[Evaluation Not Found]\nRun benchmark to generate",
                    ha="center", va="center", fontsize=11, color="gray"
                )
                ax.set_title(display_name, fontsize=12, fontweight="bold")
                ax.set_xticks([])
                ax.set_yticks([])

    plt.suptitle("Brain Tumor MRI Classification — Confusion Matrix Comparison (2x2 Grid)", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"2x2 Confusion Matrix Grid saved to: {output_path.resolve()}")
    return all_present


if __name__ == "__main__":
    generate_confusion_grid()
