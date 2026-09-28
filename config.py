"""
Central Configuration Module for Brain Tumor MRI Transfer Learning Benchmark.
Provides hardware-aware defaults, dataset paths, training hyperparameters,
and model settings for Windows local development.
"""

from pathlib import Path
import torch

# ==============================================================================
# Hardware & Environment Configuration
# ==============================================================================
RANDOM_SEED: int = 42
# Conservative default for Windows laptops. Adjust to 32 if VRAM allows, or 8/4 if OOM occurs.
BATCH_SIZE: int = 16
NUM_WORKERS: int = 0  # Safe default on Windows to prevent multiprocessing issues
IMAGE_SIZE: int = 224
EPOCHS: int = 5
LEARNING_RATE: float = 1e-4
WEIGHT_DECAY: float = 1e-2
VAL_SPLIT: float = 0.15

# ImageNet normalization statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Supported image file extensions
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

# ==============================================================================
# Path Configuration
# ==============================================================================
BASE_DIR: Path = Path(__file__).resolve().parent
DATA_DIR: Path = BASE_DIR / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"

TRAIN_DATA_DIR: Path = PROCESSED_DATA_DIR / "train"
VAL_DATA_DIR: Path = PROCESSED_DATA_DIR / "val"
TEST_DATA_DIR: Path = PROCESSED_DATA_DIR / "test"

# Fallback paths to search for raw dataset if not in data/raw
ALTERNATIVE_RAW_DIRS = [
    BASE_DIR / "dataset",
    BASE_DIR / "data" / "raw",
    BASE_DIR / "data",
]

CHECKPOINTS_DIR: Path = BASE_DIR / "checkpoints"
RESULTS_DIR: Path = BASE_DIR / "results"
LOGS_DIR: Path = RESULTS_DIR / "logs"
PLOTS_DIR: Path = RESULTS_DIR / "plots"
CONFUSION_DIR: Path = RESULTS_DIR / "confusion_matrices"

BENCHMARK_CSV: Path = RESULTS_DIR / "benchmark_results.csv"
BENCHMARK_JSON: Path = RESULTS_DIR / "benchmark_results.json"
RUN_CONFIG_JSON: Path = RESULTS_DIR / "run_config.json"
GRID_PLOT_PATH: Path = PLOTS_DIR / "confusion_matrix_grid.png"

# ==============================================================================
# Benchmark Models Specification
# ==============================================================================
MODELS_TO_BENCHMARK = [
    "AlexNet",
    "ResNet18",
    "MobileNetV3Small",
    "EfficientNetB0"
]

DEFAULT_MODEL_CHOICE: str = "all"

# ==============================================================================
# Directory Setup & Helper Functions
# ==============================================================================
def ensure_directories() -> None:
    """Create all required output and cache directories if they do not exist."""
    directories = [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        TRAIN_DATA_DIR,
        VAL_DATA_DIR,
        TEST_DATA_DIR,
        CHECKPOINTS_DIR,
        RESULTS_DIR,
        LOGS_DIR,
        PLOTS_DIR,
        CONFUSION_DIR,
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def get_device() -> torch.device:
    """
    Detect available computing hardware.
    Returns CUDA device if available, otherwise CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def print_device_info() -> torch.device:
    """
    Print formatted hardware and acceleration information to console.
    """
    cuda_available = torch.cuda.is_available()
    device = get_device()
    print("=" * 60)
    print(f"Device: {device.type}")
    if cuda_available:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    print(f"CUDA available: {cuda_available}")
    if not cuda_available:
        print("Note: Running in CPU mode. Training may take longer.")
    print("=" * 60)
    return device


if __name__ == "__main__":
    ensure_directories()
    print_device_info()
    print("Configuration directories verified successfully.")
