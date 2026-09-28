# Brain Tumor MRI Transfer Learning Benchmark

A complete, beginner-friendly PyTorch transfer learning benchmark on a local Windows laptop. This project fine-tunes four pretrained torchvision convolutional neural network (CNN) architectures on a 4-class Brain Tumor MRI classification dataset for **exactly 5 epochs per model** and compares them systematically across classification performance, computational efficiency, and model size.

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [What This Experiment Measures](#2-what-this-experiment-measures)
3. [Hardware Requirements](#3-hardware-requirements)
4. [Software Requirements](#4-software-requirements)
5. [Installation on Windows](#5-installation-on-windows)
6. [VS Code Setup](#6-vs-code-setup)
7. [Dataset Download](#7-dataset-download)
8. [Dataset Folder Placement](#8-dataset-folder-placement)
9. [Dataset Preparation](#9-dataset-preparation)
10. [Verify Dataset](#10-verify-dataset)
11. [Run One Model](#11-run-one-model)
12. [Run Full Benchmark](#12-run-full-benchmark)
13. [Understand the Results](#13-understand-the-results)
14. [Where Output Files Are Saved](#14-where-output-files-are-saved)
15. [Troubleshooting](#15-troubleshooting)
16. [How to Reduce RAM/VRAM Usage](#16-how-to-reduce-ramvram-usage)
17. [CPU-only Mode](#17-cpu-only-mode)
18. [GPU/CUDA Check](#18-gpucuda-check)
19. [Research/Reporting Notes](#19-researchreporting-notes)
20. [Limitations](#20-limitations)

---

## 1. Project Overview

This repository benchmarks four distinct torchvision CNN architectures using pretrained ImageNet weights:
1. **AlexNet** — classic deep CNN baseline
2. **ResNet-18** — standard modern residual CNN baseline
3. **MobileNetV3-Small** — lightweight, mobile-optimized architecture
4. **EfficientNet-B0** — modern compound-scaled efficient architecture

All models are fine-tuned under strictly identical conditions:
- **Same dataset split** (85% train, 15% validation from training set; 100% untouched test set)
- **Same resolution** (224 × 224 pixels)
- **Same optimizer & hyperparameters** (AdamW, learning rate $1 \times 10^{-4}$, 5 epochs)
- **Same evaluation protocol** (best checkpoint chosen strictly by validation accuracy, evaluated once on untouched test set)

---

## 2. What This Experiment Measures

For each model, the benchmark measures and records:
- **Accuracy**: Overall fraction of correct test classifications.
- **Macro F1-Score**: Unweighted mean of F1-scores across all 4 classes.
- **Macro Sensitivity (Recall)**: $\frac{\text{TP}}{\text{TP} + \text{FN}}$ averaged across all classes.
- **Macro Specificity**: $\frac{\text{TN}}{\text{TN} + \text{FP}}$ averaged across all classes.
- **Inference Latency (ms/image)**: Milliseconds required per single image forward pass (measured with warm-up batches and GPU synchronization).
- **Model Checkpoint Size (MB)**: Exact disk size of the saved PyTorch `state_dict` file.
- **Per-Class Metrics & Confusion Matrices**: Individual 300 DPI heatmaps and a consolidated 2×2 grid.

---

## 3. Hardware Requirements

- **Processor**: Intel Core i5/i7/i9 or AMD Ryzen 5/7/9
- **RAM**: Minimum 8 GB (16 GB recommended)
- **GPU (Optional)**: NVIDIA GPU with 4+ GB VRAM (CUDA accelerated)
- **CPU Mode**: Fully supported if no dedicated GPU is available.

---

## 4. Software Requirements

- **Operating System**: Windows 10 / Windows 11 (64-bit)
- **Python**: Python 3.10, 3.11, 3.12, or 3.13
- **IDE**: Visual Studio Code (VS Code) recommended
- **Key Libraries**: PyTorch, TorchVision, Scikit-Learn, Pandas, NumPy, Matplotlib, Seaborn, Pillow, Tqdm

---

## 5. Installation on Windows

Open **PowerShell** in the project directory (`C:\Users\<user>\...\dl_project`):

```powershell
# 1. Check Python version
python --version

# 2. Create virtual environment
python -m venv .venv

# 3. Activate virtual environment
# (If execution policy error occurs, run: Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass)
.\.venv\Scripts\Activate.ps1

# 4. Upgrade pip and install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> **Note for NVIDIA GPU users**: If you have an NVIDIA GPU and want CUDA acceleration, install the CUDA build of PyTorch from [pytorch.org](https://pytorch.org/get-started/locally/), for example:
> ```powershell
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> ```

---

## 6. VS Code Setup

1. Open VS Code: `File` → `Open Folder...` → Select the project folder.
2. Open the Command Palette (`Ctrl + Shift + P`) and type: `Python: Select Interpreter`.
3. Choose the virtual environment: `.\.venv\Scripts\python.exe`.
4. Open a new terminal inside VS Code: `Terminal` → `New Terminal` (`Ctrl + Shift + \``).

---

## 7. Dataset Download

Download the **Brain Tumor Classification (MRI)** dataset from Kaggle:
- Link: [Kaggle Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) or [Brain Tumor Classification (MRI)](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri)
- The dataset contains 4 categories: `glioma`, `meningioma`, `notumor` (or `no_tumor`), `pituitary`.

---

## 8. Dataset Folder Placement

Extract the downloaded dataset into `data/raw/` or `dataset/` inside the project root:

```text
dl_project/
├── dataset/ (or data/raw/)
│   ├── Training/
│   │   ├── glioma/
│   │   ├── meningioma/
│   │   ├── notumor/
│   │   └── pituitary/
│   └── Testing/
│       ├── glioma/
│       ├── meningioma/
│       ├── notumor/
│       └── pituitary/
```

---

## 9. Dataset Preparation

Run the deterministic preparation script to create 85% train / 15% val split from Training and preserve the 100% untouched Testing set:

```powershell
python prepare_dataset.py
```

This creates structured files in `data/processed/`:
- `data/processed/train/`
- `data/processed/val/`
- `data/processed/test/`

---

## 10. Verify Dataset

Inspect dataset splits, counts, and check every image file for corruption:

```powershell
python verify_dataset.py
```

Expected output:
```text
============================================================
DATASET VERIFICATION & INTEGRITY REPORT
============================================================
Detected Classes (4): glioma, meningioma, notumor, pituitary

Class                | Train      | Val        | Test       | Total     
-----------------------------------------------------------------
glioma               | 1190       | 210        | 400        | 1800      
meningioma           | 1190       | 210        | 400        | 1800      
notumor              | 1190       | 210        | 400        | 1800      
pituitary            | 1190       | 210        | 400        | 1800      
-----------------------------------------------------------------
TOTAL                | 4760       | 840        | 1600       | 7200      

Image Integrity Check: PASSED (No corrupted images detected).
============================================================
STATUS: Dataset is VERIFIED and ready for benchmarking!
============================================================
```

---

## 11. Run One Model

Test that everything works end-to-end with a single model before running all four:

```powershell
python run_benchmark.py --model ResNet18
```

Supported single model options: `AlexNet`, `ResNet18`, `MobileNetV3Small`, `EfficientNetB0`.

---

## 12. Run Full Benchmark

To train and benchmark all four models sequentially:

```powershell
python run_benchmark.py --model all
```

Or simply:

```powershell
python run_benchmark.py
```

---

## 13. Understand the Results

After execution, a formatted summary table is printed to your console:

```text
================================================================================
TRANSFER LEARNING BENCHMARK
================================================================================
Model                Accuracy   Macro F1   Sensitivity   Specificity   Latency(ms)   Size(MB)
------------------------------------------------------------------------------------------------
AlexNet                xx.xx%     xx.xx%        xx.xx%        xx.xx%         xx.xx      xx.xx
ResNet-18              xx.xx%     xx.xx%        xx.xx%        xx.xx%         xx.xx      xx.xx
MobileNetV3-Small      xx.xx%     xx.xx%        xx.xx%        xx.xx%         xx.xx      xx.xx
EfficientNet-B0        xx.xx%     xx.xx%        xx.xx%        xx.xx%         xx.xx      xx.xx
------------------------------------------------------------------------------------------------
```

- **Accuracy**: Higher is better.
- **Macro F1**: Harmonic mean of precision and recall averaged across all classes; robust to class balance.
- **Sensitivity (Recall)**: True positive detection rate.
- **Specificity**: True negative detection rate.
- **Latency (ms)**: Lower is faster for real-time inference.
- **Size (MB)**: Smaller is better for deployment to edge or mobile devices.

---

## 14. Where Output Files Are Saved

```text
results/
├── benchmark_results.csv           # Full tabular metrics
├── benchmark_results.json          # Machine-readable detailed results
├── run_config.json                 # Exact run hyperparameters for reproducibility
│
├── confusion_matrices/             # 300 DPI individual confusion matrices
│   ├── AlexNet.png
│   ├── ResNet18.png
│   ├── MobileNetV3Small.png
│   └── EfficientNetB0.png
│
├── plots/                          # Publication-quality comparison figures
│   ├── confusion_matrix_grid.png   # 2×2 Side-by-Side Confusion Matrix Grid
│   ├── performance_comparison.png  # Accuracy & Macro F1 Bar Chart
│   ├── latency_comparison.png      # Milliseconds / Image Comparison
│   ├── model_size_comparison.png   # Model Size (MB) Comparison
│   ├── sensitivity_specificity_comparison.png
│   ├── AlexNet_training_curves.png
│   ├── ResNet18_training_curves.png
│   ├── MobileNetV3Small_training_curves.png
│   └── EfficientNetB0_training_curves.png
│
└── logs/                           # Per-epoch training & validation history JSONs
```

---

## 15. Troubleshooting

- **PowerShell Script Execution Blocked**:
  ```powershell
  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
  ```
- **ModuleNotFoundError**: Make sure your `.venv` is activated before running scripts. Run `pip install -r requirements.txt`.
- **Dataset not found**: Ensure the dataset is extracted into `dataset/` or `data/raw/` with `Training` and `Testing` folders.
- **Windows multiprocessing error**: `config.py` sets `NUM_WORKERS = 0` by default, which is completely safe for Windows.

---

## 16. How to Reduce RAM/VRAM Usage

If you experience memory pressure or CUDA Out-Of-Memory errors:
1. Open [config.py](file:///c:/Users/amrut/OneDrive/Desktop/dl_project/config.py).
2. Lower `BATCH_SIZE` from `16` to `8` or `4`:
   ```python
   BATCH_SIZE = 8
   ```
3. Keep `NUM_WORKERS = 0` to avoid RAM spikes on Windows.

---

## 17. CPU-only Mode

If no NVIDIA GPU is available, the codebase automatically detects and falls back to CPU execution without crashing. Training will proceed safely on the CPU.

---

## 18. GPU/CUDA Check

To verify hardware detection at any time, run:

```powershell
python check_environment.py
```

---

## 19. Research/Reporting Notes

- This repository uses strictly measured empirical values obtained on the local setup.
- Pretrained ImageNet weights are utilized with fine-tuning on the full backbone for exactly 5 epochs using AdamW.
- Model selection is determined solely by validation set accuracy, and the test set is evaluated only once at the conclusion of training.

---

## 20. Limitations

- **Educational & Benchmark Purpose Only**: This project is designed as an empirical engineering benchmark and educational baseline. It is not intended or validated for clinical medical diagnosis.
- **Dataset Biases**: Brain MRI datasets may contain patient-level correlation, acquisition artifact variations, or scanner bias.
- **Training Budget**: Benchmarking is conducted with a fixed 5-epoch protocol for comparative efficiency; extended training schedules or clinical hyperparameters may yield different convergence profiles.
