# Antigravity Prompt — Local Laptop Transfer Learning Quick Benchmark for Brain Tumor MRI

You are an expert Python/PyTorch ML engineer helping a complete beginner build a **fully working transfer-learning benchmark on a Windows laptop**.

## 1. Main Goal

Build a complete local project that fine-tunes four torchvision pretrained CNN architectures on a public Brain Tumor MRI image classification dataset for **exactly 5 epochs per model** and compares them using:

- Accuracy
- Macro F1-score
- Macro Sensitivity
- Macro Specificity
- Inference latency in milliseconds per image
- Model size in MB on disk
- Confusion matrices

Models to benchmark:

1. AlexNet — older baseline
2. ResNet-18 — standard modern CNN baseline
3. MobileNetV3-Small — lightweight model
4. EfficientNet-B0 — modern efficient model

The final deliverable must contain a **2×2 side-by-side confusion-matrix grid** for the four models and a clean benchmark CSV/table.

The project must run on a **local Windows laptop**, not Google Colab.

---

# 2. Beginner-Friendly Requirements

Assume the user is a complete beginner.

Do NOT assume:

- Python is already installed
- virtual environments are understood
- PyTorch is installed
- CUDA/GPU is available
- the dataset is already prepared
- the user understands project structure
- the user knows how to run Python scripts

Create very clear instructions in the generated `README.md` explaining exactly what to click/type in Windows and VS Code.

Do not overcomplicate the project with web apps, Docker, databases, cloud services, or unnecessary frameworks.

Use plain Python scripts and PyTorch.

---

# 3. Target Environment

Primary environment:

- Windows 10/11
- VS Code
- Python 3.11 recommended
- PyTorch
- torchvision
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- Pillow
- tqdm

The project must work on:

### GPU mode
If an NVIDIA CUDA GPU is available, automatically use it.

### CPU mode
If CUDA is unavailable, automatically fall back to CPU.

At startup print something like:

```text
Device: cuda
GPU: NVIDIA ...
CUDA available: True
```

or:

```text
Device: cpu
CUDA available: False
```

Never crash simply because CUDA is unavailable.

---

# 4. Hardware Awareness

The code must be designed for a normal student laptop.

Recommended assumptions:

- RAM: 16 GB recommended
- RAM minimum: 8 GB
- GPU VRAM: 4 GB or more if available
- CPU-only mode must still work

Do NOT assume a high-end GPU.

Create a configuration section allowing the user to change:

```python
BATCH_SIZE = 32
IMAGE_SIZE = 224
EPOCHS = 5
LEARNING_RATE = 1e-4
NUM_WORKERS = 0
RANDOM_SEED = 42
```

Start with conservative local-laptop settings.

Recommended defaults for Windows beginner setup:

```text
BATCH_SIZE = 16
NUM_WORKERS = 0
IMAGE_SIZE = 224
EPOCHS = 5
LEARNING_RATE = 0.0001
```

Explain in README that the user can increase batch size to 32 if memory allows.

If CUDA out-of-memory occurs, explain to reduce batch size from 16 → 8 → 4.

---

# 5. Dataset

Use a public Brain Tumor MRI image classification dataset such as the Kaggle Brain Tumor Classification (MRI) dataset with approximately 3,000+ images and four classes.

Expected class names are typically:

```text
glioma_tumor
meningioma_tumor
no_tumor
pituitary_tumor
```

Do NOT hardcode one exact absolute dataset path.

Instead, create a configuration variable such as:

```python
DATASET_ROOT = "data/raw"
```

and support the following possible layout:

```text
data/raw/
    Training/
        glioma_tumor/
        meningioma_tumor/
        no_tumor/
        pituitary_tumor/
    Testing/
        glioma_tumor/
        meningioma_tumor/
        no_tumor/
        pituitary_tumor/
```

The program should detect `Training` and `Testing` directories automatically if possible.

If the dataset is not found, print a clear beginner-friendly error telling the user exactly where to place the extracted dataset.

Do not require Kaggle API credentials.

Optionally document Kaggle API download as an advanced alternative, but manual ZIP download must be the primary beginner path.

---

# 6. Dataset Split

The original dataset contains Training and Testing directories.

Preserve the original Testing directory as the final test set.

Split the original Training directory into:

- 85% training
- 15% validation

Use a fixed random seed of 42.

Create a processed structure like:

```text
data/processed/
    train/
        glioma_tumor/
        meningioma_tumor/
        no_tumor/
        pituitary_tumor/

    val/
        glioma_tumor/
        meningioma_tumor/
        no_tumor/
        pituitary_tumor/

    test/
        glioma_tumor/
        meningioma_tumor/
        no_tumor/
        pituitary_tumor/
```

The test set must remain untouched during training and model selection.

Before copying/splitting, ensure the script can be rerun without creating duplicate files.

Use deterministic splitting with `random.seed(42)` or equivalent.

---

# 7. Data Validation

Create a dataset verification script.

It must:

- count images per class
- count total images in train/validation/test
- print class names
- detect empty class folders
- detect unsupported file types
- optionally report corrupted images using Pillow

Supported image extensions:

```text
.jpg
.jpeg
.png
.bmp
```

If an image is corrupted, report it clearly.

Do not silently hide corrupted files.

---

# 8. Project Structure

Create this project structure:

```text
brain_tumor_transfer_learning/
│
├── README.md
├── requirements.txt
├── .gitignore
├── prompt.md
│
├── config.py
├── prepare_dataset.py
├── verify_dataset.py
├── train_model.py
├── run_benchmark.py
├── evaluate_models.py
├── create_confusion_grid.py
├── utils.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│
├── results/
│   ├── benchmark_results.csv
│   ├── benchmark_results.json
│   ├── confusion_matrices/
│   ├── plots/
│   └── logs/
│
└── checkpoints/
```

Keep code modular and easy to understand.

---

# 9. Python Environment Setup

In README.md provide exact Windows instructions.

Use:

```powershell
python --version
```

If Python is installed, create a virtual environment:

```powershell
python -m venv .venv
```

Activate it in PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell execution policy blocks activation, provide the normal beginner-safe command:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

then activate again.

Install dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Provide a quick verification command:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

Also provide a Python script named `check_environment.py` if useful.

---

# 10. Requirements File

Create a `requirements.txt` compatible with the project.

Include:

```text
torch
torchvision
numpy
pandas
scikit-learn
matplotlib
seaborn
Pillow
tqdm
```

Do not pin fragile CUDA-specific wheel versions unless absolutely necessary.

Explain in README that users with NVIDIA GPU should install the appropriate PyTorch build from the official PyTorch installation selector if the normal pip installation does not provide CUDA support.

---

# 11. Reproducibility

Set random seeds for:

- Python
- NumPy
- PyTorch
- CUDA when available

Use seed 42.

Print the seed at the beginning of each run.

Where practical, make results deterministic, but do not break GPU performance unnecessarily.

---

# 12. Image Preprocessing

Use 224×224 inputs.

Training transformations:

```text
Resize to 224×224
RandomHorizontalFlip
RandomRotation(10 degrees)
ToTensor
ImageNet normalization
```

Validation/test transformations:

```text
Resize to 224×224
ToTensor
ImageNet normalization
```

Use:

```python
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

Do not apply random augmentation to validation or test images.

Use `torchvision.datasets.ImageFolder`.

---

# 13. Pretrained Models

Implement these exact torchvision models using current `weights=` APIs:

## AlexNet

Use pretrained ImageNet weights.

Replace its final classifier output layer with `num_classes`.

## ResNet-18

Use pretrained ImageNet weights.

Replace `model.fc` with a linear layer for `num_classes`.

## MobileNetV3-Small

Use pretrained ImageNet weights.

Replace the last classifier layer with a linear layer for `num_classes`.

## EfficientNet-B0

Use pretrained ImageNet weights.

Replace the final classifier layer with a linear layer for `num_classes`.

Do not use the deprecated:

```python
pretrained=True
```

Use the current torchvision `weights=...` style.

---

# 14. Transfer Learning Strategy

Use true fine-tuning for the benchmark.

Do NOT freeze the entire backbone by default.

Train the full model with a small learning rate:

```text
AdamW
learning_rate = 1e-4
```

Loss:

```python
CrossEntropyLoss()
```

Epochs:

```text
5 exactly
```

Do not secretly train for more epochs.

---

# 15. Training Requirements

Build a reusable training function.

For each epoch print:

```text
Epoch 1/5
Train Loss: ...
Train Accuracy: ...
Validation Loss: ...
Validation Accuracy: ...
```

After every epoch, compare validation accuracy.

Save the checkpoint with the best validation accuracy.

Example checkpoint naming:

```text
checkpoints/AlexNet_best.pth
checkpoints/ResNet18_best.pth
checkpoints/MobileNetV3Small_best.pth
checkpoints/EfficientNetB0_best.pth
```

Store training history in CSV or JSON.

---

# 16. Important Fairness Rule

Every model must use exactly the same:

- dataset split
- image size
- training augmentation
- batch size
- optimizer
- learning rate
- number of epochs
- validation procedure
- test set

Only the architecture should change.

Document this in README.

---

# 17. Test Evaluation

After 5 epochs:

1. Load the best validation checkpoint for that model.
2. Evaluate it once on the untouched test set.
3. Collect all true labels and predictions.

Do not use test accuracy to select the checkpoint.

---

# 18. Required Metrics

Calculate all required metrics.

## Accuracy

Use:

```python
accuracy_score(y_true, y_pred)
```

## F1-score

Primary reported F1 must be **macro F1**:

```python
f1_score(y_true, y_pred, average="macro")
```

Optionally also calculate weighted F1.

## Sensitivity

For each class:

```text
Sensitivity = TP / (TP + FN)
```

Calculate class-level values from the confusion matrix.

Report macro-average sensitivity.

## Specificity

For each class:

```text
Specificity = TN / (TN + FP)
```

Calculate class-level values from the confusion matrix.

Report macro-average specificity.

Also save per-class sensitivity and specificity to CSV/JSON so the user can inspect them.

---

# 19. Confusion Matrices

For every model, create a labeled confusion matrix.

Save:

```text
results/confusion_matrices/AlexNet.png
results/confusion_matrices/ResNet18.png
results/confusion_matrices/MobileNetV3Small.png
results/confusion_matrices/EfficientNetB0.png
```

Use:

- actual class names on y-axis
- predicted class names on x-axis
- integer counts in the cells
- readable labels
- high resolution, around 300 DPI

---

# 20. Final 2×2 Confusion Matrix Grid

Create:

```text
results/plots/confusion_matrix_grid.png
```

Layout:

```text
+----------------------+----------------------+
|       AlexNet        |      ResNet-18       |
|  Confusion Matrix    |  Confusion Matrix    |
+----------------------+----------------------+
|    MobileNetV3       |   EfficientNet-B0    |
|  Confusion Matrix    |  Confusion Matrix    |
+----------------------+----------------------+
```

Make it publication/report friendly.

Do not choose one model visually as a winner.

---

# 21. Inference Latency

Measure inference latency separately from training time.

Use the test loader.

Do a warm-up first, for example at least 20 images.

Then time around 200–500 images if available.

Report:

```text
Latency = total inference time / number of images × 1000
```

Result unit:

```text
milliseconds per image
```

For CUDA timing, use:

```python
torch.cuda.synchronize()
```

before starting and after finishing the measured section.

Do not include data loading or disk I/O in the inference latency calculation if possible.

The timing code should be clearly documented.

---

# 22. Model Size

For each trained model:

```python
torch.save(model.state_dict(), path)
```

Measure file size using `os.path.getsize()`.

Convert bytes to MB:

```text
MB = bytes / (1024**2)
```

Report size of the saved `state_dict` checkpoint in MB.

Clearly label it as:

```text
Model checkpoint size (MB)
```

Do not confuse it with RAM usage or GPU VRAM usage.

---

# 23. Final Benchmark Table

Create:

```text
results/benchmark_results.csv
results/benchmark_results.json
```

Columns must include:

```text
Model
Accuracy
Macro_F1
Macro_Sensitivity
Macro_Specificity
Latency_ms_per_image
Model_Size_MB
Best_Validation_Accuracy
```

Optional additional columns:

```text
Training_Time_seconds
Device
Epochs
Batch_Size
Image_Size
```

Round display values sensibly, but preserve raw results in JSON.

---

# 24. Required Final Table Format

Generate and print a clean console table like:

```text
============================================================
TRANSFER LEARNING BENCHMARK
============================================================

Model              Accuracy   Macro F1   Sensitivity   Specificity   Latency(ms)   Size(MB)
------------------------------------------------------------------------------------------------
AlexNet             xx.xx%     xx.xx%      xx.xx%        xx.xx%        xx.xx         xx.xx
ResNet-18           xx.xx%     xx.xx%      xx.xx%        xx.xx%        xx.xx         xx.xx
MobileNetV3-Small   xx.xx%     xx.xx%      xx.xx%        xx.xx%        xx.xx         xx.xx
EfficientNet-B0     xx.xx%     xx.xx%      xx.xx%        xx.xx%        xx.xx         xx.xx
```

Use actual measured values.

Never invent benchmark numbers.

---

# 25. Additional Plots

Create these plots under `results/plots/`:

1. `performance_comparison.png`
   - Accuracy and Macro F1 for all models

2. `latency_comparison.png`
   - milliseconds per image for each model

3. `model_size_comparison.png`
   - checkpoint size in MB for each model

4. `sensitivity_specificity_comparison.png`
   - macro sensitivity and macro specificity

Use readable labels and 300 DPI.

Do not hardcode colors unless needed; use default matplotlib/seaborn styling.

---

# 26. Per-Model Training Curves

Save training history for each model.

Create plots for:

- training vs validation accuracy
- training vs validation loss

For example:

```text
results/plots/AlexNet_training_curves.png
results/plots/ResNet18_training_curves.png
results/plots/MobileNetV3Small_training_curves.png
results/plots/EfficientNetB0_training_curves.png
```

Exactly 5 epochs must appear on the x-axis.

---

# 27. README Requirements

The generated README must have these sections:

1. Project Overview
2. What This Experiment Measures
3. Hardware Requirements
4. Software Requirements
5. Installation on Windows
6. VS Code Setup
7. Dataset Download
8. Dataset Folder Placement
9. Dataset Preparation
10. Verify Dataset
11. Run One Model
12. Run Full Benchmark
13. Understand the Results
14. Where Output Files Are Saved
15. Troubleshooting
16. How to Reduce RAM/VRAM Usage
17. CPU-only Mode
18. GPU/CUDA Check
19. Research/Reporting Notes
20. Limitations

The README should use simple beginner language.

---

# 28. Beginner Run Commands

The README must eventually provide this exact simple sequence.

### Open terminal

In VS Code:

```text
Terminal → New Terminal
```

### Create environment

```powershell
python -m venv .venv
```

### Activate

```powershell
.\.venv\Scripts\Activate.ps1
```

### Install packages

```powershell
pip install -r requirements.txt
```

### Check environment

```powershell
python check_environment.py
```

### Prepare dataset

```powershell
python prepare_dataset.py
```

### Verify dataset

```powershell
python verify_dataset.py
```

### Run the entire benchmark

```powershell
python run_benchmark.py
```

### Generate/refresh confusion grid if separated

```powershell
python create_confusion_grid.py
```

If your final implementation uses fewer or different scripts, keep the commands equally simple and update the README accordingly.

---

# 29. One-Model Test Mode

Before running all four models, implement an optional one-model mode.

For example:

```powershell
python run_benchmark.py --model EfficientNetB0
```

or:

```powershell
python run_benchmark.py --model all
```

Support:

```text
AlexNet
ResNet18
MobileNetV3Small
EfficientNetB0
all
```

This lets a beginner verify that the system works before spending time training all models.

If command-line arguments make the project too complicated for a beginner, provide a simple `MODEL_TO_RUN` variable in `config.py` as an alternative.

---

# 30. Memory/VRAM Safety

The code must be designed to avoid common beginner crashes.

If GPU memory is insufficient:

- reduce batch size
- optionally use mixed precision on CUDA if implemented safely
- keep image size at 224×224

If RAM is insufficient:

- use `num_workers=0`
- avoid loading the entire dataset into RAM
- use `ImageFolder` + DataLoader
- do not convert all images into one huge NumPy array

Do not preload the whole dataset into memory.

---

# 31. Windows Compatibility

The project must work correctly on Windows.

Avoid code that assumes Linux paths such as `/home/user/...`.

Use `pathlib.Path` or `os.path.join` everywhere.

Protect the training entry point with:

```python
if __name__ == "__main__":
    main()
```

This is important for Windows multiprocessing/DataLoader behavior.

Start with:

```text
NUM_WORKERS = 0
```

for maximum beginner compatibility.

---

# 32. Error Handling

Make errors beginner-friendly.

Examples:

### Dataset missing

Print:

```text
ERROR: Dataset not found.
Please extract the Kaggle dataset into:
...
Expected folders:
Training/
Testing/
```

### Missing class folder

Print which class is missing.

### CUDA unavailable

Print:

```text
CUDA GPU not detected. Falling back to CPU.
Training will be slower.
```

### Out of memory

Catch the common CUDA OOM error where reasonable and print:

```text
CUDA out of memory.
Try reducing BATCH_SIZE from 16 to 8 or 4 in config.py.
```

Do not silently swallow exceptions.

---

# 33. Logging

Save useful run information to:

```text
results/logs/
```

Record:

- timestamp
- Python/PyTorch version
- device
- GPU name if available
- image size
- batch size
- learning rate
- seed
- model name
- epoch metrics
- final benchmark results

---

# 34. Research Reproducibility

At the end of the benchmark, save a machine-readable configuration file:

```text
results/run_config.json
```

Include:

```text
seed
image_size
batch_size
epochs
learning_rate
optimizer
models
dataset split
device
```

This makes the experiment reproducible.

---

# 35. Important Scientific/Reporting Rules

Do not claim that one model is universally “best.”

The benchmark is limited to:

- one dataset
- one train/validation split
- one test set
- 5 epochs
- one preprocessing setup
- one hardware environment

The generated README should explain that this is a **quick benchmark**, not clinical validation.

Do not call the model clinically reliable.

Do not invent results.

Do not copy metrics from papers and put them into the experiment table.

Published literature metrics and experimental metrics must remain clearly separate.

---

# 36. Important Medical-Dataset Note

Add a limitations section explaining that medical imaging benchmarks can be affected by:

- class imbalance
- duplicate or highly related images
- dataset-specific bias
- image acquisition differences
- potential data leakage if patient-level information is unavailable
- small experimental training budget

State that this experiment is intended for educational/research comparison and not clinical diagnosis.

---

# 37. Expected Output Folder

After a successful run, the user should be able to open:

```text
results/
│
├── benchmark_results.csv
├── benchmark_results.json
├── run_config.json
│
├── confusion_matrices/
│   ├── AlexNet.png
│   ├── ResNet18.png
│   ├── MobileNetV3Small.png
│   └── EfficientNetB0.png
│
├── plots/
│   ├── confusion_matrix_grid.png
│   ├── performance_comparison.png
│   ├── latency_comparison.png
│   ├── model_size_comparison.png
│   ├── sensitivity_specificity_comparison.png
│   ├── AlexNet_training_curves.png
│   ├── ResNet18_training_curves.png
│   ├── MobileNetV3Small_training_curves.png
│   └── EfficientNetB0_training_curves.png
│
└── logs/
```

---

# 38. Final Beginner Workflow

The completed README should teach the user to do this:

```text
1. Install Python 3.11
        ↓
2. Install VS Code
        ↓
3. Create/open project folder
        ↓
4. Create Python virtual environment
        ↓
5. Activate environment
        ↓
6. Install requirements
        ↓
7. Put dataset in data/raw
        ↓
8. Run environment check
        ↓
9. Prepare dataset
        ↓
10. Verify dataset
        ↓
11. Run one model first
        ↓
12. Check results
        ↓
13. Run all four models
        ↓
14. Open benchmark_results.csv
        ↓
15. Open confusion_matrix_grid.png
        ↓
16. Use results in survey/report
```

---

# 39. Do Not Do These Things

Do not:

- use Google Colab
- require a cloud server
- require Docker
- require a database
- build a web application
- use TensorFlow unless absolutely necessary
- change the model set without asking
- train more than 5 epochs by default
- use the test set for model selection
- invent benchmark numbers
- download random datasets from unknown sources
- hide errors
- silently change batch size or epochs without recording it

---

# 40. What I Expect You, Antigravity, to Deliver

Build the actual project files, not just an explanation.

At minimum create:

```text
README.md
requirements.txt
config.py
check_environment.py
prepare_dataset.py
verify_dataset.py
train_model.py
run_benchmark.py
evaluate_models.py
create_confusion_grid.py
utils.py
.gitignore
```

Also create all required folders.

The code must be runnable from the Windows terminal.

Before presenting the finished project, perform a code-level consistency check:

- imports are correct
- filenames match README commands
- model names match everywhere
- class count is detected automatically
- checkpoint directories are created automatically
- results directories are created automatically
- CPU mode works
- CUDA mode works if CUDA is installed
- Windows DataLoader behavior is safe
- CSV/JSON outputs are valid
- confusion matrices are saved
- benchmark table contains all required metrics

If the dataset itself is not available to you, do not pretend to train it. Build the complete project and tell the user exactly where to place the downloaded dataset.

---

# 41. Final Console Message

At the end of a successful run, print:

```text
============================================================
BENCHMARK COMPLETE
============================================================

Models evaluated:
- AlexNet
- ResNet-18
- MobileNetV3-Small
- EfficientNet-B0

Epochs per model: 5

Results saved to:
results/benchmark_results.csv
results/benchmark_results.json

Confusion matrix grid:
results/plots/confusion_matrix_grid.png

You can now use these measured results in your survey/report.
============================================================
```

Again: use actual measured values only.

---

# 42. Final Instruction to Antigravity

Please build this project **completely**, with beginner-friendly code and documentation.

Do not only give me snippets.

Create the full project files and explain:

1. what each file does
2. exactly where I should place the Kaggle dataset
3. exactly which terminal commands I should run
4. how to check whether my GPU is being used
5. how to run one model first
6. how to run all four models
7. where the results are saved
8. how to open the CSV and PNG results
9. what each metric means
10. what to do if I get CUDA out-of-memory, missing-package, dataset-path, or Windows multiprocessing errors

The final system should be simple enough for a beginner to operate from VS Code with minimal manual code editing.
