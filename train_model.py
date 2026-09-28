"""
Training Engine for Brain Tumor MRI Transfer Learning.
Performs fine-tuning for exactly 5 epochs using AdamW and CrossEntropyLoss.
Tracks epoch metrics, saves the best checkpoint on validation accuracy,
and records training history logs.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm

import config
from models import get_model
import utils


def get_dataloaders(
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = config.NUM_WORKERS
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """
    Create PyTorch DataLoaders for train, val, and test splits.
    """
    train_transform, eval_transform = utils.get_transforms(config.IMAGE_SIZE)

    if not config.TRAIN_DATA_DIR.exists():
        raise FileNotFoundError(
            f"Training directory not found at: {config.TRAIN_DATA_DIR}. "
            "Please run 'python prepare_dataset.py' first."
        )

    train_dataset = ImageFolder(root=str(config.TRAIN_DATA_DIR), transform=train_transform)
    val_dataset = ImageFolder(root=str(config.VAL_DATA_DIR), transform=eval_transform)
    test_dataset = ImageFolder(root=str(config.TEST_DATA_DIR), transform=eval_transform)

    class_names = train_dataset.classes

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(torch.cuda.is_available()),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(torch.cuda.is_available()),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(torch.cuda.is_available()),
    )

    return train_loader, val_loader, test_loader, class_names


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> Tuple[float, float]:
    """Train the model for one complete epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, targets in tqdm(dataloader, desc="Training", leave=False):
        images, targets = images.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    epoch_loss = running_loss / total if total > 0 else 0.0
    epoch_acc = (correct / total) * 100.0 if total > 0 else 0.0
    return epoch_loss, epoch_acc


def validate_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """Evaluate the model on the validation dataset."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Validation", leave=False):
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    val_loss = running_loss / total if total > 0 else 0.0
    val_acc = (correct / total) * 100.0 if total > 0 else 0.0
    return val_loss, val_acc


def train_model(
    model_name: str,
    epochs: int = config.EPOCHS,
    batch_size: int = config.BATCH_SIZE,
    learning_rate: float = config.LEARNING_RATE,
    device: torch.device = None
) -> Dict[str, Any]:
    """
    Fine-tune a pretrained model for exactly specified epochs (default: 5).
    Saves the best validation accuracy checkpoint and exports training history.
    """
    utils.set_seed(config.RANDOM_SEED)
    config.ensure_directories()

    if device is None:
        device = config.get_device()

    print(f"\n{'=' * 60}")
    print(f"Starting Fine-Tuning: {model_name}")
    print(f"Epochs: {epochs} | Batch Size: {batch_size} | LR: {learning_rate} | Device: {device.type}")
    print(f"{'=' * 60}")

    train_loader, val_loader, _, class_names = get_dataloaders(batch_size=batch_size)
    num_classes = len(class_names)

    model = get_model(model_name=model_name, num_classes=num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=config.WEIGHT_DECAY)

    best_val_acc = 0.0
    checkpoint_path = config.CHECKPOINTS_DIR / f"{model_name}_best.pth"

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "epoch_time": [],
    }

    start_total_time = time.time()

    try:
        for epoch in range(1, epochs + 1):
            epoch_start = time.time()

            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, device)
            epoch_duration = time.time() - epoch_start

            history["train_loss"].append(round(train_loss, 4))
            history["train_acc"].append(round(train_acc, 2))
            history["val_loss"].append(round(val_loss, 4))
            history["val_acc"].append(round(val_acc, 2))
            history["epoch_time"].append(round(epoch_duration, 2))

            print(f"Epoch {epoch}/{epochs}")
            print(f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_acc:.2f}%")
            print(f"Validation Loss: {val_loss:.4f} | Validation Accuracy: {val_acc:.2f}%")

            # Checkpoint best validation model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), checkpoint_path)
                print(f"--> Saved new best checkpoint to: {checkpoint_path.name} (Val Acc: {val_acc:.2f}%)")
            print("-" * 60)

    except torch.cuda.OutOfMemoryError:
        print("\n" + "!" * 60)
        print("CUDA Out Of Memory Error!")
        print("Suggestion: Reduce BATCH_SIZE in config.py from 16 -> 8 or 4.")
        print("!" * 60 + "\n")
        raise

    total_training_time = time.time() - start_total_time

    # Save training history
    history_file = config.LOGS_DIR / f"{model_name}_history.json"
    utils.save_json({
        "model_name": model_name,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "best_val_acc": round(best_val_acc, 2),
        "total_training_time_sec": round(total_training_time, 2),
        "history": history
    }, history_file)

    print(f"\nTraining for {model_name} completed in {total_training_time:.2f}s.")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Checkpoint saved: {checkpoint_path}")

    return {
        "model_name": model_name,
        "best_val_acc": best_val_acc,
        "checkpoint_path": checkpoint_path,
        "total_training_time": total_training_time,
        "history": history,
        "class_names": class_names,
    }


def main():
    parser = argparse.ArgumentParser(description="Train a CNN model on Brain Tumor MRI dataset.")
    parser.add_argument(
        "--model",
        type=str,
        default="ResNet18",
        choices=["AlexNet", "ResNet18", "MobileNetV3Small", "EfficientNetB0"],
        help="Model architecture to train"
    )
    parser.add_argument("--epochs", type=int, default=config.EPOCHS, help="Number of epochs (default: 5)")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE, help="Batch size (default: 16)")
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE, help="Learning rate (default: 1e-4)")

    args = parser.parse_args()
    config.print_device_info()
    train_model(
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )


if __name__ == "__main__":
    main()
