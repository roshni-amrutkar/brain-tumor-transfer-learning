"""
Dataset Preparation Script.
Prepares the Brain Tumor MRI dataset by creating deterministic splits:
- Original 'Testing' -> data/processed/test
- Original 'Training' -> 85% data/processed/train, 15% data/processed/val
Preserves class balance and handles idempotent reruns.
"""

import os
import random
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import config
import utils


def locate_raw_dataset() -> Optional[Tuple[Path, Path]]:
    """
    Search standard and fallback paths for 'Training' and 'Testing' folders.
    Returns (training_dir, testing_dir) if found, else None.
    """
    candidate_roots = [
        config.RAW_DATA_DIR,
        config.BASE_DIR / "dataset",
        config.DATA_DIR,
        config.BASE_DIR,
    ]

    for root in candidate_roots:
        if not root.exists():
            continue
        # Search for Training and Testing folders (case-insensitive)
        train_dir = None
        test_dir = None
        for item in root.iterdir():
            if item.is_dir():
                if item.name.lower() in ("training", "train"):
                    train_dir = item
                elif item.name.lower() in ("testing", "test"):
                    test_dir = item
        if train_dir and test_dir:
            return train_dir, test_dir

    return None


def get_class_images(folder: Path) -> Dict[str, List[Path]]:
    """Scan directory for class folders and their supported image files."""
    class_map: Dict[str, List[Path]] = {}
    if not folder.exists():
        return class_map

    for class_folder in sorted(folder.iterdir()):
        if class_folder.is_dir():
            images = [
                f for f in class_folder.iterdir()
                if f.is_file() and f.suffix.lower() in config.SUPPORTED_EXTENSIONS
            ]
            class_map[class_folder.name] = sorted(images)
    return class_map


def prepare_dataset(
    val_split: float = config.VAL_SPLIT,
    seed: int = config.RANDOM_SEED
) -> bool:
    """
    Split and organize images into data/processed/train, val, and test.
    """
    utils.set_seed(seed)
    config.ensure_directories()

    raw_dirs = locate_raw_dataset()
    if not raw_dirs:
        print("\n" + "!" * 60)
        print("ERROR: Dataset not found!")
        print("Please extract the Kaggle Brain Tumor MRI dataset into:")
        print(f"   {config.RAW_DATA_DIR.resolve()}  or  {(config.BASE_DIR / 'dataset').resolve()}")
        print("\nExpected directory structure:")
        print("   dataset/ (or data/raw/)")
        print("   ├── Training/")
        print("   │   ├── glioma (or glioma_tumor)/")
        print("   │   ├── meningioma (or meningioma_tumor)/")
        print("   │   ├── notumor (or no_tumor)/")
        print("   │   └── pituitary (or pituitary_tumor)/")
        print("   └── Testing/")
        print("       ├── glioma...")
        print("       ├── meningioma...")
        print("       ├── notumor...")
        print("       └── pituitary...")
        print("!" * 60 + "\n")
        return False

    raw_train_dir, raw_test_dir = raw_dirs
    print(f"Found Raw Training Data at : {raw_train_dir}")
    print(f"Found Raw Testing Data at  : {raw_test_dir}")

    # Read images by class
    train_class_map = get_class_images(raw_train_dir)
    test_class_map = get_class_images(raw_test_dir)

    if not train_class_map or not test_class_map:
        print("ERROR: Could not find class folders containing images.")
        return False

    print(f"\nDetected Classes: {list(train_class_map.keys())}")

    # Clean existing processed directory to guarantee idempotency
    for target_dir in [config.TRAIN_DATA_DIR, config.VAL_DATA_DIR, config.TEST_DATA_DIR]:
        if target_dir.exists():
            shutil.rmtree(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

    counts = {"train": 0, "val": 0, "test": 0}

    # 1. Process and split Training -> Train (85%) & Val (15%)
    print("\nSplitting Training dataset into Train (85%) and Validation (15%)...")
    for class_name, img_paths in train_class_map.items():
        # Shuffle deterministically
        shuffled = img_paths.copy()
        random.shuffle(shuffled)

        n_val = int(len(shuffled) * val_split)
        val_imgs = shuffled[:n_val]
        train_imgs = shuffled[n_val:]

        train_cls_dir = config.TRAIN_DATA_DIR / class_name
        val_cls_dir = config.VAL_DATA_DIR / class_name
        train_cls_dir.mkdir(parents=True, exist_ok=True)
        val_cls_dir.mkdir(parents=True, exist_ok=True)

        for img in train_imgs:
            shutil.copy2(img, train_cls_dir / img.name)
            counts["train"] += 1

        for img in val_imgs:
            shutil.copy2(img, val_cls_dir / img.name)
            counts["val"] += 1

    # 2. Process Testing -> Test (100% untouched)
    print("Copying untouched Testing set to data/processed/test...")
    for class_name, img_paths in test_class_map.items():
        test_cls_dir = config.TEST_DATA_DIR / class_name
        test_cls_dir.mkdir(parents=True, exist_ok=True)

        for img in img_paths:
            shutil.copy2(img, test_cls_dir / img.name)
            counts["test"] += 1

    print("\n" + "=" * 60)
    print("DATASET PREPARATION COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print(f"Processed Directory : {config.PROCESSED_DATA_DIR.resolve()}")
    print(f"Training images     : {counts['train']}")
    print(f"Validation images   : {counts['val']}")
    print(f"Testing images      : {counts['test']}")
    print(f"Total processed     : {sum(counts.values())}")
    print("=" * 60 + "\n")
    return True


if __name__ == "__main__":
    prepare_dataset()
