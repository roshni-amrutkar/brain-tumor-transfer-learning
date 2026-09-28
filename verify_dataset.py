"""
Dataset Verification and Integrity Inspection Script.
Validates:
- Image counts per class and split (train, val, test)
- Presence of required class directories
- Unsupported file formats
- Corrupted images via Pillow verification
"""

import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple
from PIL import Image

import config


def verify_split_folder(split_dir: Path) -> Tuple[Dict[str, int], List[str], List[str]]:
    """
    Inspect a split folder (train/val/test).
    Returns:
        (class_counts, corrupted_images, unsupported_files)
    """
    class_counts: Dict[str, int] = {}
    corrupted_images: List[str] = []
    unsupported_files: List[str] = []

    if not split_dir.exists():
        return class_counts, corrupted_images, unsupported_files

    for class_folder in sorted(split_dir.iterdir()):
        if not class_folder.is_dir():
            continue

        class_name = class_folder.name
        count = 0

        for file_path in class_folder.iterdir():
            if not file_path.is_file():
                continue

            if file_path.suffix.lower() not in config.SUPPORTED_EXTENSIONS:
                unsupported_files.append(str(file_path))
                continue

            # Verify image file integrity
            try:
                with Image.open(file_path) as img:
                    img.verify()
                count += 1
            except Exception as e:
                corrupted_images.append(f"{file_path} (Reason: {e})")

        class_counts[class_name] = count

    return class_counts, corrupted_images, unsupported_files


def verify_dataset() -> bool:
    print("=" * 60)
    print("DATASET VERIFICATION & INTEGRITY REPORT")
    print("=" * 60)

    if not config.PROCESSED_DATA_DIR.exists():
        print("ERROR: Processed dataset directory does not exist.")
        print(f"Looked in: {config.PROCESSED_DATA_DIR.resolve()}")
        print("Please run: python prepare_dataset.py first.")
        return False

    splits = {
        "Training (train)": config.TRAIN_DATA_DIR,
        "Validation (val)": config.VAL_DATA_DIR,
        "Testing (test)": config.TEST_DATA_DIR,
    }

    all_corrupted: List[str] = []
    all_unsupported: List[str] = []
    split_summaries: Dict[str, Dict[str, int]] = {}
    all_classes: Set[str] = set()

    for split_label, split_path in splits.items():
        counts, corrupted, unsupported = verify_split_folder(split_path)
        split_summaries[split_label] = counts
        all_classes.update(counts.keys())
        all_corrupted.extend(corrupted)
        all_unsupported.extend(unsupported)

    if not all_classes:
        print("ERROR: No class directories or images found in data/processed.")
        print("Please run: python prepare_dataset.py")
        return False

    sorted_classes = sorted(list(all_classes))
    print(f"Detected Classes ({len(sorted_classes)}): {', '.join(sorted_classes)}\n")

    # Table format
    print(f"{'Class':<20} | {'Train':<10} | {'Val':<10} | {'Test':<10} | {'Total':<10}")
    print("-" * 65)

    totals = {"train": 0, "val": 0, "test": 0}
    empty_folders_detected = False

    for cls in sorted_classes:
        tr = split_summaries["Training (train)"].get(cls, 0)
        va = split_summaries["Validation (val)"].get(cls, 0)
        te = split_summaries["Testing (test)"].get(cls, 0)
        cls_tot = tr + va + te

        totals["train"] += tr
        totals["val"] += va
        totals["test"] += te

        if tr == 0 or va == 0 or te == 0:
            empty_folders_detected = True

        print(f"{cls:<20} | {tr:<10} | {va:<10} | {te:<10} | {cls_tot:<10}")

    print("-" * 65)
    total_all = sum(totals.values())
    print(f"{'TOTAL':<20} | {totals['train']:<10} | {totals['val']:<10} | {totals['test']:<10} | {total_all:<10}\n")

    # Check warnings and errors
    has_errors = False

    if empty_folders_detected:
        print("WARNING: One or more class folders contain 0 images in a split!")
        has_errors = True

    if all_unsupported:
        print(f"WARNING: {len(all_unsupported)} unsupported file(s) found:")
        for f in all_unsupported[:5]:
            print(f"  - {f}")
        if len(all_unsupported) > 5:
            print(f"  ... and {len(all_unsupported) - 5} more.")

    if all_corrupted:
        print(f"ERROR: {len(all_corrupted)} corrupted image(s) detected:")
        for c in all_corrupted:
            print(f"  - {c}")
        has_errors = True
    else:
        print("Image Integrity Check: PASSED (No corrupted images detected).")

    print("=" * 60)
    if not has_errors and total_all > 0:
        print("STATUS: Dataset is VERIFIED and ready for benchmarking!")
        print("=" * 60)
        return True
    else:
        print("STATUS: Dataset verification FAILED or has empty classes.")
        print("=" * 60)
        return False


if __name__ == "__main__":
    success = verify_dataset()
    if not success:
        sys.exit(1)
