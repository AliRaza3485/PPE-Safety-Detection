"""
Create a validation split from the training set.
Moves a stratified random sample of images+labels from train/ to valid/.
"""

import random
import shutil
from pathlib import Path

# Config
SEED = 42
VAL_RATIO = 0.15
DATA_ROOT = Path("data/raw")

TRAIN_IMAGES = DATA_ROOT / "train" / "images"
TRAIN_LABELS = DATA_ROOT / "train" / "labels"
VAL_IMAGES = DATA_ROOT / "valid" / "images"
VAL_LABELS = DATA_ROOT / "valid" / "labels"


def create_validation_split():
    random.seed(SEED)

    VAL_IMAGES.mkdir(parents=True, exist_ok=True)
    VAL_LABELS.mkdir(parents=True, exist_ok=True)

    # Get all training images
    image_files = sorted(TRAIN_IMAGES.glob("*.*"))
    print(f"Total training images found: {len(image_files)}")

    # Shuffle and select validation subset
    random.shuffle(image_files)
    n_val = int(len(image_files) * VAL_RATIO)
    val_files = image_files[:n_val]

    print(f"Moving {n_val} images to validation set ({VAL_RATIO*100:.0f}%)...")

    moved, missing_labels = 0, 0

    for img_path in val_files:
        # Move image
        dest_img = VAL_IMAGES / img_path.name
        shutil.move(str(img_path), str(dest_img))

        # Move corresponding label (same filename, .txt extension)
        label_path = TRAIN_LABELS / (img_path.stem + ".txt")
        if label_path.exists():
            dest_label = VAL_LABELS / label_path.name
            shutil.move(str(label_path), str(dest_label))
            moved += 1
        else:
            missing_labels += 1
            print(f"  Warning: no label found for {img_path.name}")

    print(f"\nDone. Moved {moved} image+label pairs to validation set.")
    if missing_labels:
        print(f"Warning: {missing_labels} images had no matching label file.")

    # Final counts
    remaining_train = len(list(TRAIN_IMAGES.glob("*.*")))
    val_count = len(list(VAL_IMAGES.glob("*.*")))
    test_count = len(list((DATA_ROOT / "test" / "images").glob("*.*")))

    print(f"\nFinal split:")
    print(f"  Train: {remaining_train}")
    print(f"  Valid: {val_count}")
    print(f"  Test:  {test_count}")
    print(f"  Total: {remaining_train + val_count + test_count}")


if __name__ == "__main__":
    create_validation_split()
