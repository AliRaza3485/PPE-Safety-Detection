"""
Oversample minority-class (head/no-helmet) images in the training set.

For every training image that contains at least one 'head' (no-helmet)
label, this script generates N augmented copies using the heavier
augmentation pipeline (get_minority_boost_transforms). The copies are
saved back into train/images and train/labels with a suffix, effectively
increasing the minority class's representation without touching the
original raw dataset — this only affects data/raw/train, which is
already the working copy (not gitignored differently from anything
already produced).
"""

import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from ppe_detector.data.augmentation import get_minority_boost_transforms

DATA_ROOT = Path("data/raw")
TRAIN_IMAGES = DATA_ROOT / "train" / "images"
TRAIN_LABELS = DATA_ROOT / "train" / "labels"
HEAD_CLASS_ID = 0  # from data.yaml: names: ['head', 'helmet']
OVERSAMPLE_FACTOR = 2  # generate this many augmented copies per source image
IMG_SIZE = 640


def read_yolo_labels(label_path: Path):
    boxes, class_labels = [], []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            cls_id = int(parts[0])
            x, y, w, h = map(float, parts[1:5])
            boxes.append([x, y, w, h])
            class_labels.append(cls_id)
    return boxes, class_labels


def write_yolo_labels(label_path: Path, boxes, class_labels):
    with open(label_path, "w") as f:
        for cls_id, (x, y, w, h) in zip(class_labels, boxes):
            f.write(f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")


def find_head_images():
    """Return list of image paths whose label file contains the head class."""
    head_images = []
    for label_path in TRAIN_LABELS.glob("*.txt"):
        _, class_labels = read_yolo_labels(label_path)
        if HEAD_CLASS_ID in class_labels:
            img_candidates = list(TRAIN_IMAGES.glob(f"{label_path.stem}.*"))
            if img_candidates:
                head_images.append(img_candidates[0])
    return head_images


def oversample():
    transform = get_minority_boost_transforms(img_size=IMG_SIZE)
    head_images = find_head_images()
    print(f"Found {len(head_images)} training images containing 'head' class.")
    print(f"Generating {OVERSAMPLE_FACTOR} augmented copies each...")

    total_generated = 0
    total_skipped = 0

    for img_path in head_images:
        label_path = TRAIN_LABELS / f"{img_path.stem}.txt"
        boxes, class_labels = read_yolo_labels(label_path)

        image = cv2.imread(str(img_path))
        if image is None:
            total_skipped += 1
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        for i in range(OVERSAMPLE_FACTOR):
            try:
                augmented = transform(
                    image=image, bboxes=boxes, class_labels=class_labels
                )
            except Exception as e:
                total_skipped += 1
                continue

            if len(augmented["bboxes"]) == 0:
                # augmentation pushed all boxes out of frame — skip this copy
                total_skipped += 1
                continue

            aug_image = cv2.cvtColor(augmented["image"], cv2.COLOR_RGB2BGR)
            new_stem = f"{img_path.stem}_aug{i}"

            out_img_path = TRAIN_IMAGES / f"{new_stem}{img_path.suffix}"
            out_label_path = TRAIN_LABELS / f"{new_stem}.txt"

            cv2.imwrite(str(out_img_path), aug_image)
            write_yolo_labels(
                out_label_path, augmented["bboxes"], augmented["class_labels"]
            )
            total_generated += 1

    print(f"\nDone. Generated {total_generated} new image+label pairs.")
    print(f"Skipped {total_skipped} (augmentation removed all boxes, or read error).")


if __name__ == "__main__":
    oversample()
