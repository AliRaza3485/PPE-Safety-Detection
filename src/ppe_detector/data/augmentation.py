"""
Augmentation pipelines for PPE detection training.

Two pipelines:
1. `get_base_transforms()` — general-purpose augmentation applied to all
   training images. Makes the model robust to real-world variation in
   lighting, camera angle, and image quality (construction sites are
   rarely photographed under ideal conditions).
2. `get_minority_boost_transforms()` — a heavier augmentation pipeline
   applied specifically to images containing the minority class (`head`,
   i.e. no-helmet violations). This is used for offline oversampling:
   we generate multiple augmented copies of these images to increase
   their effective representation in training without collecting new data.
"""

import albumentations as A


def get_base_transforms(img_size: int = 640) -> A.Compose:
    """Standard augmentation applied to all training images."""
    return A.Compose(
        [
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(
                hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3
            ),
            A.Blur(blur_limit=3, p=0.1),
            A.RandomRotate90(p=0.2),
            A.HorizontalFlip(p=0.5),
            A.RandomScale(scale_limit=0.2, p=0.3),
            A.Resize(img_size, img_size),
        ],
        bbox_params=A.BboxParams(
            format="yolo", label_fields=["class_labels"], min_visibility=0.3
        ),
    )


def get_minority_boost_transforms(img_size: int = 640) -> A.Compose:
    """
    Heavier, more varied augmentation for minority-class (head/no-helmet)
    images. Used to create multiple distinct augmented copies of the same
    source image, effectively oversampling the minority class.
    """
    return A.Compose(
        [
            A.RandomBrightnessContrast(
                brightness_limit=0.35, contrast_limit=0.35, p=0.7
            ),
            A.HueSaturationValue(
                hue_shift_limit=15, sat_shift_limit=30, val_shift_limit=15, p=0.5
            ),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.Blur(blur_limit=4, p=0.2),
            A.RandomRotate90(p=0.3),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1, scale_limit=0.25, rotate_limit=15, p=0.5
            ),
            A.RandomShadow(p=0.2),
            A.Resize(img_size, img_size),
        ],
        bbox_params=A.BboxParams(
            format="yolo", label_fields=["class_labels"], min_visibility=0.3
        ),
    )
