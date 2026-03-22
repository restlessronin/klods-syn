---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.1
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# Dataset Loading and Verification

Download and verify the Gdańsk LEGO classification dataset.
We use only the real photos for baseline evaluation.

```python
# | default_exp data
```

```python
# | export
from pathlib import Path
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
```

## Configuration

```python
# | export
DATA_DIR = Path("../data/gdansk")
PHOTOS_DIR = DATA_DIR / "photos"
MIN_IMAGES_PER_CLASS = 5
RANDOM_SEED = 42
```

## Download instructions

The dataset must be downloaded manually from MostWiedzy.

1. **Primary:** "LEGO bricks for training classification network"
   — DOI: `10.34808/rcza-jy08`
   — URL: https://mostwiedzy.pl/en/open-research-data/lego-bricks-for-training-classification-network,618104539639776-0
   — Download `final_dataset_prepared_447_classes.tar`, extract to `data/gdansk/`

2. **Alternative (real photos only):** "Images of LEGO bricks"
   — DOI: `10.34808/arsb-4268`

We use only the `photos/` subdirectory. The `renders/` subdirectory
contains synthetic LDraw renders that we'll compare against our own
rendering pipeline later.

## Dataset verification

```python
def verify_dataset(photos_dir: Path = PHOTOS_DIR):
    """Print dataset statistics and verify structure."""
    if not photos_dir.exists():
        print(f"Photos directory not found at {photos_dir}")
        print("See download instructions above.")
        return
    class_dirs = sorted([d for d in photos_dir.glob("*/") if d.is_dir()])
    categories = sorted([d for d in photos_dir.iterdir() if d.is_dir()])
    part_dirs = sorted([d for d in photos_dir.glob("*/*") if d.is_dir()])
    n_classes = len(part_dirs)
    print(f"Photos directory: {photos_dir}")
    print(f"Categories: {len(categories)}")
    class_counts = {}
    for part_dir in part_dirs:
        count = len(
            list(part_dir.glob("*.jpg"))
            + list(part_dir.glob("*.jpeg"))
            + list(part_dir.glob("*.png"))
        )
        class_counts[part_dir.name] = count
    total = sum(class_counts.values())
    counts = list(class_counts.values())
    print(f"Classes: {n_classes}")
    print(f"Total images: {total}")
    print(
        f"Images per class — min: {min(counts)}, max: {max(counts)}, "
        f"mean: {total / n_classes:.1f}, median: {sorted(counts)[n_classes // 2]}"
    )
    print(f"Sample classes: {list(class_counts.keys())[:10]}")
    sparse = {k: v for k, v in class_counts.items() if v < MIN_IMAGES_PER_CLASS}
    if sparse:
        print(f"\nClasses with <{MIN_IMAGES_PER_CLASS} images ({len(sparse)}):")
        for k, v in sorted(sparse.items(), key=lambda x: x[1]):
            print(f"  {k}: {v}")
    return class_counts
```

```python
class_counts = verify_dataset()
```

## Dataset class

```python
# | export
class LegoDataset(Dataset):
    """LEGO brick image dataset from directory structure.

    Each subdirectory under `root` is a class, named by part number.
    """

    def __init__(
        self,
        image_paths: list[str],
        labels: np.ndarray,
        transform: transforms.Compose | None = None,
    ):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]
```

```python
# | export
def load_class_map(
    photos_dir: Path = PHOTOS_DIR, min_images: int = MIN_IMAGES_PER_CLASS
) -> tuple[list[str], dict[str, int]]:
    """Load class names and build name→index mapping.

    Walks photos_dir/category/part_number/. Skips classes with fewer
    than `min_images` images. Returns (class_names, class_to_idx).
    """
    class_names = []
    for part_dir in sorted(photos_dir.glob("*/*")):
        if not part_dir.is_dir():
            continue
        count = len(
            list(part_dir.glob("*.jpg"))
            + list(part_dir.glob("*.jpeg"))
            + list(part_dir.glob("*.png"))
        )
        if count >= min_images:
            class_names.append(part_dir.name)
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    return class_names, class_to_idx
```

```python
# | export
def load_image_paths(
    photos_dir: Path = PHOTOS_DIR, class_to_idx: dict[str, int] | None = None
) -> tuple[list[str], list[int]]:
    """Load all image paths and their integer labels.

    If class_to_idx is provided, only loads classes present in the mapping.
    """
    if class_to_idx is None:
        _, class_to_idx = load_class_map(photos_dir)
    part_dirs = {}
    for part_dir in photos_dir.glob("*/*"):
        if part_dir.is_dir() and part_dir.name in class_to_idx:
            part_dirs[part_dir.name] = part_dir
    image_paths = []
    labels = []
    for cls_name, cls_idx in class_to_idx.items():
        cls_dir = part_dirs.get(cls_name)
        if cls_dir is None:
            continue
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            for img_path in cls_dir.glob(ext):
                image_paths.append(str(img_path))
                labels.append(cls_idx)
    return image_paths, labels
```

```python
# Quick test
class_names, class_to_idx = load_class_map()
image_paths, labels = load_image_paths(class_to_idx=class_to_idx)
print(f"Loaded {len(image_paths)} images across {len(class_names)} classes")
```

```python
# | hide
import nbdev

nbdev.nbdev_export()
```
