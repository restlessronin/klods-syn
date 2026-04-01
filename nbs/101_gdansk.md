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

# Gdańsk Dataset Preparation

Download and verify the Gdańsk LEGO classification dataset.
Split into train/test, extract embeddings from pretrained backbones,
and cache for evaluation.

```python
import numpy as np
from sklearn.model_selection import train_test_split

from klods_syn.data import (
    load_class_map,
    load_image_paths,
    extract_and_cache_embeddings,
    PHOTOS_DIR,
    RANDOM_SEED,
    RESULTS_DIR,
)
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
def verify_dataset():
    """Print dataset statistics and verify structure."""
    if not PHOTOS_DIR.exists():
        print(f"Photos directory not found at {PHOTOS_DIR}")
        print("See download instructions above.")
        return
    categories = sorted([d for d in PHOTOS_DIR.iterdir() if d.is_dir()])
    part_dirs = sorted([d for d in PHOTOS_DIR.glob("*/*") if d.is_dir()])
    n_classes = len(part_dirs)
    print(f"Photos directory: {PHOTOS_DIR}")
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
    sparse = {k: v for k, v in class_counts.items() if v < 5}
    if sparse:
        print(f"\nClasses with <5 images ({len(sparse)}):")
        for k, v in sorted(sparse.items(), key=lambda x: x[1]):
            print(f"  {k}: {v}")
    return class_counts


class_counts = verify_dataset()
```

## Train/test split

```python
TEST_SPLIT = 0.3

class_names, class_to_idx = load_class_map()
image_paths, labels = load_image_paths(class_to_idx=class_to_idx)
n_classes = len(class_names)
print(f"Classes: {n_classes}, Images: {len(image_paths)}")

train_paths, test_paths, train_labels, test_labels = train_test_split(
    image_paths, labels, test_size=TEST_SPLIT, random_state=RANDOM_SEED, stratify=labels
)
print(f"Train: {len(train_paths)}, Test: {len(test_paths)}")
```

## Extract and cache embeddings

```python
train_embeddings = extract_and_cache_embeddings(
    "gdansk_train",
    train_paths,
    np.array(train_labels),
)
```

```python
test_embeddings = extract_and_cache_embeddings(
    "gdansk_test",
    test_paths,
    np.array(test_labels),
)
```

## Verify cached files

```python
emb_dir = RESULTS_DIR / "embeddings"
for p in sorted(emb_dir.glob("*gdansk*.npz")):
    data = np.load(p)
    print(f"{p.name}: emb={data['emb'].shape}, labels={data['labels'].shape}")
```

```python
# | hide
import nbdev

nbdev.nbdev_export()
```
