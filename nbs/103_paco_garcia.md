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

# Paco Garcia Dataset Preparation

Download and inspect the Paco Garcia Lego Brick Sorting dataset.
Directory-structured by class, single brick per image, conveyor
setup. Map descriptive names to part numbers, filter to classes
overlapping with Gdańsk, extract and cache embeddings.

```python
from pathlib import Path
from collections import Counter

import numpy as np

from klods_syn.data import (
    load_class_map,
    filter_to_overlap,
    extract_and_cache_embeddings,
    RESULTS_DIR,
)
```

## Configuration

```python
# Descriptive name → primary part number, with known alternates
PACO_TO_PART = {
    "Brick_1x1": "3005",
    "Brick_1x2": "3004",
    "Brick_1x3": "3622",
    "Brick_1x4": "3010",
    "Brick_2x2": "3003",
    "Brick_2x2_L": "2357",
    "Brick_2x2_Slope": "3039",
    "Brick_2x3": "3002",
    "Brick_2x4": "3001",
    "Plate_1x1": "3024",
    "Plate_1x1_Round": "6141",
    "Plate_1x1_Slope": "61252",
    "Plate_1x2": "3023",
    "Plate_1x2_Grill": "2412",
    "Plate_1x3": "3623",
    "Plate_1x4": "3710",
    "Plate_2x2": "3022",
    "Plate_2x2_L": "2420",
    "Plate_2x3": "3021",
    "Plate_2x4": "3020",
}

# Alternates to try if primary doesn't match Gdańsk
ALTERNATES = {
    "2357": ["3831"],
    "61252": ["49668"],
    "6141": ["4073"],
    "2412": ["2412b"],
}
```

## Download

```python
import kagglehub

dataset_path = Path(
    kagglehub.dataset_download("pacogarciam3/lego-brick-sorting-image-recognition")
)
print(f"Downloaded to: {dataset_path}")
```

## Use cropped images

```python
image_root = dataset_path / "Cropped Images"
assert image_root.exists(), f"Expected {image_root}"

class_dirs = sorted([d for d in image_root.iterdir() if d.is_dir()])
print(f"Classes: {len(class_dirs)}")

class_counts = {}
for d in class_dirs:
    count = len(list(d.glob("*.jpg")))
    class_counts[d.name] = count
    print(f"  {d.name}: {count}")

total = sum(class_counts.values())
counts = list(class_counts.values())
print(f"\nTotal images: {total}")
print(
    f"Images per class — min: {min(counts)}, max: {max(counts)}, "
    f"mean: {total / len(counts):.1f}"
)
```

## Sample images

```python
from PIL import Image
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 5, figsize=(15, 6))
fig.suptitle("Paco Garcia — cropped sample images")
sample_dirs = class_dirs[:10]
for ax, d in zip(axes.flat, sample_dirs):
    imgs = list(d.glob("*.jpg"))
    if imgs:
        img = Image.open(imgs[0])
        ax.imshow(img)
        ax.set_title(d.name, fontsize=8)
    ax.axis("off")
plt.tight_layout()
plt.show()
```

## Resolve part number mapping against Gdańsk

```python
gdansk_names, gdansk_to_idx = load_class_map()
gdansk_set = set(gdansk_names)

resolved = {}
unresolved = []

for desc_name, primary in PACO_TO_PART.items():
    if primary in gdansk_set:
        resolved[desc_name] = primary
    else:
        # Try alternates
        found = False
        for alt in ALTERNATES.get(primary, []):
            if alt in gdansk_set:
                resolved[desc_name] = alt
                found = True
                break
        if not found:
            # Check if Gdańsk uses compound names (e.g. "3004_3065")
            for gname in gdansk_names:
                if primary in gname.split("_"):
                    resolved[desc_name] = gname
                    found = True
                    break
        if not found:
            unresolved.append((desc_name, primary))

print(f"Resolved: {len(resolved)} / {len(PACO_TO_PART)}")
for desc, part in sorted(resolved.items()):
    print(f"  {desc} → {part}")

if unresolved:
    print(f"\nUnresolved ({len(unresolved)}):")
    for desc, primary in unresolved:
        print(f"  {desc} ({primary})")
```

## Load paths and filter to overlapping classes

```python
image_paths = []
part_labels = []

for d in class_dirs:
    desc_name = d.name
    if desc_name not in resolved:
        continue
    part_id = resolved[desc_name]
    for img_path in d.glob("*.jpg"):
        image_paths.append(str(img_path))
        part_labels.append(part_id)

print(f"Images with resolved part IDs: {len(image_paths)}")

eval_paths, eval_labels = filter_to_overlap(image_paths, part_labels, gdansk_to_idx)
print(f"Overlapping parts: {len(set(part_labels) & gdansk_set)}")
print(f"Eval images: {len(eval_paths)}")
```

## Extract and cache embeddings

```python
paco_embeddings = extract_and_cache_embeddings(
    "paco_garcia",
    eval_paths,
    np.array(eval_labels),
)
```

## Verify cached files

```python
emb_dir = RESULTS_DIR / "embeddings"
for p in sorted(emb_dir.glob("*paco*.npz")):
    data = np.load(p)
    print(f"{p.name}: emb={data['emb'].shape}, labels={data['labels'].shape}")
```
