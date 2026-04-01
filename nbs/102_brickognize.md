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

# Brickognize Dataset Preparation

Download and inspect the Brickognize test set. Crop detection
annotations to individual brick images, filter to classes
overlapping with Gdańsk, extract and cache embeddings.

```python
from pathlib import Path
import json

import numpy as np

from klods_syn.data import (
    download_and_extract,
    inspect_coco,
    crop_coco_annotations,
    load_class_map,
    filter_to_overlap,
    extract_and_cache_embeddings,
)
```

## Configuration

```python
DATA_DIR = Path("../data/brickognize")
ZIP_PATH = DATA_DIR / "brickognize_dataset.zip"
URL = "https://www.tramacsoft.com/wp-content/uploads/2022/12/brickognize_dataset.zip"
CROP_DIR = DATA_DIR / "crops"
```

## Download and extract

```python
extract_dir = download_and_extract(URL, ZIP_PATH, DATA_DIR / "extracted")
```

## Inspect structure

```python
for p in sorted(extract_dir.rglob("*"))[:30]:
    rel = p.relative_to(extract_dir)
    prefix = "  " * (len(rel.parts) - 1)
    size = f" ({p.stat().st_size / 1e3:.1f} KB)" if p.is_file() else ""
    print(f"{prefix}{p.name}{size}")
```

## Inspect COCO annotations

```python
json_files = sorted(p for p in extract_dir.rglob("*.json") if "__MACOSX" not in str(p))
print(f"Found {len(json_files)} annotation file(s):")
for jf in json_files:
    print(f"  {jf.relative_to(extract_dir)}")

cocos = {}
for jf in json_files:
    cocos[jf.stem] = inspect_coco(jf)
```

## Inspect category ID format

Category names use `{part_number}_{color_id}` format. We strip
the color suffix to match against Gdańsk class names.

```python
for name, coco in cocos.items():
    categories = coco.get("categories", [])
    cat_names = sorted(c.get("name", str(c["id"])) for c in categories)
    print(f"\n{name} — all {len(cat_names)} category names:")
    for cn in cat_names:
        print(f"  {cn}")
```

## Check overlap with Gdańsk

```python
gdansk_names, gdansk_to_idx = load_class_map()
gdansk_set = set(gdansk_names)

brickognize_parts = set(cat.rsplit("_", 1)[0] for cat in cat_names)
overlap = gdansk_set & brickognize_parts

print(f"Gdańsk classes: {len(gdansk_set)}")
print(f"Brickognize parts (ignoring color): {len(brickognize_parts)}")
print(f"Overlap: {len(overlap)}")
if overlap:
    for p in sorted(overlap):
        print(f"  {p}")
```

## Sample images

```python
from PIL import Image
import matplotlib.pyplot as plt

for name, coco in cocos.items():
    images = coco.get("images", [])
    anns = coco.get("annotations", [])
    cat_name = {c["id"]: c.get("name", str(c["id"])) for c in coco["categories"]}

    anns_by_img = {}
    for a in anns:
        anns_by_img.setdefault(a["image_id"], []).append(a)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Brickognize — {name}")
    for ax, img_info in zip(axes, images[:3]):
        img_path = extract_dir / img_info["file_name"]
        if not img_path.exists():
            for jf in json_files:
                candidate = jf.parent / img_info["file_name"]
                if candidate.exists():
                    img_path = candidate
                    break
        if img_path.exists():
            img = Image.open(img_path)
            ax.imshow(img)
            img_anns = anns_by_img.get(img_info["id"], [])
            for a in img_anns:
                if "bbox" in a:
                    x, y, w, h = a["bbox"]
                    rect = plt.Rectangle(
                        (x, y),
                        w,
                        h,
                        linewidth=1,
                        edgecolor="lime",
                        facecolor="none",
                    )
                    ax.add_patch(rect)
                    ax.text(
                        x,
                        y - 2,
                        cat_name.get(a["category_id"], "?"),
                        fontsize=6,
                        color="lime",
                        bbox=dict(boxstyle="round,pad=0.1", fc="black", alpha=0.7),
                    )
        ax.axis("off")
    plt.tight_layout()
    plt.show()
```

## Crop annotations

```python
CROP_DIR.mkdir(parents=True, exist_ok=True)

all_crop_paths = []
all_crop_labels = []

for subset in ("controlled", "uncontrolled"):
    json_path = extract_dir / "brickognize_dataset" / subset / "classification.json"
    images_dir = json_path.parent
    coco = json.load(open(json_path))
    paths, labels = crop_coco_annotations(coco, images_dir, CROP_DIR, subset)
    all_crop_paths.extend(paths)
    all_crop_labels.extend(labels)

print(f"\nTotal crops: {len(all_crop_paths)}")
print(f"Unique parts: {len(set(all_crop_labels))}")
```

## Filter to overlapping classes

```python
eval_paths, eval_labels = filter_to_overlap(
    all_crop_paths, all_crop_labels, gdansk_to_idx
)
overlap_parts = sorted(set(all_crop_labels) & gdansk_set)
print(f"Overlapping parts: {len(overlap_parts)}")
print(f"Eval crops: {len(eval_paths)}")
```

## Extract and cache embeddings

```python
brickognize_embeddings = extract_and_cache_embeddings(
    "brickognize",
    eval_paths,
    np.array(eval_labels),
)
```

## Verify cached files

```python
from klods_syn.data import RESULTS_DIR

emb_dir = RESULTS_DIR / "embeddings"
for p in sorted(emb_dir.glob("*brickognize*.npz")):
    data = np.load(p)
    print(f"{p.name}: emb={data['emb'].shape}, labels={data['labels'].shape}")
```
