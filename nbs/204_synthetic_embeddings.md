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

# Synthetic Embedding Extraction

Extract DINOv3 ViT-S/16 embeddings from rendered images produced by
`203_render_dataset`. Preserves view metadata (`view_idx`, `render_idx`)
alongside embeddings so the training pipeline can construct view-aware
contrastive pairs and 4-view fusion groups.

```python
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if platform.system() == "Linux":
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q",
         "git+https://github.com/restlessronin/klods-syn.git"],
        check=True,
    )
else:
    sys.path.insert(0, "..")
```

```python
from klods_syn.data import (
    LegoDataset,
    extract_embeddings,
    get_device,
    load_dinov3_vits16,
    BATCH_SIZE,
    NUM_WORKERS,
)
```

## Paths

```python
if platform.system() == "Linux":
    DATASET_DIR = Path("/kaggle/input/klods-syn-dataset")
    RESULTS_DIR = Path("/kaggle/working/results/synthetic")
else:
    DATASET_DIR = Path("../data/dataset_test")
    RESULTS_DIR = Path("../results/synthetic")
```

## Load render metadata

Read `labels.csv` produced by 203 and build a class mapping from
unique `ldraw_id` values.

```python
labels_df = pd.read_csv(DATASET_DIR / "labels.csv")
print(f"Images: {len(labels_df):,}")
print(f"Unique parts: {labels_df['ldraw_id'].nunique():,}")
print(f"Unique colors: {labels_df['color_id'].nunique():,}")
print(f"Views per render: {labels_df['view_idx'].unique().tolist()}")
```

```python
class_names = sorted(labels_df["ldraw_id"].unique())
class_to_idx = {name: i for i, name in enumerate(class_names)}
print(f"Classes: {len(class_names)}")
```

## Extract embeddings

```python
device = get_device()
model, transform, backbone_name = load_dinov3_vits16()
model = model.to(device)
print(f"Backbone: {backbone_name}, device: {device}")
```

```python
image_paths = [str(DATASET_DIR / p) for p in labels_df["image_path"]]
integer_labels = np.array([class_to_idx[lid] for lid in labels_df["ldraw_id"]])

dataset = LegoDataset(image_paths, integer_labels, transform)
loader = __import__("torch.utils.data", fromlist=["DataLoader"]).DataLoader(
    dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS
)

emb, lab = extract_embeddings(model, loader, device)
print(f"Embeddings: {emb.shape}")
```

## Save with view metadata

The extended `.npz` format includes `view_idx`, `ldraw_id`, and
`render_idx` alongside the standard `emb`, `labels`, `paths` fields.
This metadata enables view-aware contrastive training (301) and
4-view fusion grouping (304).

```python
emb_dir = RESULTS_DIR / "embeddings"
emb_dir.mkdir(parents=True, exist_ok=True)

safe_name = backbone_name.replace(" ", "_").replace("/", "_")
cache_path = emb_dir / f"{safe_name}_synthetic.npz"

np.savez(
    cache_path,
    emb=emb,
    labels=lab,
    view_idx=labels_df["view_idx"].values,
    ldraw_id=labels_df["ldraw_id"].values.astype(str),
    render_idx=labels_df["render_idx"].values,
    paths=np.array(image_paths),
)
print(f"Saved {emb.shape[0]} embeddings to {cache_path.name}")
```

## Verify

```python
data = np.load(cache_path, allow_pickle=True)
for key in ["emb", "labels", "view_idx", "ldraw_id", "render_idx", "paths"]:
    arr = data[key]
    print(f"  {key}: shape={arr.shape}, dtype={arr.dtype}")
```
