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

Core utilities for dataset loading, COCO annotation processing,
embedding extraction, and caching. Shared across all data prep
notebooks.

```python
# | default_exp data
```

```python
# | export
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
```

## Configuration

```python
# | export
DATA_DIR = Path("../data/gdansk")
PHOTOS_DIR = DATA_DIR / "photos"
MIN_IMAGES_PER_CLASS = 5
RANDOM_SEED = 42
RESULTS_DIR = Path("../results/baseline")
BATCH_SIZE = 64
NUM_WORKERS = 0
```

## Download and extract

```python
# | export
def download_and_extract(url: str, zip_path: Path, extract_dir: Path) -> Path:
    """Idempotent download and extract with completion marker."""
    import urllib.request
    import zipfile

    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if not zip_path.exists():
        print(f"Downloading to {zip_path}...")
        urllib.request.urlretrieve(url, zip_path)
        print(f"Done — {zip_path.stat().st_size / 1e6:.1f} MB")
    else:
        print(f"Already downloaded — {zip_path.stat().st_size / 1e6:.1f} MB")

    marker = extract_dir / ".extract_done"
    if not marker.exists():
        import shutil

        if extract_dir.exists():
            shutil.rmtree(extract_dir)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_dir)
            print(f"Extracted {len(zf.namelist())} files")
        marker.touch()
    else:
        print(f"Already extracted to {extract_dir}")

    return extract_dir
```

## COCO annotation utilities

```python
# | export
def inspect_coco(json_path: Path) -> dict:
    """Print summary of a COCO annotation file and return parsed data."""
    import json

    with open(json_path) as f:
        coco = json.load(f)

    print(f"\n{'=' * 50}")
    print(f"File: {json_path.name}")
    print(f"{'=' * 50}")
    print(f"Images: {len(coco.get('images', []))}")
    print(f"Annotations: {len(coco.get('annotations', []))}")

    categories = coco.get("categories", [])
    print(f"Categories: {len(categories)}")

    for cat in sorted(categories, key=lambda c: c["id"])[:10]:
        print(f"  id={cat['id']}: {cat.get('name', 'unnamed')}")
    if len(categories) > 10:
        print(f"  ... and {len(categories) - 10} more")

    anns = coco.get("annotations", [])
    if anns:
        has_bbox = sum(1 for a in anns if "bbox" in a and len(a["bbox"]) == 4)
        has_seg = sum(1 for a in anns if "segmentation" in a and a["segmentation"])
        print(f"With bounding box: {has_bbox}")
        print(f"With segmentation: {has_seg}")

        cat_counts = Counter(a["category_id"] for a in anns)
        cat_name = {c["id"]: c.get("name", str(c["id"])) for c in categories}
        print(f"\nInstances per category (top 10):")
        for cat_id, count in cat_counts.most_common(10):
            print(f"  {cat_name.get(cat_id, cat_id)}: {count}")
        print(f"Min instances: {min(cat_counts.values())}")
        print(f"Max instances: {max(cat_counts.values())}")

    return coco
```

```python
# | export
def crop_coco_annotations(
    coco: dict,
    images_dir: Path,
    output_dir: Path,
    label: str,
    min_size: int = 10,
) -> tuple[list[str], list[str]]:
    """Crop bounding boxes from COCO annotations, save as individual images.

    Category names are expected as `{part_id}_{color_id}`. The color
    suffix is stripped to produce part-level labels.

    Returns (image_paths, part_id_labels).
    """
    cat_name = {c["id"]: c["name"] for c in coco["categories"]}
    img_info = {i["id"]: i for i in coco["images"]}

    paths = []
    labels = []
    skipped = 0

    for ann in coco["annotations"]:
        img = img_info[ann["image_id"]]
        img_path = images_dir / img["file_name"]
        if not img_path.exists():
            skipped += 1
            continue

        x, y, w, h = ann["bbox"]
        if w < min_size or h < min_size:
            skipped += 1
            continue

        cat = cat_name[ann["category_id"]]
        part_id = cat.rsplit("_", 1)[0]

        crop_subdir = output_dir / label / part_id
        crop_subdir.mkdir(parents=True, exist_ok=True)

        crop_path = crop_subdir / f"{img['id']}_{ann['id']}.png"
        if not crop_path.exists():
            pil_img = Image.open(img_path)
            crop = pil_img.crop((x, y, x + w, y + h))
            crop.save(crop_path)

        paths.append(str(crop_path))
        labels.append(part_id)

    print(f"{label}: {len(paths)} crops, {skipped} skipped")
    return paths, labels
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

## Gdańsk class mapping

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

## Filter paths to overlapping classes

```python
# | export
def filter_to_overlap(
    paths: list[str],
    part_labels: list[str],
    class_to_idx: dict[str, int],
) -> tuple[list[str], list[int]]:
    """Keep only paths whose part ID exists in class_to_idx.

    Takes string part labels (from crop functions), returns integer
    labels compatible with cached Gdańsk embeddings.
    """
    filtered_paths = []
    filtered_labels = []
    for p, l in zip(paths, part_labels):
        if l in class_to_idx:
            filtered_paths.append(p)
            filtered_labels.append(class_to_idx[l])
    return filtered_paths, filtered_labels
```

## Backbone definitions

```python
# | export
def load_convnext_v2_tiny():
    """ConvNeXt V2 Tiny pretrained on ImageNet-22k → ImageNet-1k."""
    from timm import create_model

    model = create_model(
        "convnextv2_tiny.fcmae_ft_in22k_in1k", pretrained=True, num_classes=0
    )
    model.eval()
    transform = transforms.Compose(
        [
            transforms.Resize(288, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(288),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return model, transform, "ConvNeXt V2 Tiny"


def load_dinov2_vits14():
    """DINOv2 ViT-S/14 — self-supervised, strong k-NN features."""
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    model.eval()
    transform = transforms.Compose(
        [
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return model, transform, "DINOv2 ViT-S/14"


def load_efficientnet_b0():
    """EfficientNet-B0 — lightweight baseline."""
    from timm import create_model

    model = create_model("efficientnet_b0", pretrained=True, num_classes=0)
    model.eval()
    transform = transforms.Compose(
        [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return model, transform, "EfficientNet-B0"


# | export
def load_dinov3_vits16():
    """DINOv3 ViT-S/16 — latest self-supervised from Meta (2025)."""
    from transformers import AutoImageProcessor, AutoModel

    model_id = (
        "facebook/dinov3-vits16-pretrain-lvd1689m"  # or -base / -large as desired
    )
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id)
    model.eval()

    def transform(img):
        # Follow processor logic but match our existing pipeline style
        inputs = processor(img, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)  # remove batch dim
        return pixel_values  # already normalized by processor

    return model, transform, "DINOv3 ViT-S/16"


BACKBONES = [
    load_convnext_v2_tiny,
    load_dinov2_vits14,
    load_efficientnet_b0,
    load_dinov3_vits16,
]
```

## Embedding extraction and caching

```python
# | export
@torch.no_grad()
def extract_embeddings(model, dataloader, device) -> tuple[np.ndarray, np.ndarray]:
    """Extract L2-normalized embeddings from a pretrained backbone.

    Handles both plain tensor outputs (ConvNeXt, DINOv2, EfficientNet)
    and Transformers-style dataclass outputs (DINOv3).
    """
    embeddings = []
    all_labels = []
    for images, batch_labels in tqdm(dataloader, desc="Extracting"):
        images = images.to(device)
        outputs = model(images)
        if isinstance(outputs, torch.Tensor):
            features = outputs
        else:
            # Assume Transformers-style output (BaseModelOutputWithPooling or similar)
            # Prefer pooler_output if present; fallback to CLS token
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                features = outputs.pooler_output
            elif hasattr(outputs, "last_hidden_state"):
                features = outputs.last_hidden_state[:, 0, :]  # CLS token
            else:
                raise ValueError(
                    f"Unexpected model output type: {type(outputs)}. Cannot extract features."
                )
        features = F.normalize(features, p=2, dim=1)
        embeddings.append(features.cpu().numpy())
        all_labels.append(batch_labels.numpy())
    return np.concatenate(embeddings), np.concatenate(all_labels)
```

```python
# | export
def get_device() -> str:
    """Return best available torch device."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
```

```python
# | export
def extract_and_cache_embeddings(
    dataset_name: str,
    image_paths: list[str],
    labels: np.ndarray,
    results_dir: Path = RESULTS_DIR,
    backbones: list = None,
    device: str | None = None,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
) -> dict[str, dict]:
    """Extract embeddings for all backbones and cache to disk.

    Saves to results_dir/embeddings/{backbone}_{dataset_name}.npz.
    Skips extraction if cache already exists.

    Returns dict mapping backbone name to {emb, labels, paths} arrays.
    """
    if backbones is None:
        backbones = BACKBONES
    if device is None:
        device = get_device()

    emb_dir = results_dir / "embeddings"
    emb_dir.mkdir(parents=True, exist_ok=True)

    all_embeddings = {}
    paths_array = np.array(image_paths)

    for load_fn in backbones:
        model, transform, name = load_fn()
        safe_name = name.replace(" ", "_").replace("/", "_")
        cache_path = emb_dir / f"{safe_name}_{dataset_name}.npz"

        if cache_path.exists():
            print(f"{name}: loading cached embeddings from {cache_path.name}")
            cached = np.load(cache_path, allow_pickle=True)
            all_embeddings[name] = {
                "emb": cached["emb"],
                "labels": cached["labels"],
                "paths": cached["paths"],
            }
            continue

        print(f"\n{'=' * 50}")
        print(f"Backbone: {name}")
        print(f"{'=' * 50}")

        model = model.to(device)
        dataset = LegoDataset(image_paths, labels, transform)
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
        emb, lab = extract_embeddings(model, loader, device)

        np.savez(cache_path, emb=emb, labels=lab, paths=paths_array)
        print(
            f"Cached {emb.shape[0]} embeddings (dim={emb.shape[1]}) to {cache_path.name}"
        )

        all_embeddings[name] = {"emb": emb, "labels": lab, "paths": paths_array}

        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    return all_embeddings
```

## Quick test

```python
class_names, class_to_idx = load_class_map()
image_paths, labels = load_image_paths(class_to_idx=class_to_idx)
print(f"Loaded {len(image_paths)} images across {len(class_names)} classes")
```

```python
# | hide
import nbdev

nbdev.nbdev_export()
```
