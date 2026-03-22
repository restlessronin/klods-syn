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

# Multi-Backbone k-NN Baseline

Zero-training baseline: extract embeddings from pretrained ImageNet
backbones and classify LEGO parts by nearest-neighbor search.

This measures how much off-the-shelf visual features already capture
about LEGO part geometry — before any fine-tuning or synthetic data.

```python
# | default_exp eval
```

```python
# | export
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import top_k_accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm
```

```python
from klods_syn.data import (
    LegoDataset,
    load_class_map,
    load_image_paths,
    PHOTOS_DIR,
    RANDOM_SEED,
)
```

## Configuration

```python
DEVICE = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available() else "cpu"
)
BATCH_SIZE = 64
NUM_WORKERS = 0
TEST_SPLIT = 0.3
RESULTS_DIR = Path("../results/baseline")

print(f"Device: {DEVICE}")
```

## Load dataset and split

```python
class_names, class_to_idx = load_class_map()
image_paths, labels = load_image_paths(class_to_idx=class_to_idx)
n_classes = len(class_names)
print(f"Classes: {n_classes}, Images: {len(image_paths)}")

train_paths, test_paths, train_labels, test_labels = train_test_split(
    image_paths, labels, test_size=TEST_SPLIT, random_state=RANDOM_SEED, stratify=labels
)
print(f"Train: {len(train_paths)}, Test: {len(test_paths)}")
```

## Backbone definitions

```python
# | export
from torchvision import transforms


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


BACKBONES = [load_convnext_v2_tiny, load_dinov2_vits14, load_efficientnet_b0]
```

## Embedding extraction

```python
# | export
@torch.no_grad()
def extract_embeddings(model, dataloader, device) -> tuple[np.ndarray, np.ndarray]:
    """Extract L2-normalized embeddings from a pretrained backbone."""
    embeddings = []
    all_labels = []
    for images, batch_labels in tqdm(dataloader, desc="Extracting"):
        images = images.to(device)
        features = model(images)
        features = F.normalize(features, p=2, dim=1)
        embeddings.append(features.cpu().numpy())
        all_labels.append(batch_labels.numpy())
    return np.concatenate(embeddings), np.concatenate(all_labels)
```

## k-NN evaluation

```python
# | export
def evaluate_knn(
    train_emb: np.ndarray,
    train_labels: np.ndarray,
    test_emb: np.ndarray,
    test_labels: np.ndarray,
    n_classes: int,
    k_values: tuple[int, ...] = (1, 5),
) -> dict:
    """k-NN classification: top-1 and top-5 accuracy."""
    results = {}
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k, metric="cosine", algorithm="brute")
        knn.fit(train_emb, train_labels)

        top1 = knn.score(test_emb, test_labels)
        results[f"top1_k{k}"] = top1

        if n_classes >= 5:
            proba = knn.predict_proba(test_emb)
            top5 = top_k_accuracy_score(
                test_labels, proba, k=5, labels=np.arange(n_classes)
            )
            results[f"top5_k{k}"] = top5

    return results
```

## Run evaluation

```python
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
all_results = {
    "n_classes": n_classes,
    "n_train": len(train_paths),
    "n_test": len(test_paths),
    "test_split": TEST_SPLIT,
    "random_seed": RANDOM_SEED,
    "device": DEVICE,
    "torch_version": torch.__version__,
    "backbones": {},
}

for load_fn in BACKBONES:
    model, transform, name = load_fn()
    model = model.to(DEVICE)
    print(f"\n{'=' * 50}")
    print(f"Backbone: {name}")
    print(f"{'=' * 50}")

    train_dataset = LegoDataset(train_paths, np.array(train_labels), transform)
    test_dataset = LegoDataset(test_paths, np.array(test_labels), transform)
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS
    )

    t0 = time.time()
    train_emb, train_lab = extract_embeddings(model, train_loader, DEVICE)
    test_emb, test_lab = extract_embeddings(model, test_loader, DEVICE)
    extraction_time = time.time() - t0

    print(f"Embedding extraction: {extraction_time:.1f}s")
    print(f"Embedding dim: {train_emb.shape[1]}")

    # Cache embeddings
    emb_dir = RESULTS_DIR / "embeddings"
    emb_dir.mkdir(parents=True, exist_ok=True)
    safe_name = name.replace(" ", "_").replace("/", "_")
    np.savez(
        emb_dir / f"{safe_name}.npz",
        train_emb=train_emb,
        train_lab=train_lab,
        test_emb=test_emb,
        test_lab=test_lab,
    )
    results = evaluate_knn(train_emb, train_lab, test_emb, test_lab, n_classes)
    results["embedding_dim"] = int(train_emb.shape[1])
    results["extraction_time_s"] = round(extraction_time, 1)

    for metric, value in sorted(results.items()):
        if isinstance(value, float):
            print(f"  {metric}: {value:.4f}")

    all_results["backbones"][name] = results

    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
```

## Results

```python
# Save
results_path = RESULTS_DIR / "knn_baseline.json"
with open(results_path, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"Results saved to {results_path}")
```

```python
# Summary table
header = f"{'Backbone':<25} {'Top-1 (k=1)':<15} {'Top-5 (k=1)':<15} {'Top-1 (k=5)':<15} {'Top-5 (k=5)':<15}"
print(header)
print("-" * len(header))
for name, res in all_results["backbones"].items():
    print(
        f"{name:<25} "
        f"{res.get('top1_k1', 0):<15.4f} "
        f"{res.get('top5_k1', 0):<15.4f} "
        f"{res.get('top1_k5', 0):<15.4f} "
        f"{res.get('top5_k5', 0):<15.4f}"
    )
```

## Threshold analysis

Rejection threshold for zero-false-positive operation: accept a
classification only if the nearest-neighbor cosine distance is below
a threshold. Rejected bricks recirculate for another pass.
```python
from sklearn.metrics import pairwise_distances


def load_embeddings(name: str, results_dir: Path = RESULTS_DIR) -> dict:
    """Load cached embeddings for a backbone."""
    safe_name = name.replace(" ", "_").replace("/", "_")
    path = results_dir / "embeddings" / f"{safe_name}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"No cached embeddings at {path}. Run extraction first."
        )
    data = np.load(path)
    return {k: data[k] for k in data.files}
```
```python
name = "DINOv2 ViT-S/14"
emb = load_embeddings(name)
train_emb, train_lab = emb["train_emb"], emb["train_lab"]
test_emb, test_lab = emb["test_emb"], emb["test_lab"]

dists = pairwise_distances(test_emb, train_emb, metric="cosine")
nn_indices = dists.argmin(axis=1)
nn_dists = dists[np.arange(len(test_emb)), nn_indices]
nn_preds = train_lab[nn_indices]
correct = nn_preds == test_lab

print(
    f"Correct   — mean dist: {nn_dists[correct].mean():.4f}, "
    f"95th pct: {np.percentile(nn_dists[correct], 95):.4f}"
)
print(
    f"Incorrect — mean dist: {nn_dists[~correct].mean():.4f}, "
    f"5th pct: {np.percentile(nn_dists[~correct], 5):.4f}"
)
```
```python
print(f"\n{'Threshold':<12} {'Accepted':<12} {'Accuracy':<12} {'Rejected %':<12}")
print("-" * 48)
for t in np.arange(0.05, 0.50, 0.05):
    accepted = nn_dists <= t
    if accepted.sum() == 0:
        continue
    acc = (nn_preds[accepted] == test_lab[accepted]).mean()
    rej = 1 - accepted.mean()
    print(f"{t:<12.2f} {accepted.sum():<12} {acc:<12.4f} {rej:<12.2%}")
```

```python
# | hide
import nbdev

nbdev.nbdev_export()
```
