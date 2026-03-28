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

# Domain Gap Analysis

Visual comparison of datasets, per-class accuracy breakdown,
and retrieval failure examples. All results use DINOv3 ViT-S/16
frozen backbone with k=1 nearest neighbor.

```python
from pathlib import Path
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from klods_syn.data import RESULTS_DIR
```

## Configuration

```python
EMB_DIR = RESULTS_DIR / "embeddings"
BACKBONE = "DINOv3_ViT-S_16"
N_SAMPLE_IMAGES = 12
N_CONFUSION_EXAMPLES = 8
```

## Load embeddings and paths

```python
def load_cached(name):
    data = np.load(EMB_DIR / f"{BACKBONE}_{name}.npz", allow_pickle=True)
    assert "paths" in data, f"{name}: cache missing paths — re-run data prep notebook"
    return data["emb"], data["labels"], data["paths"]


train_emb, train_labels, train_paths = load_cached("gdansk_train")
test_emb, test_labels, test_paths = load_cached("gdansk_test")
brick_emb, brick_labels, brick_paths = load_cached("brickognize")
paco_emb, paco_labels, paco_paths = load_cached("paco_garcia")

# Build label index
from klods_syn.data import load_class_map

class_names, class_to_idx = load_class_map()
idx_to_class = {v: k for k, v in class_to_idx.items()}

print(f"Train: {train_emb.shape[0]}, Gdańsk test: {test_emb.shape[0]}")
print(f"Brickognize: {brick_emb.shape[0]}, Paco Garcia: {paco_emb.shape[0]}")
```

## Dataset sample grids

```python
def show_sample_grid(paths, title, n=N_SAMPLE_IMAGES, ncols=4):
    """Show a grid of random sample images from a dataset."""
    rng = np.random.RandomState(42)
    indices = rng.choice(len(paths), size=min(n, len(paths)), replace=False)
    nrows = (len(indices) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
    fig.suptitle(title, fontsize=14)
    for i, ax in enumerate(axes.flat):
        if i < len(indices):
            img = Image.open(paths[indices[i]])
            ax.imshow(img)
            ax.set_title(f"{img.size[0]}×{img.size[1]}", fontsize=8)
        ax.axis("off")
    plt.tight_layout()
    plt.show()
```

```python
show_sample_grid(train_paths, "Gdańsk — controlled studio photos")
```

```python
show_sample_grid(brick_paths, "Brickognize — controlled + uncontrolled crops")
```

```python
show_sample_grid(paco_paths, "Paco Garcia — conveyor crops")
```

## k-NN prediction with retrieval info

```python
def knn_predict_with_details(train_emb, train_labels, test_emb, k=1):
    """Return predictions, similarities, and neighbor indices."""
    sims = test_emb @ train_emb.T
    top_k_idx = np.argpartition(-sims, k, axis=1)[:, :k]

    preds = []
    pred_sims = []
    for i in range(len(test_emb)):
        neighbors = top_k_idx[i]
        neighbor_sims = sims[i, neighbors]
        order = np.argsort(-neighbor_sims)
        neighbors = neighbors[order]
        neighbor_sims = neighbor_sims[order]

        neighbor_labels = train_labels[neighbors]
        counts = np.bincount(neighbor_labels)
        preds.append(counts.argmax())
        pred_sims.append(neighbor_sims[0])

    return np.array(preds), np.array(pred_sims), top_k_idx
```

```python
gdansk_preds, gdansk_sims, gdansk_nn_idx = knn_predict_with_details(
    train_emb, train_labels, test_emb
)
brick_preds, brick_sims, brick_nn_idx = knn_predict_with_details(
    train_emb, train_labels, brick_emb
)
paco_preds, paco_sims, paco_nn_idx = knn_predict_with_details(
    train_emb, train_labels, paco_emb
)

print(f"Gdańsk test accuracy: {(gdansk_preds == test_labels).mean():.3f}")
print(f"Brickognize accuracy: {(brick_preds == brick_labels).mean():.3f}")
print(f"Paco Garcia accuracy: {(paco_preds == paco_labels).mean():.3f}")
```

## Per-class accuracy

```python
def per_class_accuracy(labels, preds, idx_to_class):
    """Compute accuracy per class, return sorted DataFrame."""
    import pandas as pd

    classes = sorted(set(labels))
    rows = []
    for c in classes:
        mask = labels == c
        n = mask.sum()
        correct = (preds[mask] == c).sum()
        rows.append(
            {
                "part_id": idx_to_class[c],
                "n_samples": n,
                "correct": correct,
                "accuracy": correct / n if n > 0 else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values("accuracy")
```

```python
brick_per_class = per_class_accuracy(brick_labels, brick_preds, idx_to_class)
print("Brickognize — per-class accuracy (worst first):")
print(brick_per_class.to_string(index=False))
```

```python
paco_per_class = per_class_accuracy(paco_labels, paco_preds, idx_to_class)
print("Paco Garcia — per-class accuracy (worst first):")
print(paco_per_class.to_string(index=False))
```

## Accuracy distribution

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(brick_per_class["accuracy"], bins=20, edgecolor="black")
axes[0].set_title("Brickognize — per-class accuracy distribution")
axes[0].set_xlabel("Accuracy")
axes[0].set_ylabel("Number of classes")

axes[1].hist(paco_per_class["accuracy"], bins=20, edgecolor="black")
axes[1].set_title("Paco Garcia — per-class accuracy distribution")
axes[1].set_xlabel("Accuracy")
axes[1].set_ylabel("Number of classes")

plt.tight_layout()
plt.show()
```

## Retrieval failures — query vs nearest neighbor

```python
def show_confusion_examples(
    test_paths,
    test_labels,
    preds,
    nn_idx,
    train_paths,
    idx_to_class,
    title,
    n=N_CONFUSION_EXAMPLES,
):
    """Show misclassified examples: query image + nearest neighbor."""
    wrong = np.where(preds != test_labels)[0]
    if len(wrong) == 0:
        print(f"{title}: no errors")
        return

    rng = np.random.RandomState(42)
    sample = rng.choice(wrong, size=min(n, len(wrong)), replace=False)

    fig, axes = plt.subplots(len(sample), 2, figsize=(6, 2.5 * len(sample)))
    fig.suptitle(f"{title} — retrieval failures", fontsize=13)

    if len(sample) == 1:
        axes = axes.reshape(1, -1)

    for row, idx in enumerate(sample):
        query_img = Image.open(test_paths[idx])
        axes[row, 0].imshow(query_img)
        true_label = idx_to_class[test_labels[idx]]
        axes[row, 0].set_title(f"Query: {true_label}", fontsize=9)
        axes[row, 0].axis("off")

        nn_i = nn_idx[idx][0]
        nn_img = Image.open(train_paths[nn_i])
        axes[row, 1].imshow(nn_img)
        pred_label = idx_to_class[preds[idx]]
        axes[row, 1].set_title(f"NN: {pred_label}", fontsize=9)
        axes[row, 1].axis("off")

    plt.tight_layout()
    plt.show()
```

```python
show_confusion_examples(
    brick_paths,
    brick_labels,
    brick_preds,
    brick_nn_idx,
    train_paths,
    idx_to_class,
    "Brickognize",
)
```

```python
show_confusion_examples(
    paco_paths,
    paco_labels,
    paco_preds,
    paco_nn_idx,
    train_paths,
    idx_to_class,
    "Paco Garcia",
)
```

## Similarity score distributions

```python
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("Cosine similarity to nearest neighbor (DINOv3 frozen)")

for ax, (sims, labels, preds, title) in zip(
    axes,
    [
        (gdansk_sims, test_labels, gdansk_preds, "Gdańsk test"),
        (brick_sims, brick_labels, brick_preds, "Brickognize"),
        (paco_sims, paco_labels, paco_preds, "Paco Garcia"),
    ],
):
    correct_mask = preds == labels
    ax.hist(sims[correct_mask], bins=50, alpha=0.6, label="correct", color="green")
    ax.hist(sims[~correct_mask], bins=50, alpha=0.6, label="wrong", color="red")
    ax.set_title(title)
    ax.set_xlabel("Cosine similarity")
    ax.legend()

plt.tight_layout()
plt.show()
```

```python
# | hide
import nbdev

nbdev.nbdev_export()
```
