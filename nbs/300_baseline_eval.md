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

# Cross-Domain k-NN Evaluation

Load cached embeddings from all datasets, train k-NN on Gdańsk
train split, evaluate on Gdańsk test, Brickognize, and Paco Garcia.
Produce accuracy tables and per-backbone comparisons.

```python
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, top_k_accuracy_score

from klods_syn.data import RESULTS_DIR, BACKBONES
```

## Configuration

```python
EMB_DIR = RESULTS_DIR / "embeddings"
K_VALUES = [1, 3, 5]
from klods_syn.data import BACKBONES

# Dynamically get backbone names from the loader functions
BACKBONE_NAMES = [load_fn()[2] for load_fn in BACKBONES]  # index 2 = name string
EVAL_DATASETS = ["gdansk_test", "brickognize", "paco_garcia"]
```

## Load all cached embeddings

```python
def load_embeddings(emb_dir: Path, backbone: str, dataset: str) -> dict | None:
    """Load cached embeddings, return None if not found."""
    safe_name = backbone.replace(" ", "_").replace("/", "_")
    path = emb_dir / f"{safe_name}_{dataset}.npz"
    if not path.exists():
        print(f"  Missing: {path.name}")
        return None
    data = np.load(path)
    return {"emb": data["emb"], "labels": data["labels"]}


train_cache = {}
eval_cache = {}

for bb in BACKBONE_NAMES:
    train_data = load_embeddings(EMB_DIR, bb, "gdansk_train")
    if train_data is not None:
        train_cache[bb] = train_data
        print(f"{bb} train: {train_data['emb'].shape}")

    for ds in EVAL_DATASETS:
        data = load_embeddings(EMB_DIR, bb, ds)
        if data is not None:
            eval_cache[(bb, ds)] = data
            print(f"{bb} {ds}: {data['emb'].shape}")
```

## k-NN classifier

```python
def knn_predict(
    train_emb: np.ndarray,
    train_labels: np.ndarray,
    test_emb: np.ndarray,
    k: int,
) -> np.ndarray:
    """Predict labels via k-NN with cosine similarity.

    Embeddings are assumed L2-normalized (dot product = cosine sim).
    """
    # (n_test, n_train)
    sims = test_emb @ train_emb.T
    top_k_idx = np.argpartition(-sims, k, axis=1)[:, :k]

    preds = []
    for i in range(len(test_emb)):
        neighbors = top_k_idx[i]
        neighbor_labels = train_labels[neighbors]
        counts = np.bincount(neighbor_labels)
        preds.append(counts.argmax())
    return np.array(preds)


def knn_top_k_scores(
    train_emb: np.ndarray,
    train_labels: np.ndarray,
    test_emb: np.ndarray,
    test_labels: np.ndarray,
    k_values: list[int],
) -> dict[str, float]:
    """Compute top-1 accuracy for each k, plus top-5 retrieval accuracy."""
    sims = test_emb @ train_emb.T
    results = {}

    for k in k_values:
        preds = knn_predict(train_emb, train_labels, test_emb, k)
        results[f"k={k} top-1"] = accuracy_score(test_labels, preds)

    # Top-5 retrieval: is correct label among 5 nearest neighbors?
    top5_idx = np.argpartition(-sims, 5, axis=1)[:, :5]
    top5_correct = 0
    for i in range(len(test_emb)):
        if test_labels[i] in train_labels[top5_idx[i]]:
            top5_correct += 1
    results["top-5 retrieval"] = top5_correct / len(test_emb)

    return results
```

## Run evaluation

```python
RESULTS_CSV = RESULTS_DIR / "cross_domain_knn_results.csv"

existing_df = None
if RESULTS_CSV.exists():
    try:
        existing_df = pd.read_csv(RESULTS_CSV)
        print(
            f"Loaded {len(existing_df)} existing evaluation results from {RESULTS_CSV.name}"
        )
    except Exception as e:
        print(
            f"Warning: Could not load {RESULTS_CSV.name}: {e}. Computing all from scratch."
        )
        existing_df = None

computed = set()
all_results = []

if existing_df is not None:
    all_results = existing_df.to_dict("records")
    for _, row in existing_df.iterrows():
        computed.add((row["backbone"], row["eval_dataset"], row["metric"]))

for bb in BACKBONE_NAMES:
    if bb not in train_cache:
        print(f"Skipping {bb} — no training embeddings available")
        continue
    train_emb = train_cache[bb]["emb"]
    train_labels = train_cache[bb]["labels"]

    for ds in EVAL_DATASETS:
        key = (bb, ds)
        if key not in eval_cache:
            print(f"Skipping {bb} / {ds} — no eval embeddings")
            continue
        test_emb = eval_cache[key]["emb"]
        test_labels = eval_cache[key]["labels"]

        missing_metrics = []
        for metric in ["k=1 top-1", "k=3 top-1", "k=5 top-1", "top-5 retrieval"]:
            if (bb, ds, metric) not in computed:
                missing_metrics.append(metric)
        if not missing_metrics:
            print(f"Skipping already computed: {bb} / {ds}")
            continue
        print(
            f"Computing missing metrics for {bb} / {ds}: {', '.join(missing_metrics)}"
        )
        scores = knn_top_k_scores(
            train_emb, train_labels, test_emb, test_labels, K_VALUES
        )
        for metric, value in scores.items():
            if (bb, ds, metric) not in computed:
                all_results.append(
                    {
                        "backbone": bb,
                        "eval_dataset": ds,
                        "metric": metric,
                        "accuracy": value,
                    }
                )
                computed.add((bb, ds, metric))

results_df = pd.DataFrame(all_results)
```

## Results table

```python
pivot = results_df.pivot_table(
    index=["backbone", "metric"],
    columns="eval_dataset",
    values="accuracy",
)
pivot = pivot[EVAL_DATASETS]  # consistent column order

print("\n" + "=" * 70)
print("Cross-Domain k-NN Evaluation (train: Gdańsk train split)")
print("=" * 70)
print(pivot.to_string(float_format="{:.3f}".format))
```

## Accuracy comparison chart

```python
fig, axes = plt.subplots(1, len(K_VALUES), figsize=(5 * len(K_VALUES), 5))
fig.suptitle("k-NN Top-1 Accuracy — Gdańsk Train → Cross-Domain Eval")

for ax, k in zip(axes, K_VALUES):
    metric = f"k={k} top-1"
    subset = results_df[results_df["metric"] == metric]
    pivot_k = subset.pivot(index="backbone", columns="eval_dataset", values="accuracy")
    pivot_k = pivot_k[EVAL_DATASETS]
    pivot_k.plot(kind="bar", ax=ax, rot=15)
    ax.set_title(metric)
    ax.set_ylabel("Top-1 Accuracy")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7)

plt.tight_layout()
plt.show()
```

## Domain gap summary

```python
print("\nDomain gap (drop from Gdańsk test → cross-domain):")
print("-" * 50)
for bb in BACKBONE_NAMES:
    subset = results_df[
        (results_df["backbone"] == bb) & (results_df["metric"] == "k=1 top-1")
    ]
    gdansk_acc = subset[subset["eval_dataset"] == "gdansk_test"]["accuracy"].values
    if len(gdansk_acc) == 0:
        continue
    gdansk_acc = gdansk_acc[0]
    for ds in EVAL_DATASETS[1:]:
        ds_acc = subset[subset["eval_dataset"] == ds]["accuracy"].values
        if len(ds_acc) == 0:
            continue
        drop = gdansk_acc - ds_acc[0]
        print(f"  {bb}: {ds} drop = {drop:+.3f} ({gdansk_acc:.3f} → {ds_acc[0]:.3f})")
```

## Save results

```python
results_path = RESULTS_DIR / "cross_domain_knn_results.csv"
results_df.to_csv(results_path, index=False)
print(f"Saved to {results_path}")
```
