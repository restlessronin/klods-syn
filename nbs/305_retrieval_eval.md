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

# Retrieval Evaluation

Two-tier evaluation against a synthetic reference database:

**Tier 1 — Single-view cross-domain diagnostic:** Project Brickognize
and Paco Garcia embeddings through the trained projection head, k-NN
against single-view synthetic references. Measures how well the
view-aware SupCon training closed the domain gap.

**Tier 2 — 4-view synthetic retrieval (target metric):** Hold out
synthetic renders by `render_idx`, fuse held-out 4-view sets into
queries, k-NN against a fused reference database.

```python
# | default_exp retrieval
```

```python
# | export
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from collections import defaultdict
```

```python
import pandas as pd
```

## Configuration

```python
RESULTS_DIR = Path("../results/synthetic")
BASELINE_DIR = Path("../results/baseline")
EMB_DIR = RESULTS_DIR / "embeddings"
METRIC_DIR = RESULTS_DIR / "metric"
FUSION_DIR = RESULTS_DIR / "fusion"
RETRIEVAL_DIR = RESULTS_DIR / "retrieval"
RETRIEVAL_DIR.mkdir(parents=True, exist_ok=True)

BACKBONE = "DINOv3_ViT-S_16"
INPUT_DIM = 384
PROJ_DIM = 128
K_VALUES = [1, 3, 5]
HOLDOUT_RATIO = 0.2
RANDOM_SEED = 42
```

## k-NN utilities

```python
# | export
def knn_predict(
    ref_emb: np.ndarray,
    ref_labels: np.ndarray,
    query_emb: np.ndarray,
    k: int,
) -> np.ndarray:
    """Predict labels via k-NN majority vote on cosine similarity."""
    sims = query_emb @ ref_emb.T
    top_k_idx = np.argpartition(-sims, k, axis=1)[:, :k]
    preds = []
    for i in range(len(query_emb)):
        neighbor_labels = ref_labels[top_k_idx[i]]
        counts = np.bincount(neighbor_labels)
        preds.append(counts.argmax())
    return np.array(preds)


def knn_eval(
    ref_emb: np.ndarray,
    ref_labels: np.ndarray,
    query_emb: np.ndarray,
    query_labels: np.ndarray,
    k_values: list[int],
) -> dict[str, float]:
    """Compute top-1 accuracy at each k and top-5 retrieval accuracy."""
    results = {}
    sims = query_emb @ ref_emb.T
    for k in k_values:
        preds = knn_predict(ref_emb, ref_labels, query_emb, k)
        results[f"k={k} top-1"] = float((preds == query_labels).mean())
    top5_idx = np.argpartition(-sims, 5, axis=1)[:, :5]
    top5_correct = sum(
        1 for i in range(len(query_emb))
        if query_labels[i] in ref_labels[top5_idx[i]]
    )
    results["top-5 retrieval"] = top5_correct / len(query_emb)
    return results
```

## Reference database construction

```python
# | export
def build_reference_db(
    projected_emb: np.ndarray,
    labels: np.ndarray,
    view_idx: np.ndarray,
    ldraw_id: np.ndarray,
    render_idx: np.ndarray,
    fusion_layer: torch.nn.Module,
    n_views: int = 4,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Build a fused 4-view reference database.

    Groups embeddings by (ldraw_id, render_idx), fuses each complete
    group through the fusion layer, then averages across render_idx
    to produce one reference embedding per part.

    Returns (ref_emb, ref_labels) where ref_labels are integer class IDs.
    """
    groups: dict[tuple[str, int], dict[int, int]] = defaultdict(dict)
    for i, (lid, rid, vid) in enumerate(zip(ldraw_id, render_idx, view_idx)):
        groups[(str(lid), int(rid))][int(vid)] = i

    fusion_layer.eval()
    fused_by_part: dict[str, list[np.ndarray]] = defaultdict(list)

    with torch.no_grad():
        for (lid, rid), view_map in groups.items():
            if len(view_map) < n_views:
                continue
            indices = [view_map[v] for v in range(n_views)]
            views = torch.from_numpy(projected_emb[indices]).float().unsqueeze(0).to(device)
            fused = fusion_layer(views).cpu().numpy().squeeze(0)
            fused_by_part[lid].append(fused)

    ref_embs = []
    ref_lids = []
    label_map = {str(lid): int(labels[i]) for i, lid in enumerate(ldraw_id)}

    for lid, embs in fused_by_part.items():
        avg = np.mean(embs, axis=0)
        avg = avg / (np.linalg.norm(avg) + 1e-12)
        ref_embs.append(avg)
        ref_lids.append(label_map[lid])

    return np.array(ref_embs), np.array(ref_lids)
```

## Load embeddings and models

```python
from klods_syn.data import get_device
from klods_syn.metric import ProjectionHead
from klods_syn.fusion import FusionLayer

device = get_device()

data = np.load(EMB_DIR / f"{BACKBONE}_synthetic.npz", allow_pickle=True)
emb = data["emb"]
labels = data["labels"]
view_idx = data["view_idx"]
ldraw_id = data["ldraw_id"]
render_idx = data["render_idx"]
print(f"Loaded {emb.shape[0]} synthetic embeddings")
```

```python
head = ProjectionHead(INPUT_DIM, PROJ_DIM)
head.load_state_dict(torch.load(METRIC_DIR / "projection_head.pt", weights_only=True))
head = head.to(device)
head.eval()

fusion = FusionLayer(PROJ_DIM, mode="attention")
fusion_path = FUSION_DIR / "fusion_layer.pt"
if fusion_path.exists():
    fusion.load_state_dict(torch.load(fusion_path, weights_only=True))
fusion = fusion.to(device)
fusion.eval()
print("Loaded projection head and fusion layer")
```

Project all synthetic embeddings.

```python
with torch.no_grad():
    t = torch.from_numpy(emb).float().to(device)
    chunks = []
    for i in range(0, len(t), 1024):
        chunks.append(head(t[i : i + 1024]).cpu().numpy())
    projected = np.concatenate(chunks)
print(f"Projected: {projected.shape}")
```

## Train/holdout split by render_idx

Use most render indices for the reference database, hold out some
for query evaluation.

```python
unique_ridx = np.unique(render_idx)

if len(unique_ridx) < 2:
    print(f"Only {len(unique_ridx)} render index — using all data for both ref and query")
    ref_mask = np.ones(len(render_idx), dtype=bool)
    query_mask = ref_mask
else:
    rng = np.random.default_rng(RANDOM_SEED)
    rng.shuffle(unique_ridx)
    n_holdout = max(1, int(len(unique_ridx) * HOLDOUT_RATIO))
    holdout_set = set(unique_ridx[:n_holdout])
    ref_mask = np.array([int(r) not in holdout_set for r in render_idx])
    query_mask = ~ref_mask

print(f"Reference renders: {sum(ref_mask):,}, holdout queries: {sum(query_mask):,}")
```

## Build reference database

```python
ref_emb, ref_labels = build_reference_db(
    projected[ref_mask], labels[ref_mask], view_idx[ref_mask],
    ldraw_id[ref_mask], render_idx[ref_mask],
    fusion, device=device,
)
print(f"Reference DB: {ref_emb.shape[0]} parts, dim={ref_emb.shape[1]}")

np.savez(RETRIEVAL_DIR / "reference_db.npz", emb=ref_emb, labels=ref_labels)
```

## Tier 2 — 4-view synthetic retrieval

Fuse held-out 4-view groups into query embeddings and run k-NN
against the reference database.

```python
query_proj = projected[query_mask]
query_labels_arr = labels[query_mask]
query_view = view_idx[query_mask]
query_lid = ldraw_id[query_mask]
query_ridx = render_idx[query_mask]

groups: dict[tuple[str, int], dict[int, int]] = defaultdict(dict)
for i, (lid, rid, vid) in enumerate(zip(query_lid, query_ridx, query_view)):
    groups[(str(lid), int(rid))][int(vid)] = i

query_fused = []
query_fused_labels = []
label_map = {str(lid): int(lab) for lid, lab in zip(query_lid, query_labels_arr)}

fusion.eval()
with torch.no_grad():
    for (lid, rid), view_map in groups.items():
        if len(view_map) < 4:
            continue
        indices = [view_map[v] for v in range(4)]
        views = torch.from_numpy(query_proj[indices]).float().unsqueeze(0).to(device)
        fused = fusion(views).cpu().numpy().squeeze(0)
        query_fused.append(fused)
        query_fused_labels.append(label_map[lid])

query_fused = np.array(query_fused)
query_fused_labels = np.array(query_fused_labels)
print(f"Fused queries: {len(query_fused)}")
```

```python
tier2_results = knn_eval(ref_emb, ref_labels, query_fused, query_fused_labels, K_VALUES)
print("\nTier 2 — 4-view fused retrieval:")
for metric, value in tier2_results.items():
    print(f"  {metric}: {value:.4f}")
```

## Tier 1 — Single-view cross-domain diagnostic

Project real-photo embeddings (Brickognize, Paco Garcia) through the
trained head and evaluate against single-view synthetic references.

```python
baseline_emb_dir = BASELINE_DIR / "embeddings"

ref_single = projected[ref_mask]
ref_single_labels = labels[ref_mask]

all_results = []

for tier, results_dict in [("tier2_4view", tier2_results)]:
    for metric, value in results_dict.items():
        all_results.append({"tier": tier, "dataset": "synthetic_holdout", "metric": metric, "accuracy": value})

for ds_name in ["brickognize", "paco_garcia"]:
    ds_path = baseline_emb_dir / f"{BACKBONE}_{ds_name}.npz"
    if not ds_path.exists():
        print(f"  Skipping {ds_name} — {ds_path.name} not found")
        continue

    ds_data = np.load(ds_path, allow_pickle=True)
    ds_emb = ds_data["emb"]
    ds_labels = ds_data["labels"]

    with torch.no_grad():
        t = torch.from_numpy(ds_emb).float().to(device)
        chunks = []
        for i in range(0, len(t), 1024):
            chunks.append(head(t[i : i + 1024]).cpu().numpy())
        ds_projected = np.concatenate(chunks)

    scores = knn_eval(ref_single, ref_single_labels, ds_projected, ds_labels, K_VALUES)
    print(f"\nTier 1 — {ds_name}:")
    for metric, value in scores.items():
        print(f"  {metric}: {value:.4f}")
        all_results.append({"tier": "tier1_single_view", "dataset": ds_name, "metric": metric, "accuracy": value})
```

## Results summary

```python
results_df = pd.DataFrame(all_results)
results_df.to_csv(RETRIEVAL_DIR / "eval_results.csv", index=False)

pivot = results_df.pivot_table(index=["tier", "metric"], columns="dataset", values="accuracy")
print("\n" + "=" * 60)
print("Retrieval Evaluation Summary")
print("=" * 60)
print(pivot.to_string(float_format="{:.4f}".format))
```
