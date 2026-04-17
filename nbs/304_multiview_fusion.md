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

# Multi-View Fusion

Combine 4 per-view projected embeddings into a single part embedding.
Ablates three aggregation modes: mean pooling, max pooling, and
learned attention. Mean and max are parameter-free; attention trains
a lightweight scorer with standard SupCon on fused embeddings.

```python
# | default_exp fusion
```

```python
# | export
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
```

```python
import matplotlib.pyplot as plt
import pandas as pd
```

## Configuration

```python
RESULTS_DIR = Path("../results/synthetic")
EMB_DIR = RESULTS_DIR / "embeddings"
METRIC_DIR = RESULTS_DIR / "metric"
FUSION_DIR = RESULTS_DIR / "fusion"
FUSION_DIR.mkdir(parents=True, exist_ok=True)

BACKBONE = "DINOv3_ViT-S_16"
INPUT_DIM = 384
PROJ_DIM = 128
N_VIEWS = 4
BATCH_SIZE = 128
LR = 1e-3
EPOCHS = 30
TEMPERATURE = 0.1
RANDOM_SEED = 42
```

## Fusion layer

```python
# | export
class FusionLayer(nn.Module):
    """Aggregate N view embeddings into one part embedding.

    Modes:
        mean — average across views, L2-normalize.
        max  — element-wise max across views, L2-normalize.
        attention — learned per-view scores, softmax, weighted sum, L2-normalize.
    """

    def __init__(self, embed_dim: int, mode: str = "mean"):
        super().__init__()
        self.mode = mode
        if mode == "attention":
            self.scorer = nn.Linear(embed_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, n_views, embed_dim) → (batch, embed_dim)"""
        if self.mode == "mean":
            fused = x.mean(dim=1)
        elif self.mode == "max":
            fused = x.max(dim=1).values
        elif self.mode == "attention":
            scores = self.scorer(x).squeeze(-1)
            weights = F.softmax(scores, dim=1).unsqueeze(-1)
            fused = (x * weights).sum(dim=1)
        else:
            raise ValueError(f"Unknown fusion mode: {self.mode}")
        return F.normalize(fused, p=2, dim=1)
```

## Four-view dataset

Groups projected embeddings by `(ldraw_id, render_idx)` into complete
4-view tuples. Incomplete groups (fewer than 4 views) are discarded.

```python
# | export
class FourViewDataset(Dataset):
    """Groups projected embeddings into 4-view tuples by render instance."""

    def __init__(
        self,
        projected_emb: np.ndarray,
        labels: np.ndarray,
        view_idx: np.ndarray,
        ldraw_id: np.ndarray,
        render_idx: np.ndarray,
        n_views: int = 4,
    ):
        groups: dict[tuple[str, int], dict[int, int]] = defaultdict(dict)
        for i, (lid, rid, vid) in enumerate(zip(ldraw_id, render_idx, view_idx)):
            groups[(str(lid), int(rid))][int(vid)] = i

        self.items = []
        self.group_labels = []
        for (lid, rid), view_map in groups.items():
            if len(view_map) < n_views:
                continue
            indices = [view_map[v] for v in range(n_views)]
            self.items.append(indices)
            self.group_labels.append(labels[indices[0]])

        self.projected_emb = torch.from_numpy(projected_emb).float()
        self.labels_array = np.array(self.group_labels)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        indices = self.items[idx]
        views = self.projected_emb[indices]
        label = self.group_labels[idx]
        return views, label
```

## Load and project embeddings

```python
from klods_syn.data import get_device
from klods_syn.metric import ProjectionHead

device = get_device()
print(f"Device: {device}")

data = np.load(EMB_DIR / f"{BACKBONE}_synthetic.npz", allow_pickle=True)
emb = data["emb"]
labels = data["labels"]
view_idx = data["view_idx"]
ldraw_id = data["ldraw_id"]
render_idx = data["render_idx"]
print(f"Loaded {emb.shape[0]} embeddings")
```

Project through the trained projection head from 301.

```python
head = ProjectionHead(INPUT_DIM, PROJ_DIM)
head.load_state_dict(torch.load(METRIC_DIR / "projection_head.pt", weights_only=True))
head = head.to(device)
head.eval()

with torch.no_grad():
    t = torch.from_numpy(emb).float().to(device)
    chunks = []
    for i in range(0, len(t), 1024):
        chunks.append(head(t[i : i + 1024]).cpu().numpy())
    projected = np.concatenate(chunks)
print(f"Projected: {projected.shape}")
```

## Build 4-view dataset

```python
fv_dataset = FourViewDataset(projected, labels, view_idx, ldraw_id, render_idx)
print(f"Complete 4-view groups: {len(fv_dataset):,}")
print(f"Unique parts: {len(np.unique(fv_dataset.labels_array)):,}")
```

## Ablation

Run all three fusion modes. Mean and max need no training — just
evaluate. Attention trains for a few epochs.

```python
from klods_syn.metric import SupConLoss


def evaluate_fusion(fusion: FusionLayer, dataset: FourViewDataset, device: str) -> dict:
    """Compute mean intra-class and inter-class cosine similarity of fused embeddings."""
    fusion.eval()
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    all_fused = []
    all_labels = []
    with torch.no_grad():
        for views, lab in loader:
            views = views.to(device)
            fused = fusion(views)
            all_fused.append(fused.cpu())
            lab_t = lab if isinstance(lab, torch.Tensor) else torch.tensor(lab)
            all_labels.append(lab_t)
    fused_emb = torch.cat(all_fused).numpy()
    fused_labels = torch.cat(all_labels).numpy()

    sim = fused_emb @ fused_emb.T
    same = (fused_labels[:, None] == fused_labels[None, :])
    np.fill_diagonal(same, False)
    diff = ~(fused_labels[:, None] == fused_labels[None, :])

    intra = sim[same].mean() if same.any() else 0.0
    inter = sim[diff].mean() if diff.any() else 0.0
    return {"intra_sim": float(intra), "inter_sim": float(inter), "gap": float(intra - inter)}


def train_attention_fusion(
    fusion: FusionLayer,
    dataset: FourViewDataset,
    device: str,
    epochs: int,
    lr: float,
    temperature: float,
) -> list[float]:
    """Train attention fusion with SupCon on fused embeddings."""
    criterion = SupConLoss(temperature=temperature)
    optimizer = torch.optim.Adam(fusion.parameters(), lr=lr)
    bs = min(BATCH_SIZE, len(dataset))
    loader = DataLoader(dataset, batch_size=bs, shuffle=True, drop_last=len(dataset) > bs)
    losses = []
    for epoch in range(epochs):
        fusion.train()
        epoch_loss = 0.0
        n = 0
        for views, lab in loader:
            views = views.to(device)
            lab = torch.tensor(lab, device=device) if not isinstance(lab, torch.Tensor) else lab.to(device)
            fused = fusion(views)
            loss = criterion(fused, lab)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n += 1
        avg = epoch_loss / n
        losses.append(avg)
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch + 1:3d}/{epochs}  loss={avg:.4f}")
    return losses
```

```python
torch.manual_seed(RANDOM_SEED)

results = []

for mode in ["mean", "max", "attention"]:
    print(f"\n{'=' * 40}")
    print(f"Fusion mode: {mode}")
    print(f"{'=' * 40}")

    fusion = FusionLayer(PROJ_DIM, mode=mode).to(device)

    if mode == "attention":
        attn_losses = train_attention_fusion(
            fusion, fv_dataset, device,
            epochs=EPOCHS, lr=LR, temperature=TEMPERATURE,
        )

    metrics = evaluate_fusion(fusion, fv_dataset, device)
    metrics["mode"] = mode
    results.append(metrics)
    print(f"  intra={metrics['intra_sim']:.4f}  inter={metrics['inter_sim']:.4f}  gap={metrics['gap']:.4f}")

    if mode == "attention":
        torch.save(fusion.state_dict(), FUSION_DIR / "fusion_layer.pt")
        print(f"  Saved attention weights to {FUSION_DIR / 'fusion_layer.pt'}")
```

## Results

```python
results_df = pd.DataFrame(results)
results_df.to_csv(FUSION_DIR / "ablation_results.csv", index=False)
print(results_df.to_string(index=False, float_format="{:.4f}".format))
```

## Attention training loss

```python
if "attn_losses" in dir():
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(attn_losses)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("SupCon Loss")
    ax.set_title("Attention Fusion — Training Loss")
    plt.tight_layout()
    plt.show()
```
