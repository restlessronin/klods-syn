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

# Metric Learning — View-Aware Supervised Contrastive

Train a projection head on frozen DINOv3 ViT-S/16 embeddings extracted
from synthetic renders. Uses view-aware contrastive pairs: same part +
same viewpoint = positive, same part + different viewpoint = excluded
(neither positive nor negative). This preserves viewpoint information
in the embedding space so the downstream fusion layer (304) has
meaningful per-view signals to combine.

```python
# | default_exp metric
```

```python
# | export
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, Sampler
from collections import defaultdict
```

```python
import matplotlib.pyplot as plt
```

## Configuration

```python
RESULTS_DIR = Path("../results/synthetic")
EMB_DIR = RESULTS_DIR / "embeddings"
METRIC_DIR = RESULTS_DIR / "metric"
METRIC_DIR.mkdir(parents=True, exist_ok=True)

BACKBONE = "DINOv3_ViT-S_16"
INPUT_DIM = 384
PROJ_DIM = 128
GROUPS_PER_BATCH = 64
SAMPLES_PER_GROUP = 4
BATCH_SIZE = GROUPS_PER_BATCH * SAMPLES_PER_GROUP
LR = 1e-3
EPOCHS = 50
TEMPERATURE = 0.1
RANDOM_SEED = 42
```

## Projection head

```python
# | export
class ProjectionHead(nn.Module):
    """MLP projection head: linear → BN → ReLU → linear → L2-normalize."""

    def __init__(self, input_dim: int, proj_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.BatchNorm1d(input_dim),
            nn.ReLU(inplace=True),
            nn.Linear(input_dim, proj_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), p=2, dim=1)
```

## Supervised contrastive loss

Standard SupCon (Khosla et al., 2020) treats all same-label samples as
positives. Used by the fusion layer (304) where view distinction is
irrelevant.

```python
# | export
class SupConLoss(nn.Module):
    """Supervised contrastive loss (Khosla et al., 2020)."""

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        device = embeddings.device
        n = embeddings.shape[0]

        sim = embeddings @ embeddings.T / self.temperature

        labels = labels.unsqueeze(1)
        pos_mask = (labels == labels.T).float().to(device)
        pos_mask.fill_diagonal_(0)

        logits_max = sim.max(dim=1, keepdim=True).values
        logits = sim - logits_max.detach()

        self_mask = torch.eye(n, device=device)
        logits = logits - self_mask * 1e9

        exp_logits = torch.exp(logits) * (1 - self_mask)
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        pos_count = pos_mask.sum(dim=1)
        mean_log_prob = (pos_mask * log_prob).sum(dim=1) / (pos_count + 1e-12)

        valid = pos_count > 0
        loss = -mean_log_prob[valid].mean()
        return loss
```

## View-aware supervised contrastive loss

Positive pairs share both the same part label and the same view index.
Same-part-different-view pairs are excluded from the denominator so
they exert no gradient — they neither attract nor repel.

```python
# | export
class ViewAwareSupConLoss(nn.Module):
    """Supervised contrastive loss with view-aware pair selection.

    Positives: same part AND same viewpoint.
    Excluded: same part, different viewpoint (no gradient).
    Negatives: different part.
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        view_idx: torch.Tensor,
    ) -> torch.Tensor:
        device = embeddings.device
        n = embeddings.shape[0]

        sim = embeddings @ embeddings.T / self.temperature

        same_part = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()
        same_view = (view_idx.unsqueeze(1) == view_idx.unsqueeze(0)).float()

        pos_mask = same_part * same_view
        pos_mask.fill_diagonal_(0)

        exclude_mask = same_part * (1 - same_view)

        self_mask = torch.eye(n, device=device)
        valid_mask = (1 - self_mask - exclude_mask).clamp(min=0)

        logits_max = sim.max(dim=1, keepdim=True).values
        logits = sim - logits_max.detach()

        exp_logits = torch.exp(logits) * valid_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        pos_count = pos_mask.sum(dim=1)
        mean_log_prob = (pos_mask * log_prob).sum(dim=1) / (pos_count + 1e-12)

        valid = pos_count > 0
        loss = -mean_log_prob[valid].mean()
        return loss
```

## View-aware embedding dataset

```python
# | export
class ViewAwareEmbeddingDataset(Dataset):
    """Dataset wrapping embeddings with part labels and view indices."""

    def __init__(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        view_idx: np.ndarray,
    ):
        self.embeddings = torch.from_numpy(embeddings).float()
        self.labels = torch.from_numpy(labels).long()
        self.view_idx = torch.from_numpy(view_idx).long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx], self.view_idx[idx]
```

## View-grouped batch sampler

Each batch contains `groups_per_batch` random (part, view) groups with
`samples_per_group` embeddings each, ensuring every anchor has at least
`samples_per_group - 1` same-view positives.

```python
# | export
class ViewGroupedSampler(Sampler):
    """Samples batches of (part, view) groups for view-aware SupCon."""

    def __init__(
        self,
        labels: np.ndarray,
        view_idx: np.ndarray,
        groups_per_batch: int = 64,
        samples_per_group: int = 4,
        num_batches: int | None = None,
        rng_seed: int = 42,
    ):
        self.groups_per_batch = groups_per_batch
        self.samples_per_group = samples_per_group
        self.rng = np.random.default_rng(rng_seed)

        self.group_indices: dict[tuple[int, int], list[int]] = defaultdict(list)
        for i, (lab, vid) in enumerate(zip(labels, view_idx)):
            self.group_indices[(int(lab), int(vid))].append(i)

        self.valid_groups = [
            k for k, v in self.group_indices.items()
            if len(v) >= samples_per_group
        ]

        n_samples = sum(len(v) for v in self.group_indices.values())
        self._num_batches = num_batches or (
            n_samples // (groups_per_batch * samples_per_group)
        )

    def __iter__(self):
        for _ in range(self._num_batches):
            chosen = self.rng.choice(
                len(self.valid_groups),
                size=min(self.groups_per_batch, len(self.valid_groups)),
                replace=False,
            )
            batch = []
            for g_idx in chosen:
                group_key = self.valid_groups[g_idx]
                members = self.group_indices[group_key]
                selected = self.rng.choice(
                    members, size=self.samples_per_group, replace=len(members) < self.samples_per_group
                )
                batch.extend(selected.tolist())
            yield batch

    def __len__(self):
        return self._num_batches
```

## Load synthetic embeddings

```python
data = np.load(EMB_DIR / f"{BACKBONE}_synthetic.npz", allow_pickle=True)
train_emb = data["emb"]
train_labels = data["labels"]
train_view_idx = data["view_idx"]
print(f"Train: {train_emb.shape[0]} embeddings, dim={train_emb.shape[1]}")
print(f"Classes: {len(np.unique(train_labels))}")
print(f"Views: {np.unique(train_view_idx).tolist()}")
```

## Train

```python
from klods_syn.data import get_device

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

device = get_device()
print(f"Device: {device}")

head = ProjectionHead(INPUT_DIM, PROJ_DIM).to(device)
criterion = ViewAwareSupConLoss(temperature=TEMPERATURE)
optimizer = torch.optim.Adam(head.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

dataset = ViewAwareEmbeddingDataset(train_emb, train_labels, train_view_idx)
sampler = ViewGroupedSampler(
    train_labels, train_view_idx,
    groups_per_batch=GROUPS_PER_BATCH,
    samples_per_group=SAMPLES_PER_GROUP,
    rng_seed=RANDOM_SEED,
)
loader = DataLoader(dataset, batch_sampler=sampler)

losses = []
for epoch in range(EPOCHS):
    head.train()
    epoch_loss = 0.0
    n_batches = 0
    for emb_batch, label_batch, view_batch in loader:
        emb_batch = emb_batch.to(device)
        label_batch = label_batch.to(device)
        view_batch = view_batch.to(device)

        projected = head(emb_batch)
        loss = criterion(projected, label_batch, view_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        n_batches += 1

    scheduler.step()
    avg_loss = epoch_loss / n_batches
    losses.append(avg_loss)
    if (epoch + 1) % 10 == 0:
        print(
            f"Epoch {epoch + 1:3d}/{EPOCHS}  loss={avg_loss:.4f}  lr={scheduler.get_last_lr()[0]:.2e}"
        )
```

## Training loss curve

```python
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(losses)
ax.set_xlabel("Epoch")
ax.set_ylabel("View-Aware SupCon Loss")
ax.set_title("Metric Learning — Training Loss")
plt.tight_layout()
plt.show()
```

## Save model

```python
torch.save(head.state_dict(), METRIC_DIR / "projection_head.pt")
print(f"Model saved to {METRIC_DIR / 'projection_head.pt'}")
```
