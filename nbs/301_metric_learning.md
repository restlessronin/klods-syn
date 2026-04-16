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

# Metric Learning — Supervised Contrastive Projection Head

Freeze DINOv2 ViT-S/14 backbone, train a projection head with
supervised contrastive loss on cached Gdańsk train embeddings.
Re-run the same cross-domain k-NN eval to measure improvement
over the baseline.

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
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from collections import Counter
```

```python
from klods_syn.data import RESULTS_DIR, BACKBONES, get_device
import matplotlib.pyplot as plt
import pandas as pd
```

## Configuration

```python
EMB_DIR = RESULTS_DIR / "embeddings"
METRIC_DIR = RESULTS_DIR / "metric"
METRIC_DIR.mkdir(parents=True, exist_ok=True)

BACKBONE = "DINOv3_ViT-S_16"  # safe name for filenames
BACKBONE_DISPLAY = "DINOv3 ViT-S/16"  # must match backbone name in baseline CSV
INPUT_DIM = 384
PROJ_DIM = 128
BATCH_SIZE = 256
SAMPLES_PER_CLASS = 4
LR = 1e-3
EPOCHS = 50
TEMPERATURE = 0.1
RANDOM_SEED = 42
EVAL_DATASETS = ["gdansk_test", "brickognize", "paco_garcia"]
K_VALUES = [1, 3, 5]
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

```python
# | export
class SupConLoss(nn.Module):
    """Supervised contrastive loss (Khosla et al., 2020).

    For each anchor, all samples with the same label in the batch
    are positives — the loss pulls them together and pushes apart
    all negatives.
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        device = embeddings.device
        n = embeddings.shape[0]

        # Cosine similarity matrix (embeddings assumed L2-normalized)
        sim = embeddings @ embeddings.T / self.temperature

        # Mask: same-label pairs (excluding self)
        labels = labels.unsqueeze(1)
        pos_mask = (labels == labels.T).float().to(device)
        pos_mask.fill_diagonal_(0)

        # For numerical stability
        logits_max = sim.max(dim=1, keepdim=True).values
        logits = sim - logits_max.detach()

        # Exclude self-similarity
        self_mask = torch.eye(n, device=device)
        logits = logits - self_mask * 1e9

        # Log-softmax over all non-self entries
        exp_logits = torch.exp(logits) * (1 - self_mask)
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        # Mean log-prob over positive pairs
        pos_count = pos_mask.sum(dim=1)
        mean_log_prob = (pos_mask * log_prob).sum(dim=1) / (pos_count + 1e-12)

        # Only include anchors that have at least one positive
        valid = pos_count > 0
        loss = -mean_log_prob[valid].mean()
        return loss
```

## Embedding dataset

```python
# | export
class EmbeddingDataset(Dataset):
    """Dataset wrapping pre-extracted embeddings and labels."""

    def __init__(self, embeddings: np.ndarray, labels: np.ndarray):
        self.embeddings = torch.from_numpy(embeddings).float()
        self.labels = torch.from_numpy(labels).long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]
```

## Balanced batch sampler

SupCon needs multiple positives per class in each batch. We use a
weighted sampler to oversample rare classes, combined with a batch
size large enough to get ~4 samples per class on average.

```python
def make_balanced_sampler(labels: np.ndarray) -> WeightedRandomSampler:
    """Create a sampler that balances class frequencies."""
    counts = Counter(labels.tolist())
    weights = np.array([1.0 / counts[l] for l in labels])
    return WeightedRandomSampler(weights, num_samples=len(labels), replacement=True)
```

## Load train embeddings

```python
train_data = np.load(EMB_DIR / f"{BACKBONE}_gdansk_train.npz")
train_emb = train_data["emb"]
train_labels = train_data["labels"]
print(f"Train: {train_emb.shape[0]} embeddings, dim={train_emb.shape[1]}")
print(f"Classes: {len(set(train_labels.tolist()))}")
```

## Train

```python
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

device = get_device()
print(f"Device: {device}")

head = ProjectionHead(INPUT_DIM, PROJ_DIM).to(device)
criterion = SupConLoss(temperature=TEMPERATURE)
optimizer = torch.optim.Adam(head.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

dataset = EmbeddingDataset(train_emb, train_labels)
sampler = make_balanced_sampler(train_labels)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler, drop_last=True)

losses = []
for epoch in range(EPOCHS):
    head.train()
    epoch_loss = 0.0
    n_batches = 0
    for emb_batch, label_batch in loader:
        emb_batch = emb_batch.to(device)
        label_batch = label_batch.to(device)

        projected = head(emb_batch)
        loss = criterion(projected, label_batch)

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
ax.set_ylabel("SupCon Loss")
ax.set_title("Metric Learning — Training Loss")
plt.tight_layout()
plt.show()
```

## Project all embeddings

```python
head.eval()


def project_embeddings(emb: np.ndarray, model: nn.Module, device: str) -> np.ndarray:
    """Project embeddings through the trained head."""
    with torch.no_grad():
        t = torch.from_numpy(emb).float().to(device)
        # Process in chunks to stay within memory
        chunks = []
        for i in range(0, len(t), 1024):
            chunks.append(model(t[i : i + 1024]).cpu().numpy())
    return np.concatenate(chunks)


projected = {}

# Train
proj_train = project_embeddings(train_emb, head, device)
projected["gdansk_train"] = {"emb": proj_train, "labels": train_labels}
print(f"gdansk_train: {proj_train.shape}")

# Eval datasets
for ds in EVAL_DATASETS:
    path = EMB_DIR / f"{BACKBONE}_{ds}.npz"
    if not path.exists():
        print(f"  Missing: {path.name}")
        continue
    data = np.load(path)
    proj = project_embeddings(data["emb"], head, device)
    projected[ds] = {"emb": proj, "labels": data["labels"]}
    print(f"{ds}: {proj.shape}")
```

## k-NN evaluation

```python
def knn_predict(train_emb, train_labels, test_emb, k):
    sims = test_emb @ train_emb.T
    top_k_idx = np.argpartition(-sims, k, axis=1)[:, :k]
    preds = []
    for i in range(len(test_emb)):
        neighbors = top_k_idx[i]
        neighbor_labels = train_labels[neighbors]
        counts = np.bincount(neighbor_labels)
        preds.append(counts.argmax())
    return np.array(preds)


def knn_top_k_scores(train_emb, train_labels, test_emb, test_labels, k_values):
    from sklearn.metrics import accuracy_score

    sims = test_emb @ train_emb.T
    results = {}
    for k in k_values:
        preds = knn_predict(train_emb, train_labels, test_emb, k)
        results[f"k={k} top-1"] = accuracy_score(test_labels, preds)
    top5_idx = np.argpartition(-sims, 5, axis=1)[:, :5]
    top5_correct = sum(
        1 for i in range(len(test_emb)) if test_labels[i] in train_labels[top5_idx[i]]
    )
    results["top-5 retrieval"] = top5_correct / len(test_emb)
    return results
```

```python
all_results = []

train_proj = projected["gdansk_train"]

for ds in EVAL_DATASETS:
    if ds not in projected:
        continue
    test_proj = projected[ds]
    scores = knn_top_k_scores(
        train_proj["emb"],
        train_proj["labels"],
        test_proj["emb"],
        test_proj["labels"],
        K_VALUES,
    )
    for metric, value in scores.items():
        all_results.append({"eval_dataset": ds, "metric": metric, "accuracy": value})
    print(f"{ds}: {scores}")

results_df = pd.DataFrame(all_results)
```

## Results table

```python
pivot = results_df.pivot_table(
    index="metric", columns="eval_dataset", values="accuracy"
)
pivot = pivot[EVAL_DATASETS]

print("\n" + "=" * 60)
print(f"Metric Learning k-NN Eval ({BACKBONE_DISPLAY} + SupCon projection head)")
print("=" * 60)
print(pivot.to_string(float_format="{:.3f}".format))
```

## Comparison with baseline

```python
baseline_path = RESULTS_DIR / "cross_domain_knn_results.csv"
if baseline_path.exists():
    baseline_df = pd.read_csv(baseline_path)
    baseline_backbone = baseline_df[baseline_df["backbone"] == BACKBONE_DISPLAY]

    print("\n" + "=" * 60)
    print(f"Baseline ({BACKBONE_DISPLAY} frozen) vs Metric Learning (SupCon head)")
    print("=" * 60)

    comparison = []
    for ds in EVAL_DATASETS:
        for metric in ["k=1 top-1", "top-5 retrieval"]:
            base_val = baseline_backbone[
                (baseline_backbone["eval_dataset"] == ds)
                & (baseline_backbone["metric"] == metric)
            ]["accuracy"].values
            metric_val = results_df[
                (results_df["eval_dataset"] == ds) & (results_df["metric"] == metric)
            ]["accuracy"].values
            if len(base_val) > 0 and len(metric_val) > 0:
                comparison.append(
                    {
                        "dataset": ds,
                        "metric": metric,
                        "baseline": base_val[0],
                        "metric_learning": metric_val[0],
                        "delta": metric_val[0] - base_val[0],
                    }
                )

    if comparison:
        comp_df = pd.DataFrame(comparison)
        print(comp_df.to_string(index=False, float_format="{:.3f}".format))
    else:
        print(f"No matching baseline results found for {BACKBONE_DISPLAY}")
else:
    print("Baseline results not found — run 04_baseline_eval first")
```

## Save model and results

```python
torch.save(head.state_dict(), METRIC_DIR / "projection_head.pt")
results_df.to_csv(METRIC_DIR / "metric_learning_results.csv", index=False)
print(f"Model saved to {METRIC_DIR / 'projection_head.pt'}")
print(f"Results saved to {METRIC_DIR / 'metric_learning_results.csv'}")
```
