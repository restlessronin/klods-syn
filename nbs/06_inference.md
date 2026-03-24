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

# Inference Latency — DINOv3 ViT-S/16

Measure single-image inference latency for the DINOv3 backbone
on MPS (M1 GPU) and CPU. Include backbone-only and
backbone + projection head to confirm the head is negligible.

```python
import time

import numpy as np
import torch
import torch.nn.functional as F

from klods_syn.data import load_dinov3_vits16
from klods_syn.metric import ProjectionHead
from pathlib import Path
```

## Configuration

```python
N_WARMUP = 20
N_ITERATIONS = 200
INPUT_DIM = 384
PROJ_DIM = 128
METRIC_DIR = Path("../results/baseline/metric")
```

## Load model

```python
model, transform, name = load_dinov3_vits16()
print(f"Backbone: {name}")

head = ProjectionHead(INPUT_DIM, PROJ_DIM)
head_weights = METRIC_DIR / "projection_head.pt"
if head_weights.exists():
    head.load_state_dict(
        torch.load(head_weights, map_location="cpu", weights_only=True)
    )
    print(f"Loaded projection head from {head_weights.name}")
else:
    print("No saved projection head found — using random init (timing still valid)")
head.eval()
```

## Create dummy input

```python
from PIL import Image

dummy_pil = Image.fromarray(np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8))
dummy_input = transform(dummy_pil).unsqueeze(0)
print(f"Input shape: {dummy_input.shape}")
```

## Benchmark function

```python
def benchmark_latency(
    model,
    input_tensor: torch.Tensor,
    device: str,
    n_warmup: int = N_WARMUP,
    n_iter: int = N_ITERATIONS,
    head=None,
) -> dict:
    """Measure single-image inference latency.

    Returns dict with mean, std, median, min, max in milliseconds.
    """
    model = model.to(device)
    if head is not None:
        head = head.to(device)
    x = input_tensor.to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            out = model(x)
            if head is not None:
                if isinstance(out, torch.Tensor):
                    features = out
                elif hasattr(out, "pooler_output") and out.pooler_output is not None:
                    features = out.pooler_output
                elif hasattr(out, "last_hidden_state"):
                    features = out.last_hidden_state[:, 0, :]
                features = F.normalize(features, p=2, dim=1)
                head(features)
    if device == "mps":
        torch.mps.synchronize()

    # Timed iterations
    times = []
    with torch.no_grad():
        for _ in range(n_iter):
            if device == "mps":
                torch.mps.synchronize()
            t0 = time.perf_counter()

            out = model(x)
            if head is not None:
                if isinstance(out, torch.Tensor):
                    features = out
                elif hasattr(out, "pooler_output") and out.pooler_output is not None:
                    features = out.pooler_output
                elif hasattr(out, "last_hidden_state"):
                    features = out.last_hidden_state[:, 0, :]
                features = F.normalize(features, p=2, dim=1)
                head(features)

            if device == "mps":
                torch.mps.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)

    times = np.array(times)
    return {
        "mean_ms": times.mean(),
        "std_ms": times.std(),
        "median_ms": np.median(times),
        "min_ms": times.min(),
        "max_ms": times.max(),
    }
```

## Run benchmarks

```python
DEVICES = ["mps", "cpu"] if torch.backends.mps.is_available() else ["cpu"]

results = []

for device in DEVICES:
    print(f"\n{'=' * 50}")
    print(f"Device: {device}")
    print(f"{'=' * 50}")

    # Backbone only
    stats = benchmark_latency(model, dummy_input, device)
    stats["device"] = device
    stats["mode"] = "backbone only"
    results.append(stats)
    print(
        f"  Backbone only:  {stats['mean_ms']:.1f} ± {stats['std_ms']:.1f} ms  (median {stats['median_ms']:.1f}, min {stats['min_ms']:.1f}, max {stats['max_ms']:.1f})"
    )

    # Backbone + projection head
    stats = benchmark_latency(model, dummy_input, device, head=head)
    stats["device"] = device
    stats["mode"] = "backbone + head"
    results.append(stats)
    print(
        f"  Backbone + head: {stats['mean_ms']:.1f} ± {stats['std_ms']:.1f} ms  (median {stats['median_ms']:.1f}, min {stats['min_ms']:.1f}, max {stats['max_ms']:.1f})"
    )

    # Head overhead
    head_overhead = results[-1]["mean_ms"] - results[-2]["mean_ms"]
    print(f"  Head overhead:   {head_overhead:.2f} ms")
```

## Summary

```python
import pandas as pd

df = pd.DataFrame(results)
print("\n" + "=" * 60)
print("DINOv3 ViT-S/16 — Single-Image Inference Latency")
print("=" * 60)
print(
    df[["device", "mode", "mean_ms", "std_ms", "median_ms"]].to_string(
        index=False, float_format="{:.1f}".format
    )
)
```

```python
# | hide
import nbdev

nbdev.nbdev_export()
```
