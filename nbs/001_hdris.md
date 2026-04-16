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

# HDRI Environment Maps

Download and score HDRI environment maps from [Poly Haven](https://polyhaven.com/)
for synthetic rendering. Category quotas bias toward low dynamic range
environments (indoor, studio) that are closer to the scanner's operating
conditions, while keeping some high-DR outdoor scenes for robustness.

```python
# | default_exp assets.hdris
```

```python
# | export
import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

type Vec3 = tuple[float, float, float]
```

## Poly Haven API

API: `https://api.polyhaven.com` — requires a `User-Agent` header.
All requests require unique User-Agent per [API terms](https://polyhaven.com/our-api).

```python
# | export
POLYHAVEN_API = "https://api.polyhaven.com"
HDRI_DIR_DEFAULT = Path("../data/hdri")


def _polyhaven_get(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "klods-syn/0.1"})
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def fetch_hdri_list(category: str = "all") -> list[str]:
    url = f"{POLYHAVEN_API}/assets?t=hdris"
    if category != "all":
        url += f"&c={category}"
    return list(_polyhaven_get(url).keys())


def download_hdri(
    name: str, out_dir: Path = HDRI_DIR_DEFAULT, resolution: str = "1k"
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}_{resolution}.exr"
    if out_path.exists():
        return out_path
    files = _polyhaven_get(f"{POLYHAVEN_API}/files/{name}")
    exr_url = files["hdri"][resolution]["exr"]["url"]
    req = urllib.request.Request(exr_url, headers={"User-Agent": "klods-syn/0.1"})
    with urllib.request.urlopen(req) as resp:
        out_path.write_bytes(resp.read())
    return out_path
```

## Category quotas

Available Poly Haven HDRI categories: outdoor, skies, indoor, studio,
sunrise/sunset, night, nature, urban.

We bias toward indoor/studio (closest to scanner environment, low DR)
and keep a small quota of outdoor/skies for robustness.

```python
# | export
@dataclass(frozen=True)
class HdriQuota:
    category: str
    count: int


DEFAULT_HDRI_QUOTAS: tuple[HdriQuota, ...] = (
    HdriQuota("indoor", 20),
    HdriQuota("studio", 15),
    HdriQuota("night", 8),
    HdriQuota("urban", 8),
    HdriQuota("outdoor", 5),
    HdriQuota("skies", 3),
    HdriQuota("sunrise-sunset", 3),
)


def ensure_hdris(
    quotas: tuple[HdriQuota, ...] = DEFAULT_HDRI_QUOTAS,
    out_dir: Path = HDRI_DIR_DEFAULT,
    resolution: str = "1k",
) -> list[Path]:
    rng = np.random.default_rng(42)
    seen: set[str] = set()
    paths: list[Path] = []

    for quota in quotas:
        names = [n for n in fetch_hdri_list(quota.category) if n not in seen]
        selected = (
            rng.choice(names, size=min(quota.count, len(names)), replace=False)
            if names
            else []
        )
        for name in selected:
            seen.add(name)
            try:
                paths.append(download_hdri(name, out_dir, resolution))
            except Exception as e:
                print(f"Skipping {name}: {e}")

    return paths
```

```python
hdri_paths = ensure_hdris()
print(f"HDRIs downloaded: {len(hdri_paths)}")
```

## Available categories

```python
for cat in ("indoor", "studio", "night", "urban", "outdoor", "skies", "sunrise-sunset"):
    names = fetch_hdri_list(cat)
    print(f"  {cat}: {len(names)} available")
```

## Scoring and weighted pool

Score each HDRI by dynamic range (log ratio of bright to dark
luminance percentiles). Build a sampling pool with inverse-DR
weighting so low dynamic range environments dominate the training
distribution while high-DR scenes still appear for robustness.

Mitsuba must be initialized before `mi.Bitmap` can be used.

```python
import mitsuba as mi

mi.set_variant("scalar_rgb")
```

```python
# | export
import mitsuba as mi


@dataclass(frozen=True)
class ScoredHdri:
    path: Path
    luminance: float
    dynamic_range: float  # log ratio of 99th to 10th percentile


def score_hdri(path: Path) -> ScoredHdri:
    img = np.array(mi.Bitmap(str(path)))
    lum = img[:, :, :3] @ [0.2126, 0.7152, 0.0722]
    p10, p99 = np.percentile(lum, [10, 99])
    dr = float(np.log1p(p99) - np.log1p(p10))
    return ScoredHdri(path, float(np.mean(lum)), dr)


def build_hdri_pool(
    paths: list[Path], min_luminance: float = 0.1
) -> tuple[list[ScoredHdri], np.ndarray]:
    scored = [score_hdri(p) for p in paths]
    pool = [s for s in scored if s.luminance >= min_luminance]
    if not pool:
        return [], np.array([])
    drs = np.array([s.dynamic_range for s in pool])
    weights = 1.0 / (1.0 + drs)
    weights /= weights.sum()
    return pool, weights
```

```python
hdri_pool, hdri_weights = build_hdri_pool(hdri_paths)
print(f"HDRIs: {len(hdri_paths)} downloaded, {len(hdri_pool)} passed luminance filter")
for s in sorted(hdri_pool, key=lambda s: s.dynamic_range):
    print(
        f"  {s.path.stem}: DR={s.dynamic_range:.2f}"
        f"  lum={s.luminance:.3f}"
        f"  weight={hdri_weights[hdri_pool.index(s)]:.3f}"
    )
```

## Visual preview

Thumbnail grid of the downloaded HDRIs, sorted by dynamic range.

```python
import matplotlib.pyplot as plt

sorted_pool = sorted(hdri_pool, key=lambda s: s.dynamic_range)
cols = 5
rows = (len(sorted_pool) + cols - 1) // cols

fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 2 * rows))
axes = axes.ravel() if rows > 1 else [axes] if rows == 1 else axes

for i, s in enumerate(sorted_pool):
    img = np.array(mi.Bitmap(str(s.path)))
    axes[i].imshow(np.clip(img[:, :, :3] / (1 + img[:, :, :3]), 0, 1))
    axes[i].set_axis_off()
    axes[i].set_title(
        f"{s.path.stem}\nDR={s.dynamic_range:.2f} w={hdri_weights[hdri_pool.index(s)]:.3f}",
        fontsize=8,
    )

for i in range(len(sorted_pool), len(axes)):
    axes[i].set_axis_off()

fig.suptitle("HDRI pool — sorted by dynamic range", fontsize=12)
plt.tight_layout()
plt.show()
```
