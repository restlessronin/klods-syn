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

Asset curation for the HDRI pool used by domain-randomized renders. This
notebook owns downloading, scoring, and filtering — the rendering layer
consumes a pre-built pool and doesn't know about sources or filter
thresholds.

## Sources

[Poly Haven](https://polyhaven.com/) is the primary source — CC0-licensed
HDRIs with a public API, integrated here.
[ambientCG](https://ambientcg.com/) is a secondary CC0 source; its API shape
differs so its downloader lives in `scripts/download_ambientcg.py`. Both
sources write into the same `data/hdri/` directory and share scoring /
filtering.

```python
# | default_exp assets.hdris
```

```python
# | export
import json
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
```

## Poly Haven API

API: `https://api.polyhaven.com` — requires a `User-Agent` header per the
[API terms](https://polyhaven.com/our-api).

```python
# | export
POLYHAVEN_API = "https://api.polyhaven.com"
HDRI_DIR_DEFAULT = Path("../data/hdri")


def _polyhaven_get(url: str, timeout: float = 30.0) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "klods-syn/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def fetch_hdri_list(category: str = "all") -> list[str]:
    url = f"{POLYHAVEN_API}/assets?t=hdris"
    if category != "all":
        url += f"&c={urllib.parse.quote(category)}"
    return list(_polyhaven_get(url).keys())


def download_hdri(
    name: str,
    out_dir: Path = HDRI_DIR_DEFAULT,
    resolution: str = "1k",
    timeout: float = 60.0,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}_{resolution}.exr"
    if out_path.exists():
        return out_path
    files = _polyhaven_get(f"{POLYHAVEN_API}/files/{name}")
    exr_url = files["hdri"][resolution]["exr"]["url"]
    req = urllib.request.Request(exr_url, headers={"User-Agent": "klods-syn/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        out_path.write_bytes(resp.read())
    return out_path
```

## Category quotas

Quotas are oversized relative to the target pool — most downloads fail the
filters below. Polyhaven has 70–300 HDRIs per category; `min(quota, available)`
caps automatically. The `all` category catches HDRIs not tagged with any
specific category.

```python
# | export
@dataclass(frozen=True)
class HdriQuota:
    category: str
    count: int


DEFAULT_HDRI_QUOTAS: tuple[HdriQuota, ...] = (
    HdriQuota("indoor", 300),
    HdriQuota("studio", 80),
    HdriQuota("night", 100),
    HdriQuota("urban", 150),
    HdriQuota("outdoor", 500),
    HdriQuota("skies", 100),
    HdriQuota("sunrise-sunset", 120),
    HdriQuota("overcast", 80),
    HdriQuota("low contrast", 80),
    HdriQuota("nature", 80),
    HdriQuota("all", 300),
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

## Mitsuba setup

`mi.Bitmap` (used by scoring) requires a variant to be set.

```python
import mitsuba as mi

mi.set_variant("scalar_rgb")
```

## Scoring

Each HDRI is reduced to a handful of scalars that flag the rendering failure
modes we've hit:

- `luminance` (p75) — "typical bright pixel"; robust to sparse hotspots
  that would skew a mean.
- `dynamic_range` — `log1p(p99) − log1p(p10)`; discourages very peaky
  lighting in the sampling weights.
- `peak_luminance` — max pixel value; flags a visible sun or lamp disc.
- `bright_fraction` / `near_bright_fraction` — fraction of pixels above
  1.0 / 0.7; flag HDRIs that clip to white in non-HDR viewers or have
  dominant upper-mid-tones (white-backdrop studios).
- `shadow_luminance` (p25) — darkest-quartile brightness; flags HDRIs
  with no real shadow regions.
- `ground_luminance` — mean of the lower half of the equirect; flags
  pure-sky HDRIs with pitch-black ground hemispheres.

```python
# | export
import mitsuba as mi


@dataclass(frozen=True)
class ScoredHdri:
    path: Path
    luminance: float
    dynamic_range: float  # log ratio of 99th to 10th percentile
    peak_luminance: float  # max pixel value — flags visible sun/lamp hotspots
    bright_fraction: float  # fraction of pixels with luminance > 1.0
    near_bright_fraction: float  # fraction of pixels with luminance > 0.7
    shadow_luminance: float  # 25th percentile — "darkest quartile" brightness
    ground_luminance: float  # mean of lower half of equirect — flags pure-sky HDRIs


def score_hdri(path: Path) -> ScoredHdri:
    img = np.array(mi.Bitmap(str(path)))
    lum = img[:, :, :3] @ [0.2126, 0.7152, 0.0722]
    p10, p25, p75, p99 = np.percentile(lum, [10, 25, 75, 99])
    dr = float(np.log1p(p99) - np.log1p(p10))
    h = lum.shape[0]
    return ScoredHdri(
        path,
        float(p75),
        dr,
        float(lum.max()),
        float((lum > 1.0).mean()),
        float((lum > 0.7).mean()),
        float(p25),
        float(lum[h // 2:].mean()),
    )
```

## Pool construction

`build_hdri_pool` applies all six filters and returns the surviving HDRIs
weighted by `luminance² / (1 + dynamic_range)` — favors bright, low-DR
environments without fully excluding variety.

```python
# | export
def build_hdri_pool(
    paths: list[Path],
    min_luminance: float = 0.3,
    max_peak: float = 100.0,
    max_bright_fraction: float = 0.08,
    max_near_bright_fraction: float = 0.2,
    max_shadow_luminance: float = 0.3,
    min_ground_luminance: float = 0.04,
) -> tuple[list[ScoredHdri], np.ndarray]:
    scored = [score_hdri(p) for p in paths]
    pool = [
        s for s in scored
        if s.luminance >= min_luminance
        and s.peak_luminance <= max_peak
        and s.bright_fraction <= max_bright_fraction
        and s.near_bright_fraction <= max_near_bright_fraction
        and s.shadow_luminance <= max_shadow_luminance
        and s.ground_luminance >= min_ground_luminance
    ]
    if not pool:
        return [], np.array([])
    drs = np.array([s.dynamic_range for s in pool])
    lums = np.array([s.luminance for s in pool])
    weights = (lums ** 2) / (1.0 + drs)
    weights /= weights.sum()
    return pool, weights
```

```python
hdri_pool, hdri_weights = build_hdri_pool(hdri_paths)
print(f"HDRIs: {len(hdri_paths)} downloaded, {len(hdri_pool)} passed filters")
for s in sorted(hdri_pool, key=lambda s: s.dynamic_range):
    print(
        f"  {s.path.stem}: DR={s.dynamic_range:.2f}"
        f"  lum={s.luminance:.3f}"
        f"  weight={hdri_weights[hdri_pool.index(s)]:.3f}"
    )
```

## Curation workflow

Filter thresholds get iterated as you inspect renders. To avoid re-fetching
known failures, `scripts/download_hdris.py`, `scripts/download_ambientcg.py`,
and `scripts/hdri_cleanup.py` maintain two sidecar files in `data/hdri/`:

- `manifest.csv` — HDRIs intentionally kept, tagged by source category.
- `rejected.txt` — HDRIs that previously failed the filters; downloaders
  skip these instead of re-fetching.

Tightening a threshold and re-running `hdri_cleanup.py` moves newly-failing
HDRIs to `rejected.txt` and deletes them from disk. To reconsider a
previously-rejected HDRI after loosening a threshold, remove its line from
`rejected.txt` before re-running the downloader.

## Visual preview

Thumbnail grid of the downloaded pool, sorted by dynamic range.

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
