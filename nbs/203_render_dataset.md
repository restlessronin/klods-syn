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

# Dataset Rendering

Orchestration layer: render the full working subset to disk as a
labeled image dataset. Reads `render_plan.csv` from the catalog pipeline,
renders each (part, color) pair `n_per_pair` times across 4 viewpoints,
and writes images + `labels.csv` to an output directory.

A manifest tracks completed renders so interrupted runs can resume
without re-rendering.

```python
# | default_exp rendering.dataset
```

```python
import platform
import sys
from pathlib import Path

if platform.system() == "Linux":
    import subprocess

    # Pre-built release ZIPs cannot be used here. ldr_tools_py.so is a CPython
    # extension compiled against Blender's bundled Python. Release history:
    #   0.4.x → Python 3.11 (Blender 4.1+)
    #   0.5.x → Python 3.13 (Blender 5.1+)
    # Kaggle runs Python 3.12, so no release binary matches. Building from
    # source takes ~3-5 minutes but always matches the runtime Python version.
    #
    # # Alternative (kept for reference, does not work on Kaggle Python 3.12):
    # import urllib.request, zipfile
    # LDR_TOOLS_VERSION = "0.5.0"
    # LDR_TOOLS_ZIP = f"https://github.com/ScanMountGoat/ldr_tools_blender/releases/download/{LDR_TOOLS_VERSION}/ldr_tools_blender_linux_x64.zip"
    # zip_path = Path("/tmp/ldr_tools.zip")
    # urllib.request.urlretrieve(LDR_TOOLS_ZIP, zip_path)
    # with zipfile.ZipFile(zip_path) as zf:
    #     matches = [n for n in zf.namelist() if Path(n).name == "ldr_tools_py.so"]
    #     zf.extract(matches[0], "/tmp/ldr_tools")

    LDR_TOOLS_DIR = Path("/tmp/ldr_tools")
    so_path = LDR_TOOLS_DIR / "ldr_tools_py.so"

    if not so_path.exists():
        LDR_TOOLS_DIR.mkdir(exist_ok=True)
        subprocess.run(["curl", "--proto", "=https", "--tlsv1.2", "-sSf",
                        "https://sh.rustup.rs", "-o", "/tmp/rustup.sh"], check=True)
        subprocess.run(["sh", "/tmp/rustup.sh", "-y", "--no-modify-path"], check=True)
        env = {**__import__("os").environ, "PATH": f"{Path.home()}/.cargo/bin:{__import__('os').environ['PATH']}"}
        subprocess.run(["git", "clone", "--depth=1", "--branch", "0.5.0",
                        "https://github.com/ScanMountGoat/ldr_tools_blender.git",
                        "/tmp/ldr_tools_blender"], check=True)
        subprocess.run(["cargo", "build", "--release"],
                       cwd="/tmp/ldr_tools_blender/ldr_tools_py", env=env, check=True)
        built = next(Path("/tmp/ldr_tools_blender/target/release").glob("libldr_tools_py*.so"))
        built.rename(so_path)

    sys.path.insert(0, str(LDR_TOOLS_DIR))

    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q",
         "git+https://github.com/restlessronin/klods-syn.git"],
        check=True,
    )
else:
    sys.path.insert(0, "..")
```

```python
# | export
import time
from pathlib import Path

import mitsuba as mi
import numpy as np
import pandas as pd
from PIL import Image

import ldr_tools_py as ldr
```

## Mitsuba variant

Auto-detect CUDA for GPU rendering, fall back to LLVM on CPU.
All rendering code uses `ScalarTransform4f` which is variant-independent
for the scalar backends.

```python
# | export
def _init_mitsuba() -> str:
    try:
        mi.set_variant("cuda_ad_rgb")
        return "cuda_ad_rgb"
    except Exception:
        mi.set_variant("llvm_ad_rgb")
        return "llvm_ad_rgb"

VARIANT = _init_mitsuba()
print(f"Mitsuba variant: {VARIANT}")
```

## Mesh loading

Load an LDraw `.dat` file and return vertices and triangulated faces,
with the Y-flip applied to convert from LDraw (-Y up) to Mitsuba (+Y up).

```python
# | export
def load_part_mesh(
    ldraw_id: str, ldraw_dir: Path
) -> tuple[np.ndarray, np.ndarray]:
    scene = ldr.load_file(
        str(ldraw_dir / "parts" / f"{ldraw_id}.dat"),
        str(ldraw_dir),
        [],
        ldr.GeometrySettings(),
    )
    key = list(scene.geometry_cache.keys())[0]
    geom = scene.geometry_cache[key]

    vertices = geom.vertices.astype(np.float32)
    vertices[:, 1] *= -1

    idx = geom.vertex_indices
    triangles = []
    for s, n in zip(geom.face_start_indices, geom.face_sizes):
        for i in range(1, n - 1):
            triangles.append([idx[s], idx[s + i], idx[s + i + 1]])
    faces = np.array(triangles, dtype=np.uint32)[:, [0, 2, 1]]
    return vertices, faces
```

## Color construction

Build an `LDrawColor` from the Rebrickable colors table row.
Alpha is 0.5 for transparent materials, 1.0 otherwise.

```python
# | export
from klods_syn.rendering.domain_randomization import LDrawColor


def _hex_to_rgb(hex_str: str) -> tuple[float, float, float]:
    h = hex_str.lstrip("#")
    return tuple(int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4))


def ldraw_color_from_rebrickable(
    color_id: int, color_name: str, material: str, rgb_hex: str
) -> LDrawColor:
    alpha = 0.5 if material == "transparent" else 1.0
    return LDrawColor(
        code=color_id,
        name=color_name,
        rgb=_hex_to_rgb(rgb_hex),
        alpha=alpha,
        material=material,
    )
```

## Manifest

Tracks completed `(ldraw_id, color_id, render_idx)` triples so runs can
resume after interruption. Written to `manifest.csv` in the output directory.

```python
# | export
def load_manifest(output_dir: Path) -> set[tuple[str, int, int]]:
    path = output_dir / "manifest.csv"
    if not path.exists():
        return set()
    df = pd.read_csv(path)
    return set(zip(df["ldraw_id"], df["color_id"], df["render_idx"]))


def update_manifest(
    output_dir: Path, ldraw_id: str, color_id: int, render_idx: int
) -> None:
    path = output_dir / "manifest.csv"
    row = pd.DataFrame(
        [{"ldraw_id": ldraw_id, "color_id": color_id, "render_idx": render_idx}]
    )
    row.to_csv(path, mode="a", header=not path.exists(), index=False)
```

## Render pair

Render one (part, color) pair `n_renders` times. Each render produces
4 images (1 direct + 3 mirror views). Images are saved as PNGs and
a list of label rows is returned.

```python
# | export
from klods_syn.rendering.domain_randomization import (
    RandomizationSpace,
    build_scene,
    build_hdri_pool,
    compute_exposure,
    postprocess,
    transform_mesh,
)
from klods_syn.rendering.viewpoints import TriMirrorRig


def render_pair(
    ldraw_id: str,
    color: LDrawColor,
    vertices: np.ndarray,
    faces: np.ndarray,
    render_idxs: list[int],
    space: RandomizationSpace,
    rig: TriMirrorRig,
    hdri_pool,
    hdri_weights,
    output_dir: Path,
    rng: np.random.Generator,
    spp: int,
    resolution: int,
) -> list[dict]:
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    label_rows = []

    for render_idx in render_idxs:
        cfg = space.sample(rng, hdri_pool, hdri_weights, color=color)
        tilted_rig = rig.with_tilt(cfg.rig_tilt)
        config = tilted_rig.viewpoint_config()
        zooms = [1.0] + [config.mirror_zoom] * len(config.mirrors)

        mesh, _ = transform_mesh(vertices, faces, cfg)

        masks = [
            np.array(
                mi.render(
                    mi.load_dict(
                        build_scene(
                            mesh, cfg, origin=vp, fov=35 / zoom,
                            resolution=resolution, mask_mode=True,
                        )
                    )
                )
            )
            for vp, zoom in zip(config.viewpoints, zooms)
        ]
        exposure = compute_exposure(masks)

        for view_idx, (vp, zoom) in enumerate(zip(config.viewpoints, zooms)):
            raw = np.array(
                mi.render(
                    mi.load_dict(
                        build_scene(
                            mesh, cfg, origin=vp, fov=35 / zoom,
                            resolution=resolution, spp=spp,
                        )
                    )
                )
            )
            img = postprocess(raw, cfg, exposure=exposure)
            fname = f"{ldraw_id}_{color.code}_{view_idx}_{render_idx}.png"
            Image.fromarray((img * 255).astype(np.uint8)).save(images_dir / fname)
            label_rows.append(
                {
                    "image_path": f"images/{fname}",
                    "ldraw_id": ldraw_id,
                    "color_id": color.code,
                    "color_name": color.name,
                    "material": color.material,
                    "view_idx": view_idx,
                    "render_idx": render_idx,
                }
            )

    return label_rows
```

## Dataset orchestration

Main entry point. Iterates the render plan, skips completed pairs via
the manifest, and appends to `labels.csv` incrementally so partial runs
are immediately usable.

`n_per_pair` is the primary lever for test vs production runs:
a small value (1–5) enables a quick smoke-test; the full dataset
uses whatever N was decided for training.

`max_seconds` and `max_pairs` are stopping conditions for bounded runs;
the loop exits cleanly after the current pair finishes when either
fires. `max_pairs` counts pairs with actual work done (skipped pairs
from the manifest don't count), so it's a predictable unit of work for
session batching. `max_seconds` is useful as a hard safety bound under
Kaggle's session-kill.

```python
# | export
def render_dataset(
    render_plan_path: Path,
    output_dir: Path,
    ldraw_dir: Path,
    rebrickable_dir: Path,
    n_per_pair: int = 20,
    spp: int = 64,
    resolution: int = 512,
    max_seconds: float | None = None,
    max_pairs: int | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    render_plan = pd.read_csv(render_plan_path)
    colors_df = pd.read_csv(rebrickable_dir / "colors.csv.gz").set_index("id")

    hdri_dir = rebrickable_dir.parent / "hdri"
    hdri_paths = list(hdri_dir.glob("*.exr"))
    hdri_pool, hdri_weights = build_hdri_pool(hdri_paths)

    rig = TriMirrorRig.optimize(camera_dist=250.0)
    space = RandomizationSpace()
    rng = np.random.default_rng(42)

    manifest = load_manifest(output_dir)
    labels_path = output_dir / "labels.csv"
    labels_header = not labels_path.exists()

    t_start = time.monotonic()
    n_rendered = 0
    n_pairs_done = 0

    for _, row in render_plan.iterrows():
        if max_seconds and (time.monotonic() - t_start) >= max_seconds:
            break
        if max_pairs is not None and n_pairs_done >= max_pairs:
            break

        ldraw_id = row["ldraw_id"]
        color_id = int(row["color_id"])

        pending = [
            i for i in range(n_per_pair)
            if (ldraw_id, color_id, i) not in manifest
        ]
        if not pending:
            continue

        try:
            vertices, faces = load_part_mesh(ldraw_id, ldraw_dir)
        except Exception as e:
            print(f"Skipping {ldraw_id}: {e}")
            continue

        rgb_hex = colors_df.loc[color_id, "rgb"]
        color = ldraw_color_from_rebrickable(
            color_id, row["color_name"], row["material"], rgb_hex
        )

        label_rows = render_pair(
            ldraw_id=ldraw_id,
            color=color,
            vertices=vertices,
            faces=faces,
            render_idxs=pending,
            space=space,
            rig=rig,
            hdri_pool=hdri_pool,
            hdri_weights=hdri_weights,
            output_dir=output_dir,
            rng=rng,
            spp=spp,
            resolution=resolution,
        )

        pd.DataFrame(label_rows).to_csv(
            labels_path, mode="a", header=labels_header, index=False
        )
        labels_header = False

        for i in pending:
            update_manifest(output_dir, ldraw_id, color_id, i)
            manifest.add((ldraw_id, color_id, i))

        n_rendered += len(pending)
        n_pairs_done += 1

    elapsed = time.monotonic() - t_start
    print(f"Rendered {n_rendered} renders ({n_pairs_done} pairs) in {elapsed:.1f}s")
```

## Paths and run configuration

Paths and render parameters are the only things that differ between
local and Kaggle runs. Everything else is the same code.

```python
if platform.system() == "Linux":
    # Kaggle: input datasets mounted under /kaggle/input/
    DATA_DIR      = Path("/kaggle/input/klods-syn-data")
    OUTPUT_DIR    = Path("/kaggle/working/dataset")
    N_PER_PAIR    = 20
    MAX_SECONDS   = 11 * 3600  # hard safety under Kaggle's 12h kill
    MAX_PAIRS     = None       # set after a calibration run
else:
    DATA_DIR      = Path("../data")
    OUTPUT_DIR    = DATA_DIR / "dataset_test"
    N_PER_PAIR    = 1
    MAX_SECONDS   = 600    # 10-minute smoke test
    MAX_PAIRS     = 10     # small calibration batch
```

## Run

```python
render_dataset(
    render_plan_path=DATA_DIR / "rebrickable" / "render_plan.csv",
    output_dir=OUTPUT_DIR,
    ldraw_dir=DATA_DIR / "ldraw",
    rebrickable_dir=DATA_DIR / "rebrickable",
    n_per_pair=N_PER_PAIR,
    spp=64,
    resolution=512,
    max_seconds=MAX_SECONDS,
    max_pairs=MAX_PAIRS,
)
```

```python
labels = pd.read_csv(DATA_DIR / "dataset_test" / "labels.csv")
print(f"Images rendered: {len(labels):,}")
print(f"Unique parts:    {labels['ldraw_id'].nunique():,}")
print(f"Unique colors:   {labels['color_id'].nunique():,}")
print(f"Views per render: {labels.groupby(['ldraw_id','color_id','render_idx']).size().value_counts().to_dict()}")
```
