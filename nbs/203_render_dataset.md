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

Orchestration layer: render the working-subset parts to disk as a
labeled image dataset. Reads `render_plan.csv` from the catalog pipeline
as a per-part color distribution, then for each part renders
`n_per_part` samples across 4 viewpoints — the color for each render is
drawn from that part's real-world color distribution, not pinned per
class. The class label is `ldraw_id`; color is a randomization axis.

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
        [sys.executable, "-m", "pip", "install", "-q", "mitsuba>=3.8.0"],
        check=True,
    )
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "--force-reinstall", "--no-deps",
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

Tracks completed `(ldraw_id, render_idx)` pairs so runs can resume
after interruption. Written to `manifest.csv` in the output directory.
Color is not part of the key — it's sampled per-render from the part's
distribution and recorded in `labels.csv`.

```python
# | export
def load_manifest(output_dir: Path) -> set[tuple[str, int]]:
    path = output_dir / "manifest.csv"
    if not path.exists():
        return set()
    df = pd.read_csv(path)
    return set(zip(df["ldraw_id"], df["render_idx"]))


def update_manifest(
    output_dir: Path, ldraw_id: str, render_idx: int
) -> None:
    path = output_dir / "manifest.csv"
    row = pd.DataFrame(
        [{"ldraw_id": ldraw_id, "render_idx": render_idx}]
    )
    row.to_csv(path, mode="a", header=not path.exists(), index=False)
```

## Render helpers

Low-level primitives shared by the full dataset pipeline and the
diagnostic scripts (SPP sweep, lighting check, HDRI preview). Each is
composable — callers assemble them to cover their specific loop shape.

```python
# | export
from dataclasses import dataclass, replace as dc_replace

from klods_syn.assets.hdris import build_hdri_pool
from klods_syn.rendering.domain_randomization import (
    RandomizationSpace,
    RenderConfig,
    build_scene,
    compute_exposure,
    postprocess,
    transform_mesh,
)
from klods_syn.rendering.viewpoints import TriMirrorRig


DEFAULT_CAMERA_DIST = 380.0
DEFAULT_FOV = 35.0


@dataclass(frozen=True)
class RenderContext:
    """Bundle of everything needed to render a pair: plan, colors,
    HDRI pool, rig, randomization space, and paths to mesh data."""

    plan: pd.DataFrame
    colors_df: pd.DataFrame
    ldraw_dir: Path
    hdri_pool: list
    hdri_weights: np.ndarray
    rig: TriMirrorRig
    space: RandomizationSpace

    @staticmethod
    def load(
        data_dir: Path,
        camera_dist: float = DEFAULT_CAMERA_DIST,
        ldraw_dir: Path | None = None,
    ) -> "RenderContext":
        plan = pd.read_csv(data_dir / "rebrickable" / "render_plan.csv")
        plan["ldraw_id"] = plan["ldraw_id"].astype(str)
        colors_df = pd.read_csv(data_dir / "rebrickable" / "colors.csv.gz").set_index("id")
        hdri_paths = list((data_dir / "hdri").glob("*.exr"))
        hdri_pool, hdri_weights = build_hdri_pool(hdri_paths)
        rig = TriMirrorRig.optimize(camera_dist=camera_dist)
        return RenderContext(
            plan=plan, colors_df=colors_df,
            ldraw_dir=ldraw_dir if ldraw_dir is not None else data_dir / "ldraw",
            hdri_pool=hdri_pool, hdri_weights=hdri_weights,
            rig=rig, space=RandomizationSpace(),
        )


def part_max_translation(vertices: np.ndarray, platform_radius: float) -> float:
    """Given a part and the platform radius, return the max XZ distance
    from origin at which the part's center can land while keeping the
    part fully on the platform under any rotation."""
    part_radius = float(np.linalg.norm(vertices, axis=1).max())
    return max(0.0, platform_radius - part_radius)


def sample_cfg(
    ctx: RenderContext,
    ldraw_id: str,
    color_id: int,
    rng: np.random.Generator,
    overrides: dict | None = None,
) -> tuple[RenderConfig, np.ndarray, np.ndarray, LDrawColor]:
    """Resolve a (ldraw_id, color_id) pair to a sampled RenderConfig
    with per-part translation cap. Returns (cfg, vertices, faces, color)."""
    row = ctx.plan[
        (ctx.plan["ldraw_id"] == ldraw_id)
        & (ctx.plan["color_id"] == color_id)
    ].iloc[0]
    vertices, faces = load_part_mesh(ldraw_id, ctx.ldraw_dir)
    rgb_hex = ctx.colors_df.loc[color_id, "rgb"]
    color = ldraw_color_from_rebrickable(
        color_id, row["color_name"], row["material"], rgb_hex
    )
    max_t = part_max_translation(vertices, ctx.space.platform_radius)
    cfg = ctx.space.sample(
        rng, ctx.hdri_pool, ctx.hdri_weights,
        color=color, max_translation=max_t,
    )
    if overrides:
        cfg = dc_replace(cfg, **overrides)
    return cfg, vertices, faces, color


def _viewpoints(rig: TriMirrorRig, cfg: RenderConfig):
    tilted = rig.with_tilt(cfg.rig_tilt)
    vp_cfg = tilted.viewpoint_config()
    zooms = [1.0] + [vp_cfg.mirror_zoom] * len(vp_cfg.mirrors)
    return vp_cfg, zooms


def exposure_for(mesh, cfg: RenderConfig, rig: TriMirrorRig, resolution: int) -> float:
    """Compute the shared exposure across all viewpoints via a mask pass."""
    vp_cfg, zooms = _viewpoints(rig, cfg)
    masks = [
        np.array(mi.render(mi.load_dict(build_scene(
            mesh, cfg, origin=vp, fov=DEFAULT_FOV / z,
            resolution=resolution, mask_mode=True,
        ))))
        for vp, z in zip(vp_cfg.viewpoints, zooms)
    ]
    return compute_exposure(masks)


def render_views(
    mesh,
    cfg: RenderConfig,
    rig: TriMirrorRig,
    resolution: int,
    spp: int,
    view_idxs: list[int] | None = None,
    exposure: float | None = None,
) -> tuple[list[np.ndarray], float]:
    """Render given viewpoints of `mesh` under `cfg`. Returns (list of
    post-processed images in [0,1], exposure). If `exposure` is None,
    it is computed via a mask pass over ALL viewpoints so direct + mirror
    views tonemap consistently."""
    vp_cfg, zooms = _viewpoints(rig, cfg)
    if exposure is None:
        exposure = exposure_for(mesh, cfg, rig, resolution)

    idxs = view_idxs if view_idxs is not None else list(range(len(vp_cfg.viewpoints)))
    imgs = []
    for i in idxs:
        raw = np.array(mi.render(mi.load_dict(build_scene(
            mesh, cfg, origin=vp_cfg.viewpoints[i], fov=DEFAULT_FOV / zooms[i],
            resolution=resolution, spp=spp,
        ))))
        imgs.append(postprocess(raw, cfg, exposure=exposure))
    return imgs, exposure


def render_raw(
    mesh,
    cfg: RenderConfig,
    rig: TriMirrorRig,
    resolution: int,
    spp: int,
    view_idx: int = 0,
) -> np.ndarray:
    """Render a single viewpoint and return the raw mitsuba output (HDR
    float array), no postprocessing. Useful for diagnostics that need
    to compare before/after tonemapping."""
    vp_cfg, zooms = _viewpoints(rig, cfg)
    return np.array(mi.render(mi.load_dict(build_scene(
        mesh, cfg,
        origin=vp_cfg.viewpoints[view_idx],
        fov=DEFAULT_FOV / zooms[view_idx],
        resolution=resolution, spp=spp,
    ))))
```

## Render part

Render one part `n_renders` times. For each render the color is sampled
from the part's real-world color distribution (weighted by `total_qty`),
so a single part's renders cover many colors. Each render produces 4
images (1 direct + 3 mirror views). Images are saved as PNGs and a list
of label rows is returned.

Production renders use a uniform `spp=2048`. An `scripts/spp_sweep.py`
sweep showed that caustics and specular highlights on transparent and
chrome materials stop converging meaningfully past ~256 SPP, while
solid and metallic surfaces still show speckle below 2048. In
principle transparent could be dispatched at 256 to save render
budget, but non-solid colors are a small share of the dataset, so we
keep a single `spp` arg rather than branching on `color.material`.

```python
# | export
def render_part(
    ldraw_id: str,
    color_dist: pd.DataFrame,
    colors_lookup: pd.DataFrame,
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

    max_translation = part_max_translation(vertices, space.platform_radius)
    color_weights = color_dist["total_qty"].to_numpy(dtype=float)
    color_weights /= color_weights.sum()

    for render_idx in render_idxs:
        color_row = color_dist.iloc[rng.choice(len(color_dist), p=color_weights)]
        color_id = int(color_row["color_id"])
        rgb_hex = colors_lookup.loc[color_id, "rgb"]
        color = ldraw_color_from_rebrickable(
            color_id, color_row["color_name"], color_row["material"], rgb_hex
        )

        cfg = space.sample(
            rng, hdri_pool, hdri_weights,
            color=color, max_translation=max_translation,
        )
        mesh, _ = transform_mesh(vertices, faces, cfg)
        imgs, _ = render_views(mesh, cfg, rig, resolution, spp)

        for view_idx, img in enumerate(imgs):
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

Main entry point. Aggregates the render plan by part, iterates parts in
descending `total_qty` order, skips completed renders via the manifest,
and appends to `labels.csv` incrementally so partial runs are
immediately usable.

`n_per_part` is the primary lever for test vs production runs: a small
value (1–5) enables a quick smoke-test; the full dataset uses whatever
N was decided for training. Color is sampled fresh per render from the
part's real-world color distribution, so `n_per_part` renders of a
single part cover many colors rather than pinning one.

`top_n_parts` restricts the iteration to the N most-common parts (by
summed `total_qty`), for quick evaluation runs before committing to the
full catalog. `max_seconds` and `max_parts` are stopping conditions;
the loop exits cleanly after the current part finishes when either
fires. `max_parts` counts parts with actual work done (parts already
fully in the manifest don't count).

```python
# | export
def render_dataset(
    render_plan_path: Path,
    output_dir: Path,
    ldraw_dir: Path,
    rebrickable_dir: Path,
    n_per_part: int = 20,
    spp: int = 2048,
    resolution: int = 512,
    max_seconds: float | None = None,
    max_parts: int | None = None,
    top_n_parts: int | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    data_dir = rebrickable_dir.parent
    ctx = RenderContext.load(data_dir, ldraw_dir=ldraw_dir)
    color_dist_df = pd.read_csv(render_plan_path)
    color_dist_df["ldraw_id"] = color_dist_df["ldraw_id"].astype(str)
    rng = np.random.default_rng(42)

    part_qty = (
        color_dist_df.groupby("ldraw_id")["total_qty"].sum()
        .sort_values(ascending=False)
    )
    if top_n_parts is not None:
        part_qty = part_qty.head(top_n_parts)
    part_color_dists = {
        pid: group for pid, group in color_dist_df.groupby("ldraw_id")
    }

    manifest = load_manifest(output_dir)
    labels_path = output_dir / "labels.csv"
    labels_header = not labels_path.exists()

    t_start = time.monotonic()
    n_rendered = 0
    n_parts_done = 0

    for ldraw_id in part_qty.index:
        if max_seconds and (time.monotonic() - t_start) >= max_seconds:
            break
        if max_parts is not None and n_parts_done >= max_parts:
            break

        pending = [
            i for i in range(n_per_part)
            if (ldraw_id, i) not in manifest
        ]
        if not pending:
            continue

        try:
            vertices, faces = load_part_mesh(ldraw_id, ctx.ldraw_dir)
        except Exception as e:
            print(f"Skipping {ldraw_id}: {e}")
            continue

        t_part_start = time.monotonic()
        label_rows = render_part(
            ldraw_id=ldraw_id,
            color_dist=part_color_dists[ldraw_id],
            colors_lookup=ctx.colors_df,
            vertices=vertices,
            faces=faces,
            render_idxs=pending,
            space=ctx.space,
            rig=ctx.rig,
            hdri_pool=ctx.hdri_pool,
            hdri_weights=ctx.hdri_weights,
            output_dir=output_dir,
            rng=rng,
            spp=spp,
            resolution=resolution,
        )
        part_elapsed = time.monotonic() - t_part_start

        pd.DataFrame(label_rows).to_csv(
            labels_path, mode="a", header=labels_header, index=False
        )
        labels_header = False

        for i in pending:
            update_manifest(output_dir, ldraw_id, i)
            manifest.add((ldraw_id, i))

        n_rendered += len(pending)
        n_parts_done += 1
        print(
            f"part {n_parts_done}: {ldraw_id} "
            f"{part_elapsed:.1f}s ({len(pending)} renders)"
        )

    elapsed = time.monotonic() - t_start
    print(f"Rendered {n_rendered} renders ({n_parts_done} parts) in {elapsed:.1f}s")
```

## Paths and run configuration

Paths and render parameters are the only things that differ between
local and Kaggle runs. Everything else is the same code.

```python
if platform.system() == "Linux":
    # Kaggle: HDRIs + render_plan.csv come from the attached dataset;
    # LDraw + Rebrickable CSVs are fetched upstream at session start.
    DATA_DIR      = Path("/kaggle/working/data")
    OUTPUT_DIR    = Path("/kaggle/working/dataset")
    N_PER_PART    = 20
    MAX_SECONDS   = 11 * 3600  # hard safety under Kaggle's 12h kill
    MAX_PARTS     = None       # set after eval run decides full catalog
    TOP_N_PARTS   = 100        # eval baseline: top parts by total_qty
else:
    DATA_DIR      = Path("../data")
    OUTPUT_DIR    = DATA_DIR / "dataset_test"
    N_PER_PART    = 1
    MAX_SECONDS   = 600    # 10-minute smoke test
    MAX_PARTS     = 10     # small calibration batch
    TOP_N_PARTS   = None
```

## Upstream data (Kaggle only)

Kaggle auto-moderation flags LDraw content (LEGO trademark references in
`.dat` headers), so the LDraw library and Rebrickable CSVs are fetched
from their upstream sources at session start rather than mirrored in a
Kaggle dataset. Only HDRIs and `render_plan.csv` come from the attached
input dataset.

Downloads are idempotent — re-running the cell is a no-op once the
staging directory is populated.

```python
if platform.system() == "Linux":
    import shutil
    import urllib.request
    import zipfile

    KAGGLE_INPUT = next(Path("/kaggle/input").rglob("klods-syn-hdri"))

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    hdri_link = DATA_DIR / "hdri"
    if not (hdri_link.is_symlink() or hdri_link.exists()):
        hdri_link.symlink_to(KAGGLE_INPUT / "hdri")

    rebrickable_dir = DATA_DIR / "rebrickable"
    rebrickable_dir.mkdir(exist_ok=True)

    render_plan_dst = rebrickable_dir / "render_plan.csv"
    if not render_plan_dst.exists():
        shutil.copy(KAGGLE_INPUT / "render_plan.csv", render_plan_dst)

    colors_dst = rebrickable_dir / "colors.csv.gz"
    if not colors_dst.exists():
        urllib.request.urlretrieve(
            "https://cdn.rebrickable.com/media/downloads/colors.csv.gz",
            colors_dst,
        )

    ldraw_root = DATA_DIR / "ldraw"
    if not ldraw_root.exists():
        complete_zip = Path("/tmp/ldraw-complete.zip")
        req = urllib.request.Request(
            "https://library.ldraw.org/library/updates/complete.zip",
            headers={"User-Agent": "Mozilla/5.0"},
        )
        with urllib.request.urlopen(req) as resp, open(complete_zip, "wb") as f:
            shutil.copyfileobj(resp, f)
        with zipfile.ZipFile(complete_zip) as zf:
            zf.extractall(DATA_DIR)
```

## Run

```python
render_dataset(
    render_plan_path=DATA_DIR / "rebrickable" / "render_plan.csv",
    output_dir=OUTPUT_DIR,
    ldraw_dir=DATA_DIR / "ldraw",
    rebrickable_dir=DATA_DIR / "rebrickable",
    n_per_part=N_PER_PART,
    spp=2048,
    resolution=512,
    max_seconds=MAX_SECONDS,
    max_parts=MAX_PARTS,
    top_n_parts=TOP_N_PARTS,
)
```

```python
labels = pd.read_csv(OUTPUT_DIR / "labels.csv")
print(f"Images rendered: {len(labels):,}")
print(f"Unique parts:    {labels['ldraw_id'].nunique():,}")
print(f"Unique colors:   {labels['color_id'].nunique():,}")
print(f"Views per render: {labels.groupby(['ldraw_id','color_id','render_idx']).size().value_counts().to_dict()}")
```
