# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
uv sync

# Launch notebooks
uv run jupyter lab

# Export notebooks to klods_syn/ package
uv run nbdev-export
```

## Architecture

klods-syn is split into two independent pipelines with no runtime dependency between them:

**Rendering:** LDraw `.dat` files → `ldr_tools_py.so` (prebuilt Rust extension, in repo root) → triangle meshes → Mitsuba 3 (Python, `klods_syn/rendering/`). Fast Rust/C++ paths (wgpu rasterizer, ChameleonRT) exist externally for bulk production renders but are not in this repo.

**Training (Python — this repo):** Consumes rendered datasets. Current model: frozen DINOv3 ViT-S/16 backbone + supervised contrastive projection head, with k-NN retrieval against a reference embedding database. Future: multi-view fusion layer, then detection with YOLO or Mask R-CNN. See `docs/training-strategy.md` for architecture rationale and decisions.

### Dataset Format

Classification datasets consumed by training code:
```
dataset/
  images/
    <part_id>_<color>_<view>_<idx>.png
  labels.csv    # image_path, part_id, color_id, view_angle, render_method
```
Directory structure for real-photo datasets: `category/part_number/` with ≥5 images per class.

### Notebook Series

Notebooks in `nbs/` are organized by series prefix:
- `00x_`: catalog and rendering assets (parts catalog, HDRIs)
- `1xx_`: datasets (real-world data sources: Gdańsk, Brickognize, Paco Garcia)
- `2xx_`: rendering pipeline (viewpoints, Mitsuba rendering, domain randomization, dataset orchestration at 203)
- `3xx_`: training and evaluation (baseline eval, metric learning, inference, domain gap)

### Notebook-Driven Development (nbdev + jupytext)

- **Source of truth is `.md` files** in `nbs/`, not `.ipynb` files (`.ipynb` are gitignored)
- nbdev exports annotated cells (`# | export`) to `klods_syn/`
- Cell annotation `# | default_exp <module>` sets the target module for a notebook
- Edit `.md` files directly; jupytext syncs to `.ipynb` on open

**Converting .md → .ipynb:** Use `uv run jupytext --to ipynb --update <file.md>`. The `--update` flag preserves existing output cells when the input cell is unchanged. Do not use `jupytext --sync`.

**After executing a notebook:** Do not sync .ipynb back to .md. Execution only produces outputs; input cells (the .md source of truth) are unchanged.

## Code Style

**Notebooks:**
- One logical concept per cell
- Zero comments in code cells — use markdown cells for context/rationale
- Markdown explains *why*, not *what*

**Python:**
- `@dataclass(frozen=True)` as default class pattern; delegate complex construction to `@staticmethod create()`
- Comprehensive type hints; use `jaxtyping` for array shapes (`Float[Array, "batch features"]`)
- `pathlib.Path` over string paths
- Imports always at module top (never function-level)

## Commit Messages

Single-line conventional commit title with co-author:

```
<type>: <description>

Co-authored-by: Claude Sonnet 4.6 <claude-sonnet-4.6@claude-cli>
```

Keep the body short. Don't enumerate file-by-file changes — the diff
already shows that. Use the body only for context not visible from the
diff itself: the *why*, key decisions, build/deploy commands worth
recording, or non-obvious follow-ups.
