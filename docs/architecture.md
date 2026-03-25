# klodssyn

**Computer vision models and synthetic training data for LEGO brick
recognition.**

klodssyn (Danish: "brick vision") is an open-source toolkit for generating
synthetic training images from the LDraw parts library and training models
for LEGO brick classification and detection. It is designed to serve
sorted-studs but is usable independently by any LEGO ML project.

---

## Scope

### What klodssyn does

- Parse the LDraw parts library into renderable triangle meshes
- Render synthetic training images with domain randomization
- Train and evaluate classification models (part ID from cropped image)
- Train and evaluate detection models (locate and identify parts in scene)
- Export trained models for deployment (CoreML, ONNX, TensorRT)

### What klodssyn does not do

- Control physical hardware (that's sorted-studs)
- Manage inventory or interface with Rebrickable (that's sorted-studs)
- Provide a user-facing application (that's sorted-studs)

## Architecture

### Rendering Pipeline

Two render paths, shared mesh loading, shared domain randomization:

```
LDraw .dat files
  → ldr_tools (Rust) → triangle mesh + normals + color/material
    │
    ├── wgpu rasterizer (Rust)         ← bulk path, ~100-500 img/sec
    │   PBR shaders, domain randomization per frame
    │   Output: single-part images for classification
    │           multi-part scenes for detection (future)
    │
    └── ChameleonRT (C++)              ← quality path, ~1-10 img/sec
        Disney BSDF path tracing
        Metal (M4 Mac) / OptiX (rented NVIDIA GPU)
        Output: transparent/chrome/pearl parts
                reference embeddings for metric learning DB
                validation renders
```

### Interface Between Rendering and Training

The rendering pipeline produces a dataset on disk in a standard format:

**Classification dataset:**

```
dataset/
  images/
    3001_red_top_001.png
    3001_red_side_001.png
    ...
  labels.csv    # image_path, part_id, color_id, view_angle, render_method
```

**Detection dataset (future):**

```
dataset/
  images/
    scene_001.png
    ...
  annotations/
    scene_001.json   # COCO format: bounding boxes, class IDs, masks
```

The training code consumes these directories. No runtime dependency
between the Rust/C++ rendering code and the Python training code.

### Models

**Classification (v1 — sorted-studs Phase 3):**
ConvNeXt V2 Tiny as a metric learning feature encoder. Trained with
contrastive or triplet loss. Classification by nearest-neighbor search
against a reference embedding database. See sorted-studs Module 4 spec
for architecture rationale.

**Detection (future):**
Architecture TBD. Likely YOLO-family or Mask R-CNN. Requires scene
composition with physics-consistent brick placement (not yet designed).
Enables pile-scanning workflows without mechanical singulation.

### Scene Composition (Future)

For detection training data, bricks must be rendered in multi-part scenes
with physically plausible placement (no floating, no interpenetration).
This requires:

- Rigid body physics simulation (Rapier in Rust, or Bullet/PhysX in C++)
- Random part selection and drop onto a surface
- Settle simulation until stable
- Automatic bounding box and segmentation mask generation

Not designed yet. Will be added when detection work begins.

## Build

### Rendering Pipeline

LDraw mesh loading via ldr_tools (Rust), rendering via Mitsuba 3 (Python):
```
LDraw .dat files
  → ldr_tools (Rust) → triangle mesh + normals + color/material
    → Mitsuba 3 (Python)
        scalar_rgb variant         ← bulk path, direct integrator
        spectral variant           ← quality path, path tracing
        Output: single-part images for classification
                multi-part scenes for detection (future)
```

Multi-view capture uses a tri-mirror rig geometry: camera above the
target on the vertical axis, three mirrors on a ring below, each
reflecting a distinct virtual viewpoint. Mirror positions and
orientations are optimized to maximize angular spread across all
viewpoints. See `rendering.viewpoints` module.

### Training (Python)

```
cd models
pip install -e .
```

## Relationship to sorted-studs

klodssyn produces trained models. sorted-studs deploys them.

```
klodssyn                          sorted-studs
  rendering → dataset               Module 3: scanner captures images
  training → model weights    →      Module 4: loads weights, runs inference
  export → .mlpackage/.onnx         Module 9: orchestration
```

sorted-studs references klodssyn as a development dependency, not a
runtime dependency. The deployed iPhone/Mac app ships only the exported
model weights, not the training pipeline.

## References

- Rendering pipeline research: docs/synthetic-rendering-pipeline.md
- sorted-studs Module 4: classification architecture and training strategy
- LDraw parts library: https://ldraw.org
- ldr_tools: https://github.com/ScanMountGoat/ldr_tools_blender
- Mitsuba 3: https://mitsuba.readthedocs.io/
- ChameleonRT: https://github.com/Twinklebear/ChameleonRT (future: bulk path tracing)

```

Commit for when this becomes a real repo:
```

docs: initial klodssyn architecture

Co-authored-by: Claude Opus 4.6 <claude-opus-4-6@llm-context>
