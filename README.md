# klods-syn

**Computer vision models and synthetic training data for LEGO brick
recognition.**

klodssyn (Danish: "brick vision") is an open-source toolkit for generating
synthetic training images from the LDraw parts library and training models
for LEGO brick classification and detection.

Built to serve [sorted-studs](https://github.com/restlessronin/sorted-studs)
but usable independently by any LEGO ML project.

---

## What It Does

- Parse the LDraw parts library into renderable triangle meshes
- Render synthetic training images with domain randomization
- Train and evaluate classification models (part ID from cropped image)
- Train and evaluate detection models (locate and identify parts in scene)
- Export trained models for deployment (CoreML, ONNX, TensorRT)

## Approach

**Rendering:** LDraw parts are parsed via ldr_tools (Rust) and rendered
through two paths — a fast wgpu rasterizer for bulk image generation and
ChameleonRT path tracing for materials that require physically accurate
light transport (transparent, chrome, pearl). Domain randomization
(lighting, camera, material, background) bridges the synthetic-to-real
gap.

**Classification:** ConvNeXt V2 Tiny as a metric learning feature
encoder, trained with contrastive loss. Classification by nearest-neighbor
search against a reference embedding database. Designed for the
sorted-studs multi-view scanning station but applicable to any single-part
identification task.

**Detection:** (Future) Object detection for identifying multiple bricks
in a scene, enabling pile-scanning workflows without mechanical
singulation.

## Documentation

- [Architecture](docs/architecture.md) — system design, rendering
  pipeline, model architecture, interface between components
- [Synthetic Rendering Pipeline](docs/synthetic-rendering-pipeline.md) —
  research on rendering approaches, domain randomization, what works

## Relationship to sorted-studs

klodssyn produces trained models. sorted-studs deploys them.

klodssyn is a development dependency of sorted-studs, not a runtime
dependency. The deployed system ships only the exported model weights.

## References

- [sorted-studs](https://github.com/restlessronin/sorted-studs) —
  the sorting machine that uses these models
- [LDraw](https://ldraw.org) — 3D parts library
- [ldr_tools](https://github.com/ScanMountGoat/ldr_tools_blender) —
  Rust LDraw parser
- [ChameleonRT](https://github.com/Twinklebear/ChameleonRT) —
  multi-backend path tracer
- [awesome-lego-machine-learning](https://github.com/360er0/awesome-lego-machine-learning) —
  curated list of LEGO ML projects

---

_klodssyn is not affiliated with the LEGO Group. LEGO® is a trademark
of the LEGO Group, which does not sponsor, authorize, or endorse this
project._
