# Synthetic Training Data: Rendering Pipeline Research

**Status:** Research complete — pipeline selection pending
**Parent:** [Module 4 — Classification Engine](m04-classification-engine.md)

---

## Purpose

Render the LDraw parts library into synthetic training images for the
classification pipeline. This is a data augmentation strategy: we generate
large volumes of labeled images from 3D models, with domain randomization
to bridge the synthetic-to-real gap, supplemented by a small real-world
fine-tuning set captured from the actual scanner.

## Key Research Findings

### The Synthetic-to-Real Gap Is the Central Problem

Every LEGO sorting project that has attempted synthetic training data has
encountered the same issue: models trained on synthetic renders alone
perform poorly on real-world images.

Daniel West (Universal LEGO Sorting Machine) generated 25 million synthetic
images from LDraw models rendered in Blender. The model initially failed
on real parts entirely. He spent months attempting image-to-image
translation approaches before discovering that domain randomization was
the solution. Final accuracy: 93% on parts present in the real training
set, 74% on parts only in the synthetic set. He supplemented with ~200,000
manually-labeled real images.

The LegoBrickClassification project (Theiner et al.) fine-tuned ResNeXt50
on synthetic LDraw renders and found the results unsatisfactory — the
distribution gap between synthetic training data and real-world test
images was too large without randomization.

### Domain Randomization Closes the Gap

Domain randomization trains the model to treat real-world images as "just
another variation" rather than requiring photorealistic synthesis. The
foundational NVIDIA paper (Tremblay et al., CVPR 2018) demonstrated that
synthetic data with non-realistic randomization can rival real data for
training neural networks, and that fine-tuning on real data after synthetic
pretraining outperforms real data alone.

The parameters that matter most (from ablation studies across multiple
projects):

- **Lighting direction, intensity, and color temperature** — fixed lighting
  causes the largest accuracy drop when removed from the randomization.
- **Camera pose and distance** — random viewpoint is essential.
- **Background variation** — random textures, random images, or solid
  colors all work; fixed backgrounds overfit.
- **Material perturbation** — slight variation in roughness, color jitter,
  and specular intensity. Teaches the model to be robust to real-world
  surface variation.
- **Post-processing noise and blur** — simulates sensor imperfections.
  Consistently improves generalization.
- **Distractors** — random geometric shapes or objects in the scene.
  Improves robustness.

### Path Tracing vs Rasterization for Synthetic Data

A comprehensive 2025 study on domain randomization for manufacturing
object detection (using Blender Cycles vs Eevee) found that path tracing
with PBR materials significantly outperformed rasterization — but
specifically for metallic and highly reflective surfaces where accurate
light simulation captures real-world surface reflectance that rasterization
misses.

ABS plastic (the vast majority of LEGO parts) is matte and lightly glossy.
The specular characteristics that path tracing handles well are less
critical for diffuse materials. Transparent, chrome, and pearl LEGO
materials are the exception — these benefit meaningfully from physically
accurate light transport.

**Conclusion:** Rasterization with PBR shaders and aggressive domain
randomization is sufficient for most parts. Path tracing should be reserved
for transparent, chrome, pearl, and metallic material classes where the
reflectance model matters.

### Brickognize: Best Published Results

The Brickognize paper (Vidal et al., MDPI Sensors 2023) achieved the
strongest published results for synthetic LEGO training data:

- Rendering pipeline: Blender + BlenderProc + Cycles path tracer.
- Scene: square room with ceiling-plane emitter, randomized camera,
  brick placement, background, and light position.
- Results: 91.33% AP50 on uncontrolled real-world scenarios, 98.7% on
  controlled scenarios.
- Real data requirement: only 20 manually annotated real images for
  few-shot fine-tuning.
- Key insight: the Brickognize production system uses metric learning
  (image search), not classification. This is more forgiving of
  imperfect synthetic data because the model learns similarity, not
  per-class memorization.

### Our Advantage: Known Scanner Geometry

Most projects render from arbitrary viewpoints because they don't control
the capture environment. We do. Our multi-mirror scanner (Module 3) has
fixed camera and mirror geometry. We can render synthetic views that
exactly match the angles the model will see in production. This collapses
a major source of domain gap — viewpoint variation — by construction.

### A Small Real-World Fine-Tuning Set Is Non-Negotiable

Every successful project required some real-world data. Brickognize needed
20 images. Daniel West used 200,000. The amount varies, but zero real data
has never been sufficient. Plan for capturing a few hundred bricks through
the actual scanner as soon as hardware is operational.

## Existing Rendering Pipelines (What Everyone Uses)

All existing LEGO ML rendering pipelines use the same basic stack.

### Rendering Stack

| Component   | Role                                          |
| ----------- | --------------------------------------------- |
| LDraw       | 3D part library (~70,000 parts as .dat files) |
| ImportLDraw | Blender addon to load LDraw .dat files        |
| Blender     | 3D scene construction and rendering           |
| BlenderProc | Procedural pipeline for domain randomization  |
| Cycles      | Blender's built-in GPU path tracer            |

### Known Rendering Tools (from awesome-lego-machine-learning)

- **BrickRenderer** (2023) — renders realistic training images of LDraw
  pieces, built for the Nexus sorting machine.
- **Lego Rendering Pipeline** (2023) — semi-realistic individual part
  rendering pipeline.
- **BrickRegistration** (2021) — generates synthetic 3D scenes with
  LEGO parts and segmentation information.
- **Lego Renderer for ML Projects** (2020) — Blender/Python utilities
  with tracked camera, scripts for rendering images, normals, and masks.

### Existing Datasets

- **B200C** — 800,000 renders of 200 parts.
- **Photos and rendered images of LEGO bricks** (Nature Scientific Data, 2023) — ~155,000 real photos + ~1.5M renders. Simulates bricks on a
  conveyor belt. Mixed real+synthetic training yielded strong YOLOv5
  results, including generalization to unseen bricks.

### Limitations of the Blender Stack

Blender is a creation tool, not a rendering library. Using it for batch
synthetic data generation means:

- **Startup overhead** — Blender is a ~2GB GUI application driven
  headlessly via `bpy` Python scripting. Each invocation has significant
  initialization cost.
- **Idiosyncratic API** — scene construction goes through Blender's
  internal scene graph, which is designed for interactive editing, not
  programmatic batch rendering.
- **Serialization overhead** — geometry passes through Blender's internal
  representation before reaching the renderer.
- **Speed** — Cycles path tracing is capable but not optimized for the
  batch throughput use case. Eevee rasterization is faster but still
  carries Blender's scene graph overhead.

For projects generating thousands to low millions of images, Blender works.
For our scale (70,000 parts × multiple views × multiple colors × multiple
randomizations), a purpose-built pipeline would be significantly faster.

## Pipeline Option 1: ldr_tools + Mitsuba 3

A Python-first approach that replaces Blender with a proper rendering
library.

### Architecture

```
LDraw .dat files
  → ldr_tools (Rust, Python bindings via PyO3)
    → triangle mesh + normals + color/material IDs
      → Mitsuba 3 scene (Python dict)
        → mi.render() → numpy array → training image
```

### Components

**ldr_tools** (ScanMountGoat) — Rust-based LDraw processing library with
Python bindings. Handles the full LDraw spec: MPD, subpart resolution,
BFC normals, color resolution. Outputs clean triangle meshes with material
assignments. Uses numpy arrays to minimize Python conversion overhead.

**Mitsuba 3** (EPFL) — research-grade physically based renderer. Python-
first design: scenes constructed as Python dicts, no intermediate file
formats. JIT-compiles rendering code via Dr.Jit. Backends: LLVM (CPU),
CUDA/OptiX (NVIDIA GPU). Supports macOS aarch64 (M-series Macs). Installs
via `pip install mitsuba`. Smooth/rough plastic BSDFs available out of the
box for ABS material simulation.

### Strengths

- No Blender dependency. No subprocess calls. No scene file serialization.
- Pure Python control of the entire pipeline — domain randomization is a
  tight loop: perturb params → `mi.render()` → write image.
- Mitsuba's LLVM backend runs on M4 Mac for development; CUDA/OptiX
  backend available for rented GPU batch runs.
- Differentiable rendering available if needed for future domain adaptation
  experiments.

### Limitations

- Path tracing only — no rasterization fast path. Every image pays the
  cost of physically based light transport, even for matte ABS plastic
  where it provides minimal benefit over rasterization.
- Python overhead in the rendering loop (mitigated by Mitsuba's JIT
  compilation, but still present in scene construction and I/O).
- ldr_tools Python bindings add a Rust→Python boundary that wouldn't exist
  in a Rust-native or C++-native pipeline.

### Estimated Throughput

Path tracing at 512×512 with ~64 samples per pixel on an M4 Mac (LLVM):
roughly 5–20 images/second for simple LEGO part scenes (low poly, simple
materials). Faster on rented NVIDIA GPU with OptiX backend.

## Pipeline Option 2: ldr_tools + wgpu + ChameleonRT

A performance-first approach using Rust for rasterization bulk rendering
and C++ for selective path tracing.

### Architecture

```
LDraw .dat files
  → ldr_tools (Rust)
    → triangle mesh + normals + color/material IDs
      ├── FAST PATH: wgpu rasterizer (Rust)
      │   PBR fragment shader, domain randomization loop.
      │   Output: bulk training images (~95% of dataset).
      │
      └── QUALITY PATH: ChameleonRT (C++)
          Disney BSDF path tracer.
          Output: transparent/chrome/pearl parts, reference
          embeddings, validation renders.
```

### Components

**ldr_tools** (ScanMountGoat) — same Rust LDraw parser as Option 1, but
used directly as a Rust crate with no Python boundary. Zero serialization
overhead from parse to render.

**wgpu** (Rust) — cross-platform GPU abstraction over Metal (macOS),
Vulkan, DX12. Headless rendering with no window required. PBR fragment
shaders for ABS plastic simulation. Domain randomization parameters
(camera, lighting, material, background, noise) controlled
programmatically per frame. The entire rendering loop — mesh loading,
scene setup, GPU dispatch, framebuffer readback — is a single Rust binary
with no external dependencies.

**ChameleonRT** (Will Usher, C++) — production-quality path tracer with
backends for Metal, OptiX, Vulkan, Embree, and DXR. Disney BSDF material
model (roughness/metallic workflow). Loads OBJ and glTF natively. Intel
forked it as their Real-time Path Tracing Research Framework — it is the
industry reference for cross-backend path tracing benchmarks. Metal
backend runs on M4 Mac with hardware ray tracing; OptiX backend uses
NVIDIA RT cores for maximum throughput on rented GPU instances.

### Interface Between Components

ldr_tools outputs meshes that the wgpu rasterizer consumes directly in
Rust (shared memory, no I/O). For ChameleonRT, ldr_tools exports OBJ or
glTF files to disk — both formats ChameleonRT loads natively. The domain
randomization parameters (camera pose, light configuration, material
perturbations) are shared between both render paths via a common
configuration, ensuring the rasterized and path-traced datasets are
statistically compatible.

### Strengths

- **Maximum throughput for the bulk case.** GPU rasterization at 512×512
  for low-poly LEGO parts: estimated 100–500+ images/second on M4 Mac.
  This is 10–100× faster than any path tracing approach.
- **Path tracing where it matters.** ChameleonRT provides physically
  correct rendering for the material classes that need it (transparent,
  chrome, pearl), without imposing that cost on the 95% of parts that
  are matte ABS.
- **No Python in the hot path.** Rust and C++ only. No interpreter
  overhead, no GIL, no serialization boundaries in the rendering loop.
- **Hardware ray tracing on both dev and production platforms.** Metal
  on M4 Mac, OptiX on rented NVIDIA. ChameleonRT abstracts the backend.
- **Single-binary rasterizer.** The Rust/wgpu bulk renderer has zero
  external dependencies beyond the GPU driver.

### Limitations

- Two rendering codebases (Rust + C++) to maintain.
- ChameleonRT integration requires OBJ/glTF export as an intermediate
  step (acceptable overhead for the selective path-tracing use case).
- wgpu PBR shaders must be written and validated to match ChameleonRT's
  Disney BSDF closely enough that the two datasets are compatible for
  mixed training.

### Estimated Throughput

**wgpu rasterizer:** 100–500 images/second at 512×512 on M4 Mac. 70,000
parts × 20 views × 5 colors = 7 million images in under a day on a single
machine.

**ChameleonRT path tracer:** 1–10 images/second at 512×512 on M4 Mac
(Metal). Significantly faster on rented NVIDIA with OptiX (hardware RT
cores). Used for a small fraction of the total dataset.

## Pipeline Comparison

| Factor              | Blender Stack          | Mitsuba 3                | wgpu + ChameleonRT                |
| ------------------- | ---------------------- | ------------------------ | --------------------------------- |
| Language            | Python (bpy)           | Python + Rust (bindings) | Rust + C++                        |
| LDraw parsing       | ImportLDraw addon      | ldr_tools (PyO3)         | ldr_tools (native)                |
| Bulk renderer       | Eevee (rasterization)  | Mitsuba (path tracing)   | wgpu (rasterization)              |
| Quality renderer    | Cycles (path tracing)  | Mitsuba (path tracing)   | ChameleonRT (path tracing)        |
| macOS Metal support | Yes (Blender native)   | Yes (LLVM backend)       | Yes (wgpu + ChameleonRT Metal)    |
| NVIDIA GPU support  | Yes (Cycles CUDA)      | Yes (OptiX backend)      | Yes (ChameleonRT OptiX)           |
| Bulk throughput     | ~10–50 img/sec (Eevee) | ~5–20 img/sec            | ~100–500 img/sec                  |
| External deps       | Blender (~2GB)         | pip install mitsuba      | None (wgpu) / CMake (ChameleonRT) |
| Community precedent | All existing projects  | None for LEGO            | None for LEGO                     |
| Risk                | Low (proven path)      | Medium                   | Higher (novel)                    |

## Recommendation

Option 2 (wgpu + ChameleonRT) for maximum throughput. The 10–100×
speed advantage for bulk rendering directly translates to faster
iteration on the training pipeline — more experiments per day, more
randomization coverage, faster feedback on what domain randomization
parameters matter for our specific scanner geometry.

Option 1 (Mitsuba 3) is the pragmatic fallback if the Rust/C++ pipeline
proves too costly to build and maintain. It is still a large improvement
over the Blender stack.

The Blender stack (Option 0) is the safe fallback. If we get stuck on
either custom pipeline, we can always fall back to BlenderProc + Cycles
and still produce viable training data — every other LEGO sorting project
has done exactly this.

## References

- Brickognize paper: Vidal et al., MDPI Sensors 2023.
  https://www.mdpi.com/1424-8220/23/4/1898
- Domain randomization: Tremblay et al., CVPR 2018.
  https://arxiv.org/abs/1804.06516
- DR for manufacturing (path tracing vs rasterization ablation):
  https://arxiv.org/html/2506.07539v1
- Photos and rendered images of LEGO bricks: Nature Scientific Data 2023.
  https://www.nature.com/articles/s41597-023-02682-2
- Daniel West synthetic data:
  https://www.theregister.com/2019/12/06/lego_sorting_machine/
- LegoBrickClassification: https://github.com/jtheiner/LegoBrickClassification
- awesome-lego-machine-learning: https://github.com/360er0/awesome-lego-machine-learning
- ldr_tools: https://github.com/ScanMountGoat/ldr_tools_blender
- Mitsuba 3: https://github.com/mitsuba-renderer/mitsuba3
- ChameleonRT: https://github.com/Twinklebear/ChameleonRT
- Intel RTPTRF (ChameleonRT fork):
  https://github.com/intel/RealTimePathTracingResearchFramework

