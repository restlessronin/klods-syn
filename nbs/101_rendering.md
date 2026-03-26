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

# Synthetic Rendering Pipeline

Render LDraw parts to synthetic training images using Mitsuba 3.
First milestone: load a single part via ldr_tools and render it.

```python
import sys

sys.path.insert(0, "..")
import numpy as np
import mitsuba as mi

mi.set_variant("scalar_rgb")

import ldr_tools_py as ldr
```

## Load an LDraw part

```python
from pathlib import Path

LDRAW_DIR = str(Path("../data/ldraw").resolve())

settings = ldr.GeometrySettings()
scene = ldr.load_file(f"{LDRAW_DIR}/parts/3001.dat", LDRAW_DIR, [], settings)

key = list(scene.geometry_cache.keys())[0]
geom = scene.geometry_cache[key]

print(f"Part: {scene.root_node.name}")
print(f"Vertices: {geom.vertices.shape}")
print(f"Faces: {geom.face_sizes.shape}")
```

## Convert to Mitsuba mesh

LDraw uses a right-handed coordinate system where **-Y is up**.
Mitsuba 3 uses +Y up. Flip Y and reverse face winding to compensate.

```python
vertices = geom.vertices.astype(np.float32)
vertices[:, 1] *= -1
idx = geom.vertex_indices
starts = geom.face_start_indices
sizes = geom.face_sizes

triangles = []
for s, n in zip(starts, sizes):
    for i in range(1, n - 1):
        triangles.append([idx[s], idx[s + i], idx[s + i + 1]])

faces = np.array(triangles, dtype=np.uint32)
faces = faces[:, [0, 2, 1]]
print(f"Triangles: {len(faces)}")
```

```python
mesh = mi.Mesh(
    "brick",
    vertex_count=len(vertices),
    face_count=len(faces),
)

mesh_params = mi.traverse(mesh)
mesh_params["vertex_positions"] = vertices.ravel()
mesh_params["faces"] = faces.ravel()
mesh_params.update()

print(f"Mitsuba mesh: {mesh}")
```

## Scene helpers

```python
type Vec3 = tuple[float, float, float]


def make_bsdf(color: Vec3 = (0.85, 0.05, 0.05), ior: float = 1.53) -> dict:
    return {
        "type": "plastic",
        "diffuse_reflectance": {"type": "rgb", "value": list(color)},
        "int_ior": ior,
    }


def make_scene(
    mesh: mi.Mesh,
    origin: Vec3,
    target: Vec3 = (0, -5, 0),
    fov: float = 35,
    resolution: int = 512,
    spp: int = 128,
) -> mi.Scene:
    return mi.load_dict(
        {
            "type": "scene",
            "integrator": {"type": "path", "max_depth": 4},
            "sensor": {
                "type": "perspective",
                "fov": fov,
                "to_world": mi.ScalarTransform4f.look_at(
                    origin=list(origin),
                    target=list(target),
                    up=[0, 1, 0],
                ),
                "film": {"type": "hdrfilm", "width": resolution, "height": resolution},
                "sampler": {"type": "independent", "sample_count": spp},
            },
            "key_light": {
                "type": "directional",
                "direction": [-1, -2, -0.5],
                "irradiance": {"type": "spectrum", "value": 3.0},
            },
            "fill_light": {
                "type": "directional",
                "direction": [1, -1, 0.5],
                "irradiance": {"type": "spectrum", "value": 1.0},
            },
            "background": {
                "type": "constant",
                "radiance": {"type": "spectrum", "value": 0.4},
            },
            "brick": mesh,
        }
    )
```

## Render

```python
bsdf = mi.load_dict(make_bsdf())
mesh.set_bsdf(bsdf)

image = mi.render(make_scene(mesh, origin=(100, 50, 100)))
```

```python
import matplotlib.pyplot as plt

bitmap = mi.util.convert_to_bitmap(image)
fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(bitmap)
ax.set_axis_off()
ax.set_title("3001 — 2×4 Brick")
plt.show()
```

## Multi-view rendering

```python
from klods_syn.rendering.viewpoints import TriMirrorRig, angular_spread

rig = TriMirrorRig.optimize(camera_dist=250.0).with_tilt((np.pi / 4, 0.0, 0.0))
config = rig.viewpoint_config()

zooms = [1.0] + [config.mirror_zoom] * len(config.mirrors)

images = [
    mi.render(make_scene(mesh, origin=vp, fov=35 / zoom))
    for vp, zoom in zip(config.viewpoints, zooms)
]

labels = ["direct"] + [f"mirror {i+1}" for i in range(len(config.mirrors))]

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
for ax, img, label in zip(axes, images, labels):
    ax.imshow(mi.util.convert_to_bitmap(img))
    ax.set_axis_off()
    ax.set_title(label)
fig.suptitle("3001 — 2×4 Brick — Multi-view")
plt.tight_layout()
plt.show()
```

```python

```
