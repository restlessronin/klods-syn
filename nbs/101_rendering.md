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

# Fan-triangulate: tris pass through, quads become two triangles
triangles = []
for s, n in zip(starts, sizes):
    for i in range(1, n - 1):
        triangles.append([idx[s], idx[s + i], idx[s + i + 1]])

faces = np.array(triangles, dtype=np.uint32)
faces = faces[:, [0, 2, 1]]
print(f"Triangles: {len(faces)}")  # expect 700
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

## Render

```python
bsdf = mi.load_dict(
    {
        "type": "plastic",
        "diffuse_reflectance": {"type": "rgb", "value": [0.8, 0.1, 0.1]},
        "int_ior": 1.49,
    }
)
mesh.set_bsdf(bsdf)

scene = mi.load_dict(
    {
        "type": "scene",
        "integrator": {"type": "direct"},
        "sensor": {
            "type": "perspective",
            "fov": 35,
            "to_world": mi.ScalarTransform4f.look_at(
                origin=[100, 50, 100],
                target=[0, -5, 0],
                up=[0, 1, 0],
            ),
            "film": {"type": "hdrfilm", "width": 512, "height": 512},
            "sampler": {"type": "independent", "sample_count": 64},
        },
        "light": {
            "type": "constant",
            "radiance": {"type": "spectrum", "value": 1.5},
        },
        "brick": mesh,
    }
)

image = mi.render(scene)
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
