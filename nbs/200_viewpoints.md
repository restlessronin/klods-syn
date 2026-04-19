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

# Viewpoint Geometry

Multi-view capture configuration for scanner mirror rigs.

```python
# | default_exp rendering.viewpoints
```

```python
# | export
from dataclasses import dataclass

import numpy as np
from scipy.optimize import differential_evolution
from scipy.spatial.transform import Rotation

type Vec3 = tuple[float, float, float]
```

## Core types

```python
# | export
@dataclass(frozen=True)
class PhysicalMirror:
    center: Vec3
    normal: Vec3


@dataclass(frozen=True)
class ViewpointConfig:
    camera: Vec3
    viewpoints: tuple[Vec3, ...]
    mirrors: tuple[PhysicalMirror, ...]
    target: Vec3 = (0.0, 0.0, 0.0)

    @property
    def mirror_zoom(self) -> float:
        direct_dist = np.linalg.norm(np.array(self.camera) - np.array(self.target))
        mirror_dist = np.linalg.norm(
            np.array(self.viewpoints[1]) - np.array(self.target)
        )
        return mirror_dist / direct_dist
```

## Angular spread analysis

```python
# | export
def angular_spread(config: ViewpointConfig) -> dict[str, float]:
    pts = np.array(config.viewpoints)
    dirs = pts - np.array(config.target)
    dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)

    cosines = np.clip(dirs @ dirs.T, -1.0, 1.0)
    angles = np.degrees(np.arccos(cosines))

    pairwise = angles[np.triu_indices(len(dirs), k=1)]

    return {
        "min": float(pairwise.min()),
        "max": float(pairwise.max()),
        "mean": float(pairwise.mean()),
        "angles": pairwise.tolist(),
    }
```

## Tri-mirror rig

```python
# | export
@dataclass(frozen=True)
class TriMirrorRig:
    camera_dist: float
    ring_radius: float
    ring_height: float
    tilt: Vec3 = (0.0, 0.0, 0.0)

    def with_tilt(self, tilt: Vec3) -> "TriMirrorRig":
        return TriMirrorRig(self.camera_dist, self.ring_radius, self.ring_height, tilt)

    def viewpoint_config(self, target: Vec3 = (0.0, 0.0, 0.0)) -> ViewpointConfig:
        cam = np.array([0.0, self.camera_dist, 0.0])
        centers = self._mirror_centers()

        rotvec = np.array(self.tilt)
        if np.linalg.norm(rotvec) > 1e-12:
            rot = Rotation.from_rotvec(rotvec)
            cam = rot.apply(cam)
            centers = [rot.apply(c) for c in centers]

        tgt = np.array(target)
        mirrors = []
        viewpoints = [tuple(cam.tolist())]
        for c in centers:
            d_cam = cam - c
            d_tgt = tgt - c
            n = d_cam / np.linalg.norm(d_cam) + d_tgt / np.linalg.norm(d_tgt)
            n = n / np.linalg.norm(n)
            virtual = cam - 2 * np.dot(cam - c, n) * n
            mirrors.append(
                PhysicalMirror(
                    center=tuple(c.tolist()),
                    normal=tuple(n.tolist()),
                )
            )
            viewpoints.append(tuple(virtual.tolist()))
        return ViewpointConfig(
            camera=viewpoints[0],
            viewpoints=tuple(viewpoints),
            mirrors=tuple(mirrors),
            target=target,
        )

    def _mirror_centers(self) -> list[np.ndarray]:
        return [
            np.array(
                [
                    self.ring_radius * np.cos(2 * np.pi * i / 3),
                    -self.ring_height,
                    self.ring_radius * np.sin(2 * np.pi * i / 3),
                ]
            )
            for i in range(3)
        ]

    @staticmethod
    def optimize(
        camera_dist: float = 380.0,
        r_bounds: tuple[float, float] = (20.0, 300.0),
        h_bounds: tuple[float, float] = (10.0, 200.0),
    ) -> "TriMirrorRig":
        def objective(params: np.ndarray) -> float:
            rig = TriMirrorRig(camera_dist, params[0], params[1])
            config = rig.viewpoint_config()
            spread = angular_spread(config)
            return -spread["min"]

        result = differential_evolution(
            objective,
            bounds=[r_bounds, h_bounds],
            seed=42,
            tol=1e-10,
            maxiter=5000,
        )
        return TriMirrorRig(camera_dist, result.x[0], result.x[1])
```

## Default scanner configuration

```python
rig = TriMirrorRig.optimize(camera_dist=380.0)
config = rig.viewpoint_config()
spread = angular_spread(config)

print(f"Ring radius: {rig.ring_radius:.1f}, Ring height: {rig.ring_height:.1f}")
print(
    f"Angular spread — min: {spread['min']:.1f}° max: {spread['max']:.1f}° mean: {spread['mean']:.1f}°"
)
print(f"Pairwise: {[f'{a:.1f}°' for a in spread['angles']]}")

print(f"\nCamera: {tuple(np.round(config.camera, 1))}")
print("Viewpoints:")
for i, v in enumerate(config.viewpoints):
    label = "direct" if i == 0 else f"mirror {i}"
    print(f"  {label}: {tuple(np.round(v, 1))}")
print("Physical mirrors:")
for i, m in enumerate(config.mirrors):
    print(
        f"  {i}: center={tuple(np.round(m.center, 1))} normal={tuple(np.round(m.normal, 3))}"
    )
```

## Rig visualization

```python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def plot_rig(config: ViewpointConfig, mirror_size: float = 40.0, ax=None):
    if ax is None:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection="3d")

    t = np.array(config.target)
    c = np.array(config.camera)

    ax.scatter(*t, color="black", s=80, marker="x", label="target")

    sz = 60.0
    yp = t[1] - 5.0
    corners_p = [
        [t[0] - sz, yp, t[2] - sz],
        [t[0] + sz, yp, t[2] - sz],
        [t[0] + sz, yp, t[2] + sz],
        [t[0] - sz, yp, t[2] + sz],
    ]
    ax.add_collection3d(
        Poly3DCollection(
            [corners_p],
            alpha=0.3,
            facecolor="tab:gray",
            edgecolor="tab:gray",
            label="platform",
        )
    )

    ax.scatter(*c, color="tab:blue", s=100, zorder=5, label="camera")
    ax.quiver(*t, *(c - t), arrow_length_ratio=0.05, color="tab:blue", alpha=0.6)

    for i, (vp, m) in enumerate(zip(config.viewpoints[1:], config.mirrors)):
        vp = np.array(vp)
        mc = np.array(m.center)
        mn = np.array(m.normal)
        mn = mn / np.linalg.norm(mn)

        ax.scatter(
            *vp,
            color="tab:orange",
            s=100,
            zorder=5,
            label="virtual" if i == 0 else None,
        )

        ax.plot(
            [vp[0], mc[0]],
            [vp[1], mc[1]],
            [vp[2], mc[2]],
            color="tab:orange",
            linestyle="--",
            alpha=0.5,
        )

        ax.plot(
            [c[0], mc[0]],
            [c[1], mc[1]],
            [c[2], mc[2]],
            color="tab:blue",
            linestyle=":",
            alpha=0.3,
        )

        arb = np.array([1, 0, 0]) if abs(mn[0]) < 0.9 else np.array([0, 1, 0])
        u_vec = np.cross(mn, arb)
        u_vec = u_vec / np.linalg.norm(u_vec)
        v_vec = np.cross(mn, u_vec)

        corners = [
            mc + s1 * mirror_size * u_vec + s2 * mirror_size * v_vec
            for s1, s2 in [(-1, -1), (1, -1), (1, 1), (-1, 1)]
        ]
        ax.add_collection3d(
            Poly3DCollection(
                [corners],
                alpha=0.25,
                facecolor="tab:green",
                edgecolor="tab:green",
                label="mirror" if i == 0 else None,
            )
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    ax.set_aspect("equal")
    return ax


plot_rig(config)
plt.title("Optimized tri-mirror rig")
plt.show()
```

```python

```
