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

# Domain Randomization

Parameterized synthetic data generation for training.
Builds on the rendering pipeline from `101_rendering.md`.

```python
# | default_exp rendering.domain_randomization
```

```python
# | export
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

type Vec3 = tuple[float, float, float]
```

## LDraw color table

Parse `LDConfig.ldr` for the full LEGO color palette.
Each entry carries RGB, alpha, and material class — we need
the material class to route to different BSDFs.

Reference: [LDraw Color Definition Language](https://www.ldraw.org/article/547.html)

```python
# | export
@dataclass(frozen=True)
class LDrawColor:
    code: int
    name: str
    rgb: Vec3
    alpha: float
    material: str  # solid, transparent, pearl, chrome, metallic, rubber

    @property
    def is_special(self) -> bool:
        return self.material in ("transparent", "pearl", "chrome", "metallic")


def _parse_hex(h: str) -> Vec3:
    h = h.lstrip("#")
    return tuple(int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4))


def parse_ldconfig(path: Path) -> tuple[LDrawColor, ...]:
    colors = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line.startswith("0 !COLOUR"):
            continue
        tokens = line.split()
        name = tokens[2]
        kv = {
            tokens[i]: tokens[i + 1]
            for i in range(3, len(tokens) - 1, 2)
            if tokens[i].startswith(("CODE", "VALUE", "ALPHA", "EDGE"))
        }
        code = int(kv["CODE"])
        rgb = _parse_hex(kv["VALUE"])
        alpha = int(kv.get("ALPHA", "255")) / 255.0

        material = "solid"
        if alpha < 1.0:
            material = "transparent"
        elif "CHROME" in line:
            material = "chrome"
        elif "PEARLESCENT" in line:
            material = "pearl"
        elif "METAL" in line:
            material = "metallic"
        elif "RUBBER" in line:
            material = "rubber"

        colors.append(LDrawColor(code, name, rgb, alpha, material))
    return tuple(colors)
```

```python
LDRAW_DIR = Path("../data/ldraw").resolve()
color_table = parse_ldconfig(LDRAW_DIR / "LDConfig.ldr")
print(f"Colors: {len(color_table)}")

by_material = {}
for c in color_table:
    by_material.setdefault(c.material, []).append(c)
for mat, cs in sorted(by_material.items()):
    print(f"  {mat}: {len(cs)}")
```

## HDRI environment maps

[Poly Haven](https://polyhaven.com/) provides CC0 HDRIs via a public API.
We fetch 1K resolution maps (~1–6MB each) and cache them locally.

Category quotas bias toward low dynamic range environments (indoor,
studio) that are closer to the scanner's operating conditions, while
keeping some high-DR outdoor scenes for robustness.

API: `https://api.polyhaven.com` — requires a `User-Agent` header.
All requests require unique User-Agent per [API terms](https://polyhaven.com/our-api).

```python
# | export
import json
import urllib.request

POLYHAVEN_API = "https://api.polyhaven.com"
HDRI_DIR_DEFAULT = Path("../data/hdri")


def _polyhaven_get(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "klods-syn/0.1"})
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def fetch_hdri_list(category: str = "all") -> list[str]:
    url = f"{POLYHAVEN_API}/assets?t=hdris"
    if category != "all":
        url += f"&c={category}"
    return list(_polyhaven_get(url).keys())


def download_hdri(
    name: str, out_dir: Path = HDRI_DIR_DEFAULT, resolution: str = "1k"
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}_{resolution}.exr"
    if out_path.exists():
        return out_path
    files = _polyhaven_get(f"{POLYHAVEN_API}/files/{name}")
    exr_url = files["hdri"][resolution]["exr"]["url"]
    req = urllib.request.Request(exr_url, headers={"User-Agent": "klods-syn/0.1"})
    with urllib.request.urlopen(req) as resp:
        out_path.write_bytes(resp.read())
    return out_path


@dataclass(frozen=True)
class HdriQuota:
    category: str
    count: int


DEFAULT_HDRI_QUOTAS: tuple[HdriQuota, ...] = (
    HdriQuota("indoor", 20),
    HdriQuota("studio", 15),
    HdriQuota("night", 8),
    HdriQuota("urban", 8),
    HdriQuota("outdoor", 5),
    HdriQuota("skies", 3),
    HdriQuota("sunrise-sunset", 3),
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
hdri_paths_raw = ensure_hdris()
print(f"HDRIs downloaded: {len(hdri_paths_raw)}")
```

## Mitsuba setup

Variant must be set before any Mitsuba type annotations or
`mi.Bitmap` calls are evaluated.

```python
import mitsuba as mi

mi.set_variant("scalar_rgb")
```

## HDRI scoring and weighted pool

Score each HDRI by dynamic range (log ratio of bright to dark
percentiles). Build a sampling pool with inverse-DR weighting
so low dynamic range environments dominate the training
distribution while high-DR scenes still appear for robustness.

```python
# | export
@dataclass(frozen=True)
class ScoredHdri:
    path: Path
    luminance: float
    dynamic_range: float  # log ratio of 99th to 10th percentile


def score_hdri(path: Path) -> ScoredHdri:
    img = np.array(mi.Bitmap(str(path)))
    lum = img[:, :, :3] @ [0.2126, 0.7152, 0.0722]
    p10, p99 = np.percentile(lum, [10, 99])
    dr = float(np.log1p(p99) - np.log1p(p10))
    return ScoredHdri(path, float(np.mean(lum)), dr)


def build_hdri_pool(
    paths: list[Path], min_luminance: float = 0.1
) -> tuple[list[ScoredHdri], np.ndarray]:
    scored = [score_hdri(p) for p in paths]
    pool = [s for s in scored if s.luminance >= min_luminance]
    if not pool:
        return [], np.array([])
    drs = np.array([s.dynamic_range for s in pool])
    weights = 1.0 / (1.0 + drs)
    weights /= weights.sum()
    return pool, weights
```

```python
hdri_pool, hdri_weights = build_hdri_pool(hdri_paths_raw)
print(f"HDRIs in pool: {len(hdri_pool)}")
for s in sorted(hdri_pool, key=lambda s: s.dynamic_range):
    print(
        f"  {s.path.stem}: DR={s.dynamic_range:.2f} lum={s.luminance:.3f} weight={hdri_weights[hdri_pool.index(s)]:.3f}"
    )
```

## Randomization space

```python
# | export
@dataclass(frozen=True)
class RenderConfig:
    part_rotation: np.ndarray  # (3,) rotvec
    part_translation: Vec3  # XZ offset + Y=0
    rig_tilt: Vec3  # rotvec for rig
    color: LDrawColor
    roughness: float  # ABS plastic roughness
    ior: float  # index of refraction
    hdri_path: Path | None
    hdri_rotation: float  # Y-axis rotation in radians
    hdri_intensity: float
    fill_light_dir: Vec3
    fill_light_intensity: float
    fill_light_temp: float  # color temperature K
    background_mode: str  # "solid" | "hdri"
    background_color: Vec3
    noise_sigma: float
    blur_sigma: float
    brightness_delta: float
    contrast_scale: float
    jpeg_quality: int


@dataclass(frozen=True)
class RandomizationSpace:
    platform_radius: float = 30.0
    roughness_range: tuple[float, float] = (0.15, 0.45)
    ior_range: tuple[float, float] = (1.48, 1.58)
    hdri_intensity_range: tuple[float, float] = (1.0, 3.0)
    fill_intensity_range: tuple[float, float] = (0.5, 2.0)
    fill_temp_range: tuple[float, float] = (3000.0, 8000.0)
    noise_sigma_range: tuple[float, float] = (0.0, 0.02)
    blur_sigma_range: tuple[float, float] = (0.0, 1.0)
    brightness_range: tuple[float, float] = (-0.15, 0.15)
    contrast_range: tuple[float, float] = (0.85, 1.15)
    jpeg_quality_range: tuple[int, int] = (70, 100)

    def sample(
        self,
        rng: np.random.Generator,
        color_table: tuple[LDrawColor, ...],
        hdri_pool: list[ScoredHdri],
        hdri_weights: np.ndarray,
    ) -> RenderConfig:
        part_rotation = Rotation.random(random_state=rng.integers(2**31)).as_rotvec()
        angle = rng.uniform(0, 2 * np.pi)
        radius = self.platform_radius * np.sqrt(rng.uniform())
        part_translation = (
            float(radius * np.cos(angle)),
            0.0,
            float(radius * np.sin(angle)),
        )
        rig_tilt = Rotation.random(random_state=rng.integers(2**31)).as_rotvec()

        color = color_table[rng.integers(len(color_table))]

        has_hdri = len(hdri_pool) > 0
        hdri_path = (
            hdri_pool[rng.choice(len(hdri_pool), p=hdri_weights)].path
            if has_hdri
            else None
        )

        fill_dir = _random_upper_hemisphere(rng)

        bg_mode = "hdri" if has_hdri and rng.random() < 0.5 else "solid"
        bg_color = tuple(rng.uniform(0, 1, 3).tolist())

        return RenderConfig(
            part_rotation=part_rotation,
            part_translation=part_translation,
            rig_tilt=rig_tilt,
            color=color,
            roughness=float(rng.uniform(*self.roughness_range)),
            ior=float(rng.uniform(*self.ior_range)),
            hdri_path=hdri_path,
            hdri_rotation=float(rng.uniform(0, 2 * np.pi)),
            hdri_intensity=float(rng.uniform(*self.hdri_intensity_range)),
            fill_light_dir=fill_dir,
            fill_light_intensity=float(rng.uniform(*self.fill_intensity_range)),
            fill_light_temp=float(rng.uniform(*self.fill_temp_range)),
            background_mode=bg_mode,
            background_color=bg_color,
            noise_sigma=float(rng.uniform(*self.noise_sigma_range)),
            blur_sigma=float(rng.uniform(*self.blur_sigma_range)),
            brightness_delta=float(rng.uniform(*self.brightness_range)),
            contrast_scale=float(rng.uniform(*self.contrast_range)),
            jpeg_quality=int(rng.integers(*self.jpeg_quality_range)),
        )


def _random_upper_hemisphere(rng: np.random.Generator) -> Vec3:
    v = rng.standard_normal(3)
    v[1] = abs(v[1])
    v = v / np.linalg.norm(v)
    return tuple(v.tolist())
```

```python
rng = np.random.default_rng(0)
space = RandomizationSpace()
cfg = space.sample(rng, color_table, hdri_pool, hdri_weights)
print(f"Color: {cfg.color.name} ({cfg.color.material})")
print(f"Roughness: {cfg.roughness:.3f}, IOR: {cfg.ior:.3f}")
print(f"Background: {cfg.background_mode}")
print(f"HDRI: {cfg.hdri_path.name if cfg.hdri_path else 'none'}")
```

## Scene builder

Converts a `RenderConfig` + mesh into a Mitsuba scene dict.

```python
# | export
def _temp_to_rgb(temp_k: float) -> Vec3:
    """Approximate blackbody color temperature to RGB (Tanner Helland)."""
    t = temp_k / 100.0
    if t <= 66:
        r = 1.0
        g = np.clip((99.4708 * np.log(t) - 161.1196) / 255.0, 0, 1)
        b = (
            np.clip((138.5177 * np.log(t - 10) - 305.0448) / 255.0, 0, 1)
            if t > 10
            else 0.0
        )
    else:
        r = np.clip((329.6987 * ((t - 60) ** -0.1332)) / 255.0, 0, 1)
        g = np.clip((288.1222 * ((t - 60) ** -0.0755)) / 255.0, 0, 1)
        b = 1.0
    return (float(r), float(g), float(b))


def make_bsdf(cfg: RenderConfig) -> dict:
    c = cfg.color
    if c.material == "transparent":
        return {
            "type": "dielectric",
            "int_ior": cfg.ior,
            "specular_transmittance": {"type": "rgb", "value": list(c.rgb)},
        }
    if c.material == "chrome":
        return {
            "type": "conductor",
            "specular_reflectance": {"type": "rgb", "value": list(c.rgb)},
        }
    if c.material in ("pearl", "metallic"):
        return {
            "type": "roughplastic",
            "distribution": "ggx",
            "alpha": cfg.roughness + 0.1,
            "diffuse_reflectance": {"type": "rgb", "value": list(c.rgb)},
            "int_ior": cfg.ior + 0.1,
        }
    return {
        "type": "roughplastic",
        "distribution": "ggx",
        "alpha": cfg.roughness,
        "diffuse_reflectance": {"type": "rgb", "value": list(c.rgb)},
        "int_ior": cfg.ior,
    }


_AMBIENT_DIRS = []


def build_scene(
    mesh,
    cfg: RenderConfig,
    origin: Vec3,
    target: Vec3 = (0, 0, 0),
    fov: float = 35,
    resolution: int = 512,
    spp: int = 128,
    mask_mode: bool = False,
) -> dict:
    scene = {
        "type": "scene",
        "integrator": {
            "type": "path",
            "max_depth": 6 if cfg.color.is_special else 4,
            **({"hide_emitters": True} if mask_mode else {}),
        },
        "sensor": {
            "type": "perspective",
            "fov": fov,
            "to_world": mi.ScalarTransform4f.look_at(
                origin=list(origin),
                target=list(target),
                up=[0, 1, 0],
            ),
            "film": {
                "type": "hdrfilm",
                "width": resolution,
                "height": resolution,
                **({"pixel_format": "rgba"} if mask_mode else {}),
            },
            "sampler": {
                "type": "independent",
                "sample_count": 16 if mask_mode else spp,
            },
        },
        "brick": mesh,
    }

    if cfg.hdri_path and cfg.hdri_path.exists():
        scene["envmap"] = {
            "type": "envmap",
            "filename": str(cfg.hdri_path),
            "scale": cfg.hdri_intensity,
            "to_world": mi.ScalarTransform4f.rotate(
                axis=[0, 1, 0], angle=np.degrees(cfg.hdri_rotation)
            ),
        }
    else:
        scene["background"] = {
            "type": "constant",
            "radiance": {"type": "rgb", "value": list(cfg.background_color)},
        }

    # Ambient fill from 3 directions — guarantees visibility from all viewpoints
    for i, d in enumerate(_AMBIENT_DIRS):
        scene[f"ambient_{i}"] = {
            "type": "directional",
            "direction": list(d),
            "irradiance": {"type": "spectrum", "value": 1.0},
        }

    # Randomized fill light for additional variation
    fill_rgb = _temp_to_rgb(cfg.fill_light_temp)
    if cfg.fill_light_intensity > 0.1:
        scene["fill_light"] = {
            "type": "directional",
            "direction": list(cfg.fill_light_dir),
            "irradiance": {
                "type": "rgb",
                "value": [c * cfg.fill_light_intensity for c in fill_rgb],
            },
        }

    return scene
```

## Mesh transforms

Apply part rotation and translation before rendering.

```python
# | export
def transform_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    cfg: RenderConfig,
) -> tuple:
    rot = Rotation.from_rotvec(cfg.part_rotation)
    v = rot.apply(vertices.astype(np.float64)).astype(np.float32)
    v[:, 0] += cfg.part_translation[0]
    v[:, 2] += cfg.part_translation[2]

    mesh = mi.Mesh("brick", vertex_count=len(v), face_count=len(faces))
    params = mi.traverse(mesh)
    params["vertex_positions"] = v.ravel()
    params["faces"] = faces.ravel()
    params.update()

    bsdf_dict = make_bsdf(cfg)
    mesh.set_bsdf(mi.load_dict(bsdf_dict))
    return mesh, bsdf_dict
```

## Post-processing

Numpy-level image transforms applied after rendering.

```python
# | export
from scipy.ndimage import gaussian_filter
import io


# | export
def compute_exposure(masks: list[np.ndarray], percentile: float = 99) -> float:
    brick_pixels = np.concatenate([m[:, :, :3][m[:, :, 3] > 0.5] for m in masks])
    return (
        float(np.percentile(brick_pixels, percentile)) if brick_pixels.size > 0 else 1.0
    )


def postprocess(
    image: np.ndarray, cfg: RenderConfig, exposure: float = 1.0
) -> np.ndarray:
    img = np.clip(image[:, :, :3] if image.shape[2] > 3 else image, 0, None)
    img = img / (exposure + 1e-8)

    img = img * cfg.contrast_scale + cfg.brightness_delta
    img = np.clip(img, 0, 1)

    if cfg.blur_sigma > 0.05:
        for c in range(3):
            img[:, :, c] = gaussian_filter(img[:, :, c], sigma=cfg.blur_sigma)

    if cfg.noise_sigma > 0.001:
        rng = np.random.default_rng()
        img = img + rng.normal(0, cfg.noise_sigma, img.shape)
        img = np.clip(img, 0, 1)

    if cfg.jpeg_quality < 98:
        img = _jpeg_roundtrip(img, cfg.jpeg_quality)

    return img


def _jpeg_roundtrip(img: np.ndarray, quality: int) -> np.ndarray:
    from PIL import Image

    pil = Image.fromarray((img * 255).astype(np.uint8))
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return np.array(Image.open(buf)).astype(np.float32) / 255.0
```

## Demo: randomized renders of a single part

```python
import sys

sys.path.insert(0, "..")
import ldr_tools_py as ldr
import matplotlib.pyplot as plt

settings = ldr.GeometrySettings()
ldraw_scene = ldr.load_file(f"{LDRAW_DIR}/parts/3001.dat", str(LDRAW_DIR), [], settings)
key = list(ldraw_scene.geometry_cache.keys())[0]
geom = ldraw_scene.geometry_cache[key]

vertices = geom.vertices.astype(np.float32)
vertices[:, 1] *= -1
idx = geom.vertex_indices
starts = geom.face_start_indices
sizes = geom.face_sizes

triangles = []
for s, n in zip(starts, sizes):
    for i in range(1, n - 1):
        triangles.append([idx[s], idx[s + i], idx[s + i + 1]])
faces = np.array(triangles, dtype=np.uint32)[:, [0, 2, 1]]
```

```python
from klods_syn.rendering.viewpoints import TriMirrorRig

space = RandomizationSpace()
rng = np.random.default_rng(42)
rig_base = TriMirrorRig.optimize(camera_dist=250.0)

fig, axes = plt.subplots(3, 4, figsize=(24, 15))
for row in range(3):
    cfg = space.sample(rng, color_table, hdri_pool, hdri_weights)
    rig = rig_base.with_tilt(cfg.rig_tilt)
    config = rig.viewpoint_config()
    zooms = [1.0] + [config.mirror_zoom] * len(config.mirrors)
    mesh, _ = transform_mesh(vertices, faces, cfg)

    # Mask renders — cheap, for exposure computation
    masks = [
        np.array(
            mi.render(
                mi.load_dict(
                    build_scene(mesh, cfg, origin=vp, fov=35 / zoom, mask_mode=True)
                )
            )
        )
        for vp, zoom in zip(config.viewpoints, zooms)
    ]
    exposure = compute_exposure(masks)

    # Full renders
    for col, (vp, zoom) in enumerate(zip(config.viewpoints, zooms)):
        scene_dict = build_scene(mesh, cfg, origin=vp, fov=35 / zoom)
        scene = mi.load_dict(scene_dict)
        raw = np.array(mi.render(scene))
        img = postprocess(raw, cfg, exposure=exposure)
        axes[row, col].imshow(img)
        axes[row, col].set_axis_off()
        if row == 0:
            label = "direct" if col == 0 else f"mirror {col}"
            axes[row, col].set_title(label)

fig.suptitle("3001 — 2×4 Brick — Domain Randomization", fontsize=14)
plt.tight_layout()
plt.show()
```
