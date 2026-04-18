---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.1
  kernelspec:
    display_name: python3
    language: python
    name: python3
---

# klods-syn

> Recognize ~60,000 LEGO parts from photographs — trained on synthetic CAD renders, not real labeled images.

```python
# | hide
```

## The problem

The Rebrickable catalog lists **61,705 distinct parts**. Photographing each from enough angles for supervised learning is infeasible, and the long tail keeps shifting as new parts are released. A small subset dominates real-world inventories — 90% coverage requires just **851 LDraw geometries** out of 5,638 unique base parts — but the rest still matter.

The classic obstacle is distribution shift. A frozen DINOv3 ViT-S/16 backbone hits **98.9%** top-1 on a controlled studio dataset (Gdańsk), but only **30.4%** on a different photo distribution (Brickognize). Closing that gap is what this project is about.

## The approach

Three independent stages:

1. **Render** synthetic training data from LDraw `.dat` CAD files via a Mitsuba 3 path tracer. A tri-mirror rig captures four viewpoints with **109.5° pairwise angular spread** — tetrahedral symmetry, the theoretical maximum for 4 points on a sphere. Each render randomizes color, material type, roughness, lighting, and HDRI environment.
2. **Embed** each render with a frozen DINOv3 backbone, then train a small SupCon projection head (384→384→128) on the synthetic distribution.
3. **Retrieve**, not classify: at inference we k-NN against a reference embedding bank rather than predict a fixed-vocabulary label. Adding a new part is cheap — render it, embed it, append.

## Headline result (frozen backbone baseline, k=1)

| Backbone | Same-domain (Gdańsk) | Cross-domain (Brickognize) | Cross-domain (Paco Garcia) |
|---|---|---|---|
| ConvNeXt V2 Tiny | 97.2% | 15.2% | 18.7% |
| EfficientNet-B0 | 96.7% | 13.6% | 20.1% |
| DINOv2 ViT-S/14 | 97.8% | 23.9% | 41.0% |
| **DINOv3 ViT-S/16** | **98.9%** | **30.4%** | **38.1%** |

Self-supervised ViT backbones outperform ImageNet-pretrained CNNs on cross-domain retrieval. DINOv3 is the current SOTA in this benchmark.

## What we're learning

- **Real studio data hurts cross-domain.** Training a SupCon head on Gdańsk gives +0.4pp on Gdańsk itself but *drops* Brickognize by 13.4pp and Paco Garcia by 9.4pp. The head overfits to studio lighting. This is the empirical justification for synthetic + domain randomization.
- **Single viewpoint isn't enough.** A top-down view of a 1×2 plate is visually identical to a 1×3 or 1×4 plate — and these are exactly the parts the baseline gets wrong (3023 1×2 plate: 15% accuracy on Paco Garcia). The 4-view rig exists to break this ambiguity.
- **Latency is fine.** 16.9ms per image on MPS, 28.4ms on CPU. Backbone-bound; the projection head adds <0.1ms.

## Dive deeper

**Rendering pipeline**
- [Parts catalog](parts_catalog.html) — Rebrickable + LDraw coverage analysis
- [HDRI environment maps](hdris.html) — Poly Haven scoring and weighting
- [Viewpoints](viewpoints.html) — tri-mirror rig optimization
- [Rendering](rendering.html) — Mitsuba scene assembly
- [Domain randomization](domain_randomization.html) — color, material, lighting
- [Render dataset orchestration](render_dataset.html)

**Training & evaluation**
- [Synthetic embeddings](synthetic_embeddings.html) — view-aware extraction
- [Baseline evaluation](baseline_eval.html) — backbone comparison
- [Metric learning](metric_learning.html) — SupCon head
- [Multi-view fusion](multiview_fusion.html) — combining 4 views per part
- [Retrieval evaluation](retrieval_eval.html) — two-tier k-NN
- [Inference latency](inference.html)
- [Domain gap analysis](domain_gap_analysis.html) — per-part failures

## Code

[github.com/restlessronin/klods-syn](https://github.com/restlessronin/klods-syn) — Apache-2.0.
