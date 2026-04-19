# Training Strategy

**Status:** Active — guides rendering pipeline and model training decisions

---

## Overview

Frozen DINOv3 ViT-S/16 backbone → supervised contrastive projection head
→ multi-view fusion layer → k-NN retrieval against a 4-view reference
embedding database.

Training data is synthetic renders from the LDraw parts library with
aggressive domain randomization. No real-world photographic datasets are
used for training. Real scanner captures are reserved for fine-tuning
once hardware is operational.

---

## Decisions and Evidence

### 1. No Gdańsk for training

**Decision:** Exclude the Gdańsk University dataset (54K training images)
from model training entirely. Use only as a sanity-check diagnostic.

**Evidence:**

- Frozen DINOv3 ViT-S/16 achieves 98.9% top-1 k-NN accuracy on the
  Gdańsk test split — near-ceiling with no training at all.
- A SupCon projection head trained on Gdańsk embeddings improved
  same-domain accuracy slightly (98.9% → 99.2%) but degraded
  cross-domain performance:
  - Brickognize: 30.4% → 17.0% (−13.4pp)
  - Paco Garcia: 38.1% → 28.7% (−9.4pp)

**Rationale:** The Gdańsk studio photos (white background, LED lamps,
controlled angles) occupy a distribution very close to clean synthetic
renders. Training on Gdańsk teaches the projection head "what studio
photos look like," not "what parts look like." The cross-domain
degradation proves the head overfits to domain-specific features — it
pulls same-part embeddings together within the studio distribution while
pushing cross-domain embeddings apart.

Gdańsk remains useful as a sanity check: if accuracy drops below ~95%
on Gdańsk test, something is fundamentally broken in the feature
pipeline. But optimizing for it is counterproductive.

### 2. Frozen backbone — do not pretrain or fine-tune DINOv3

**Decision:** Use DINOv3 ViT-S/16 as a frozen feature extractor. Train
only the projection head and fusion layer on top.

**Evidence:**

- 98.9% same-domain accuracy from frozen 384-dimensional embeddings
  with no task-specific training whatsoever.
- The 30.4% / 38.1% cross-domain accuracy is a training data problem,
  not a backbone capacity problem — the backbone separates parts well
  within any given domain.

**Rationale:** The backbone is not the bottleneck. The DINOv3 features
already encode rich part geometry and surface properties. The failure
mode is domain shift, not feature quality. Training data diversity is
what moves the cross-domain numbers — not a larger or fine-tuned
backbone. Keeping the backbone frozen also enables caching embeddings
and training the projection head on cached vectors, which makes
iteration fast.

### 3. Synthetic renders as primary training data

**Decision:** Generate training data from LDraw 3D models rendered with
Mitsuba 3 and aggressive domain randomization. No real-world
photographic datasets for training.

**Evidence from prior work (documented in docs/data-augmentation.md):**

- Brickognize (Vidal et al., MDPI Sensors 2023) achieved 91.3% AP50 on
  uncontrolled real-world scenarios using synthetic-only training with
  domain randomization, plus 20 real images for few-shot fine-tuning.
- Daniel West (Universal LEGO Sorting Machine) reached 93% accuracy
  with synthetic renders + domain randomization after initially failing
  with synthetic data alone.
- Ablation studies across multiple projects identify lighting direction,
  camera pose, and background variation as the highest-impact
  randomization parameters.
- Every published project that attempted zero real-world data failed.

**Evidence from our eval data:**

- The 30.4% Brickognize and 38.1% Paco Garcia accuracy on frozen
  features establishes the baseline domain gap. Synthetic training data
  with sufficient visual diversity is the mechanism to close this gap.

**Rationale:** Synthetic rendering gives us unlimited labeled data with
exact part IDs, controllable domain randomization, and coverage of all
70,000+ LDraw parts regardless of whether physical samples exist. The
rendering pipeline is the scalable path. Real scanner captures for
fine-tuning remain non-negotiable — but they complement synthetic data,
they don't replace it.

### 4. Color randomization over monochrome

**Decision:** Randomize the LDraw color assigned to each part at render
time, sampling from the full LEGO color palette. Do not convert to
grayscale.

**Evidence:**

- Brickognize per-class failures include Technic pins, axles, and
  connectors (32034, 87083, 32054, 55013) that differ from visually
  similar parts primarily in surface geometry and material properties.
  Monochrome rendering would destroy material cues (transparency,
  pearl, chrome) needed to distinguish these parts.
- Standard SupCon treats all images of the same part ID as positives.
  If training data includes part 3001 in red, blue, yellow, green,
  black, and white, the contrastive loss forces the model to find
  features that are invariant to color — because color varies within
  the positive set.

**Rationale:** Color randomization achieves the same goal as monochrome
(preventing color-based shortcuts) without destroying material
information. Transparent, chrome, pearl, and metallic surfaces interact
with light in ways that are shape-relevant and visible only in color.
Grayscale can be added as an additional augmentation with some
probability for further robustness.

### 5. Environment maps over light box simulation

**Decision:** Use HDRI environment maps for scene illumination during
rendering. Do not simulate the physical scanner's light box geometry
as the primary lighting model.

**Evidence:**

- Paco Garcia (conveyor camera, varied industrial lighting) achieves
  only 38.1% accuracy — the domain gap from Gdańsk's controlled studio
  lighting is severe.
- Brickognize's uncontrolled subset (varied real-world surfaces and
  lighting) shows the same pattern.
- Ablation studies in the domain randomization literature (cited in
  docs/data-augmentation.md) identify fixed lighting as the single
  largest source of accuracy degradation when removed from the
  randomization schedule.

**Rationale:** Environment maps provide massive lighting diversity
essentially for free — thousands of HDRIs are available from sources
like Poly Haven, covering indoor, outdoor, studio, and industrial
lighting conditions. This forces the model to learn part features that
are invariant to illumination. Light box simulation can be added later
as a domain-specific refinement for scanner fine-tuning, but the
primary training data should maximize lighting diversity.

### 6. Two-stage architecture: single-view → multi-view fusion

**Decision:** Train a single-view encoder first, then a multi-view
fusion layer that combines 4 rig views into one part embedding.

**Evidence:**

- Paco Garcia's worst failures are geometrically similar basic bricks
  viewed from a single conveyor angle:
  - 3002 (2×3 brick): 2.0% accuracy
  - 3010 (1×4 brick): 14.3%
  - 3023 (1×2 plate): 15.5%
    These parts are genuinely ambiguous from one viewpoint — a top-down
    view of a 2×4 plate is visually identical to a 2×4 tile.
- The tri-mirror rig provides 4 views spanning ~60–90° of angular
  spread (from viewpoint optimization in nbs/100_viewpoints), giving
  enough geometric coverage to resolve single-view ambiguities.

**Rationale:** The single-view encoder produces view-specific embeddings
that encode "what this part looks like from this angle." The multi-view
fusion layer produces a viewpoint-invariant part embedding from 4 such
view-specific embeddings. This separation is important: forcing
single-view embeddings to be viewpoint-invariant would collapse parts
that are distinguishable only from specific angles (plates vs tiles,
bricks of different heights). The fusion layer is what resolves these
ambiguities using the combined evidence from all 4 views.

### 7. View-aware contrastive pairs for single-view training

**Decision:** The single-view SupCon loss should not treat all views of
the same part as equivalent positives. Same part + similar viewpoint =
strong positive. Same part + very different viewpoint = weak positive
or excluded.

**Evidence:**

- A top-down view of a 2×4 plate and a top-down view of a 2×4 tile
  are near-identical images of different parts. Standard SupCon would
  pull the top-down plate embedding toward a side-view plate embedding
  (same part, strong positive) — but this moves it away from the
  top-down tile embedding, which is what distinguishes them.
- The single-view embedding space must preserve viewpoint information
  so the fusion layer has meaningful per-view signals to combine.

**Rationale:** If single-view embeddings are fully viewpoint-invariant,
the fusion layer adds no information — averaging 4 copies of the same
embedding doesn't help. The fusion layer is useful precisely because
each view contributes different geometric evidence. The single-view
contrastive loss should encode "what this part looks like from here,"
not "what this part is."

### 8. Evaluation protocol

**Decision:** Three-tier eval, with single-view cross-domain as the
development diagnostic and 4-view retrieval as the target metric.

**Evidence:**

- Gdańsk test (98.9%) is near-ceiling and provides no signal for
  improvement.
- Brickognize (30.4%) and Paco Garcia (38.1%) represent the real-world
  distributions where improvement is needed.
- 4-view retrieval accuracy can be evaluated on synthetic rig renders
  before scanner hardware exists.

**Eval tiers:**

| Tier       | Metric                      | Data source                             | Purpose                                   |
| ---------- | --------------------------- | --------------------------------------- | ----------------------------------------- |
| Diagnostic | Single-view k-NN accuracy   | Brickognize + Paco Garcia               | Fast feedback during pipeline development |
| Target     | 4-view fused k-NN retrieval | Synthetic rig renders (tilt-randomized) | Primary optimization target               |
| Production | 4-view fused k-NN retrieval | Real scanner captures                   | Final deployment metric                   |

**Rationale:** Single-view eval is a proxy. If single-view features
improve, multi-view fusion will improve too — but the converse is not
guaranteed. Mediocre single-view accuracy with excellent 4-view accuracy
is a valid and expected outcome, because the fusion resolves ambiguities
that are genuinely unresolvable from one angle. The production metric is
always 4-view retrieval on real scanner data.

### 9. Render quality — uniform 2048 SPP

**Decision:** Render all training images at `spp=2048`, regardless of
`color.material`. Do not branch SPP by material class.

**Evidence:**

- `scripts/spp_sweep.py` shows caustics and specular highlights on
  transparent and chrome colors plateau by ~256 SPP — additional
  samples do not reduce visible noise for those materials.
- Solid and metallic surfaces continue to show speckle on brick faces
  below 2048 SPP.

**Rationale:** In principle transparent parts could render at 256 SPP
to save budget. In practice, non-solid colors are a small share of the
dataset, and a single uniform `spp` keeps `render_pair` simple and the
per-image cost predictable. If the non-solid share grows or the render
budget tightens, dispatching transparent at 256 is the obvious lever.

---

## Reference Database

The k-NN reference database contains fused 4-view embeddings:

- One entry per part ID × color (or per part ID if color-randomized
  training makes color irrelevant to the embedding).
- Initially populated from synthetic rig renders covering the full
  LDraw parts library.
- Augmented with real scanner captures as they become available.
- Query at inference time: 4 scanner images → single-view encoder →
  4 embeddings → fusion layer → 1 fused query → k-NN against database.

---

## Training Pipeline Sequence

1. **Build rendering pipeline** (nbs/102_domain_randomization) —
   environment maps, lighting randomization, color randomization,
   material perturbation, background variation, post-processing
   noise/blur. Output: single-view renders for stage 1.

2. **Train single-view SupCon head** on synthetic renders with
   view-aware contrastive pairs. Eval against Brickognize and Paco
   Garcia as diagnostic.

3. **Add 4-view rig rendering** — coherent 4-view captures with tilt
   randomization, same domain randomization as single-view.

4. **Train multi-view fusion layer** on synthetic 4-view rig renders.
   Eval on synthetic 4-view retrieval.

5. **Scanner fine-tuning** — capture a few hundred parts through the
   real scanner, fine-tune the fusion layer (and optionally the
   projection head) on real 4-view data.

---

## Open Questions

- **View-aware contrastive pair design:** Exact definition of "similar
  viewpoint" for positive pair selection. Angular threshold? Continuous
  weighting by viewpoint similarity?
- **Fusion architecture:** Mean pooling vs attention vs learned
  aggregation for combining 4 view embeddings. Needs ablation once
  4-view data exists.
- **Reference database granularity:** One fused embedding per part ID,
  or multiple entries per part covering different rig tilts for
  robustness?
- **Scanner fine-tuning strategy:** How many real captures are needed?
  Fine-tune projection head only, or also fusion layer? Brickognize
  needed 20, Daniel West used 200,000 — our number is somewhere in
  between.
