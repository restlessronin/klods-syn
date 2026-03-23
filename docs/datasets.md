# Datasets

**Status:** Inventory complete — download and validation pending

---

## Category A — Real Photos Labeled by Part ID

Datasets with per-image LEGO part identification, usable for supervised
classification training and evaluation.

### Primary

**Gdańsk University of Technology (Boiński et al., Nature Scientific
Data 2023)**

The dominant dataset for LEGO part classification. Part of a series of
5 datasets hosted on the MostWiedzy institutional repository.

- ~155K total images (real + synthetic combined; exact real-only count
  needs verification — estimates range from 52K to 77K real photos)
- 447 distinct part classes, labeled by official LEGO brick type number
- Labels via directory structure and PASCAL VOC XML
- Single isolated bricks, white non-reflecting background, LED lamps,
  handheld camera at random angles
- Some subsets include household backgrounds and conveyor simulation
- CC BY 4.0
- Paper: https://www.nature.com/articles/s41597-023-02682-2
- Data: MostWiedzy repository (multiple DOIs, ~6GB+ total)
- Note: The Kaggle dataset at marwin1665/synthetic-lego-images-images22
  is synthetic renders only, not a mirror of the real photos.

TODO: Download, verify real vs synthetic split, confirm class count
matches our target part list.

### Secondary

**Brickognize Test Set (Vidal et al., MDPI Sensors 2023)**

- Real-world photos with COCO-format segmentation labels
- 76 distinct brick types
- Two subsets: controlled (uniform background) and uncontrolled
  (varied real-world surfaces)
- Publicly released test split only (training was synthetic +
  20 real images for few-shot fine-tuning)
- License unspecified in paper
- Data: https://www.tramacsoft.com/brickognize
- Paper: https://www.mdpi.com/1424-8220/23/4/1898

Small but unique: only public dataset with both detection bounding
boxes and instance segmentation masks on real-world uncontrolled
photos. Useful as an evaluation set.

TODO: Download, verify accessibility (link from 2022), confirm
annotation format and brick type overlap with our target list.

**VSLID — Very Small Lego Image Dataset (Delft University, Zenodo 2021)**

- ~1,800 real photos of piles with 1–10 bricks per image
- 85 distinct part types, multi-label annotations per image
- Semi-controlled household-style setups
- CC BY 4.0
- https://zenodo.org/records/3901857

Unique value: only dataset with multi-label pile annotations. Useful
for both classification (cropped regions) and future detection work.

**Paco Garcia — Lego Brick Sorting (Kaggle)**

- ~4,580 real photos from Raspberry Pi sorting-machine cameras
- 20 distinct part classes
- Controlled conveyor setup, multiple angles
- CC BY-SA 3.0
- https://www.kaggle.com/datasets/pacogarciam3/lego-brick-sorting-image-recognition

**Legoland Lego Brick Classifier (Roboflow Universe)**

- ~1,194 real photos
- Labeled by specific LEGO part numbers (e.g., 3003)
- Controlled single-brick photos
- CC BY 4.0
- https://universe.roboflow.com/legoland/lego-brick-classifier

**Lego Brick Recognition (Roboflow Universe)**

- ~13,000 real photos
- Object detection bounding box labels
- CC BY 4.0
- https://universe.roboflow.com (search Lego Brick Recognition)

TODO: Verify class count, label granularity (part ID vs generic
"brick"), and image quality.

**Lego Identifier (Roboflow Universe)**

- ~9,900 images
- Labeled by part ID (details unverified)
- https://universe.roboflow.com (search Lego Identifier)

TODO: Verify whether this is a later version of the "Legoland Lego
Brick Classifier" entry above or a distinct project. Confirm class
count and license.

**NTOU Lego Front/Side/Back (Roboflow Universe)**

- 288 real photos
- 2 classes (4x2 and 6x2 bricks), labeled by orientation
  (front, side, back)
- Niche: orientation/pose estimation, not part classification
- https://universe.roboflow.com (search NTOU Lego)

TODO: Confirm license and annotation format.

---

## Category B — Real Photos for Self-Supervised Pretraining

Datasets with loose, coarse, or no part-level labels. Useful for
contrastive learning, masked autoencoders, and other self-supervised
methods that learn general LEGO visual features without requiring
part IDs.

### Larger (>5K images)

**bulu/lego_blip_512 (HuggingFace)**

- 100K–1M images (size category on HuggingFace; exact count unverified)
- BLIP-generated text captions, no part IDs
- In-the-wild photos of built sets, models, scenes
- License unspecified
- https://huggingface.co/datasets/bulu/lego_blip_512

TODO: Verify actual image count, license, and image quality.

**Open Images V6 — "Lego" class**

- ~4,600 images with bounding boxes; ~15K with image-level "Lego" tag
- Single generic class, no part differentiation
- In-the-wild consumer photos (built sets, piles, children playing,
  mixed with other toys)
- CC BY 4.0
- https://storage.googleapis.com/openimages/web/factsfigures.html
- Download via class filter `/m/0d_2m`

Maximum visual diversity. Excellent for learning robustness to
real-world conditions.

**Lego vs. Generic Brick (Paco Garcia, Kaggle)**

- ~20K real photos (some descriptions say ~46K)
- 12 classes: 6 brick types × LEGO vs generic
- Multi-camera Raspberry Pi sorter setup
- CC-style open license
- https://www.kaggle.com/datasets/pacogarciam3/lego-vs-generic-brick-image-recognition

**Hex Lego (Roboflow Universe)**

- ~8.3K real photos
- 28–32 loose classes (size + color)
- Piles, varied lighting
- CC BY 4.0

**magichampz/lego-technic-pieces (HuggingFace)**

- ~6K real photos of Technic pieces
- Loose category labels
- Adds geometric variety (Technic parts are structurally distinct
  from System bricks)
- https://huggingface.co/datasets/magichampz/lego-technic-pieces

**Multiple Lego Tracking Dataset (Eötvös Loránd University, Kaggle)**

- Thousands of frames extracted from 12 high-resolution videos
- Multiple bricks per frame, including overlapping/occluded scenarios
- White conveyor belt, smartphone camera, 13–25 FPS
- Loosely labeled by complexity (Simple, Overlapping)
- License unspecified
- https://www.kaggle.com/datasets/hbahruz/multiple-lego-tracking-dataset

Unique value: temporal sequences and occlusion scenarios. Useful for
learning segmentation under clutter — relevant to future detection work.

TODO: Download, verify frame count and labeling granularity.

**DTTD-Mobile / MobileBrick (UC Berkeley, 2023–2025)**

- ~47,700 RGB-D frames from iPhone 14 Pro LiDAR
- 153 LEGO sets, 18 specific tracking objects
- 3D bounding boxes and semantic segmentation labels
- CC BY-SA 4.0
- https://github.com/augcog/DTTD2

Not directly applicable to our single-part classification pipeline,
but relevant for future detection work and understanding how LEGO
surfaces interact with consumer depth sensors.

TODO: Verify download size and whether individual part labels are
extractable from the set-level annotations.

### Smaller (<5K images)

**Gdańsk University — Tagged images / conveyor subsets**

- Part of the Gdańsk series (same MostWiedzy repository)
- ~2,933 multi-brick photos with bounding boxes (no part IDs)
- Conveyor belt simulation, white background
- Videos also available (extractable to frames)

**Gdańsk University — Conveyor Belt Videos (2022)**

- Video recordings of LEGO bricks moving on a white conveyor belt
- Part of the same MostWiedzy repository series
- Intended for training sorting machine classifiers
- Extractable to frames for pretraining data
- https://mostwiedzy.pl (search for LEGO conveyor video dataset)

TODO: Download, verify frame count and labeling (if any).

**Spiled_LEGO_Bricks (Kaggle)**

- ~1,800 real photos
- Single-class YOLO bounding boxes (detection only)
- Scattered bricks on various surfaces, in-the-wild-ish
- Apache 2.0
- https://www.kaggle.com/datasets/migueldilalla/spiled-lego-bricks

**Simplified Object Detection for Manufacturing (Zenodo)**

- 1,500 real photos at 640×640
- YOLO detection + quality/defect labels
- Low-cost manufacturing camera setup
- CC BY 4.0
- https://zenodo.org/records/10731976

**Lego_V2 (Roboflow Universe)**

- 450 images
- 13 descriptive classes (e.g., 1x1x3, 2x4x1, Circle)
- Object detection format
- https://universe.roboflow.com (search Lego_V2)

**LEGO Minifigures (Isaienkov, Kaggle)**

- ~500 real photos of minifigures in varied poses
- CC BY-NC-SA 4.0
- Minifigure-specific; not relevant unless sorted-studs handles
  minifig sorting

**Lego Figures (Roboflow Universe)**

- 776 real photos
- Labeled for complete/missing parts (head, arms, legs)
- Open source license
- Anomaly detection angle: identifies incomplete minifigures

**Hardhat Detection FOMO (Roboflow Universe)**

- 192 images
- Specific focus on red/white hardhat minifigure accessories
- Too narrow for general use

---

## Category C — Reference Data (Non-Photographic)

Structured data useful for part identification, class list construction,
and reference embedding databases, but not photographic training data.

**Rebrickable — LDraw Renders by Color**

- Pre-rendered catalog images of the full LDraw parts library
- Organized by color, including chrome, pearl, transparent,
  glow-in-dark, and other special material classes
- Single viewpoint (45° top-down), white background
- Explicitly provided for download with permissive terms
  ("You can use these files for any purpose")
- https://rebrickable.com/downloads/

Not suitable for training directly (single viewpoint, no domain
randomization), but valuable for: part ID → visual reference mapping,
color/material coverage validation, seed images for metric learning
reference database.

**Rebrickable — Parts/Sets/Colors CSV Metadata**

- Complete catalog of official LEGO parts, sets, colors, and
  inventories, updated daily
- CSV format, freely downloadable
- API available for real-time queries (rate-limited)
- https://rebrickable.com/downloads/

Useful for: building the target part list, part ID ↔ LDraw ID
mapping, color ID resolution, identifying part relationships
(prints, molds, alternates).

---

## Watch List

Datasets not yet publicly available but worth monitoring.

**BrickSortingMachine — Classification Training Data**

- Images collected through an operational sorting machine with a
  dual-view mirror-based scanner (two perspectives per part)
- Currently used to train a 21-class category classifier
- Author plans to publicly release labeled datasets at regular
  intervals after manual labeling
- Multi-view image pairs — architecturally relevant to our
  multi-mirror scanner design
- Contact via Discord for updates
- https://bricksortingmachine.com/classification-webservice

---

## Excluded Datasets

Evaluated and rejected. Preserved here to avoid re-evaluation.

| Dataset                                                 | Reason                                                                                                                      |
| ------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| B200C LEGO Classification (Kaggle)                      | Synthetic renders only                                                                                                      |
| B200 LEGO Detection (Kaggle)                            | Synthetic renders only                                                                                                      |
| Synthetic LEGO Images / Images22 (Kaggle)               | Synthetic renders only                                                                                                      |
| pvrancx/legobricks (HuggingFace)                        | Synthetic LDraw renders only                                                                                                |
| Lego EMMET B200 (Roboflow)                              | Synthetic from Kaggle                                                                                                       |
| korra-pickell/LEGO-Classification-Dataset (GitHub)      | Machine-generated synthetic                                                                                                 |
| Biggest LEGO Dataset (Reddit/Kaggle)                    | Blender synthetic renders                                                                                                   |
| LEGO-Parts-3D (GitHub)                                  | 3D model files (.ldraw/.obj), not photos                                                                                    |
| Daniel West sorting machine (~200K images)              | Not publicly released                                                                                                       |
| Theiner et al. real test set (~300 images)              | Not separately downloadable                                                                                                 |
| Kaggle: LEGO Set Recognition                            | Box/packaging images, not parts                                                                                             |
| HuggingFace: lego-set-classification                    | Text metadata only, no images                                                                                               |
| Lego Product Dataset (Zenodo)                           | Product metadata, no images                                                                                                 |
| Rebrickable / BrickLink — Element & Photo images        | ToS prohibits ML use; not available for bulk download. (LDraw renders and CSV metadata listed separately in Category C)     |
| Flickr Creative Commons "lego"                          | Not a versioned dataset; requires manual curation                                                                           |
| Google Image Search "lego"                              | Copyright issues, unfiltered noise                                                                                          |
| Image2Lego (Papers With Code)                           | Research paper, no released photo dataset                                                                                   |
| RebrickNet training frames (Rebrickable)                | Not available for bulk download; privacy and bandwidth restrictions                                                         |
| York University LEGO Dataset (Brouhani)                 | Synthetic only; designed for dataset bias research                                                                          |
| LEGO-Dataset (bolinlai / Ego4D)                         | Egocentric action generation dataset, not part identification                                                               |
| Joost Hazelzet LEGO Bricks                              | Rendered in Autodesk Maya and Blender; synthetic only                                                                       |
| Mantas Gribulis Dataset                                 | 100% synthetic; used to demonstrate sim-to-real transfer                                                                    |
| Lego-Data-Analysis (kduvvuri1)                          | CSV metadata/history only; no image data                                                                                    |
| ShanghaiTech Anomaly                                    | Generic surveillance dataset; unrelated to LEGO                                                                             |
| Sphero Tracking Dataset                                 | Tracks Sphero robots, not LEGO parts                                                                                        |
| LegoLAS sorting machine (THI Ingolstadt, LegoAS/LegoAS) | Dataset not publicly released; wiki documents ~18K images across 243 classes from bachelor theses but no download available |

---

## Usage Plan

1. **Fixed test set:** Gdańsk real photos (held-out split, never used
   for training). This is the constant for all before/after comparisons.
2. **Supervised baseline:** Train on Gdańsk real photos only.
3. **Self-supervised pretraining:** Combine Category B datasets for
   maximum volume and visual diversity.
4. **Synthetic augmentation:** After rendering pipeline is built, train
   with synthetic + real, evaluate against the same fixed test set.
5. **Scanner fine-tuning:** Once hardware is operational, capture a few
   hundred parts through the actual scanner for domain-specific
   fine-tuning.
