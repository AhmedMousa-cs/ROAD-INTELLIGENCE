# modules/road_damage — Road Damage Detection (Person 1)

Bounding-box detection of four road-damage types:
`pothole`, `longitudinal_crack`, `transverse_crack`, `alligator_crack`.

```python
from modules.road_damage import RoadDamageModel

model = RoadDamageModel("models/road_damage_best.pt")
result = model.predict(frame)
```

`result` is a shared-schema `FrameResult` dict. This module fills
**`detections`** only and leaves `segmentations` (Person 2) and
`tracks`/`alerts`/`analytics` (Person 4) at their empty defaults, so
`shared.schemas.merge_frame_results()` can concatenate all four modules'
output without duplication.

The model is **stateless** — the same frame always gives the same result, and
there is no `reset()`.

> ## ⚠️ Current status: pipeline complete, model NOT trained
>
> Every `.py` file here is finished and tested (38 tests pass). What is
> missing is `models/road_damage_best.pt` and the measured metrics, because
> training needs the RDD2022 download and a GPU. `evaluation_results.json`
> is therefore all `null` — deliberately, per spec section 15. **Do not fill
> those numbers in by hand**; run `evaluate.py`, which overwrites the file.
> See "How to train" below for the three commands.

---

## Pipeline

```text
Dataset (RDD2022 primary + Roboflow pothole supplementary)
      ↓
Preprocessing (prepare_dataset.py: normalize ids, validate, dedup, split)
      ↓
Training (train.py: transfer learning from pretrained YOLO)
      ↓
Validation (evaluate.py on the val split)
      ↓
Testing (evaluate.py on the internal held-out test split)
      ↓
Model (models/road_damage_best.pt)
      ↓
Inference (RoadDamageModel.predict(frame))
      ↓
Output Schema (shared FrameResult: detections)
      ↓
Integration (merge_frame_results)
```

## Files

| File | Responsibility |
|---|---|
| `road_damage_model.py` | `RoadDamageModel` inference class |
| `class_mapping.py` | RDD/pothole labels → ontology, exclusions, id remap |
| `prepare_dataset.py` | Annotation validation, dedup, normalized train/val/internal_test split |
| `train.py` | Transfer-learning training + `training_run.json` record |
| `evaluate.py` | mAP50, mAP50-95, P, R, per-class AP, confusion matrix |
| `inference.py` | Image / folder / video runner + CLI + `--demo` |
| `visualization.py` | Optional overlays (theme injectable) |
| `data.yaml` | Dataset config template (real one is generated) |

---

## Datasets

### Primary — RDD2022

The 4-class YOLO representation. Mapping is exactly as the spec requires:

| RDD2022 code | Normalized class |
|---|---|
| `D00` | `longitudinal_crack` |
| `D10` | `transverse_crack` |
| `D20` | `alligator_crack` |
| `D40` | `pothole` |

License: RDD2022 is released for research use by its authors (Arya et al.);
check the release you download, as terms differ between the challenge
distribution and the archived dataset. **Size and per-class annotation counts
are not stated here** — they are measured by `prepare_dataset.py` and written
to `preparation_report.json`, rather than quoted from memory.

### Supplementary — Roboflow pothole dataset

Used only as extra pothole data. `potholes` → `pothole`. **No second pothole
class is created.** Its published train/val/test organisation is preserved via
`--respect-split pothole_roboflow`.

### Excluded labels (documented, never force-mapped)

`D43` and `D44` are road-*marking* wear, not structural damage, and the shared
ontology has no marking class. `D50` and `D0w0` are utility/repair markers —
and `road_patch` belongs to Person 2's segmentation ontology, not here.
`D01`/`D11` are construction-joint variants that overlap but are not identical
to `D00`/`D10`. `hcrack`/`vcrack` are ambiguous abbreviations whose meaning
depends on undocumented capture orientation. Full list with reasons:
`class_mapping.EXCLUDED_CLASSES`.

### Splits

`prepare_dataset.py` produces `train` / `val` / `internal_test`, with the test
split held out **before** training. Default ratio 0.75 / 0.15 / 0.10, seeded
and deterministic.

The split directory is named `internal_test`, and the generated `data.yaml`
carries a comment saying so, because it is **an internal held-out split we
created — not the official RDD2022 challenge test set**.

### Annotation verification

Everything the spec asks to be checked is checked, counted, and reported in
`preparation_report.json` — nothing is silently dropped:

| Problem | Handling |
|---|---|
| Incorrect class ids | Remapped via `build_id_remap`; unmappable lines dropped and counted by source class |
| Coordinates outside `[0,1]` | Clamped if off by ≤ 0.02, otherwise dropped |
| Invalid / zero-area boxes | Dropped |
| Malformed label lines | Dropped |
| Empty labels | Kept as explicit background images (and counted) |
| Corrupted / unreadable images | Dropped |
| Duplicate images | md5 grouping |
| **Near**-duplicate images | dHash + Hamming ≤ 5 (catches re-compressed, resized and augmented copies) |

Duplicate groups are assigned to a **single split**, so a test image can never
be a near-copy of a training image — which is the failure mode that silently
inflates reported mAP.

## How to train

```bash
# 1. Normalize, validate, dedup and split
python -m modules.road_damage.prepare_dataset \
    --source data/raw/rdd2022  --name rdd2022 \
    --source data/raw/pothole  --name pothole_roboflow \
    --respect-split pothole_roboflow \
    --output data/road_damage_prepared

# 2. Train (transfer learning from pretrained weights)
python -m modules.road_damage.train \
    --data data/road_damage_prepared/data.yaml \
    --model yolov8s.pt --epochs 100 --batch 16 --imgsz 640

# 3. Evaluate — this writes evaluation_results.json
python -m modules.road_damage.evaluate \
    --weights models/road_damage_best.pt \
    --data data/road_damage_prepared/data.yaml \
    --split val --split internal_test
```

`train.py` records model, epochs, batch size, image size, learning rate,
optimizer, augmentation, device and wall-clock training time to
`training_run.json` next to the run, then copies the best checkpoint to
`models/road_damage_best.pt` (the project naming convention — never
`best.pt` or `final2.pt`).

### Model architecture

Default `yolov8s.pt`, fine-tuned. `yolov8s` over `yolov8n` because road damage
is a small-object problem where the extra capacity matters; over `yolov8m/l`
because the integrated system runs five modules per frame and inference speed
is a stated priority. Swap with `--model`; nothing else needs changing.

### Augmentation choices worth knowing

Two settings in `train.py:ROAD_AUGMENTATION` are deliberate and should not be
"fixed" without discussion:

- **`flipud=0.0`** — road imagery is captured looking down/forward at the
  surface. A vertically flipped road is not a real view.
- **`degrees=3.0`** (small) — a large rotation turns a longitudinal crack into
  a transverse one. That is not augmentation, it is relabelling. The
  distinction between `D00` and `D10` is *orientation*, so rotation has to
  stay small.

`close_mosaic=10` disables mosaic for the final epochs, which stabilises box
regression — helpful given how elongated crack boxes are.

## Output

```json
{
  "class": "pothole",
  "confidence": 0.93,
  "bbox": [300.0, 430.0, 430.0, 530.0],
  "source": "road_damage",
  "class_id": 0
}
```

`bbox` is `[x1, y1, x2, y2]`, origin top-left, in **original frame pixel
coordinates**. Ultralytics reverses its own letterbox internally; the module
never resizes the frame itself and does not assume a fixed input resolution.
Boxes are clipped into the frame and degenerate boxes dropped before anything
is returned, so `shared.schemas.validate_detection()` always passes.

`class_id` is additive (the shared validator tolerates extras) and is included
so the dashboard can colour-code without a name lookup.

A full sample is in `example_output.json`, and a rendered frame in
`samples/demo_frame.jpg`. Regenerate both — no weights needed:

```bash
python -m modules.road_damage.inference --demo \
    --json modules/road_damage/example_output.json \
    --image-out modules/road_damage/samples/demo_frame.jpg
```

## Metrics

`evaluate.py` reports, for each of `val` and `internal_test` separately:
mAP50, mAP50-95, precision, recall, per-class AP50 and AP50-95 for all four
classes, and the confusion matrix. The split name and its meaning are recorded
alongside every number.

**Currently all null** — see the status warning at the top.

## Defence in depth against label leakage

The shared rule ("never return dataset-specific class names") is enforced at
two independent stages:

1. **Offline**, in `prepare_dataset.py` — dataset class ids are rewritten to
   project ids, so the trained model has ontology names baked in.
2. **At inference**, in `road_damage_model.py` — every predicted label goes
   through `normalize_class()` again, and anything unmappable is dropped.

So even weights accidentally trained on raw RDD codes cannot leak a `D00` into
the dashboard. Loading such weights also raises a `RuntimeWarning` at load
time, because the class *ordering* assumption would be wrong even though the
names get normalized.

## Installation

```bash
pip install -r requirements.txt                        # repo base
pip install -r modules/road_damage/requirements.txt    # module extras
```

`ultralytics` + `torch` are needed to load real weights, train, or evaluate.
The test suite and `--demo` run on NumPy + OpenCV alone.

## CLI

```bash
# one image
python -m modules.road_damage.inference --source road.jpg \
    --weights models/road_damage_best.pt --image-out out/annotated.jpg

# folder of images -> JSONL
python -m modules.road_damage.inference --source data/samples/ \
    --weights models/road_damage_best.pt --json out/damage.jsonl

# video
python -m modules.road_damage.inference --source road.mp4 \
    --weights models/road_damage_best.pt --video-out out/annotated.mp4

# no weights needed
python -m modules.road_damage.inference --demo
```

## Integration

```python
from modules.road_damage import RoadDamageModel
from shared.schemas import merge_frame_results

damage = RoadDamageModel("models/road_damage_best.pt")
damage_result = damage.predict(frame)

final_result = merge_frame_results(
    damage_result, segmentation_result, obstacle_result, tracking_result,
    frame_id=frame_id, timestamp=timestamp,
)
```

Notes for whoever wires up the dashboard:

1. This module also sets `analytics.road_damage_count` from its own
   detections. Person 4's module recomputes that field across *all* modules
   when it receives `other_results`, so pass the tracking result **last** in
   the merge and let it win.
2. `visualization.draw_frame_result()` filters on `source == "road_damage"`,
   so it is safe to call on a fully merged result — it will not draw Person
   3's obstacles in road-damage colours.
3. No weights are committed to git (`models/` is gitignored). Ask for
   `road_damage_best.pt` separately, or run the three training commands.

## Known limitations

- **No metrics yet.** The model is untrained; see the status warning.
- **Crack orientation is camera-relative.** `D00` vs `D10` is defined by
  orientation *in the image*. A camera mounted at a different angle than
  RDD2022's capture setup will systematically confuse the two. If the team
  films with a different rig, expect this pair to be the weakest.
- **Cracks are badly served by boxes.** A diagonal hairline crack's bounding
  box is mostly intact road. IoU-based metrics are harsh here, and heavy
  overlap between neighbouring crack boxes makes NMS thresholds touchy.
  Person 2's segmentation module is the better signal for crack *extent*;
  this module is the better signal for crack *presence and type*.
- **RDD2022 is geographically skewed** (Japan/India/Czechia/Norway/USA in the
  2022 release). Road surface, paint and lighting in Cairo differ; expect a
  domain gap, and consider fine-tuning on local imagery before trusting the
  Road Health Score in production.
- **Severity is not estimated.** The ontology has no severity levels, so a
  hairline crack and a spalled one are both `longitudinal_crack`. Person 4's
  Road Health Score weights by class and count only.
- **Small/distant damage is missed.** Damage more than ~15–20 m ahead is a few
  pixels at 640 px inference. Raising `--imgsz` to 960/1280 helps at a
  proportional speed cost.
- **AGPL-3.0.** Ultralytics weights and package are AGPL-3.0. Fine for
  coursework; commercial deployment needs a license or a different detector.
