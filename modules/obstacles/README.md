# modules/obstacles — Road Debris + Speed Bump Detection (Person 3)

Two specialized YOLO detectors behind one wrapper, per the project spec:
road debris (from TACO) and speed bumps (from Roboflow "Speed Bump
Detection v10"). Datasets are **not** merged for training — different
ontologies, different annotation sources — but both are exposed through a
single `RoadObstacleModel` interface so the integration layer only loads
one object.

```
Dataset (TACO)              Dataset (Speed Bump Detection v10)
     │                                  │
     ▼                                  ▼
prepare_taco_dataset.py          (use Roboflow's own split)
     │                                  │
     ▼                                  ▼
train_debris.py                  train_speed_bump.py
     │                                  │
     ▼                                  ▼
models/debris_best.pt            models/speed_bump_best.pt
     │                                  │
     ▼                                  ▼
DebrisModel                      SpeedBumpModel
     └──────────────┬───────────────────┘
                     ▼
            RoadObstacleModel.predict(frame)
                     ▼
     {"detections": [...], "analytics": {...}}  (shared schema)
```

## Status of this delivery

This delivery contains the **complete, production-ready module code**
(inference classes, class mapping, visualization, CLI, tests, training/
evaluation scripts) built to the shared integration contract. It does
**not** include trained weights: training TACO and the speed-bump dataset
requires downloading them locally (Roboflow/TACO are outside this
environment's network access) and a GPU training run, neither of which is
possible in the current environment. `models/debris_best.pt` and
`models/speed_bump_best.pt` are the two files you need to produce (via
`training/train_debris.py` / `training/train_speed_bump.py`) and drop in —
after that, `RoadObstacleModel` works exactly as documented below with zero
code changes. See "What's simulated vs. real" for the honest breakdown.

## Dataset — Part A: Road Debris (TACO)

- **Source**: [TACO](http://tacodataset.org/) — Trash Annotations in
  Context. License: CC BY 4.0 (images/annotations); verify the current
  license on the dataset site before redistribution, since TACO aggregates
  images from multiple contributors.
- **Size**: TACO grows over time via community contributions — do not
  quote a fixed image/annotation count without checking the
  `annotations.json` you actually download (`prepare_taco_dataset.py`
  prints the real counts it found).
- **Class mapping**: all ~60 TACO leaf categories collapse to the single
  project class `road_debris`, with an optional non-schema `subclass` field
  (`plastic_bottle`, `can`, `glass`, `paper`, `food_waste`,
  `plastic_wrapper`, `misc_debris`) for UI/analytics use. See
  `class_mapping.py` for the full table and rationale, and note it is a
  **starter mapping** — `verify_taco_categories()` must be run against your
  real downloaded `annotations.json` before training; unmapped categories
  are excluded, never force-mapped.
- **Split**: `prepare_taco_dataset.py` builds train/val/**internal_test**
  from TACO (no official test set exists, so it is never called that),
  after removing images with zero mapped annotations and de-duplicating by
  file hash so the same photo can't appear in both train and test.

## Dataset — Part B: Speed Bump Detection v10 (Roboflow)

- **Source**: Roboflow "Speed Bump Detection", version 10.
- **Published split**: 5498 train / 266 validation / 240 test images (per
  the dataset's own published numbers — used, not re-derived).
- **Class mapping**: passthrough to `speed_bump`; `train_speed_bump.py`
  refuses to train if the downloaded `data.yaml`'s `names:` list contains
  anything outside `class_mapping.SPEED_BUMP_RAW_TO_NORMALIZED`, per the
  spec's "do not assume additional classes exist."
- **Split**: used as published — not re-split.

## Task / architecture

Both detectors are YOLOv8 (`ultralytics`) object detectors, transfer-learned
from a pretrained checkpoint (`yolov8n.pt` by default — swap for `yolov8s.pt`
etc. via `--weights` if accuracy needs outweigh the inference-speed
priority the spec calls out). Bounding-box detection, not segmentation —
debris and speed bumps are both well-suited to boxes, and per-obstacle mask
area isn't part of any downstream analytic in this module (Road Health
Score reads *segmentation* `area_ratio` from Person 2's module for surface
damage, not object-level obstacle area).

## Training

```bash
# 1. Prepare TACO -> YOLO-format road_debris dataset
python -m modules.obstacles.training.prepare_taco_dataset \
    --taco-root /path/to/TACO --out-dir data/taco_road_debris

# 2. Train road-debris detector
python -m modules.obstacles.training.train_debris \
    --data data/taco_road_debris/data.yaml --epochs 100 --batch 16

# 3. Train speed-bump detector (using Roboflow's own data.yaml directly)
python -m modules.obstacles.training.train_speed_bump \
    --data /path/to/speed-bump-detection-v10/data.yaml --epochs 100 --batch 16

# 4. Evaluate both against their held-out/test split
python -m modules.obstacles.training.evaluate \
    --weights runs/detect/road_debris/weights/best.pt \
    --data data/taco_road_debris/data.yaml --split test \
    --out modules/obstacles/evaluation_results_debris.json

python -m modules.obstacles.training.evaluate \
    --weights runs/detect/speed_bump/weights/best.pt \
    --data /path/to/speed-bump-detection-v10/data.yaml --split test \
    --out modules/obstacles/evaluation_results_speed_bump.json

# 5. Copy weights into place, per "11. MODEL NAMING"
cp runs/detect/road_debris/weights/best.pt models/debris_best.pt
cp runs/detect/speed_bump/weights/best.pt models/speed_bump_best.pt
```

Each training run writes its own `run_config.json` (model, epochs, batch
size, image size, optimizer, device, training time) next to the weights —
copy those numbers into this README's "Training runs" section below once
you've actually trained. **Do not hand-fill placeholder numbers**; leave
this section stating "not yet trained" until real numbers exist.

### Training runs

> Not yet trained in this delivery — no dataset access / GPU in this
> environment. Fill in after running the commands above:

| Detector | Model | Epochs | Batch | Img size | Optimizer | Device | Training time |
|---|---|---|---|---|---|---|---|
| road_debris | — | — | — | — | — | — | — |
| speed_bump | — | — | — | — | — | — | — |

## Evaluation

Metrics are produced by `training/evaluate.py` directly from Ultralytics'
own `model.val()` — never hand-typed. Run it against:
- road_debris: the `internal_held_out` test split `prepare_taco_dataset.py`
  creates (explicitly **not** an official TACO test set, since none exists)
- speed_bump: the dataset's own published `test` split (240 images)

### Results

> Not yet evaluated in this delivery. Fill in with the real
> `mAP50` / `mAP50-95` / `precision` / `recall` / per-class AP50 from
> `evaluation_results_debris.json` and `evaluation_results_speed_bump.json`
> once training has actually run.

## Inference interface

```python
from modules.obstacles import RoadObstacleModel

model = RoadObstacleModel(
    "models/debris_best.pt",
    "models/speed_bump_best.pt",
)

result = model.predict(frame)  # frame: numpy.ndarray, BGR, HxWx3, any resolution
```

`result` is shaped like `shared.schemas.empty_frame_result()`:

```python
{
    "frame_id": 0, "timestamp": 0.0,
    "detections": [
        {"class": "road_debris", "confidence": 0.87, "bbox": [x1, y1, x2, y2],
         "source": "debris", "subclass": "plastic_bottle"},
        {"class": "speed_bump", "confidence": 0.94, "bbox": [x1, y1, x2, y2],
         "source": "speed_bump"},
    ],
    "segmentations": [], "tracks": [], "alerts": [],
    "analytics": {"vehicle_count": 0, "traffic_density": "low",
                  "road_damage_count": 0, "obstacle_count": 1,
                  "speed_bump_count": 1},
}
```

`subclass` on `road_debris` detections is metadata for the dashboard, not
part of the shared schema — don't rely on it being present if the mapping
table is later trimmed.

`DebrisModel` and `SpeedBumpModel` are individually importable from
`modules.obstacles` if only one detector is needed. `predict()` on either
raises `FileNotFoundError` if `model_path` doesn't exist and
`TypeError`/`ValueError` on a malformed `frame` — no silent failures.

## Configuration

`confidence_threshold`, `iou_threshold`, and `image_size` all come from
`shared/config.py`'s `InferenceConfig` by default — pass a custom
`InferenceConfig` (or plain dict with the same keys) to
`RoadObstacleModel(..., config=...)` to override for both detectors at
once, or construct `DebrisModel`/`SpeedBumpModel` directly for per-detector
overrides. Nothing in this module hard-codes a threshold.

## Visualization

```python
from modules.obstacles.visualization import draw_detections
annotated = draw_detections(frame, result["detections"])
```

Default colors are in `visualization.DEFAULT_COLOR_MAP`; pass
`color_map={"road_debris": (r, g, b), ...}` to override for the dashboard's
own theme.

## CLI

```bash
python -m modules.obstacles.inference \
    --debris-model models/debris_best.pt \
    --speed-bump-model models/speed_bump_best.pt \
    --image path/to/road.jpg --out annotated.jpg
```

Also supports `--video` (with optional `--sample-every-n-frames` to skip
frames for speed) instead of `--image`. See `inference.py` docstring.

## Tests

```bash
pytest tests/test_obstacles.py -v
```

Two tiers, matching "12. REQUIRED TEST":
- **Unit tests** (always run): validate `class_mapping.py`'s mapping/
  exclusion logic and `RoadObstacleModel`'s merge/schema behavior against a
  stub backend, so the interface contract is verified independent of
  whether trained weights exist yet.
- **Integration tests** (auto-skipped if `models/debris_best.pt` /
  `models/speed_bump_best.pt` aren't present): load the real weights, run
  inference on a real image and a synthetic frame, and validate the output
  with `shared.schemas.validate_frame_result()`.

## Known limitations

- No trained weights ship with this delivery (see "Status of this
  delivery" above) — the integration tests are skipped until you train and
  drop in `models/debris_best.pt` / `models/speed_bump_best.pt`.
- `class_mapping.TACO_TO_ROAD_DEBRIS` is a best-effort starting table, not
  verified against a specific downloaded TACO release. Run
  `class_mapping.verify_taco_categories()` before trusting it for training.
- TACO is a general litter dataset, not road-obstacle-specific — some
  mapped categories (e.g. small wrappers) may be too small/low-severity to
  matter for driving-hazard use cases even though they're visually litter;
  consider a minimum-bbox-area filter at the dashboard/analytics layer if
  that turns out to be noisy in practice.
- Road-debris detection quality on frames very different from TACO's
  photography style (e.g. dashcam footage vs. TACO's largely
  ground-level/close-up litter photos) is untested and likely to need
  additional fine-tuning data from actual road/dashcam scenes.
- Speed-bump detection assumes bumps are visually distinct from shadows/
  road markings under the training set's lighting conditions; no
  night-time-specific data has been verified.
