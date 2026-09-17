# modules/road_segmentation — Person 2

Pixel/mask-level road surface & defect segmentation.

Outputs: `pothole`, `surface_damage`, `road_patch`, `ravelling`
(`alligator_crack` optionally, only if Dataset 2's `alligator` class maps
cleanly — see `class_mapping.py`).

```
Dataset          -> pothole_seg (720 train / 60 val, no official test) +
                    roboflow_road_defect (verify real split from export)
        ↓
Preprocessing    -> validate polygons, dedupe, class_mapping.py normalization,
                    verify_mapping_against_dataset() run against real data.yaml
        ↓
Training         -> YOLOv8-seg, transfer learning from COCO-pretrained weights
        ↓
Validation       -> published val split (pothole_seg) / dataset's val split
                    (road_defect); track mask mAP50, mAP50-95, IoU per class
        ↓
Testing          -> internal held-out test split where possible; documented
                    explicitly as "internal held-out", never "official"
        ↓
Model            -> models/road_segmentation_best.pt
        ↓
Inference        -> modules/road_segmentation/road_segmentation_model.py
        ↓
Output Schema    -> {"segmentations": [{class, confidence, mask, area_px,
                    area_ratio, source}]}  (see shared/schemas.py)
        ↓
Integration      -> merged into the final FrameResult's "segmentations" list
                    via shared.schemas.merge_frame_results()
```

## Dataset

**Dataset 1 — Pothole Image Segmentation Dataset**
- Format: YOLOv8-seg
- Single class: `pothole`
- Published split: 720 train / 60 validation. **No official test split.**
  If you create one, carve it from original (non-augmented) images before
  augmentation and call it an "internal held-out test split" — never claim
  it's official.
- License: record the exact license/attribution from the dataset's source
  page here once downloaded (not filled in by this scaffold — do not
  invent a license).

**Dataset 2 — Roboflow "Road Defect Segmentation"**
- Classes and split: **unknown until the actual export is inspected.**
  `class_mapping.py` ships a *provisional* mapping taken directly from the
  spec, marked clearly as unverified. Run:
  ```bash
  python -m modules.road_segmentation.class_mapping \
      --verify data/road_defect_seg/data.yaml --dataset roboflow_road_defect
  ```
  before trusting it, and update `ROBOFLOW_ROAD_DEFECT_MAPPING` /
  `EXCLUDED_CLASSES` in `class_mapping.py` based on what it reports.
- Record dataset size, license, and split sizes here after inspection.

## Class mapping

See `class_mapping.py`. Rules enforced there:
- A dataset class not present in the mapping is **never** force-mapped —
  `normalize_class()` returns `None` and the instance is dropped from the
  output, not silently renamed.
- Explicitly excluded classes go in `EXCLUDED_CLASSES` with a reason in a
  comment, not just omitted.

## Training

Record (fill in after actual training runs — do not invent numbers):

| Parameter | Value |
|---|---|
| Base model | `yolov8n-seg.pt` (or whichever variant chosen) |
| Epochs | — |
| Batch size | — |
| Image size | `shared.config.IMAGE_SIZE` (640 by default) |
| Optimizer | — |
| Learning rate | — |
| Augmentation | — |
| Device | — |
| Training time | — |

Command sketch:
```bash
yolo segment train \
  model=yolov8n-seg.pt \
  data=modules/road_segmentation/data_pothole_seg.yaml \
  epochs=100 imgsz=640 device=0 \
  project=runs/road_segmentation name=pothole_seg
```
Repeat / fine-tune for the second dataset per the two-dataset strategy
below, then export the final checkpoint as `models/road_segmentation_best.pt`.

### Two-dataset strategy

The spec does not mandate merging Dataset 1 and Dataset 2 into one
training run, and their class sets only partially overlap (`pothole` is
shared; the rest is Dataset 2-only). Two reasonable approaches:
1. **Sequential fine-tune**: train on the larger/cleaner dataset first,
   fine-tune on the second, keeping a combined validation check on both.
2. **Merged dataset**: only if both datasets' images and masks can be
   combined into one YOLO-seg folder structure with a single `names` list
   built from the normalized classes (i.e. merge AFTER `class_mapping.py`
   normalization, not before).
Document whichever approach is actually used here once decided.

## Evaluation

Report mask `mAP50`, `mAP50-95`, precision, recall, IoU, per class. See
`evaluation_results.json` for the exact template to fill in — it currently
contains only `null` placeholders and a `_note` explaining that no
training has been run yet in this scaffold. **State which split every
number comes from.**

## Model

`models/road_segmentation_best.pt` — not included in this scaffold (see
`.gitignore`: `*.pt` files aren't committed). Train per above and place
the exported weights there.

## Inference

```python
from modules.road_segmentation import RoadSegmentationModel

model = RoadSegmentationModel(
    "models/road_segmentation_best.pt",
    dataset="roboflow_road_defect",  # or "pothole_seg", matching how the
                                      # weights were trained/labeled
)
result = model.predict(frame)  # frame: BGR numpy.ndarray, any resolution
# result == {"segmentations": [{"class": "pothole", "confidence": 0.91,
#             "mask": {"format": "polygon", "points": [[x,y], ...]},
#             "area_px": 12345, "area_ratio": 0.034,
#             "source": "road_segmentation"}, ...]}
```

CLI sanity check on one image/video:
```bash
python -m modules.road_segmentation.inference \
  --model models/road_segmentation_best.pt \
  --image data/samples/road1.jpg \
  --out outputs/road1_segmented.jpg
```

## Output schema

Exactly `shared.schemas.Segmentation`: `class`, `confidence`, `mask`,
`area_px`, `area_ratio` (`= area_px / image_area`, computed identically to
every other module — see `shared/schemas.py`), `source="road_segmentation"`.
The `mask` payload here is `{"format": "polygon", "points": [[x, y], ...]}`
in original-frame pixel coordinates — lightweight and JSON-serializable;
rasterize with `cv2.fillPoly` if a dense bitmap is needed downstream.

## Integration

`model.predict(frame)` returns only `{"segmentations": [...]}`. The final
app merges this into the shared `FrameResult` via
`shared.schemas.merge_frame_results(...)` alongside the other three
modules' outputs — this module never constructs the full `FrameResult`
itself and never touches `detections`, `tracks`, `alerts`, or `analytics`.

## Known limitations

- Dataset 2's real class list/split has not been verified in this
  scaffold — `class_mapping.py`'s mapping is provisional until
  `verify_mapping_against_dataset()` is run against the actual downloaded
  `data.yaml`.
- No official test split exists for Dataset 1; only an internal held-out
  split (if created) or the published validation set should be quoted in
  results.
- Mask output is polygon-based, not a dense bitmap — sufficient for
  overlay visualization and area computation, but downstream code needing
  per-pixel masks must rasterize the polygon itself.
