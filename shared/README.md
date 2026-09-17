# shared/ — Integration Contract

This folder is the contract every module (`modules/road_damage`,
`modules/road_segmentation`, `modules/obstacles`, `modules/tracking_analytics`)
must build against. **Do not modify these files unless the whole team agrees**
— every module's tests import from here.

## Files

| File | Purpose |
|---|---|
| `classes.yaml` | The single source of truth for normalized class names. Never return a dataset-specific label (`D00`, `hcrack`, `plastic bottle`, ...) from a module — map it to one of these names in your `class_mapping.py`, or exclude it and document why. |
| `config.py` | Centralized thresholds (`CONFIDENCE_THRESHOLD`, `IOU_THRESHOLD`, `IMAGE_SIZE`, `DEVICE`), path anchors, and the `InferenceConfig` dataclass every `ModuleModel.__init__(model_path, config=None)` should accept. No module should hard-code these values locally. |
| `schemas.py` | The standardized `FrameResult` dict shape, `empty_frame_result()` / `merge_frame_results()` helpers, and `validate_*()` functions your `tests/test_<module>.py` should call. |

## The output contract, in one paragraph

Every module returns a plain dict shaped like `empty_frame_result()` from
`schemas.py`, filling in only the field(s) it owns (`detections` for road
damage/obstacles/speed bumps, `segmentations` for road segmentation, `tracks`
+ `alerts` + `analytics` for tracking). Every item inside `detections` is
`{"class", "confidence", "bbox", "source"}` with `class` drawn from
`classes.yaml`, `bbox` as `[x1, y1, x2, y2]` in original-frame pixel
coordinates with top-left origin, and `confidence` in `[0, 1]`. Every item in
`segmentations` additionally carries `mask`, `area_px`, and
`area_ratio = area_px / image_area`. See `schemas.py` for the exact
validators used in tests.

## Model interface every module exposes

```python
class SomeModuleModel:
    def __init__(self, model_path, config=None):
        ...

    def predict(self, frame):
        """frame: numpy.ndarray, BGR, HxWx3, any resolution.
        Returns a dict shaped like shared.schemas.empty_frame_result()."""
        ...
```

## Final merge (what the Streamlit app consumes)

```python
from shared.schemas import merge_frame_results

final_result = merge_frame_results(
    damage_result, segmentation_result, obstacle_result, tracking_result,
    frame_id=frame_id, timestamp=timestamp,
)
```

## Checklist before you open a PR into this repo

- [ ] Classes returned are only from `shared/classes.yaml`
- [ ] `bbox` = `[x1, y1, x2, y2]`, top-left origin, original frame dims
- [ ] `confidence` in `[0, 1]`
- [ ] Output validated with `shared.schemas.validate_frame_result()`
- [ ] No hard-coded thresholds — imported from `shared/config.py`
- [ ] No hard-coded absolute/local paths
- [ ] Works from a clean environment without the training pipeline
- [ ] `tests/test_<module>.py` passes
- [ ] README documents dataset → preprocessing → training → ... → integration
