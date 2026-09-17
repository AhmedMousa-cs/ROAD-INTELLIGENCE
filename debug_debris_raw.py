"""
Quick standalone check: does debris_best.pt actually detect anything on a
given image, BEFORE class_mapping.py has a chance to drop it?

This bypasses DebrisModel entirely (and therefore class_mapping.py /
shared/config.py's confidence threshold) so you can see:
  1. What raw class names your model actually outputs (model.names)
  2. Whether it finds anything at all, at a very low confidence bar

Usage:
    python debug_debris_raw.py path/to/image.jpg
"""

import sys

from ultralytics import YOLO

MODEL_PATH = "models/debris_best.pt"


def main():
    if len(sys.argv) != 2:
        print("Usage: python debug_debris_raw.py path/to/image.jpg")
        sys.exit(1)

    image_path = sys.argv[1]

    model = YOLO(MODEL_PATH)

    print("=" * 60)
    print(f"Model: {MODEL_PATH}")
    print("Raw class names baked into this model (model.names):")
    for class_id, name in model.names.items():
        print(f"  {class_id}: {name!r}")
    print("=" * 60)

    # conf=0.01 on purpose -- we want to see EVERYTHING the model considers,
    # not just what would survive shared/config.py's real threshold.
    results = model.predict(source=image_path, conf=0.01, verbose=False)
    result = results[0]
    boxes = result.boxes

    if boxes is None or len(boxes) == 0:
        print("Model found NOTHING at all, even at conf=0.01.")
        print("-> This points to the model itself (undertrained / wrong weights),")
        print("   not the class_mapping.py step.")
        return

    print(f"Model found {len(boxes)} raw box(es) at conf>=0.01:")
    for box, conf, cls_id in zip(boxes.xyxy, boxes.conf, boxes.cls):
        cls_id = int(cls_id)
        raw_name = model.names.get(cls_id, str(cls_id))
        print(f"  class={raw_name!r}  confidence={float(conf):.3f}  bbox={[round(v, 1) for v in box.tolist()]}")

    print()
    print("Now compare each 'class=...' name above against the keys in")
    print("modules/obstacles/class_mapping.py's TACO_TO_ROAD_DEBRIS dict.")
    print("If a printed name isn't an EXACT match (case-sensitive) to a key")
    print("there, map_taco_category() returns None and DebrisModel silently")
    print("drops it -- that's almost certainly your '0 debris found' bug.")


if __name__ == "__main__":
    main()
