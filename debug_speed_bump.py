"""Standalone speed-bump model test — auto-finds an image, no editing needed."""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from ultralytics import YOLO

MODEL_PATH = "models/speed_bump_best.pt"

# Auto-find the first image anywhere under data/
image_paths = list(Path("data").rglob("*.jpg")) + list(Path("data").rglob("*.png")) + list(Path("data").rglob("*.jpeg"))

if not image_paths:
    print("No images found under data/. Put a test image there and rerun.")
    sys.exit(1)

IMAGE_PATH = str(image_paths[0])
print(f"Using image: {IMAGE_PATH}")

model = YOLO(MODEL_PATH)
print("Class names:", model.names)

for conf in [0.40, 0.25, 0.10, 0.05]:
    results = model.predict(IMAGE_PATH, conf=conf, verbose=False)[0]
    boxes = [(model.names[int(b.cls[0])], round(float(b.conf[0]), 3)) for b in results.boxes]
    print(f"conf={conf}: {len(boxes)} detections -> {boxes}")