"""
Road Intelligence — Streamlit Dashboard
==========================================
Consumes ONLY the standardized FrameResult dict from shared/schemas.py.
Per the spec, this app doesn't own any model logic — it just loads each
module's ModuleModel, calls .predict()/.process(), merges the results, and
displays them. All the actual logic lives in app/dashboard_logic.py so it
can be unit-tested without a Streamlit runtime; this file is UI glue only.

Each module is loaded lazily and independently: if a module's weight file
isn't on disk yet (a teammate hasn't trained/shared it), the dashboard skips
that module with a visible warning instead of crashing. This lets you run
the dashboard the moment ANY one module (e.g. road_damage) has real
weights, without waiting on the whole team.

Run with:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.dashboard_logic import (
    draw_all_detections,
    load_image_from_bytes,
    load_models,
    render_segmentation_overlay,
    run_pipeline,
)

st.set_page_config(page_title="Road Intelligence", layout="wide")

cached_load_models = st.cache_resource(show_spinner=False)(load_models)

st.title("🛣️ Road Intelligence Dashboard")
st.caption(
    "Uploads run through every module that has trained weights available; "
    "modules without weights yet are skipped and shown in the sidebar."
)

models, status = cached_load_models()

with st.sidebar:
    st.header("Module status")
    for name, state in status.items():
        icon = "✅" if state.startswith("loaded") else ("⚠️" if state.startswith("missing") else "❌")
        st.write(f"{icon} **{name}** — {state}")
    st.divider()
    st.caption(
        "Drop trained `.pt` files into `models/` (see shared/README.md for "
        "exact filenames) and reload the page to pick them up."
    )
    show_json = st.checkbox("Show raw merged FrameResult JSON", value=False)

uploaded = st.file_uploader("Upload a road image", type=["jpg", "jpeg", "png", "bmp"])

if uploaded is None:
    st.info("Upload an image to run it through the available modules.")
    st.stop()

frame = load_image_from_bytes(uploaded.read())
st.write(f"Image: {frame.shape[1]}×{frame.shape[0]} px")

with st.spinner("Running inference..."):
    started = time.perf_counter()
    final_result = run_pipeline(models, frame)
    elapsed = time.perf_counter() - started

st.success(f"Done in {elapsed:.2f}s")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Original")
    st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
with col2:
    st.subheader("Detections + Segmentation")
    annotated = render_segmentation_overlay(frame, final_result["segmentations"])
    annotated = draw_all_detections(annotated, final_result["detections"])
    st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

st.subheader("Analytics")
analytics = final_result["analytics"]
metric_cols = st.columns(6)
metric_cols[0].metric("Vehicles", analytics.get("vehicle_count", 0))
metric_cols[1].metric("Traffic density", analytics.get("traffic_density", "n/a"))
metric_cols[2].metric("Road damage", analytics.get("road_damage_count", 0))
metric_cols[3].metric("Obstacles", analytics.get("obstacle_count", 0))
metric_cols[4].metric("Speed bumps", analytics.get("speed_bump_count", 0))
health_score = analytics.get("road_health_score")
metric_cols[5].metric(
    "Road Health Score", f"{health_score:.0f}/100" if health_score is not None else "n/a"
)
if analytics.get("road_health_note"):
    st.caption(analytics["road_health_note"])

if final_result["alerts"]:
    st.subheader("Alerts")
    for alert in final_result["alerts"]:
        st.warning(f"**{alert.get('type', 'alert')}** — {alert.get('message', alert)}")

if final_result["detections"]:
    st.subheader("Detections")
    st.dataframe(
        [
            {
                "class": d["class"],
                "confidence": round(d["confidence"], 3),
                "source": d["source"],
                "bbox": [round(v, 1) for v in d["bbox"]],
            }
            for d in final_result["detections"]
        ],
        use_container_width=True,
    )

if final_result["segmentations"]:
    st.subheader("Segmentations")
    st.dataframe(
        [
            {
                "class": s["class"],
                "confidence": round(s["confidence"], 3),
                "area_ratio": round(s["area_ratio"], 4),
                "source": s["source"],
            }
            for s in final_result["segmentations"]
        ],
        use_container_width=True,
    )

if show_json:
    st.subheader("Raw merged FrameResult")
    st.json(final_result)
