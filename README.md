# Road Intelligence

Modular Road Intelligence Computer Vision System. Four independent modules
(road damage, road segmentation, obstacles/speed bumps, vehicle
tracking+analytics) share one architecture, class ontology, and output
schema, and feed one Streamlit dashboard.

**Start here: [`shared/README.md`](shared/README.md)** — the integration
contract every module must follow.

## Structure

```text
road_intelligence/
├── shared/                  # contract: config, classes.yaml, schemas, README
├── modules/
│   ├── road_damage/         # Person 1
│   ├── road_segmentation/   # Person 2
│   ├── obstacles/           # Person 3
│   └── tracking_analytics/  # Person 4
├── models/                  # trained weights (road_damage_best.pt, etc.)
├── data/                    # datasets (not committed — see .gitignore)
├── tests/                   # tests/test_<module>.py per module
├── app/
│   └── streamlit_app.py     # final dashboard — built after modules are ready
└── requirements.txt
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest tests/ -v
```

## Adding your module

1. Read `shared/README.md` and `shared/schemas.py` first.
2. Work inside `modules/<your_module>/` — you own that folder.
3. Your `ModuleModel.__init__(model_path, config=None)` / `.predict(frame)`
   must return a dict shaped like `shared.schemas.empty_frame_result()`.
4. Only use class names from `shared/classes.yaml`; map dataset-specific
   labels in your own `class_mapping.py`.
5. Add `tests/test_<module>.py` that calls `shared.schemas.validate_*`.
6. Fill in your module's `README.md` (dataset → preprocessing → training →
   validation → testing → model → inference → output schema → integration).

Do not build a separate Streamlit app, and do not redesign `shared/` without
team agreement — see the spec's "MOST IMPORTANT RULE".
