# AIDOOG Dexterous Triage Lab - Judge Brief

Registration UUID: `6c3b08a9-5fb8-4e60-bd5d-d02d90f40ab9`

## Inspect First

1. `demo.mp4` - generated 60 second MuJoCo video.
2. `data/rollout_summary.json` - 20/20 gates and object-specific skill evidence.
3. `data/sensor_log.csv` - synchronized phase, touch, route, alignment, rotation, and recovery telemetry.
4. `scene.xml` - self-contained MuJoCo workcell.
5. `run_demo.py` - deterministic behavior policy that regenerates all artifacts.

## What Changed In v24

- Red cube: obstacle-aware dogleg route around hazard bars.
- Blue cylinder: upright alignment before release.
- Amber capsule: visible 216 degree precision rotation.
- Green sphere: planned slip pulse with regrasp recovery.

## Quantitative Evidence

- Local task suite: 20/20 gates.
- Five active finger contacts are logged.
- Maximum capsule rotation: 216 degrees.
- Slip recovery: 0.36 mm with 9x load hold.
- One command regenerates video, logs, summary, policy card, and randomized layout report.
