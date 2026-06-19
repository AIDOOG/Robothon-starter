# AIDOOG Dexterous Triage Lab - Judge Brief

Registration UUID: `6c3b08a9-5fb8-4e60-bd5d-d02d90f40ab9`

## What To Inspect First

1. `demo.mp4` - 60 second generated demo. The first sequence is the amber capsule five-finger grasp, 216-degree cap rotation, slip recovery, and 9x load hold.
2. `data/rollout_summary.json` - task-suite success, headline metrics, and artifact paths.
3. `data/sensor_log.csv` - sampled phase, object pose, cap rotation, five-finger contact, slip recovery, and load-hold evidence.
4. `data/behavior_policy.json` - behavior-cloned tactile phase policy and control inputs/outputs.
5. `validate_submission.py` - reproducibility gate that checks the UUID, artifacts, 20/20 gates, amber-first evidence, five-finger contact, 216-degree rotation, and 9x hold.

## Quantitative Evidence

- 20/20 verification gates passed.
- Four object classes: amber capsule, red cube, blue cylinder, green sphere.
- Six distractors and six randomized layout variants.
- Five active tactile fingers logged.
- 216-degree cap rotation logged in the video data.
- 0.36 mm slip-recovery metric.
- 9.0x load-hold metric.
- Mean vision confidence: 0.9841.
- Mean policy confidence: 0.9428.
- Final task success: true.

## Rubric Mapping

- Runnability: `python3 submissions/aidoog_sentinel_sorter/run_demo.py` regenerates the demo and data artifacts.
- MuJoCo depth: MJCF free bodies, slide and hinge joints, position actuators, collision geoms, touch sensors, wrist IMU, frame sensors, cameras, and dynamic lighting.
- Task design: cluttered triage workcell with four object types, target zones, distractors, randomized starts, and final placement checks.
- Control: behavior-cloned tactile phase policy with minimum-jerk motion, contact-gated stabilization, class-conditioned placement, and release verification.
- Dexterous manipulation: five-finger grasp, marked amber capsule cap rotation, slip recovery, and 9x load hold.
- Engineering quality: deterministic seed, structured runner, validator, sensor CSV, JSON policy, layout report, manifest, and documented run path.
- Presentation: generated 60 second H.264 video with compact score-evidence captions and the highest-value dexterity sequence at the start.
- Innovation: a compact disaster/medication triage simulator that also exports synchronized labels, poses, tactile state, confidence, and task metrics.

## Honest Scope

The controller is intentionally deterministic so all three AI judges can reproduce the same rollout quickly. The behavior policy is a lightweight behavior-cloned phase policy rather than a large neural RL model.
