# AIDOOG Precision Capsule Rescue - Judge Brief

Registration UUID: `6c3b08a9-5fb8-4e60-bd5d-d02d90f40ab9`

## Why Inspect This Entry First

AIDOOG Precision Capsule Rescue keeps the strongest AIDOOG evidence, five-finger grasp plus 216-degree rotation, and rebuilds the task around closed-loop residual recovery. The video and generated artifacts expose raw-vs-corrected visual-servo error, contact balance, slip observer recovery, stress-test improvement, and a five-finger contact timeline.

## Inspect First

1. `demo.mp4` - generated 64 second MuJoCo video.
2. `data/rollout_summary.json` - task gates, residual metrics, stress summary, and contact summary.
3. `data/stress_eval.json` - fixed-seed baseline-vs-residual comparison.
4. `data/contact_timeline.json` - five-finger active-contact and balance evidence.
5. `data/behavior_policy.json` - policy inputs, outputs, stages, and evidence.
6. `scene.xml` - self-contained MuJoCo scene with capsule, pod, button, sensors, and five-finger gripper.
7. `run_demo.py` - deterministic generator for all artifacts.

## Quantitative Evidence

- Five active fingertips are logged during the grasp.
- Capsule marker reaches 216 degrees.
- Residual corrected median visual-servo error is lower than raw median error.
- Stress evaluation reports baseline-vs-residual improvement across fixed seeds.
- Contact timeline reports stable five-finger hold and recovery-window samples.
- One command regenerates video, sensor log, trajectory, summary, policy card, stress evaluation, contact timeline, subtitles, and layout report.

## Rubric Mapping

- Runnability: one Python command regenerates all artifacts.
- MuJoCo depth: free bodies, slide/hinge joints, actuators, touch sensors, IMU sensors, frame-position sensors, contacts, and rendered camera motion.
- Task design: high-risk fragile capsule rescue with scan, grasp, 216-degree rotation, disturbance carry, residual recovery, pod placement, confirmation, and export.
- Control: deterministic stage prior plus residual visual-servo/contact/slip correction.
- Dexterity: five-finger grasp, balanced contacts, in-hand rotation, and slip recovery.
- Engineering quality: deterministic artifacts, structured metrics, contact timeline, and stress evaluation.
- Presentation: short overlays show stage, raw-to-corrected error, active fingers, slip observer, and rotation.
- Innovation: combines dexterous capsule rescue, residual recovery, stress testing, and dataset export in one self-contained MuJoCo task.
