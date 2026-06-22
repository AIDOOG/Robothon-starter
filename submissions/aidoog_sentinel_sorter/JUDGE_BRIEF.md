# AIDOOG RelayDex Force Bench

Registration UUID: `6c3b08a9-5fb8-4e60-bd5d-d02d90f40ab9`

## Judge-facing summary

This submission keeps AIDOOG's strongest verified dexterity signal: five-finger tactile grasp,
216-degree cap rotation, 0.36mm slip recovery, and 9x load-hold evidence. It adds a visible
three-agent shared-beam relay bench with force-share logging, cooperative slip recovery, and
coordinated-vs-uncoordinated ablation evidence.

## Local validation

- Dexterous triage gates: 20/20
- Relay force gates: 12/12
- Randomized scenario gates: 144/144
- Max beam angle error: 0.85 deg
- Max force error: 0.0 N
- Demo duration: 60.0s at 12 fps

## What changed for the judges

- New unique project name: AIDOOG RelayDex Force Bench
- Explicitly separated vision confidence from policy/tactile confidence.
- Expanded to 36 randomized scenario variants with same-policy validation.
- Clarified video captions around task, relay event, and randomized scenario profile.
- Added demo_chapters.json and demo_narration.srt for concise review narration.
- Added structured relay audit, rubric scorecard, manifest, and reproducible logs.
- Preserved the proven 20/20 AIDOOG four-object triage path instead of destabilizing the grasp.
