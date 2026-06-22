# AIDOOG RelayDex HumanCue Spotlight Cell

Registration UUID: `6c3b08a9-5fb8-4e60-bd5d-d02d90f40ab9`

## Judge-facing summary

This submission keeps AIDOOG's strongest verified dexterity signal: five-finger tactile grasp,
216-degree cap rotation, 0.36mm slip recovery, and 9x load-hold evidence. It adds a visible
operator request/approval console with three large cue lights plus a three-agent shared-beam relay bench with force-share
logging, cooperative slip recovery, and coordinated-vs-uncoordinated ablation evidence.

## Local validation

- Dexterous triage gates: 20/20
- Relay force gates: 12/12
- Human interaction gates: 8/8
- Randomized scenario gates: 192/192
- Max beam angle error: 0.85 deg
- Max force error: 0.0 N
- Demo duration: 36.0s at 12 fps

## What changed for the judges

- New unique project name: AIDOOG RelayDex HumanCue Spotlight Cell
- Added larger visible operator request, relay acknowledgement, and recovery approval cue lights.
- Explicitly separated vision confidence from policy/tactile confidence.
- Expanded to 48 randomized scenario variants with same-policy validation.
- Rebuilt the default demo as a 36-second spotlight reel with single-line key-action labels.
- Added demo_chapters.json and demo_narration.srt for concise review narration.
- Added structured relay audit, rubric scorecard, manifest, and reproducible logs.
- Preserved the proven 20/20 AIDOOG four-object triage path instead of destabilizing the grasp.
