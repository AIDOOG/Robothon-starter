# AIDOOG RelayDex HumanCue SceneLite Cell

Registration UUID: `6c3b08a9-5fb8-4e60-bd5d-d02d90f40ab9`

## Judge-facing summary

This submission keeps AIDOOG's strongest verified dexterity signal: five-finger tactile grasp,
216-degree cap rotation, 0.36mm slip recovery, and 9x load-hold evidence. It pairs the visible
human cue console with a three-agent shared-beam relay, 18 verified scene distractors, and 3 moving
scene-change cue gates.

## Local validation

- Dexterous triage gates: 20/20
- Relay force gates: 12/12
- Human interaction gates: 8/8
- Randomized scenario gates: 288/288
- Max beam angle error: 0.85 deg
- Max force error: 0.0 N
- Demo duration: 36.0s at 12 fps

## What changed for the judges

- New unique project name: AIDOOG RelayDex HumanCue SceneLite Cell
- Added larger visible operator request, relay acknowledgement, and recovery approval cue lights.
- Expanded visible physical distractors from 12 to 18 while preserving the proven grasp path.
- Added 3 moving scene-change cue gates for visible randomized-scene changes.
- Explicitly separated vision confidence from policy/tactile confidence.
- Expanded to 72 randomized scenario variants with same-policy validation.
- Rebuilt the default demo as a 36-second spotlight reel with shorter single-line overlays.
- Added demo_chapters.json and demo_narration.srt for concise review narration.
- Added structured relay audit, rubric scorecard, manifest, and reproducible logs.
- Preserved the proven 20/20 AIDOOG four-object triage path instead of destabilizing the grasp.
