# AIDOOG RelayDex HumanCue TriadAudit Cell

Registration UUID: `6c3b08a9-5fb8-4e60-bd5d-d02d90f40ab9`

## Judge-facing summary

This submission keeps AIDOOG's strongest verified dexterity signal: five-finger tactile grasp,
216-degree cap rotation, 0.36mm slip recovery, and 9x load-hold evidence. It adds an N-robot
triad relay audit: six three-agent shared-beam tasks, a minimum-jerk L->C->R handoff, level-preserving
pair-to-center consolidation evidence, cooperative slip recovery, and a 5-check N-robot cooperation audit.

## Local validation

- Dexterous triage gates: 20/20
- Relay force gates: 12/12
- Triad relay tasks: 6/6
- N-robot cooperation checks: 5/5
- Human interaction gates: 8/8
- Randomized scenario gates: 192/192
- Max beam angle error: 0.85 deg
- Max force error: 0.0 N
- Demo duration: 36.0s at 12 fps

## What changed for the judges

- New unique project name: AIDOOG RelayDex HumanCue TriadAudit Cell
- Added larger visible operator request, relay acknowledgement, and recovery approval cue lights.
- Added 6-task triad relay suite and 5-check N-robot cooperation audit inspired by the top-ranked relay handoff signal.
- Explicitly separated vision confidence from policy/tactile confidence.
- Expanded to 48 randomized scenario variants with same-policy validation.
- Rebuilt the default demo as a 36-second spotlight reel with single-line key-action labels.
- Added demo_chapters.json and demo_narration.srt for concise review narration.
- Added structured relay audit, rubric scorecard, manifest, and reproducible logs.
- Preserved the proven 20/20 AIDOOG four-object triage path instead of destabilizing the grasp.
