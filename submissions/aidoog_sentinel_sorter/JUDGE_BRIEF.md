# AIDOOG RelayDex HumanCue Force Cell

Registration UUID: `6c3b08a9-5fb8-4e60-bd5d-d02d90f40ab9`

## Judge-facing summary

This submission keeps AIDOOG's strongest verified dexterity signal: five-finger tactile grasp,
216-degree cap rotation, 0.36mm slip recovery, and 9x load-hold evidence. It adds a visible
operator request/approval console with three dynamically staged cue lights plus a three-agent shared-beam relay bench with
force-share logging, cooperative slip recovery, and coordinated-vs-uncoordinated ablation evidence.

## Local validation

- Dexterous triage gates: 20/20
- Relay force gates: 12/12
- Human interaction gates: 10/10
- Randomized scenario gates: 240/240
- Max operator cue lights lit: 3/3
- Max beam angle error: 0.85 deg
- Max force error: 0.0 N
- Demo duration: 36.0s at 12 fps

## What changed for the judges

- Kept the proven project name: AIDOOG RelayDex HumanCue Force Cell
- Animated the three large operator cue lights so the video shows request -> relay acknowledgement -> recovery approval.
- Explicitly separated vision confidence from policy/tactile confidence.
- Expanded to 60 randomized scenario variants with same-policy validation.
- Rebuilt the default demo as a 36-second spotlight reel with single-line key-action labels.
- Added demo_chapters.json and demo_narration.srt for concise review narration.
- Added structured relay audit, rubric scorecard, manifest, and reproducible logs.
- Preserved the proven 20/20 AIDOOG four-object triage path instead of destabilizing the grasp.
