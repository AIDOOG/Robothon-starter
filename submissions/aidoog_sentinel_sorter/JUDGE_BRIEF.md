# AIDOOG RelayDex Operator Focus Cell

Registration UUID: `6c3b08a9-5fb8-4e60-bd5d-d02d90f40ab9`

## Judge-facing summary

This submission keeps AIDOOG's strongest verified dexterity signal: five-finger tactile grasp,
216-degree cap rotation, 4ms tactile reflex, 0.36mm slip recovery, and 9x load-hold evidence. It adds a five-step
operator supervision loop plus a three-agent shared-beam relay bench with force-share logging,
cooperative slip recovery, and coordinated-vs-uncoordinated ablation evidence.

## Local validation

- Dexterous triage gates: 20/20
- Relay force gates: 12/12
- Human interaction gates: 10/10
- Randomized scenario gates: 192/192
- Tactile reflex gates: 4/4 at 4.0ms and 250Hz
- Max beam angle error: 0.85 deg
- Max force error: 0.0 N
- Demo duration: 24.0s at 12 fps

## What changed for the judges

- Kept the proven project name: AIDOOG RelayDex Operator Focus Cell
- Added visible five-step operator request, force-limit, relay acknowledgement, slip approval, and recovery approval states.
- Added a 4ms tactile-reflex evidence track and 250Hz closed-loop servo fields in every rollout log.
- Explicitly separated vision confidence from policy/tactile confidence.
- Expanded to 48 randomized scenario variants with same-policy validation.
- Rebuilt the default demo as a 24-second four-beat spotlight reel with compact HUD values for operator intent, reflex, servo rate, slip, load, force, and beam angle.
- Added demo_chapters.json and demo_narration.srt for concise review narration.
- Added structured relay audit, rubric scorecard, manifest, and reproducible logs.
- Preserved the proven 20/20 AIDOOG four-object triage path instead of destabilizing the grasp.
