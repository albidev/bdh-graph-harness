# Jev Golden Set — Divergence Analysis

> Running analysis of human-vs-Jev divergences in the Curate pipeline.
> Source of truth: candidate jsons (`extra.jev_*` + status) in each vault's
> `.bdh-candidates/`. This doc records the PATTERNS, not the raw pairs.

## Snapshot 2026-09-20 (29 pairs)

| Pattern | N | Meaning |
|---|---|---|
| jev reject + auto-reject accepted | 16 | dedup gate confirmed correct |
| jev needs_review -> human approve | 8 | THE signature (see below) |
| jev reject -> human approve | 4 | strong divergence |
| jev needs_review -> human reject | 1 | over-call (guards against always-approve bias) |

## Pattern 1 — "Principles distilled from just-finished work" (dominant)

Jev systematically UNDER-CALLS concepts born from a live work session:
definitions like "A mechanism/pattern/rule that governs..." extracted from
an implementation that just landed. Jev scores them 0.41-0.73 needs_review
or even reject; the human (who lived the session) approves them.

Why: the classifier sees a thin definition and no session context. The
concept is real and durable, but its evidence trail (commits, configs,
incidents) is invisible to the gate.

Planned fix (at ~50 pairs): criteria tweak adding to `promote`:
"principles distilled from a just-completed work session are valid promote
candidates". Plus the keyword hardcode in pipeline.py `_state_for`
(context_note for gate/jev/curate titles) becomes obsolete and gets removed.

## Pattern 2 — needs_review -> human reject (1 case)

Generic UX pattern (Bottom-Anchored Response) — Jev correctly smelled
weakness but hedged. Confirms Jev's confidence BAND, not just direction,
carries signal.

## Known provider issue — variance

Same input scored 0.68 then 0.94 across two classify runs (provider
non-determinism). Mitigated operationally by sticky verdicts (idempotent
gate), but any measured precision must also quantify variance (N runs,
same state). TODO at calibration time.

## Calibration plan (at ~50 pairs)

1. Recompute placeholder band vs real-concept band on enriched state
   (criteria_version 2026-09-20-enriched-state changed the distributions;
   old 0.8 threshold was calibrated on the poor state)
2. If the gap is clean: set new auto-reject threshold in the gap
3. Add the session-distilled principle definition to the promote criteria
4. Remove the keyword hardcode
5. Re-run golden validation A/B (v1 vs v2 criteria) before enabling any
   auto-action; auto-approve only at measured promote-precision >= 90%
