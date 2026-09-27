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

## Snapshot 2026-09-27 (68 pairs)

Scope: both candidate vaults (core + one client vault), 7 days after the previous
snapshot. Pair definition unchanged: `extra.jev_choice` present AND status in
`{applied -> approve, pre_approved -> curator_approve, rejected/auto_rejected -> reject}`.
`pending_review` and `merged` are not pairs; 10 candidates carrying a Jev verdict are
excluded here (8 merged, 2 pending).

| Pattern | N | Delta vs 09-20 |
|---|---|---|
| jev needs_review -> human approve | 34 | +26 |
| jev reject -> auto-reject accepted | 16 | +0 |
| jev needs_review -> human reject | 11 | +10 |
| jev reject -> human reject | 2 | +2 |
| jev reject -> human approve | 5 | +1 |

Split: core 44 pairs, client vault 24 pairs (first snapshot for the second vault).

### Pattern 1 confirmed — and it is not a confidence problem

core: needs_review -> approve = 23/24 (96%). The approved set is exactly the
"principle distilled from a just-finished work session" class (credential decoupling,
typed decision contracts, span-level PII, worktree verification, ...). Jev scores them
0.41-0.75, and no threshold separates them from the single reject in the same band
(0.68). The signal is not in the number. The criteria tweak remains the right fix;
a threshold change alone cannot work.

### NEW Pattern 3 — promote rate is vault-relative, not universal

client vault: needs_review -> approve = 11/21 (52%), and confidence does not separate
the classes at all: approve spread 0.27-0.77, reject spread 0.26-0.77, fully
interleaved — the same 0.77 value appears on both sides. The curator notes name the
discriminator explicitly: "artifact / policy / duplicate, not vault knowledge" for the
rejects, versus routing and architecture concepts for the approves.

Jev scores generic durability; it has no model of what belongs in *this* vault.
Consequence for calibration: a single global promote rule cannot serve both vaults.

### NEW Pattern 4 — near-duplicate trios make the ground truth noisy

Three candidates expressing the same rule ("a validation gate must precede writing to
the vault") landed approve / reject / approve inside 3 seconds at confidence
0.27 / 0.38 / 0.57, the reject reason being "duplicate". The labels disagree with
themselves, so any precision measured against them has a ceiling. The dedup signal
(cluster_id) is computed but was not visible to the reviewer at decision time.
Before calibrating, review with duplicates grouped — otherwise the measurement
punishes Jev for the reviewer's inconsistency.

### Criteria v1 -> v2 removed a real false-reject class

reject-choice at confidence 0.94 (the auto-reject band):
`2026-09-19-core15` = 11 correct / 4 wrong (27% false reject, all four reverted by
hand within 30 s) -> `2026-09-20-enriched-state` = 5 correct / 0 wrong.
Overall reject-choice false-reject rate 27% -> 12%.
The revert path plus the revert-as-signal loop is what caught the v1 failures; it is
doing real work and should be kept.

### Auto-reject threshold is not load-bearing on current data

In the enriched-state era reject-choice only ever fires at 0.94 (n=5) or hedges
(0.31 / 0.60 / 0.75, n=3). Nothing lands in 0.80-0.93, so moving AUTO_REJECT_CONF
anywhere between 0.80 and 0.94 changes no decision. The v1 misses sat *at* 0.94, so
raising the threshold would not have prevented them — only the criteria change did.

### The pipeline.py keyword hardcode is measurably dead weight

`_state_for` injects `context_note` only when the title contains
gate/jev/curate/pipeline. Of the 34 needs_review -> approve pairs it covers 5 (15%);
of the 18 true rejects it fires on 1. The divergence class it was written for lies
85% outside its trigger. Remove at recalibration, as already planned.

## Calibration proposal (from 68 pairs)

1. Enable promote on `needs_review` **for core only** — measured precision 96% (23/24).
   Gate it behind the criteria tweak below and keep the one known counterexample
   (generic UX heuristic, 0.68) as a regression case.
2. Do **not** enable auto-promote for the client vault (52%). Human stays in the loop.
3. Leave AUTO_REJECT_CONF at 0.80 — no decision sits in the 0.80-0.93 gap.
4. Add to the `promote` criteria: "principles distilled from a completed work session
   are valid promote candidates" — the measured dominant approve class.
5. Add to the `reject` criteria: "process artifacts, repo/PII hygiene rules and
   policies that are not vault knowledge" — the measured dominant reject class in the
   client vault, currently absent from the criteria map.
6. Remove the `context_note` keyword hardcode in `curate/pipeline.py`.
7. Re-run golden validation A/B before enabling any auto-action. Auto-approve only at
   measured promote-precision >= 90%: core clears it (96%), the pooled figure (76%)
   does not — so the rule must be scoped per vault.

Caveat unchanged: provider variance (identical input scored 0.68 then 0.94) means every
number above is a single-draw estimate; quantify variance at calibration time.
