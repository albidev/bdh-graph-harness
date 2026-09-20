# Design Proposal: Nightly Automatic Retrieval for Synapse Bootstrapping

**Status:** Proposal — for evaluation 2026-09-27 (cron verdict on dynamic-edge traversal)
**Author:** Hermes (design), Albi (approval pending)
**Repo:** bdh-graph-harness
**Related:** docs/hebbian-dynamic-edges.md, memory/source_policy.py, memory/hebbian.py

---

## Problem statement

A note created by nightly consolidation / Curate approval starts life with **zero
synapses**. Synapses are only born from co-activation during real retrieval queries.
Consequences:

1. A newly-synthesized note is invisible to the learned graph until a user happens
   to query in its neighborhood — potentially for weeks.
2. The "blind-spot" case (weak embedding + no wikilink to its semantic partners) is
   *never discovered*, because nothing ever tries to reach the note. Measured on
   2026-09-20: 1 of 9 strong non-wikilink synapse pairs is truly unreachable by
   vector search (`auth-header-interpolation-audit ↔ profile-scoped-mcp-auth`,
   cos=0.42, w=0.45).
3. The vault produces knowledge that its own retrieval cannot find again — a
   lifecycle gap between synthesis (write) and learning (use).

## Insight: the source is already reserved

`source_policy.py` registers an unused source:

```python
register_source(
    "automatic_retrieval",
    frequency_increment=1.0,          # ← too hot; must be lowered (see below)
    provenance_label="automatic_retrieval",
    allow_neurogenesis=True,          # ← must be False (see below)
)
```

No code path invokes it yet. This design completes a mechanism the source registry
already anticipated — it is not a new architectural direction.

## Proposal

A nightly job (cron, after semantic consolidation completes) runs **bootstrap
queries** through the standard attention pipeline with
`source="automatic_retrieval"`:

1. **Query candidates** (bounded, in priority order):
   a. Recent real queries from the vault's query log (past 48h), deduplicated.
   b. Approved-but-never-queried concepts (notes in `wiki/concepts/` with zero
      synapses): generate a query from the note title + first heading.
2. **Execution**: normal `POST /api/query` path — same `attention()`, same gates
   (dynamic relevance floor 0.35, trust model, propagation threshold). No
   privileged routing. The only difference is the source policy.
3. **Learning**: `hebbian_update(..., source="automatic_retrieval")` with
   dampened increment — see config.
4. **Audit**: every bootstrap query and its `activation_details` summary appended
   to the audit log with `source: automatic_retrieval`, so the effect is
   attributable and reversible.
5. **Shadow telemetry**: events flow to `.bdh-hebbian-shadow.jsonl` as usual —
   night-time dynamic-only recoveries enrich the eval dataset.

## Config specification (proposed defaults)

```yaml
# Nightly synapse bootstrap (automatic retrieval)
bootstrap_queries_enabled: true
bootstrap_max_queries_per_night: 5          # hard cap; conservative start
bootstrap_frequency_increment: 0.2          # dampened; do NOT use registry default 1.0
bootstrap_allow_neurogenesis: false         # queries never create notes
bootstrap_new_notes_only: false             # false = also replay recent real queries
bootstrap_min_hours_between_replays: 72     # don't reinforce the same query too often
bootstrap_query_source: recent_and_unqueried
bootstrap_audit: true
```

Required source-policy change (one line):

```python
register_source(
    "automatic_retrieval",
    frequency_increment=0.2,      # was 1.0 — match session_synthesis dampening
    provenance_label="automatic_retrieval",
    allow_neurogenesis=False,     # was True — automatic queries must not write notes
)
```

## Safety analysis

| Risk | Mitigation |
|---|---|
| Graph pollution from unverified associations | dampened increment 0.2; same trust/decay/pruning lifecycle as all synapses; consolidation downscale applies nightly |
| Rich-get-richer reinforcement loop | query selection bounded per night; replays rate-limited (72h); priority on zero-synapse notes, not popular ones |
| Bypassing user-intent evidence | learning increment 0.2 means a bootstrap co-activation needs ~5 repetitions to reach the strength of one real query; single-shot accidental pairings stay weak |
| Neurogenesis from noise | `allow_neurogenesis: false` at both config and source-policy level |
| Interference with the current experiment | NOT enabled until the traversal eval verdict (2026-09-27); the experiment measures dynamic-edge behavior under real-traffic learning only |

## What this fixes vs what it doesn't

**Fixes:**
- New notes get first synapses within one night instead of weeks.
- The blind-spot case (point 1 of the retrieval-gap analysis) is *exercised*: the
  nightly query is the discovery mechanism for "learned but unreachable" notes.
- Shadow telemetry gets a controlled probe surface (same query families nightly →
  comparable dynamic-only counts over time).

**Does not fix:**
- The seeder itself remains vector+BM25 (point 1's full fix — synapse-aware
  seeding — remains a separate design, gated on this one's measurement).
- Query generation quality: title-based queries are crude; they seed association,
  they don't curate relevance.

## Rollout plan

1. **2026-09-27 (cron verdict):** if traversal experiment passes, land the
   source-policy dampening + config behind `bootstrap_queries_enabled: false`.
2. **Week of 2026-09-29:** enable in dry-run for 3 nights (`bootstrap_audit: true`,
   dry_run flag) — verify query selection, activation distributions, no regressions.
3. **2026-10-02:** enable live with `bootstrap_max_queries_per_night: 5`.
4. **2026-10-09:** first measurement — new-synapse count from bootstrap vs real
   traffic, blind-spot count trend, shadow dynamic-only events.
5. Revert path: single config flag; synapses born from bootstrap carry the
   `semantic:automatic_retrieval`-style provenance label and can be bulk-reviewed
   via the audit log.

## Success criteria (30 days)

- ≥10 new synapses attributable to bootstrap queries (provenance-tagged) with
  weight ≥ 0.15 surviving one consolidation cycle.
- Blind-spot count (strong synapse + no wikilink + cos < 0.45) does not grow.
- Zero consolidation anomalies (rollback events) attributable to bootstrap.
- Real-traffic seed boost and dynamic-only retrieval behavior unchanged in kind.