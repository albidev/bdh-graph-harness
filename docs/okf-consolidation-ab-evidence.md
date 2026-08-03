# OKF -> Consolidation Candidate A/B Evidence (P1, task t_363cbaf3)

**Question:** Does OKF retrieval inflate consolidation candidates via the
indirect OKF -> retrieval -> Hebbian candidate path?

**Verdict: OKF is CONTRIBUTORY to synapse *composition*, EXONERATED from
candidate *count inflation*.** It changes WHICH learning seeds/synapses get
created, not HOW MANY. The per-query Hebbian write budget is bounded and
OKF-independent.

## Causal path traced (read-only)

1. `attention()` (`retrieval/attention.py`) adjusts seed scores via
   `apply_okf_retrieval_policy` (`retrieval/okf_policy.py`) — a pure ranking
   multiplier, never mutating state.
2. `routes._run_attention_and_plasticity_unlocked` extracts `learning_active`
   = notes with `role == 'seed'` from the top-k seeds.
3. `hebbian_update()` (`memory/hebbian.py`) filters by `hebbian_min_score`
   (0.15), takes the top `hebbian_learning_seed_count` (default **2**) as
   learning seeds, and creates exactly one undirected pair (clique among the
   top-2) per query.
4. `consolidate()` (`memory/consolidation.py`) later prunes stale-weak
   synapses from the accumulated state. It never reads OKF metadata.

**Therefore:** OKF can change *which* seeds rank highest (composition), but it
cannot change the number of new synapses per query, which is hard-capped by
`hebbian_learning_seed_count`. Consolidation candidate count is bounded at the
creation boundary, independent of OKF.

## Measured evidence (deterministic, offline, no live writes)

Fixture: 4 nodes with fixed vector scores; beta ranks highest on raw vectors
but is deprecated+stale; alpha/others carry positive OKF signals. Same query,
same graph (no edges -> no traversal), same state seed. OKF policy ON vs OFF.
Python 3.11.13, full suite 459 passed.

| Metric | OKF OFF | OKF ON |
|---|---|---|
| Query-1 learning seeds | `beta, alpha` | `alpha, gamma` |
| Query-1 new synapse | `{alpha,beta}` | `{alpha,gamma}` |
| Accumulated synapses (20 queries) | 1 | 1 |
| Synapse weight (min/max/mean) | 1.1595 | 1.1441 |
| Consolidation candidate proxy (weak below 0.15) | 0 of 1 | 0 of 1 |
| Distinct endpoints learned | `{alpha,beta}` | `{alpha,gamma}` |

**Key observations**
- Per-query new-synapse creation is **1 in both arms** on query 1, 0 on
  reinforces — bounded by `hebbian_learning_seed_count=2` regardless of OKF.
- Accumulated synapse **count is identical** (1) across both arms over 20
  queries. No inflation.
- The **pair created differs** (`alpha-beta` vs `alpha-gamma`) — OKF changes
  composition.
- Neither arm produced any synapse below the consolidation weak threshold, so
  neither arm creates candidate pressure from this loop.

## Deliverables

- `tests/test_okf_consolidation_ab.py` — 3 deterministic A/B regression tests
  (composition differs, no count inflation, bound holds over 20 queries).
- `scripts/okf_ab_evidence.py` — self-contained reproducible evidence harness.
- Commit `e337eb9b22791a1d90f599fde54dbef140e6940c` (`feature/okf-compatibility`).
- No changes to `consolidation.py`; no thresholds tuned; no live vault/state
  mutation.

## Reproduction

```bash
cd /Users/albi/Projects/bdh-graph-harness-okf
PYTHON=/Users/albi/.pyenv/versions/3.11.13/bin/python
"$PYTHON" scripts/okf_ab_evidence.py
"$PYTHON" -m pytest tests/test_okf_consolidation_ab.py -v
```
