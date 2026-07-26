# Trusted Dynamic Hebbian Edges

BDH keeps two distinct graph layers:

- **Static edges** are declared knowledge: Obsidian wikilinks and other materialized graph relations.
- **Dynamic Hebbian edges** are learned associations: they are created by repeated co-activation and can be traversed during retrieval without becoming permanent wikilinks.

Static edges remain the source of explicit structure. Dynamic edges can improve reachability between related notes that have no declared link, but are deliberately guarded against weak or accidental correlations.

## Retrieval behavior

During attention traversal, eligible learned synapses add candidate neighbors to the current node's static adjacency. A dynamic-only result has explicit provenance:

```json
{
  "role": "hebbian_neighbor",
  "matched_by": "hebbian_edge",
  "hebbian_edge_weight": 0.8,
  "hebbian_edge_trust": 0.75
}
```

An existing static relation remains a static route even if a learned synapse also exists (`matched_by: "static_and_hebbian_edge"`). Trust attenuation applies only to `hebbian_edge`; it never down-ranks the declared graph.

Federated IDs stored by the Hebbian state are resolved back to canonical graph node IDs before traversal, so a logical learned relation is not silently lost because its storage ID differs from the graph ID.

## Trust score

A learned weight measures association strength. A separate trust factor measures whether that association has earned the right to influence retrieval.

The dynamic-edge score is:

```text
seed activation × dynamic hop decay × (0.5 + learned weight) × dynamic gain × trust
```

Trust is bounded to `[trust_floor, 1]` and combines:

1. **Frequency** — recurrent co-activation reaches full confidence at `hebbian_dynamic_frequency_saturation`.
2. **Consolidation** — an edge that survived at least one consolidation candidate cycle receives full consolidation confidence; an unconsolidated one is discounted.
3. **Recency** — inactive associations decay toward, but never below, the configured recency floor.

This is intentionally conservative. A strong but one-shot association can still be traversed, but it cannot outrank a repeated, consolidated, recent relation by accident.

## Configuration

| Parameter | Default | Meaning |
|---|---:|---|
| `hebbian_dynamic_edges_enabled` | `true` | Enable learned-only traversal. |
| `hebbian_dynamic_min_weight` | `0.15` | Ignore synapses below this learned weight. |
| `hebbian_dynamic_top_n` | `3` | Maximum dynamic candidates per active source note. |
| `hebbian_dynamic_gain` | `1.5` | Dynamic-edge score multiplier. |
| `hebbian_dynamic_hop_decay` | `0.6` | Per-hop decay for learned-only traversal. |
| `hebbian_dynamic_frequency_saturation` | `2.0` | Frequency at which recurrence confidence reaches 1. |
| `hebbian_dynamic_unconsolidated_trust` | `0.6` | Trust multiplier before consolidation evidence. |
| `hebbian_dynamic_recency_days` | `30` | Recency interpolation horizon in days. |
| `hebbian_dynamic_recency_floor` | `0.4` | Minimum recency confidence for older associations. |
| `hebbian_dynamic_trust_floor` | `0.25` | Lower bound for valid dynamic-edge trust. |
| `hebbian_dynamic_shadow_enabled` | `true` | Emit privacy-safe telemetry for dynamic-only retrieval. |

## Shadow telemetry

When a dynamic-only result is returned, BDH adds `routing.hebbian_dynamic_shadow` and appends a vault-local JSONL event. Events contain:

- a truncated SHA-256 query fingerprint, never the query text;
- dynamic-only note ID, rank, parent, learned weight, trust, and score;
- timestamp and count of dynamic-only results.

Telemetry is emitted only when there is a dynamic-only result. It is intended for evaluating learned-edge behavior under real usage without logging user prompts.

## Evaluation status

Dynamic adjacency is **operational and measured**, not treated as a universal retrieval-quality claim.

An edge-conditioned evaluation uses real, non-static Hebbian synapses, source-derived queries without target-title leakage, and compares static-only traversal with trusted dynamic traversal. In the 2026-07-26 run (65 source-seeded queries):

| Metric | Static-only | Trusted dynamic | Delta |
|---|---:|---:|---:|
| MRR | .0344 | .0904 | +.0560 |
| Recall@5 | .0923 | .1846 | +.0923 |
| Precision@5 | .0185 | .0369 | +.0184 |
| NDCG@5 | .0487 | .1118 | +.0631 |

This demonstrates that, when learned non-static relations are relevant to the query, trusted traversal can improve their retrieval. It does **not** demonstrate a general uplift across every vault query, and it is not a substitute for a human-validated general retrieval benchmark.

The earlier train-to-holdout trajectory benchmark remained flat because its training and holdout paths did not create or traverse the same learned relations. That result is a protocol limitation, not evidence that dynamic adjacency has no value.

## Operational rule

Do not tune `gain` or `hop_decay` upward merely because a local example looks better. Use the shadow telemetry and a fixed evaluation set to inspect regressions, dynamic-only recovery, and trust distributions before changing defaults.
