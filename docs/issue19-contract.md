# Issue #19 — Multi-Query Retrieval Contract

This document records the **compatibility audit**, **mixed-language benchmark coverage**, and **English-language documentation** for Issue #19: *Language-agnostic multi-query retrieval and visualization*.

## Scope

- Add a new golden set and audit runner focused on the Issue #19 contract.
- Document the opt-in multi-query request contract, legacy fallback, original-query write semantics, bounds, fusion, provenance, event ordering, and rollback.
- Verify the backend, frontend, and WebSocket compatibility requirements against the implementation.
- Keep the new fields additive so existing single-query clients remain valid.

## What Issue #19 asks for

The user message should remain the authoritative signal for writes (Hebbian update, neurogenesis).  Optional rewritten retrieval variants may be supplied, but:

1. the backend deduplicates and bounds variants;
2. retrieval and fusion happen once, in the backend;
3. graph expansion also happens once;
4. one ordered activation event is emitted;
5. visualization keeps canonical node/edge identity;
6. provenance is additive, not a breaking change.

The bridge and backend must remain language-agnostic: `language` is opaque metadata, never a retrieval gate.

## Current contract status

| Requirement | Status | Evidence |
|-------------|--------|----------|
| Legacy single-query request remains valid | **already satisfied** | `POST /api/query {"query": "..."}` is the only path exercised today |
| Original query is the write-path signal | **already satisfied** | `api_query` uses `query` for `run_attention_and_plasticity` and `run_neurogenesis`; `user_prompt` is only prepended to the LLM context string |
| Optional `query_variants` accepted in request model | **satisfied** | `api_query` accepts the optional list and preserves the legacy `{query}` request |
| Variant deduplication and empty filtering | **satisfied** | `normalize_query_variants()` removes empty and case-insensitive duplicate candidates |
| Max-variant bound enforced | **satisfied** | `multi_query_max_variants` bounds the normalized list, including the original query |
| Parallel per-variant retrieval | **satisfied** | Variant seed retrieval runs off the event loop and is fused in `retrieval/multi_query.py` |
| Backend fusion once, then graph expansion once | **satisfied** | Variant results are fused, followed by one canonical original-query attention/expansion pass |
| One ordered activation event after fusion | **satisfied** | `api_query` emits one `activation` event with monotonic `sequence` after retrieval and fusion |
| Canonical node/edge identity preserved | **satisfied** | `api_graph` and the frontend use canonical IDs; activation never creates variant-specific nodes or edges |
| Additive provenance in activated notes | **satisfied** | Activated notes expose `matched_by` and `variant_hits` alongside the existing routing fields |
| Feature can be disabled without code changes | **satisfied** | `multi_query_enabled: false` is the default and omitting variants preserves the single-query path |
| Mixed-language benchmark coverage | **satisfied** | `benchmarks/golden_set_issue19.yaml` contains 18 contract queries; the evidence report also records 40-query EN/ITA retrieval results |

## Golden set: `benchmarks/golden_set_issue19.yaml`

A 18-query set covering the Issue #19 contract surface:

| Category | Count | What it tests |
|----------|-------|---------------|
| compatibility-legacy | 2 | Single-query requests identical to today's API |
| write-semantics | 2 | Original query must drive Hebbian write path |
| mixed-language | 5 | Italian + English technical terms in the same query |
| bounds | 2 | Empty/duplicate variant removal and max-variant cap |
| fusion-provenance | 2 | Notes matched by multiple variants |
| code-identifier | 3 | Technical identifiers that must survive rewrite/filtering |
| event-ordering | 1 | Read-only retrieval path (`learn=false`) |
| rollback | 1 | Single-query fallback when variants are disabled |
| **Total** | **18** | |

Every entry uses the same YAML schema as the existing golden sets, with an optional `variants:` list.  Each variant has `query`, `language`, and `weight`.

## Audit runner: `benchmarks/run_issue19_audit.py`

Two modes:

- `python benchmarks/run_issue19_audit.py --mode contract`
  Validates the golden-set contract: bounds, deduplication, mixed-language labeling, and write-semantics preservation. This fails only if the golden set itself is malformed; it does not need the vault or embeddings.

- `python benchmarks/run_issue19_audit.py --mode baseline`
  Runs the Issue #19 golden set through the existing ablation runner to record the **current** single-query baseline. This requires the real vault and embeddings.

- `python benchmarks/run_issue19_audit.py --mode both --output benchmarks/issue19_audit.json`
  Runs both and writes the full report.

## Request contract

Existing clients stay valid:

```json
{"query": "Original user message"}
```

Opt-in multi-query form:

```json
{
  "query": "Original user message",
  "query_variants": [
    {"query": "First retrieval representation", "language": "opaque", "weight": 1.0},
    {"query": "Seconda rappresentazione", "language": "opaque", "weight": 1.0}
  ],
  "source": "automatic_retrieval",
  "learn": false,
  "respond": false
}
```

Rules:

- `query` is always required and is the **write-path signal**.
- `query_variants` is optional.  Empty and exact-duplicate variants are removed.
- No more than `max_variants` distinct variants are accepted (default 3).
- `language` is opaque metadata; the backend may ignore it.
- `weight` is a per-variant fusion hint; `rrb` ignores it, `weighted` uses it.
- The legacy bridge field `search_query` remains accepted as an alias for the single variant `query`.

## Provenance fields (implemented, backward-compatible)

Activated notes expose additive provenance without changing the canonical note identity:

```json
"matched_by": [
  {"variant": "variant-0", "language": "en", "rank": 1, "score": 0.86}
],
"variant_hits": 2
```

The UI renders this provenance as compact variant badges and the retrieval trace summarizes the per-variant counts. Older clients can ignore these fields safely.

## Configuration proposal

```yaml
multi_query_enabled: false        # opt-in flag
multi_query_max_variants: 3       # hard bound
multi_query_fusion: rrb           # 'rrb' | 'weighted'
query_rewrite_enabled: false    # bridge-side rewrite toggle
query_rewrite_timeout: 5          # seconds
```

These names are wired into the backend configuration.  The bridge can remain provider-neutral by mapping its opaque `search_queries` output to `query_variants`.

## Rollback

To fall back to single-query behavior, either:

1. Set `multi_query_enabled: false` in config; or
2. Omit `query_variants` from the request.

In both cases the backend must produce results identical to the current single-query path, including the same `activation`, `query_response`, and WebSocket event sequence.

## Residual risks (recorded, not fixed here)

1. **BM25 still hurts Italian paraphrase.** The Italian golden set (`benchmarks/golden_set_ita.yaml`) shows vector-only MRR 0.53 vs RRF 0.43.  Multi-query rewrite will not fix this if variants are still fed into the same BM25 index.
2. **Latency.** Even with parallel execution, two variants roughly double embedding and ChromaDB query time. The default `max_variants=3` is a safety cap; real deployments may want `2`.
3. **Bridge contract drift.** The bridge uses `search_queries`, while the backend uses `query_variants`.  The bridge mapping remains the integration boundary; language labels stay opaque metadata.
4. **Provenance payload size.** If every activated note lists every matching variant, large result sets will bloat the WebSocket payload.  The implementation should cap `matched_by` to the top 2 variants per note.

## Links

- Issue #19: `gh issue view 19`
- Main golden set: `benchmarks/golden_set.yaml`
- Italian golden set: `benchmarks/golden_set_ita.yaml`
- Evidence refactor report: `benchmarks/EVIDENCE_REFACTOR_REPORT.md`
- Audit runner: `benchmarks/run_issue19_audit.py`
- This contract doc: `docs/issue19-contract.md`
