# Issue #19 — Multi-Query Retrieval Contract

This document records the **compatibility audit**, **mixed-language benchmark coverage**, and **English-language documentation** for Issue #19: *Language-agnostic multi-query retrieval and visualization*.  It lives in the evidence worktree while the implementation runs in parallel in task `t_09a2f4c9`.

## Scope

- Add a new golden set and audit runner focused on the Issue #19 contract.
- Document the opt-in multi-query request contract, legacy fallback, original-query write semantics, bounds, fusion, provenance, event ordering, and rollback.
- Verify that the **current** backend already satisfies the compatibility requirements that do not require new code.
- **Do not** change core retrieval, the frontend, or the WebSocket event shape.

## What Issue #19 asks for

The user message should remain the authoritative signal for writes (Hebbian update, neurogenesis).  Optional rewritten retrieval variants may be supplied, but:

1. the backend deduplicates and bounds variants;
2. retrieval and fusion happen once, in the backend;
3. graph expansion also happens once;
4. one ordered activation event is emitted;
5. visualization keeps canonical node/edge identity;
6. provenance is additive, not a breaking change.

The bridge and backend must remain language-agnostic: `language` is opaque metadata, never a retrieval gate.

## Current contract status (this commit)

| Requirement | Status | Evidence |
|-------------|--------|----------|
| Legacy single-query request remains valid | **already satisfied** | `POST /api/query {"query": "..."}` is the only path exercised today |
| Original query is the write-path signal | **already satisfied** | `api_query` uses `query` for `run_attention_and_plasticity` and `run_neurogenesis`; `user_prompt` is only prepended to the LLM context string |
| Optional `query_variants` accepted in request model | **not implemented** | Reserved in `docs/issue19-contract.md` for the implementation task |
| Variant deduplication and empty filtering | **not implemented** | Implemented only in the audit helper `_valid_variants` for contract testing |
| Max-variant bound enforced | **not implemented** | Audit defaults to `max_variants=4` |
| Parallel per-variant retrieval | **not implemented** | Current `attention()` takes a single query |
| Backend fusion once, then graph expansion once | **not implemented** | `attention()` currently expands one seed set |
| One ordered activation event after fusion | **already satisfied** | `api_query` emits exactly one `activation`, one optional `neurogenesis`, and one final `query_response` event with monotonic `sequence` |
| Canonical node/edge identity preserved | **already satisfied** | `api_graph` returns canonical IDs; no variant-specific nodes exist |
| Additive provenance in activated notes | **partially satisfied** | `routing.activation_details` already expose `role`, `hop`, `parent_id`, `vector_score`, `bm25_score`, `hybrid_score`, `hebbian_boost`, and `final_score`.  A `matched_by` field is reserved for the future |
| Feature can be disabled without code changes | **not implemented** | Configuration flags documented but not wired |
| Mixed-language benchmark coverage | **added here** | `benchmarks/golden_set_issue19.yaml` (21 queries) |

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
  Validates the golden-set contract: bounds, deduplication, mixed-language labeling, and write-semantics preservation.  This fails only if the golden set itself is malformed; it does not need the vault or embeddings.

- `python benchmarks/run_issue19_audit.py --mode baseline`  
  Runs the Issue #19 golden set through the existing ablation runner to record the **current** single-query baseline.  This requires the real vault and embeddings.

- `python benchmarks/run_issue19_audit.py --mode both --output benchmarks/issue19_audit.json`  
  Runs both and writes the full report.

## Expected future request contract

Existing clients stay valid:

```json
{"query": "Original user message"}
```

Opt-in multi-query form (not yet accepted by the server):

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
- No more than `max_variants` distinct variants are accepted (default 4).
- `language` is opaque metadata; the backend may ignore it.
- `weight` is a per-variant fusion hint; `rrf` ignores it, `weighted` uses it.
- The legacy bridge field `search_query` remains accepted as an alias for the single variant `query`.

## Provenance fields (reserved, backward-compatible)

Current activated-note payload shape already supports diagnostic fields:

```json
{
  "id": "canonical-note-id",
  "title": "...",
  "score": 0.84,
  "role": "seed",
  "hop": 0,
  "parent_id": null,
  "vector_score": 0.81,
  "bm25_score": 0.44,
  "hybrid_score": 0.84,
  "hebbian_boost": 0.0,
  "final_score": 0.84
}
```

Future additive field:

```json
"matched_by": [
  {"variant": "variant-0", "language": "en", "rank": 1, "score": 0.86}
]
```

Adding `matched_by` must not break existing clients; older payloads simply omit it.

## Configuration proposal

```yaml
multi_query_enabled: false        # opt-in flag
multi_query_max_variants: 4       # hard bound
multi_query_fusion: rrf         # 'rrf' | 'weighted'
query_rewrite_enabled: false    # bridge-side rewrite toggle
query_rewrite_timeout: 5          # seconds
```

These names are reserved in this document.  The implementation task will wire them into `bdh-config.yaml` and the bridge.

## Rollback

To fall back to single-query behavior, either:

1. Set `multi_query_enabled: false` in config; or
2. Omit `query_variants` from the request.

In both cases the backend must produce results identical to the current single-query path, including the same `activation`, `query_response`, and WebSocket event sequence.

## Residual risks (recorded, not fixed here)

1. **BM25 still hurts Italian paraphrase.** The Italian golden set (`benchmarks/golden_set_ita.yaml`) shows vector-only MRR 0.53 vs RRF 0.43.  Multi-query rewrite will not fix this if variants are still fed into the same BM25 index.
2. **Latency.** Even with parallel execution, two variants roughly double embedding and ChromaDB query time.  The default `max_variants=4` is a safety cap; real deployments may want `2`.
3. **Bridge contract drift.** The bridge uses `search_queries`, the backend will use `query_variants`.  The audit runner includes a normalization helper (`_valid_variants`) that both sides can share.
4. **Provenance payload size.** If every activated note lists every matching variant, large result sets will bloat the WebSocket payload.  The implementation should cap `matched_by` to the top 2 variants per note.

## Links

- Issue #19: `gh issue view 19`
- Main golden set: `benchmarks/golden_set.yaml`
- Italian golden set: `benchmarks/golden_set_ita.yaml`
- Evidence refactor report: `benchmarks/EVIDENCE_REFACTOR_REPORT.md`
- Audit runner: `benchmarks/run_issue19_audit.py`
- This contract doc: `docs/issue19-contract.md`
