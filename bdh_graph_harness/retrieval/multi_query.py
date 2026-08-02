"""
BDH Graph Harness — Multi-query retrieval module.

Language-agnostic multi-query retrieval: accept one or more query variants,
retrieve candidates in parallel, deduplicate by note ID, fuse rankings once,
then perform a single graph expansion and Hebbian ranking pass.

Design constraints (from issue #19):
- No language-specific branching.
- Optional query_variants; missing variants fall back to the original query.
- Bounded variant count; empty/duplicate variants removed.
- Deterministic fusion (RRF or existing weighted hybrid abstraction).
- One graph expansion, one activation event, canonical note provenance.
"""
from collections import defaultdict
from dataclasses import dataclass

from bdh_graph_harness.config import CONFIG
from bdh_graph_harness.retrieval.attention import attention
from bdh_graph_harness.retrieval.okf_policy import (
    evaluate_okf_metadata,
    is_okf_retrieval_policy_enabled,
)


__all__ = [
    "QueryVariant",
    "normalize_query_variants",
    "multi_query_attention",
    "rrb_fuse",
    "weighted_fuse",
    "count_multivariant_hits",
]


@dataclass
class QueryVariant:
    """A single query variant used for retrieval."""

    query: str
    language: str | None = None
    weight: float = 1.0

    def as_dict(self) -> dict:
        return {
            "query": self.query,
            "language": self.language,
            "weight": self.weight,
        }


DEFAULT_MAX_VARIANTS = 3


def _variant_label(index: int, language: str | None) -> str:
    if language:
        return f"variant-{index} ({language})"
    return f"variant-{index}"


def normalize_query_variants(
    original_query: str,
    variants_in: list | None = None,
    max_variants: int | None = None,
) -> list[QueryVariant]:
    """Validate, deduplicate, and bound query variants.

    The original query is always kept as variant-0 so that the fused result
    set never loses the user's exact wording. If no variants are supplied,
    returns a single-item list containing only the original query.

    Args:
        original_query: the raw user query.
        variants_in: optional list of dicts/strings describing extra variants.
        max_variants: hard cap on the number of accepted variants
            (defaults to CONFIG['multi_query_max_variants'] or 3).

    Returns:
        A list of normalized QueryVariant objects, deduplicated and bounded.
    """
    max_variants = int(
        max_variants
        or CONFIG.get('multi_query_max_variants', DEFAULT_MAX_VARIANTS)
    )

    base = original_query.strip()
    if not base:
        return []

    seen = {base.lower()}
    accepted = [QueryVariant(query=base, language='original', weight=1.0)]

    if not variants_in:
        return accepted

    for item in variants_in:
        if isinstance(item, str):
            candidate = QueryVariant(query=item.strip())
        elif isinstance(item, dict):
            q = str(item.get('query', '')).strip()
            if not q:
                continue
            language = item.get('language')
            if language is not None:
                language = str(language).strip() or None
            weight = item.get('weight', 1.0)
            try:
                weight = float(weight)
            except (TypeError, ValueError):
                weight = 1.0
            candidate = QueryVariant(query=q, language=language, weight=weight)
        else:
            continue

        if not candidate.query:
            continue
        key = candidate.query.lower()
        if key in seen:
            continue
        seen.add(key)
        accepted.append(candidate)
        if len(accepted) >= max(1, int(max_variants)):
            break

    return accepted


def rrb_fuse(
    variant_results: list[tuple[QueryVariant, dict]],
    k: int = 60,
) -> dict[str, float]:
    """Reciprocal Rank Fusion over per-variant rankings.

    Each variant contributes 1 / (k + rank). Notes are ranked by the sum
    across variants and normalized to [0, 1].
    """
    if not variant_results:
        return {}

    per_note = defaultdict(float)
    for _variant, active in variant_results:
        sorted_notes = sorted(active.items(), key=lambda x: -x[1])
        for rank, (note_id, _) in enumerate(sorted_notes, start=1):
            per_note[note_id] += 1.0 / (k + rank)

    if not per_note:
        return {}

    max_score = max(per_note.values())
    if max_score == 0:
        return {nid: 0.0 for nid in per_note}
    return {nid: score / max_score for nid, score in per_note.items()}


def weighted_fuse(
    variant_results: list[tuple[QueryVariant, dict]],
) -> dict[str, float]:
    """Weighted max-normalized fusion over per-variant scores."""
    if not variant_results:
        return {}

    per_note_max = {}
    for variant, active in variant_results:
        weight = max(0.0, variant.weight)
        for note_id, score in active.items():
            s = score * weight
            per_note_max[note_id] = max(per_note_max.get(note_id, 0.0), s)

    if not per_note_max:
        return {}

    max_score = max(per_note_max.values())
    if max_score == 0:
        return {nid: 0.0 for nid in per_note_max}
    return {nid: score / max_score for nid, score in per_note_max.items()}


def count_multivariant_hits(
    variant_results: list[tuple[QueryVariant, dict]],
) -> dict[str, int]:
    """Return how many distinct variants matched each note_id."""
    counts = defaultdict(int)
    for _variant, active in variant_results:
        for note_id in active:
            counts[note_id] += 1
    return dict(counts)


def _relabel_matches(
    variant_results: list[tuple[QueryVariant, dict]],
) -> dict[str, list[dict]]:
    """Build canonical matched_by provenance for each note_id."""
    matched_by: dict[str, list[dict]] = defaultdict(list)
    for idx, (variant, active) in enumerate(variant_results):
        if not active:
            continue
        label = _variant_label(idx, variant.language)
        sorted_notes = sorted(active.items(), key=lambda x: -x[1])
        for rank, (note_id, score) in enumerate(sorted_notes, start=1):
            matched_by[note_id].append({
                "variant": label,
                "language": variant.language,
                "rank": rank,
                "score": round(score, 4),
            })
    return dict(matched_by)


def _gather_variant_results(
    variants: list[QueryVariant],
    nodes: dict,
    edges: dict,
    collection,
    bm25_index,
    hebbian_state: dict | None,
    k: int,
    max_hop: int,
) -> list[tuple[QueryVariant, dict]]:
    """Run attention() for each variant and return (variant, active) pairs.

    This helper is intentionally synchronous: callers already run it inside
    asyncio.to_thread when invoked from an async handler, so we must not call
    asyncio.to_thread again or inspect the event loop here. That avoids nested
    executor deadlocks and makes the function testable without a running loop.
    """
    gathered = []
    for variant in variants:
        # Variant retrieval is seed-only. Graph expansion belongs exclusively
        # to the canonical original-query attention call below; expanding each
        # variant would make provenance metrics count graph neighbors as
        # independent variant matches.
        active = attention(
            variant.query, nodes, edges, collection, k, 0,
            bm25_index, hebbian_state, None,
        )
        gathered.append((variant, active))
    return gathered


def multi_query_attention(
    original_query: str,
    query_variants: list | None,
    nodes: dict,
    edges: dict,
    collection,
    bm25_index=None,
    hebbian_state=None,
    *,
    k: int | None = None,
    max_hop: int | None = None,
    max_variants: int | None = None,
    enabled: bool = True,
) -> tuple[dict, dict]:
    """Retrieve once per query variant, fuse, and return canonical activation.

    This function does **not** modify Hebbian state. It returns a tuple
    ``(active, routing_meta)`` where ``active`` is the canonical fused note
    set and ``routing_meta`` contains provenance:

        - query: original query
        - query_variants: accepted normalized variants
        - activation_details: canonical list with role/hop/scores
        - matched_by: per-note variant provenance
        - multi_query_fusion: method used
        - multi_query_variant_count
        - multi_query_unique_notes
        - multi_query_multivariant_hits

    The single-query path (no query_variants) is behaviorally equivalent to
    a single attention() call and carries the same routing keys. The
    caller is responsible for invoking this function inside a thread when
    running in an async handler (the routes do this via asyncio.to_thread).
    """
    variants = normalize_query_variants(
        original_query, query_variants, max_variants=max_variants
    )
    k = k or CONFIG.get('seed_count', 5)
    max_hop = max_hop if max_hop is not None else CONFIG.get('max_hop', 2)

    # Run retrieval for every variant. attention() is CPU/IO-bound
    # (embeddings + graph traversal); callers run this function inside
    # asyncio.to_thread, so the per-variant calls stay off the event loop.
    gathered = _gather_variant_results(
        variants, nodes, edges, collection, bm25_index, hebbian_state, k, max_hop
    )

    fusion_method = CONFIG.get('multi_query_fusion', 'rrb')
    if fusion_method == 'rrb':
        fused_scores = rrb_fuse(gathered, k=CONFIG.get('rrb_k', 60))
    else:
        fused_scores = weighted_fuse(gathered)

    # Build matched_by provenance before re-ranking.
    matched_by = _relabel_matches(gathered)
    multivariant_hits = count_multivariant_hits(gathered)

    # Canonical attention run on the original query: this is the single graph
    # expansion we keep regardless of how many variants were supplied. It also
    # provides the standard routing keys (vector/bm25/hybrid top scores, etc.).
    routing_meta: dict = {}
    canonical_active = attention(
        original_query, nodes, edges, collection, k, max_hop,
        bm25_index, hebbian_state, routing_meta=routing_meta,
    )

    details_by_id = {
        d['id']: d for d in routing_meta.get('activation_details', [])
    }

    # Apply fused scores to notes that were found by variants, keeping the
    # canonical graph traversal structure (role/hop/parent_id) intact. If a
    # note only appears via variants, we add it as a seed-level result.
    active: dict[str, float] = {}
    for note_id, score in canonical_active.items():
        active[note_id] = fused_scores.get(note_id, score)
        if note_id in details_by_id:
            details_by_id[note_id]['final_score'] = round(active[note_id], 4)

    for note_id, fused_score in sorted(fused_scores.items(), key=lambda x: -x[1]):
        if note_id in active:
            continue
        active[note_id] = fused_score
        details_by_id[note_id] = {
            'id': note_id,
            'role': 'seed',
            'hop': 0,
            'parent_id': None,
            'final_score': round(fused_score, 4),
        }

    # Enrich activation details with provenance.
    enriched_details = []
    for detail in details_by_id.values():
        note_id = detail['id']
        enriched = dict(detail)
        if is_okf_retrieval_policy_enabled() and "okf_policy" not in enriched:
            enriched["okf_policy"] = evaluate_okf_metadata(nodes.get(note_id))
        if note_id in matched_by:
            enriched['matched_by'] = matched_by[note_id]
        if note_id in multivariant_hits and multivariant_hits[note_id] > 1:
            enriched['variant_hits'] = multivariant_hits[note_id]
        enriched_details.append(enriched)

    routing_meta.update({
        'query': original_query,
        'query_variants': [v.as_dict() for v in variants],
        'activation_details': enriched_details,
        'multi_query_fusion': fusion_method,
        'multi_query_variant_count': len(variants),
        'multi_query_unique_notes': len(active),
        'multi_query_multivariant_hits': sum(
            1 for c in multivariant_hits.values() if c > 1
        ),
        'multi_query_enabled': bool(enabled),
        'vector_top_score': routing_meta.get('vector_top_score', 0.0),
        'bm25_top_score': routing_meta.get('bm25_top_score', 0.0),
        'hybrid_top_score': routing_meta.get('hybrid_top_score', 0.0),
        'hybrid_enabled': routing_meta.get('hybrid_enabled', False),
        'hybrid_fusion': routing_meta.get('hybrid_fusion'),
    })

    return active, routing_meta
