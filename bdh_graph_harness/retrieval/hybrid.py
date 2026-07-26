"""
BDH Graph Harness — Hybrid scoring module.

Encapsulates the α*vector + β*BM25 combination logic used by attention().
"""
from bdh_graph_harness.config import CONFIG


_RRF_K = 60


def _rank_from_scores(scores: dict) -> dict:
    """Convert a dict of {id: score} into {id: 1-based rank}, highest score first."""
    sorted_items = sorted(scores.items(), key=lambda x: -x[1])
    return {nid: i + 1 for i, (nid, _) in enumerate(sorted_items)}


def reciprocal_rank_fusion_score(
    note_id,
    raw_vector_scores: dict,
    bm25_scores: dict,
    pool_size: int,
    k: int = _RRF_K,
) -> float:
    """RRF score over a shared candidate pool.

    Missing side gets rank = len(pool) + 1.
    """
    vec_rank = _rank_from_scores(raw_vector_scores).get(note_id, pool_size + 1)
    bm25_rank = _rank_from_scores(bm25_scores).get(note_id, pool_size + 1)
    return 1.0 / (k + vec_rank) + 1.0 / (k + bm25_rank)


def reciprocal_rank_fusion(
    raw_vector_scores: dict,
    bm25_scores: dict,
    k: int = _RRF_K,
) -> dict:
    """Return fused scores for the union of vector and BM25 candidates.

    Scores are min-max normalized so they are on the same scale as vector/BM25
    scores and can pass through the same adaptive threshold and hub-dampening
    path.
    """
    candidate_ids = sorted(set(raw_vector_scores) | set(bm25_scores))
    pool_size = len(candidate_ids)
    fused = {
        nid: reciprocal_rank_fusion_score(nid, raw_vector_scores, bm25_scores, pool_size, k)
        for nid in candidate_ids
    }
    if not fused:
        return fused
    min_v = min(fused.values())
    max_v = max(fused.values())
    span = max_v - min_v
    if span == 0:
        return {nid: 0.0 for nid in fused}
    return {nid: (v - min_v) / span for nid, v in fused.items()}
def hybrid_score(note_id, raw_vector_scores, bm25_index, query, alpha=None, beta=None):
    """Compute hybrid score: α * vector_sim + β * BM25_score.

    Both components are in [0, 1]. Falls back to pure vector score
    if bm25_index is None or the note isn't in the BM25 index.

    Uses batch BM25 normalization to avoid clamping artifacts.

    Args:
        note_id: the note being scored
        raw_vector_scores: dict {note_id -> vector_similarity}
        bm25_index: BM25Index instance or None
        query: the original query string (for BM25 scoring)
        alpha: weight for vector similarity (defaults to CONFIG['hybrid_alpha'])
        beta: weight for BM25 score (defaults to CONFIG['hybrid_beta'])

    Returns:
        float combined score in [0, 1]
    """
    if alpha is None:
        alpha = CONFIG.get('hybrid_alpha', 0.7)
    if beta is None:
        beta = CONFIG.get('hybrid_beta', 0.3)

    vec_s = raw_vector_scores.get(note_id, 0.0)

    if bm25_index is not None:
        # Use batch normalization for proper [0,1] BM25 scores
        bm25_scores = bm25_index.score_batch(query, list(raw_vector_scores.keys()))
        bm_s = bm25_scores.get(note_id, 0.0)
        return alpha * vec_s + beta * bm_s
    else:
        return vec_s
