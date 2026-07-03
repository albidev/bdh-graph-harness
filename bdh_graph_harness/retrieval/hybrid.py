"""
BDH Graph Harness — Hybrid scoring module.

Encapsulates the α*vector + β*BM25 combination logic used by attention().
"""

from bdh_graph_harness.config import CONFIG


def hybrid_score(note_id, raw_vector_scores, bm25_index, query, alpha=None, beta=None):
    """Compute hybrid score: α * vector_sim + β * BM25_score.

    Both components are in [0, 1]. Falls back to pure vector score
    if bm25_index is None or the note isn't in the BM25 index.

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
        bm_s = bm25_index.score(query, note_id)
        return alpha * vec_s + beta * bm_s
    else:
        return vec_s