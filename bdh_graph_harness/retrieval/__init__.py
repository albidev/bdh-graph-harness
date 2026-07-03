"""
BDH Graph Harness — Retrieval subpackage.

Re-exports main functions for easy importing.
"""

from bdh_graph_harness.retrieval.embeddings import (
    get_embeddings,
    cosine_similarity,
)
from bdh_graph_harness.retrieval.bm25 import BM25Index
from bdh_graph_harness.retrieval.hybrid import hybrid_score
from bdh_graph_harness.retrieval.attention import (
    compute_adaptive_threshold,
    attention as run_attention,
    format_context,
)
from bdh_graph_harness.retrieval.chroma_store import compute_all_embeddings

__all__ = [
    'get_embeddings',
    'cosine_similarity',
    'BM25Index',
    'hybrid_score',
    'compute_adaptive_threshold',
    'run_attention',
    'format_context',
    'compute_all_embeddings',
]