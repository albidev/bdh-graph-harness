"""
BDH Graph Harness — Attention module.

BDH-style attention: embedding seed (ChromaDB KNN) + graph traversal (k-hop expansion).
Includes adaptive threshold and hybrid search support.
"""

import os
import statistics
from collections import defaultdict, deque

from bdh_graph_harness.config import CONFIG, logger
from bdh_graph_harness.retrieval.embeddings import get_embeddings
from bdh_graph_harness.retrieval.bm25 import BM25Index
from bdh_graph_harness.retrieval.hybrid import hybrid_score
from bdh_graph_harness.graph.builder import _resolve_target


# ---------------------------------------------------------------------------
# Adaptive Threshold (Phase 3.3)
# ---------------------------------------------------------------------------

def compute_adaptive_threshold(scores, floor=0.15):
    """Compute a dynamic threshold from score distribution.

    Uses max(percentile_75, mean + 1*std, floor) to adaptively
    select only genuinely relevant notes per query.

    Args:
        scores: list of float scores from attention
        floor: minimum threshold (never go below this)
    Returns:
        float threshold value
    """
    if not scores or len(scores) < 3:
        return floor

    sorted_scores = sorted(scores)
    n = len(sorted_scores)

    # Percentile 75 (Q3)
    q75_idx = int(n * 0.75)
    q75 = sorted_scores[min(q75_idx, n - 1)]

    # Mean + 1 std
    mean = statistics.mean(scores)
    stdev = statistics.stdev(scores) if len(scores) > 1 else 0.0
    mean_plus_std = mean + stdev

    threshold = max(q75, mean_plus_std, floor)
    logger.info(f"Adaptive threshold: Q75={q75:.3f}, mean+std={mean_plus_std:.3f}, floor={floor} → {threshold:.3f}")
    return threshold


# ---------------------------------------------------------------------------
# Attention: Embedding seed + Graph traversal
# ---------------------------------------------------------------------------

def attention(query, nodes, edges, collection, k=None, max_hop=None, bm25_index=None):
    """
    BDH-style attention: find relevant notes via ChromaDB KNN search (seed)
    + graph traversal (k-hop expansion). Returns active note set with scores.

    Uses ChromaDB for vector similarity (HNSW index, cosine space).
    Phase 3.1: Hybrid search — combines vector similarity with BM25 keyword score.
    Phase 3.3: Adaptive threshold — dynamic threshold from score distribution.

    Improvements over v1:
    - Hub dampening: high-degree nodes get activation scaled by 1/log(degree)
    - Neighbor cap: only top-k neighbors per hop (by embedding similarity to query)
    - Hybrid: α * vector_sim + β * BM25_score (if bm25_index provided)
    - Adaptive threshold: max(Q75, mean+1std, floor) instead of fixed 0.25
    """
    if k is None:
        k = CONFIG['seed_count']
    if max_hop is None:
        max_hop = CONFIG['max_hop']

    # Precompute degree for each node
    degree = defaultdict(int)
    for src, links in edges.items():
        degree[src] += len(links)
        for link in links:
            target = _resolve_target(link['target'], nodes)
            if target:
                degree[target] += 1

    # Step 1: Embedding seed — ChromaDB KNN search
    query_emb = get_embeddings([query])[0]
    if not query_emb:
        return {}

    # ChromaDB returns distances (lower = more similar), convert to similarity
    overfetch = min(k * 5, collection.count()) if collection.count() > 0 else k * 5
    results = collection.query(
        query_embeddings=[query_emb],
        n_results=overfetch,
        include=['metadatas', 'distances'],
    )

    # Collect raw vector scores
    raw_vector_scores = {}
    if results['ids'] and results['ids'][0]:
        for i, note_id in enumerate(results['ids'][0]):
            dist = results['distances'][0][i]
            sim = max(0.0, 1.0 - dist)
            raw_vector_scores[note_id] = sim

    # Hybrid search: combine vector + BM25
    hybrid_enabled = CONFIG.get('hybrid_search', False) and bm25_index is not None

    if hybrid_enabled:
        # Get BM25 scores via hybrid_score for the same candidate set
        candidate_ids = list(raw_vector_scores.keys())
        scores = {}
        for nid in candidate_ids:
            combined = hybrid_score(nid, raw_vector_scores, bm25_index, query)
            if combined > CONFIG['active_threshold']:
                # Hub dampening
                if CONFIG['hub_dampening'] and degree.get(nid, 0) > CONFIG['hub_degree_threshold']:
                    dampen = 1.0 / (1.0 + 0.15 * (degree[nid] - CONFIG['hub_degree_threshold']))
                    combined *= dampen
                scores[nid] = combined
    else:
        scores = {}
        for note_id, sim in raw_vector_scores.items():
            if sim > CONFIG['active_threshold']:
                if CONFIG['hub_dampening'] and degree.get(note_id, 0) > CONFIG['hub_degree_threshold']:
                    dampen = 1.0 / (1.0 + 0.15 * (degree[note_id] - CONFIG['hub_degree_threshold']))
                    sim *= dampen
                scores[note_id] = sim

    # Adaptive threshold (Phase 3.3)
    if CONFIG.get('adaptive_threshold', False) and len(scores) >= 5:
        threshold = compute_adaptive_threshold(
            list(scores.values()),
            floor=CONFIG.get('threshold_floor', 0.15),
        )
        scores = {nid: s for nid, s in scores.items() if s >= threshold}

    # Top-k seeds
    seeds = sorted(scores.items(), key=lambda x: -x[1])[:k]

    # Step 2: Graph traversal — expand from seeds via wikilinks
    active = dict(seeds)
    queue = deque()
    for note_id, score in seeds:
        queue.append((note_id, score, 0))

    while queue:
        current_id, score, hop = queue.popleft()
        if hop >= max_hop:
            continue

        # Get neighbors — use ChromaDB to rank them by similarity to query
        neighbors = []
        for edge in edges.get(current_id, []):
            target = edge['target']
            target_id = _resolve_target(target, nodes)
            if target_id is None:
                continue
            # Get similarity from ChromaDB if available, else use 0.1
            n_sim = scores.get(target_id, 0.1)
            neighbors.append((target_id, n_sim))

        # Cap: only top max_neighbors_per_hop by similarity
        neighbors.sort(key=lambda x: -x[1])
        neighbors = neighbors[:CONFIG['max_neighbors_per_hop']]

        for target_id, n_sim in neighbors:
            # Decay score by hop distance
            new_score = score * (0.5 ** (hop + 1))

            # Hub dampening for the target
            if CONFIG['hub_dampening'] and degree.get(target_id, 0) > CONFIG['hub_degree_threshold']:
                dampen = 1.0 / (1.0 + 0.15 * (degree[target_id] - CONFIG['hub_degree_threshold']))
                new_score *= dampen

            if target_id in active:
                active[target_id] = max(active[target_id], new_score)
            elif new_score > CONFIG['active_threshold']:
                active[target_id] = new_score
                queue.append((target_id, new_score, hop + 1))

    return active


# ---------------------------------------------------------------------------
# Context formatting
# ---------------------------------------------------------------------------

def format_context(active_notes, nodes):
    """Format active notes as context for an LLM."""
    sorted_notes = sorted(active_notes.items(), key=lambda x: -x[1])
    parts = []
    for note_id, score in sorted_notes:
        node = nodes.get(note_id)
        if not node:
            continue
        parts.append(f"### {node['title']} (activation: {score:.3f})\n{node['text'][:300]}\n")
    return "\n---\n".join(parts)