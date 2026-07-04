"""
BDH Graph Harness — Attention module.

BDH-style attention: embedding seed (ChromaDB KNN) + graph traversal (k-hop expansion).
Includes adaptive threshold, hybrid search support, and Integrate-and-Fire model.
"""
import math
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

    Uses max(percentile_75, median + 0.5*std, floor) to adaptively
    select only genuinely relevant notes per query.

    The 0.5*std multiplier (instead of 1*std) prevents clustered
    scores from pushing the threshold too high and filtering out
    relevant notes that are close to the median.

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

    # Median + 0.5 std — less aggressive than mean + 1 std
    median = statistics.median(scores)
    stdev = statistics.stdev(scores) if len(scores) > 1 else 0.0
    median_half_std = median + 0.5 * stdev

    threshold = max(q75, median_half_std, floor)
    logger.info(f"Adaptive threshold: Q75={q75:.3f}, median+0.5std={median_half_std:.3f}, floor={floor} → {threshold:.3f}")
    return threshold


# ---------------------------------------------------------------------------
# Attention: Embedding seed + Graph traversal
# ---------------------------------------------------------------------------

def attention(query, nodes, edges, collection, k=None, max_hop=None, bm25_index=None):
    """
    BDH-style attention: find relevant notes via ChromaDB KNN search (seed)
    + graph traversal (k-hop expansion). Returns active note set with scores.

    When integrate_and_fire=True (default), uses the IaF model for iterative
    accumulation + threshold firing. Otherwise falls back to single-pass k-hop.

    Uses ChromaDB for vector similarity (HNSW index, cosine space).
    Phase 3.1: Hybrid search — combines vector similarity with BM25 keyword score.
    Phase 3.3: Adaptive threshold — dynamic threshold from score distribution.
    Phase 4: Integrate-and-Fire — iterative accumulation with per-neuron τ.
    """
    # Dispatch to IaF if enabled
    if CONFIG.get('integrate_and_fire', False):
        return integrate_and_fire_attention(query, nodes, edges, collection, k, max_hop, bm25_index)

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
# Integrate-and-Fire attention model
# ---------------------------------------------------------------------------

def compute_tau(node_id, degree_map):
    """Compute per-neuron firing threshold: τ_j = base + k * log(1 + deg(j)).

    Hub nodes naturally require more accumulated input to fire, replacing
    the post-hoc hub_dampening factor with a structural mechanism.
    """
    base = CONFIG.get('iaf_tau_base', 0.15)
    k = CONFIG.get('iaf_tau_k', 0.075)
    deg = degree_map.get(node_id, 0)
    return base + k * math.log1p(deg)


def _build_reverse_edges(edges, nodes):
    """Build reverse edge index: for each node, list of (source_id, weight=1.0).

    This lets us efficiently compute which nodes can send input to a target.
    """
    reverse = defaultdict(list)
    for src_id, out_edges in edges.items():
        for edge in out_edges:
            target_id = _resolve_target(edge['target'], nodes)
            if target_id:
                reverse[target_id].append(src_id)
    return reverse


def _compute_degree(edges, nodes):
    """Compute degree (outgoing + incoming) for each node."""
    degree = defaultdict(int)
    for src, links in edges.items():
        degree[src] += len(links)
        for link in links:
            target = _resolve_target(link['target'], nodes)
            if target:
                degree[target] += 1
    return degree


def integrate_and_fire_attention(query, nodes, edges, collection, k=None, max_hop=None, bm25_index=None):
    """Integrate-and-Fire attention: iterative accumulation + threshold firing.

    Implements the simplified LIF model:
        A_j^{(t+1)} = max(0, Σ A_i^{(t)} · W_ij - τ_j)

    where τ_j = τ_base + τ_k * log(1 + degree(j)).

    Unlike the single-pass k-hop traversal, nodes accumulate input from
    multiple fired neighbors across timesteps. Firing is all-or-nothing:
    once a node crosses its threshold, it sends its full activation to
    downstream neighbors.

    Args:
        query: search query string
        nodes: dict of node_id → node dict
        edges: dict of node_id → list of edge dicts
        collection: ChromaDB collection
        k: seed count (default from config)
        max_hop: max integration steps (default from config)
        bm25_index: optional BM25Index for hybrid search
    Returns:
        dict of node_id → float (accumulated activation of fired nodes)
    """
    if k is None:
        k = CONFIG['seed_count']
    if max_hop is None:
        max_hop = CONFIG.get('iaf_max_steps', 5)

    tau_base = CONFIG.get('iaf_tau_base', 0.15)
    tau_k = CONFIG.get('iaf_tau_k', 0.075)
    max_steps = CONFIG.get('iaf_max_steps', 5)
    conv_threshold = CONFIG.get('iaf_convergence_threshold', 1e-4)

    degree = _compute_degree(edges, nodes)
    reverse = _build_reverse_edges(edges, nodes)

    # Step 1: Embedding seed — hybrid search scores (same as single-pass)
    query_emb = get_embeddings([query])[0]
    if not query_emb:
        return {}

    overfetch = min(k * 10, collection.count()) if collection.count() > 0 else k * 5
    results = collection.query(
        query_embeddings=[query_emb],
        n_results=overfetch,
        include=['metadatas', 'distances'],
    )

    raw_vector_scores = {}
    if results['ids'] and results['ids'][0]:
        for i, note_id in enumerate(results['ids'][0]):
            dist = results['distances'][0][i]
            sim = max(0.0, 1.0 - dist)
            raw_vector_scores[note_id] = sim

    hybrid_enabled = CONFIG.get('hybrid_search', False) and bm25_index is not None

    if hybrid_enabled:
        candidate_ids = list(raw_vector_scores.keys())
        seed_scores = {}
        for nid in candidate_ids:
            combined = hybrid_score(nid, raw_vector_scores, bm25_index, query)
            seed_scores[nid] = combined
    else:
        seed_scores = dict(raw_vector_scores)

    # Adaptive threshold on seed scores
    if CONFIG.get('adaptive_threshold', False) and len(seed_scores) >= 5:
        threshold = compute_adaptive_threshold(
            list(seed_scores.values()),
            floor=CONFIG.get('threshold_floor', tau_base),
        )
        seed_scores = {nid: s for nid, s in seed_scores.items() if s >= threshold}

    # Step 2: Integrate-and-Fire propagation
    # Seeds are initialized as "pre-fired" with their seed scores
    fired = {}  # node_id → accumulated activation (fired nodes)
    for nid, score in sorted(seed_scores.items(), key=lambda x: -x[1])[:k]:
        fired[nid] = score

    # Track all activation for convergence
    activation = dict(fired)

    for step in range(max_steps):
        # Collect input from fired neighbors
        incoming = defaultdict(float)
        for node_id in nodes:
            if node_id in fired:
                continue  # already fired, skip
            for src_id in reverse.get(node_id, []):
                if src_id in fired:
                    # Weight from hybrid/embedding scores or default 0.3
                    w_ij = seed_scores.get(src_id, 0.3)
                    incoming[node_id] += fired[src_id] * w_ij

        if not incoming:
            break

        # Apply threshold: potential = accumulated - τ_j
        newly_fired = {}
        for node_id, total_input in incoming.items():
            tau_j = compute_tau(node_id, degree)
            potential = total_input - tau_j

            if potential > 0:
                # Cap activation to prevent runaway
                activation[node_id] = min(potential, 1.0)
                newly_fired[node_id] = activation[node_id]

        if not newly_fired:
            break

        # Check convergence: did total activation change meaningfully?
        prev_total = sum(fired.values())
        fired.update(newly_fired)
        new_total = sum(fired.values())
        if abs(new_total - prev_total) < conv_threshold:
            break

        logger.info(
            f"IaF step {step + 1}: {len(newly_fired)} new fired, "
            f"total={len(fired)}/{len(nodes)}, activation={new_total:.3f}"
        )

    return fired


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