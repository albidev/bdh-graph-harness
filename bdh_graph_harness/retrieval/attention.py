"""
BDH Graph Harness — Attention module.

BDH-style attention: embedding seed (ChromaDB KNN) + graph traversal (k-hop expansion).
Includes adaptive threshold, hybrid search support, and Integrate-and-Fire model.
"""
import math
import statistics
from collections import defaultdict, deque
from datetime import datetime

from bdh_graph_harness.config import CONFIG, logger
from bdh_graph_harness.memory.hebbian import (
    encode_synapse_key,
    safe_decode_synapse_key,
)
from bdh_graph_harness.retrieval.embeddings import get_embeddings
from bdh_graph_harness.retrieval.bm25 import BM25Index
from bdh_graph_harness.retrieval.hybrid import hybrid_score
from bdh_graph_harness.graph.builder import _resolve_target


def _dynamic_query_relevance(query_embedding, collection, node_ids):
    """Cosine relevance of dynamic-edge targets to the current query.

    Historical Hebbian trust says an association persisted; it does not prove
    that the association belongs in this retrieval. This gate supplies that
    missing query-conditioned evidence without mutating learned state.
    """
    if not node_ids:
        return {}
    payload = collection.get(ids=list(node_ids), include=["embeddings"])
    scores = {}
    query_norm = math.sqrt(sum(value * value for value in query_embedding))
    for node_id, embedding in zip(payload.get("ids", []), payload.get("embeddings", [])):
        target_norm = math.sqrt(sum(value * value for value in embedding))
        scores[node_id] = (sum(a * b for a, b in zip(query_embedding, embedding)) / (query_norm * target_norm)
                           if query_norm and target_norm else 0.0)
    return scores


def _build_associative_context(seeds, adjacency, primary_ids):
    """Return a bounded, non-ranking Hebbian context lane from primary seeds."""
    max_items = CONFIG.get('hebbian_associative_context_max_items', 2)
    max_per_seed = CONFIG.get('hebbian_associative_context_max_per_seed', 1)
    candidates = []
    for seed_id, seed_score in seeds:
        selected = 0
        for target_id, weight, trust in adjacency.get(seed_id, []):
            if target_id in primary_ids:
                continue
            association_score = seed_score * CONFIG.get('hebbian_dynamic_hop_decay', 0.6) * (0.5 + weight) * CONFIG.get('hebbian_dynamic_gain', 1.0) * trust
            candidates.append({
                'id': target_id,
                'source_seed_id': seed_id,
                'matched_by': 'hebbian_edge',
                'association_score': association_score,
                'hebbian_edge_weight': weight,
                'hebbian_edge_trust': trust,
            })
            selected += 1
            if selected >= max_per_seed:
                break
    candidates.sort(key=lambda item: item['association_score'], reverse=True)
    seen = set()
    return [item for item in candidates if not (item['id'] in seen or seen.add(item['id']))][:max_items]


def _resolve_hebbian_node_id(node_id, valid_node_ids):
    """Map persisted federated IDs to the canonical IDs of this graph view."""
    if node_id in valid_node_ids:
        return node_id
    if not node_id.startswith("vault:"):
        return None
    relative = node_id.removeprefix("vault:")
    for candidate in (relative, relative.removesuffix(".md")):
        if candidate in valid_node_ids:
            return candidate
    return None


def _hebbian_dynamic_trust(synapse, now=None):
    """Return a bounded confidence factor for a learned dynamic edge.

    Weight expresses association strength. Trust expresses whether that strength
    has repeated, consolidated, and remained current enough to traverse.
    """
    frequency = max(0.0, float(synapse.get("frequency", 0.0)))
    saturation = max(0.01, float(CONFIG.get("hebbian_dynamic_frequency_saturation", 2.0)))
    frequency_factor = min(1.0, frequency / saturation)

    cycles = max(0, int(synapse.get("consolidation_candidate_cycles", 0)))
    consolidation_factor = 1.0 if cycles else float(
        CONFIG.get("hebbian_dynamic_unconsolidated_trust", 0.6)
    )

    recency_factor = 1.0
    timestamp = synapse.get("last_coactivated")
    if timestamp:
        try:
            reference = now or datetime.now()
            last_coactivated = datetime.fromisoformat(timestamp)
            if last_coactivated.tzinfo is not None and reference.tzinfo is None:
                reference = reference.replace(tzinfo=last_coactivated.tzinfo)
            elif last_coactivated.tzinfo is None and reference.tzinfo is not None:
                last_coactivated = last_coactivated.replace(tzinfo=reference.tzinfo)
            age_days = max(0.0, (reference - last_coactivated).total_seconds() / 86400)
            horizon = max(1.0, float(CONFIG.get("hebbian_dynamic_recency_days", 30.0)))
            recency_floor = float(CONFIG.get("hebbian_dynamic_recency_floor", 0.4))
            recency_factor = max(recency_floor, 1.0 - age_days / horizon * (1.0 - recency_floor))
        except (TypeError, ValueError):
            recency_factor = float(CONFIG.get("hebbian_dynamic_recency_floor", 0.4))

    floor = float(CONFIG.get("hebbian_dynamic_trust_floor", 0.25))
    return min(1.0, max(floor, frequency_factor * consolidation_factor * recency_factor))


def _synapse_weight_for_pair(synapses, note_a, note_b):
    """Read a pair weight from v2 state, falling back to decodable legacy keys."""
    v2_synapse = synapses.get(encode_synapse_key(note_a, note_b))
    if v2_synapse is not None:
        return v2_synapse.get("weight", 0.0)

    for key, synapse in synapses.items():
        pair = safe_decode_synapse_key(key)
        if pair is None:
            continue
        left, right = pair
        if (left == note_a and right == note_b) or (left == note_b and right == note_a):
            return synapse.get("weight", 0.0)
    return 0.0


# ---------------------------------------------------------------------------
# Adaptive Threshold (Phase 3.3)
# ---------------------------------------------------------------------------

def compute_adaptive_threshold(scores, floor=0.05, min_activations=3):
    """Compute a dynamic threshold from score distribution.

    Implementation: median + 0.3*std with a low floor. Also guarantees at
    least min_activations notes are kept by relaxing the threshold if needed.

    This matches the code in this module. README / IMPLEMENTATION_PLAN / other
    docs that state ``max(Q75, mean+1std, 0.15)`` are stale; code is truth.

    Args:
        scores: list of float scores from attention
        floor: minimum threshold (never go below this)
        min_activations: guaranteed minimum number of notes to activate
    Returns:
        float threshold value
    """
    if not scores:
        return floor

    sorted_scores = sorted(scores, reverse=True)
    # Guarantee min_activations before any statistical filtering.
    if len(sorted_scores) <= min_activations:
        return floor

    # Statistical threshold: median + 0.3*std
    median = statistics.median(scores)
    stdev = statistics.stdev(scores) if len(scores) > 1 else 0.0
    threshold = max(median + 0.3 * stdev, floor)

    # If threshold is so strict it drops below min_activations, relax it,
    # but never below floor.
    kept = sum(1 for s in scores if s >= threshold)
    if kept < min_activations:
        relaxed = sorted_scores[min_activations - 1]
        threshold = max(relaxed, floor)
        logger.info(f"Adaptive threshold relaxed to {threshold:.3f} to keep {min_activations} activations")
    else:
        logger.info(f"Adaptive threshold: median+0.3std={median + 0.3 * stdev:.3f}, floor={floor}, "
                    f"min_activations={min_activations} → {threshold:.3f}")
    return threshold


# ---------------------------------------------------------------------------
# Attention: Embedding seed + Graph traversal
# ---------------------------------------------------------------------------

def _compute_hebbian_boost(candidate_id, hebbian_state, recently_active):
    """Compute Hebbian boost for a candidate note.

    Looks at the candidate's Hebbian synapses towards recently active notes.
    Instead of summing ALL synapse weights (which dilutes the signal), it takes
    only the top-N strongest synapses — this ensures the boost is selective:
    a candidate needs a *strong* connection to *specific* recent notes, not
    weak connections to many.

    Args:
        candidate_id: The note ID being evaluated as a seed.
        hebbian_state: The Hebbian state dict with 'synapses'.
        recently_active: Set of note IDs activated in recent queries.

    Config params:
        hebbian_boost_max: Maximum boost factor (0.5 = up to +50%).
        hebbian_boost_top_n: Only consider the N strongest synapses.
        hebbian_boost_weight_factor: Multiplier applied to summed weight.

    Returns:
        Float boost factor in [0, hebbian_boost_max].
    """
    if not hebbian_state or not recently_active:
        return 0.0

    synapses = hebbian_state.get('synapses', {})
    if not synapses:
        return 0.0

    top_n = CONFIG.get('hebbian_boost_top_n', 3)
    max_boost = CONFIG.get('hebbian_boost_max', 0.5)
    weight_factor = CONFIG.get('hebbian_boost_weight_factor', 0.3)

    # Persisted state can use federated ``vault:...md`` IDs while attention
    # operates on canonical graph IDs. Resolve against this candidate/context.
    valid_ids = set(recently_active) | {candidate_id}

    # Collect synapse weights from candidate to recently active notes
    weights = []
    for key, syn in synapses.items():
        pair = safe_decode_synapse_key(key)
        if pair is None:
            continue
        left, right = pair
        left = _resolve_hebbian_node_id(left, valid_ids)
        right = _resolve_hebbian_node_id(right, valid_ids)
        if left is None or right is None:
            continue
        if left == candidate_id and right in recently_active:
            weights.append(syn.get('weight', 0.0))
        elif right == candidate_id and left in recently_active:
            weights.append(syn.get('weight', 0.0))

    if not weights:
        return 0.0

    # Take only the top-N strongest synapses (selectivity)
    weights.sort(reverse=True)
    top_weights = weights[:top_n]

    total_weight = sum(top_weights)
    return min(total_weight * weight_factor, max_boost)


def _get_recently_active_notes(hebbian_state, valid_node_ids=None):
    """Get notes activated in recent queries from Hebbian state.

    Uses last_coactivated timestamps on synapses to determine which notes
    have been active recently. The time window is configurable.

    Also filters out synapses to notes that no longer exist in the vault
    (dead synapses from deleted notes).

    Args:
        hebbian_state: The Hebbian state dict.

    Config params:
        hebbian_boost_window_minutes: How far back to look for recent activity.
        hebbian_boost_min_weight: Minimum synapse weight to consider "active".

    Returns:
        Set of note IDs that were recently active.
    """
    if not hebbian_state:
        return set()

    synapses = hebbian_state.get('synapses', {})
    if not synapses:
        return set()

    from datetime import datetime, timedelta
    now = datetime.now()
    window_minutes = CONFIG.get('hebbian_boost_window_minutes', 10)
    recent_cutoff = now - timedelta(minutes=window_minutes)
    min_weight = CONFIG.get('hebbian_boost_min_weight', 0.15)

    # Get the set of valid note IDs (to filter dead synapses)

    recent_notes = set()
    for key, syn in synapses.items():
        weight = syn.get('weight', 0.0)
        if weight < min_weight:
            continue
        last_co = syn.get('last_coactivated')
        if last_co:
            try:
                co_time = datetime.fromisoformat(last_co)
                if co_time > recent_cutoff:
                    pair = safe_decode_synapse_key(key)
                    if pair is None:
                        continue
                    parts = list(pair)
                    # Filter dead synapses if we have the valid node set
                    if valid_node_ids is not None:
                        parts = [
                            resolved for p in parts
                            if (resolved := _resolve_hebbian_node_id(p, valid_node_ids)) is not None
                        ]
                        if not parts:
                            continue
                    recent_notes.update(parts)
            except (ValueError, TypeError):
                pass

    return recent_notes


def _build_hebbian_adjacency(hebbian_state, valid_node_ids):
    """Build a bounded adjacency index from strong learned synapses.

    Unlike ``_compute_hebbian_boost``, this makes an association traversable
    even when its target did not survive the initial vector/BM25 candidate
    pool. The index is scoped to one attention call and only retains the
    strongest configured neighbors per source node.
    """
    if not hebbian_state or not CONFIG.get("hebbian_dynamic_edges_enabled", True):
        return {}

    min_weight = CONFIG.get("hebbian_dynamic_min_weight", 0.15)
    top_n = CONFIG.get("hebbian_dynamic_top_n", 3)
    adjacency = defaultdict(list)
    for key, synapse in hebbian_state.get("synapses", {}).items():
        weight = float(synapse.get("weight", 0.0))
        if weight < min_weight:
            continue
        pair = safe_decode_synapse_key(key)
        if pair is None:
            continue
        left, right = pair
        left = _resolve_hebbian_node_id(left, valid_node_ids)
        right = _resolve_hebbian_node_id(right, valid_node_ids)
        if left is None or right is None:
            continue
        trust = _hebbian_dynamic_trust(synapse)
        adjacency[left].append((right, weight, trust))
        adjacency[right].append((left, weight, trust))

    return {
        node_id: sorted(neighbors, key=lambda item: -(item[1] * item[2]))[:top_n]
        for node_id, neighbors in adjacency.items()
    }


def attention(query, nodes, edges, collection, k=None, max_hop=None, bm25_index=None,
              hebbian_state=None, routing_meta=None):
    """
    BDH-style attention: find relevant notes via ChromaDB KNN search (seed)
    + graph traversal (k-hop expansion). Returns active note set with scores.

    When experimental_integrate_fire=True, uses the IaF model for iterative
    accumulation + threshold firing. Otherwise falls back to single-pass k-hop.

    Uses ChromaDB for vector similarity (HNSW index, cosine space).
    Phase 3.1: Hybrid search — combines vector similarity with BM25 keyword score.
    Phase 3.3: Adaptive threshold — dynamic threshold from score distribution.
    Phase 4: Integrate-and-Fire — iterative accumulation with per-neuron τ.
    Phase 5: Hebbian-aware seed ranking — boosts seeds that have strong Hebbian
             synapses with notes activated in recent queries.

    Args:
        hebbian_state: Optional Hebbian state dict. When provided, seed selection
                      is boosted by Hebbian synapse weights to recently active notes.
    """
    # Dispatch to IaF if enabled
    if CONFIG.get('experimental_integrate_fire', False):
        logger.warning("⚠️  Integrate-and-Fire attention ENABLED (experimental)")
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

    # Hybrid search: combine vector + BM25 (with proper batch normalization)
    hybrid_enabled = CONFIG.get('hybrid_search', False) and bm25_index is not None
    bm25_scores = {}

    if hybrid_enabled and bm25_index is not None:
        vector_candidate_ids = list(raw_vector_scores.keys())
        bm25_candidate_ids = [
            note_id for note_id, _score in bm25_index.search(query, top_k=overfetch)
        ]
        candidate_ids = list(dict.fromkeys(vector_candidate_ids + bm25_candidate_ids))
        # Compute BM25 scores ONCE for the union of vector and lexical candidates.
        bm25_scores = bm25_index.score_batch(query, candidate_ids)

        fusion = CONFIG.get('hybrid_fusion', 'weighted')
        if fusion == 'rrf':
            from bdh_graph_harness.retrieval.hybrid import reciprocal_rank_fusion
            scores = reciprocal_rank_fusion(raw_vector_scores, bm25_scores, k=CONFIG.get('rrf_k', 60))
        else:
            alpha = CONFIG.get('hybrid_alpha', 0.7)
            beta = CONFIG.get('hybrid_beta', 0.3)
            scores = {}
            for nid in candidate_ids:
                vec_s = raw_vector_scores.get(nid, 0.0)
                bm_s = bm25_scores.get(nid, 0.0)
                combined = alpha * vec_s + beta * bm_s
                # Hub dampening
                if CONFIG['hub_dampening'] and degree.get(nid, 0) > CONFIG['hub_degree_threshold']:
                    dampen = 1.0 / (1.0 + 0.15 * (degree[nid] - CONFIG['hub_degree_threshold']))
                    combined *= dampen
                scores[nid] = combined
    else:
        scores = {}
        for note_id, sim in raw_vector_scores.items():
            if CONFIG['hub_dampening'] and degree.get(note_id, 0) > CONFIG['hub_degree_threshold']:
                dampen = 1.0 / (1.0 + 0.15 * (degree[note_id] - CONFIG['hub_degree_threshold']))
                sim *= dampen
            scores[note_id] = sim

    # Update routing metadata with chosen fusion and component scores.
    if routing_meta is not None:
        ranked = sorted(scores.values(), reverse=True)
        bm25_ranked = sorted(bm25_scores.items(), key=lambda item: -item[1]) if hybrid_enabled else []
        bm25_top_id = bm25_ranked[0][0] if bm25_ranked else None
        bm25_query_terms = (
            bm25_index._tokenize(query)
            if hybrid_enabled and bm25_index is not None else []
        )
        bm25_matched_terms = (
            bm25_index.matched_terms(query, bm25_top_id)
            if hybrid_enabled and bm25_index is not None and bm25_top_id is not None else []
        )
        routing_meta.update({
            "vector_top_score": max(raw_vector_scores.values(), default=0.0),
            "bm25_top_score": max(bm25_scores.values(), default=0.0),
            "bm25_query_term_count": len(set(bm25_query_terms)),
            "bm25_matched_term_count": len(bm25_matched_terms),
            "bm25_matched_terms": bm25_matched_terms,
            "hybrid_top_score": ranked[0] if ranked else 0.0,
            "hybrid_second_score": ranked[1] if len(ranked) > 1 else 0.0,
            "hybrid_margin": (ranked[0] - ranked[1]) if len(ranked) > 1 else ranked[0] if ranked else 0.0,
            "hybrid_enabled": hybrid_enabled,
            "hybrid_fusion": CONFIG.get('hybrid_fusion', 'weighted') if hybrid_enabled else None,
        })

    # Keep retrieval-only scores separate from Hebbian and hop propagation scores.
    retrieval_scores = dict(scores)

    # Phase 5: Hebbian-aware seed ranking
    # Boost candidates that have strong Hebbian synapses with recently active notes.
    # This makes the graph "remember" what you're working on and prefer seeds
    # in the same context, even if their pure cosine similarity is slightly lower.
    if hebbian_state and CONFIG.get('hebbian_seed_boost', True):
        valid_node_ids = set(nodes.keys())
        recently_active = _get_recently_active_notes(
            hebbian_state, valid_node_ids=valid_node_ids
        )
        if recently_active:
            logger.info(f"Phase 5: {len(recently_active)} recently active notes, boosting seeds")
            boosted = {}
            for nid, score in scores.items():
                boost = _compute_hebbian_boost(nid, hebbian_state, recently_active)
                if boost > 0:
                    logger.debug(f"  boost {nid}: +{boost:.3f}")
                boosted[nid] = score * (1.0 + boost)
            scores = boosted

    # Adaptive threshold (Phase 3.3)
    if CONFIG.get('adaptive_threshold', False) and len(scores) >= 5:
        threshold = compute_adaptive_threshold(
            list(scores.values()),
            floor=CONFIG.get('threshold_floor', 0.05),
        )
        scores = {nid: s for nid, s in scores.items() if s >= threshold}

    # Top-k seeds — always keep at least k regardless of threshold
    seeds = sorted(scores.items(), key=lambda x: -x[1])[:k]
    seed_ids = {nid for nid, _score in seeds}
    activation_details = {}
    for note_id, score in seeds:
        retrieval_score = retrieval_scores.get(note_id, 0.0)
        boost = max(0.0, score / retrieval_score - 1.0) if retrieval_score > 0 else 0.0
        activation_details[note_id] = {
            'role': 'seed',
            'hop': 0,
            'parent_id': None,
            'vector_score': round(raw_vector_scores.get(note_id, 0.0), 4),
            'bm25_score': round(bm25_scores.get(note_id, 0.0), 4),
            'hybrid_score': round(retrieval_score, 4),
            'hebbian_boost': round(boost, 4),
            'final_score': round(score, 4),
        }

    # Step 2: Graph traversal — expand from seeds via wikilinks
    active = dict(seeds)
    learned_adjacency = _build_hebbian_adjacency(hebbian_state, set(nodes))
    associative_context_enabled = CONFIG.get('hebbian_associative_context_enabled', False)
    hebbian_adjacency = {} if associative_context_enabled else learned_adjacency
    dynamic_targets = {
        target_id
        for dynamic_neighbors in hebbian_adjacency.values()
        for target_id, _weight, _trust in dynamic_neighbors
    }
    dynamic_relevance = _dynamic_query_relevance(query_emb, collection, dynamic_targets)
    dynamic_relevance_floor = CONFIG.get('hebbian_dynamic_query_relevance_floor', 0.0)
    queue = deque()
    for note_id, score in seeds:
        queue.append((note_id, score, 0, None))

    # Unified threshold policy for propagation
    traversal_threshold = CONFIG['active_threshold']
    if CONFIG.get('adaptive_threshold', False) and len(scores) >= 5:
        traversal_threshold = compute_adaptive_threshold(
            list(scores.values()),
            floor=CONFIG.get('threshold_floor', 0.05),
        )

    while queue:
        current_id, score, hop, parent_id = queue.popleft()
        if hop >= max_hop:
            continue

        # Static wikilinks and learned co-activations share the same bounded
        # traversal queue, but retain their origin for scoring and provenance.
        neighbors = {}
        for edge in edges.get(current_id, []):
            target = edge['target']
            target_id = _resolve_target(target, nodes)
            if target_id is None:
                continue
            neighbors[target_id] = {
                'matched_by': 'static_edge',
                'hebbian_edge_weight': 0.0,
                'hebbian_edge_trust': 0.0,

                'priority': scores.get(target_id, 0.1),
            }

        for target_id, weight, trust in hebbian_adjacency.get(current_id, []):
            # Historical co-activation alone cannot displace declared graph
            # knowledge. A learned-only edge also needs query-local semantic
            # support before it enters this traversal.
            if dynamic_relevance.get(target_id, 0.0) < dynamic_relevance_floor:
                continue
            existing = neighbors.get(target_id)
            if existing:
                existing['matched_by'] = 'static_and_hebbian_edge'
                existing['hebbian_edge_weight'] = max(existing['hebbian_edge_weight'], weight)
                existing['hebbian_edge_trust'] = max(existing['hebbian_edge_trust'], trust)

                existing['priority'] = max(existing['priority'], weight)
            else:
                neighbors[target_id] = {
                    'matched_by': 'hebbian_edge',
                    'hebbian_edge_weight': weight,
                    'hebbian_edge_trust': trust,

                    'priority': max(scores.get(target_id, 0.0), weight * trust),
                }

        # Cap static + dynamic expansion together, preserving strong learned
        # associations even if vector/BM25 did not include their target.
        neighbors = sorted(neighbors.items(), key=lambda item: -item[1]['priority'])
        neighbors = neighbors[:CONFIG['max_neighbors_per_hop']]

        for target_id, neighbor in neighbors:
            dynamic_weight = neighbor['hebbian_edge_weight']
            dynamic_trust = neighbor['hebbian_edge_trust']

            # Dynamic edges get an independently tunable decay. Static and
            # mixed edges preserve the existing graph traversal behavior.
            decay = (
                CONFIG.get('hebbian_dynamic_hop_decay', CONFIG.get('hop_decay', 0.5))
                if neighbor['matched_by'] == 'hebbian_edge'
                else CONFIG.get('hop_decay', 0.5)
            )
            new_score = score * decay
            if neighbor['matched_by'] == 'hebbian_edge':
                new_score *= ((0.5 + dynamic_weight) * CONFIG.get('hebbian_dynamic_gain', 1.0)
                              * dynamic_trust)
            elif dynamic_weight > 0:
                new_score *= 1.0 + dynamic_weight * CONFIG.get('hebbian_dynamic_gain', 1.0)

            # Hub dampening for the target
            if CONFIG['hub_dampening'] and degree.get(target_id, 0) > CONFIG['hub_degree_threshold']:
                dampen = 1.0 / (1.0 + 0.15 * (degree[target_id] - CONFIG['hub_degree_threshold']))
                new_score *= dampen

            # Hebbian gain: learned synaptic weight between current_id and target_id
            # boosts propagation when the synapse exists. hebbian_gain=0 reproduces old behavior.
            hebbian_gain = CONFIG.get('hebbian_gain', 0.0)
            if hebbian_gain > 0 and hebbian_state and neighbor['matched_by'] == 'static_edge':
                synapses = hebbian_state.get('synapses', {})
                syn_weight = _synapse_weight_for_pair(synapses, current_id, target_id)
                if syn_weight > 0:
                    new_score *= (1.0 + hebbian_gain * syn_weight)

            if target_id in active:
                if new_score > active[target_id]:
                    active[target_id] = new_score
                    if target_id not in seed_ids:
                        activation_details[target_id].update({
                            'hop': hop + 1,
                            'parent_id': current_id,
                            'matched_by': neighbor['matched_by'],
                            'hebbian_edge_weight': round(dynamic_weight, 4),
                            'hebbian_edge_trust': round(dynamic_trust, 4),
                            'final_score': round(new_score, 4),
                        })
            elif new_score > traversal_threshold:
                active[target_id] = new_score
                activation_details[target_id] = {
                    'role': 'hebbian_neighbor' if neighbor['matched_by'] == 'hebbian_edge' else 'graph_neighbor',
                    'hop': hop + 1,
                    'parent_id': current_id,
                    'matched_by': neighbor['matched_by'],
                    'hebbian_edge_weight': round(dynamic_weight, 4),
                    'hebbian_edge_trust': round(dynamic_trust, 4),
                    'vector_score': round(raw_vector_scores.get(target_id, 0.0), 4),
                    'bm25_score': round(bm25_scores.get(target_id, 0.0), 4),
                    'hybrid_score': round(retrieval_scores.get(target_id, 0.0), 4),
                    'hebbian_boost': 0.0,
                    'final_score': round(new_score, 4),
                }
                queue.append((target_id, new_score, hop + 1, current_id))

    if routing_meta is not None:
        if associative_context_enabled:
            routing_meta['associative_context'] = _build_associative_context(
                seeds=seeds,
                adjacency=learned_adjacency,
                primary_ids=set(active),
            )
        routing_meta['activation_details'] = [
            {'id': nid, **activation_details[nid]}
            for nid, _score in sorted(active.items(), key=lambda item: -item[1])
            if nid in activation_details
        ]

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
        # Compute BM25 scores ONCE for all candidates, normalized [0,1]
        bm25_scores = bm25_index.score_batch(query, candidate_ids)
        alpha = CONFIG.get('hybrid_alpha', 0.7)
        beta = CONFIG.get('hybrid_beta', 0.3)
        seed_scores = {}
        for nid in candidate_ids:
            vec_s = raw_vector_scores.get(nid, 0.0)
            bm_s = bm25_scores.get(nid, 0.0)
            seed_scores[nid] = alpha * vec_s + beta * bm_s
    else:
        seed_scores = dict(raw_vector_scores)

    # Adaptive threshold on seed scores
    if CONFIG.get('adaptive_threshold', False) and len(seed_scores) >= 5:
        threshold = compute_adaptive_threshold(
            list(seed_scores.values()),
            floor=CONFIG.get('threshold_floor', 0.05),
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