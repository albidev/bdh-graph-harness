"""Node quality scoring and dormant-node pruning.

Every node in the Hebbian graph accumulates a *quality score* based on
three independent signals:

1. **Strong-edge ratio** — fraction of edges with weight > 0.7.
2. **Mean weight** — average weight across all edges.
3. **Activation frequency** — how often the node is co-activated in queries.

The composite score is a weighted sum:

    Q = 0.40 * strong_ratio + 0.35 * mean_weight + 0.25 * freq_norm

where ``freq_norm = min(freq / max_freq, 1.0)``.

Nodes with Q below a configurable threshold are marked *dormant* and
excluded from visualization until a future query re-activates them
strongly (activation score > reactivation_threshold).

Dormancy is tracked in the state dict under ``node_quality``::

    "node_quality": {
        "wiki/concepts/bdh": {
            "score": 0.72,
            "dormant": false,
            "evaluated_at": "2026-07-05T12:00:00"
        },
        ...
    }
"""

from datetime import datetime

from bdh_graph_harness.config import CONFIG

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_QUALITY_THRESHOLD = 0.25     # below this → dormant
DEFAULT_REACTIVATION_SCORE = 0.50    # activation needed to re-awaken
DEFAULT_PRUNE_INTERVAL = 50          # re-evaluate every N queries
STRONG_EDGE_THRESHOLD = 0.7          # weight above this counts as "strong"


# ---------------------------------------------------------------------------
# Quality computation
# ---------------------------------------------------------------------------
def compute_node_quality(node_id: str, synapses: dict, max_freq: int) -> dict:
    """Compute the quality score for a single node.

    Parameters
    ----------
    node_id : str
        The node identifier (e.g. ``wiki/concepts/bdh``).
    synapses : dict
        Full ``state['synapses']`` mapping.
    max_freq : int
        Maximum frequency across all synapses (for normalisation).

    Returns
    -------
    dict
        ``{"score": float, "strong_ratio": float, "mean_weight": float,
          "frequency": int}``
    """
    edges = []
    total_freq = 0
    for key, syn in synapses.items():
        a, b = key.split('|')
        if a == node_id or b == node_id:
            edges.append(syn.get('weight', 0.0))
            total_freq += syn.get('frequency', 0)

    if not edges:
        return {
            'score': 0.0,
            'strong_ratio': 0.0,
            'mean_weight': 0.0,
            'frequency': 0,
        }

    strong_count = sum(1 for w in edges if w > STRONG_EDGE_THRESHOLD)
    strong_ratio = strong_count / len(edges)
    mean_weight = sum(edges) / len(edges)
    freq_norm = min(total_freq / max(max_freq, 1), 1.0)

    score = (
        0.40 * strong_ratio
        + 0.35 * mean_weight
        + 0.25 * freq_norm
    )

    return {
        'score': round(score, 4),
        'strong_ratio': round(strong_ratio, 4),
        'mean_weight': round(mean_weight, 4),
        'frequency': total_freq,
    }


def compute_all_qualities(synapses: dict, nodes: dict) -> dict:
    """Compute quality scores for every node in the graph.

    Returns
    -------
    dict
        ``{node_id: {"score": ..., "dormant": ..., "evaluated_at": ...}}``
    """
    # Find max frequency for normalisation
    max_freq = max(
        (syn.get('frequency', 0) for syn in synapses.values()),
        default=1,
    )

    now = datetime.now().isoformat()
    threshold = CONFIG.get('quality_threshold', DEFAULT_QUALITY_THRESHOLD)

    qualities = {}
    for node_id in nodes:
        q = compute_node_quality(node_id, synapses, max_freq)
        qualities[node_id] = {
            'score': q['score'],
            'strong_ratio': q['strong_ratio'],
            'mean_weight': q['mean_weight'],
            'frequency': q['frequency'],
            'dormant': q['score'] < threshold,
            'evaluated_at': now,
        }

    return qualities


# ---------------------------------------------------------------------------
# Pruning (dormancy)
# ---------------------------------------------------------------------------
def prune_dormant(state: dict, nodes: dict) -> dict:
    """Re-evaluate all node qualities and mark low-scoring nodes dormant.

    Modifies ``state['node_quality']`` in place and returns the updated state.
    Also injects a ``dormant_nodes`` list into the state for quick lookup.

    Returns
    -------
    dict
        The updated state (same reference as input).
    """
    synapses = state.get('synapses', {})

    qualities = compute_all_qualities(synapses, nodes)
    state['node_quality'] = qualities

    # Build quick-lookup list (stored as sorted list for JSON serialisation)
    state['dormant_nodes'] = sorted(
        nid for nid, q in qualities.items() if q['dormant']
    )

    return state


def try_reactivate(node_id: str, activation_score: float, state: dict) -> bool:
    """Check if a dormant node should be re-activated.

    If the node is dormant and the activation score exceeds the
    ``reactivation_score`` threshold, un-dorm it.

    Returns
    -------
    bool
        True if the node was re-activated.
    """
    nq = state.get('node_quality', {})
    if node_id not in nq:
        return False

    entry = nq[node_id]
    if not entry.get('dormant', False):
        return False

    reactivation = CONFIG.get(
        'quality_reactivation_score', DEFAULT_REACTIVATION_SCORE
    )
    if activation_score >= reactivation:
        entry['dormant'] = False
        entry['score'] = max(entry['score'], activation_score)
        entry['evaluated_at'] = datetime.now().isoformat()
        # Update lookup list
        dormant = state.get('dormant_nodes', [])
        if node_id in dormant:
            state['dormant_nodes'] = [n for n in dormant if n != node_id]
        return True

    return False


# ---------------------------------------------------------------------------
# Stats helper
# ---------------------------------------------------------------------------
def quality_stats(state: dict) -> dict:
    """Return summary statistics about node quality."""
    nq = state.get('node_quality', {})
    if not nq:
        return {'total': 0, 'dormant': 0, 'active': 0, 'avg_score': 0.0}

    scores = [q['score'] for q in nq.values()]
    dormant_count = sum(1 for q in nq.values() if q.get('dormant', False))

    return {
        'total': len(nq),
        'dormant': dormant_count,
        'active': len(nq) - dormant_count,
        'avg_score': round(sum(scores) / len(scores), 4) if scores else 0.0,
        'threshold': CONFIG.get('quality_threshold', DEFAULT_QUALITY_THRESHOLD),
    }
