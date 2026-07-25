"""Hebbian update — reinforce links between co-activated notes.

Key design choices
------------------
* Recency is computed from actual Δt, not a constant.
* Weight updates are incremental: decay persists between touches.
* Frequency contribution uses a non-saturating compression so popular
  synapses remain distinguishable.
* Co-activation strength weights the frequency increment by the product of
  the two nodes' activation scores.
* Topology is seeds → activated nodes (directed), not a clique.
* Pruning is deferred to the consolidation cycle; here synapses are only
  decayed and never deleted.
"""
from datetime import datetime, timedelta, timezone
from math import exp, log1p

from bdh_graph_harness.config import CONFIG
from bdh_graph_harness.memory.quality import prune_dormant, try_reactivate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ordered_key(a: str, b: str) -> str:
    """Canonical ordering for undirected synapse keys."""
    return f"{a}|{b}" if a < b else f"{b}|{a}"


def _recency_factor(last_coactivated: str | None, tau_hours: float) -> float:
    """Exponential recency: exp(-Δt / τ) with Δt in hours."""
    if not last_coactivated:
        return 0.0
    try:
        last = datetime.fromisoformat(last_coactivated)
        if last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)
        delta_hours = (datetime.now(timezone.utc) - last).total_seconds() / 3600.0
        return exp(-max(delta_hours, 0.0) / max(tau_hours, 1e-6))
    except (ValueError, TypeError):
        return 0.0


def _frequency_component(freq: float, freq_scale: float) -> float:
    """Non-saturating frequency compression in [0, 1]."""
    denom = log1p(freq_scale)
    return log1p(freq) / denom if denom > 0 else 0.0


def _compute_weight(
    syn: dict,
    alpha: float,
    beta: float,
    freq_scale: float,
    tau_recency_hours: float,
) -> float:
    """Recompute synapse weight from stored state."""
    freq = syn.get('frequency', 0.0)
    recency = _recency_factor(syn.get('last_coactivated'), tau_recency_hours)
    return alpha * _frequency_component(freq, freq_scale) + beta * recency


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def hebbian_update(active_notes, state, nodes=None, source=None):
    """Reinforce links from strong seeds to activated notes.

    Args:
        active_notes: dict {note_id: activation_score} in descending order.
        state: Hebbian state dict with 'synapses', 'queries', 'dormant_nodes'.
        nodes: optional full node map; used only by the periodic quality prune.
        source: 'assistant_response' or 'nightly_semantic_consolidation' dampen
                the update; anything else uses full strength.

    Returns:
        (state, updated_keys, pruned_count)
    """
    min_score = CONFIG.get('hebbian_min_score', 0.15)

    # Dampening for derived/secondary signals.
    if source == "assistant_response":
        freq_increment = 0.3
    elif source == "nightly_semantic_consolidation":
        freq_increment = CONFIG.get('semantic_consolidation_frequency_increment', 0.3)
    else:
        freq_increment = 1.0

    # Only notes above threshold participate.
    strong = {nid: s for nid, s in active_notes.items() if s >= min_score}
    if not strong:
        return state, set(), 0

    # Seeds: the top-k strongest notes. Use config seed_count, capped to available.
    seed_count = CONFIG.get('seed_count', 5)
    seeds = [nid for nid, _ in sorted(strong.items(), key=lambda x: -x[1])[:seed_count]]

    alpha = CONFIG.get('alpha', 0.7)
    beta = CONFIG.get('beta', 0.3)
    freq_scale = CONFIG.get('hebbian_frequency_scale', 10.0)
    tau_recency = CONFIG.get('tau_recency_hours', 24.0)

    now = datetime.now(timezone.utc).isoformat()
    updated_keys = set()

    # Directed seed → later seed edges. The key is stored canonical/undirected,
    # but we create each pair exactly once using the score product of the two ends.
    for i, seed_id in enumerate(seeds):
        seed_score = strong[seed_id]
        for target_id in seeds[i + 1:]:
            target_score = strong[target_id]
            key = _ordered_key(seed_id, target_id)
            updated_keys.add(key)

            if key not in state['synapses']:
                state['synapses'][key] = {
                    'weight': 0.0,
                    'frequency': 0.0,
                    'last_coactivated': None,
                    'created': now,
                }

            syn = state['synapses'][key]
            # Increment weighted by co-activation strength.
            increment = freq_increment * seed_score * target_score
            syn['frequency'] += increment
            syn['last_coactivated'] = now
            # Incremental: recompute from stored state so decay persists.
            syn['weight'] = _compute_weight(syn, alpha, beta, freq_scale, tau_recency)

    # Decay synapses that were not touched by this query.
    decay_rate = CONFIG.get('decay', 0.95)
    touched = updated_keys
    for key, syn in list(state['synapses'].items()):
        if key in touched:
            continue
        syn['weight'] *= decay_rate
        # Floor instead of delete; consolidation owns pruning.
        if syn['weight'] < 0:
            syn['weight'] = 0.0

    # Try to re-activate dormant nodes that received strong activation.
    reactivated = 0
    for nid, score in active_notes.items():
        if score >= min_score and try_reactivate(nid, score, state):
            reactivated += 1

    # Periodic quality pruning.
    pruned_count = 0
    state['queries'] += 1
    prune_interval = CONFIG.get('quality_prune_interval', 50)
    if nodes and state['queries'] % prune_interval == 0:
        old_dormant = set(state.get('dormant_nodes', []))
        state = prune_dormant(state, nodes)
        new_dormant = set(state.get('dormant_nodes', []))
        pruned_count = len(new_dormant - old_dormant)

    return state, updated_keys, pruned_count
