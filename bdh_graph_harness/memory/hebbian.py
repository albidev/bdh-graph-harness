"""Hebbian update — reinforce links between co-activated notes."""

from datetime import datetime

from bdh_graph_harness.config import CONFIG
from bdh_graph_harness.memory.quality import prune_dormant, try_reactivate


def hebbian_update(active_notes, state, nodes=None, source=None):
    """
    Hebbian update: reinforce links between co-activated notes.
    'Neurons that fire together, wire together.'

    Only creates synapses between notes that BOTH score above
    hebbian_min_score (default 0.15). Prevents spurious connections
    between weakly-activated peripheral nodes.

    Every ``quality_prune_interval`` queries (default 50), runs
    ``prune_dormant`` to re-evaluate node quality and mark weak nodes
    as dormant.

    Dormant nodes with a strong re-activation are automatically woken up.

    Args:
        source: When ``"assistant_response"``, dampens frequency increment
                to 0.3 (instead of 1.0) to prevent echo-loop reinforcement
                where Hermes repeats BDH context back into the graph.

    Returns (state, updated_keys, pruned_count) — updated_keys is the
    set of synapse keys that were created or reinforced in this call;
    pruned_count is the number of newly dormant nodes (0 if no prune
    happened).
    """
    min_score = CONFIG.get('hebbian_min_score', 0.15)

    # Dampening for assistant responses and semantic sleep: both are
    # derived/secondary signals and must not reinforce the graph like a direct
    # user-driven query.
    if source == "assistant_response":
        freq_increment = 0.3
    elif source == "nightly_semantic_consolidation":
        freq_increment = CONFIG.get('semantic_consolidation_frequency_increment', 0.3)
    else:
        freq_increment = 1.0

    # Filter to only notes above threshold
    strong = {nid: s for nid, s in active_notes.items() if s >= min_score}

    note_ids = sorted(strong.keys())
    now = datetime.now().isoformat()
    updated_keys = set()

    for i, a in enumerate(note_ids):
        for b in note_ids[i + 1:]:
            key = f"{a}|{b}"
            updated_keys.add(key)
            if key not in state['synapses']:
                state['synapses'][key] = {
                    'weight': 0.0,
                    'frequency': 0,
                    'last_coactivated': None,
                    'created': now,
                }

            syn = state['synapses'][key]
            syn['frequency'] += freq_increment
            syn['last_coactivated'] = now

            # Weight = alpha * normalized_freq + beta * recency
            # Recency: 1.0 if just activated, decays over time
            syn['weight'] = CONFIG['alpha'] * min(syn['frequency'] / 10.0, 1.0) + CONFIG['beta'] * 1.0

    # Decay unused synapses only during ordinary waking queries. Semantic sleep
    # must not globally weaken unrelated memory: structural consolidation owns
    # downscaling and pruning for the whole graph.
    if source != "nightly_semantic_consolidation":
        for key, syn in list(state['synapses'].items()):
            if key.split('|')[0] not in active_notes and key.split('|')[1] not in active_notes:
                syn['weight'] *= CONFIG['decay']
                if syn['weight'] < 0.01:
                    del state['synapses'][key]

    # Try to re-activate dormant nodes with strong activation
    reactivated = 0
    for nid, score in active_notes.items():
        if score >= min_score and try_reactivate(nid, score, state):
            reactivated += 1

    # Periodic quality pruning
    pruned_count = 0
    state['queries'] += 1
    prune_interval = CONFIG.get('quality_prune_interval', 50)
    if nodes and state['queries'] % prune_interval == 0:
        old_dormant = set(state.get('dormant_nodes', []))
        state = prune_dormant(state, nodes)
        new_dormant = set(state.get('dormant_nodes', []))
        pruned_count = len(new_dormant - old_dormant)

    return state, updated_keys, pruned_count