"""Hebbian update — reinforce links between co-activated notes."""

from datetime import datetime

from bdh_graph_harness.config import CONFIG


def hebbian_update(active_notes, state):
    """
    Hebbian update: reinforce links between co-activated notes.
    'Neurons that fire together, wire together.'
    """
    note_ids = sorted(active_notes.keys())
    now = datetime.now().isoformat()

    for i, a in enumerate(note_ids):
        for b in note_ids[i + 1:]:
            key = f"{a}|{b}"
            if key not in state['synapses']:
                state['synapses'][key] = {
                    'weight': 0.0,
                    'frequency': 0,
                    'last_coactivated': None,
                    'created': now,
                }

            syn = state['synapses'][key]
            syn['frequency'] += 1
            syn['last_coactivated'] = now

            # Weight = alpha * normalized_freq + beta * recency
            # Recency: 1.0 if just activated, decays over time
            syn['weight'] = CONFIG['alpha'] * min(syn['frequency'] / 10.0, 1.0) + CONFIG['beta'] * 1.0

    # Decay unused synapses
    for key, syn in list(state['synapses'].items()):
        if key.split('|')[0] not in active_notes and key.split('|')[1] not in active_notes:
            syn['weight'] *= CONFIG['decay']
            if syn['weight'] < 0.01:
                del state['synapses'][key]

    state['queries'] += 1
    return state