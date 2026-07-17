from __future__ import annotations

import json

from bdh_graph_harness.memory.state_store import (
    load_state,
    reconcile_state_to_nodes,
    save_state,
)


def _synapse(weight=0.8):
    return {'weight': weight, 'frequency': 1, 'last_coactivated': '2026-07-17T00:00:00'}


def test_reconcile_state_drops_removed_node_references():
    state = {
        'synapses': {'a|b': _synapse(), 'a|deleted': _synapse()},
        'node_quality': {'a': {'score': 0.8}, 'deleted': {'score': 0.8}},
        'dormant_nodes': ['deleted'],
        'phantom_links': [
            {'source': 'a', 'target': 'b', 'similarity': 0.8},
            {'source': 'a', 'target': 'deleted', 'similarity': 0.9},
        ],
    }

    reconcile_state_to_nodes(state, {'a': {}, 'b': {}, 'new': {}})

    assert set(state['synapses']) == {'a|b'}
    assert set(state['node_quality']) == {'a', 'b', 'new'}
    assert state['dormant_nodes'] == ['new']
    assert state['phantom_links'] == [{'source': 'a', 'target': 'b', 'similarity': 0.8}]


def test_save_state_valid_node_ids_does_not_resurrect_dead_synapses(tmp_path):
    (tmp_path / '.bdh-state.json').write_text(json.dumps({
        'synapses': {'a|deleted': _synapse(), 'a|b': _synapse()},
        'queries': 2,
    }), encoding='utf-8')

    save_state(
        str(tmp_path),
        {'synapses': {'a|b': _synapse(0.9)}, 'queries': 3},
        valid_node_ids={'a', 'b'},
    )

    persisted = load_state(str(tmp_path))
    assert set(persisted['synapses']) == {'a|b'}
    assert persisted['queries'] == 3
