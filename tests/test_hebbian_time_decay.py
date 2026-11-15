"""Time-based forgetting and learning from travelled edges."""
from datetime import datetime, timedelta, timezone
from math import log1p

import pytest

import bdh_graph_harness.api.routes as bdh_routes
from bdh_graph_harness.config import config_overlay
from bdh_graph_harness.memory.consolidation import consolidate, effective_downscale_factor
from bdh_graph_harness.memory.hebbian import (
    current_weight,
    encode_synapse_key,
    hebbian_update,
    refresh_weights,
)
from bdh_graph_harness.retrieval.attention import _hebbian_dynamic_trust

from tests.test_multi_query_api import _capture_app, mock_app_mq_setup  # noqa: F401

TIME = {'hebbian_decay_mode': 'time', 'hebbian_frequency_half_life_days': 30.0}


def _state(**synapses):
    return {'synapses': dict(synapses), 'queries': 0}


def _ago(**delta):
    return (datetime.now(timezone.utc) - timedelta(**delta)).isoformat()


def _frequency_term(frequency):
    return 0.7 * log1p(frequency) / log1p(10.0)


# ---------------------------------------------------------------------------
# Forgetting follows time, not traffic
# ---------------------------------------------------------------------------

def test_unrelated_queries_do_not_erode_a_synapse_in_time_mode():
    key = encode_synapse_key('a', 'b')
    with config_overlay(TIME):
        state = _state()
        state, _, _ = hebbian_update({'a': 0.9, 'b': 0.8}, state)
        learned = state['synapses'][key]['weight']
        for _ in range(60):
            state, _, _ = hebbian_update({'x': 0.9, 'y': 0.8}, state)
        assert state['synapses'][key]['weight'] == pytest.approx(learned, abs=1e-3)


def test_the_same_traffic_erases_it_in_the_legacy_mode():
    key = encode_synapse_key('a', 'b')
    with config_overlay({'hebbian_decay_mode': 'query'}):
        state = _state()
        state, _, _ = hebbian_update({'a': 0.9, 'b': 0.8}, state)
        learned = state['synapses'][key]['weight']
        for _ in range(60):
            state, _, _ = hebbian_update({'x': 0.9, 'y': 0.8}, state)
        assert state['synapses'][key]['weight'] < learned * 0.1


def test_weight_fades_with_elapsed_time_on_a_half_life():
    recent = {'frequency': 2.0, 'last_coactivated': _ago(hours=1)}
    month_old = {'frequency': 2.0, 'last_coactivated': _ago(days=30)}
    two_months_old = {'frequency': 2.0, 'last_coactivated': _ago(days=60)}
    with config_overlay(TIME):
        assert current_weight(month_old) == pytest.approx(_frequency_term(1.0), abs=1e-3)
        assert current_weight(two_months_old) == pytest.approx(_frequency_term(0.5), abs=1e-3)
        assert current_weight(recent) > current_weight(month_old) > current_weight(two_months_old)


def test_reuse_after_a_long_gap_builds_on_what_is_left_not_on_the_old_count():
    key = encode_synapse_key('a', 'b')
    with config_overlay(TIME):
        state = _state(**{key: {'weight': 0.4, 'frequency': 2.0, 'last_coactivated': _ago(days=30)}})
        state, updated, _ = hebbian_update({'a': 1.0, 'b': 1.0}, state)
        assert key in updated
        assert state['synapses'][key]['frequency'] == pytest.approx(1.0 + 1.0, abs=1e-3)


def test_refresh_recomputes_stale_weights_and_spares_undated_synapses():
    dated, undated = encode_synapse_key('a', 'b'), encode_synapse_key('c', 'd')
    state = _state(**{
        dated: {'weight': 0.9, 'frequency': 1.0, 'last_coactivated': _ago(days=90)},
        undated: {'weight': 0.33, 'frequency': 1.0},
    })
    with config_overlay(TIME):
        assert refresh_weights(state) == 1
    assert state['synapses'][dated]['weight'] == pytest.approx(_frequency_term(0.125), abs=1e-3)
    assert state['synapses'][undated]['weight'] == 0.33


# ---------------------------------------------------------------------------
# Travelled edges
# ---------------------------------------------------------------------------

def test_a_travelled_edge_is_reinforced_alongside_the_seed_pair():
    state, updated, _ = hebbian_update(
        {'a': 1.0, 'b': 0.9}, _state(), traversed_edges=[('a', 'n', 0.5)],
    )
    pair, travelled = encode_synapse_key('a', 'b'), encode_synapse_key('a', 'n')
    assert updated == {pair, travelled}
    assert state['synapses'][travelled]['frequency'] == pytest.approx(0.5)
    assert state['synapses'][pair]['frequency'] == pytest.approx(0.9)


def test_travelled_edges_honour_the_source_dampening():
    state, _, _ = hebbian_update(
        {'a': 1.0, 'b': 0.9}, _state(), source='assistant_response',
        traversed_edges=[('a', 'n', 0.5)],
    )
    assert state['synapses'][encode_synapse_key('a', 'n')]['frequency'] == pytest.approx(0.15)


def test_degenerate_travelled_edges_are_ignored():
    state, updated, _ = hebbian_update(
        {'a': 1.0, 'b': 0.9}, _state(),
        traversed_edges=[('a', 'a', 0.5), ('a', 'n', 0.0), ('a', 'm', -1.0)],
    )
    assert updated == {encode_synapse_key('a', 'b')}
    assert len(state['synapses']) == 1


# ---------------------------------------------------------------------------
# Consolidation
# ---------------------------------------------------------------------------

def test_time_mode_owns_forgetting_so_consolidation_does_not_downscale():
    legacy = {'hebbian_state_file': '.bdh-state.json', 'consolidation_downscale_factor': 0.9}
    assert effective_downscale_factor(legacy) == 0.9
    assert effective_downscale_factor({**legacy, **TIME}) == 1.0


def test_consolidation_refreshes_weights_and_marks_survivors_in_time_mode():
    stale, fresh = encode_synapse_key('a', 'b'), encode_synapse_key('c', 'd')
    state = {
        'synapses': {
            stale: {'weight': 0.8, 'frequency': 4.0, 'last_coactivated': _ago(days=60)},
            fresh: {'weight': 0.1, 'frequency': 4.0, 'last_coactivated': _ago(hours=2)},
        },
        'queries': 0,
    }
    nodes = {nid: {'id': nid, 'title': nid} for nid in 'abcd'}
    cfg = {**TIME, 'phantom_links_enabled': False, 'hebbian_state_file': '.bdh-state.json'}
    result = consolidate(state, nodes, {}, config=cfg)
    assert result['downscale_factor'] == 1.0
    assert state['synapses'][stale]['weight'] == pytest.approx(_frequency_term(1.0), abs=1e-3)
    assert state['synapses'][fresh]['weight'] > 0.5
    assert all(syn['consolidated_cycles'] == 1 for syn in state['synapses'].values())


def test_dry_run_leaves_the_state_untouched_in_time_mode():
    key = encode_synapse_key('a', 'b')
    state = {'synapses': {key: {'weight': 0.8, 'frequency': 4.0, 'last_coactivated': _ago(days=60)}}, 'queries': 0}
    cfg = {**TIME, 'phantom_links_enabled': False}
    consolidate(state, {'a': {}, 'b': {}}, {}, config=cfg, dry_run=True)
    assert state['synapses'][key]['weight'] == 0.8
    assert 'consolidated_cycles' not in state['synapses'][key]


def test_an_association_that_survived_a_sleep_cycle_is_trusted_more():
    base = {'frequency': 4.0, 'last_coactivated': _ago(hours=1)}
    learned_today = _hebbian_dynamic_trust(dict(base))
    survived = _hebbian_dynamic_trust({**base, 'consolidated_cycles': 1})
    assert survived == pytest.approx(1.0, abs=0.01)
    assert learned_today == pytest.approx(0.6, abs=0.01)


# ---------------------------------------------------------------------------
# /api/query wiring
# ---------------------------------------------------------------------------

def _attention_with_neighbour(query, nodes, edges, collection, k=None, max_hop=None,
                              bm25_index=None, hebbian_state=None, routing_meta=None,
                              vault_settings=None):
    if routing_meta is not None:
        routing_meta.update({
            'vector_top_score': 0.9, 'bm25_top_score': 0.0, 'hybrid_top_score': 0.9,
            'hybrid_enabled': False, 'hybrid_fusion': None,
            'activation_details': [
                {'id': 'alpha', 'role': 'seed', 'hop': 0, 'parent_id': None, 'final_score': 0.9},
                {'id': 'beta', 'role': 'seed', 'hop': 0, 'parent_id': None, 'final_score': 0.8},
                {'id': 'gamma', 'role': 'graph_neighbor', 'hop': 1, 'parent_id': 'alpha',
                 'matched_by': 'static_edge', 'query_relevance': 0.5, 'final_score': 0.4},
            ],
        })
    return {'alpha': 0.9, 'beta': 0.8, 'gamma': 0.4}


async def _learned_pairs(monkeypatch, setup, enabled):
    from aiohttp.test_utils import TestClient, TestServer

    nodes, edges, collection, state, config, _ = setup
    config['hebbian_learn_traversed_edges'] = enabled
    monkeypatch.setattr(bdh_routes, 'attention', _attention_with_neighbour)
    app = _capture_app(monkeypatch, config, nodes, edges, collection, state)
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        resp = await client.post('/api/query', json={'query': 'anything', 'respond': False})
        assert resp.status == 200
        data = await resp.json()
    finally:
        await client.close()
    return {
        frozenset((update['note_a'], update['note_b'])): update['frequency']
        for update in data['hebbian_updates']
    }


@pytest.mark.asyncio
async def test_api_query_reinforces_the_edge_a_query_travelled(mock_app_mq_setup, monkeypatch):  # noqa: F811
    pairs = await _learned_pairs(monkeypatch, mock_app_mq_setup, enabled=True)
    assert set(pairs) == {frozenset(('alpha', 'beta')), frozenset(('alpha', 'gamma'))}
    # seed score 0.9 x query relevance 0.5
    assert pairs[frozenset(('alpha', 'gamma'))] == pytest.approx(0.45)


@pytest.mark.asyncio
async def test_api_query_learns_only_the_seed_pair_by_default(mock_app_mq_setup, monkeypatch):  # noqa: F811
    pairs = await _learned_pairs(monkeypatch, mock_app_mq_setup, enabled=False)
    assert set(pairs) == {frozenset(('alpha', 'beta'))}
