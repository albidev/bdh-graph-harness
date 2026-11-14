"""Quota traversal: the seeds are expanded with their best few direct neighbours."""
import math
from datetime import datetime, timezone

import chromadb
import pytest

import bdh_graph_harness.retrieval.attention as attention_mod
from bdh_graph_harness.config import config_overlay
from bdh_graph_harness.memory.hebbian import encode_synapse_key

QUERY = [1.0, 0.0, 0.0, 0.0]


def _vec(cosine):
    """A unit vector whose cosine with QUERY is ``cosine``."""
    return [cosine, math.sqrt(1.0 - cosine * cosine), 0.0, 0.0]


EMBEDDINGS = {
    's1': _vec(1.0),
    's2': _vec(0.99),
    'near': _vec(0.60),       # linked from s1
    'mid': _vec(0.50),        # linked from s2
    'low': _vec(0.30),        # linked from s1
    'far': _vec(0.10),        # linked from s2
    'deep': _vec(0.65),       # two links away, through near
    'unlinked': _vec(0.70),   # relevant, but no link from a seed
    'stranger': _vec(0.05),   # neither linked nor relevant
}
EDGES = {
    's1': [{'target': 'near', 'display': 'near'}, {'target': 'low', 'display': 'low'}],
    's2': [
        {'target': 'mid', 'display': 'mid'},
        {'target': 'far', 'display': 'far'},
        {'target': 's1', 'display': 's1'},
    ],
    'near': [{'target': 'deep', 'display': 'deep'}],
}
BASE = {
    'graph_traversal_mode': 'quota',
    'graph_neighbor_quota': 2,
    'graph_neighbor_min_relevance': 0.0,
    'adaptive_threshold': False,
    'hybrid_search': False,
    'hebbian_seed_boost': False,
    'okf_mode': False,
}


@pytest.fixture
def graph(monkeypatch):
    nodes = {nid: {'id': nid, 'title': nid.title(), 'text': f'{nid} body'} for nid in EMBEDDINGS}
    collection = chromadb.EphemeralClient().get_or_create_collection(
        'test_traversal_quota', metadata={'hnsw:space': 'cosine'},
    )
    if collection.count() > 0:
        collection.delete(ids=collection.get()['ids'])
    collection.add(
        ids=list(EMBEDDINGS),
        embeddings=list(EMBEDDINGS.values()),
        documents=[nodes[nid]['text'] for nid in EMBEDDINGS],
    )
    monkeypatch.setattr(attention_mod, 'get_embeddings', lambda texts: [QUERY])
    return nodes, EDGES, collection


def _run(graph, overrides=None, state=None, max_hop=None):
    nodes, edges, collection = graph
    routing = {}
    with config_overlay({**BASE, **(overrides or {})}):
        active = attention_mod.attention(
            'query', nodes, edges, collection, k=2, max_hop=max_hop,
            hebbian_state=state, routing_meta=routing,
        )
    return active, routing


def _synapse(a, b, weight):
    return {
        encode_synapse_key(a, b): {
            'weight': weight,
            'frequency': 4.0,
            'last_coactivated': datetime.now(timezone.utc).isoformat(),
            'consolidation_candidate_cycles': 1,
        }
    }


def test_the_most_relevant_direct_neighbours_fill_the_quota(graph):
    active, routing = _run(graph)
    assert set(active) == {'s1', 's2', 'near', 'mid'}
    assert routing['traversal'] == {'mode': 'quota', 'quota': 2, 'candidates': 4, 'admitted': 2}


def test_neighbours_rank_below_every_seed_and_keep_their_order(graph):
    active, _ = _run(graph, {'graph_neighbor_quota': 4})
    ranked = [nid for nid, _score in sorted(active.items(), key=lambda item: -item[1])]
    assert ranked == ['s1', 's2', 'near', 'mid', 'low', 'far']
    assert max(active['near'], active['mid']) < min(active['s1'], active['s2'])


def test_a_note_two_links_away_or_not_linked_is_never_admitted(graph):
    active, _ = _run(graph, {'graph_neighbor_quota': 10})
    assert 'deep' not in active        # more relevant than 'mid', but two links away
    assert 'unlinked' not in active    # more relevant than any neighbour, but not a neighbour
    assert 'stranger' not in active


def test_minimum_relevance_leaves_the_quota_unfilled_rather_than_padding_it(graph):
    active, routing = _run(graph, {'graph_neighbor_quota': 4, 'graph_neighbor_min_relevance': 0.35})
    assert set(active) == {'s1', 's2', 'near', 'mid'}
    assert routing['traversal']['admitted'] == 2


def test_admitted_neighbours_carry_traversal_provenance(graph):
    _active, routing = _run(graph)
    details = {item['id']: item for item in routing['activation_details']}
    assert details['near']['role'] == 'graph_neighbor'
    assert details['near']['hop'] == 1
    assert details['near']['parent_id'] == 's1'
    assert details['near']['matched_by'] == 'static_edge'
    assert details['near']['query_relevance'] == pytest.approx(0.60, abs=1e-3)
    assert details['mid']['parent_id'] == 's2'
    assert details['s1']['role'] == 'seed'


def test_no_hops_means_no_expansion(graph):
    active, routing = _run(graph, max_hop=0)
    assert set(active) == {'s1', 's2'}
    assert routing['traversal']['admitted'] == 0


def test_zero_quota_keeps_only_the_seeds(graph):
    active, _ = _run(graph, {'graph_neighbor_quota': 0})
    assert set(active) == {'s1', 's2'}


def test_threshold_mode_remains_the_default_walk(graph):
    _active, routing = _run(graph, {'graph_traversal_mode': 'threshold'})
    assert routing['traversal']['mode'] == 'threshold'


def test_a_learned_association_lifts_a_linked_neighbour_into_the_quota(graph):
    plain, _ = _run(graph, {'graph_neighbor_quota': 1})
    assert set(plain) == {'s1', 's2', 'near'}

    state = {'synapses': _synapse('s2', 'mid', 0.5)}
    learned, routing = _run(graph, {'graph_neighbor_quota': 1}, state=state)
    assert set(learned) == {'s1', 's2', 'mid'}
    detail = {item['id']: item for item in routing['activation_details']}['mid']
    assert detail['matched_by'] == 'static_and_hebbian_edge'
    assert detail['hebbian_edge_weight'] == 0.5


def test_a_learned_association_makes_an_unlinked_note_reachable(graph):
    state = {'synapses': _synapse('s1', 'unlinked', 0.5)}
    active, routing = _run(graph, {'graph_neighbor_quota': 1}, state=state)
    assert set(active) == {'s1', 's2', 'unlinked'}
    detail = {item['id']: item for item in routing['activation_details']}['unlinked']
    assert detail['role'] == 'hebbian_neighbor'
    assert detail['matched_by'] == 'hebbian_edge'


def test_a_learned_association_alone_does_not_admit_an_irrelevant_note(graph):
    state = {'synapses': _synapse('s1', 'stranger', 0.9)}
    active, _ = _run(
        graph,
        {'graph_neighbor_quota': 10, 'hebbian_dynamic_query_relevance_floor': 0.35},
        state=state,
    )
    assert 'stranger' not in active
    assert 'far' in active  # a declared link is still allowed at low relevance


def test_learned_edges_can_be_switched_off(graph):
    state = {'synapses': _synapse('s1', 'unlinked', 0.5)}
    active, _ = _run(graph, {'hebbian_dynamic_edges_enabled': False}, state=state)
    assert set(active) == {'s1', 's2', 'near', 'mid'}
