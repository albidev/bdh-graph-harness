"""Tests for node quality scoring and dormant-node pruning."""

import pytest
from bdh_graph_harness.memory.quality import (
    compute_node_quality,
    compute_all_qualities,
    prune_dormant,
    try_reactivate,
    quality_stats,
)
from bdh_graph_harness.config import CONFIG


@pytest.fixture(autouse=True)
def _default_config():
    """Ensure CONFIG has quality defaults for all tests."""
    CONFIG['quality_threshold'] = 0.25
    CONFIG['quality_reactivation_score'] = 0.50
    CONFIG['quality_prune_interval'] = 50
    yield
    CONFIG.pop('quality_threshold', None)
    CONFIG.pop('quality_reactivation_score', None)
    CONFIG.pop('quality_prune_interval', None)


@pytest.fixture
def simple_synapses():
    """Synapses for a small graph: a-b (strong), a-c (weak), b-c (medium)."""
    return {
        'a|b': {'weight': 0.9, 'frequency': 10},
        'a|c': {'weight': 0.1, 'frequency': 2},
        'b|c': {'weight': 0.5, 'frequency': 5},
    }


@pytest.fixture
def simple_nodes():
    return {'a': {'title': 'A'}, 'b': {'title': 'B'}, 'c': {'title': 'C'}}


@pytest.fixture
def fresh_state():
    return {
        'synapses': {},
        'queries': 0,
        'node_quality': {},
        "dormant_nodes": [],
    }


# ---------------------------------------------------------------------------
# compute_node_quality
# ---------------------------------------------------------------------------
class TestComputeNodeQuality:
    def test_node_with_strong_edges(self, simple_synapses):
        q = compute_node_quality('a', simple_synapses, max_freq=10)
        assert q['score'] > 0.4
        assert q['strong_ratio'] == 0.5  # 1 of 2 edges > 0.7
        assert q['frequency'] == 12  # 10 + 2

    def test_node_with_no_edges(self):
        q = compute_node_quality('z', {}, max_freq=1)
        assert q['score'] == 0.0
        assert q['strong_ratio'] == 0.0
        assert q['mean_weight'] == 0.0
        assert q['frequency'] == 0

    def test_node_all_strong_edges(self, simple_synapses):
        """Node b has edges a|b (0.9) and b|c (0.5) — 50% strong."""
        q = compute_node_quality('b', simple_synapses, max_freq=10)
        assert q['strong_ratio'] == 0.5
        assert q['mean_weight'] == pytest.approx(0.7, abs=0.01)

    def test_node_all_weak_edges(self, simple_synapses):
        """Node c has edges a|c (0.1) and b|c (0.5) — 0% strong."""
        q = compute_node_quality('c', simple_synapses, max_freq=10)
        assert q['strong_ratio'] == 0.0
        assert q['mean_weight'] == pytest.approx(0.3, abs=0.01)

    def test_max_freq_normalisation(self, simple_synapses):
        """Frequency normalisation uses max_freq across all synapses."""
        q1 = compute_node_quality('a', simple_synapses, max_freq=10)
        q2 = compute_node_quality('a', simple_synapses, max_freq=100)
        # q2 should have lower freq_norm → lower score
        assert q1['score'] > q2['score']

    def test_score_range(self, simple_synapses):
        """Score is between 0 and 1."""
        for node_id in ['a', 'b', 'c']:
            q = compute_node_quality(node_id, simple_synapses, max_freq=10)
            assert 0.0 <= q['score'] <= 1.0


# ---------------------------------------------------------------------------
# compute_all_qualities
# ---------------------------------------------------------------------------
class TestComputeAllQualities:
    def test_all_nodes_scored(self, simple_synapses, simple_nodes):
        result = compute_all_qualities(simple_synapses, simple_nodes)
        assert len(result) == 3
        for nid in ['a', 'b', 'c']:
            assert nid in result
            assert 'score' in result[nid]
            assert 'dormant' in result[nid]
            assert 'evaluated_at' in result[nid]

    def test_dormant_flag_respects_threshold(self, simple_nodes):
        CONFIG['quality_threshold'] = 0.25
        # Node z with very weak edges — should be dormant
        synapses = {
            'a|z': {'weight': 0.05, 'frequency': 1},
            'b|z': {'weight': 0.08, 'frequency': 1},
        }
        result = compute_all_qualities(synapses, simple_nodes)
        assert result['c']['dormant'] is True  # no edges at all

    def test_empty_synapses(self, simple_nodes):
        result = compute_all_qualities({}, simple_nodes)
        assert len(result) == 3
        for nid in simple_nodes:
            assert result[nid]['dormant'] is True  # score 0 < threshold

    def test_empty_nodes(self, simple_synapses):
        result = compute_all_qualities(simple_synapses, {})
        assert len(result) == 0


# ---------------------------------------------------------------------------
# prune_dormant
# ---------------------------------------------------------------------------
class TestPruneDormant:
    def test_sets_dormant_nodes_set(self, simple_nodes, fresh_state):
        # Add very weak edges so one node is clearly dormant
        fresh_state['synapses'] = {
            'a|b': {'weight': 0.9, 'frequency': 10},
            'a|z': {'weight': 0.05, 'frequency': 1},
        }
        nodes = {'a': {'title': 'A'}, 'b': {'title': 'B'}, 'z': {'title': 'Z'}}
        state = prune_dormant(fresh_state, nodes)
        assert 'dormant_nodes' in state
        assert isinstance(state['dormant_nodes'], list)
        assert 'z' in state['dormant_nodes']  # weak node

    def test_updates_node_quality(self, simple_synapses, simple_nodes, fresh_state):
        fresh_state['synapses'] = simple_synapses
        state = prune_dormant(fresh_state, simple_nodes)
        assert 'node_quality' in state
        assert len(state['node_quality']) == 3

    def test_empty_graph(self, fresh_state):
        state = prune_dormant(fresh_state, {})
        assert state['dormant_nodes'] == []
        assert state['node_quality'] == {}


# ---------------------------------------------------------------------------
# try_reactivate
# ---------------------------------------------------------------------------
class TestTryReactivate:
    def test_reactivate_dormant_node(self, fresh_state):
        fresh_state['node_quality'] = {
            'x': {'score': 0.1, 'dormant': True, 'evaluated_at': '2026-01-01'},
        }
        fresh_state['dormant_nodes'] = ['x']
        result = try_reactivate('x', 0.6, fresh_state)
        assert result is True
        assert fresh_state['node_quality']['x']['dormant'] is False
        assert 'x' not in fresh_state['dormant_nodes']

    def test_dont_reactivate_below_threshold(self, fresh_state):
        fresh_state['node_quality'] = {
            'x': {'score': 0.1, 'dormant': True, 'evaluated_at': '2026-01-01'},
        }
        fresh_state['dormant_nodes'] = ['x']
        result = try_reactivate('x', 0.3, fresh_state)
        assert result is False
        assert fresh_state['node_quality']['x']['dormant'] is True
        assert 'x' in fresh_state['dormant_nodes']

    def test_skip_active_node(self, fresh_state):
        fresh_state['node_quality'] = {
            'x': {'score': 0.5, 'dormant': False, 'evaluated_at': '2026-01-01'},
        }
        fresh_state['dormant_nodes'] = []
        result = try_reactivate('x', 0.9, fresh_state)
        assert result is False

    def test_unknown_node(self, fresh_state):
        result = try_reactivate('nonexistent', 0.9, fresh_state)
        assert result is False


# ---------------------------------------------------------------------------
# quality_stats
# ---------------------------------------------------------------------------
class TestQualityStats:
    def test_empty_state(self):
        stats = quality_stats({})
        assert stats['total'] == 0
        assert stats['dormant'] == 0

    def test_with_dormant_nodes(self, simple_nodes):
        state = {
            'node_quality': {
                'a': {'score': 0.8, 'dormant': False},
                'b': {'score': 0.6, 'dormant': False},
                'c': {'score': 0.1, 'dormant': True},
            }
        }
        stats = quality_stats(state)
        assert stats['total'] == 3
        assert stats['dormant'] == 1
        assert stats['active'] == 2
        assert stats['avg_score'] == pytest.approx(0.5, abs=0.01)
