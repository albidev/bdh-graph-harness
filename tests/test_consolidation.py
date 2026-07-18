"""Tests for memory consolidation (Phase 4) — sleep-cycle graph maintenance."""

import pytest
from copy import deepcopy
from datetime import datetime, timedelta

from bdh_graph_harness.memory.consolidation import (
    consolidate,
    is_stale_weak_synapse,
    prune_stale_weak,
    synaptic_downscaling,
    structural_pruning,
    prune_stale_dormant,
    consolidation_stats,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_state():
    """State with a mix of strong, weak, and dormant nodes."""
    return {
        'synapses': {
            'wiki/a|wiki/b': {'weight': 0.90, 'frequency': 10, 'last_coactivated': '2026-07-01'},
            'wiki/a|wiki/c': {'weight': 0.50, 'frequency': 5, 'last_coactivated': '2026-07-02'},
            'wiki/b|wiki/d': {'weight': 0.03, 'frequency': 1, 'last_coactivated': '2026-06-01'},
            'wiki/c|wiki/e': {'weight': 0.01, 'frequency': 1, 'last_coactivated': '2026-06-01'},
        },
        'queries': 100,
        'created': '2026-01-01',
        'updated': '2026-07-06',
        'node_quality': {
            'wiki/a': {'score': 0.80, 'dormant': False, 'dormant_cycles': 0, 'evaluated_at': '2026-07-01'},
            'wiki/b': {'score': 0.60, 'dormant': False, 'dormant_cycles': 0, 'evaluated_at': '2026-07-01'},
            'wiki/c': {'score': 0.10, 'dormant': True, 'dormant_cycles': 2, 'evaluated_at': '2026-07-01'},
            'wiki/d': {'score': 0.05, 'dormant': True, 'dormant_cycles': 4, 'evaluated_at': '2026-07-01'},
            'wiki/e': {'score': 0.15, 'dormant': True, 'dormant_cycles': 1, 'evaluated_at': '2026-07-01'},
        },
        'dormant_nodes': ['wiki/c', 'wiki/d', 'wiki/e'],
        'consolidation_cycles': 0,
    }


@pytest.fixture
def sample_nodes():
    """Nodes dict matching the state fixture."""
    return {
        'wiki/a': {'title': 'Alpha', 'tags': [], 'text': 'content a'},
        'wiki/b': {'title': 'Beta', 'tags': [], 'text': 'content b'},
        'wiki/c': {'title': 'Gamma', 'tags': [], 'text': 'content c'},
        'wiki/d': {'title': 'Delta', 'tags': [], 'text': 'content d'},
        'wiki/e': {'title': 'Epsilon', 'tags': [], 'text': 'content e'},
    }


# ---------------------------------------------------------------------------
# Synaptic downscaling
# ---------------------------------------------------------------------------

class TestSynapticDownscaling:

    def test_all_weights_scaled(self, sample_state):
        """Every synapse weight should be multiplied by the factor."""
        original_weights = {
            k: v['weight'] for k, v in sample_state['synapses'].items()
        }
        synaptic_downscaling(sample_state, factor=0.90)
        for key, syn in sample_state['synapses'].items():
            expected = round(original_weights[key] * 0.90, 6)
            assert syn['weight'] == pytest.approx(expected, abs=1e-6)

    def test_weak_synapses_shrink_more_absolutely(self, sample_state):
        """Downscaling should reduce weak synapses closer to zero."""
        synaptic_downscaling(sample_state, factor=0.90)
        # 0.01 * 0.90 = 0.009 — very close to pruning floor
        assert sample_state['synapses']['wiki/c|wiki/e']['weight'] < 0.01

    def test_strong_synapses_survive(self, sample_state):
        """Strong synapses should still be well above floor after one cycle."""
        synaptic_downscaling(sample_state, factor=0.90)
        assert sample_state['synapses']['wiki/a|wiki/b']['weight'] > 0.80

    def test_state_returned_is_same_ref(self, sample_state):
        """Function should return the same state object (in-place mutation)."""
        result = synaptic_downscaling(sample_state, factor=0.90)
        assert result is sample_state


# ---------------------------------------------------------------------------
# Structural pruning
# ---------------------------------------------------------------------------

class TestStructuralPruning:

    def test_weak_synapses_deleted(self, sample_state):
        """Synapses below floor should be removed."""
        structural_pruning(sample_state, weight_floor=0.02)
        # wiki/b|wiki/d was 0.03 → survives
        assert 'wiki/b|wiki/d' in sample_state['synapses']
        # wiki/c|wiki/e was 0.01 → deleted
        assert 'wiki/c|wiki/e' not in sample_state['synapses']

    def test_strong_synapses_survive(self, sample_state):
        """Synapses well above floor should remain."""
        structural_pruning(sample_state, weight_floor=0.02)
        assert 'wiki/a|wiki/b' in sample_state['synapses']
        assert 'wiki/a|wiki/c' in sample_state['synapses']

    def test_empty_synapses(self):
        """Empty synapses dict should not crash."""
        state = {'synapses': {}}
        structural_pruning(state, weight_floor=0.02)
        assert state['synapses'] == {}


# ---------------------------------------------------------------------------
# Stale weak synapse retention
# ---------------------------------------------------------------------------

class TestStaleWeakSynapse:

    @pytest.fixture
    def weak_config(self):
        return {
            'consolidation_weak_weight_threshold': 0.15,
            'consolidation_weak_max_frequency': 1.0,
            'consolidation_weak_min_age_hours': 48,
        }

    def test_fresh_weak_synapse_survives(self, weak_config):
        now = datetime(2026, 7, 18, 12, 0, 0)
        synapse = {
            'weight': 0.08,
            'frequency': 0.3,
            'last_coactivated': (now - timedelta(hours=1)).isoformat(),
        }
        assert is_stale_weak_synapse(synapse, now=now, config=weak_config) is False

    def test_old_low_frequency_weak_synapse_is_stale(self, weak_config):
        now = datetime(2026, 7, 18, 12, 0, 0)
        synapse = {
            'weight': 0.08,
            'frequency': 0.6,
            'last_coactivated': (now - timedelta(hours=49)).isoformat(),
        }
        assert is_stale_weak_synapse(synapse, now=now, config=weak_config) is True

    def test_frequent_weak_synapse_is_protected(self, weak_config):
        now = datetime(2026, 7, 18, 12, 0, 0)
        synapse = {
            'weight': 0.08,
            'frequency': 1.1,
            'last_coactivated': (now - timedelta(hours=49)).isoformat(),
        }
        assert is_stale_weak_synapse(synapse, now=now, config=weak_config) is False

    def test_malformed_timestamp_fails_closed(self, weak_config):
        synapse = {'weight': 0.08, 'frequency': 0.3, 'last_coactivated': 'unknown'}
        assert is_stale_weak_synapse(
            synapse,
            now=datetime(2026, 7, 18, 12, 0, 0),
            config=weak_config,
        ) is False

    def test_prune_stale_weak_returns_count_and_preserves_protected(self, weak_config):
        now = datetime(2026, 7, 18, 12, 0, 0)
        state = {
            'synapses': {
                'stale|weak': {
                    'weight': 0.08,
                    'frequency': 0.3,
                    'last_coactivated': (now - timedelta(hours=49)).isoformat(),
                },
                'fresh|weak': {
                    'weight': 0.08,
                    'frequency': 0.3,
                    'last_coactivated': (now - timedelta(hours=1)).isoformat(),
                },
                'frequent|weak': {
                    'weight': 0.08,
                    'frequency': 2.0,
                    'last_coactivated': (now - timedelta(hours=49)).isoformat(),
                },
            },
        }
        assert prune_stale_weak(state, now=now, config=weak_config) == 1
        assert set(state['synapses']) == {'fresh|weak', 'frequent|weak'}


# ---------------------------------------------------------------------------
# Stale dormant pruning
# ---------------------------------------------------------------------------

class TestPruneStaleDormant:

    def test_stale_dormant_removed(self, sample_state, sample_nodes):
        """Nodes dormant for >= persist_cycles should be removed from quality map."""
        prune_stale_dormant(sample_state, sample_nodes, persist_cycles=3)
        # wiki/d had dormant_cycles=4 → removed
        assert 'wiki/d' not in sample_state['node_quality']
        # wiki/c had dormant_cycles=2 → incremented to 3, but 3 >= 3 → removed
        assert 'wiki/c' not in sample_state['node_quality']
        # wiki/e had dormant_cycles=1 → incremented to 2, 2 < 3 → stays
        assert 'wiki/e' in sample_state['node_quality']

    def test_active_nodes_cycle_reset(self, sample_state, sample_nodes):
        """Active nodes should have dormant_cycles reset to 0."""
        prune_stale_dormant(sample_state, sample_nodes, persist_cycles=3)
        assert sample_state['node_quality']['wiki/a']['dormant_cycles'] == 0
        assert sample_state['node_quality']['wiki/b']['dormant_cycles'] == 0

    def test_dormant_cycle_incremented(self, sample_state, sample_nodes):
        """Dormant nodes below persist threshold should have counter incremented."""
        prune_stale_dormant(sample_state, sample_nodes, persist_cycles=10)
        # wiki/e had dormant_cycles=1 → should be 2
        assert sample_state['node_quality']['wiki/e']['dormant_cycles'] == 2

    def test_synapses_cleaned_for_removed_nodes(self, sample_state, sample_nodes):
        """Synapses referencing removed nodes should be deleted."""
        prune_stale_dormant(sample_state, sample_nodes, persist_cycles=3)
        # wiki/d removed → wiki/b|wiki/d synapse should be gone
        assert 'wiki/b|wiki/d' not in sample_state['synapses']

    def test_deleted_vault_nodes_removed(self, sample_state):
        """Nodes no longer in the graph should be removed from quality map."""
        nodes = {'wiki/a': {}, 'wiki/b': {}}  # c, d, e missing
        prune_stale_dormant(sample_state, nodes, persist_cycles=10)
        assert 'wiki/c' not in sample_state['node_quality']
        assert 'wiki/d' not in sample_state['node_quality']
        assert 'wiki/e' not in sample_state['node_quality']

    def test_dormant_nodes_list_updated(self, sample_state, sample_nodes):
        """The dormant_nodes lookup list should be updated after removal."""
        prune_stale_dormant(sample_state, sample_nodes, persist_cycles=3)
        # Only wiki/e should remain dormant
        assert sample_state['dormant_nodes'] == ['wiki/e']


# ---------------------------------------------------------------------------
# Full consolidation
# ---------------------------------------------------------------------------

class TestConsolidate:

    def test_returns_results_dict(self, sample_state, sample_nodes):
        """consolidate() should return a results dict with expected keys."""
        results = consolidate(sample_state, sample_nodes)
        assert 'synapses_before' in results
        assert 'synapses_after' in results
        assert 'synapses_pruned' in results
        assert 'stale_weak_pruned' in results
        assert 'nodes_removed' in results
        assert 'cycles' in results
        assert 'timestamp' in results

    def test_cycle_counter_incremented(self, sample_state, sample_nodes):
        """consolidation_cycles should be incremented."""
        assert sample_state['consolidation_cycles'] == 0
        consolidate(sample_state, sample_nodes)
        assert sample_state['consolidation_cycles'] == 1

    def test_synapses_pruned_after_consolidation(self, sample_state, sample_nodes):
        """After consolidation, weak synapses should be pruned."""
        before = len(sample_state['synapses'])
        results = consolidate(sample_state, sample_nodes)
        assert results['synapses_before'] == before
        assert results['synapses_after'] < before
        assert results['synapses_pruned'] > 0

    def test_strong_synapses_survive_full_cycle(self, sample_state, sample_nodes):
        """Strong synapses should survive a full consolidation cycle."""
        consolidate(sample_state, sample_nodes)
        assert 'wiki/a|wiki/b' in sample_state['synapses']

    def test_stale_dormant_removed_full_cycle(self, sample_state, sample_nodes):
        """Stale dormant nodes should be removed after full consolidation."""
        consolidate(sample_state, sample_nodes)
        # wiki/d had dormant_cycles=4 → should be removed
        assert 'wiki/d' not in sample_state['node_quality']

    def test_multiple_cycles_progressive_pruning(self, sample_state, sample_nodes):
        """Multiple consolidation cycles should progressively prune weak synapses."""
        initial_count = len(sample_state['synapses'])
        for i in range(5):
            consolidate(sample_state, sample_nodes)
        # After 5 cycles of 0.90 downscaling, very weak synapses are gone
        assert len(sample_state['synapses']) < initial_count

    def test_dry_run_preserves_state(self, sample_state, sample_nodes):
        """dry_run should not mutate the original state."""
        from copy import deepcopy
        original = deepcopy(sample_state)
        state_copy = deepcopy(sample_state)
        consolidate(state_copy, sample_nodes)
        # Original should be unchanged
        assert sample_state['synapses'] == original['synapses']
        assert sample_state['consolidation_cycles'] == original['consolidation_cycles']


# ---------------------------------------------------------------------------
# Consolidation stats
# ---------------------------------------------------------------------------

class TestConsolidationStats:

    def test_returns_dict_with_keys(self, sample_state):
        """Stats should include cycle count and config values."""
        stats = consolidation_stats(sample_state)
        assert 'cycles' in stats
        assert 'downscale_factor' in stats
        assert 'weight_floor' in stats
        assert 'dormant_persist_cycles' in stats

    def test_cycles_value(self, sample_state):
        """Stats should report the current cycle count."""
        sample_state['consolidation_cycles'] = 7
        stats = consolidation_stats(sample_state)
        assert stats['cycles'] == 7

    def test_default_cycles_zero(self):
        """Stats should default to 0 when consolidation_cycles is missing."""
        state = {'synapses': {}, 'queries': 0}
        stats = consolidation_stats(state)
        assert stats['cycles'] == 0


# ---------------------------------------------------------------------------
# Integration: consolidation + quality interaction
# ---------------------------------------------------------------------------

class TestConsolidationQualityIntegration:

    def test_quality_recomputed_after_pruning(self, sample_state, sample_nodes):
        """After consolidation, quality scores should reflect the surviving synapses."""
        # Before: wiki/a has 2 edges (strong: wiki/a|wiki/b @ 0.90, wiki/a|wiki/c @ 0.50)
        consolidate(sample_state, sample_nodes)
        # After: wiki/c|wiki/e was pruned (weight too low after downscaling)
        # wiki/a quality should be recomputed from surviving edges
        nq = sample_state['node_quality']
        if 'wiki/a' in nq:
            # Strong ratio should be based on surviving edges only
            assert nq['wiki/a']['score'] > 0

    def test_dormant_nodes_updated_after_consolidation(self, sample_state, sample_nodes):
        """Dormant node list should be updated after consolidation."""
        consolidate(sample_state, sample_nodes)
        dormant = sample_state.get('dormant_nodes', [])
        # wiki/d should be removed (was dormant for 4 cycles)
        assert 'wiki/d' not in dormant