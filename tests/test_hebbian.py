"""Tests for hebbian_update, load_state, save_state."""
import os
import json
import tempfile
from math import log1p
import pytest
import harness


def _synapse_key(a: str, b: str) -> str:
    from bdh_graph_harness.memory.hebbian import encode_synapse_key
    return encode_synapse_key(a, b)


@pytest.fixture
def temp_vault():
    d = tempfile.mkdtemp()
    return d


@pytest.fixture
def fresh_state():
    return {
        'synapses': {},
        'created': '2026-01-01T00:00:00',
        'updated': '2026-01-01T00:00:00',
        'queries': 0,
    }


# ---------------------------------------------------------------------------
# load_state / save_state
# ---------------------------------------------------------------------------

def test_load_state_creates_default(temp_vault):
    """load_state on empty vault returns default state structure."""
    state = harness.load_state(temp_vault)
    assert state['synapses'] == {}
    assert 'created' in state
    assert 'updated' in state
    assert state['queries'] == 0


def test_clean_room_state_file_does_not_touch_legacy_state(temp_vault, fresh_state, monkeypatch):
    """A configured clean-room file isolates new learning from legacy state."""
    legacy_path = os.path.join(temp_vault, harness.STATE_FILE)
    with open(legacy_path, 'w') as f:
        json.dump({'synapses': {'legacy|edge': {}}, 'queries': 99}, f)

    monkeypatch.setitem(harness.CONFIG, 'hebbian_state_file', '.bdh-state-primary-seeds-v2.json')
    fresh_state['synapses'] = {'new|edge': {'weight': 0.5}}
    harness.save_state(temp_vault, fresh_state)

    with open(legacy_path) as f:
        assert json.load(f)['queries'] == 99
    assert harness.load_state(temp_vault)['synapses'] == {'new|edge': {'weight': 0.5}}


def test_load_state_existing(temp_vault, fresh_state):
    """load_state reads existing state file."""
    fresh_state['synapses'] = {'a|b': {'weight': 0.5, 'frequency': 2, 'last_coactivated': 'x'}}
    harness.save_state(temp_vault, fresh_state)
    state = harness.load_state(temp_vault)
    assert state['synapses']['a|b']['weight'] == 0.5
    assert state['synapses']['a|b']['frequency'] == 2


def test_save_state_persists(temp_vault, fresh_state):
    """save_state writes to file and load_state reads it back identically."""
    fresh_state['synapses'] = {'x|y': {'weight': 0.9, 'frequency': 5, 'last_coactivated': 'now'}}
    harness.save_state(temp_vault, fresh_state)

    state_path = os.path.join(temp_vault, harness.STATE_FILE)
    assert os.path.isfile(state_path)

    loaded = harness.load_state(temp_vault)
    assert loaded['synapses']['x|y']['weight'] == 0.9
    assert loaded['synapses']['x|y']['frequency'] == 5


def test_save_state_updates_timestamp(temp_vault, fresh_state):
    """save_state updates the 'updated' field."""
    old_updated = fresh_state['updated']
    harness.save_state(temp_vault, fresh_state)
    loaded = harness.load_state(temp_vault)
    assert loaded['updated'] != old_updated


def test_load_state_lock_file_created(temp_vault):
    """Verify lock file is created during load_state."""
    harness.load_state(temp_vault)
    lock_path = os.path.join(temp_vault, harness.LOCK_FILE)
    assert os.path.isfile(lock_path)


def test_save_state_lock_file_created(temp_vault, fresh_state):
    """Verify lock file is created during save_state."""
    harness.save_state(temp_vault, fresh_state)
    lock_path = os.path.join(temp_vault, harness.LOCK_FILE)
    assert os.path.isfile(lock_path)


# ---------------------------------------------------------------------------
# hebbian_update
# ---------------------------------------------------------------------------

def test_hebbian_update_respects_conservative_learning_seed_budget(temp_vault, fresh_state):
    """Learning uses only its own top-2 budget, not all retrieval seeds."""
    active = {'a': 0.8, 'b': 0.6, 'c': 0.4}
    state, _, _ = harness.hebbian_update(active, fresh_state)

    assert len(state['synapses']) == 1
    assert _synapse_key('a', 'b') in state['synapses']
    assert _synapse_key('a', 'c') not in state['synapses']
    assert _synapse_key('b', 'c') not in state['synapses']


def test_hebbian_update_weight_formula(temp_vault, fresh_state):
    """Weight uses non-saturating frequency compression + actual recency."""
    alpha = harness.CONFIG['alpha']
    beta = harness.CONFIG['beta']
    freq_scale = harness.CONFIG.get('hebbian_frequency_scale', 10.0)

    active = {'a': 0.8, 'b': 0.6}
    state, _, _ = harness.hebbian_update(active, fresh_state)

    syn = state['synapses'][_synapse_key('a', 'b')]
    # Topology is seeds -> activated: only one seed here, so seed score * target score
    assert syn['frequency'] == pytest.approx(0.8 * 0.6, abs=1e-6)
    # Recency is ~1 because last_coactivated is just now.
    expected_weight = alpha * (log1p(syn['frequency']) / log1p(freq_scale)) + beta * 1.0
    assert syn['weight'] == pytest.approx(expected_weight, abs=1e-6)



def test_hebbian_update_frequency_increment(fresh_state):
    """Repeated co-activation increments frequency by the score product."""
    active = {'a': 0.8, 'b': 0.6}
    state, _, _ = harness.hebbian_update(active, fresh_state)
    first_freq = state['synapses'][_synapse_key('a', 'b')]['frequency']
    assert first_freq == pytest.approx(0.8 * 0.6, abs=1e-6)

    state, _, _ = harness.hebbian_update(active, state)
    assert state['synapses'][_synapse_key('a', 'b')]['frequency'] == pytest.approx(2 * first_freq, abs=1e-6)

    state, _, _ = harness.hebbian_update(active, state)
    assert state['synapses'][_synapse_key('a', 'b')]['frequency'] == pytest.approx(3 * first_freq, abs=1e-6)

    # Weight should have increased
    assert state['synapses'][_synapse_key('a', 'b')]['weight'] > 0


def test_hebbian_update_decay(fresh_state):
    """Test that synapses between non-active notes decay."""
    # Conservative write budget creates only the top pair.
    active1 = {'a': 0.8, 'b': 0.6, 'c': 0.4}
    state, _, _ = harness.hebbian_update(active1, fresh_state)
    assert len(state['synapses']) == 1

    original_weight = state['synapses'][_synapse_key('a', 'b')]['weight']

    # Now: activate D, E — a|b, a|c, b|c should decay
    active2 = {'d': 0.7, 'e': 0.5}
    state, _, _ = harness.hebbian_update(active2, state)

    # a|b should have decayed
    if _synapse_key('a', 'b') in state['synapses']:
        decayed_weight = state['synapses'][_synapse_key('a', 'b')]['weight']
        decay = harness.CONFIG['decay']
        assert decayed_weight < original_weight
        assert abs(decayed_weight - original_weight * decay) < 0.001


def test_hebbian_update_prune_low_weight(fresh_state):
    """Synapses below 0.01 are pruned by consolidation, not hebbian_update."""
    active1 = {'a': 0.8, 'b': 0.6}
    state, _, _ = harness.hebbian_update(active1, fresh_state)

    # hebbian_update no longer deletes; it only floors weights at 0.
    # After many decays the synapse remains with a small weight.
    for _ in range(80):
        state, _, _ = harness.hebbian_update({'c': 0.5, 'd': 0.4}, state)

    # a|b still exists (no hard delete here)
    assert _synapse_key('a', 'b') in state['synapses']
    assert state['synapses'][_synapse_key('a', 'b')]['weight'] < 0.01


def test_hebbian_update_queries_increment(fresh_state):
    """Test that queries counter increments."""
    active = {'a': 0.8, 'b': 0.6}
    state, _, _ = harness.hebbian_update(active, fresh_state)
    assert state['queries'] == 1
    state, _, _ = harness.hebbian_update(active, state)
    assert state['queries'] == 2


def test_hebbian_update_single_note(fresh_state):
    """Test that single active note creates no synapses."""
    active = {'a': 0.8}
    state, _, _ = harness.hebbian_update(active, fresh_state)
    assert len(state['synapses']) == 0
    assert state['queries'] == 1


def test_hebbian_update_semantic_sleep_is_dampened(fresh_state):
    """Semantic sleep reinforces co-activation without full user-query weight."""
    previous = harness.CONFIG.get('semantic_consolidation_frequency_increment', 0.3)
    harness.CONFIG['semantic_consolidation_frequency_increment'] = 0.3
    try:
        fresh_state['synapses']['old|memory'] = {
            'weight': 0.03,
            'frequency': 0.3,
            'last_coactivated': '2026-01-01T00:00:00',
            'created': '2026-01-01T00:00:00',
        }
        state, _, _ = harness.hebbian_update(
            {'a': 0.8, 'b': 0.6},
            fresh_state,
            source='nightly_semantic_consolidation',
        )
        assert state['synapses'][_synapse_key('a', 'b')]['frequency'] == pytest.approx(0.3 * 0.8 * 0.6, abs=1e-6)
        # old|memory was not touched, so it only decays (weight stays >=0)
        assert state['synapses']['old|memory']['weight'] <= 0.03
    finally:
        harness.CONFIG['semantic_consolidation_frequency_increment'] = previous
def test_hebbian_update_sets_last_coactivated(fresh_state):
    """Test that last_coactivated is set on new synapses."""
    active = {'a': 0.8, 'b': 0.6}
    state, _, _ = harness.hebbian_update(active, fresh_state)
    assert state['synapses'][_synapse_key('a', 'b')]['last_coactivated'] is not None
    assert state['synapses'][_synapse_key('a', 'b')]['created'] is not None


def test_hebbian_update_triggers_pruning_at_interval(fresh_state):
    """Test that pruning runs every N queries."""
    from bdh_graph_harness.config import CONFIG
    CONFIG['quality_prune_interval'] = 3  # every 3 queries
    nodes = {'a': {'title': 'A'}, 'b': {'title': 'B'}}

    # First 2 queries — no pruning
    state, _, pruned = harness.hebbian_update({'a': 0.8, 'b': 0.6}, fresh_state, nodes=nodes)
    assert pruned == 0
    assert state['queries'] == 1

    state, _, pruned = harness.hebbian_update({'a': 0.8, 'b': 0.6}, state, nodes=nodes)
    assert pruned == 0
    assert state['queries'] == 2

    # 3rd query — pruning triggers
    state, _, pruned = harness.hebbian_update({'a': 0.8, 'b': 0.6}, state, nodes=nodes)
    assert state['queries'] == 3
    assert 'node_quality' in state  # quality computed
    assert 'dormant_nodes' in state


def test_hebbian_update_reactivates_dormant(fresh_state):
    """Test that a dormant node is re-activated with strong activation."""
    from bdh_graph_harness.config import CONFIG
    CONFIG['quality_reactivation_score'] = 0.50
    fresh_state['node_quality'] = {
        'a': {'score': 0.1, 'dormant': True, 'evaluated_at': '2026-01-01'},
    }
    fresh_state['dormant_nodes'] = {'a'}

    state, _, _ = harness.hebbian_update({'a': 0.8, 'b': 0.6}, fresh_state)
    # Node a should be re-activated
    assert state['node_quality']['a']['dormant'] is False


# ---------------------------------------------------------------------------
# TDD: synapse key encoding — v2 base64url JSON codec
# ---------------------------------------------------------------------------

def test_v2_round_trip_pipe_in_note_ids(fresh_state):
    """P1: note IDs containing pipes produce exactly one v2: key that round-trips."""
    from bdh_graph_harness.memory.hebbian import (
        encode_synapse_key, decode_synapse_key,
    )

    a = "vault:Notes/A|B.md"   # pipe in note ID
    b = "vault:Notes/C|D.md"   # pipe in note ID

    key = encode_synapse_key(a, b)
    assert key.startswith("v2:"), f"new key must be v2: prefix, got {key!r}"

    decoded_a, decoded_b = decode_synapse_key(key)
    assert {decoded_a, decoded_b} == {a, b}

    # Canonical ordering: lexicographic
    assert decoded_a == min(a, b)
    assert decoded_b == max(a, b)


def test_v2_key_in_hebbian_update_producer(fresh_state):
    """P1: hebbian_update produces v2 keys for pipe-containing IDs."""
    a = "vault:Notes/A|B.md"
    b = "vault:Notes/C|D.md"

    active = {a: 0.8, b: 0.6}
    state, updated_keys, _ = harness.hebbian_update(active, fresh_state)

    # Exactly one synapse created
    assert len(state['synapses']) == 1
    key = list(state['synapses'].keys())[0]
    assert key.startswith("v2:"), f"producer must emit v2 key, got {key!r}"

    # The updated_keys set must also use v2
    assert len(updated_keys) == 1
    uk = list(updated_keys)[0]
    assert uk.startswith("v2:")


def test_v2_round_trip_consumer_read(fresh_state):
    """P1: consumer decode_synapse_key correctly reads v2 keys produced by hebbian_update."""
    from bdh_graph_harness.memory.hebbian import decode_synapse_key

    a = "vault:Notes/X|Y.md"
    b = "vault:Notes/Z.md"

    active = {a: 0.8, b: 0.6}
    state, _, _ = harness.hebbian_update(active, fresh_state)

    for key in state['synapses']:
        decoded_a, decoded_b = decode_synapse_key(key)
        assert {decoded_a, decoded_b} == {a, b}


def test_v2_note_ids_without_pipes(fresh_state):
    """v2 keys are used even for simple note IDs (consistent format)."""
    from bdh_graph_harness.memory.hebbian import encode_synapse_key, decode_synapse_key

    key = encode_synapse_key("note_a", "note_b")
    assert key.startswith("v2:")
    a, b = decode_synapse_key(key)
    assert {a, b} == {"note_a", "note_b"}


def test_legacy_simple_pipe_decodes():
    """Legacy a|b format with exactly one pipe is decoded correctly."""
    from bdh_graph_harness.memory.hebbian import decode_synapse_key

    a, b = decode_synapse_key("note_x|note_y")
    assert a == "note_x"
    assert b == "note_y"


def test_legacy_pipe_ordering_preserved():
    """Legacy keys keep their original ordering (not reordered)."""
    from bdh_graph_harness.memory.hebbian import decode_synapse_key

    a, b = decode_synapse_key("z_note|a_note")
    assert a == "z_note"
    assert b == "a_note"


def test_malformed_multi_pipe_raises():
    """Legacy keys with more than one pipe raise ValueError."""
    from bdh_graph_harness.memory.hebbian import decode_synapse_key

    with pytest.raises(ValueError, match="ambiguous"):
        decode_synapse_key("a|b|c")


def test_malformed_multi_pipe_safe_returns_none():
    """safe_decode_synapse_key returns None for ambiguous keys."""
    from bdh_graph_harness.memory.hebbian import safe_decode_synapse_key

    assert safe_decode_synapse_key("a|b|c") is None


def test_safe_decode_rejects_v2_non_string_endpoints():
    """Malformed v2 JSON must not leak fabricated non-string endpoints."""
    import base64
    from bdh_graph_harness.memory.hebbian import safe_decode_synapse_key

    key = "v2:" + base64.urlsafe_b64encode(b'["a",7]').rstrip(b'=').decode()
    assert safe_decode_synapse_key(key) is None
