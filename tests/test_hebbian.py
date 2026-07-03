"""Tests for hebbian_update, load_state, save_state."""
import os
import json
import tempfile
import pytest
import harness


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
    assert 'synapses' in state
    assert state['synapses'] == {}
    assert 'created' in state
    assert 'updated' in state
    assert state['queries'] == 0


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

def test_hebbian_update_creates_synapses(temp_vault, fresh_state):
    """Test that hebbian_update with 3 active notes creates all pairs."""
    active = {'a': 0.8, 'b': 0.6, 'c': 0.4}
    state = harness.hebbian_update(active, fresh_state)

    # 3 notes → C(3,2) = 3 pairs
    assert len(state['synapses']) == 3
    assert 'a|b' in state['synapses']
    assert 'a|c' in state['synapses']
    assert 'b|c' in state['synapses']


def test_hebbian_update_weight_formula(temp_vault, fresh_state):
    """Test weight = alpha * min(freq/10, 1) + beta * 1.0."""
    alpha = harness.CONFIG['alpha']
    beta = harness.CONFIG['beta']

    active = {'a': 0.8, 'b': 0.6}
    state = harness.hebbian_update(active, fresh_state)

    syn = state['synapses']['a|b']
    assert syn['frequency'] == 1
    expected_weight = alpha * min(1 / 10.0, 1.0) + beta * 1.0
    assert abs(syn['weight'] - expected_weight) < 0.001


def test_hebbian_update_frequency_increment(fresh_state):
    """Test that repeated co-activation increments frequency."""
    active = {'a': 0.8, 'b': 0.6}
    state = harness.hebbian_update(active, fresh_state)
    assert state['synapses']['a|b']['frequency'] == 1

    state = harness.hebbian_update(active, state)
    assert state['synapses']['a|b']['frequency'] == 2

    state = harness.hebbian_update(active, state)
    assert state['synapses']['a|b']['frequency'] == 3

    # Weight should have increased
    alpha = harness.CONFIG['alpha']
    beta = harness.CONFIG['beta']
    w1 = alpha * min(1 / 10.0, 1.0) + beta
    assert state['synapses']['a|b']['weight'] > w1


def test_hebbian_update_decay(fresh_state):
    """Test that synapses between non-active notes decay."""
    # First: activate A, B, C → creates synapses a|b, a|c, b|c
    active1 = {'a': 0.8, 'b': 0.6, 'c': 0.4}
    state = harness.hebbian_update(active1, fresh_state)
    assert len(state['synapses']) == 3

    original_weight = state['synapses']['a|b']['weight']

    # Now: activate D, E — a|b, a|c, b|c should decay
    active2 = {'d': 0.7, 'e': 0.5}
    state = harness.hebbian_update(active2, state)

    # a|b should have decayed
    if 'a|b' in state['synapses']:
        decayed_weight = state['synapses']['a|b']['weight']
        decay = harness.CONFIG['decay']
        assert decayed_weight < original_weight
        assert abs(decayed_weight - original_weight * decay) < 0.001


def test_hebbian_update_prune_low_weight(fresh_state):
    """Test that synapses below 0.01 are pruned after decay."""
    active1 = {'a': 0.8, 'b': 0.6}
    state = harness.hebbian_update(active1, fresh_state)

    # Repeatedly activate different notes to decay a|b below threshold
    # 0.37 * 0.95^n < 0.01 requires n > ~69 iterations
    for _ in range(80):
        state = harness.hebbian_update({'c': 0.5, 'd': 0.4}, state)

    # a|b should have been pruned (weight decays as 0.37 * 0.95^80 ≈ 0.0006)
    assert 'a|b' not in state['synapses']


def test_hebbian_update_queries_increment(fresh_state):
    """Test that queries counter increments."""
    active = {'a': 0.8, 'b': 0.6}
    state = harness.hebbian_update(active, fresh_state)
    assert state['queries'] == 1
    state = harness.hebbian_update(active, state)
    assert state['queries'] == 2


def test_hebbian_update_single_note(fresh_state):
    """Test that single active note creates no synapses."""
    active = {'a': 0.8}
    state = harness.hebbian_update(active, fresh_state)
    assert len(state['synapses']) == 0
    assert state['queries'] == 1


def test_hebbian_update_sets_last_coactivated(fresh_state):
    """Test that last_coactivated is set on new synapses."""
    active = {'a': 0.8, 'b': 0.6}
    state = harness.hebbian_update(active, fresh_state)
    assert state['synapses']['a|b']['last_coactivated'] is not None
    assert state['synapses']['a|b']['created'] is not None