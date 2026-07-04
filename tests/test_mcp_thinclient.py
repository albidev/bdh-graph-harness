"""Tests for state_store merge logic and MCP thin client behaviour."""
import os
import json
import tempfile
import pytest
import bdh_graph_harness.memory.state_store as bdh_state_store


# ---------------------------------------------------------------------------
# merge_states unit tests
# ---------------------------------------------------------------------------

def test_merge_disk_only_synapse_preserved():
    """Synapse only on disk should survive the merge."""
    disk = {'synapses': {'a|b': {'weight': 0.5, 'frequency': 2, 'last_coactivated': '2026-01-01'}}, 'queries': 10}
    mem = {'synapses': {}, 'queries': 5}
    merged = bdh_state_store.merge_states(disk, mem)
    assert 'a|b' in merged['synapses']
    assert merged['synapses']['a|b']['frequency'] == 2


def test_merge_mem_only_synapse_preserved():
    """Synapse only in memory should survive the merge."""
    disk = {'synapses': {}, 'queries': 5}
    mem = {'synapses': {'x|y': {'weight': 0.9, 'frequency': 7, 'last_coactivated': '2026-07-04'}}, 'queries': 3}
    merged = bdh_state_store.merge_states(disk, mem)
    assert 'x|y' in merged['synapses']
    assert merged['synapses']['x|y']['frequency'] == 7


def test_merge_shared_synapse_higher_frequency_wins():
    """When a synapse exists in both, the entry with higher frequency wins."""
    disk = {'synapses': {'a|b': {'weight': 0.5, 'frequency': 4, 'last_coactivated': '2026-01-01'}}, 'queries': 10}
    mem = {'synapses': {'a|b': {'weight': 0.8, 'frequency': 2, 'last_coactivated': '2026-07-04'}}, 'queries': 5}
    merged = bdh_state_store.merge_states(disk, mem)
    assert merged['synapses']['a|b']['frequency'] == 4
    assert merged['synapses']['a|b']['weight'] == 0.5


def test_merge_shared_synapse_frequency_tie_weight_breaks():
    """When frequency ties, higher weight wins."""
    disk = {'synapses': {'a|b': {'weight': 0.3, 'frequency': 5, 'last_coactivated': '2026-01-01'}}, 'queries': 10}
    mem = {'synapses': {'a|b': {'weight': 0.9, 'frequency': 5, 'last_coactivated': '2026-07-04'}}, 'queries': 5}
    merged = bdh_state_store.merge_states(disk, mem)
    assert merged['synapses']['a|b']['weight'] == 0.9


def test_merge_queries_takes_max():
    """The queries count should be the maximum of the two."""
    disk = {'synapses': {}, 'queries': 42}
    mem = {'synapses': {}, 'queries': 17}
    merged = bdh_state_store.merge_states(disk, mem)
    assert merged['queries'] == 42


def test_merge_other_keys_memory_wins():
    """Non-synapse, non-queries keys: memory version wins, disk-only keys preserved."""
    disk = {'synapses': {}, 'queries': 1, 'created': '2026-01-01', 'disk_only': 'disk_val'}
    mem = {'synapses': {}, 'queries': 2, 'created': '2026-07-04', 'mem_only': 'mem_val'}
    merged = bdh_state_store.merge_states(disk, mem)
    assert merged['created'] == '2026-07-04'  # memory wins
    assert merged['disk_only'] == 'disk_val'  # disk-only preserved
    assert merged['mem_only'] == 'mem_val'  # mem-only added


def test_save_state_merges_with_disk(tmp_path):
    """save_state should merge with on-disk state, not overwrite blindly."""
    # Create initial state on disk
    initial = {
        'synapses': {'a|b': {'weight': 0.5, 'frequency': 10, 'last_coactivated': '2026-01-01'}},
        'queries': 20,
        'created': '2026-01-01',
    }
    state_path = os.path.join(str(tmp_path), bdh_state_store.STATE_FILE)
    with open(state_path, 'w') as f:
        json.dump(initial, f)

    # Simulate an in-memory state that doesn't know about the disk synapse
    mem_state = {
        'synapses': {'x|y': {'weight': 0.9, 'frequency': 3, 'last_coactivated': '2026-07-04'}},
        'queries': 5,
        'created': '2026-07-04',
    }
    bdh_state_store.save_state(str(tmp_path), mem_state)

    # Reload and verify both synapses survived
    loaded = bdh_state_store.load_state(str(tmp_path))
    assert 'a|b' in loaded['synapses']  # disk-only synapse preserved
    assert 'x|y' in loaded['synapses']  # mem-only synapse preserved
    assert loaded['queries'] == 20  # max(20, 5)


# ---------------------------------------------------------------------------
# MCP thin client tests
# ---------------------------------------------------------------------------

def test_mcp_http_get_returns_none_when_server_down():
    """_http_get should return None when the server is not reachable."""
    from bdh_graph_harness.mcp_server import _http_get
    result = _http_get("http://localhost:99999/api/stats", timeout=0.5)
    assert result is None


def test_mcp_http_post_returns_none_when_server_down():
    """_http_post should return None when the server is not reachable."""
    from bdh_graph_harness.mcp_server import _http_post
    result = _http_post("http://localhost:99999/api/query", {"query": "test"}, timeout=0.5)
    assert result is None


def test_mcp_api_url_construction():
    """_api_url should build correct URLs with defaults and overrides."""
    from bdh_graph_harness.mcp_server import _api_url
    assert _api_url("/api/stats") == "http://localhost:8643/api/stats"
    assert _api_url("/api/query", host="1.2.3.4", port=9999) == "http://1.2.3.4:9999/api/query"


def test_mcp_api_url_env_override(monkeypatch):
    """_api_url should respect BDH_API_HOST and BDH_API_PORT env vars."""
    monkeypatch.setenv("BDH_API_HOST", "100.84.148.17")
    monkeypatch.setenv("BDH_API_PORT", "8080")
    from bdh_graph_harness.mcp_server import _api_url
    assert _api_url("/api/stats") == "http://100.84.148.17:8080/api/stats"