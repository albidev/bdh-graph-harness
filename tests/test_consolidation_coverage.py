"""Coverage regressions for consolidation configuration paths."""

from bdh_graph_harness import config
from bdh_graph_harness.memory import consolidation


def test_consolidation_helpers_read_config_defaults(monkeypatch):
    state = {"synapses": {"a|b": {"weight": 1.0}}}
    monkeypatch.setitem(config.CONFIG, "consolidation_downscale_factor", 0.5)
    monkeypatch.setitem(config.CONFIG, "consolidation_prune_weight_floor", 0.6)

    consolidation.synaptic_downscaling(state)
    consolidation.structural_pruning(state)

    assert state["synapses"] == {}


def test_prune_stale_dormant_reads_config_default(monkeypatch):
    state = {
        "synapses": {},
        "node_quality": {"gone": {"dormant": True, "dormant_cycles": 0}},
        "dormant_nodes": ["gone"],
    }
    monkeypatch.setitem(config.CONFIG, "consolidation_dormant_persist_cycles", 1)

    consolidation.prune_stale_dormant(state, {"gone": {}})

    assert state["node_quality"] == {}
    assert state["dormant_nodes"] == []


def test_consolidate_skips_optional_pruning_and_phantom_updates(monkeypatch):
    state = {"synapses": {}, "node_quality": {}, "dormant_nodes": []}
    monkeypatch.setitem(config.CONFIG, "consolidation_prune_dormant_nodes", False)
    monkeypatch.setitem(config.CONFIG, "phantom_links_enabled", False)

    result = consolidation.consolidate(state, {"node": {}}, edges={"node": []})

    assert result["cycles"] == 1


def test_consolidate_updates_phantom_links_when_vault_and_edges_exist(monkeypatch):
    state = {"synapses": {}, "node_quality": {}, "dormant_nodes": []}
    calls = []

    def fake_update(current_state, nodes, edges, vault_root, *, config=None, collection=None):
        calls.append((nodes, edges, vault_root, config, collection))
        current_state["phantom_links"] = {"a|b": {"score": 0.9}}
        return current_state

    from bdh_graph_harness.memory import phantom

    monkeypatch.setattr(phantom, "update_phantom_links", fake_update)
    monkeypatch.setitem(config.CONFIG, "consolidation_prune_dormant_nodes", False)
    monkeypatch.setitem(config.CONFIG, "phantom_links_enabled", True)
    monkeypatch.setitem(config.CONFIG, "vault_path", "/tmp/vault")

    consolidation.consolidate(state, {"a": {}, "b": {}}, edges={"a": [{"target": "b"}]})

    assert len(calls) == 1
    assert calls[0][:3] == ({"a": {}, "b": {}}, {"a": [{"target": "b"}]}, "/tmp/vault")
    assert state["phantom_links"]["a|b"]["score"] == 0.9
