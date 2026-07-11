"""Branch regression for reactivation with a stale dormant lookup list."""

from bdh_graph_harness.memory.quality import try_reactivate


def test_reactivate_succeeds_when_node_is_absent_from_lookup_list():
    state = {
        "node_quality": {"node": {"dormant": True, "score": 0.1}},
        "dormant_nodes": [],
    }

    assert try_reactivate("node", 1.0, state) is True
    assert state["node_quality"]["node"]["dormant"] is False
    assert state["dormant_nodes"] == []
