"""Approval tests for safe BDH consolidation guardrails."""

from copy import deepcopy
from datetime import datetime, timedelta

from bdh_graph_harness.memory.consolidation import consolidate, prune_stale_weak


def stale_synapse(now, weight=0.08):
    return {
        "weight": weight,
        "frequency": 0.2,
        "last_coactivated": (now - timedelta(hours=100)).isoformat(),
    }


def make_state(count=10, now=None):
    now = now or datetime(2026, 7, 23, 12, 0, 0)
    synapses = {f"n{i}|n{i + 1}": stale_synapse(now) for i in range(count)}
    nodes = {f"n{i}": {} for i in range(count + 1)}
    return {
        "synapses": synapses,
        "node_quality": {nid: {"dormant": False, "dormant_cycles": 0} for nid in nodes},
        "dormant_nodes": [],
        "consolidation_cycles": 0,
    }, nodes


def guardrail_config(**overrides):
    config = {
        "consolidation_downscale_factor": 1.0,
        "consolidation_prune_weight_floor": 0.02,
        "consolidation_weak_weight_threshold": 0.15,
        "consolidation_weak_max_frequency": 1.0,
        "consolidation_weak_min_age_hours": 48,
        "consolidation_prune_confirm_cycles": 1,
        "consolidation_max_prune_ratio": 0.35,
        "consolidation_max_prune_per_cycle": 0.15,
        "consolidation_protect_backbone": True,
        "consolidation_protect_recent_hours": 72,
        "consolidation_prune_dormant_nodes": False,
        "phantom_links_enabled": False,
        **overrides,
    }
    return config


def test_kill_switch_aborts_and_restores_state_when_candidates_exceed_ratio():
    state, nodes = make_state(10)
    before = deepcopy(state)

    result = consolidate(state, nodes, config=guardrail_config())

    assert result["aborted"] is True
    assert result["abort_reason"] == "candidate_prune_ratio_exceeded"
    assert result["candidate_prune_ratio"] == 1.0
    assert state == before


def test_dry_run_returns_detailed_report_without_mutating_state():
    state, nodes = make_state(4)
    before = deepcopy(state)

    result = consolidate(
        state,
        nodes,
        config=guardrail_config(consolidation_max_prune_ratio=1.0),
        dry_run=True,
    )

    assert result["dry_run"] is True
    assert result["would_commit"] is True
    assert result["phase_order"] == [
        "snapshot", "downscale", "protection", "candidate_scan",
        "hysteresis", "safety_gate", "apply_prune", "quality", "dormant", "phantom",
    ]
    assert "candidate_synapses" in result
    assert "protected_synapses" in result
    assert "pending_confirmation" in result
    assert state == before


def test_structural_and_bridge_synapses_are_protected():
    now = datetime(2026, 7, 23, 12, 0, 0)
    state = {
        "synapses": {
            "a|b": stale_synapse(now),
            "c|b": stale_synapse(now),
            "a|c": stale_synapse(now),
        },
        "node_quality": {},
        "dormant_nodes": [],
        "consolidation_cycles": 0,
    }
    nodes = {"a": {}, "b": {}, "c": {}}
    edges = {"a": [{"target": "b"}], "b": [{"target": "c"}], "c": []}

    result = consolidate(
        state,
        nodes,
        edges=edges,
        config=guardrail_config(consolidation_max_prune_ratio=1.0),
    )

    assert result["aborted"] is False
    assert "a|b" in state["synapses"]
    assert "c|b" in state["synapses"]
    assert result["protected_synapses"] >= 2


def test_hysteresis_requires_two_consecutive_candidate_cycles():
    now = datetime(2026, 7, 23, 12, 0, 0)
    state = {"synapses": {"a|b": stale_synapse(now)}, "consolidation_cycles": 0}
    config = guardrail_config(consolidation_prune_confirm_cycles=2)

    assert prune_stale_weak(state, now=now, config=config) == 0
    assert state["synapses"]["a|b"]["consolidation_candidate_cycles"] == 1
    assert prune_stale_weak(state, now=now, config=config) == 1
    assert "a|b" not in state["synapses"]


def test_per_cycle_cap_prunes_at_most_fifteen_percent():
    state, nodes = make_state(20)
    config = guardrail_config(
        consolidation_max_prune_ratio=1.0,
        consolidation_max_prune_per_cycle=0.15,
    )

    result = consolidate(state, nodes, config=config)

    assert result["aborted"] is False
    assert result["candidate_synapses"] == 20
    assert result["synapses_pruned"] == 3
    assert result["capped"] is True


def test_consolidation_does_not_run_phantom_update_after_aborted_cycle(monkeypatch):
    state, nodes = make_state(10)
    called = []

    from bdh_graph_harness.memory import phantom
    monkeypatch.setattr(
        phantom,
        "update_phantom_links",
        lambda *args, **kwargs: called.append(True),
    )

    result = consolidate(
        state,
        nodes,
        edges={"n0": []},
        config=guardrail_config(
            consolidation_max_prune_ratio=0.35,
            phantom_links_enabled=True,
            vault_path="/tmp/vault",
        ),
    )

    assert result["aborted"] is True
    assert called == []
