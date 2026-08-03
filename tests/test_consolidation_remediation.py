"""P0/P1 remediation for the consolidation budget gate and clean-room double decay.

P0: raw candidate explosion must NOT abort the cycle. Candidates are ranked,
    planned deletion is capped at the configured per-cycle budget, and the hard
    safety gate is applied to the *planned* (budget-capped) mutation. The raw
    candidate ratio is surfaced as an explicit anomaly warning/metric.

P1: the clean-room state path (``hebbian_state_file=.bdh-state-primary-seeds-v2.json``)
    inherits ``consolidation_downscale_factor`` from the global default (0.90)
    when the live config omits it. That downscale would compound on top of the
    online Hebbian decay. The clean-room path therefore uses an effective factor
    of 1.0 (no extra persisted downscale), without changing the global default.
"""

from copy import deepcopy
from datetime import datetime, timedelta

from bdh_graph_harness.memory.consolidation import (
    consolidate,
    effective_downscale_factor,
)


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


def base_config(**overrides):
    cfg = {
        "consolidation_downscale_factor": 1.0,
        "consolidation_prune_weight_floor": 0.02,
        "consolidation_weak_weight_threshold": 0.15,
        "consolidation_weak_max_frequency": 1.0,
        "consolidation_weak_min_age_hours": 48,
        "consolidation_prune_confirm_cycles": 1,
        "consolidation_max_prune_ratio": 0.35,
        "consolidation_max_prune_per_cycle": 0.15,
        "consolidation_protect_backbone": False,
        "consolidation_protect_recent_hours": 72,
        "consolidation_prune_dormant_nodes": False,
        "phantom_links_enabled": False,
    }
    cfg.update(overrides)
    return cfg


# ---------------------------------------------------------------------------
# P0: budget-gated pruning (raw explosion is capped, not aborted)
# ---------------------------------------------------------------------------

def test_raw_candidate_explosion_is_capped_not_aborted():
    """All 20 synapses stale (raw ratio 1.0) must cap at budget, not abort."""
    state, nodes = make_state(20)
    cfg = base_config(
        consolidation_max_prune_ratio=0.35,
        consolidation_max_prune_per_cycle=0.15,
    )

    result = consolidate(state, nodes, config=cfg)

    assert result["aborted"] is False
    # Raw candidate ratio is observable as an anomaly warning.
    assert result["candidate_prune_ratio"] == 1.0
    assert result["candidate_ratio_anomaly"] is True
    # Planned deletion is capped at 15% of 20 = 3.
    assert result["planned_prune_count"] == 3
    assert result["planned_prune_ratio"] == 3 / 20
    assert result["synapses_pruned"] == 3
    assert result["capped"] is True
    assert result["decision"] == "commit"


def test_hard_gate_aborts_on_planned_ratio_when_budget_does_not_cap():
    """When the budget does not cap below max_ratio, the hard gate still aborts."""
    state, nodes = make_state(10)
    cfg = base_config(
        consolidation_max_prune_ratio=0.2,
        consolidation_max_prune_per_cycle=1.0,  # budget does not cap
    )
    before = deepcopy(state)

    result = consolidate(state, nodes, config=cfg)

    assert result["aborted"] is True
    assert result["abort_reason"] == "candidate_prune_ratio_exceeded"
    assert result["planned_prune_ratio"] == 1.0
    assert state == before


def test_report_includes_candidate_reasons():
    """The report exposes per-candidate reasons (no placeholders)."""
    state, nodes = make_state(4)
    cfg = base_config(consolidation_max_prune_ratio=1.0)

    result = consolidate(state, nodes, config=cfg)

    assert len(result["candidate_reasons"]) == 4
    assert set(result["candidate_reasons"].values()) == {"stale_weak"}


def test_dry_run_reports_decision_and_restores_state():
    """A dry run reports a dry_run decision and restores the original state."""
    state, nodes = make_state(4)
    before = deepcopy(state)
    cfg = base_config(consolidation_max_prune_ratio=1.0)

    result = consolidate(state, nodes, config=cfg, dry_run=True)

    assert result["dry_run"] is True
    assert result["decision"] == "dry_run"
    assert result["state_target"] == ".bdh-state.json"
    assert result["state_mode"] == "legacy_active"
    assert state == before


def test_report_carries_state_target_and_no_placeholders():
    """The report names the active state target and decision/abort are explicit."""
    state, nodes = make_state(4)
    cfg = base_config(
        consolidation_max_prune_ratio=1.0,
        hebbian_state_file=".bdh-state-primary-seeds-v2.json",
    )

    result = consolidate(state, nodes, config=cfg)

    assert result["state_target"] == ".bdh-state-primary-seeds-v2.json"
    assert result["state_mode"] == "clean_room_shadow"
    assert result["decision"] == "commit"
    assert result["abort_reason"] is None
    for field in ("planned_prune_count", "planned_prune_ratio", "candidate_prune_ratio",
                  "candidate_synapses", "synapses_pruned", "cycles"):
        assert result[field] is not None
        assert not (isinstance(result[field], str) and result[field] == "?")


# ---------------------------------------------------------------------------
# P1: clean-room effective downscale factor (no double decay)
# ---------------------------------------------------------------------------

def test_effective_downscale_factor_is_1_0_for_clean_room_shadow():
    """Clean-room shadow path must not downscale even if factor inherits 0.90."""
    cfg = base_config(
        consolidation_downscale_factor=0.9,  # inherited global default
        hebbian_state_file=".bdh-state-primary-seeds-v2.json",
    )
    assert effective_downscale_factor(cfg) == 1.0


def test_effective_downscale_factor_uses_configured_for_legacy():
    """Non-clean-room paths keep the configured factor unchanged."""
    cfg = base_config(consolidation_downscale_factor=0.9)
    assert effective_downscale_factor(cfg) == 0.9


def test_effective_downscale_factor_custom_clean_room_override():
    """A profile may opt into a non-1.0 clean-room factor via the seam."""
    cfg = base_config(
        consolidation_downscale_factor=0.9,
        consolidation_clean_room_downscale_factor=0.95,
        hebbian_state_file=".bdh-state-primary-seeds-v2.json",
    )
    assert effective_downscale_factor(cfg) == 0.95


def test_consolidate_no_double_decay_on_clean_room():
    """Consolidating the clean-room state must leave weights untouched (factor 1.0)."""
    state = {
        "synapses": {
            "a|b": {
                "weight": 0.8,
                "frequency": 3,
                "last_coactivated": "2026-08-01T00:00:00",
            }
        },
        "node_quality": {},
        "dormant_nodes": [],
        "consolidation_cycles": 0,
    }
    cfg = base_config(
        consolidation_downscale_factor=0.9,  # inherited global, must be neutralised
        hebbian_state_file=".bdh-state-primary-seeds-v2.json",
    )

    result = consolidate(state, {"a": {}, "b": {}}, config=cfg)

    assert result["downscale_factor"] == 1.0
    assert state["synapses"]["a|b"]["weight"] == 0.8  # unchanged, no double decay


def test_consolidate_still_downscales_legacy_path():
    """The legacy path keeps its configured downscale factor."""
    state = {
        "synapses": {
            "a|b": {
                "weight": 0.8,
                "frequency": 3,
                "last_coactivated": "2026-08-01T00:00:00",
            }
        },
        "node_quality": {},
        "dormant_nodes": [],
        "consolidation_cycles": 0,
    }
    cfg = base_config(consolidation_downscale_factor=0.9)

    result = consolidate(state, {"a": {}, "b": {}}, config=cfg)

    assert result["downscale_factor"] == 0.9
    assert state["synapses"]["a|b"]["weight"] == round(0.8 * 0.9, 6)
