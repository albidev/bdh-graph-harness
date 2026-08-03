"""State/config safety contract for the clean-room consolidation lane.

Locks in the clean-room state-selection and consolidation-observability
contract that the live config must uphold:

1. Active-state selection — ``.bdh-state-primary-seeds-v2.json`` is the
   clean-room state and the API classifies it as ``clean_room_shadow``.
   The quarantined legacy states (``.bdh-state.json``,
   ``.bdh-state-legacy-curated-v1.json``) must never be labelled as the
   clean-room mode. The default fallback (no config override) is
   intentionally ``legacy_active``, not ``clean_room_shadow`` — a missing
   override must not auto-promote shadow learning.
2. Profile-scoped consolidation settings — an explicit per-profile config
   dict drives consolidation deterministically, independent of the global
   ``CONFIG``.
3. Dry-run read-only behavior — a dry run returns a full report and
   restores the state exactly (including on a would-abort cycle), and
   never enriches the external phantom collection during a successful dry
   run.
4. Report provenance — the API ``hebbian_state`` block classifies the
   active state file/mode, and existing report fields are concrete (no
   unknown placeholders).

These are contract tests; they deliberately do not re-implement the runtime
safety gate (covered by the runtime lane in ``test_consolidation_guardrails.py``).
"""

from copy import deepcopy
from datetime import datetime, timedelta

from bdh_graph_harness.api.routes import _hebbian_state_status
from bdh_graph_harness.memory.consolidation import consolidate
from bdh_graph_harness.memory.state_store import _state_path
from bdh_graph_harness import config as bdh_config


CLEAN_ROOM_FILE = ".bdh-state-primary-seeds-v2.json"
LEGACY_FILE = ".bdh-state.json"
CURATED_FILE = ".bdh-state-legacy-curated-v1.json"


# ---------------------------------------------------------------------------
# 1. Active-state selection
# ---------------------------------------------------------------------------

def test_clean_room_state_file_resolves_as_active_target():
    """The clean-room filename is the resolved per-vault state path."""
    vault = "/vaults/core"
    original = bdh_config.CONFIG.get("hebbian_state_file")
    had_key = "hebbian_state_file" in bdh_config.CONFIG
    bdh_config.CONFIG["hebbian_state_file"] = CLEAN_ROOM_FILE
    try:
        assert _state_path(vault) == f"{vault}/{CLEAN_ROOM_FILE}"
    finally:
        if had_key:
            bdh_config.CONFIG["hebbian_state_file"] = original
        else:
            bdh_config.CONFIG.pop("hebbian_state_file", None)


def test_status_classifies_clean_room_as_clean_room_shadow():
    """API status must report primary-seeds-v2 as the clean-room shadow mode."""
    status = _hebbian_state_status(
        {"hebbian_state_file": CLEAN_ROOM_FILE}, {"synapses": {}}
    )
    assert status["mode"] == "clean_room_shadow"
    assert status["state_file"] == CLEAN_ROOM_FILE


def test_status_never_labels_quarantined_states_as_active_clean_room():
    """Legacy and curated files are never reported as the clean-room mode."""
    legacy = _hebbian_state_status(
        {"hebbian_state_file": LEGACY_FILE}, {"synapses": {}}
    )
    assert legacy["mode"] == "legacy_active"
    assert legacy["mode"] != "clean_room_shadow"

    curated = _hebbian_state_status(
        {"hebbian_state_file": CURATED_FILE}, {"synapses": {}}
    )
    assert curated["mode"] == "curated_experimental"
    assert curated["mode"] != "clean_room_shadow"


def test_status_default_falls_back_to_legacy_active_when_unconfigured():
    """Without a config override the status reports the quarantined legacy file.

    The contract requires the clean-room file to be pinned explicitly in the
    live config; the fallback is intentionally NOT the clean-room file (that
    would auto-promote shadow learning on a missing override).
    """
    status = _hebbian_state_status({}, {"synapses": {}})
    assert status["mode"] == "legacy_active"
    assert status["state_file"] == LEGACY_FILE


# ---------------------------------------------------------------------------
# 2. Profile-scoped consolidation settings
# ---------------------------------------------------------------------------

def _profile_config(**overrides):
    cfg = {
        "consolidation_downscale_factor": 1.0,  # clean-room: no double decay
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
    }
    cfg.update(overrides)
    return cfg


def _base_state():
    return {
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


def test_clean_room_profile_avoids_double_downscale():
    """A clean-room profile (factor 1.0) must not downscale weights."""
    cfg = _profile_config(consolidation_downscale_factor=1.0)
    state = _base_state()
    before = deepcopy(state)

    result = consolidate(state, {"a": {}, "b": {}}, config=cfg)

    assert result["downscale_factor"] == 1.0
    assert state["synapses"]["a|b"]["weight"] == before["synapses"]["a|b"]["weight"]


def test_consolidation_settings_are_profile_scoped_not_global():
    """An explicit per-profile config drives the result independently of CONFIG."""
    state_clean = _base_state()
    state_legacy = _base_state()

    clean_cfg = _profile_config(consolidation_downscale_factor=1.0)
    legacy_cfg = _profile_config(consolidation_downscale_factor=0.5)

    # Even if the global CONFIG disagrees, the profile config wins.
    original = bdh_config.CONFIG.get("consolidation_downscale_factor")
    bdh_config.CONFIG["consolidation_downscale_factor"] = 0.9
    try:
        r_clean = consolidate(state_clean, {"a": {}, "b": {}}, config=clean_cfg)
        r_legacy = consolidate(state_legacy, {"a": {}, "b": {}}, config=legacy_cfg)
    finally:
        bdh_config.CONFIG["consolidation_downscale_factor"] = original

    assert r_clean["downscale_factor"] == 1.0
    assert r_legacy["downscale_factor"] == 0.5
    assert state_clean["synapses"]["a|b"]["weight"] == 0.8
    assert state_legacy["synapses"]["a|b"]["weight"] == 0.4


# ---------------------------------------------------------------------------
# 3. Dry-run read-only behavior
# ---------------------------------------------------------------------------

def test_dry_run_returns_report_and_restores_state_exactly():
    """A dry run computes the full report but restores the original state."""
    cfg = _profile_config(consolidation_max_prune_ratio=1.0)
    state = _base_state()
    before = deepcopy(state)

    result = consolidate(state, {"a": {}, "b": {}}, config=cfg, dry_run=True)

    assert result["dry_run"] is True
    assert result["would_commit"] is True
    assert "synapses_before" in result
    assert state == before
    # Cycle counter is NOT advanced on dry-run restore.
    assert state["consolidation_cycles"] == before["consolidation_cycles"]


def test_dry_run_never_enriches_phantom_collection(monkeypatch):
    """A successful dry run must not touch the external phantom collection."""
    from bdh_graph_harness.memory import phantom

    calls = []
    monkeypatch.setattr(
        phantom,
        "update_phantom_links",
        lambda *args, **kwargs: calls.append(True),
    )

    cfg = _profile_config(
        consolidation_max_prune_ratio=1.0,
        phantom_links_enabled=True,
        vault_path="/tmp/vault",
    )
    state = _base_state()

    result = consolidate(
        state,
        {"a": {}, "b": {}},
        edges={"a": [{"target": "b"}]},
        config=cfg,
        dry_run=True,
    )

    assert calls == []  # phantom enrichment never ran during dry-run
    assert result["dry_run"] is True


def test_dry_run_abort_still_restores_state():
    """Even a would-abort dry run must not mutate the original state."""
    cfg = _profile_config(consolidation_max_prune_ratio=0.0)  # forces abort
    now = datetime(2026, 8, 3, 12, 0, 0)
    state = {
        "synapses": {
            "n0|n1": {
                "weight": 0.08,
                "frequency": 0.2,
                "last_coactivated": (now - timedelta(hours=100)).isoformat(),
            }
        },
        "node_quality": {},
        "dormant_nodes": [],
        "consolidation_cycles": 0,
    }
    before = deepcopy(state)

    result = consolidate(state, {"n0": {}, "n1": {}}, config=cfg, dry_run=True)

    assert result["dry_run"] is True
    assert result["aborted"] is True
    assert result["would_commit"] is False
    assert state == before


# ---------------------------------------------------------------------------
# 4. Report provenance
# ---------------------------------------------------------------------------

def test_report_carries_observable_state_provenance():
    """The active-state provenance is observable at the API status surface."""
    cfg = _profile_config(consolidation_downscale_factor=1.0)
    state = _base_state()

    result = consolidate(state, {"a": {}, "b": {}}, config=cfg)

    # Provenance via the API classification surface (independent of report).
    status = _hebbian_state_status(
        {"hebbian_state_file": CLEAN_ROOM_FILE}, {"synapses": {}}
    )
    assert status["mode"] == "clean_room_shadow"
    assert status["state_file"] == CLEAN_ROOM_FILE
    # The report identifies the clean-room profile by its downscale factor.
    assert result["downscale_factor"] == 1.0


def test_report_existing_fields_are_explicit_no_unknown_placeholders():
    """Consolidation report fields present today are concrete — no '?' or None."""
    cfg = _profile_config(consolidation_max_prune_ratio=1.0)
    state = _base_state()

    result = consolidate(state, {"a": {}, "b": {}}, config=cfg)

    for key in ("synapses_before", "candidate_synapses", "synapses_pruned", "cycles"):
        assert key in result
        assert result[key] is not None
        assert not isinstance(result[key], str) or result[key] != "?"

    # Explicit decision fields must be boolean/str, never '?' placeholders.
    assert isinstance(result["aborted"], bool)
    assert isinstance(result["would_commit"], bool)
    assert result["abort_reason"] in (None, "candidate_prune_ratio_exceeded")
    assert isinstance(result["downscale_factor"], float)
    assert result["weight_floor"] is not None
