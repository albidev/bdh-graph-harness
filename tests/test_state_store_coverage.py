"""Behavioral regression tests for durable, atomic state persistence."""

import json

import pytest

from bdh_graph_harness.memory import state_store


def test_load_state_recovers_from_corrupt_json(tmp_path):
    """A malformed state file must not stop the graph from starting."""
    state_path = tmp_path / state_store.STATE_FILE
    state_path.write_text("{definitely-not-json", encoding="utf-8")

    state = state_store.load_state(str(tmp_path))

    assert state["synapses"] == {}
    assert state["queries"] == 0
    assert state["created"]
    assert state["updated"]


def test_load_state_creates_default_when_state_file_is_missing(tmp_path):
    state = state_store.load_state(str(tmp_path))

    assert state["synapses"] == {}
    assert state["queries"] == 0


def test_save_state_is_atomic_and_cleans_temporary_file_after_write_failure(tmp_path, monkeypatch):
    """A failed JSON write leaves no partial temporary state file behind."""
    state = {"synapses": {}, "queries": 1}
    tmp_state_path = tmp_path / f"{state_store.STATE_FILE}.tmp"

    def fail_dump(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(state_store.json, "dump", fail_dump)

    with pytest.raises(OSError, match="disk full"):
        state_store.save_state(str(tmp_path), state)

    assert not tmp_state_path.exists()


def test_save_state_reraises_when_temporary_file_was_never_created(tmp_path, monkeypatch):
    """Cleanup handles a failed write before the temporary file exists."""
    state = {"synapses": {}, "queries": 1}
    real_isfile = state_store.os.path.isfile

    def fail_dump(*_args, **_kwargs):
        raise OSError("permission denied")

    def isfile_without_temp(path):
        if str(path).endswith(".tmp"):
            return False
        return real_isfile(path)

    monkeypatch.setattr(state_store.json, "dump", fail_dump)
    monkeypatch.setattr(state_store.os.path, "isfile", isfile_without_temp)

    with pytest.raises(OSError, match="permission denied"):
        state_store.save_state(str(tmp_path), state)


def test_save_state_uses_replace_and_preserves_disk_only_synapse(tmp_path):
    """Concurrent writer data is merged and the final state is valid JSON."""
    disk_state = {
        "synapses": {"disk|only": {"weight": 0.1, "frequency": 1}},
        "queries": 2,
        "disk_flag": True,
    }
    (tmp_path / state_store.STATE_FILE).write_text(json.dumps(disk_state), encoding="utf-8")
    memory_state = {
        "synapses": {"memory|only": {"weight": 0.9, "frequency": 4}},
        "queries": 5,
        "memory_flag": True,
    }

    state_store.save_state(str(tmp_path), memory_state)

    saved = json.loads((tmp_path / state_store.STATE_FILE).read_text(encoding="utf-8"))
    assert set(saved["synapses"]) == {"disk|only", "memory|only"}
    assert saved["queries"] == 5
    assert saved["disk_flag"] is True
    assert saved["memory_flag"] is True
