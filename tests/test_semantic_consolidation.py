"""Tests for semantic sleep source selection, checkpoints, and API orchestration."""

import asyncio
import json
import os
import sqlite3
import time
from types import SimpleNamespace

import pytest

from bdh_graph_harness.memory.semantic_consolidation import (
    load_checkpoint,
    save_checkpoint_atomic,
    select_candidate_notes,
    select_candidate_sessions,
)
import bdh_graph_harness.api.routes as routes


class _Request:
    def __init__(self, payload):
        self._payload = payload

    async def json(self):
        return self._payload

    @property
    def query(self):
        return {}


def _config(**overrides):
    config = {
        "semantic_consolidation_enabled": True,
        "semantic_consolidation_checkpoint": ".checkpoint.json",
        "semantic_consolidation_source_globs": ["wiki/entities/*.md"],
        "semantic_consolidation_max_age_hours": 48,
        "semantic_consolidation_exclude_globs": ["wiki/concepts/*", ".bdh-*"],
        "semantic_consolidation_max_sources": 3,
        "semantic_consolidation_max_source_chars": 8000,
        "semantic_consolidation_max_concepts": 5,
        "semantic_consolidation_session_enabled": False,
    }
    config.update(overrides)
    return config


def test_select_candidate_notes_uses_content_hash_and_excludes_noise(tmp_path):
    daily = tmp_path / "wiki" / "entities"
    daily.mkdir(parents=True)
    source = daily / "2026-07-13.md"
    source.write_text("durable decision", encoding="utf-8")
    ignored = tmp_path / "wiki" / "concepts"
    ignored.mkdir(parents=True)
    (ignored / "generated.md").write_text("ignore", encoding="utf-8")

    config = _config()
    checkpoint = load_checkpoint(tmp_path, config)
    candidates = select_candidate_notes(tmp_path, config, checkpoint)
    assert [item["path"] for item in candidates] == ["wiki/entities/2026-07-13.md"]
    assert len(candidates[0]["sha256"]) == 64

    checkpoint["processed"] = {
        candidates[0]["path"]: {"sha256": candidates[0]["sha256"]}
    }
    assert select_candidate_notes(tmp_path, config, checkpoint) == []

    source.write_text("changed durable decision", encoding="utf-8")
    changed = select_candidate_notes(tmp_path, config, checkpoint)
    assert len(changed) == 1
    assert changed[0]["sha256"] != candidates[0]["sha256"]

    os.utime(source, (time.time() - 72 * 3600, time.time() - 72 * 3600))
    assert select_candidate_notes(tmp_path, config, checkpoint) == []


def test_select_candidate_sessions_ignores_cron_and_tool_output(tmp_path):
    db_path = tmp_path / "state.db"
    db = sqlite3.connect(db_path)
    db.executescript(
        """
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY, source TEXT, title TEXT,
            started_at REAL, ended_at REAL
        );
        CREATE TABLE messages (
            id INTEGER PRIMARY KEY, session_id TEXT, role TEXT,
            content TEXT, timestamp REAL
        );
        """
    )
    now = time.time()
    db.executemany(
        "INSERT INTO sessions VALUES (?, ?, ?, ?, ?)",
        [
            ("human-1", "discord", "Architecture discussion", now, now),
            ("cron-1", "cron", "Cron noise", now, now),
        ],
    )
    db.executemany(
        "INSERT INTO messages VALUES (?, ?, ?, ?, ?)",
        [
            (1, "human-1", "user", "We decided to isolate episodic memory from the core vault because cross-vault retrieval can contaminate durable concepts and make provenance ambiguous.", now),
            (2, "human-1", "tool", "Ignore this tool output", now),
            (3, "human-1", "assistant", "The durable decision is to keep vault boundaries explicit, record the source vault, and require an intentional promotion step before core neurogenesis.", now),
            (4, "cron-1", "user", "A cron status message", now),
        ],
    )
    db.commit()
    db.close()

    config = _config(
        semantic_consolidation_session_db_path=str(db_path),
        semantic_consolidation_session_enabled=True,
        semantic_consolidation_include_cron_sessions=False,
        semantic_consolidation_max_session_chars=12000,
    )
    candidates = select_candidate_sessions(config, {"sessions": {}}, max_sessions=5)
    assert len(candidates) == 1
    assert candidates[0]["session_id"] == "human-1"
    assert "Ignore this tool output" not in candidates[0]["content"]
    assert "durable decision" in candidates[0]["content"]

    checkpoint = {"sessions": {"human-1": {"last_message_id": 3}}}
    assert select_candidate_sessions(config, checkpoint, max_sessions=5) == []


def test_checkpoint_save_is_atomic_and_reloadable(tmp_path):
    config = _config()
    checkpoint = {"version": 1, "last_run_at": None, "processed": {}}
    path = save_checkpoint_atomic(tmp_path, checkpoint, config)
    assert path == tmp_path / ".checkpoint.json"
    loaded = load_checkpoint(tmp_path, config)
    assert loaded["version"] == 1
    assert loaded["processed"] == {}
    assert loaded["last_run_at"]
    assert not list(tmp_path.glob(".*.checkpoint.json.*"))


@pytest.mark.asyncio
async def test_semantic_endpoint_is_idempotent(monkeypatch, tmp_path):
    source_dir = tmp_path / "wiki" / "entities"
    source_dir.mkdir(parents=True)
    (source_dir / "daily.md").write_text("A durable architectural decision", encoding="utf-8")
    config = _config()
    ctx = SimpleNamespace(
        config=SimpleNamespace(id="core", path=str(tmp_path), settings=config),
        semantic_lock=asyncio.Lock(),
        event_sequence=0,
    )
    monkeypatch.setattr(routes, "_resolve_vault_ctx", lambda _state, _vault: (ctx, None))
    monkeypatch.setattr(routes, "broadcast_activation", _noop_broadcast)
    calls = []

    async def fake_source(source, ctx, ws_clients, *, dry_run, max_concepts):
        calls.append((source["path"], dry_run, max_concepts))
        return {
            "path": source["path"],
            "new_concepts": [{"id": "wiki/concepts/decision", "title": "Decision"}],
            "hebbian_updates": 2,
            "activated_notes": 3,
            "dry_run": dry_run,
        }

    monkeypatch.setattr(routes, "_run_semantic_source", fake_source)
    first = await routes.api_semantic_consolidate(_Request({}), {}, set())
    first_data = json.loads(first.text)
    assert first.status == 200
    assert first_data["sources_processed"] == 1
    assert first_data["checkpoint_updated"] is True
    assert len(calls) == 1

    second = await routes.api_semantic_consolidate(_Request({}), {}, set())
    second_data = json.loads(second.text)
    assert second.status == 200
    assert second_data["sources_discovered"] == 0
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_semantic_endpoint_dry_run_does_not_write_checkpoint(monkeypatch, tmp_path):
    source_dir = tmp_path / "wiki" / "entities"
    source_dir.mkdir(parents=True)
    (source_dir / "daily.md").write_text("A source", encoding="utf-8")
    config = _config()
    ctx = SimpleNamespace(
        config=SimpleNamespace(id="core", path=str(tmp_path), settings=config),
        semantic_lock=asyncio.Lock(),
        event_sequence=0,
    )
    monkeypatch.setattr(routes, "_resolve_vault_ctx", lambda _state, _vault: (ctx, None))
    monkeypatch.setattr(routes, "broadcast_activation", _noop_broadcast)

    async def fake_source(source, ctx, ws_clients, *, dry_run, max_concepts):
        assert dry_run is True
        return {"path": source["path"], "new_concepts": [], "hebbian_updates": 0}

    monkeypatch.setattr(routes, "_run_semantic_source", fake_source)
    response = await routes.api_semantic_consolidate(_Request({"dry_run": True}), {}, set())
    data = json.loads(response.text)
    assert response.status == 200
    assert data["dry_run"] is True
    assert data["checkpoint_updated"] is False
    assert not (tmp_path / ".checkpoint.json").exists()


@pytest.mark.asyncio
async def test_semantic_endpoint_can_process_filtered_delta_without_sessions(monkeypatch, tmp_path):
    config = _config(semantic_consolidation_session_enabled=True)
    ctx = SimpleNamespace(
        config=SimpleNamespace(id="core", path=str(tmp_path), settings=config),
        semantic_lock=asyncio.Lock(),
        event_sequence=0,
    )
    monkeypatch.setattr(routes, "_resolve_vault_ctx", lambda _state, _vault: (ctx, None))
    monkeypatch.setattr(routes, "broadcast_activation", _noop_broadcast)
    captured = {}

    def fake_notes(root, cfg, checkpoint, *, max_sources):
        captured["notes_config"] = cfg
        return []

    def fake_sessions(cfg, checkpoint, *, max_sessions):
        captured["sessions_config"] = cfg
        return []

    monkeypatch.setattr(routes, "select_candidate_notes", fake_notes)
    monkeypatch.setattr(routes, "select_candidate_sessions", fake_sessions)
    response = await routes.api_semantic_consolidate(
        _Request({
            "session_enabled": False,
            "source_globs": ["memory/learned/bdh-session-recovery-delta.md"],
            "max_age_hours": 48,
            "max_sources": 3,
            "dry_run": True,
        }),
        {},
        set(),
    )
    data = json.loads(response.text)
    assert response.status == 200
    assert data["sources_discovered"] == 0
    assert captured["notes_config"]["semantic_consolidation_source_globs"] == [
        "memory/learned/bdh-session-recovery-delta.md"
    ]
    assert captured["sessions_config"]["semantic_consolidation_session_enabled"] is False


async def _noop_broadcast(*_args, **_kwargs):
    return None
