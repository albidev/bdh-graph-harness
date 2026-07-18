"""Regression tests for per-vault API routing and async query isolation."""

import asyncio
from types import SimpleNamespace

import pytest
from aiohttp.test_utils import make_mocked_request

from bdh_graph_harness.api import routes
from bdh_graph_harness.vaults import VaultConfig, VaultContext, VaultRegistry


def make_context(vault_id="research"):
    config = VaultConfig(
        id=vault_id,
        name=vault_id.title(),
        path=f"/tmp/{vault_id}",
        chroma_path=f"/tmp/{vault_id}/chroma",
        chroma_collection=f"vault_{vault_id}_notes",
        settings={"online_plasticity": True, "neurogenesis_enabled": False},
    )
    return VaultContext(
        config=config,
        nodes={"note": {"title": "Scoped note", "text": "content", "tags": "test"}},
        edges={"note": []},
        collection=object(),
        state={"synapses": {}, "queries": 2},
    )


def make_registry(ctx):
    registry = VaultRegistry.__new__(VaultRegistry)
    registry._vault_configs = [ctx.config]
    registry._contexts = {ctx.config.id: ctx}
    registry._default_id = ctx.config.id
    return registry


def test_resolve_vault_returns_clear_400_for_unknown_id():
    ctx = make_context()
    resolved, error = routes._resolve_vault_ctx({"registry": make_registry(ctx)}, "missing")

    assert resolved is None
    assert error.status == 400
    assert 'missing' in error.text
    assert 'research' in error.text


def test_resolve_vault_returns_500_without_registry():
    resolved, error = routes._resolve_vault_ctx({}, "anything")

    assert resolved is None
    assert error.status == 500


def test_vault_id_helpers_ignore_empty_values():
    assert routes._vault_id_from_query(make_mocked_request("GET", "/?vault_id=")) is None
    assert routes._vault_id_from_body({"vault_id": ""}) is None
    assert routes._vault_id_from_body({"vault_id": "research"}) == "research"


@pytest.mark.asyncio
async def test_attention_read_only_does_not_mutate_hebbian_state(monkeypatch):
    ctx = make_context("research")
    calls = []

    monkeypatch.setattr(routes, "attention", lambda *args, **kwargs: {"note": 0.9})
    monkeypatch.setattr(
        routes, "hebbian_update", lambda *args: calls.append(args) or pytest.fail("learn=false called Hebbian update")
    )
    monkeypatch.setattr(routes, "broadcast_activation", lambda *_args: asyncio.sleep(0))

    active, notes, updates, routing = await routes.run_attention_and_plasticity(
        "technical query", ctx, set(), source="automatic_retrieval", learn=False
    )

    assert active == {"note": 0.9}
    assert notes[0]["title"] == "Scoped note"
    assert updates == []
    assert calls == []
    assert ctx.state == {"synapses": {}, "queries": 2}


@pytest.mark.asyncio
async def test_attention_and_plasticity_only_mutates_resolved_vault(monkeypatch):
    ctx = make_context("research")
    calls = []

    def fake_attention(query, nodes, edges, collection, *args, **kwargs):
        assert nodes is ctx.nodes
        assert collection is ctx.collection
        return {"note": 0.9}

    def fake_hebbian(active, state, nodes, source):
        calls.append((active, state, nodes, source))
        state["synapses"]["note|note"] = {"weight": 0.8, "frequency": 1}
        return state, {"note|note"}, 0

    async def no_broadcast(event, clients):
        assert event["vault_id"] == "research"
        assert clients == set()

    monkeypatch.setattr(routes, "attention", fake_attention)
    monkeypatch.setattr(routes, "hebbian_update", fake_hebbian)
    monkeypatch.setattr(routes, "save_state", lambda *_args: None)
    monkeypatch.setattr(routes, "broadcast_activation", no_broadcast)

    active, notes, updates, routing = await routes.run_attention_and_plasticity(
        "scoped query", ctx, set(), source="assistant_response"
    )

    assert active == {"note": 0.9}
    assert notes == [{"id": "note", "title": "Scoped note", "score": 0.9}]
    assert updates == [{"pair": "note|note", "weight": 0.8, "frequency": 1}]
    assert calls[0][3] == "assistant_response"


@pytest.mark.asyncio
async def test_neurogenesis_is_scoped_to_context_path(monkeypatch):
    ctx = make_context("research")
    ctx.config.settings["neurogenesis_enabled"] = True
    ctx.config.settings["neurogenesis_dir"] = "research-concepts"
    monkeypatch.setattr(routes, "extract_new_concepts", lambda *_args: [
        {"title": "New concept", "definition": "A definition"},
        {"title": "", "definition": "ignored"},
    ])
    captured = []

    def fake_create(path, title, definition, source_notes, query, **kwargs):
        captured.append((path, title, definition, source_notes, query, kwargs))
        return "wiki/concepts/new-concept"

    monkeypatch.setattr(routes, "create_note", fake_create)

    result = routes.run_neurogenesis("response", "query", {"note": 0.8}, ctx)

    assert captured == [(
        "/tmp/research", "New concept", "A definition", ["Scoped note"],
        "query", {"neurogenesis_dir": "research-concepts", "source_node_ids": ["note"]},
    )]
    assert result == [{
        "id": "wiki/concepts/new-concept",
        "title": "New concept",
        "source_notes": ["Scoped note"],
    }]
