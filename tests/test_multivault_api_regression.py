"""Regression tests for per-vault API routing and async query isolation."""

import asyncio
import threading
from types import SimpleNamespace

import pytest
from aiohttp.test_utils import make_mocked_request

from bdh_graph_harness.api import routes
from bdh_graph_harness.graph.federated import project_runtime_state_to_persisted
from bdh_graph_harness.memory.hebbian import encode_synapse_key
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
    assert updates == [{
        "note_a": "note", "note_b": "note", "weight": 0.8, "frequency": 1,
    }]
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


def make_federated_context():
    """A canonical runtime backed by a raw legacy federated state."""
    ctx = make_context("federated")
    first = "vault:wiki/first.md"
    second = "vault:wiki/second.md"
    raw_key = "wiki/first|wiki/second"
    canonical_key = encode_synapse_key(first, second)
    synapse = {"weight": 0.8, "frequency": 3, "last_coactivated": None}
    ctx.nodes = {
        first: {"title": "First", "text": "first", "tags": "test"},
        second: {"title": "Second", "text": "second", "tags": "test"},
    }
    ctx.edges = {first: [], second: []}
    ctx.state = {"synapses": {canonical_key: dict(synapse)}, "queries": 2}
    ctx.persisted_state = {"synapses": {raw_key: dict(synapse)}, "queries": 2}
    return ctx, raw_key, canonical_key


def test_federated_projection_preserves_raw_key_absent_from_runtime():
    ctx, raw_key, _ = make_federated_context()
    assert ctx.persisted_state is not None

    projected = project_runtime_state_to_persisted(
        ctx.persisted_state, {"synapses": {}, "queries": 3}, ctx.nodes,
    )

    assert raw_key in projected["synapses"]
    assert projected["queries"] == 3


@pytest.mark.asyncio
async def test_federated_plasticity_persists_raw_legacy_synapse_keys(monkeypatch):
    ctx, raw_key, canonical_key = make_federated_context()
    saved = []

    monkeypatch.setattr(
        routes, "attention",
        lambda *args, **kwargs: {
            "vault:wiki/first.md": 0.9,
            "vault:wiki/second.md": 0.8,
        },
    )
    monkeypatch.setattr(routes, "save_state", lambda _path, state, **_kwargs: saved.append(state.copy()))
    monkeypatch.setattr(routes, "broadcast_activation", lambda *_args: asyncio.sleep(0))

    await routes.run_attention_and_plasticity("federated query", ctx, set())

    assert canonical_key in ctx.state["synapses"]
    assert raw_key in saved[-1]["synapses"]
    assert canonical_key not in saved[-1]["synapses"]


@pytest.mark.asyncio
async def test_federated_consolidation_persists_raw_legacy_synapse_keys(monkeypatch):
    ctx, raw_key, canonical_key = make_federated_context()
    saved = []

    async def json_body():
        return {"vault_id": "federated"}

    monkeypatch.setattr(routes, "save_state", lambda _path, state, **_kwargs: saved.append(state.copy()))
    monkeypatch.setattr(routes, "broadcast_activation", lambda *_args: asyncio.sleep(0))

    response = await routes.api_consolidate(
        SimpleNamespace(json=json_body), {"registry": make_registry(ctx)}, set(),
    )

    assert response.status == 200
    assert canonical_key in ctx.state["synapses"]
    assert raw_key in saved[-1]["synapses"]
    assert canonical_key not in saved[-1]["synapses"]


@pytest.mark.asyncio
async def test_embedding_refresh_waits_for_vault_runtime_lock(monkeypatch):
    ctx = make_context("research")
    started = threading.Event()

    class Collection:
        def count(self):
            return 1

    def compute_embeddings(*_args, **_kwargs):
        started.set()
        return Collection()

    import bdh_graph_harness.retrieval as retrieval
    monkeypatch.setattr(retrieval, "compute_all_embeddings", compute_embeddings)

    async def json_body():
        return {"vault_id": "research"}

    await ctx.runtime_lock.acquire()
    task = asyncio.create_task(
        routes.api_refresh(SimpleNamespace(json=json_body), {"registry": make_registry(ctx)})
    )
    try:
        await asyncio.sleep(0.05)
        assert not started.is_set(), "embedding build ran while graph context was locked"
    finally:
        ctx.runtime_lock.release()

    response = await task
    assert response.status == 200
    assert started.is_set()
