"""End-to-end: room_synthesis is accepted and Curate-gated through HTTP.

Posts a room_synthesis request the way the room watcher does, against the real
aiohttp app, and asserts the pre-write gate held: the request is accepted, no
vault note is created, and a candidate is staged as ``pending_review`` with the
room source attached so the Curate correlation check can match it at apply time.

An unregistered source is asserted to still be rejected: registering room
synthesis must widen the gate, not remove it.
"""
import tempfile

import chromadb
import pytest
from aiohttp.test_utils import TestClient, TestServer

import bdh_graph_harness.api.routes as bdh_routes
import bdh_graph_harness.retrieval.attention as bdh_attention_mod

TRANSCRIPT_SHA = "d" * 64


@pytest.fixture
def app_setup(monkeypatch):
    """Real app against a throwaway vault with staging enabled."""
    vault = tempfile.mkdtemp()
    nodes = {
        "alpha": {"id": "alpha", "title": "Alpha", "tags": "concept",
                  "text": "Alpha content", "path": "/fake/alpha.md"},
        "beta": {"id": "beta", "title": "Beta", "tags": "concept",
                 "text": "Beta content", "path": "/fake/beta.md"},
    }
    edges = {"alpha": [{"target": "beta", "display": "beta"}]}
    state = {"synapses": {}, "created": "2026-01-01T00:00:00",
             "updated": "2026-01-01T00:00:00", "queries": 0}

    collection = chromadb.EphemeralClient().get_or_create_collection(
        "test_room_synthesis_e2e", metadata={"hnsw:space": "cosine"}
    )
    if collection.count() > 0:
        collection.delete(ids=collection.get()["ids"])
    collection.add(
        ids=["alpha", "beta"],
        embeddings=[[1.0, 0.0, 0.0], [0.9, 0.1, 0.0]],
        documents=["Alpha content", "Beta content"],
        metadatas=[{"title": "Alpha", "tags": "concept"},
                   {"title": "Beta", "tags": "concept"}],
    )

    import harness
    config = dict(harness.CONFIG)
    config["vault_path"] = vault
    config["neurogenesis_enabled"] = False
    # Staging on, one candidate, no model call.
    config["session_synthesis_staging_enabled"] = True
    config["session_synthesis_staging_max_concepts"] = 1
    config["llm_fallbacks"] = []

    monkeypatch.setattr(bdh_attention_mod, "get_embeddings",
                        lambda texts: [[1.0, 0.0, 0.0]])
    monkeypatch.setattr(bdh_routes, "llm_respond",
                        lambda q, a, n, **kwargs: "Mock LLM response")
    monkeypatch.setattr(bdh_routes, "save_state", lambda vr, s: None)
    monkeypatch.setattr(bdh_routes, "extract_new_concepts",
                        lambda r, q, a, n, **kwargs: [])
    # Deterministic extraction; no provider involved.
    from bdh_graph_harness.memory import session_synthesis_staging as staging
    monkeypatch.setattr(
        staging, "_extract_concepts_safely",
        lambda *a, **k: [{
            "title": "Room Agreed Concept",
            "definition": "A durable concept that emerged across the room.",
            "confidence": "high",
        }],
    )

    captured = {}
    monkeypatch.setattr("aiohttp.web.run_app",
                        lambda app, **kwargs: captured.update(app=app))
    harness.start_api_server(config, nodes, edges, collection, state)
    return captured["app"], vault


def _room_request(source="room_synthesis", vault_id=None):
    body = {
        "query": "Synthesis of an entire group chat room.",
        "user_prompt": "USER: how should the gate behave?\nMEMBER: by actor, not topic.",
        "source": source,
        "metadata": {
            "session_id": "room-abc",
            "synthesis_id": "11111111-2222-3333-4444-555555555555",
            "transcript_sha256": TRANSCRIPT_SHA,
            "queued_at": 1.0,
        },
    }
    if vault_id is not None:
        body["vault_id"] = vault_id
    return body


@pytest.mark.asyncio
async def test_room_synthesis_is_accepted_and_curate_gated(app_setup):
    app, vault = app_setup
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        resp = await client.post("/api/query", json=_room_request())
        assert resp.status == 200, await resp.text()
    finally:
        await client.close()

    import json
    import pathlib
    staged = []
    candidates_dir = pathlib.Path(vault) / ".bdh-candidates"
    for path in sorted(candidates_dir.glob("*.json")):
        staged.append(json.loads(path.read_text(encoding="utf-8")))

    assert staged, "room synthesis must stage a Curate candidate"
    for data in staged:
        assert data["source"] == "room_synthesis"
        assert data["status"] == "pending_review"
        assert data["session_id"] == "room-abc"
        assert data["transcript_sha256"] == TRANSCRIPT_SHA
    # The pre-write gate held: no note was materialised.
    assert not list(pathlib.Path(vault).rglob("*.md"))


@pytest.mark.asyncio
async def test_unknown_source_is_still_rejected(app_setup):
    app, _ = app_setup
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        resp = await client.post("/api/query", json=_room_request(source="not_a_source"))
        assert resp.status == 400
    finally:
        await client.close()
