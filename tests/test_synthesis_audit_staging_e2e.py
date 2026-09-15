"""End-to-end: /api/query must record a STAGED run as `staged`, not `noop`.

Drives the real aiohttp route with staging enabled and the extractor returning
concepts, then reads the audit log. On the previous implementation this recorded
`outcome="noop", concept_ids=[]` — the same row a run that extracted nothing
produced, which made a working pipeline indistinguishable from a dead one.

The extractor is stubbed (no network), but everything else is the production
path: the staging branch, the candidate files, the audit row.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import harness  # noqa: E402
import bdh_graph_harness.api.routes as bdh_routes  # noqa: E402
import bdh_graph_harness.retrieval.attention as bdh_attention_mod  # noqa: E402
from bdh_graph_harness.memory.synthesis_audit import read_synthesis_audit  # noqa: E402


@pytest.fixture
def staged_app(monkeypatch):
    """App with Curate staging ENABLED and an extractor that returns concepts."""
    import tempfile

    import chromadb

    d = tempfile.mkdtemp()
    nodes = {
        "alpha": {"id": "alpha", "title": "Alpha", "tags": "concept",
                  "text": "Alpha content", "path": "/fake/alpha.md"},
        "beta": {"id": "beta", "title": "Beta", "tags": "concept",
                 "text": "Beta content", "path": "/fake/beta.md"},
    }
    edges = {"alpha": [{"target": "beta", "display": "beta"}]}
    state = {"synapses": {}, "created": "2026-01-01T00:00:00",
             "updated": "2026-01-01T00:00:00", "queries": 0}

    client = chromadb.EphemeralClient()
    collection = client.get_or_create_collection(
        "test_audit_staging_e2e", metadata={"hnsw:space": "cosine"}
    )
    if collection.count() > 0:
        collection.delete(ids=collection.get()["ids"])
    collection.add(
        ids=["alpha", "beta"],
        embeddings=[[1.0, 0.0, 0.0], [0.9, 0.1, 0.0]],
        documents=["Alpha content", "Beta content"],
        metadatas=[{"title": "Alpha", "tags": "concept"}, {"title": "Beta", "tags": "concept"}],
    )

    config = dict(harness.CONFIG)
    config["vault_path"] = d
    config["neurogenesis_enabled"] = False
    # The condition under test: staging is the write path.
    config["session_synthesis_staging_enabled"] = True

    monkeypatch.setattr(bdh_attention_mod, "get_embeddings", lambda texts: [[1.0, 0.0, 0.0]])
    monkeypatch.setattr(bdh_routes, "llm_respond", lambda q, a, n, **kw: "Mock LLM response")
    # The staging path calls the extractor from its OWN module namespace, so the
    # stub must land there; patching routes alone leaves the real one running.
    from bdh_graph_harness.memory import session_synthesis_staging as staging
    monkeypatch.setattr(
        staging, "extract_new_concepts",
        lambda *a, **kw: [
            {"title": "Curate Staging Gate", "definition": "Candidates are staged before any note is written."},
            {"title": "Actor Scoped Routing", "definition": "A vault is authorised by actor, never by topic."},
        ],
    )
    # semantic dedupe must not reach a vector store
    monkeypatch.setattr(staging, "find_semantic_match", lambda *a, **kw: None)
    monkeypatch.setattr(bdh_routes, "save_state", lambda vr, s: None)

    captured = {}
    from aiohttp import web

    monkeypatch.setattr("aiohttp.web.run_app", lambda app, **kw: captured.update(app=app))
    harness.start_api_server(config, nodes, edges, collection, state)
    return captured["app"], d


@pytest.mark.asyncio
async def test_api_query_records_staged_when_candidates_are_queued(staged_app):
    """A run that queued candidates is audited as `staged`, not `noop`."""
    app, vault_dir = staged_app
    from aiohttp.test_utils import TestClient, TestServer

    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        resp = await client.post("/api/query", json={
            "query": "synthesis of the room",
            "source": "room_synthesis",
            "vault_id": "default",
            "metadata": {
                "session_id": "sess-staged",
                "synthesis_id": "syn-staged",
                "transcript_sha256": "a" * 64,
            },
        })
        assert resp.status == 200

        entries = read_synthesis_audit(vault_dir)
        assert len(entries) == 1, f"expected one audit row, got {entries}"
        entry = entries[0]

        # The regression: this used to be outcome="noop", staged_count absent.
        assert entry.outcome == "staged", (
            f"a run that queued candidates recorded outcome={entry.outcome!r} — "
            f"indistinguishable from one that extracted nothing"
        )
        assert entry.staged_count >= 1
        # concept_ids stays empty in staging mode, and that is expected
        assert entry.concept_ids == []
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_api_query_records_noop_with_a_reason_when_nothing_extracted(
    monkeypatch,
):
    """A genuinely empty run still records `noop`, now with a stated reason."""
    import tempfile

    import chromadb

    d = tempfile.mkdtemp()
    nodes = {"alpha": {"id": "alpha", "title": "Alpha", "tags": "concept",
                       "text": "A", "path": "/fake/a.md"}}
    edges = {}
    state = {"synapses": {}, "created": "x", "updated": "x", "queries": 0}
    client = chromadb.EphemeralClient()
    collection = client.get_or_create_collection(
        "test_audit_staging_empty", metadata={"hnsw:space": "cosine"}
    )
    if collection.count() > 0:
        collection.delete(ids=collection.get()["ids"])
    collection.add(ids=["alpha"], embeddings=[[1.0, 0.0, 0.0]], documents=["A"],
                   metadatas=[{"title": "Alpha", "tags": "concept"}])

    config = dict(harness.CONFIG)
    config["vault_path"] = d
    config["neurogenesis_enabled"] = False
    config["session_synthesis_staging_enabled"] = True

    monkeypatch.setattr(bdh_attention_mod, "get_embeddings", lambda t: [[1.0, 0.0, 0.0]])
    monkeypatch.setattr(bdh_routes, "llm_respond", lambda q, a, n, **kw: "nothing durable here")
    monkeypatch.setattr(bdh_routes, "extract_new_concepts", lambda r, q, a, n, **kw: [])
    monkeypatch.setattr(bdh_routes, "save_state", lambda vr, s: None)

    captured = {}
    monkeypatch.setattr("aiohttp.web.run_app", lambda app, **kw: captured.update(app=app))
    harness.start_api_server(config, nodes, edges, collection, state)
    app = captured["app"]

    from aiohttp.test_utils import TestClient, TestServer

    c = TestClient(TestServer(app))
    await c.start_server()
    try:
        resp = await c.post("/api/query", json={
            "query": "synthesis",
            "source": "room_synthesis",
            "vault_id": "default",
            "metadata": {
                "session_id": "sess-empty",
                "synthesis_id": "syn-empty",
                "transcript_sha256": "b" * 64,
            },
        })
        assert resp.status == 200

        entries = read_synthesis_audit(d)
        assert len(entries) == 1
        entry = entries[0]
        assert entry.outcome == "noop"
        assert entry.staged_count == 0
        # the log now says WHY, instead of leaving it to guesswork
        assert entry.reason, "an empty run must state a reason"
    finally:
        await c.close()
