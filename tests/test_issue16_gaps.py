"""Focused regression tests for issue #16 acceptance gaps.

Gap 1 — Note provenance: create_note carries generation_source;
         assimilate_evidence carries provenance in evidence section.
Gap 2 — Policy-controlled neurogenesis: run_neurogenesis honours
         source_policy.allow_neurogenesis instead of hard-coding 'cron'.
Gap 3 — Streaming metadata: api_stream exposes source/source_policy
         routing metadata identical to api_query.
"""
import json
import os
import tempfile
from types import SimpleNamespace

import pytest
import bdh_graph_harness.api.routes as routes
import bdh_graph_harness.api.routes as bdh_routes
import bdh_graph_harness.neurogenesis.creator as creator
from bdh_graph_harness.neurogenesis.merge import assimilate_evidence
from bdh_graph_harness.memory.source_policy import get_source_policy


# ===========================================================================
# Gap 1 — Note provenance
# ===========================================================================

class TestNoteProvenance:
    """create_note must persist generation_source in frontmatter."""

    def test_create_note_includes_generation_source_when_source_provided(self, tmp_path):
        note_id = creator.create_note(
            tmp_path, "Provenance Test", "A durable concept.",
            ["Source A"], "test query",
            source="session_synthesis",
        )
        assert note_id is not None
        with open(os.path.join(tmp_path, note_id + ".md"), encoding="utf-8") as f:
            content = f.read()
        assert 'generation_source: "session_synthesis"' in content

    def test_create_note_defaults_generation_source_to_interactive_query(self, tmp_path):
        """When no source is passed (None), generation_source = interactive_query."""
        note_id = creator.create_note(
            tmp_path, "Default Source", "A concept.",
            ["Source A"], "test query",
        )
        assert note_id is not None
        with open(os.path.join(tmp_path, note_id + ".md"), encoding="utf-8") as f:
            content = f.read()
        assert 'generation_source: "interactive_query"' in content

    def test_create_note_preserves_source_policy_label_for_cron(self, tmp_path):
        note_id = creator.create_note(
            tmp_path, "Cron Note", "Definition.",
            ["cron"], "cron query",
            source="cron",
        )
        assert note_id is not None
        with open(os.path.join(tmp_path, note_id + ".md"), encoding="utf-8") as f:
            content = f.read()
        assert 'generation_source: "cron"' in content

    def test_create_note_backward_compat_no_source_arg(self, tmp_path):
        """Old callers that don't pass source still work."""
        note_id = creator.create_note(
            tmp_path, "Old Style", "Definition.",
            ["Source"], "query",
        )
        assert note_id is not None
        with open(os.path.join(tmp_path, note_id + ".md"), encoding="utf-8") as f:
            content = f.read()
        # Default generation_source should be present
        assert "generation_source:" in content


class TestMergeProvenance:
    """assimilate_evidence must record provenance in the evidence section."""

    def test_assimilate_evidence_includes_provenance_when_source_provided(self, tmp_path):
        note = tmp_path / "wiki" / "concepts" / "note.md"
        note.parent.mkdir(parents=True)
        note.write_text(
            "---\ntitle: Existing Note\n---\n\nOriginal content.\n",
            encoding="utf-8",
        )
        result = assimilate_evidence(
            tmp_path,
            "wiki/concepts/note",
            {"absolute_path": str(note), "title": "Existing Note"},
            "New evidence here.",
            source="session_synthesis",
        )
        assert result["status"] == "merged"
        content = note.read_text(encoding="utf-8")
        assert "provenance: session_synthesis" in content

    def test_assimilate_evidence_omits_provenance_when_source_none(self, tmp_path):
        note = tmp_path / "wiki" / "concepts" / "note.md"
        note.parent.mkdir(parents=True)
        note.write_text(
            "---\ntitle: Existing Note\n---\n\nOriginal content.\n",
            encoding="utf-8",
        )
        result = assimilate_evidence(
            tmp_path,
            "wiki/concepts/note",
            {"absolute_path": str(note), "title": "Existing Note"},
            "New evidence here.",
        )
        assert result["status"] == "merged"
        content = note.read_text(encoding="utf-8")
        assert "provenance:" not in content

    def test_assimilate_evidence_backward_compat(self, tmp_path):
        """Old callers that don't pass source still work."""
        note = tmp_path / "wiki" / "concepts" / "note.md"
        note.parent.mkdir(parents=True)
        note.write_text(
            "---\ntitle: Note\n---\n\nContent.\n",
            encoding="utf-8",
        )
        result = assimilate_evidence(
            tmp_path,
            "wiki/concepts/note",
            {"absolute_path": str(note), "title": "Note"},
            "Evidence.",
            source_notes=["A"],
            source_node_ids=["vault:a.md"],
            query="q",
        )
        assert result["status"] == "merged"


# ===========================================================================
# Gap 2 — Policy-controlled neurogenesis
# ===========================================================================

class TestPolicyControlledNeurogenesis:
    """run_neurogenesis must honour source_policy.allow_neurogenesis."""

    def test_cron_blocked_by_policy(self, monkeypatch, tmp_path):
        """cron has allow_neurogenesis=False — must skip."""
        called = False

        def extractor(*args, **kwargs):
            nonlocal called
            called = True
            return [{"title": "Should Not Exist", "definition": "No."}]

        monkeypatch.setattr(routes, "extract_new_concepts", extractor)
        ctx = SimpleNamespace(
            config=SimpleNamespace(
                settings={"neurogenesis_enabled": True, "neurogenesis_max_concepts": 1},
                path=str(tmp_path),
            ),
            nodes={},
        )
        result = routes.run_neurogenesis("response", "cron q", {}, ctx, source="cron")
        assert result == []
        assert called is False

    def test_none_source_allows_neurogenesis(self, monkeypatch, tmp_path):
        """None (interactive) defaults to allowing neurogenesis."""
        monkeypatch.setattr(
            routes, "extract_new_concepts",
            lambda *a, **k: [{"title": "Allowed", "definition": "Yes."}],
        )
        monkeypatch.setattr(routes, "find_semantic_match", lambda *a, **k: None)
        monkeypatch.setattr(routes, "looks_conflicting", lambda d: False)
        monkeypatch.setattr(
            routes, "create_note",
            lambda *a, **k: "concepts/allowed",
        )
        ctx = SimpleNamespace(
            config=SimpleNamespace(
                settings={"neurogenesis_enabled": True, "neurogenesis_max_concepts": 1},
                path=str(tmp_path),
            ),
            nodes={},
        )
        result = routes.run_neurogenesis("resp", "query", {}, ctx, source=None)
        assert len(result) == 1

    def test_session_synthesis_allows_neurogenesis(self, monkeypatch, tmp_path):
        """session_synthesis has allow_neurogenesis=True."""
        monkeypatch.setattr(
            routes, "extract_new_concepts",
            lambda *a, **k: [{"title": "Synth Note", "definition": "From session."}],
        )
        monkeypatch.setattr(routes, "find_semantic_match", lambda *a, **k: None)
        monkeypatch.setattr(routes, "looks_conflicting", lambda d: False)
        monkeypatch.setattr(
            routes, "create_note",
            lambda *a, **k: "concepts/synth-note",
        )
        ctx = SimpleNamespace(
            config=SimpleNamespace(
                settings={"neurogenesis_enabled": True, "neurogenesis_max_concepts": 1},
                path=str(tmp_path),
            ),
            nodes={},
        )
        result = routes.run_neurogenesis("resp", "query", {}, ctx, source="session_synthesis")
        assert len(result) == 1

    def test_user_query_allows_neurogenesis(self, monkeypatch, tmp_path):
        """user_query has allow_neurogenesis=True."""
        monkeypatch.setattr(
            routes, "extract_new_concepts",
            lambda *a, **k: [{"title": "User Note", "definition": "From user."}],
        )
        monkeypatch.setattr(routes, "find_semantic_match", lambda *a, **k: None)
        monkeypatch.setattr(routes, "looks_conflicting", lambda d: False)
        monkeypatch.setattr(
            routes, "create_note",
            lambda *a, **k: "concepts/user-note",
        )
        ctx = SimpleNamespace(
            config=SimpleNamespace(
                settings={"neurogenesis_enabled": True, "neurogenesis_max_concepts": 1},
                path=str(tmp_path),
            ),
            nodes={},
        )
        result = routes.run_neurogenesis("resp", "query", {}, ctx, source="user_query")
        assert len(result) == 1

    def test_source_policy_allow_neurogenesis_field_is_consulted(self):
        """Verify the policy fields for known sources match expectations."""
        for name in ("user_query", "assistant_response", "session_synthesis",
                      "nightly_semantic_consolidation", "automatic_retrieval"):
            policy = get_source_policy(name)
            assert policy is not None, f"{name} should be registered"
            assert policy.allow_neurogenesis is True, f"{name} should allow neurogenesis"
        cron_policy = get_source_policy("cron")
        assert cron_policy is not None
        assert cron_policy.allow_neurogenesis is False

    def test_run_neurogenesis_source_passed_to_create_note(self, monkeypatch, tmp_path):
        """run_neurogenesis threads source through to create_note."""
        captured_kwargs = {}

        def capturing_create_note(*args, **kwargs):
            captured_kwargs.update(kwargs)
            return "concepts/test-note"

        monkeypatch.setattr(
            routes, "extract_new_concepts",
            lambda *a, **k: [{"title": "Test Note", "definition": "Def."}],
        )
        monkeypatch.setattr(routes, "find_semantic_match", lambda *a, **k: None)
        monkeypatch.setattr(routes, "looks_conflicting", lambda d: False)
        monkeypatch.setattr(routes, "create_note", capturing_create_note)
        ctx = SimpleNamespace(
            config=SimpleNamespace(
                settings={"neurogenesis_enabled": True, "neurogenesis_max_concepts": 1},
                path=str(tmp_path),
            ),
            nodes={},
        )
        routes.run_neurogenesis("resp", "q", {}, ctx, source="session_synthesis")
        assert captured_kwargs.get("source") == "session_synthesis"

    def test_run_neurogenesis_source_passed_to_assimilate_evidence(self, monkeypatch, tmp_path):
        """run_neurogenesis threads source through to assimilate_evidence."""
        note = tmp_path / "wiki" / "concepts" / "canonical.md"
        note.parent.mkdir(parents=True)
        note.write_text("---\ntitle: Canonical\n---\n\nOriginal.\n", encoding="utf-8")
        node = {
            "title": "Canonical",
            "absolute_path": str(note),
            "relative_path": "wiki/concepts/canonical.md",
        }
        ctx = SimpleNamespace(
            nodes={"wiki/concepts/canonical": node},
            config=SimpleNamespace(
                path=str(tmp_path),
                settings={"neurogenesis_enabled": True, "neurogenesis_dir": "wiki/concepts"},
            ),
        )
        monkeypatch.setattr(
            routes, "extract_new_concepts",
            lambda *a, **k: [{"title": "Canonical", "definition": "New evidence."}],
        )
        monkeypatch.setattr(routes, "create_note", lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("should merge, not create"),
        ))

        routes.run_neurogenesis("resp", "q", {}, ctx, source="session_synthesis")
        content = note.read_text(encoding="utf-8")
        assert "provenance: session_synthesis" in content


# ===========================================================================
# Gap 3 — Streaming metadata
# ===========================================================================

class TestStreamMetadata:
    """api_stream must expose source/source_policy in routing."""

    @pytest.mark.asyncio
    async def test_stream_includes_source_policy(self, mock_app_setup, monkeypatch):
        from aiohttp.test_utils import TestClient, TestServer
        import bdh_graph_harness.api.routes as bdh_routes

        nodes, edges, collection, state, config, _ = mock_app_setup
        app = _capture_app(monkeypatch, config, nodes, edges, collection, state)
        server = TestServer(app)
        client = TestClient(server)
        await client.start_server()
        try:
            resp = await client.post("/api/stream", json={
                "query": "stream test",
                "source": "session_synthesis",
                "user_prompt": "actual transcript",
            })
            assert resp.status == 200
            body = await resp.text()
            # First SSE message should be the activation event with routing
            lines = body.strip().split("\n")
            first_data_line = next(line for line in lines if line.startswith("data: ") and "[DONE]" not in line)
            payload = json.loads(first_data_line[len("data: "):])
            assert payload["type"] == "activation"
            routing = payload["routing"]
            assert routing["source"] == "session_synthesis"
            sp = routing["source_policy"]
            assert sp["frequency_increment"] == pytest.approx(0.2)
            assert sp["provenance_label"] == "session_synthesis"
            assert sp["use_user_prompt_for_retrieval"] is True
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_stream_default_source_includes_source_policy(self, mock_app_setup, monkeypatch):
        from aiohttp.test_utils import TestClient, TestServer

        nodes, edges, collection, state, config, _ = mock_app_setup
        app = _capture_app(monkeypatch, config, nodes, edges, collection, state)
        server = TestServer(app)
        client = TestClient(server)
        await client.start_server()
        try:
            resp = await client.post("/api/stream", json={
                "query": "default stream",
            })
            assert resp.status == 200
            body = await resp.text()
            lines = body.strip().split("\n")
            first_data_line = next(line for line in lines if line.startswith("data: ") and "[DONE]" not in line)
            payload = json.loads(first_data_line[len("data: "):])
            routing = payload["routing"]
            assert "source" in routing
            assert "source_policy" in routing
            sp = routing["source_policy"]
            assert sp["frequency_increment"] == pytest.approx(1.0)
            assert sp["use_user_prompt_for_retrieval"] is False
        finally:
            await client.close()


# -----------------------------------------------------------------------
# Test fixtures reused from test_session_synthesis_contract.py
# -----------------------------------------------------------------------

import asyncio
import chromadb
import bdh_graph_harness.retrieval.attention as bdh_attention_mod


@pytest.fixture
def mock_app_setup(monkeypatch):
    """Create a mock app setup: nodes, edges, collection, state, config."""
    d = tempfile.mkdtemp()

    nodes = {
        "alpha": {"id": "alpha", "title": "Alpha", "tags": "concept", "text": "Alpha content", "path": "/fake/alpha.md"},
        "beta": {"id": "beta", "title": "Beta", "tags": "concept", "text": "Beta content", "path": "/fake/beta.md"},
        "gamma": {"id": "gamma", "title": "Gamma", "tags": "concept", "text": "Gamma content", "path": "/fake/gamma.md"},
    }
    edges = {
        "alpha": [{"target": "beta", "display": "beta"}],
        "beta": [{"target": "gamma", "display": "gamma"}],
    }
    state = {
        "synapses": {},
        "created": "2026-01-01T00:00:00",
        "updated": "2026-01-01T00:00:00",
        "queries": 0,
    }

    client = chromadb.EphemeralClient()
    collection = client.get_or_create_collection("test_gap16", metadata={"hnsw:space": "cosine"})
    if collection.count() > 0:
        collection.delete(ids=collection.get()["ids"])
    collection.add(
        ids=["alpha", "beta", "gamma"],
        embeddings=[[1.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.0, 1.0, 0.0]],
        documents=["Alpha content", "Beta content", "Gamma content"],
        metadatas=[
            {"title": "Alpha", "tags": "concept"},
            {"title": "Beta", "tags": "concept"},
            {"title": "Gamma", "tags": "concept"},
        ],
    )

    import harness
    config = dict(harness.CONFIG)
    config["vault_path"] = d
    config["neurogenesis_enabled"] = False

    monkeypatch.setattr(bdh_attention_mod, "get_embeddings", lambda texts: [[1.0, 0.0, 0.0]])
    monkeypatch.setattr(bdh_routes, "llm_respond", lambda q, a, n, **kwargs: "Mock LLM response")
    monkeypatch.setattr(bdh_routes, "llm_stream", lambda q, a, n, **kwargs: iter(["Mock", " ", "stream"]))
    monkeypatch.setattr(bdh_routes, "extract_new_concepts", lambda r, q, a, n, **kwargs: [])
    monkeypatch.setattr(bdh_routes, "save_state", lambda vr, s: None)

    return nodes, edges, collection, state, config, d


def _capture_app(monkeypatch, config, nodes, edges, collection, state):
    """Monkeypatch web.run_app to capture the app without starting a server."""
    import harness
    captured = {}
    from aiohttp import web

    def fake_run_app(app, **kwargs):
        captured["app"] = app
    monkeypatch.setattr("aiohttp.web.run_app", fake_run_app)
    harness.start_api_server(config, nodes, edges, collection, state)
    return captured["app"]
