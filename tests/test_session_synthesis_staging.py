"""Tests for session_synthesis candidate staging.

Covers:
  1. Dry-run extraction returns candidates without creating files/audit entries.
  2. Non-dry run writes pending candidate files and Curate audit entries.
  3. Candidate metadata contains provenance but never the raw transcript.
  4. No vault note creation, no Hebbian mutation, no merge side effects.
  5. Listing and loading candidates works, including filters.
  6. Invalid transcript_sha256 is rejected.
  7. API endpoints /api/synthesis/stage and /api/synthesis/candidates behave.
"""
import json
import os
from pathlib import Path

import pytest

from bdh_graph_harness.memory import session_synthesis_staging as staging
from bdh_graph_harness.memory import curate_audit as ca


SHA = "a" * 64


class TestStageSessionSynthesisCandidates:
    def test_dry_run_returns_candidates_without_files(self, tmp_path):
        result = staging.stage_session_synthesis_candidates(
            str(tmp_path),
            synthesis_id="syn-1",
            vault_id="vault-1",
            session_id="sess-1",
            transcript_sha256=SHA,
            response_text="We discussed sparse attention and KV caching.",
            query="session synthesis",
            active={},
            nodes={},
            dry_run=True,
        )
        assert result["dry_run"] is True
        assert result["count"] == 2
        assert not (tmp_path / ".bdh-candidates").exists()
        assert not (tmp_path / ".bdh-audit").exists()

    def test_non_dry_run_writes_candidate_files_and_audit(self, tmp_path):
        result = staging.stage_session_synthesis_candidates(
            str(tmp_path),
            synthesis_id="syn-2",
            vault_id="vault-1",
            session_id="sess-1",
            transcript_sha256=SHA,
            response_text="A new concept called Hebbian retrieval boosting.",
            query="session synthesis",
            active={},
            nodes={},
            dry_run=False,
        )
        assert result["dry_run"] is False
        assert result["count"] == 1
        cand_dir = tmp_path / ".bdh-candidates"
        assert cand_dir.is_dir()
        files = list(cand_dir.glob("*.json"))
        assert len(files) == 1
        data = json.loads(files[0].read_text(encoding="utf-8"))
        assert data["synthesis_id"] == "syn-2"
        assert data["status"] == "pending_review"

        audit = ca.read_curate_audit(str(tmp_path))
        assert len(audit) == 1
        assert audit[0].state == "pending_review"
        assert audit[0].candidate_id == data["candidate_id"]

    def test_candidate_has_required_metadata(self, tmp_path):
        result = staging.stage_session_synthesis_candidates(
            str(tmp_path),
            synthesis_id="syn-3",
            vault_id="vault-1",
            session_id="sess-1",
            transcript_sha256=SHA,
            response_text="Durable idea: context-only turns should be retained.",
            query="session synthesis",
            active={},
            nodes={},
            dry_run=False,
        )
        files = list((tmp_path / ".bdh-candidates").glob("*.json"))
        data = json.loads(files[0].read_text(encoding="utf-8"))
        assert data["source"] == "session_synthesis"
        assert data["vault_id"] == "vault-1"
        assert data["synthesis_id"] == "syn-3"
        assert data["session_id"] == "sess-1"
        assert data["transcript_sha256"] == SHA
        assert "definition" in data
        assert "confidence" in data
        assert "provenance" in data

    def test_candidate_never_contains_raw_transcript(self, tmp_path):
        transcript = "this is the raw secret transcript"
        result = staging.stage_session_synthesis_candidates(
            str(tmp_path),
            synthesis_id="syn-4",
            vault_id="vault-1",
            session_id="sess-1",
            transcript_sha256=SHA,
            response_text=f"Concept extracted from: {transcript}",
            query=transcript,
            active={},
            nodes={},
            dry_run=False,
        )
        files = list((tmp_path / ".bdh-candidates").glob("*.json"))
        data = json.loads(files[0].read_text(encoding="utf-8"))
        serialized = json.dumps(data)
        assert transcript not in serialized
        assert SHA in serialized  # hash is present

    def test_no_vault_note_created(self, tmp_path):
        staging.stage_session_synthesis_candidates(
            str(tmp_path),
            synthesis_id="syn-5",
            vault_id="vault-1",
            session_id="sess-1",
            transcript_sha256=SHA,
            response_text="New concept should not create a note yet.",
            query="session synthesis",
            active={},
            nodes={},
            dry_run=False,
        )
        concept_dir = tmp_path / "wiki" / "concepts"
        assert not concept_dir.exists() or not any(concept_dir.glob("*.md"))

    def test_no_hebbian_mutation(self, tmp_path):
        staging.stage_session_synthesis_candidates(
            str(tmp_path),
            synthesis_id="syn-6",
            vault_id="vault-1",
            session_id="sess-1",
            transcript_sha256=SHA,
            response_text="Another durable concept.",
            query="session synthesis",
            active={},
            nodes={},
            dry_run=False,
        )
        state_file = tmp_path / ".bdh-state.json"
        assert not state_file.exists()

    def test_invalid_sha256_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="transcript_sha256"):
            staging.stage_session_synthesis_candidates(
                str(tmp_path),
                synthesis_id="syn-7",
                vault_id="vault-1",
                session_id="sess-1",
                transcript_sha256="not-a-hash",
                response_text="A concept.",
            )


class TestListAndLoadCandidates:
    def test_list_candidates_filters_by_status(self, tmp_path):
        staging.stage_session_synthesis_candidates(
            str(tmp_path),
            synthesis_id="syn-list",
            vault_id="vault-1",
            session_id="sess-1",
            transcript_sha256=SHA,
            response_text="Concept one. Concept two.",
            dry_run=False,
        )
        candidates = staging.list_candidates(str(tmp_path))
        assert len(candidates) >= 1
        first = candidates[0]
        staging.update_candidate_status(str(tmp_path), first.candidate_id, "rejected")
        pending = staging.list_candidates(str(tmp_path), status="pending_review")
        rejected = staging.list_candidates(str(tmp_path), status="rejected")
        assert len(pending) == len(candidates) - 1
        assert len(rejected) == 1

    def test_load_candidate_missing(self, tmp_path):
        assert staging.load_candidate(str(tmp_path), "missing") is None

    def test_list_candidates_filters_by_synthesis(self, tmp_path):
        staging.stage_session_synthesis_candidates(
            str(tmp_path),
            synthesis_id="syn-a",
            vault_id="vault-1",
            session_id="sess-1",
            transcript_sha256=SHA,
            response_text="One concept.",
            dry_run=False,
        )
        staging.stage_session_synthesis_candidates(
            str(tmp_path),
            synthesis_id="syn-b",
            vault_id="vault-1",
            session_id="sess-1",
            transcript_sha256=SHA,
            response_text="Another concept.",
            dry_run=False,
        )
        a_only = staging.list_candidates(str(tmp_path), synthesis_id="syn-a")
        assert len(a_only) == 1
        assert a_only[0].synthesis_id == "syn-a"


class TestStageFromApiResponse:
    def test_derives_source_notes_from_activated_notes(self, tmp_path):
        activated = [
            {"id": "alpha", "title": "Alpha"},
            {"id": "beta", "title": "Beta"},
        ]
        result = staging.stage_from_api_response(
            str(tmp_path),
            synthesis_id="syn-api",
            vault_id="vault-1",
            session_id="sess-1",
            transcript_sha256=SHA,
            response_text="Derived from alpha and beta.",
            query="session synthesis",
            nodes={},
            activated_notes=activated,
            dry_run=False,
        )
        assert result["count"] == 1
        candidate = staging.list_candidates(str(tmp_path))[0]
        assert candidate.provenance["source_notes"] == ["Alpha", "Beta"]
        assert candidate.provenance["source"] == "session_synthesis"
        assert candidate.provenance["activated_note_count"] == 2


class TestStagingApiEndpoints:
    @pytest.mark.asyncio
    async def test_api_stage_dry_run(self, mock_app_setup, monkeypatch):
        from aiohttp.test_utils import TestClient, TestServer
        import harness
        import bdh_graph_harness.api.routes as bdh_routes

        nodes, edges, collection, state, config, d = mock_app_setup
        app = _capture_app(monkeypatch, config, nodes, edges, collection, state)
        server = TestServer(app)
        client = TestClient(server)
        await client.start_server()
        try:
            resp = await client.post('/api/synthesis/stage', json={
                'synthesis_id': 'syn-stage',
                'session_id': 'sess-stage',
                'transcript_sha256': SHA,
                'response_text': 'A new durable concept called Sparse Attention.',
                'dry_run': True,
            })
            assert resp.status == 200
            data = await resp.json()
            assert data['dry_run'] is True
            assert data['count'] >= 1
            assert not (Path(d) / '.bdh-candidates').exists()
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_api_stage_writes_candidate(self, mock_app_setup, monkeypatch):
        from aiohttp.test_utils import TestClient, TestServer
        import harness
        import bdh_graph_harness.api.routes as bdh_routes

        nodes, edges, collection, state, config, d = mock_app_setup
        app = _capture_app(monkeypatch, config, nodes, edges, collection, state)
        server = TestServer(app)
        client = TestClient(server)
        await client.start_server()
        try:
            resp = await client.post('/api/synthesis/stage', json={
                'synthesis_id': 'syn-stage-2',
                'session_id': 'sess-stage-2',
                'transcript_sha256': SHA,
                'response_text': 'Second durable concept called KV Caching.',
                'dry_run': False,
            })
            assert resp.status == 200
            data = await resp.json()
            assert data['dry_run'] is False
            assert data['count'] >= 1
            files = list(Path(d).glob('.bdh-candidates/*.json'))
            assert len(files) >= 1
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_api_candidates_lists_staged(self, mock_app_setup, monkeypatch):
        from aiohttp.test_utils import TestClient, TestServer
        import harness
        import bdh_graph_harness.api.routes as bdh_routes

        nodes, edges, collection, state, config, d = mock_app_setup
        app = _capture_app(monkeypatch, config, nodes, edges, collection, state)
        server = TestServer(app)
        client = TestClient(server)
        await client.start_server()
        try:
            await client.post('/api/synthesis/stage', json={
                'synthesis_id': 'syn-list',
                'session_id': 'sess-list',
                'transcript_sha256': SHA,
                'response_text': 'Listable concept.',
                'dry_run': False,
            })
            resp = await client.get('/api/synthesis/candidates')
            assert resp.status == 200
            data = await resp.json()
            assert data['count'] == 1
            assert data['candidates'][0]['synthesis_id'] == 'syn-list'
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_api_stage_rejects_invalid_sha256(self, mock_app_setup, monkeypatch):
        from aiohttp.test_utils import TestClient, TestServer
        import harness
        import bdh_graph_harness.api.routes as bdh_routes

        nodes, edges, collection, state, config, d = mock_app_setup
        app = _capture_app(monkeypatch, config, nodes, edges, collection, state)
        server = TestServer(app)
        client = TestClient(server)
        await client.start_server()
        try:
            resp = await client.post('/api/synthesis/stage', json={
                'synthesis_id': 'syn-bad',
                'session_id': 'sess-bad',
                'transcript_sha256': 'bad',
                'response_text': 'Concept.',
            })
            assert resp.status == 400
        finally:
            await client.close()


# Fixtures reused from test_api.py style
import asyncio
import tempfile
import chromadb
import bdh_graph_harness.retrieval.attention as bdh_attention_mod


@pytest.fixture
def mock_app_setup(monkeypatch):
    d = tempfile.mkdtemp()

    nodes = {
        'alpha': {'id': 'alpha', 'title': 'Alpha', 'tags': 'concept', 'text': 'Alpha content', 'path': '/fake/alpha.md'},
        'beta': {'id': 'beta', 'title': 'Beta', 'tags': 'concept', 'text': 'Beta content', 'path': '/fake/beta.md'},
        'gamma': {'id': 'gamma', 'title': 'Gamma', 'tags': 'concept', 'text': 'Gamma content', 'path': '/fake/gamma.md'},
    }
    edges = {
        'alpha': [{'target': 'beta', 'display': 'beta'}],
        'beta': [{'target': 'gamma', 'display': 'gamma'}],
    }
    state = {
        'synapses': {},
        'created': '2026-01-01T00:00:00',
        'updated': '2026-01-01T00:00:00',
        'queries': 0,
    }

    client = chromadb.EphemeralClient()
    collection = client.get_or_create_collection('test_staging', metadata={'hnsw:space': 'cosine'})
    if collection.count() > 0:
        collection.delete(ids=collection.get()['ids'])
    collection.add(
        ids=['alpha', 'beta', 'gamma'],
        embeddings=[[1.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.0, 1.0, 0.0]],
        documents=['Alpha content', 'Beta content', 'Gamma content'],
        metadatas=[{'title': 'Alpha', 'tags': 'concept'}, {'title': 'Beta', 'tags': 'concept'}, {'title': 'Gamma', 'tags': 'concept'}],
    )

    import harness
    config = dict(harness.CONFIG)
    config['vault_path'] = d
    config['neurogenesis_enabled'] = False

    monkeypatch.setattr(bdh_attention_mod, 'get_embeddings', lambda texts: [[1.0, 0.0, 0.0]])
    monkeypatch.setattr(bdh_routes, 'llm_respond', lambda q, a, n, **kwargs: 'Mock LLM response')
    monkeypatch.setattr(bdh_routes, 'extract_new_concepts', lambda r, q, a, n, **kwargs: [])
    monkeypatch.setattr(bdh_routes, 'save_state', lambda vr, s: None)

    return nodes, edges, collection, state, config, d


def _capture_app(monkeypatch, config, nodes, edges, collection, state):
    captured = {}
    from aiohttp import web

    def fake_run_app(app, **kwargs):
        captured['app'] = app

    monkeypatch.setattr('aiohttp.web.run_app', fake_run_app)
    import harness
    harness.start_api_server(config, nodes, edges, collection, state)
    return captured['app']


import bdh_graph_harness.api.routes as bdh_routes
