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
    def test_dry_run_returns_candidates_without_files(self, tmp_path, monkeypatch):
        # Stub the live extractor so the dry-run count is deterministic and
        # does not depend on a running Ollama/LLM endpoint (CI is offline).
        monkeypatch.setattr(
            staging,
            "extract_new_concepts",
            lambda *a, **k: [
                {"title": "Sparse Attention", "definition": "A technique.", "confidence": "low"},
                {"title": "KV Caching", "definition": "A technique.", "confidence": "low"},
            ],
        )
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

    def test_placeholder_and_duplicate_loop_concepts_are_filtered(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            staging,
            "extract_new_concepts",
            lambda *a, **k: [
                {"title": "A Hermes", "definition": "A durable concept related to Hermes extracted from session synthesis."},
                {"title": "A Hermes", "definition": "A durable concept related to Hermes extracted from session synthesis."},
                {"title": "Real concept", "definition": "A useful durable concept from the actual discussion."},
                {"title": "Real concept", "definition": "A useful durable concept from the actual discussion."},
            ],
        )
        result = staging.stage_session_synthesis_candidates(
            str(tmp_path), synthesis_id="syn-filter", vault_id="v", session_id="s",
            transcript_sha256=SHA, response_text="discussion", query="session synthesis",
            active={}, nodes={}, dry_run=False,
        )
        assert result["count"] == 1
        assert result["filtered_count"] == 3
        data = json.loads(next((tmp_path / ".bdh-candidates").glob("*.json")).read_text())
        assert data["title"] == "Real concept"
        assert "extracted from session synthesis" not in data["definition"]

    def test_only_placeholder_concepts_do_not_enter_curate(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            staging,
            "extract_new_concepts",
            lambda *a, **k: [{"title": "A Hermes", "definition": "A durable concept related to Hermes extracted from session synthesis."}],
        )
        result = staging.stage_session_synthesis_candidates(
            str(tmp_path), synthesis_id="syn-placeholder", vault_id="v", session_id="s",
            transcript_sha256=SHA, response_text="discussion", query="session synthesis",
            active={}, nodes={}, dry_run=False,
        )
        assert result["count"] == 0
        assert result["filtered_count"] == 1
        assert not (tmp_path / ".bdh-candidates").exists()

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

    def test_repeated_synthesis_correlation_is_idempotent(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            staging,
            "extract_new_concepts",
            lambda *a, **k: [{"title": "Durable Epoch", "definition": "A persisted synthesis unit.", "confidence": "low"}],
        )
        kwargs = {
            "synthesis_id": "syn-idempotent",
            "vault_id": "vault-1",
            "session_id": "sess-1",
            "transcript_sha256": SHA,
            "response_text": "Durable Epoch is a persisted synthesis unit.",
            "query": "session synthesis",
            "active": {},
            "nodes": {},
            "dry_run": False,
        }
        first = staging.stage_session_synthesis_candidates(str(tmp_path), **kwargs)
        second = staging.stage_session_synthesis_candidates(str(tmp_path), **kwargs)

        assert first["count"] == 1
        assert second["idempotent"] is True
        assert second["candidates"] == first["candidates"]
        assert len(list((tmp_path / ".bdh-candidates").glob("*.json"))) == 1
        assert len(ca.read_curate_audit(str(tmp_path))) == 1

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

class TestStagedDuplicateSuppression:
    """A live session re-stages the same concept on every run.

    ``transcript_sha256`` changes as the conversation grows, so the synthesis-id
    idempotency cannot catch it, and the vault-level dedupe only sees notes that
    were already promoted. The staged queue is where the copies pile up.
    """

    @staticmethod
    def _concepts(monkeypatch, concepts):
        monkeypatch.setattr(staging, "extract_new_concepts", lambda *a, **k: concepts)

    def _stage(self, tmp_path, *, synthesis_id, sha, concepts):
        return staging.stage_session_synthesis_candidates(
            str(tmp_path),
            synthesis_id=synthesis_id,
            vault_id="vault-1",
            session_id="sess-live",
            transcript_sha256=sha,
            response_text="We discussed the concept.",
            query="session synthesis",
            active={},
            nodes={},
            dry_run=False,
        )

    def test_same_concept_from_a_later_run_is_not_staged_twice(self, tmp_path, monkeypatch):
        concept = {"title": "Action Beats Injection", "definition": "First phrasing.", "confidence": "low"}
        self._concepts(monkeypatch, [concept])
        first = self._stage(tmp_path, synthesis_id="syn-1", sha="a" * 64, concepts=[concept])
        assert first["count"] == 1
        assert first["duplicate_count"] == 0

        # Second run of the SAME live session: new transcript digest, new synthesis id,
        # and the extractor rephrases the definition — all three of which the old
        # checks treated as "new".
        again = {"title": "Action Beats Injection", "definition": "Rephrased by the model.", "confidence": "low"}
        self._concepts(monkeypatch, [again])
        second = self._stage(tmp_path, synthesis_id="syn-2", sha="b" * 64, concepts=[again])

        assert second["count"] == 0, "the same concept must not be staged again"
        assert second["duplicate_count"] == 1
        assert len(staging.list_candidates(str(tmp_path))) == 1, "the ledger keeps one row"

    def test_a_different_concept_is_still_staged(self, tmp_path, monkeypatch):
        one = {"title": "Action Beats Injection", "definition": "A principle.", "confidence": "low"}
        two = {"title": "Sparse Attention", "definition": "A separate technique.", "confidence": "low"}
        self._concepts(monkeypatch, [one])
        self._stage(tmp_path, synthesis_id="syn-1", sha="a" * 64, concepts=[one])
        self._concepts(monkeypatch, [two])
        second = self._stage(tmp_path, synthesis_id="syn-2", sha="b" * 64, concepts=[two])

        assert second["count"] == 1, "an unrelated concept must survive dedupe"
        assert second["duplicate_count"] == 0
        assert len(staging.list_candidates(str(tmp_path))) == 2

    def test_two_distinct_concepts_in_one_run_are_both_kept(self, tmp_path, monkeypatch):
        a = {"title": "Write Path Dampening", "definition": "A.", "confidence": "low"}
        b = {"title": "Read Path Independence", "definition": "B.", "confidence": "low"}
        self._concepts(monkeypatch, [a, b])
        result = self._stage(tmp_path, synthesis_id="syn-1", sha="a" * 64, concepts=[a, b])

        assert result["count"] == 2, "dedupe must not collapse distinct concepts in one run"
        assert result["duplicate_count"] == 0

    def test_token_order_and_punctuation_do_not_hide_a_duplicate(self, tmp_path, monkeypatch):
        first = {"title": "Skill-Graph Separation", "definition": "A.", "confidence": "low"}
        self._concepts(monkeypatch, [first])
        self._stage(tmp_path, synthesis_id="syn-1", sha="a" * 64, concepts=[first])

        reordered = {"title": "separation, skill graph", "definition": "B.", "confidence": "low"}
        self._concepts(monkeypatch, [reordered])
        second = self._stage(tmp_path, synthesis_id="syn-2", sha="b" * 64, concepts=[reordered])

        assert second["duplicate_count"] == 1, "same token set in another order is the same concept"

    def test_exact_definition_match_is_not_required(self, tmp_path, monkeypatch):
        """The extractor rephrases every run; requiring definition agreement matched nothing."""
        first = {"title": "Canonical Bot Chat Identity", "definition": "Original wording.", "confidence": "low"}
        self._concepts(monkeypatch, [first])
        self._stage(tmp_path, synthesis_id="syn-1", sha="a" * 64, concepts=[first])

        reworded = {"title": "Canonical Bot Chat Identity", "definition": "Completely different words here.", "confidence": "low"}
        self._concepts(monkeypatch, [reworded])
        second = self._stage(tmp_path, synthesis_id="syn-2", sha="b" * 64, concepts=[reworded])

        assert second["duplicate_count"] == 1

    def test_an_applied_candidate_does_not_block_a_restaged_concept(self, tmp_path, monkeypatch):
        """Only concepts still awaiting a decision absorb a duplicate."""
        from bdh_graph_harness.memory.session_synthesis_staging import update_candidate_status

        concept = {"title": "Launchd Keepalive Sigterm Limitation", "definition": "A.", "confidence": "low"}
        self._concepts(monkeypatch, [concept])
        first = self._stage(tmp_path, synthesis_id="syn-1", sha="a" * 64, concepts=[concept])
        cid = first["candidates"][0]["candidate_id"]
        update_candidate_status(str(tmp_path), cid, "applied")

        self._concepts(monkeypatch, [concept])
        second = self._stage(tmp_path, synthesis_id="syn-2", sha="b" * 64, concepts=[concept])

        assert second["count"] == 1, "an already-applied concept may legitimately be re-observed"
        assert second["duplicate_count"] == 0

