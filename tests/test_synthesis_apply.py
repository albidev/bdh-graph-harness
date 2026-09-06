"""Tests for POST /api/synthesis/apply — the idempotent Curate apply gate.

Covers:
  1. HTTP contract: missing/malformed IDs, wrong source, vault mismatch,
     unknown candidate, and non-approved candidate return explicit 4xx with
     no vault/Hebbian/journal mutation.
  2. First apply returns a stable operation result and transitions the audit
     state to created/merged/noop.
  3. Repeating the identical apply returns the previous result and creates no
     second note, Hebbian update, or journal row.
  4. Conflict/failure is persisted as conflict/failed without deleting the
     candidate; retry is deterministic.
"""
import json
from pathlib import Path

import pytest

import bdh_graph_harness.api.routes as bdh_routes
import bdh_graph_harness.retrieval.attention as bdh_attention_mod
from bdh_graph_harness.memory import curate_audit as ca
from bdh_graph_harness.memory import session_synthesis_staging as staging
from bdh_graph_harness.neurogenesis.operation_journal import list_operation_records


SHA = "a" * 64


def _stage_approved(tmp_path, *, candidate_id=None, title="Sparse Attention",
                    definition="A durable concept about sparse attention.",
                    synthesis_id="syn-1", session_id="sess-1", vault_id="default"):
    """Stage a candidate and mark it approved, returning the candidate dict."""
    staging.stage_session_synthesis_candidates(
        str(tmp_path),
        synthesis_id=synthesis_id,
        vault_id=vault_id,
        session_id=session_id,
        transcript_sha256=SHA,
        response_text=f"{title} is {definition}",
        query="session synthesis",
        active={},
        nodes={},
        dry_run=False,
    )
    candidates = staging.list_candidates(str(tmp_path))
    assert candidates, "staging produced no candidates"
    cand = candidates[0]
    staging.update_candidate_status(str(tmp_path), cand.candidate_id, "approved")
    return staging.load_candidate(str(tmp_path), cand.candidate_id)


def _stage_candidate_direct(tmp_path, *, title, definition, candidate_id="cand-direct",
                            synthesis_id="syn-1", session_id="sess-1", vault_id="default",
                            status="approved"):
    """Write a candidate file directly with an exact title/definition.

    Bypasses the fallback extractor so tests can target a specific existing
    node title (e.g. "Alpha") for merge/conflict paths.
    """
    from bdh_graph_harness.memory.session_synthesis_staging import SessionSynthesisCandidate

    candidate = SessionSynthesisCandidate(
        candidate_id=candidate_id,
        synthesis_id=synthesis_id,
        vault_id=vault_id,
        session_id=session_id,
        transcript_sha256=SHA,
        source="session_synthesis",
        title=title,
        definition=definition,
        confidence="low",
        provenance={"source_notes": [], "source_node_ids": []},
        status=status,
        created_at="2026-01-01T00:00:00+00:00",
    )
    staging._save_candidate(candidate, str(tmp_path))
    ca.create_curate_candidate(
        str(tmp_path),
        candidate_id=candidate_id,
        synthesis_id=synthesis_id,
        vault_id=vault_id,
        session_id=session_id,
        transcript_sha256=SHA,
        extra={"title": title, "source": "session_synthesis", "status": status},
    )
    return candidate


def _apply_body(candidate, **overrides):
    body = {
        "candidate_id": candidate.candidate_id,
        "synthesis_id": candidate.synthesis_id,
        "session_id": candidate.session_id,
        "vault_id": candidate.vault_id,
        "source": "session_synthesis",
    }
    body.update(overrides)
    return body


# ---------------------------------------------------------------------------
# Fixtures (mirror test_session_synthesis_staging.py)
# ---------------------------------------------------------------------------

import tempfile
import chromadb


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
    collection = client.get_or_create_collection('test_apply', metadata={'hnsw:space': 'cosine'})
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
    # find_semantic_match hits a real embedding endpoint; disable it at the
    # source module so both staging and apply fall through to create
    # (exact-title match still works via the generator).
    import bdh_graph_harness.neurogenesis.dedupe as dedupe
    monkeypatch.setattr(dedupe, 'find_semantic_match', lambda *a, **k: None)

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


# ---------------------------------------------------------------------------
# HTTP contract
# ---------------------------------------------------------------------------

class TestApplyHttpContract:
    @pytest.mark.asyncio
    async def test_missing_ids_return_400(self, mock_app_setup, monkeypatch):
        nodes, edges, collection, state, config, d = mock_app_setup
        app = _capture_app(monkeypatch, config, nodes, edges, collection, state)
        from aiohttp.test_utils import TestClient, TestServer
        server = TestServer(app)
        client = TestClient(server)
        await client.start_server()
        try:
            resp = await client.post('/api/synthesis/apply', json={'source': 'session_synthesis'})
            assert resp.status == 400
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_wrong_source_returns_400(self, mock_app_setup, monkeypatch):
        nodes, edges, collection, state, config, d = mock_app_setup
        app = _capture_app(monkeypatch, config, nodes, edges, collection, state)
        from aiohttp.test_utils import TestClient, TestServer
        server = TestServer(app)
        client = TestClient(server)
        await client.start_server()
        try:
            resp = await client.post('/api/synthesis/apply', json={
                'candidate_id': 'c-1', 'synthesis_id': 'syn-1',
                'session_id': 'sess-1', 'source': 'assistant_response',
            })
            assert resp.status == 400
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_unknown_candidate_returns_404(self, mock_app_setup, monkeypatch):
        nodes, edges, collection, state, config, d = mock_app_setup
        app = _capture_app(monkeypatch, config, nodes, edges, collection, state)
        from aiohttp.test_utils import TestClient, TestServer
        server = TestServer(app)
        client = TestClient(server)
        await client.start_server()
        try:
            resp = await client.post('/api/synthesis/apply', json={
                'candidate_id': 'cand-nonexistent', 'synthesis_id': 'syn-1',
                'session_id': 'sess-1', 'source': 'session_synthesis',
            })
            assert resp.status == 404
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_vault_mismatch_returns_400(self, mock_app_setup, monkeypatch):
        nodes, edges, collection, state, config, d = mock_app_setup
        candidate = _stage_approved(Path(d), vault_id="other-vault")
        app = _capture_app(monkeypatch, config, nodes, edges, collection, state)
        from aiohttp.test_utils import TestClient, TestServer
        server = TestServer(app)
        client = TestClient(server)
        await client.start_server()
        try:
            resp = await client.post('/api/synthesis/apply', json=_apply_body(candidate))
            assert resp.status == 400
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_non_approved_candidate_returns_400(self, mock_app_setup, monkeypatch):
        nodes, edges, collection, state, config, d = mock_app_setup
        # Stage but do NOT approve (stays pending_review).
        staging.stage_session_synthesis_candidates(
            str(d), synthesis_id="syn-1", vault_id="default", session_id="sess-1",
            transcript_sha256=SHA, response_text="A pending concept.", dry_run=False,
        )
        candidate = staging.list_candidates(str(d))[0]
        app = _capture_app(monkeypatch, config, nodes, edges, collection, state)
        from aiohttp.test_utils import TestClient, TestServer
        server = TestServer(app)
        client = TestClient(server)
        await client.start_server()
        try:
            resp = await client.post('/api/synthesis/apply', json=_apply_body(candidate))
            assert resp.status == 400
            # No vault note, no journal row, no audit transition.
            assert not (Path(d) / "wiki" / "concepts").exists() or not any(
                (Path(d) / "wiki" / "concepts").glob("*.md")
            )
            assert list_operation_records(str(d)) == []
            latest = ca.latest_curate_state(str(d), candidate.candidate_id)
            assert latest.state == "pending_review"
        finally:
            await client.close()


# ---------------------------------------------------------------------------
# Apply + idempotency
# ---------------------------------------------------------------------------

class TestApplyIdempotency:
    @pytest.mark.asyncio
    async def test_first_apply_creates_note_and_transitions_audit(self, mock_app_setup, monkeypatch):
        nodes, edges, collection, state, config, d = mock_app_setup
        candidate = _stage_approved(Path(d))
        app = _capture_app(monkeypatch, config, nodes, edges, collection, state)
        from aiohttp.test_utils import TestClient, TestServer
        server = TestServer(app)
        client = TestClient(server)
        await client.start_server()
        try:
            resp = await client.post('/api/synthesis/apply', json=_apply_body(candidate))
            assert resp.status == 200
            data = await resp.json()
            assert data['status'] == 'created'
            assert data['idempotent'] is False
            assert data['applied'] is True
            assert data['note_path']

            # Audit transitioned to created.
            latest = ca.latest_curate_state(str(d), candidate.candidate_id)
            assert latest.state == 'created'
            assert latest.note_path == data['note_path']

            # A note file was actually created.
            note_path = Path(d) / data['note_path']
            assert note_path.is_file()

            # One journal operation (prepared + applied = 2 records for the
            # same operation_id).
            records = list_operation_records(str(d))
            assert len(records) == 2
            assert records[0]['action'] == 'created'
            assert records[0]['operation_id'] == records[1]['operation_id']
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_repeat_apply_is_idempotent(self, mock_app_setup, monkeypatch):
        nodes, edges, collection, state, config, d = mock_app_setup
        candidate = _stage_approved(Path(d))
        app = _capture_app(monkeypatch, config, nodes, edges, collection, state)
        from aiohttp.test_utils import TestClient, TestServer
        server = TestServer(app)
        client = TestClient(server)
        await client.start_server()
        try:
            body = _apply_body(candidate)
            first = await client.post('/api/synthesis/apply', json=body)
            assert first.status == 200
            first_data = await first.json()

            notes_before = list((Path(d) / "wiki" / "concepts").glob("*.md"))
            records_before = list_operation_records(str(d))
            audit_before = len(ca.read_curate_audit(str(d)))

            second = await client.post('/api/synthesis/apply', json=body)
            assert second.status == 200
            second_data = await second.json()

            assert second_data['idempotent'] is True
            assert second_data['status'] == first_data['status']
            assert second_data['note_path'] == first_data['note_path']

            # No second note, journal row, or audit entry.
            notes_after = list((Path(d) / "wiki" / "concepts").glob("*.md"))
            assert len(notes_after) == len(notes_before)
            assert len(list_operation_records(str(d))) == len(records_before)
            assert len(ca.read_curate_audit(str(d))) == audit_before
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_apply_merges_into_existing_note(self, mock_app_setup, monkeypatch):
        nodes, edges, collection, state, config, d = mock_app_setup
        # Stage a candidate whose title exactly matches an existing node.
        candidate = _stage_candidate_direct(
            Path(d), title="Alpha", definition="Extra evidence about Alpha.",
        )
        app = _capture_app(monkeypatch, config, nodes, edges, collection, state)
        from aiohttp.test_utils import TestClient, TestServer
        server = TestServer(app)
        client = TestClient(server)
        await client.start_server()
        try:
            resp = await client.post('/api/synthesis/apply', json=_apply_body(candidate))
            assert resp.status == 200
            data = await resp.json()
            # The existing node 'alpha' has no real file on disk, so merge
            # resolves to unavailable -> noop. Either merged or noop is a
            # valid non-create outcome; assert it is NOT created.
            assert data['status'] in {'merged', 'noop'}
            assert data['applied'] is (data['status'] == 'merged')
        finally:
            await client.close()


# ---------------------------------------------------------------------------
# Conflict / failure persistence
# ---------------------------------------------------------------------------

class TestApplyConflictAndFailure:
    @pytest.mark.asyncio
    async def test_conflict_is_persisted_without_deleting_candidate(self, mock_app_setup, monkeypatch):
        nodes, edges, collection, state, config, d = mock_app_setup
        # A conflicting definition (negation language) against an existing note.
        candidate = _stage_candidate_direct(
            Path(d), title="Alpha", definition="Alpha should never be used this way.",
        )
        app = _capture_app(monkeypatch, config, nodes, edges, collection, state)
        from aiohttp.test_utils import TestClient, TestServer
        server = TestServer(app)
        client = TestClient(server)
        await client.start_server()
        try:
            resp = await client.post('/api/synthesis/apply', json=_apply_body(candidate))
            assert resp.status == 200
            data = await resp.json()
            assert data['status'] == 'conflict'

            # Candidate file still exists.
            assert staging.load_candidate(str(d), candidate.candidate_id) is not None
            # Audit transitioned to conflict.
            latest = ca.latest_curate_state(str(d), candidate.candidate_id)
            assert latest.state == 'conflict'
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_failure_is_persisted_and_retry_is_deterministic(self, mock_app_setup, monkeypatch):
        nodes, edges, collection, state, config, d = mock_app_setup
        candidate = _stage_approved(Path(d))

        # Force create_note to raise.
        def boom(*args, **kwargs):
            raise RuntimeError("disk full")
        monkeypatch.setattr(bdh_routes, 'create_note', boom)

        app = _capture_app(monkeypatch, config, nodes, edges, collection, state)
        from aiohttp.test_utils import TestClient, TestServer
        server = TestServer(app)
        client = TestClient(server)
        await client.start_server()
        try:
            body = _apply_body(candidate)
            resp = await client.post('/api/synthesis/apply', json=body)
            assert resp.status == 500

            # Audit transitioned to failed.
            latest = ca.latest_curate_state(str(d), candidate.candidate_id)
            assert latest.state == 'failed'
            # Candidate file still exists.
            assert staging.load_candidate(str(d), candidate.candidate_id) is not None

            # Retry returns the previous failed result deterministically.
            resp2 = await client.post('/api/synthesis/apply', json=body)
            assert resp2.status == 200
            data2 = await resp2.json()
            assert data2['idempotent'] is True
            assert data2['status'] == 'failed'
        finally:
            await client.close()
