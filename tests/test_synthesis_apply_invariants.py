"""Invariant tests for the staged session_synthesis extraction + apply gate.

These are behavioral invariants over the real BDH path (staging -> Curate
review -> apply), not source-shape assertions. They close the gaps left by
the focused unit tests:

  1. Reject is side-effect free: rejecting a candidate never creates a vault
     note, merges evidence, mutates Hebbian state, or writes a journal row.
  2. Wrong-vault and wrong-candidate apply requests are rejected without any
     vault/audit/journal mutation.
  3. Non-session sources (assistant_response, nightly_semantic_consolidation,
     interactive) are unaffected by the staging/apply gate.
"""
import json
import tempfile
from pathlib import Path

import pytest

import bdh_graph_harness.api.routes as bdh_routes
import bdh_graph_harness.retrieval.attention as bdh_attention_mod
from bdh_graph_harness.memory import curate_audit as ca
from bdh_graph_harness.memory import session_synthesis_staging as staging
from bdh_graph_harness.neurogenesis.operation_journal import list_operation_records


SHA = "a" * 64


def _stage_pending(tmp_path, *, title="Sparse Attention",
                   definition="A durable concept about sparse attention.",
                   synthesis_id="syn-1", session_id="sess-1", vault_id="default"):
    """Stage a candidate and leave it pending_review (not approved)."""
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
    return candidates[0]


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
# Fixtures (mirror test_synthesis_apply.py)
# ---------------------------------------------------------------------------

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
    collection = client.get_or_create_collection('test_apply_invariants', metadata={'hnsw:space': 'cosine'})
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


def _vault_mutation_snapshot(d):
    """Return a snapshot of every durable artifact the apply gate may touch."""
    return {
        "notes": sorted(str(p) for p in Path(d).rglob("*.md")),
        "journal": list_operation_records(str(d)),
        "audit": ca.read_curate_audit(str(d)),
        "state_file": (Path(d) / ".bdh-state.json").is_file(),
    }


# ---------------------------------------------------------------------------
# 1. Reject is side-effect free
# ---------------------------------------------------------------------------

class TestRejectSideEffectFree:
    def test_reject_creates_no_note_merge_hebbian_or_journal(self, tmp_path):
        candidate = _stage_pending(tmp_path)

        before = _vault_mutation_snapshot(tmp_path)

        # The Curate reject action: mark the candidate rejected.
        rejected = staging.update_candidate_status(
            str(tmp_path), candidate.candidate_id, "rejected", reason="low confidence",
        )
        assert rejected is not None
        assert rejected.status == "rejected"

        after = _vault_mutation_snapshot(tmp_path)

        # No vault note, no merge, no Hebbian state, no journal row.
        assert after["notes"] == before["notes"]
        assert after["journal"] == before["journal"]
        assert after["state_file"] is False
        # The candidate file still exists (reject is not a delete).
        assert staging.load_candidate(str(tmp_path), candidate.candidate_id) is not None

    def test_reject_does_not_transition_audit_to_created_or_merged(self, tmp_path):
        candidate = _stage_pending(tmp_path)
        staging.update_candidate_status(
            str(tmp_path), candidate.candidate_id, "rejected", reason="low confidence",
        )
        latest = ca.latest_curate_state(str(tmp_path), candidate.candidate_id)
        # Reject must never leave the audit in a created/merged state.
        assert latest is not None
        assert latest.state not in {"created", "merged"}

    @pytest.mark.asyncio
    async def test_apply_rejected_candidate_is_noop(self, mock_app_setup, monkeypatch):
        nodes, edges, collection, state, config, d = mock_app_setup
        candidate = _stage_pending(Path(d))
        staging.update_candidate_status(str(d), candidate.candidate_id, "rejected")

        app = _capture_app(monkeypatch, config, nodes, edges, collection, state)
        from aiohttp.test_utils import TestClient, TestServer
        server = TestServer(app)
        client = TestClient(server)
        await client.start_server()
        try:
            before = _vault_mutation_snapshot(d)
            resp = await client.post('/api/synthesis/apply', json=_apply_body(candidate))
            # A rejected candidate fails the approval gate and must not mutate
            # the vault, journal, or audit state.
            assert resp.status == 400
            after = _vault_mutation_snapshot(d)
            assert after["notes"] == before["notes"]
            assert after["journal"] == before["journal"]
            # Audit must not have moved to a created/merged state.
            latest = ca.latest_curate_state(str(d), candidate.candidate_id)
            assert latest is None or latest.state not in {"created", "merged"}
        finally:
            await client.close()


# ---------------------------------------------------------------------------
# 2. Wrong-vault / wrong-candidate rejected without mutation
# ---------------------------------------------------------------------------

class TestWrongTargetNoMutation:
    @pytest.mark.asyncio
    async def test_wrong_vault_apply_mutates_nothing(self, mock_app_setup, monkeypatch):
        nodes, edges, collection, state, config, d = mock_app_setup
        # Stage a candidate that belongs to a different vault.
        candidate = _stage_pending(Path(d), vault_id="other-vault")
        staging.update_candidate_status(str(d), candidate.candidate_id, "approved")

        app = _capture_app(monkeypatch, config, nodes, edges, collection, state)
        from aiohttp.test_utils import TestClient, TestServer
        server = TestServer(app)
        client = TestClient(server)
        await client.start_server()
        try:
            before = _vault_mutation_snapshot(d)
            resp = await client.post('/api/synthesis/apply', json=_apply_body(candidate))
            assert resp.status == 400
            after = _vault_mutation_snapshot(d)
            assert after["notes"] == before["notes"]
            assert after["journal"] == before["journal"]
            # Audit must not have moved to a created/merged state.
            latest = ca.latest_curate_state(str(d), candidate.candidate_id)
            assert latest is None or latest.state not in {"created", "merged"}
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_wrong_candidate_apply_mutates_nothing(self, mock_app_setup, monkeypatch):
        nodes, edges, collection, state, config, d = mock_app_setup
        app = _capture_app(monkeypatch, config, nodes, edges, collection, state)
        from aiohttp.test_utils import TestClient, TestServer
        server = TestServer(app)
        client = TestClient(server)
        await client.start_server()
        try:
            before = _vault_mutation_snapshot(d)
            resp = await client.post('/api/synthesis/apply', json={
                'candidate_id': 'cand-nonexistent',
                'synthesis_id': 'syn-1',
                'session_id': 'sess-1',
                'vault_id': 'default',
                'source': 'session_synthesis',
            })
            assert resp.status == 404
            after = _vault_mutation_snapshot(d)
            assert after["notes"] == before["notes"]
            assert after["journal"] == before["journal"]
            assert after["audit"] == before["audit"]
        finally:
            await client.close()




# ---------------------------------------------------------------------------
# 3. Session synthesis staging is a strict pre-write gate
# ---------------------------------------------------------------------------

class TestSessionSynthesisStrictStaging:
    @pytest.mark.asyncio
    async def test_staging_disables_learning_before_neurogenesis(self, mock_app_setup, monkeypatch):
        nodes, edges, collection, state, config, d = mock_app_setup
        config['session_synthesis_staging_enabled'] = True
        captured = {}

        async def fake_attention(query, ctx, ws_clients, **kwargs):
            captured['learn'] = kwargs['learn']
            return {}, [], [], {}

        monkeypatch.setattr(bdh_routes, 'run_attention_and_plasticity', fake_attention)
        monkeypatch.setattr(bdh_routes, 'llm_respond', lambda *a, **k: 'Staged response')
        monkeypatch.setattr(bdh_routes, 'run_neurogenesis', lambda *a, **k: (_ for _ in ()).throw(AssertionError('must not run before approval')))
        monkeypatch.setattr(bdh_routes, 'stage_from_api_response', lambda *a, **k: captured.setdefault('staged', True))

        app = _capture_app(monkeypatch, config, nodes, edges, collection, state)
        from aiohttp.test_utils import TestClient, TestServer
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            resp = await client.post('/api/query', json={
                'query': 'session synthesis query',
                'user_prompt': 'USER: durable architecture\nASSISTANT: Staged response',
                'source': 'session_synthesis',
                'learn': True,
                'respond': True,
                'metadata': {
                    'session_id': 'sess-1',
                    'synthesis_id': 'syn-1',
                    'transcript_sha256': SHA,
                    'queued_at': '1.0',
                },
            })
            assert resp.status == 200
            assert captured['learn'] is False
            assert captured['staged'] is True
        finally:
            await client.close()



class TestNonSessionSourcesUnaffected:
    @pytest.mark.asyncio
    async def test_assistant_response_does_not_stage_candidates(self, mock_app_setup, monkeypatch):
        nodes, edges, collection, state, config, d = mock_app_setup
        # Enable staging so we can prove the gate is source-scoped.
        config['session_synthesis_staging_enabled'] = True

        app = _capture_app(monkeypatch, config, nodes, edges, collection, state)
        from aiohttp.test_utils import TestClient, TestServer
        server = TestServer(app)
        client = TestClient(server)
        await client.start_server()
        try:
            resp = await client.post('/api/query', json={
                'query': 'assistant response query',
                'source': 'assistant_response',
                'learn': False,
                'respond': False,
            })
            assert resp.status == 200
            # No candidate files, no curate audit entries for a non-session source.
            assert not (Path(d) / '.bdh-candidates').exists()
            assert ca.read_curate_audit(str(d)) == []
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_nightly_consolidation_does_not_stage_candidates(self, mock_app_setup, monkeypatch):
        nodes, edges, collection, state, config, d = mock_app_setup
        config['session_synthesis_staging_enabled'] = True

        app = _capture_app(monkeypatch, config, nodes, edges, collection, state)
        from aiohttp.test_utils import TestClient, TestServer
        server = TestServer(app)
        client = TestClient(server)
        await client.start_server()
        try:
            resp = await client.post('/api/query', json={
                'query': 'nightly consolidation query',
                'source': 'nightly_semantic_consolidation',
                'learn': False,
                'respond': False,
            })
            assert resp.status == 200
            assert not (Path(d) / '.bdh-candidates').exists()
            assert ca.read_curate_audit(str(d)) == []
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_interactive_query_does_not_stage_candidates(self, mock_app_setup, monkeypatch):
        nodes, edges, collection, state, config, d = mock_app_setup
        config['session_synthesis_staging_enabled'] = True

        app = _capture_app(monkeypatch, config, nodes, edges, collection, state)
        from aiohttp.test_utils import TestClient, TestServer
        server = TestServer(app)
        client = TestClient(server)
        await client.start_server()
        try:
            resp = await client.post('/api/query', json={
                'query': 'interactive query',
                'learn': False,
                'respond': False,
            })
            assert resp.status == 200
            assert not (Path(d) / '.bdh-candidates').exists()
            assert ca.read_curate_audit(str(d)) == []
        finally:
            await client.close()
