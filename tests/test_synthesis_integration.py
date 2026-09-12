"""Integration tests for session synthesis audit + oMLX routing."""
import json
import os
import pytest
from types import SimpleNamespace

import harness
import bdh_graph_harness.api.routes as bdh_routes
import bdh_graph_harness.retrieval.attention as bdh_attention_mod
from bdh_graph_harness.config import resolve_llm_config_for_source
from bdh_graph_harness.llm import providers as bdh_providers


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_app_setup(monkeypatch):
    """Create a mock app setup: nodes, edges, collection, state, config."""
    import tempfile
    import chromadb

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
    collection = client.get_or_create_collection('test_synth_integration', metadata={'hnsw:space': 'cosine'})
    if collection.count() > 0:
        collection.delete(ids=collection.get()['ids'])
    collection.add(
        ids=['alpha', 'beta', 'gamma'],
        embeddings=[[1.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.0, 1.0, 0.0]],
        documents=['Alpha content', 'Beta content', 'Gamma content'],
        metadatas=[{'title': 'Alpha', 'tags': 'concept'}, {'title': 'Beta', 'tags': 'concept'}, {'title': 'Gamma', 'tags': 'concept'}],
    )

    config = dict(harness.CONFIG)
    config['vault_path'] = d
    config['neurogenesis_enabled'] = False

    monkeypatch.setattr(bdh_attention_mod, 'get_embeddings', lambda texts: [[1.0, 0.0, 0.0]])
    monkeypatch.setattr(bdh_routes, 'llm_respond', lambda q, a, n, **kwargs: 'Mock LLM response')
    monkeypatch.setattr(bdh_routes, 'extract_new_concepts', lambda r, q, a, n, **kwargs: [])
    monkeypatch.setattr(bdh_routes, 'save_state', lambda vr, s: None)

    return nodes, edges, collection, state, config, d


def _capture_app(monkeypatch, config, nodes, edges, collection, state):
    """Monkeypatch web.run_app to capture the app without starting a server."""
    captured = {}
    from aiohttp import web

    def fake_run_app(app, **kwargs):
        captured['app'] = app

    monkeypatch.setattr('aiohttp.web.run_app', fake_run_app)
    harness.start_api_server(config, nodes, edges, collection, state)
    return captured['app']


# ---------------------------------------------------------------------------
# Tests: source-specific session_synthesis routing
# ---------------------------------------------------------------------------


class TestSessionSynthesisSourceRouting:
    def test_session_synthesis_inherits_configured_ollama_without_override(self):
        """A Linux-style Ollama config is not replaced by oMLX."""
        base = {
            'llm_provider': 'ollama',
            'llm_model': 'qwen3.8:27b',
            'ollama_url': 'http://127.0.0.1:11434',
        }
        config = resolve_llm_config_for_source(base, 'session_synthesis')
        assert config['llm_provider'] == 'ollama'
        assert config['llm_model'] == 'qwen3.8:27b'
        assert config['llm_endpoint'] == 'http://127.0.0.1:11434/api/chat'

    def test_session_synthesis_explicit_omlx_override_is_preserved(self):
        """The existing macOS path remains available through explicit config."""
        base = {
            'llm_provider': 'ollama-cloud',
            'llm_model': 'deepseek-v4-pro',
            'llm_source_overrides': {
                'session_synthesis': {
                    'provider': 'omlx',
                    'model': 'qwen3.8-27b-oq4e-mtp',
                    'base_url': 'http://127.0.0.1:8083/v1',
                    'local_only': True,
                    'chat_template_kwargs': {'enable_thinking': False, 'thinking': False},
                },
            },
        }
        config = resolve_llm_config_for_source(base, 'session_synthesis')
        assert config['llm_provider'] == 'omlx'
        assert config['llm_model'] == 'qwen3.8-27b-oq4e-mtp'
        assert config['llm_local_only'] is True
        assert config['llm_base_url'] == 'http://127.0.0.1:8083/v1'
        assert config['llm_chat_template_kwargs'] == {
            'enable_thinking': False, 'thinking': False,
        }

    def test_session_synthesis_explicit_ollama_override_wins_over_global(self):
        """A source-specific Ollama model/base URL beats cloud global settings."""
        base = {
            'llm_provider': 'ollama-cloud',
            'llm_model': 'deepseek-v4-pro',
            'llm_base_url': 'https://ollama.com/v1',
            'llm_source_overrides': {
                'session_synthesis': {
                    'provider': 'ollama',
                    'model': 'qwen3.8:27b',
                    'base_url': 'http://127.0.0.1:11434',
                },
            },
        }
        config = resolve_llm_config_for_source(base, 'session_synthesis')
        assert config['llm_provider'] == 'ollama'
        assert config['llm_model'] == 'qwen3.8:27b'
        assert config['llm_endpoint'] == 'http://127.0.0.1:11434/api/chat'

    def test_session_synthesis_has_no_fallback_chain(self):
        """An unavailable configured backend is not silently routed elsewhere."""
        base = {
            'llm_provider': 'ollama',
            'llm_model': 'qwen3.8:27b',
            'llm_fallbacks': [
                {'provider': 'nous', 'model': 'upstage/solar-pro4:free'},
                {'provider': 'openrouter', 'model': 'openrouter/free'},
            ],
        }
        config = resolve_llm_config_for_source(base, 'session_synthesis')
        assert config['llm_fallbacks'] == []
        assert bdh_providers.resolve_llm_candidates(config) == [config]

    def test_other_sources_keep_global_provider_and_fallbacks(self):
        """The source policy does not affect normal interactive responses."""
        base = {
            'llm_provider': 'ollama-cloud',
            'llm_model': 'deepseek-v4-pro',
            'llm_fallbacks': [{'provider': 'omlx', 'model': 'local'}],
        }
        config = resolve_llm_config_for_source(base, 'assistant_response')
        assert config['llm_provider'] == 'ollama-cloud'
        assert config['llm_model'] == 'deepseek-v4-pro'
        assert config['llm_fallbacks'] == base['llm_fallbacks']


# ---------------------------------------------------------------------------
# Tests: chat_template_kwargs in payload
# ---------------------------------------------------------------------------


class TestChatTemplateKwargs:
    def test_payload_includes_chat_template_kwargs(self):
        """OpenAI-compatible payload includes chat_template_kwargs when set."""
        config = {
            'llm_provider': 'omlx',
            'llm_model': 'qwen3.8-27b-oq4e-mtp',
            'llm_temperature': 0.3,
            'llm_max_ctx': 4096,
            'llm_chat_template_kwargs': {'enable_thinking': False, 'thinking': False},
        }
        data, _ = bdh_providers._build_llm_payload(
            'test', {'a': 0.8}, {'a': {'id': 'a', 'title': 'A', 'text': 't'}}, config=config,
        )
        payload = json.loads(data)
        assert payload['chat_template_kwargs'] == {'enable_thinking': False, 'thinking': False}

    def test_payload_omits_chat_template_kwargs_when_empty(self):
        """chat_template_kwargs is omitted from payload when empty/missing."""
        config = {
            'llm_provider': 'omlx',
            'llm_model': 'qwen3.8-27b-oq4e-mtp',
            'llm_temperature': 0.3,
            'llm_max_ctx': 4096,
        }
        data, _ = bdh_providers._build_llm_payload(
            'test', {'a': 0.8}, {'a': {'id': 'a', 'title': 'A', 'text': 't'}}, config=config,
        )
        payload = json.loads(data)
        assert 'chat_template_kwargs' not in payload


# ---------------------------------------------------------------------------
# Tests: API audit recording with session_synthesis
# ---------------------------------------------------------------------------


class TestApiQueryRecordsAudit:
    @pytest.mark.asyncio
    async def test_api_query_records_audit_on_synthesis_with_concepts(
        self, mock_app_setup, monkeypatch,
    ):
        """api_query records audit entry when session_synthesis + concept created."""
        nodes, edges, collection, state, config, d = mock_app_setup
        config['neurogenesis_enabled'] = True

        monkeypatch.setattr(
            bdh_routes, 'extract_new_concepts',
            lambda r, q, a, n, **kwargs: [{'title': 'New Concept', 'definition': 'A new concept.'}],
        )

        app = _capture_app(monkeypatch, config, nodes, edges, collection, state)
        from aiohttp.test_utils import TestClient, TestServer

        server = TestServer(app)
        client = TestClient(server)
        await client.start_server()
        try:
            resp = await client.post('/api/query', json={
                'query': 'session synthesis test',
                'source': 'session_synthesis',
                'metadata': {
                    'session_id': 'sess-001',
                    'synthesis_id': 'syn-001',
                    'transcript_sha256': 'abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890',
                },
            })
            assert resp.status == 200
            data = await resp.json()
            assert len(data['new_concepts']) == 1

            # Verify audit file was created
            from bdh_graph_harness.memory.synthesis_audit import read_synthesis_audit
            entries = read_synthesis_audit(d)
            assert len(entries) == 1
            assert entries[0].session_id == 'sess-001'
            assert entries[0].synthesis_id == 'syn-001'
            assert entries[0].outcome == 'created'
            assert entries[0].vault == 'default'
            assert entries[0].provider == 'ollama'
            assert entries[0].hebbian_updates >= 0
            assert len(entries[0].transcript_sha256) == 64  # SHA-256 hex
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_api_query_records_noop_when_no_concepts(
        self, mock_app_setup, monkeypatch,
    ):
        """api_query records noop when session_synthesis produces no concepts."""
        nodes, edges, collection, state, config, d = mock_app_setup

        app = _capture_app(monkeypatch, config, nodes, edges, collection, state)
        from aiohttp.test_utils import TestClient, TestServer

        server = TestServer(app)
        client = TestClient(server)
        await client.start_server()
        try:
            resp = await client.post('/api/query', json={
                'query': 'session synthesis test',
                'source': 'session_synthesis',
                'metadata': {
                    'session_id': 'sess-002',
                    'synthesis_id': 'syn-002',
                    'transcript_sha256': 'b' * 64,
                },
            })
            assert resp.status == 200
            data = await resp.json()
            assert data['new_concepts'] == []

            from bdh_graph_harness.memory.synthesis_audit import read_synthesis_audit
            entries = read_synthesis_audit(d)
            assert len(entries) == 1
            assert entries[0].outcome == 'noop'
            assert entries[0].concept_ids == []
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_api_query_does_not_record_audit_without_synthesis_meta(
        self, mock_app_setup, monkeypatch,
    ):
        """api_query does NOT record audit when no session_id/synthesis_id provided."""
        nodes, edges, collection, state, config, d = mock_app_setup

        app = _capture_app(monkeypatch, config, nodes, edges, collection, state)
        from aiohttp.test_utils import TestClient, TestServer

        server = TestServer(app)
        client = TestClient(server)
        await client.start_server()
        try:
            resp = await client.post('/api/query', json={
                'query': 'normal query without synthesis',
                'source': 'session_synthesis',
            })
            assert resp.status == 200

            from bdh_graph_harness.memory.synthesis_audit import read_synthesis_audit
            entries = read_synthesis_audit(d)
            assert entries == []
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_api_query_records_invalid_metadata(self, mock_app_setup, monkeypatch):
        """Malformed session synthesis metadata is audited as invalid and rejected."""
        nodes, edges, collection, state, config, d = mock_app_setup
        app = _capture_app(monkeypatch, config, nodes, edges, collection, state)
        from aiohttp.test_utils import TestClient, TestServer

        server = TestServer(app)
        client = TestClient(server)
        await client.start_server()
        try:
            resp = await client.post('/api/query', json={
                'query': 'session synthesis invalid metadata',
                'source': 'session_synthesis',
                'metadata': {'session_id': 'sess-invalid'},
            })
            assert resp.status == 400

            from bdh_graph_harness.memory.synthesis_audit import read_synthesis_audit
            entries = read_synthesis_audit(d)
            assert len(entries) == 1
            assert entries[0].outcome == 'invalid'
            assert entries[0].session_id == 'sess-invalid'
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_api_query_records_failed_synthesis_on_llm_error(
        self, mock_app_setup, monkeypatch,
    ):
        """An LLM error must not be misclassified as a successful noop."""
        nodes, edges, collection, state, config, d = mock_app_setup
        monkeypatch.setattr(
            bdh_routes, 'llm_respond',
            lambda *args, **kwargs: '[LLM error: local oMLX unavailable]',
        )
        app = _capture_app(monkeypatch, config, nodes, edges, collection, state)
        from aiohttp.test_utils import TestClient, TestServer

        server = TestServer(app)
        client = TestClient(server)
        await client.start_server()
        try:
            resp = await client.post('/api/query', json={
                'query': 'session synthesis failed model',
                'source': 'session_synthesis',
                'metadata': {
                    'session_id': 'sess-failed',
                    'synthesis_id': 'syn-failed',
                    'transcript_sha256': 'a' * 64,
                    'queued_at': 1700000000.0,
                },
            })
            assert resp.status == 200

            from bdh_graph_harness.memory.synthesis_audit import read_synthesis_audit
            entries = read_synthesis_audit(d)
            assert len(entries) == 1
            assert entries[0].outcome == 'failed'
            assert entries[0].synthesis_id == 'syn-failed'
        finally:
            await client.close()
