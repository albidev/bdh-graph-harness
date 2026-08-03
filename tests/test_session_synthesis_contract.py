"""Tests for the session_synthesis API contract (GitHub issue #16).

Covers:
  1. Source policy registry — dampening tiers, anti-fall-through
  2. Hebbian frequency increment for session_synthesis
  3. Provenance metadata on newly-created synapses
  4. Retrieval routing — user_prompt used for attention when appropriate
  5. Backward compatibility for pre-existing sources
"""
import pytest
from math import log1p

import harness
from bdh_graph_harness.memory.source_policy import (
    get_frequency_increment,
    get_source_policy,
    use_user_prompt_for_retrieval,
    allowed_sources,
    SourcePolicy,
)
from bdh_graph_harness.memory.hebbian import encode_synapse_key


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _synapse_key(a: str, b: str) -> str:
    return encode_synapse_key(a, b)


def _fresh_state():
    return {
        'synapses': {},
        'created': '2026-01-01T00:00:00',
        'updated': '2026-01-01T00:00:00',
        'queries': 0,
    }


# ===========================================================================
# 1. Source policy registry
# ===========================================================================

class TestSourcePolicyRegistry:
    """The registry must be explicit, known sources only."""

    def test_session_synthesis_is_registered(self):
        policy = get_source_policy("session_synthesis")
        assert policy is not None
        assert isinstance(policy, SourcePolicy)

    def test_session_synthesis_dampened(self):
        """session_synthesis must use reduced frequency_increment (< 1.0)."""
        policy = get_source_policy("session_synthesis")
        assert 0.0 < policy.frequency_increment < 1.0
        assert policy.frequency_increment == pytest.approx(0.2)

    def test_session_synthesis_uses_user_prompt_for_retrieval(self):
        assert use_user_prompt_for_retrieval("session_synthesis") is True

    def test_session_synthesis_provenance_label(self):
        policy = get_source_policy("session_synthesis")
        assert policy.provenance_label == "session_synthesis"

    def test_session_synthesis_allows_neurogenesis(self):
        policy = get_source_policy("session_synthesis")
        assert policy.allow_neurogenesis is True

    def test_unknown_source_raises_value_error(self):
        """Unknown sources must NOT silently fall through to full strength."""
        with pytest.raises(ValueError, match="Unknown source"):
            get_frequency_increment("bogus_source_xyz")

    def test_unknown_source_get_policy_returns_none(self):
        assert get_source_policy("bogus_source_xyz") is None

    def test_user_query_is_full_strength(self):
        assert get_frequency_increment("user_query") == 1.0

    def test_assistant_response_is_dampened(self):
        assert get_frequency_increment("assistant_response") == pytest.approx(0.3)

    def test_nightly_semantic_consolidation_is_dampened(self):
        assert get_frequency_increment("nightly_semantic_consolidation") == pytest.approx(0.3)

    def test_cron_source_is_dampened(self):
        assert get_frequency_increment("cron") == pytest.approx(0.3)

    def test_automatic_retrieval_is_full_strength(self):
        assert get_frequency_increment("automatic_retrieval") == 1.0

    def test_none_source_returns_full_strength(self):
        """None (omitted source) defaults to interactive full strength."""
        assert get_frequency_increment(None) == 1.0

    def test_none_source_retrieval_flag_is_false(self):
        assert use_user_prompt_for_retrieval(None) is False

    def test_allowed_sources_contains_session_synthesis(self):
        sources = allowed_sources()
        assert "session_synthesis" in sources
        assert isinstance(sources, list)
        assert sources == sorted(sources)  # always sorted


# ===========================================================================
# 2. Hebbian frequency increment dampening
# ===========================================================================

class TestHebbianDampening:
    """Verify that hebbian_update respects the source policy increments."""

    def test_session_synthesis_increment(self):
        """session_synthesis (0.2) should produce less frequency than interactive (1.0)."""
        active = {'a': 0.8, 'b': 0.6}

        state_interactive, _, _ = harness.hebbian_update(active, _fresh_state(), source=None)
        state_session, _, _ = harness.hebbian_update(active, _fresh_state(), source='session_synthesis')

        freq_interactive = state_interactive['synapses'][_synapse_key('a', 'b')]['frequency']
        freq_session = state_session['synapses'][_synapse_key('a', 'b')]['frequency']

        # Interactive: 1.0 * 0.8 * 0.6 = 0.48
        assert freq_interactive == pytest.approx(0.8 * 0.6, abs=1e-6)
        # Session synthesis: 0.2 * 0.8 * 0.6 = 0.096
        assert freq_session == pytest.approx(0.2 * 0.8 * 0.6, abs=1e-6)
        assert freq_session < freq_interactive

    def test_assistant_response_matches_expected_dampening(self):
        """assistant_response uses 0.3 increment."""
        active = {'a': 0.8, 'b': 0.6}
        state, _, _ = harness.hebbian_update(active, _fresh_state(), source='assistant_response')
        freq = state['synapses'][_synapse_key('a', 'b')]['frequency']
        assert freq == pytest.approx(0.3 * 0.8 * 0.6, abs=1e-6)

    def test_nightly_consolidation_matches_expected_dampening(self):
        """nightly_semantic_consolidation uses 0.3 increment."""
        active = {'a': 0.8, 'b': 0.6}
        state, _, _ = harness.hebbian_update(
            active, _fresh_state(), source='nightly_semantic_consolidation',
        )
        freq = state['synapses'][_synapse_key('a', 'b')]['frequency']
        assert freq == pytest.approx(0.3 * 0.8 * 0.6, abs=1e-6)

    def test_weight_reflects_dampened_frequency(self):
        """Weight formula must reflect the dampened frequency, not full strength."""
        active = {'a': 0.8, 'b': 0.6}
        alpha = harness.CONFIG['alpha']
        beta = harness.CONFIG['beta']
        freq_scale = harness.CONFIG.get('hebbian_frequency_scale', 10.0)

        state, _, _ = harness.hebbian_update(active, _fresh_state(), source='session_synthesis')
        syn = state['synapses'][_synapse_key('a', 'b')]

        expected_freq = 0.2 * 0.8 * 0.6
        assert syn['frequency'] == pytest.approx(expected_freq, abs=1e-6)
        expected_weight = alpha * (log1p(expected_freq) / log1p(freq_scale)) + beta * 1.0
        assert syn['weight'] == pytest.approx(expected_weight, abs=1e-6)

    def test_unknown_source_raises_in_hebbian(self):
        """hebbian_update must raise for unknown sources (no silent fall-through)."""
        active = {'a': 0.8, 'b': 0.6}
        with pytest.raises(ValueError, match="Unknown source"):
            harness.hebbian_update(active, _fresh_state(), source='nonexistent_signal')


# ===========================================================================
# 3. Provenance metadata on new synapses
# ===========================================================================

class TestProvenance:
    """Newly created synapses carry provenance labels from the source policy."""

    def test_session_synthesis_provenance_on_new_synapse(self):
        active = {'a': 0.8, 'b': 0.6}
        state, _, _ = harness.hebbian_update(active, _fresh_state(), source='session_synthesis')
        syn = state['synapses'][_synapse_key('a', 'b')]
        assert syn.get('source') == 'session_synthesis'

    def test_assistant_response_provenance_on_new_synapse(self):
        active = {'a': 0.8, 'b': 0.6}
        state, _, _ = harness.hebbian_update(active, _fresh_state(), source='assistant_response')
        syn = state['synapses'][_synapse_key('a', 'b')]
        assert syn.get('source') == 'assistant_response'

    def test_nightly_consolidation_provenance(self):
        active = {'a': 0.8, 'b': 0.6}
        state, _, _ = harness.hebbian_update(
            active, _fresh_state(), source='nightly_semantic_consolidation',
        )
        syn = state['synapses'][_synapse_key('a', 'b')]
        assert syn.get('source') == 'semantic_consolidation'

    def test_interactive_query_provenance(self):
        active = {'a': 0.8, 'b': 0.6}
        state, _, _ = harness.hebbian_update(active, _fresh_state(), source=None)
        syn = state['synapses'][_synapse_key('a', 'b')]
        # None source has no SourcePolicy, so no provenance label is stored
        assert 'source' not in syn

    def test_existing_synapse_preserves_old_provenance(self):
        """When a synapse already exists, its original provenance is kept."""
        state = _fresh_state()
        # Pre-create a synapse with explicit provenance
        state['synapses'][_synapse_key('a', 'b')] = {
            'weight': 0.5, 'frequency': 2.0,
            'last_coactivated': '2026-01-01T00:00:00',
            'created': '2026-01-01T00:00:00',
            'source': 'original_source',
        }
        active = {'a': 0.8, 'b': 0.6}
        state, _, _ = harness.hebbian_update(active, state, source='session_synthesis')
        syn = state['synapses'][_synapse_key('a', 'b')]
        assert syn.get('source') == 'original_source'


# ===========================================================================
# 4. Retrieval routing — user_prompt semantics
# ===========================================================================

class TestRetrievalRouting:
    """Verify that use_user_prompt_for_retrieval drives the correct path."""

    def test_session_synthesis_uses_user_prompt(self):
        assert use_user_prompt_for_retrieval("session_synthesis") is True

    def test_interactive_query_does_not_use_user_prompt(self):
        assert use_user_prompt_for_retrieval("user_query") is False

    def test_assistant_response_does_not_use_user_prompt(self):
        assert use_user_prompt_for_retrieval("assistant_response") is False

    def test_nightly_consolidation_does_not_use_user_prompt(self):
        assert use_user_prompt_for_retrieval("nightly_semantic_consolidation") is False

    def test_none_source_does_not_use_user_prompt(self):
        assert use_user_prompt_for_retrieval(None) is False


# ===========================================================================
# 5. API-level contract: source_policy in routing response
# ===========================================================================

class TestAPIRoutingProvenance:
    """The routing dict returned by /api/query must carry source_policy."""

    @pytest.mark.asyncio
    async def test_session_synthesis_routing_includes_source_policy(self, mock_app_setup, monkeypatch):
        """When source=session_synthesis, routing.source_policy must be present."""
        from aiohttp.test_utils import TestClient, TestServer
        import bdh_graph_harness.api.routes as bdh_routes

        nodes, edges, collection, state, config, _ = mock_app_setup
        app = _capture_app(monkeypatch, config, nodes, edges, collection, state)
        server = TestServer(app)
        client = TestClient(server)
        await client.start_server()
        try:
            resp = await client.post('/api/query', json={
                'query': 'generic session label',
                'source': 'session_synthesis',
                'user_prompt': 'This is the actual session transcript with details...',
                'learn': False,
                'respond': False,
            })
            assert resp.status == 200
            data = await resp.json()
            routing = data['routing']
            assert routing['source'] == 'session_synthesis'
            sp = routing['source_policy']
            assert sp['frequency_increment'] == pytest.approx(0.2)
            assert sp['provenance_label'] == 'session_synthesis'
            assert sp['use_user_prompt_for_retrieval'] is True
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_user_query_routing_includes_source_policy(self, mock_app_setup, monkeypatch):
        """Default source also gets source_policy in routing."""
        from aiohttp.test_utils import TestClient, TestServer

        nodes, edges, collection, state, config, _ = mock_app_setup
        app = _capture_app(monkeypatch, config, nodes, edges, collection, state)
        server = TestServer(app)
        client = TestClient(server)
        await client.start_server()
        try:
            resp = await client.post('/api/query', json={
                'query': 'test query',
                'learn': False,
                'respond': False,
            })
            assert resp.status == 200
            data = await resp.json()
            routing = data['routing']
            assert 'source' in routing
            assert 'source_policy' in routing
            sp = routing['source_policy']
            assert sp['frequency_increment'] == pytest.approx(1.0)
            assert sp['use_user_prompt_for_retrieval'] is False
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_unknown_source_returns_error(self, mock_app_setup, monkeypatch):
        """Unknown source at the API level must fail with a clear error."""
        from aiohttp.test_utils import TestClient, TestServer

        nodes, edges, collection, state, config, _ = mock_app_setup

        # Patch hebbian_update to raise on unknown sources (the new behavior)
        original_hebbian = bdh_routes.hebbian_update
        def strict_hebbian(*args, **kwargs):
            source = kwargs.get('source') or (args[3] if len(args) > 3 else None)
            if source not in (None, 'user_query', 'assistant_response',
                              'nightly_semantic_consolidation', 'session_synthesis',
                              'cron', 'automatic_retrieval'):
                raise ValueError(f"Unknown source {source!r}")
            return original_hebbian(*args, **kwargs)

        monkeypatch.setattr(bdh_routes, 'hebbian_update', strict_hebbian)

        app = _capture_app(monkeypatch, config, nodes, edges, collection, state)
        server = TestServer(app)
        client = TestClient(server)
        await client.start_server()
        try:
            resp = await client.post('/api/query', json={
                'query': 'test',
                'source': 'bogus_unknown_source',
                'learn': True,
            })
            # The server should return a 500 error (ValueError propagation)
            assert resp.status == 500
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_session_synthesis_user_prompt_used_for_retrieval(self, mock_app_setup, monkeypatch):
        """When source=session_synthesis, the attention pass receives user_prompt
        as the retrieval query, not the fixed generic query."""
        from aiohttp.test_utils import TestClient, TestServer
        import bdh_graph_harness.api.routes as bdh_routes

        captured_queries = []

        def capturing_attention(query, *args, **kwargs):
            captured_queries.append(query)
            routing_meta = kwargs.get('routing_meta') or (args[6] if len(args) > 6 else None)
            if routing_meta is not None:
                routing_meta.update({
                    'activation_details': [],
                    'vector_top_score': 0.8,
                    'bm25_top_score': 0.5,
                    'hybrid_top_score': 0.7,
                    'hybrid_second_score': 0.3,
                    'hybrid_margin': 0.4,
                    'hybrid_enabled': False,
                })
            return {}

        monkeypatch.setattr(bdh_routes, 'attention', capturing_attention)

        nodes, edges, collection, state, config, _ = mock_app_setup
        app = _capture_app(monkeypatch, config, nodes, edges, collection, state)
        server = TestServer(app)
        client = TestClient(server)
        await client.start_server()
        try:
            resp = await client.post('/api/query', json={
                'query': 'fixed generic label',
                'source': 'session_synthesis',
                'user_prompt': 'The actual session evidence goes here',
                'learn': False,
                'respond': False,
            })
            assert resp.status == 200
            # The attention call should have received user_prompt, not the fixed query
            assert len(captured_queries) >= 1
            assert captured_queries[-1] == 'The actual session evidence goes here'
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_non_session_source_uses_query_for_retrieval(self, mock_app_setup, monkeypatch):
        """For non-session-synthesis sources, user_prompt does NOT override query."""
        from aiohttp.test_utils import TestClient, TestServer
        import bdh_graph_harness.api.routes as bdh_routes

        captured_queries = []

        def capturing_attention(query, *args, **kwargs):
            captured_queries.append(query)
            routing_meta = kwargs.get('routing_meta') or (args[6] if len(args) > 6 else None)
            if routing_meta is not None:
                routing_meta.update({
                    'activation_details': [],
                    'vector_top_score': 0.8,
                    'bm25_top_score': 0.5,
                    'hybrid_top_score': 0.7,
                    'hybrid_second_score': 0.3,
                    'hybrid_margin': 0.4,
                    'hybrid_enabled': False,
                })
            return {}

        monkeypatch.setattr(bdh_routes, 'attention', capturing_attention)

        nodes, edges, collection, state, config, _ = mock_app_setup
        app = _capture_app(monkeypatch, config, nodes, edges, collection, state)
        server = TestServer(app)
        client = TestClient(server)
        await client.start_server()
        try:
            resp = await client.post('/api/query', json={
                'query': 'the real query',
                'source': 'assistant_response',
                'user_prompt': 'extra context',
                'learn': False,
                'respond': False,
            })
            assert resp.status == 200
            assert len(captured_queries) >= 1
            # Should use query, NOT user_prompt
            assert captured_queries[-1] == 'the real query'
        finally:
            await client.close()


# ---------------------------------------------------------------------------
# Test fixtures reused from test_api.py (duplicated here for isolation)
# ---------------------------------------------------------------------------

import asyncio
import tempfile
import chromadb
import bdh_graph_harness.retrieval.attention as bdh_attention_mod


@pytest.fixture
def mock_app_setup(monkeypatch):
    """Create a mock app setup: nodes, edges, collection, state, config."""
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
    collection = client.get_or_create_collection('test_session_contract', metadata={'hnsw:space': 'cosine'})
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


import bdh_graph_harness.api.routes as bdh_routes
