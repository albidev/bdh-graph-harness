"""Tests for API endpoints using aiohttp TestClient."""
import os
import json
import tempfile
import pytest
import chromadb
import harness
import bdh_graph_harness.api.routes as bdh_routes
import bdh_graph_harness.retrieval.attention as bdh_attention_mod
import bdh_graph_harness.llm.providers as bdh_providers
import bdh_graph_harness.neurogenesis.creator as bdh_creator
import bdh_graph_harness.memory.state_store as bdh_state_store

pytest_plugins = ['pytest_asyncio']


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
        'synapses': {
            'alpha|beta': {'weight': 0.8, 'frequency': 3, 'last_coactivated': '2026-01-01T00:00:00'},
        },
        'created': '2026-01-01T00:00:00',
        'updated': '2026-01-01T00:00:00',
        'queries': 5,
    }

    # ChromaDB EphemeralClient
    client = chromadb.EphemeralClient()
    collection = client.get_or_create_collection('test_api', metadata={'hnsw:space': 'cosine'})
    # Clear any existing data from prior test runs
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
    config['neurogenesis_enabled'] = False  # disable to avoid file creation in query tests

    # Monkeypatch LLM and embedding calls — must patch the modules that routes.py imports from
    monkeypatch.setattr(bdh_attention_mod, 'get_embeddings', lambda texts: [[1.0, 0.0, 0.0]])
    monkeypatch.setattr(bdh_routes, 'llm_respond', lambda q, a, n: 'Mock LLM response')
    monkeypatch.setattr(bdh_routes, 'extract_new_concepts', lambda r, q, a, n: [])
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
# GET /api/stats
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_api_stats(mock_app_setup, monkeypatch):
    from aiohttp.test_utils import TestClient, TestServer

    nodes, edges, collection, state, config, _ = mock_app_setup
    app = _capture_app(monkeypatch, config, nodes, edges, collection, state)

    server = TestServer(app)
    client = TestClient(server)
    await client.start_server()

    try:
        resp = await client.get('/api/stats')
        assert resp.status == 200
        data = await resp.json()
        assert 'neurons' in data
        assert data['neurons'] == 3
        assert 'synapses' in data
        assert 'avg_degree' in data
        assert 'hebbian_synapses' in data
        assert data['hebbian_synapses'] == 1
        assert 'queries_processed' in data
        assert data['queries_processed'] == 5
        assert 'top_hebbian' in data
        assert isinstance(data['top_hebbian'], list)
    finally:
        await client.close()


# ---------------------------------------------------------------------------
# GET /api/graph
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_api_graph(mock_app_setup, monkeypatch):
    from aiohttp.test_utils import TestClient, TestServer

    nodes, edges, collection, state, config, _ = mock_app_setup
    app = _capture_app(monkeypatch, config, nodes, edges, collection, state)

    server = TestServer(app)
    client = TestClient(server)
    await client.start_server()

    try:
        resp = await client.get('/api/graph')
        assert resp.status == 200
        data = await resp.json()
        assert 'nodes' in data
        assert 'edges' in data
        assert len(data['nodes']) == 3
        # Edges should have source, target, display
        assert len(data['edges']) >= 1
        edge = data['edges'][0]
        assert 'source' in edge
        assert 'target' in edge
        assert 'display' in edge
    finally:
        await client.close()


# ---------------------------------------------------------------------------
# GET /api/hebbian
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_api_hebbian(mock_app_setup, monkeypatch):
    from aiohttp.test_utils import TestClient, TestServer

    nodes, edges, collection, state, config, _ = mock_app_setup
    app = _capture_app(monkeypatch, config, nodes, edges, collection, state)

    server = TestServer(app)
    client = TestClient(server)
    await client.start_server()

    try:
        resp = await client.get('/api/hebbian')
        assert resp.status == 200
        data = await resp.json()
        assert 'total' in data
        assert data['total'] == 1
        assert 'queries' in data
        assert 'synapses' in data
        assert len(data['synapses']) == 1
        syn = data['synapses'][0]
        assert 'note_a' in syn
        assert 'note_b' in syn
        assert 'weight' in syn
        assert 'frequency' in syn
        assert 'last_coactivated' in syn
    finally:
        await client.close()


# ---------------------------------------------------------------------------
# POST /api/query
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_api_query_success(mock_app_setup, monkeypatch):
    from aiohttp.test_utils import TestClient, TestServer

    nodes, edges, collection, state, config, _ = mock_app_setup
    app = _capture_app(monkeypatch, config, nodes, edges, collection, state)

    server = TestServer(app)
    client = TestClient(server)
    await client.start_server()

    try:
        resp = await client.post('/api/query', json={'query': 'test query'})
        assert resp.status == 200
        data = await resp.json()
        assert 'response' in data
        assert data['response'] == 'Mock LLM response'
        assert 'activated_notes' in data
        assert isinstance(data['activated_notes'], list)
        assert 'new_concepts' in data
        assert 'hebbian_synapses' in data
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_api_query_read_only_skips_learning_and_neurogenesis(mock_app_setup, monkeypatch):
    from aiohttp.test_utils import TestClient, TestServer

    nodes, edges, collection, state, config, _ = mock_app_setup
    app = _capture_app(monkeypatch, config, nodes, edges, collection, state)
    before = json.dumps(state, sort_keys=True)
    monkeypatch.setattr(
        bdh_routes, 'run_neurogenesis',
        lambda *args: pytest.fail('read-only query ran neurogenesis'),
    )
    monkeypatch.setattr(
        bdh_routes, 'llm_respond',
        lambda *args: pytest.fail('read-only query ran synthesis LLM'),
    )

    server = TestServer(app)
    client = TestClient(server)
    await client.start_server()

    try:
        resp = await client.post('/api/query', json={
            'query': 'technical retrieval',
            'source': 'automatic_retrieval',
            'learn': False,
            'respond': False,
        })
        assert resp.status == 200
        data = await resp.json()
        assert data['hebbian_updates'] == []
        assert data['new_concepts'] == []
        assert set(data['routing']) >= {
            'vector_top_score', 'bm25_top_score', 'hybrid_top_score',
            'hybrid_second_score', 'hybrid_margin', 'hybrid_enabled',
        }
        assert data['routing']['hybrid_enabled'] is True
        assert data['activated_notes']
        for note in data['activated_notes']:
            assert note['role'] in {'seed', 'graph_neighbor'}
            assert isinstance(note['hop'], int)
            assert 'final_score' in note
            assert 'hybrid_score' in note
        assert json.dumps(state, sort_keys=True) == before
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_api_query_empty_returns_400(mock_app_setup, monkeypatch):
    from aiohttp.test_utils import TestClient, TestServer

    nodes, edges, collection, state, config, _ = mock_app_setup
    app = _capture_app(monkeypatch, config, nodes, edges, collection, state)

    server = TestServer(app)
    client = TestClient(server)
    await client.start_server()

    try:
        resp = await client.post('/api/query', json={'query': ''})
        assert resp.status == 400
        data = await resp.json()
        assert 'error' in data
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_api_query_missing_field_returns_400(mock_app_setup, monkeypatch):
    from aiohttp.test_utils import TestClient, TestServer

    nodes, edges, collection, state, config, _ = mock_app_setup
    app = _capture_app(monkeypatch, config, nodes, edges, collection, state)

    server = TestServer(app)
    client = TestClient(server)
    await client.start_server()

    try:
        resp = await client.post('/api/query', json={})
        assert resp.status == 400
    finally:
        await client.close()


# ---------------------------------------------------------------------------
# GET / (index)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_api_index_page(mock_app_setup, monkeypatch):
    from aiohttp.test_utils import TestClient, TestServer

    nodes, edges, collection, state, config, _ = mock_app_setup
    app = _capture_app(monkeypatch, config, nodes, edges, collection, state)

    server = TestServer(app)
    client = TestClient(server)
    await client.start_server()

    try:
        resp = await client.get('/')
        assert resp.status == 200
        text = await resp.text()
        assert 'BDH Graph Harness' in text
    finally:
        await client.close()