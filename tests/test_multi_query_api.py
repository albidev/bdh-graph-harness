"""Integration tests for multi-query retrieval via the API."""
import json
import tempfile
import pytest
import chromadb
import harness
import bdh_graph_harness.api.routes as bdh_routes
import bdh_graph_harness.retrieval.attention as bdh_attention_mod
import bdh_graph_harness.retrieval.embeddings as bdh_embeddings_mod


@pytest.fixture
def mock_app_mq_setup(monkeypatch):
    """Build a mock app setup with deterministic multi-query attention."""
    d = tempfile.mkdtemp()

    nodes = {
        'alpha': {'id': 'alpha', 'title': 'Alpha', 'tags': 'concept', 'text': 'Alpha content', 'path': '/fake/alpha.md'},
        'beta': {'id': 'beta', 'title': 'Beta', 'tags': 'concept', 'text': 'Beta content', 'path': '/fake/beta.md'},
        'gamma': {'id': 'gamma', 'title': 'Gamma', 'tags': 'concept', 'text': 'Gamma content', 'path': '/fake/gamma.md'},
        'delta': {'id': 'delta', 'title': 'Delta', 'tags': 'concept', 'text': 'Delta content', 'path': '/fake/delta.md'},
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
    collection = client.get_or_create_collection('test_mq_api', metadata={'hnsw:space': 'cosine'})
    if collection.count() > 0:
        collection.delete(ids=collection.get()['ids'])
    collection.add(
        ids=list(nodes.keys()),
        embeddings=[[1.0, 0.0, 0.0]] * len(nodes),
        documents=[n['text'] for n in nodes.values()],
        metadatas=[{'title': n['title'], 'tags': n['tags']} for n in nodes.values()],
    )

    config = dict(harness.CONFIG)
    config['vault_path'] = d
    config['neurogenesis_enabled'] = False
    config['hybrid_search'] = False
    config['adaptive_threshold'] = False
    config['multi_query_enabled'] = True

    # Deterministic embedding to bypass Ollama — attention imports
    # get_embeddings directly, so patch that binding on the attention module.
    monkeypatch.setattr(bdh_attention_mod, 'get_embeddings', lambda texts: [[1.0, 0.0, 0.0]] * len(texts))

    query_result_map = {
        'test query': {'alpha': 0.90, 'beta': 0.80},
        'variant one': {'beta': 0.85, 'gamma': 0.70},
        'variant two': {'gamma': 0.75, 'delta': 0.65},
    }

    def fake_attention(query, nodes, edges, collection, k=None, max_hop=None, bm25_index=None, hebbian_state=None, routing_meta=None):
        result = dict(query_result_map.get(query, {}))
        if routing_meta is not None:
            top = max(result.values(), default=0.0)
            routing_meta.update({
                'vector_top_score': top,
                'bm25_top_score': 0.0,
                'hybrid_top_score': top,
                'hybrid_enabled': False,
                'hybrid_fusion': None,
                'activation_details': [
                    {'id': nid, 'role': 'seed', 'hop': 0, 'parent_id': None, 'final_score': score}
                    for nid, score in result.items()
                ],
            })
        return result

    monkeypatch.setattr(bdh_attention_mod, 'attention', fake_attention)
    # multi_query.py imported its own reference to attention; patch that too so
    # the multi-query path uses the deterministic fake in tests.
    import bdh_graph_harness.retrieval.multi_query as _mq_mod
    monkeypatch.setattr(_mq_mod, 'attention', fake_attention)
    # routes.py also holds its own imported reference to attention; patch it so
    # single-query path uses the deterministic fake in tests.
    monkeypatch.setattr(bdh_routes, 'attention', fake_attention)
    monkeypatch.setattr(bdh_routes, 'llm_respond', lambda q, a, n: 'Mock LLM response')
    monkeypatch.setattr(bdh_routes, 'extract_new_concepts', lambda r, q, a, n: [])
    monkeypatch.setattr(bdh_routes, 'save_state', lambda vr, s: None)

    return nodes, edges, collection, state, config, d


def _capture_app(monkeypatch, config, nodes, edges, collection, state):
    """Monkeypatch web.run_app to capture the app without starting a server."""
    from aiohttp import web

    captured = {}

    def fake_run_app(app, **kwargs):
        captured['app'] = app

    monkeypatch.setattr('aiohttp.web.run_app', fake_run_app)
    harness.start_api_server(config, nodes, edges, collection, state)
    return captured['app']


@pytest.mark.asyncio
async def test_api_query_single_query_equivalent(mock_app_mq_setup, monkeypatch):
    """A request without query_variants behaves like the legacy single-query path."""
    from aiohttp.test_utils import TestClient, TestServer

    nodes, edges, collection, state, config, _ = mock_app_mq_setup
    # Even when multi_query_enabled is True, omitting query_variants must keep
    # the legacy behavior (no variant expansion) and only activate alpha/beta.
    config['multi_query_enabled'] = True
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
        assert data['routing']['multi_query_variant_count'] == 1
        assert data['routing']['query_variants'][0]['query'] == 'test query'
        assert data['routing']['multi_query_multivariant_hits'] == 0
        # Without explicit variants the path is not a multi-query expansion,
        # so the contract reports enabled=False.
        assert data['routing']['multi_query_enabled'] is False
        activated_ids = {note['id'] for note in data['activated_notes']}
        assert activated_ids == {'alpha', 'beta'}
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_api_query_single_query_opt_in_disabled(mock_app_mq_setup, monkeypatch):
    """When multi_query_enabled is False, query_variants are ignored."""
    from aiohttp.test_utils import TestClient, TestServer

    nodes, edges, collection, state, config, _ = mock_app_mq_setup
    config['multi_query_enabled'] = False
    app = _capture_app(monkeypatch, config, nodes, edges, collection, state)

    server = TestServer(app)
    client = TestClient(server)
    await client.start_server()
    try:
        resp = await client.post('/api/query', json={
            'query': 'test query',
            'query_variants': [
                {'query': 'variant one', 'language': 'en'},
                {'query': 'variant two', 'language': 'it'},
            ],
            'learn': False,
            'respond': False,
        })
        assert resp.status == 200
        data = await resp.json()
        assert data['routing']['multi_query_enabled'] is False
        assert data['routing']['multi_query_variant_count'] == 1
        activated_ids = {note['id'] for note in data['activated_notes']}
        assert activated_ids == {'alpha', 'beta'}
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_api_query_multi_query_fusion_and_provenance(mock_app_mq_setup, monkeypatch):
    """A request with query_variants returns canonical fused notes with provenance."""
    from aiohttp.test_utils import TestClient, TestServer

    nodes, edges, collection, state, config, _ = mock_app_mq_setup
    app = _capture_app(monkeypatch, config, nodes, edges, collection, state)

    server = TestServer(app)
    client = TestClient(server)
    await client.start_server()
    try:
        resp = await client.post('/api/query', json={
            'query': 'test query',
            'query_variants': [
                {'query': 'variant one', 'language': 'en'},
                {'query': 'variant two', 'language': 'it'},
            ],
            'learn': False,
            'respond': False,
        })
        assert resp.status == 200
        data = await resp.json()
        routing = data['routing']
        assert routing['multi_query_enabled'] is True
        assert routing['multi_query_variant_count'] == 3
        assert routing['multi_query_unique_notes'] == 4
        assert routing['multi_query_multivariant_hits'] == 2  # beta (original+variant) and gamma (both variants)
        assert routing['multi_query_fusion'] == 'rrb'
        assert len(routing['query_variants']) == 3

        by_id = {note['id']: note for note in data['activated_notes']}
        assert 'gamma' in by_id
        gamma = by_id['gamma']
        assert gamma.get('variant_hits') == 2
        assert 'matched_by' in gamma
        matched_variants = {m['variant'] for m in gamma['matched_by']}
        assert matched_variants == {'variant-1 (en)', 'variant-2 (it)'}

        # The response must preserve original query for the write/learn path.
        assert data['response'] == ''  # respond=False
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_api_query_multi_query_drops_duplicates_and_bounds(mock_app_mq_setup, monkeypatch):
    """Empty, duplicate, and over-limit variants are bounded/deduplicated."""
    from aiohttp.test_utils import TestClient, TestServer

    nodes, edges, collection, state, config, _ = mock_app_mq_setup
    app = _capture_app(monkeypatch, config, nodes, edges, collection, state)

    server = TestServer(app)
    client = TestClient(server)
    await client.start_server()
    try:
        resp = await client.post('/api/query', json={
            'query': 'test query',
            'query_variants': [
                {'query': 'test query'},  # duplicate of original
                {'query': ''},             # empty
                {'query': 'variant one'},
                {'query': 'variant one'},  # duplicate
                {'query': 'variant two'},
                {'query': 'variant three'},  # over limit
            ],
            'learn': False,
            'respond': False,
        })
        assert resp.status == 200
        data = await resp.json()
        variants = data['routing']['query_variants']
        queries = [v['query'] for v in variants]
        assert queries == ['test query', 'variant one', 'variant two']
        assert data['routing']['multi_query_variant_count'] == 3
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_api_query_multi_query_emits_ordered_ws_event(mock_app_mq_setup, monkeypatch):
    """The activation event contains query_variants and is ordered."""
    from aiohttp.test_utils import TestClient, TestServer

    nodes, edges, collection, state, config, _ = mock_app_mq_setup
    events = []
    app = _capture_app(monkeypatch, config, nodes, edges, collection, state)

    async def capture(event, _clients):
        events.append(event)

    monkeypatch.setattr(bdh_routes, 'broadcast_activation', capture)

    server = TestServer(app)
    client = TestClient(server)
    await client.start_server()
    try:
        resp = await client.post('/api/query', json={
            'query': 'test query',
            'query_variants': [{'query': 'variant one'}],
            'learn': False,
            'respond': False,
        })
        assert resp.status == 200
        activation_events = [e for e in events if e['type'] == 'activation']
        assert len(activation_events) == 1
        event = activation_events[0]
        assert event['sequence'] is not None
        assert event['query'] == 'test query'
        assert event['query_variants'] is not None
        assert len(event['query_variants']) == 2
        assert isinstance(event['activated_notes'], list)
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_api_stream_supports_query_variants(mock_app_mq_setup, monkeypatch):
    """The SSE stream endpoint accepts query_variants and emits them in the init payload."""
    from aiohttp.test_utils import TestClient, TestServer

    nodes, edges, collection, state, config, _ = mock_app_mq_setup
    app = _capture_app(monkeypatch, config, nodes, edges, collection, state)
    monkeypatch.setattr(bdh_routes, 'llm_stream', lambda q, a, n: iter(['token']))

    server = TestServer(app)
    client = TestClient(server)
    await client.start_server()
    try:
        resp = await client.post('/api/stream', json={
            'query': 'test query',
            'query_variants': [{'query': 'variant one'}],
        })
        assert resp.status == 200
        text = await resp.text()
        lines = text.split('\n')
        data_lines = [line[len('data: '):] for line in lines if line.startswith('data: ') and line != 'data: [DONE]']
        assert data_lines
        init = json.loads(data_lines[0])
        assert init['type'] == 'activation'
        assert init['query'] == 'test query'
        assert init['query_variants'] is not None
        assert len(init['query_variants']) == 2
    finally:
        await client.close()
