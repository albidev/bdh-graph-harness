"""Route handlers for the BDH Graph Harness API server.

All handlers were originally closures inside ``start_api_server()`` in
harness.py.  They have been extracted into top-level async functions that
accept ``app_state`` (and where needed ``ws_clients``) as explicit parameters.

``setup_routes(app, app_state, ws_clients)`` registers every route on the
aiohttp ``Application``.
"""

import json
from datetime import datetime

from aiohttp import web

from bdh_graph_harness.config import CONFIG, logger
from bdh_graph_harness.visualization import render_viz_html
from bdh_graph_harness.retrieval.attention import attention
from bdh_graph_harness.memory import hebbian_update, save_state
from bdh_graph_harness.llm import llm_respond, llm_stream
from bdh_graph_harness.neurogenesis import extract_new_concepts, create_note
from bdh_graph_harness.graph import _resolve_target
from bdh_graph_harness.api.ws import broadcast_activation

__all__ = [
    "index",
    "api_stats",
    "api_graph",
    "api_hebbian",
    "api_query",
    "api_stream",
    "api_refresh",
    "run_attention_and_plasticity",
    "run_neurogenesis",
    "setup_routes",
]


# ---------------------------------------------------------------------------
# Route handlers
# ---------------------------------------------------------------------------

async def index(request, app_state: dict) -> web.Response:
    """Serve the vis.js visualization page."""
    n = app_state['nodes']
    e = app_state['edges']
    s = app_state['state']
    html = render_viz_html(
        len(n),
        sum(len(links) for links in e.values()),
        len(s['synapses']),
    )
    return web.Response(text=html, content_type='text/html')


async def api_stats(request, app_state: dict) -> web.Response:
    """Return graph stats + Hebbian summary as JSON."""
    s = app_state['state']
    n = app_state['nodes']
    e = app_state['edges']
    stats = {
        'neurons': len(n),
        'synapses': sum(len(links) for links in e.values()),
        'avg_degree': sum(len(links) for links in e.values()) / max(len(e), 1),
        'hebbian_synapses': len(s['synapses']),
        'queries_processed': s.get('queries', 0),
        'top_hebbian': [],
    }
    if s['synapses']:
        sorted_syn = sorted(s['synapses'].items(), key=lambda x: -x[1]['weight'])[:10]
        stats['top_hebbian'] = [
            {'pair': key, 'weight': syn['weight'], 'frequency': syn['frequency']}
            for key, syn in sorted_syn
        ]
    return web.json_response(stats)


async def api_graph(request, app_state: dict) -> web.Response:
    """Return nodes and edges as JSON for visualization."""
    n = app_state['nodes']
    e = app_state['edges']

    node_list = []
    for note_id, node in n.items():
        node_list.append({
            'id': note_id,
            'title': node['title'],
            'tags': node['tags'],
            'path': node.get('path', ''),
            'text': node.get('text', '')[:200],
        })

    edge_list = []
    for src, links in e.items():
        for link in links:
            target_id = _resolve_target(link['target'], n)
            if target_id:
                edge_list.append({
                    'source': src,
                    'target': target_id,
                    'display': link['display'],
                })

    return web.json_response({'nodes': node_list, 'edges': edge_list})


async def api_hebbian(request, app_state: dict) -> web.Response:
    """Return Hebbian synaptic state as JSON."""
    s = app_state['state']
    synapses = []
    for key, syn in sorted(s['synapses'].items(), key=lambda x: -x[1]['weight']):
        a, b = key.split('|')
        synapses.append({
            'note_a': a,
            'note_b': b,
            'weight': syn['weight'],
            'frequency': syn['frequency'],
            'last_coactivated': syn['last_coactivated'],
        })
    return web.json_response({
        'total': len(s['synapses']),
        'queries': s.get('queries', 0),
        'synapses': synapses,
    })


async def run_attention_and_plasticity(
    query: str, app_state: dict, ws_clients: set
) -> tuple[dict, list, list]:
    """Run the attention + Hebbian plasticity phase shared by all query routes.

    Steps:
      1. Attention (with optional BM25 hybrid search).
      2. Build the ``activated_notes`` list (id, title, score).
      3. Online plasticity: Hebbian update + persist state.
      4. Build ``hebbian_updates`` from the current synaptic state.
      5. Broadcast an activation event to WebSocket clients.

    Returns ``(active, activated_notes, hebbian_updates)``.
    """
    config = app_state['config']
    n = app_state['nodes']
    e = app_state['edges']
    coll = app_state['collection']
    bm25 = app_state.get('bm25_index')

    # Attention (with hybrid search if enabled)
    active = attention(query, n, e, coll, bm25_index=bm25)

    activated_notes = []
    if active:
        for note_id, score in sorted(active.items(), key=lambda x: -x[1]):
            node = n.get(note_id)
            activated_notes.append({
                'id': note_id,
                'title': node['title'] if node else note_id,
                'score': round(score, 4),
            })

    # Online plasticity: Hebbian update immediately after attention
    updated_keys = set()
    if CONFIG.get('online_plasticity', True):
        app_state['state'], updated_keys = hebbian_update(active, app_state['state'])
        save_state(config['vault_path'], app_state['state'])

    # Collect ONLY hebbian synapses updated in this query (for pulse animation)
    hebbian_updates = []
    for key in updated_keys:
        syn = app_state['state']['synapses'].get(key)
        if syn:
            hebbian_updates.append({
                'pair': key,
                'weight': syn['weight'],
                'frequency': syn.get('frequency', 0),
            })

    # Broadcast activation event to WebSocket clients
    activation_event = {
        'type': 'activation',
        'query': query,
        'activated_notes': activated_notes,
        'hebbian_updates': hebbian_updates,
        'hebbian_synapses': len(app_state['state']['synapses']),
        'queries_processed': app_state['state'].get('queries', 0),
        'neuron_count': len(app_state['nodes']),
        'synapse_count': sum(len(links) for links in app_state['edges'].values()),
        'timestamp': datetime.now().isoformat(),
    }
    await broadcast_activation(activation_event, ws_clients)

    return active, activated_notes, hebbian_updates


def run_neurogenesis(
    response_text: str, query: str, active: dict, app_state: dict
) -> list:
    """Run neurogenesis on a completed LLM response.

    If ``CONFIG['neurogenesis_enabled']`` is True, extracts new concepts
    from the response and creates notes for each one in the vault.

    Returns the ``new_concepts_list`` (list of ``{'id', 'title'}`` dicts).
    """
    new_concepts_list = []
    if CONFIG.get('neurogenesis_enabled', True):
        n = app_state['nodes']
        config = app_state['config']
        active_titles = [n[nid]['title'] for nid in active if nid in n]
        new_concepts = extract_new_concepts(response_text, query, active, n)
        for concept in new_concepts:
            title = concept.get('title', '').strip()
            definition = concept.get('definition', '').strip()
            if not title or not definition:
                continue
            vault_root = config['vault_path']
            new_note_id = create_note(vault_root, title, definition, active_titles, query)
            if new_note_id:
                new_concepts_list.append({
                    'id': new_note_id,
                    'title': title,
                    'source_notes': active_titles[:3],
                })
    return new_concepts_list


async def api_query(request, app_state: dict, ws_clients: set) -> web.Response:
    """Accept {\"query\": \"...\"} and run the full BDH pipeline.

    Phase 3.2: Online plasticity — Hebbian update happens right after
    attention (not after LLM), so synaptic state reflects what was
    activated, not what the LLM happened to use.
    """
    try:
        data = await request.json()
    except Exception:
        return web.json_response({'error': 'Invalid JSON body'}, status=400)

    query = data.get('query', '').strip()
    if not query:
        return web.json_response({'error': 'Missing "query" field'}, status=400)

    # Shared attention + plasticity phase (broadcasts activation to WS)
    active, activated_notes, hebbian_updates = await run_attention_and_plasticity(
        query, app_state, ws_clients
    )

    # LLM response
    n = app_state['nodes']
    response_text = llm_respond(query, active, n)

    # Neurogenesis on the full response
    new_concepts_list = run_neurogenesis(response_text, query, active, app_state)

    return web.json_response({
        'response': response_text,
        'activated_notes': activated_notes,
        'new_concepts': new_concepts_list,
        'hebbian_synapses': len(app_state['state']['synapses']),
        'hebbian_updates': hebbian_updates,
        'queries_processed': app_state['state'].get('queries', 0),
        'neuron_count': len(app_state['nodes']),
        'synapse_count': sum(len(links) for links in app_state['edges'].values()),
    })


async def api_stream(request, app_state: dict, ws_clients: set) -> web.StreamResponse:
    """Streaming query endpoint using Server-Sent Events.

    Phase 3.2: Online plasticity with streaming.
    Streams tokens as they arrive from the LLM, with Hebbian update
    performed right after attention (before streaming starts).

    SSE format: data: {json}\\n\\n
    """
    try:
        data = await request.json()
    except Exception:
        return web.json_response({'error': 'Invalid JSON body'}, status=400)

    query = data.get('query', '').strip()
    if not query:
        return web.json_response({'error': 'Missing "query" field'}, status=400)

    n = app_state['nodes']

    # Shared attention + plasticity phase (broadcasts activation to WS)
    active, activated_notes, hebbian_updates = await run_attention_and_plasticity(
        query, app_state, ws_clients
    )

    # Stream LLM response via SSE
    resp = web.StreamResponse(
        status=200,
        headers={
            'Content-Type': 'text/event-stream',
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
        },
    )
    await resp.prepare(request)

    # Send activation info first
    init_data = json.dumps({
        'type': 'activation',
        'activated_notes': activated_notes,
        'hebbian_synapses': len(app_state['state']['synapses']),
    })
    await resp.write(f"data: {init_data}\n\n".encode())

    # Stream tokens
    full_response = []
    try:
        for token in llm_stream(query, active, n):
            full_response.append(token)
            token_data = json.dumps({'type': 'token', 'content': token})
            await resp.write(f"data: {token_data}\n\n".encode())
    except Exception as exc:
        logger.warning(f"Stream interrupted: {exc}")

    # Neurogenesis on the full response
    response_text = ''.join(full_response)
    new_concepts_list = run_neurogenesis(response_text, query, active, app_state)

    # Send final event with new concepts
    done_data = json.dumps({
        'type': 'done',
        'new_concepts': new_concepts_list,
        'hebbian_synapses': len(app_state['state']['synapses']),
    })
    await resp.write(f"data: {done_data}\n\n".encode())
    await resp.write(b'data: [DONE]\n\n')

    return resp


async def api_refresh(request, app_state: dict) -> web.Response:
    """Force refresh all embeddings."""
    config = app_state['config']
    n = app_state['nodes']
    vault_root = config['vault_path']

    # compute_all_embeddings lives in the retrieval module
    from bdh_graph_harness.retrieval import compute_all_embeddings
    coll = compute_all_embeddings(n, vault_root, force_refresh=True)
    app_state['collection'] = coll
    return web.json_response({'status': 'ok', 'embeddings': coll.count()})


async def api_refresh_graph(request, app_state: dict, ws_clients: set) -> web.Response:
    """Full graph rebuild: re-read vault, rebuild graph, re-embed, notify WS clients."""
    import asyncio
    config = app_state['config']
    vault_root = config['vault_path']

    # 1. Rebuild graph from vault (no cache — force fresh read)
    from bdh_graph_harness.graph.builder import build_graph
    nodes, edges = build_graph(vault_root, use_cache=False)

    old_count = len(app_state['nodes'])
    app_state['nodes'] = nodes
    app_state['edges'] = edges

    # 2. Re-embed all notes
    from bdh_graph_harness.retrieval import compute_all_embeddings
    coll = compute_all_embeddings(nodes, vault_root, force_refresh=True)
    app_state['collection'] = coll

    # 3. Rebuild BM25 index if hybrid search is enabled
    if config.get('hybrid_search', False):
        from bdh_graph_harness.retrieval.bm25 import BM25Index
        app_state['bm25'] = BM25Index(nodes)

    new_count = len(nodes)
    delta = new_count - old_count

    # 4. Notify all WebSocket clients about the graph update
    from bdh_graph_harness.api.ws import broadcast_activation
    event = {
        'type': 'graph_refresh',
        'neurons': new_count,
        'synapses': len(edges),
        'delta': delta,
        'message': f'Graph refreshed: {new_count} neurons ({delta:+d})',
    }
    await broadcast_activation(event, ws_clients)

    return web.json_response({
        'status': 'ok',
        'neurons': new_count,
        'synapses': len(edges),
        'embeddings': coll.count(),
        'delta': delta,
    })


# ---------------------------------------------------------------------------
# Route registration
# ---------------------------------------------------------------------------

def setup_routes(app: web.Application, app_state: dict, ws_clients: set) -> None:
    """Register all API routes on the aiohttp application."""

    from bdh_graph_harness.api.ws import websocket_handler

    async def _index(request):
        return await index(request, app_state)

    async def _ws(request):
        return await websocket_handler(request, app_state, ws_clients)

    async def _stats(request):
        return await api_stats(request, app_state)

    async def _graph(request):
        return await api_graph(request, app_state)

    async def _hebbian(request):
        return await api_hebbian(request, app_state)

    async def _query(request):
        return await api_query(request, app_state, ws_clients)

    async def _stream(request):
        return await api_stream(request, app_state, ws_clients)

    async def _refresh(request):
        return await api_refresh(request, app_state)

    async def _refresh_graph(request):
        return await api_refresh_graph(request, app_state, ws_clients)

    app.router.add_get('/', _index)
    app.router.add_get('/ws', _ws)
    app.router.add_get('/api/stats', _stats)
    app.router.add_get('/api/graph', _graph)
    app.router.add_get('/api/hebbian', _hebbian)
    app.router.add_post('/api/query', _query)
    app.router.add_post('/api/stream', _stream)
    app.router.add_post('/api/refresh', _refresh)
    app.router.add_post('/api/refresh-graph', _refresh_graph)