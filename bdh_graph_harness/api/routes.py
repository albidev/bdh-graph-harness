"""Route handlers for the BDH Graph Harness API server.

All handlers were originally closures inside ``start_api_server()`` in
harness.py.  They have been extracted into top-level async functions that
accept ``app_state`` (and where needed ``ws_clients``) as explicit parameters.

``setup_routes(app, app_state, ws_clients)`` registers every route on the
aiohttp ``Application``.
"""

import asyncio
import json
import os
from datetime import datetime

from aiohttp import web

from bdh_graph_harness.config import CONFIG, logger
from bdh_graph_harness.visualization import render_viz_html, get_template_path
from bdh_graph_harness.retrieval.attention import attention
from bdh_graph_harness.memory import hebbian_update, save_state
from bdh_graph_harness.memory.consolidation import consolidate, consolidation_stats
from bdh_graph_harness.llm import llm_respond, llm_stream
from bdh_graph_harness.neurogenesis import extract_new_concepts, create_note
from bdh_graph_harness.graph import _resolve_target
from bdh_graph_harness.api.ws import broadcast_activation

__all__ = [
    "index",
    "api_stats",
    "api_graph",
    "api_hebbian",
    "api_quality",
    "api_query",
    "api_stream",
    "api_refresh",
    "api_consolidate",
    "api_consolidation_stats",
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
    dormant = s.get('dormant_nodes', set())
    stats = {
        'neurons': len(n),
        'synapses': sum(len(links) for links in e.values()),
        'avg_degree': sum(len(links) for links in e.values()) / max(len(e), 1),
        'hebbian_synapses': len(s['synapses']),
        'queries_processed': s.get('queries', 0),
        'dormant_neurons': len(dormant),
        'active_neurons': len(n) - len(dormant),
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
    """Return nodes, edges, hebbian synapses, and stats as JSON for visualization."""
    n = app_state['nodes']
    e = app_state['edges']
    s = app_state['state']
    dormant = s.get('dormant_nodes', set())
    nq = s.get('node_quality', {})

    node_list = []
    for note_id, node in n.items():
        quality = nq.get(note_id, {})
        node_list.append({
            'id': note_id,
            'title': node['title'],
            'tags': node['tags'],
            'path': node.get('path', ''),
            'text': node.get('text', '')[:200],
            'dormant': note_id in dormant,
            'quality_score': quality.get('score', 0.0),
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

    hebbian_list = []
    for key, syn in s['synapses'].items():
        a, b = key.split('|')
        hebbian_list.append({
            'note_a': a,
            'note_b': b,
            'weight': syn['weight'],
            'frequency': syn.get('frequency', 0),
        })

    phantom_list = s.get('phantom_links', [])

    return web.json_response({
        'nodes': node_list,
        'edges': edge_list,
        'hebbian': hebbian_list,
        'phantom': phantom_list,
        'stats': {
            'neurons': len(n),
            'synapses': sum(len(links) for links in e.values()),
            'hebbian_synapses': len(s['synapses']),
            'dormant_neurons': len(s.get('dormant_nodes', [])),
            'phantom_links': len(phantom_list),
        },
    })


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
    query: str, app_state: dict, ws_clients: set, source: str | None = None
) -> tuple[dict, list, list]:
    """Run the attention + Hebbian plasticity phase shared by all query routes.

    Steps:
      1. Attention (with optional BM25 hybrid search).
      2. Build the ``activated_notes`` list (id, title, score).
      3. Online plasticity: Hebbian update + persist state.
      4. Build ``hebbian_updates`` from the current synaptic state.
      5. Broadcast an activation event to WebSocket clients.

    All blocking operations (attention, hebbian_update, save_state) are
    offloaded to ``asyncio.to_thread`` to avoid freezing the event loop.
    An ``asyncio.Lock`` protects ``app_state['state']`` from concurrent
    mutation by overlapping queries or the vault watcher.

    Args:
        source: When ``"assistant_response"``, Hebbian dampening is applied
                to prevent echo-loop reinforcement.

    Returns ``(active, activated_notes, hebbian_updates)``.
    """
    config = app_state['config']
    n = app_state['nodes']
    e = app_state['edges']
    coll = app_state['collection']
    bm25 = app_state.get('bm25_index')
    state_lock = app_state.get('state_lock')

    # Attention — run blocking I/O (embeddings) in a thread
    active = await asyncio.to_thread(
        attention, query, n, e, coll, None, None, bm25, app_state.get('state')
    )

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
    pruned_count = 0
    if CONFIG.get('online_plasticity', True):
        # Acquire lock, then run hebbian_update + save_state in a thread
        if state_lock:
            async with state_lock:
                app_state['state'], updated_keys, pruned_count = await asyncio.to_thread(
                    hebbian_update, active, app_state['state'], n, source
                )
                await asyncio.to_thread(
                    save_state, config['vault_path'], app_state['state']
                )
        else:
            # Fallback: no lock (e.g. MCP fallback path without server)
            app_state['state'], updated_keys, pruned_count = await asyncio.to_thread(
                hebbian_update, active, app_state['state'], n, source
            )
            await asyncio.to_thread(
                save_state, config['vault_path'], app_state['state']
            )

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
        'dormant_count': len(app_state['state'].get('dormant_nodes', set())),
        'pruned_count': pruned_count,
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
    """Accept {"query": "..."} and run the full BDH pipeline.

    Phase 3.2: Online plasticity — Hebbian update happens right after
    attention (not after LLM), so synaptic state reflects what was
    activated, not what the LLM happened to use.

    Optional fields:
      - source: "assistant_response" triggers Hebbian dampening (echo-loop prevention)
      - user_prompt: original user question, combined with query for LLM context
    """
    try:
        data = await request.json()
    except Exception:
        return web.json_response({'error': 'Invalid JSON body'}, status=400)

    query = data.get('query', '').strip()
    if not query:
        return web.json_response({'error': 'Missing "query" field'}, status=400)

    source = data.get('source')
    user_prompt = data.get('user_prompt', '').strip()

    # Combine user prompt + assistant response for richer LLM context
    # when user_prompt is available (prevents losing question→answer association)
    llm_query = query
    if user_prompt:
        llm_query = f"{user_prompt}\n\n---\n\n{query}"

    # Shared attention + plasticity phase (broadcasts activation to WS)
    # Pass source for echo-loop dampening
    active, activated_notes, hebbian_updates = await run_attention_and_plasticity(
        query, app_state, ws_clients, source=source
    )

    # LLM response — use combined query for richer context (run in thread to avoid blocking)
    n = app_state['nodes']
    response_text = await asyncio.to_thread(llm_respond, llm_query, active, n)

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

    source = data.get('source')
    user_prompt = data.get('user_prompt', '').strip()

    # Combine user prompt + assistant response for richer LLM context
    llm_query = query
    if user_prompt:
        llm_query = f"{user_prompt}\n\n---\n\n{query}"

    n = app_state['nodes']

    # Shared attention + plasticity phase (broadcasts activation to WS)
    active, activated_notes, hebbian_updates = await run_attention_and_plasticity(
        query, app_state, ws_clients, source=source
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

    # Stream tokens — run blocking generator in a thread, feed to SSE
    full_response = []
    loop = asyncio.get_running_loop()

    def _run_stream():
        """Run the blocking LLM stream in a thread, collecting tokens."""
        tokens = []
        try:
            for token in llm_stream(llm_query, active, n):
                tokens.append(token)
        except Exception as exc:
            logger.warning(f"Stream interrupted: {exc}")
        return tokens

    # Execute the blocking stream call in a thread
    tokens = await asyncio.to_thread(_run_stream)
    for token in tokens:
        full_response.append(token)
        token_data = json.dumps({'type': 'token', 'content': token})
        await resp.write(f"data: {token_data}\n\n".encode())

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


async def api_quality(request, app_state: dict) -> web.Response:
    """Return node quality statistics and dormant node list."""
    from bdh_graph_harness.memory.quality import quality_stats

    state = app_state['state']
    stats = quality_stats(state)

    # Include per-node quality details for dormant nodes
    nq = state.get('node_quality', {})
    dormant_details = []
    for nid, q in sorted(nq.items(), key=lambda x: x[1]['score']):
        if q.get('dormant', False):
            dormant_details.append({
                'id': nid,
                'score': q['score'],
                'strong_ratio': q['strong_ratio'],
                'mean_weight': q['mean_weight'],
                'frequency': q['frequency'],
            })

    stats['dormant_nodes'] = dormant_details
    return web.json_response(stats)


async def api_refresh(request, app_state: dict) -> web.Response:
    """Force refresh all embeddings."""
    config = app_state['config']
    n = app_state['nodes']
    vault_root = config['vault_path']

    # compute_all_embeddings lives in the retrieval module
    from bdh_graph_harness.retrieval import compute_all_embeddings
    coll = await asyncio.to_thread(compute_all_embeddings, n, vault_root, False)
    app_state['collection'] = coll
    return web.json_response({'status': 'ok', 'embeddings': coll.count()})


async def api_node_update(request, app_state: dict, ws_clients: set) -> web.Response:
    """Lightweight vault diff: detect new/changed/deleted notes without full rebuild.

    Sends targeted WebSocket events instead of rebuilding the entire graph.
    Much faster than /api/refresh-graph for single-note updates.
    """
    config = app_state['config']
    vault_root = config['vault_path']
    old_nodes = app_state['nodes'] or {}
    old_edges = app_state['edges'] or {}

    # Rebuild graph from vault (fresh read) — in thread
    from bdh_graph_harness.graph.builder import build_graph
    new_nodes, new_edges = await asyncio.to_thread(build_graph, vault_root, False)

    old_ids = set(old_nodes.keys())
    new_ids = set(new_nodes.keys())

    added = sorted(new_ids - old_ids)
    deleted = sorted(old_ids - new_ids)
    changed = []
    for nid in sorted(old_ids & new_ids):
        old_title = old_nodes[nid].get('title', '')
        new_title = new_nodes[nid].get('title', '')
        old_text = old_nodes[nid].get('text', '')
        new_text = new_nodes[nid].get('text', '')
        if old_title != new_title or old_text != new_text:
            changed.append({
                'id': nid,
                'title': new_title,
                'old_title': old_title,
            })

    # Update app state
    app_state['nodes'] = new_nodes
    app_state['edges'] = new_edges

    # Re-embed if needed — in thread
    if added or changed:
        from bdh_graph_harness.retrieval import compute_all_embeddings
        app_state['collection'] = await asyncio.to_thread(
            compute_all_embeddings, new_nodes, vault_root, False
        )

    # Send targeted WS events
    from bdh_graph_harness.api.ws import broadcast_activation

    if added:
        added_node_data = []
        new_concepts = []
        for nid in added:
            node = new_nodes[nid]
            source_notes = []
            node_edges = []
            for link in new_edges.get(nid, []):
                target_id = link['target'] if isinstance(link, dict) else link
                node_edges.append({'source': nid, 'target': target_id})
                if target_id in old_ids:
                    source_notes.append(old_nodes.get(target_id, {}).get('title', target_id))
            new_concepts.append({
                'id': nid,
                'title': node.get('title', nid.split('/')[-1]),
                'source_notes': source_notes[:5],
            })
            added_node_data.append({
                'id': nid,
                'title': node.get('title', nid.split('/')[-1]),
                'tags': node.get('tags', ''),
                'text': node.get('text', ''),
                'path': node.get('path', ''),
                'edges': node_edges,
            })
        await broadcast_activation({
            'type': 'graph_refresh',
            'neurons': len(new_nodes),
            'synapses': sum(len(links) for links in new_edges.values()),
            'delta': len(added) - len(deleted),
            'new_concepts': new_concepts,
            'changed_nodes': changed,
            'deleted_nodes': deleted,
            'added_node_data': added_node_data,
            'message': f'{len(added)} new, {len(changed)} changed, {len(deleted)} deleted',
        }, ws_clients)
    elif changed or deleted:
        await broadcast_activation({
            'type': 'node_update',
            'changed_nodes': changed,
            'deleted_nodes': deleted,
            'message': f'{len(changed)} changed, {len(deleted)} deleted',
        }, ws_clients)

    return web.json_response({
        'status': 'ok',
        'added': len(added),
        'changed': len(changed),
        'deleted': len(deleted),
        'changed_nodes': changed,
        'added_nodes': [{'id': nid, 'title': new_nodes[nid].get('title', '')} for nid in added],
    })


async def api_refresh_graph(request, app_state: dict, ws_clients: set) -> web.Response:
    """Full graph rebuild: re-read vault, rebuild graph, re-embed, notify WS clients.
    Detects new notes and returns them as new_concepts for neurogenesis animation."""
    import asyncio
    config = app_state['config']
    vault_root = config['vault_path']

    # 1. Snapshot old node data before rebuild
    old_node_ids = set(app_state['nodes'].keys()) if app_state['nodes'] else set()
    old_node_titles = {nid: n.get('title', '') for nid, n in (app_state['nodes'] or {}).items()}

    # 2. Rebuild graph from vault (no cache — force fresh read) — in thread
    from bdh_graph_harness.graph.builder import build_graph
    nodes, edges = await asyncio.to_thread(build_graph, vault_root, False)

    app_state['nodes'] = nodes
    app_state['edges'] = edges

    # 3. Detect new notes AND changed notes
    new_node_ids = set(nodes.keys()) - old_node_ids
    changed_nodes = []
    for nid in sorted(old_node_ids & set(nodes.keys())):
        old_title = old_node_titles.get(nid, '')
        new_title = nodes[nid].get('title', '')
        if old_title != new_title:
            changed_nodes.append({'id': nid, 'title': new_title})
    new_concepts = []
    added_node_data = []
    for nid in sorted(new_node_ids):
        node = nodes[nid]
        title = node.get('title', nid.split('/')[-1])
        # Find source notes: existing nodes this new note links TO (outgoing wikilinks)
        source_notes = []
        node_links = edges.get(nid, [])
        node_edges = []
        for t in node_links:
            target_id = t['target'] if isinstance(t, dict) else t
            node_edges.append({'source': nid, 'target': target_id})
            # Targets may lack wiki/ prefix — try both formats
            resolved = target_id if target_id in old_node_ids else ('wiki/' + target_id if ('wiki/' + target_id) in old_node_ids else None)
            if resolved:
                src_node = nodes.get(resolved, {})
                source_notes.append(src_node.get('title', resolved.split('/')[-1]))
        new_concepts.append({
            'id': nid,
            'title': title,
            'source_notes': source_notes[:5],  # cap at 5
        })
        added_node_data.append({
            'id': nid,
            'title': title,
            'tags': node.get('tags', ''),
            'text': node.get('text', ''),
            'path': node.get('path', ''),
            'edges': node_edges,
        })

    # 4. Re-embed all notes — in thread
    from bdh_graph_harness.retrieval import compute_all_embeddings
    coll = await asyncio.to_thread(compute_all_embeddings, nodes, vault_root, False)
    app_state['collection'] = coll

    # 5. Rebuild BM25 index if hybrid search is enabled
    if config.get('hybrid_search', False):
        from bdh_graph_harness.retrieval.bm25 import BM25Index
        app_state['bm25_index'] = BM25Index(nodes)

    new_count = len(nodes)
    delta = new_count - len(old_node_ids)

    # 6. Notify all WebSocket clients with new_concepts for neurogenesis animation
    from bdh_graph_harness.api.ws import broadcast_activation
    event = {
        'type': 'graph_refresh',
        'neurons': new_count,
        'synapses': len(edges),
        'delta': delta,
        'new_concepts': new_concepts,
        'changed_nodes': changed_nodes,
        'added_node_data': added_node_data,
        'message': f'Graph refreshed: {new_count} neurons ({delta:+d}), {len(new_concepts)} new, {len(changed_nodes)} updated',
    }
    await broadcast_activation(event, ws_clients)

    return web.json_response({
        'status': 'ok',
        'neurons': new_count,
        'synapses': len(edges),
        'embeddings': coll.count(),
        'delta': delta,
        'new_concepts': new_concepts,
        'changed_nodes': changed_nodes,
    })


async def api_consolidate(request, app_state: dict, ws_clients: set) -> web.Response:
    """Run a memory consolidation cycle (sleep phase).

    POST /api/consolidate

    Triggers a full consolidation pass:
      1. Synaptic downscaling (global weight reduction).
      2. Structural pruning (delete weak synapses).
      3. Quality re-evaluation.
      4. Stale dormant node removal.

    State is persisted after consolidation. A WebSocket event is broadcast
    to all connected clients with the results.

    Optional JSON body:
      {
        "dry_run": false  -- if true, compute results without mutating state
      }
    """
    import asyncio

    dry_run = False
    try:
        data = await request.json()
        dry_run = data.get('dry_run', False)
    except Exception:
        pass  # no body or invalid JSON — just run normally

    n = app_state['nodes']
    e = app_state.get('edges', {})
    config = app_state['config']

    if dry_run:
        from copy import deepcopy
        state_copy = deepcopy(app_state['state'])
        results = consolidate(state_copy, n, e)
        results['dry_run'] = True
        return web.json_response(results)

    # Real consolidation — mutate state in place (with lock for thread safety)
    state_lock = app_state.get('state_lock')
    if state_lock:
        async with state_lock:
            results = await asyncio.to_thread(consolidate, app_state['state'], n, e)
            await asyncio.to_thread(save_state, config['vault_path'], app_state['state'])
    else:
        results = await asyncio.to_thread(consolidate, app_state['state'], n, e)
        await asyncio.to_thread(save_state, config['vault_path'], app_state['state'])

    # Broadcast consolidation event to WebSocket clients
    from bdh_graph_harness.api.ws import broadcast_activation
    event = {
        'type': 'consolidation',
        **results,
    }
    await broadcast_activation(event, ws_clients)

    return web.json_response(results)


async def api_consolidation_stats(request, app_state: dict) -> web.Response:
    """Return consolidation configuration and cycle count."""
    stats = consolidation_stats(app_state['state'])
    return web.json_response(stats)


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

    async def _node_update(request):
        return await api_node_update(request, app_state, ws_clients)

    async def _quality(request):
        return await api_quality(request, app_state)

    async def _consolidate(request):
        return await api_consolidate(request, app_state, ws_clients)

    async def _consolidation_stats(request):
        return await api_consolidation_stats(request, app_state)

    app.router.add_get('/', _index)
    app.router.add_get('/ws', _ws)

    # Serve static files (CSS, JS) from the templates directory
    templates_dir = os.path.dirname(get_template_path())
    app.router.add_static('/static', templates_dir, show_index=False)

    app.router.add_get('/api/stats', _stats)
    app.router.add_get('/api/graph', _graph)
    app.router.add_get('/api/hebbian', _hebbian)
    app.router.add_get('/api/quality', _quality)
    app.router.add_get('/api/consolidation-stats', _consolidation_stats)
    app.router.add_post('/api/query', _query)
    app.router.add_post('/api/stream', _stream)
    app.router.add_post('/api/refresh', _refresh)
    app.router.add_post('/api/refresh-graph', _refresh_graph)
    app.router.add_post('/api/node-update', _node_update)
    app.router.add_post('/api/consolidate', _consolidate)