"""WebSocket connection management and broadcast for the BDH API server."""

import asyncio
import json

from aiohttp import web

__all__ = ["WebSocketManager", "broadcast_activation", "websocket_handler"]


class WebSocketManager:
    """Manage connected WebSocket clients and broadcast events.

    Encapsulates the ``ws_clients`` set and the broadcast/handler logic
    that was originally inline in ``start_api_server()``.
    """

    def __init__(self):
        self.clients: set = set()

    async def broadcast_activation(self, event: dict) -> None:
        """Broadcast an activation event to all connected WebSocket clients."""
        msg = json.dumps(event)
        dead = []
        for ws in self.clients:
            try:
                await ws.send_str(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.clients.discard(ws)

    async def websocket_handler(self, request, app_state: dict) -> web.WebSocketResponse:
        """Handle WebSocket connections for real-time graph visualization."""
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        self.clients.add(ws)

        # Send full graph on connect
        n = app_state['nodes']
        e = app_state['edges']
        s = app_state['state']

        node_list = []
        for note_id, node in n.items():
            node_list.append({
                'id': note_id,
                'title': node['title'],
                'tags': node.get('tags', []),
                'path': node.get('path', ''),
                'text': node.get('text', '')[:200],
            })

        # _resolve_target is in the graph module
        from bdh_graph_harness.graph import _resolve_target

        edge_list = []
        for src, links in e.items():
            for link in links:
                target_id = _resolve_target(link['target'], n)
                if target_id:
                    edge_list.append({
                        'source': src,
                        'target': target_id,
                        'display': link.get('display', ''),
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

        init_msg = {
            'type': 'graph',
            'nodes': node_list,
            'edges': edge_list,
            'hebbian': hebbian_list,
            'stats': {
                'neurons': len(n),
                'synapses': sum(len(links) for links in e.values()),
                'hebbian_synapses': len(s['synapses']),
                'dormant_neurons': len(s.get('dormant_nodes', [])),
                'phantom_links': len(s.get('phantom_links', [])),
            },
        }
        await ws.send_str(json.dumps(init_msg))

        # Listen for messages (ping/pong keepalive)
        try:
            async for msg in ws:
                if msg.type == web.WSMsgType.ERROR:
                    break
        finally:
            self.clients.discard(ws)

        return ws


# ---- Module-level convenience functions (backwards-compatible API) ----

_default_manager = WebSocketManager()


async def broadcast_activation(event: dict, ws_clients: set = None) -> None:
    """Broadcast an activation event.

    If *ws_clients* is provided, broadcast to that set directly (legacy mode
    used by routes that receive a raw set).  Otherwise fall back to the
    module-level default manager.
    """
    if ws_clients is not None:
        msg = json.dumps(event)
        dead = []
        # Copy the set to avoid RuntimeError: Set changed size during iteration
        for ws in list(ws_clients):
            target_vault = event.get('vault_id')
            client_vault = getattr(ws, '_bdh_vault_id', None)
            if target_vault and client_vault and client_vault != target_vault:
                continue
            try:
                await ws.send_str(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            ws_clients.discard(ws)
    else:
        await _default_manager.broadcast_activation(event)


async def websocket_handler(request, app_state: dict, ws_clients: set = None) -> web.WebSocketResponse:
    """WebSocket handler.

    Resolves the target vault from the ``vault_id`` query parameter
    (defaults to the registry's default vault).  Sends the full graph
    on connect and receives ping/pong keepalive messages.

    When *ws_clients* is supplied the handler registers/deregisters against
    that set (matching the original closure-based design where ``ws_clients``
    was captured from the enclosing ``start_api_server`` scope).
    """
    ws = web.WebSocketResponse(heartbeat=30.0)  # ping every 30s to keep alive
    await ws.prepare(request)

    if ws_clients is not None:
        ws_clients.add(ws)
    else:
        _default_manager.clients.add(ws)

    # Resolve vault context (default if vault_id not specified)
    vault_id = request.query.get('vault_id') or None
    registry = app_state.get('registry')
    event_sequence = 0
    if registry is not None:
        try:
            ctx = registry.get(vault_id)
        except KeyError:
            ctx = registry.get()  # fall back to default
        n = ctx.nodes
        e = ctx.edges
        s = ctx.state
        vault_id_label = ctx.config.id
        event_sequence = ctx.event_sequence
    else:
        # Legacy fallback (app_state has flat structure)
        n = app_state.get('nodes', {})
        e = app_state.get('edges', {})
        s = app_state.get('state', {'synapses': {}})
        vault_id_label = app_state.get('vault_id', 'default')

    try:
        setattr(ws, '_bdh_vault_id', vault_id_label)
    except Exception:
        pass

    node_list = []
    for note_id, node in n.items():
        node_list.append({
            'id': note_id,
            'title': node['title'],
            'tags': node.get('tags', []),
            'path': node.get('path', ''),
            'text': node.get('text', '')[:200],
        })

    from bdh_graph_harness.graph import _resolve_target

    edge_list = []
    for src, links in e.items():
        for link in links:
            target_id = _resolve_target(link['target'], n)
            if target_id:
                edge_list.append({
                    'source': src,
                    'target': target_id,
                    'display': link.get('display', ''),
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

    init_msg = {
        'type': 'graph',
        'sequence': event_sequence,
        'vault_id': vault_id_label,
        'nodes': node_list,
        'edges': edge_list,
        'hebbian': hebbian_list,
        'stats': {
            'neurons': len(n),
            'synapses': sum(len(links) for links in e.values()),
            'hebbian_synapses': len(s['synapses']),
            'dormant_neurons': len(s.get('dormant_nodes', [])),
            'phantom_links': len(s.get('phantom_links', [])),
        },
    }
    await ws.send_str(json.dumps(init_msg))

    # Keepalive: send heartbeat every 30s to prevent idle disconnect
    async def keepalive():
        try:
            while not ws.closed:
                await asyncio.sleep(30)
                if not ws.closed:
                    await ws.send_str(json.dumps({'type': 'ping'}))
        except (asyncio.CancelledError, ConnectionError):
            pass

    keepalive_task = asyncio.create_task(keepalive())

    try:
        async for msg in ws:
            if msg.type == web.WSMsgType.ERROR:
                break
    finally:
        keepalive_task.cancel()
        if ws_clients is not None:
            ws_clients.discard(ws)
        else:
            _default_manager.clients.discard(ws)

    return ws