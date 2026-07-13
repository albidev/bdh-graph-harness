"""API server startup for the BDH Graph Harness.

Supports both single-vault (legacy) and multi-vault modes via
:class:`~bdh_graph_harness.vaults.VaultRegistry`.

The ``app_state`` passed to all route handlers is::

    {
        'registry': VaultRegistry,
        'config': dict,        # global config (for server-level settings)
    }

Route handlers resolve the per-request vault via ``registry.get(vault_id)``.
"""

import asyncio
import logging

from aiohttp import web

from bdh_graph_harness.config import CONFIG
from bdh_graph_harness.api.routes import setup_routes

logger = logging.getLogger("bdh-server")

__all__ = ["start_api_server"]


def _build_registry(config, nodes, edges, collection, state):
    """Build a VaultRegistry from the server startup arguments.

    For single-vault (legacy) mode: wraps the pre-built data into a
    ``VaultContext`` registered as the default vault.

    For multi-vault mode (config has ``vaults:`` key): calls
    ``registry.load_all()`` to build graph/embeddings/state per vault.
    Pre-built data is ignored in multi-vault mode — it was built for the
    *single* ``vault_path`` entry, which may not match the multi-vault list.
    """
    from bdh_graph_harness.vaults import (
        VaultRegistry, VaultContext, normalize_vault_configs,
    )
    from bdh_graph_harness.retrieval import BM25Index

    registry = VaultRegistry(config)

    if config.get('vaults'):
        # Multi-vault mode: load everything from the config
        registry.load_all()
    else:
        # Single-vault (legacy) mode: wrap pre-built data
        vault_configs = normalize_vault_configs(config)
        vc = vault_configs[0]

        bm25_idx = None
        if config.get('hybrid_search', False):
            print("📊 Building BM25 index for hybrid search...")
            bm25_idx = BM25Index(
                nodes,
                k1=config.get('bm25_k1', 1.5),
                b=config.get('bm25_b', 0.75),
            )
            print(f"   ✓ BM25: {bm25_idx.N} docs, {len(bm25_idx.df)} terms")

        ctx = VaultContext(
            config=vc,
            nodes=nodes,
            edges=edges,
            collection=collection,
            state=state,
            bm25_index=bm25_idx,
        )
        registry.register_context(vc.id, ctx)

    return registry


def _make_watcher_callback(ctx, app_state, ws_clients):
    """Factory for the vault watcher callback — avoids closure capture bugs."""

    async def _trigger_node_update_unlocked():
        from bdh_graph_harness.graph.federated import build_configured_graph
        from bdh_graph_harness.api.ws import broadcast_activation
        from bdh_graph_harness.retrieval import compute_all_embeddings

        vault_path = ctx.config.path
        old_nodes = ctx.nodes or {}

        new_nodes, new_edges, unresolved = await asyncio.to_thread(
            build_configured_graph,
            ctx.config.settings,
            use_cache=False,
        )

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
                changed.append({'id': nid, 'title': new_title, 'old_title': old_title})

        print(
            f"👁️  trigger_node_update [{ctx.config.id}]: "
            f"added={len(added)} changed={len(changed)} deleted={len(deleted)}",
            flush=True,
        )
        if not added and not changed and not deleted:
            print(f"👁️  trigger_node_update [{ctx.config.id}]: no changes, skipping", flush=True)
            return

        ctx.nodes = new_nodes
        ctx.edges = new_edges
        ctx.state['unresolved_links'] = unresolved

        if added or changed or deleted:
            ctx.collection = await asyncio.to_thread(
                compute_all_embeddings, new_nodes, vault_path, False,
                chroma_path=ctx.config.chroma_path,
                collection_name=ctx.config.chroma_collection,
                config=ctx.config.settings,
            )
            if ctx.config.settings.get('hybrid_search', False):
                from bdh_graph_harness.retrieval.bm25 import BM25Index
                ctx.bm25_index = BM25Index(new_nodes)

        if added:
            added_node_data = []
            new_concepts = []
            for nid in added:
                node = new_nodes[nid]
                node_edges = []
                source_notes = []
                for link in new_edges.get(nid, []):
                    if isinstance(link, dict):
                        target_id = link['target']
                        edge_payload = {
                            'source': nid,
                            'target': target_id,
                            'type': link.get('type', 'wikilink'),
                            'weight': link.get('weight', 1.0),
                            'relation': link.get('relation'),
                            'group_id': link.get('group_id'),
                        }
                    else:
                        target_id = link
                        edge_payload = {'source': nid, 'target': target_id, 'type': 'wikilink'}
                    node_edges.append(edge_payload)
                    if target_id in old_ids:
                        source_notes.append(old_nodes.get(target_id, {}).get('title', target_id))
                added_node_data.append({
                    'id': nid,
                    'title': node.get('title', nid.split('/')[-1]),
                    'tags': node.get('tags', ''),
                    'text': node.get('text', ''),
                    'path': node.get('path', ''),
                    'absolute_path': node.get('absolute_path', node.get('path', '')),
                    'relative_path': node.get('relative_path', ''),
                    'source_id': node.get('source_id', 'vault'),
                    'source_type': node.get('source_type', 'vault'),
                    'project_group': node.get('project_group'),
                    'writable': node.get('writable', True),
                    'edges': node_edges,
                })
                new_concepts.append({
                    'id': nid,
                    'title': node.get('title', nid.split('/')[-1]),
                    'source_notes': source_notes[:5],
                })
            ctx.event_sequence += 1
            await broadcast_activation({
                'type': 'graph_refresh',
                'sequence': ctx.event_sequence,
                'vault_id': ctx.config.id,
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
            ctx.event_sequence += 1
            await broadcast_activation({
                'type': 'node_update',
                'sequence': ctx.event_sequence,
                'vault_id': ctx.config.id,
                'changed_nodes': changed,
                'deleted_nodes': deleted,
                'message': f'{len(changed)} changed, {len(deleted)} deleted',
            }, ws_clients)

        logger.info(
            f"Vault watcher [{ctx.config.id}]: "
            f"{len(added)} new, {len(changed)} changed, {len(deleted)} deleted"
        )

    async def trigger_node_update():
        async with ctx.runtime_lock:
            return await _trigger_node_update_unlocked()

    return trigger_node_update


def start_api_server(config, nodes, edges, collection, state):
    """Start an aiohttp web API server for the BDH Graph Harness.

    Parameters
    ----------
    config:
        Merged configuration dict.
    nodes, edges:
        Pre-built graph (used in single-vault mode; ignored in multi-vault mode).
    collection:
        Pre-built ChromaDB collection (single-vault mode).
    state:
        Pre-loaded Hebbian state dict (single-vault mode).
    """

    # Monkeypatch tcp_keepalive to avoid OSError [Errno 22] on macOS/Tailscale
    try:
        import aiohttp.tcp_helpers as _tcp_helpers
        _orig = _tcp_helpers.tcp_keepalive
        def _safe_keepalive(transport):
            try:
                _orig(transport)
            except OSError:
                pass
        _tcp_helpers.tcp_keepalive = _safe_keepalive
    except Exception:
        pass

    registry = _build_registry(config, nodes, edges, collection, state)

    # Shared mutable state container used by all route handlers
    app_state = {
        'registry': registry,
        'config': config,
    }

    # Global set of connected WebSocket clients (shared across all vaults)
    ws_clients = set()

    # Build aiohttp app and register routes
    app = web.Application()

    # Auth middleware
    auth_token = config.get('api_auth_token', '')
    if auth_token:
        @web.middleware
        async def auth_middleware(request, handler):
            if request.path == '/ws':
                token = request.query.get('token', '')
                if token != auth_token:
                    return web.json_response({'error': 'Unauthorized'}, status=401)
                return await handler(request)
            if request.path.startswith('/api/'):
                auth = request.headers.get('Authorization', '')
                if not auth.startswith('Bearer ') or auth[7:] != auth_token:
                    return web.json_response({'error': 'Unauthorized'}, status=401)
            return await handler(request)
        app.middlewares.append(auth_middleware)

    setup_routes(app, app_state, ws_clients)

    # Start one watcher per vault
    watchers = []
    try:
        from bdh_graph_harness.api.watcher import VaultWatcher

        for ctx in registry.list():
            try:
                callback = _make_watcher_callback(ctx, app_state, ws_clients)
                from bdh_graph_harness.graph.sources import sources_from_config
                watcher_sources = sources_from_config(ctx.config.settings)
                watcher = VaultWatcher(
                    ctx.config.path,
                    callback,
                    sources=watcher_sources,
                )
                ctx.watcher = watcher
                watchers.append((ctx, watcher))
            except Exception as e:
                print(f"⚠️  Vault watcher setup failed for '{ctx.config.id}': {e}", flush=True)

    except ImportError as e:
        print(f"⚠️  Vault watcher disabled: {e}", flush=True)
    except Exception as e:
        print(f"⚠️  Vault watcher global setup failed: {e}", flush=True)

    host = config['api_host']
    port = config['api_port']
    print(f"🌐 BDH Graph Harness API server starting on http://{host}:{port}")

    if watchers:
        async def _start_watchers(app):
            loop = asyncio.get_running_loop()
            for ctx, watcher in watchers:
                watcher.start(loop)
                print(f"👁️  Vault watcher active on {ctx.config.path} [{ctx.config.id}]", flush=True)

        async def _stop_watchers(app):
            for ctx, watcher in watchers:
                watcher.stop()

        app.on_startup.append(_start_watchers)
        app.on_cleanup.append(_stop_watchers)

    web.run_app(app, host=host, port=port)
