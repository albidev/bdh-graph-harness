"""API server startup for the BDH Graph Harness.

The monkeypatch, BM25 index build, ``app_state`` dict, and ``web.run_app``
call remain here.  Route handlers have been moved to ``routes.py`` and
WebSocket logic to ``ws.py``.
"""

import asyncio
import logging

from aiohttp import web

from bdh_graph_harness.config import CONFIG
from bdh_graph_harness.api.routes import setup_routes

logger = logging.getLogger("bdh-server")

__all__ = ["start_api_server"]


def start_api_server(config, nodes, edges, collection, state):
    """Start an aiohttp web API server for the BDH graph harness."""

    # Monkeypatch tcp_keepalive to avoid OSError [Errno 22] on macOS/Tailscale
    # This is a known aiohttp bug on macOS where setsockopt(SO_KEEPALIVE) fails
    try:
        import aiohttp.tcp_helpers as _tcp_helpers
        _orig = _tcp_helpers.tcp_keepalive
        def _safe_keepalive(transport):
            try:
                _orig(transport)
            except OSError:
                pass  # Ignore on macOS/Tailscale
        _tcp_helpers.tcp_keepalive = _safe_keepalive
    except Exception:
        pass

    # Build BM25 index for hybrid search (Phase 3.1)
    # BM25Index lives in the retrieval module
    from bdh_graph_harness.retrieval import BM25Index

    bm25_idx = None
    if config.get('hybrid_search', False):
        print("📊 Building BM25 index for hybrid search...")
        bm25_idx = BM25Index(
            nodes,
            k1=config.get('bm25_k1', 1.5),
            b=config.get('bm25_b', 0.75),
        )
        print(f"   ✓ BM25: {bm25_idx.N} docs, {len(bm25_idx.df)} terms")

    # Shared mutable state container
    app_state = {
        'nodes': nodes,
        'edges': edges,
        'collection': collection,
        'state': state,
        'config': config,
        'bm25_index': bm25_idx,
    }

    # Global set of connected WebSocket clients
    ws_clients = set()

    # Build the aiohttp app and register routes
    app = web.Application()
    setup_routes(app, app_state, ws_clients)

    # Start vault file watcher (auto-detect .md changes)
    vault_watcher = None
    vault_path = config.get('vault_path', './wiki')

    try:
        from bdh_graph_harness.api.watcher import VaultWatcher

        async def trigger_node_update():
            """Re-run the node-update logic when vault files change."""
            from bdh_graph_harness.graph.builder import build_graph
            from bdh_graph_harness.api.ws import broadcast_activation

            old_nodes = app_state['nodes'] or {}
            new_nodes, new_edges = build_graph(vault_path, use_cache=False)

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

            print(f"👁️  trigger_node_update: added={len(added)} changed={len(changed)} deleted={len(deleted)}", flush=True)
            if not added and not changed and not deleted:
                print(f"👁️  trigger_node_update: no changes detected, skipping", flush=True)
                return

            app_state['nodes'] = new_nodes
            app_state['edges'] = new_edges

            if added or changed:
                from bdh_graph_harness.retrieval import compute_all_embeddings
                # Use force_refresh=False for incremental updates — only compute
                # embeddings for new/changed notes, not all 400+ notes every time.
                # force_refresh=True was causing server freezes during watcher updates.
                app_state['collection'] = compute_all_embeddings(new_nodes, vault_path, force_refresh=False)

            if added:
                new_concepts = []
                for nid in added:
                    node = new_nodes[nid]
                    source_notes = []
                    for link in new_edges.get(nid, []):
                        target_id = link['target'] if isinstance(link, dict) else link
                        if target_id in old_ids:
                            source_notes.append(old_nodes.get(target_id, {}).get('title', target_id))
                    new_concepts.append({
                        'id': nid,
                        'title': node.get('title', nid.split('/')[-1]),
                        'source_notes': source_notes[:5],
                    })
                await broadcast_activation({
                    'type': 'graph_refresh',
                    'neurons': len(new_nodes),
                    'synapses': sum(len(links) for links in new_edges.values()),
                    'delta': len(added) - len(deleted),
                    'new_concepts': new_concepts,
                    'changed_nodes': changed,
                    'deleted_nodes': deleted,
                    'message': f'{len(added)} new, {len(changed)} changed, {len(deleted)} deleted',
                }, ws_clients)
            elif changed or deleted:
                await broadcast_activation({
                    'type': 'node_update',
                    'changed_nodes': changed,
                    'deleted_nodes': deleted,
                    'message': f'{len(changed)} changed, {len(deleted)} deleted',
                }, ws_clients)

            logger.info(f"Vault watcher: {len(added)} new, {len(changed)} changed, {len(deleted)} deleted")

        vault_watcher = VaultWatcher(vault_path, trigger_node_update)
    except ImportError as e:
        print(f"⚠️  Vault watcher disabled: {e}", flush=True)
    except Exception as e:
        print(f"⚠️  Vault watcher setup failed: {e}", flush=True)

    host = config['api_host']
    port = config['api_port']
    print(f"🌐 BDH Graph Harness API server starting on http://{host}:{port}")

    # Start watcher via on_startup hook (needs the running event loop)
    if vault_watcher:
        async def _start_watcher(app):
            loop = asyncio.get_running_loop()
            vault_watcher.start(loop)
            print(f"👁️  Vault watcher active on {vault_path}", flush=True)
        async def _stop_watcher(app):
            vault_watcher.stop()
        app.on_startup.append(_start_watcher)
        app.on_cleanup.append(_stop_watcher)

    web.run_app(app, host=host, port=port)