"""API server startup for the BDH Graph Harness.

The monkeypatch, BM25 index build, ``app_state`` dict, and ``web.run_app``
call remain here.  Route handlers have been moved to ``routes.py`` and
WebSocket logic to ``ws.py``.
"""

from aiohttp import web

from bdh_graph_harness.config import CONFIG
from bdh_graph_harness.api.routes import setup_routes

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

    host = config['api_host']
    port = config['api_port']
    print(f"🌐 BDH Graph Harness API server starting on http://{host}:{port}")
    web.run_app(app, host=host, port=port)