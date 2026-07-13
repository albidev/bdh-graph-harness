"""MCP (Model Context Protocol) server for BDH Graph Harness.

Exposes the Hebbian graph retrieval system as MCP tools so any MCP-compatible
client (Claude Desktop, Cursor, Windsurf, Continue, etc.) can query the vault
knowledge graph directly.

Architecture: the MCP server is a **thin HTTP client** of the web server
(:8643 by default). All write operations (query, refresh) go through the web
server's REST API, ensuring a single source of truth for Hebbian state and
real-time WebSocket broadcast. Read operations (stats, hebbian, graph) also
prefer the HTTP path. If the web server is not running, the MCP server falls
back to the direct in-process pipeline (with a warning).

Run via stdio (default, for Claude Desktop / Cursor)::

    python -m bdh_graph_harness --mcp

Or via HTTP (for web-based clients)::

    python -m bdh_graph_harness --mcp --transport http --port 8644

Tools exposed:
    - query:       Query the vault graph, get a grounded LLM response with citations
    - stats:       Get graph statistics (neurons, synapses, Hebbian state)
    - hebbian:     Get Hebbian synaptic state (learned connections)
    - graph:       Get the full graph as nodes + edges JSON
    - refresh:     Force refresh embeddings (after vault changes)
"""

import os
import sys
import json
import logging
import urllib.request
import urllib.error

from mcp.server.fastmcp import FastMCP

from bdh_graph_harness.config import load_config

logger = logging.getLogger("bdh-mcp")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_API_HOST = "localhost"
DEFAULT_API_PORT = 8643


def _api_url(path: str, host: str | None = None, port: int | None = None) -> str:
    """Build a web API URL."""
    h = host or os.environ.get("BDH_API_HOST", DEFAULT_API_HOST)
    p = port or int(os.environ.get("BDH_API_PORT", str(DEFAULT_API_PORT)))
    return f"http://{h}:{p}{path}"


def _http_get(url: str, timeout: float = 30) -> dict | None:
    """GET JSON from the web API. Returns None if the server is not reachable."""
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except (urllib.error.URLError, urllib.error.HTTPError, ConnectionError, OSError) as exc:
        logger.debug(f"HTTP GET {url} failed: {exc}")
        return None


def _http_post(url: str, body: dict, timeout: float = 120) -> dict | None:
    """POST JSON to the web API. Returns None if the server is not reachable."""
    try:
        data = json.dumps(body).encode()
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except (urllib.error.URLError, urllib.error.HTTPError, ConnectionError, OSError) as exc:
        logger.debug(f"HTTP POST {url} failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# Fallback: direct in-process pipeline (used when web server is not running)
# ---------------------------------------------------------------------------

# Per-vault fallback cache.  Key = vault_id (or 'default' for single-vault mode).
_fallback_by_vault: dict[str, dict] = {}

# Global config cache for vault resolution
_fallback_config: dict | None = None
_fallback_config_path: str | None = None


def _get_fallback_config(config_path: str | None = None) -> dict:
    """Load (or return cached) config for the fallback pipeline."""
    global _fallback_config, _fallback_config_path
    if _fallback_config is None or config_path != _fallback_config_path:
        _fallback_config = load_config(config_path)
        _fallback_config_path = config_path
    return _fallback_config


def _init_fallback(config_path: str | None = None, vault_id: str | None = None) -> dict:
    """Initialise (or return) the fallback in-process pipeline for a vault.

    In single-vault mode, vault_id is treated as 'default'.
    In multi-vault mode, the vault_id must match a configured vault.

    Returns the fallback dict for the resolved vault.
    """
    from bdh_graph_harness.graph import build_configured_graph, migrate_legacy_state_ids
    from bdh_graph_harness.retrieval import compute_all_embeddings, BM25Index
    from bdh_graph_harness.memory import load_state
    from bdh_graph_harness.vaults import normalize_vault_configs

    config = _get_fallback_config(config_path)
    vault_cfgs = normalize_vault_configs(config)

    # Resolve which vault to use
    if vault_id is None:
        target_cfg = vault_cfgs[0]
        effective_id = target_cfg.id
    else:
        matched = [v for v in vault_cfgs if v.id == vault_id]
        if not matched:
            available = [v.id for v in vault_cfgs]
            raise KeyError(f"Unknown vault '{vault_id}'. Available: {available}")
        target_cfg = matched[0]
        effective_id = vault_id

    if effective_id in _fallback_by_vault:
        return _fallback_by_vault[effective_id]

    # Build the fallback for this vault
    vault_root = target_cfg.path
    if not os.path.isdir(vault_root):
        raise FileNotFoundError(f"Vault path '{vault_root}' not found")

    nodes, edges, unresolved = build_configured_graph(
        target_cfg.settings,
        use_cache=True,
    )
    collection = compute_all_embeddings(
        nodes, vault_root,
        chroma_path=target_cfg.chroma_path,
        collection_name=target_cfg.chroma_collection,
        config=target_cfg.settings,
    )

    bm25_index = None
    cfg = target_cfg.settings
    if cfg.get("hybrid_search", False):
        bm25_index = BM25Index(
            nodes,
            k1=cfg.get("bm25_k1", 1.5),
            b=cfg.get("bm25_b", 0.75),
        )

    fb = {
        "config": cfg,
        "vault_root": vault_root,
        "nodes": nodes,
        "edges": edges,
        "collection": collection,
        "bm25_index": bm25_index,
        "hebbian_state": migrate_legacy_state_ids(load_state(vault_root), nodes),
        "unresolved_links": unresolved,
    }
    _fallback_by_vault[effective_id] = fb

    logger.info(
        f"BDH MCP fallback initialised [{effective_id}]: {len(nodes)} neurons, "
        f"{sum(len(e) for e in edges.values())} synapses, "
        f"{len(fb['hebbian_state']['synapses'])} Hebbian connections"
    )
    return fb


# Keep a module-level reference to config_path so tool functions can use it
_mcp_config_path: str | None = None


def _fallback_query(question: str, vault_id: str | None = None) -> str:
    """Run a query directly against the in-process pipeline."""
    from bdh_graph_harness.retrieval.attention import attention
    from bdh_graph_harness.memory import hebbian_update, save_state
    from bdh_graph_harness.llm import llm_respond
    from bdh_graph_harness.neurogenesis import extract_new_concepts, create_note

    fb = _init_fallback(_mcp_config_path, vault_id)
    cfg = fb["config"]
    nodes = fb["nodes"]
    edges = fb["edges"]
    collection = fb["collection"]
    bm25_idx = fb["bm25_index"]
    state = fb["hebbian_state"]
    vault_root = fb["vault_root"]

    active = attention(question, nodes, edges, collection, bm25_index=bm25_idx)
    if not active:
        return json.dumps({
            "response": "No notes activated above threshold for this query.",
            "activated_notes": [],
            "new_concepts": [],
            "hebbian_updates": [],
        }, ensure_ascii=False, indent=2)

    if cfg.get("online_plasticity", True):
        state, _updated_keys, _pruned = hebbian_update(active, state)
        save_state(vault_root, state)
        fb["hebbian_state"] = state

    response_text = llm_respond(question, active, nodes)

    active_titles = [nodes[nid]["title"] for nid in active if nid in nodes]
    new_concepts_raw = extract_new_concepts(response_text, question, active, nodes)
    new_concepts = []
    for concept in new_concepts_raw:
        title = concept.get("title", "").strip()
        definition = concept.get("definition", "").strip()
        if not title or not definition:
            continue
        new_id = create_note(vault_root, title, definition, active_titles, question)
        new_concepts.append({
            "title": title,
            "definition": definition,
            "note_id": new_id,
            "created": new_id is not None,
        })

    sorted_active = sorted(active.items(), key=lambda x: -x[1])
    activated = []
    for note_id, score in sorted_active:
        node = nodes.get(note_id)
        title = node["title"] if node else note_id
        tags = node.get("tags", "") if node else ""
        if isinstance(tags, list):
            tags = ", ".join(tags)
        activated.append({
            "note_id": note_id,
            "title": title,
            "score": round(score, 4),
            "tags": tags,
        })

    result = {
        "response": response_text,
        "activated_notes": activated,
        "new_concepts": new_concepts,
        "hebbian_updates": [],
        "hebbian_synapse_count": len(state["synapses"]),
    }
    return json.dumps(result, ensure_ascii=False, indent=2)


def _fallback_stats(vault_id: str | None = None) -> str:
    """Get stats from the in-process pipeline."""
    fb = _init_fallback(_mcp_config_path, vault_id)
    nodes = fb["nodes"]
    edges = fb["edges"]
    state = fb["hebbian_state"]

    total_edges = sum(len(e) for e in edges.values())
    degrees = [len(e) for e in edges.values()]
    avg_degree = total_edges / max(len(edges), 1)

    sorted_by_degree = sorted(edges.items(), key=lambda x: -len(x[1]))[:10]
    top_hubs = [
        {"note_id": nid, "title": nodes.get(nid, {}).get("title", nid), "degree": len(links)}
        for nid, links in sorted_by_degree
    ]

    top_hebbian = []
    if state["synapses"]:
        sorted_syn = sorted(state["synapses"].items(), key=lambda x: -x[1]["weight"])[:10]
        for key, syn in sorted_syn:
            a, b = key.split("|")
            top_hebbian.append({
                "pair": f"{a} ↔ {b}",
                "weight": round(syn["weight"], 4),
                "frequency": syn["frequency"],
            })

    result = {
        "neurons": len(nodes),
        "synapses": total_edges,
        "avg_degree": round(avg_degree, 2),
        "max_degree": max(degrees) if degrees else 0,
        "top_hubs": top_hubs,
        "hebbian_synapses": len(state["synapses"]),
        "queries_processed": state["queries"],
        "top_hebbian_connections": top_hebbian,
    }
    return json.dumps(result, ensure_ascii=False, indent=2)


def _fallback_hebbian(vault_id: str | None = None) -> str:
    """Get Hebbian state from the in-process pipeline."""
    fb = _init_fallback(_mcp_config_path, vault_id)
    state = fb["hebbian_state"]
    nodes = fb["nodes"]

    synapses = []
    for key, syn in sorted(state["synapses"].items(), key=lambda x: -x[1]["weight"]):
        a, b = key.split("|")
        title_a = nodes.get(a, {}).get("title", a)
        title_b = nodes.get(b, {}).get("title", b)
        synapses.append({
            "note_a": a,
            "title_a": title_a,
            "note_b": b,
            "title_b": title_b,
            "weight": round(syn["weight"], 4),
            "frequency": syn["frequency"],
            "last_coactivated": syn.get("last_coactivated", ""),
        })

    result = {
        "total_synapses": len(synapses),
        "queries_processed": state["queries"],
        "synapses": synapses,
    }
    return json.dumps(result, ensure_ascii=False, indent=2)


def _fallback_graph(vault_id: str | None = None) -> str:
    """Get graph structure from the in-process pipeline."""
    fb = _init_fallback(_mcp_config_path, vault_id)
    nodes = fb["nodes"]
    edges = fb["edges"]

    node_list = []
    for nid, node in nodes.items():
        tags = node.get("tags", "")
        if isinstance(tags, list):
            tags = ", ".join(tags)
        node_list.append({
            "id": nid,
            "title": node.get("title", nid),
            "path": node.get("path", ""),
            "tags": tags,
            "degree": len(edges.get(nid, [])),
        })

    edge_list = []
    for source, targets in edges.items():
        for target in targets:
            edge_list.append({"source": source, "target": target})

    result = {
        "nodes": node_list,
        "edges": edge_list,
        "node_count": len(node_list),
        "edge_count": len(edge_list),
    }
    return json.dumps(result, ensure_ascii=False, indent=2)


def _fallback_refresh(vault_id: str | None = None) -> str:
    """Force refresh via in-process pipeline."""
    from bdh_graph_harness.graph import build_configured_graph, migrate_legacy_state_ids
    from bdh_graph_harness.retrieval import compute_all_embeddings, BM25Index
    from bdh_graph_harness.memory import load_state
    from bdh_graph_harness.vaults import normalize_vault_configs

    config = _get_fallback_config(_mcp_config_path)
    vault_cfgs = normalize_vault_configs(config)

    if vault_id is None:
        target_cfg = vault_cfgs[0]
        effective_id = target_cfg.id
    else:
        matched = [v for v in vault_cfgs if v.id == vault_id]
        if not matched:
            available = [v.id for v in vault_cfgs]
            return json.dumps({"error": f"Unknown vault '{vault_id}'",
                               "available_vaults": available})
        target_cfg = matched[0]
        effective_id = vault_id

    fb = _init_fallback(_mcp_config_path, vault_id)
    cfg = target_cfg.settings
    vault_root = target_cfg.path

    nodes, edges, unresolved = build_configured_graph(
        cfg,
        use_cache=False,
    )
    fb["nodes"] = nodes
    fb["edges"] = edges
    fb["unresolved_links"] = unresolved

    collection = compute_all_embeddings(
        nodes, vault_root, force_refresh=True,
        chroma_path=target_cfg.chroma_path,
        collection_name=target_cfg.chroma_collection,
        config=cfg,
    )
    fb["collection"] = collection

    if cfg.get("hybrid_search", False):
        fb["bm25_index"] = BM25Index(
            nodes,
            k1=cfg.get("bm25_k1", 1.5),
            b=cfg.get("bm25_b", 0.75),
        )

    fb["hebbian_state"] = load_state(vault_root)

    total_edges = sum(len(e) for e in edges.values())
    result = {
        "status": "refreshed",
        "neurons": len(nodes),
        "synapses": total_edges,
        "embeddings": collection.count(),
        "hebbian_synapses": len(fb["hebbian_state"]["synapses"]),
    }
    return json.dumps(result, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# MCP server + tools
# ---------------------------------------------------------------------------

mcp = FastMCP(
    "bdh-graph-harness",
    instructions=(
        "BDH Graph Harness — a Hebbian knowledge graph over an Obsidian vault. "
        "Notes are neurons, wikilinks are synapses, and co-activation strengthens "
        "connections via Hebbian learning. Use the 'query' tool to ask questions "
        "grounded in vault content. Use 'stats' for graph overview, 'hebbian' for "
        "learned synaptic connections, and 'graph' for the full network structure."
    ),
)


@mcp.tool()
def query(question: str, vault_id: str | None = None) -> str:
    """Query the vault knowledge graph and get a grounded response with citations.

    Runs attention (hybrid vector + BM25 search + k-hop graph traversal),
    updates Hebbian synaptic weights (online plasticity), generates an LLM
    response grounded in activated notes, and optionally creates new concept
    notes (neurogenesis).

    If the BDH web server is running, this tool proxies to it via HTTP,
    ensuring real-time WebSocket updates and a single source of truth for
    Hebbian state. Falls back to in-process pipeline if the server is down.

    Args:
        question: The question to ask the knowledge graph.
        vault_id: Optional vault ID to query (for multi-vault configs).
                  Defaults to the configured default vault.

    Returns:
        A JSON string with: response (LLM answer), activated_notes (sources
        with similarity scores), new_concepts (if any were created),
        hebbian_updates (synaptic changes from this query).
    """
    # Try the web server first (thin client)
    payload: dict = {"query": question}
    if vault_id is not None:
        payload["vault_id"] = vault_id
    result = _http_post(_api_url("/api/query"), payload)
    if result is not None:
        return json.dumps(result, ensure_ascii=False, indent=2)

    # Fallback: in-process pipeline
    logger.warning("Web server not reachable, using in-process fallback pipeline")
    return _fallback_query(question, vault_id=vault_id)


@mcp.tool()
def stats(vault_id: str | None = None) -> str:
    """Get graph statistics: neuron count, synapse count, Hebbian state, top hubs.

    Args:
        vault_id: Optional vault ID (for multi-vault configs).
                  Defaults to the configured default vault.

    Returns:
        JSON string with graph statistics.
    """
    url = _api_url("/api/stats")
    if vault_id is not None:
        url = f"{url}?vault_id={vault_id}"
    result = _http_get(url)
    if result is not None:
        # Web API already returns the right keys, just pass through
        return json.dumps(result, ensure_ascii=False, indent=2)

    logger.warning("Web server not reachable, using in-process fallback pipeline")
    return _fallback_stats(vault_id=vault_id)


@mcp.tool()
def hebbian(vault_id: str | None = None) -> str:
    """Get the full Hebbian synaptic state — all learned connections between notes.

    Hebbian synapses are connections discovered through co-activation during
    queries, not from wikilinks. They represent semantic relationships the
    graph learned from usage patterns.

    Args:
        vault_id: Optional vault ID (for multi-vault configs).
                  Defaults to the configured default vault.

    Returns:
        JSON string with all synaptic connections sorted by weight.
    """
    url = _api_url("/api/hebbian")
    if vault_id is not None:
        url = f"{url}?vault_id={vault_id}"
    result = _http_get(url)
    if result is not None:
        # Normalise: web API uses 'total' + 'queries', MCP uses 'total_synapses' + 'queries_processed'
        synapses = result.get("synapses", [])
        normalised = {
            "total_synapses": result.get("total", len(synapses)),
            "queries_processed": result.get("queries", 0),
            "synapses": synapses,
        }
        return json.dumps(normalised, ensure_ascii=False, indent=2)

    logger.warning("Web server not reachable, using in-process fallback pipeline")
    return _fallback_hebbian(vault_id=vault_id)


@mcp.tool()
def graph(vault_id: str | None = None) -> str:
    """Get the full graph structure as nodes and edges.

    Returns all notes (neurons) with their metadata and all wikilink connections
    (synapses). Useful for visualisation or analysis.

    Args:
        vault_id: Optional vault ID (for multi-vault configs).
                  Defaults to the configured default vault.

    Returns:
        JSON string with nodes array and edges array.
    """
    url = _api_url("/api/graph")
    if vault_id is not None:
        url = f"{url}?vault_id={vault_id}"
    result = _http_get(url)
    if result is not None:
        # Normalise: web API returns 'nodes' + 'edges' without counts
        nodes = result.get("nodes", [])
        edges = result.get("edges", [])
        normalised = {
            "nodes": nodes,
            "edges": edges,
            "node_count": len(nodes),
            "edge_count": len(edges),
        }
        return json.dumps(normalised, ensure_ascii=False, indent=2)

    logger.warning("Web server not reachable, using in-process fallback pipeline")
    return _fallback_graph(vault_id=vault_id)


@mcp.tool()
def refresh(vault_id: str | None = None) -> str:
    """Force refresh all embeddings in ChromaDB.

    Use this after adding new notes to the vault or when the graph seems stale.
    Rebuilds the graph from scratch (skips cache) and re-computes all embeddings.

    Args:
        vault_id: Optional vault ID (for multi-vault configs).
                  Defaults to the configured default vault.

    Returns:
        JSON string with refresh results.
    """
    payload: dict = {}
    if vault_id is not None:
        payload["vault_id"] = vault_id
    result = _http_post(_api_url("/api/refresh"), payload)
    if result is not None:
        # Enrich with hebbian count if not present
        if "hebbian_synapses" not in result:
            stats_url = _api_url("/api/stats")
            if vault_id is not None:
                stats_url = f"{stats_url}?vault_id={vault_id}"
            s = _http_get(stats_url)
            if s and "hebbian_synapses" in s:
                result["hebbian_synapses"] = s["hebbian_synapses"]
        return json.dumps(result, ensure_ascii=False, indent=2)

    logger.warning("Web server not reachable, using in-process fallback pipeline")
    return _fallback_refresh(vault_id=vault_id)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_mcp_server(transport: str = "stdio", port: int = 8644, config_path: str = None):
    """Start the MCP server.

    Args:
        transport: "stdio" for Claude Desktop / Cursor, "http" for web clients.
        port: Port for HTTP transport (default 8644).
        config_path: Path to config YAML (defaults to bdh-config.yaml in cwd).
    """
    import asyncio

    # Store config path for fallback initialisation
    global _mcp_config_path
    _mcp_config_path = config_path
    if config_path:
        os.environ.setdefault("BDH_CONFIG_PATH", config_path)

    logger.info(f"BDH MCP server starting ({transport})")

    if transport == "http":
        import uvicorn

        async def _run_http():
            app = mcp.streamable_http_app()
            config = uvicorn.Config(app, host="0.0.0.0", port=port)
            server = uvicorn.Server(config)
            await server.serve()

        asyncio.run(_run_http())
    else:
        asyncio.run(mcp.run_stdio_async())


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="BDH Graph Harness MCP Server")
    parser.add_argument("--transport", choices=["stdio", "http"], default="stdio")
    parser.add_argument("--port", type=int, default=8644)
    args = parser.parse_args()
    run_mcp_server(transport=args.transport, port=args.port)