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

_fallback = {
    "config": None,
    "vault_root": None,
    "nodes": None,
    "edges": None,
    "collection": None,
    "bm25_index": None,
    "hebbian_state": None,
    "initialised": False,
}


def _init_fallback(config_path: str = None):
    """Initialise the fallback in-process pipeline. Called lazily."""
    if _fallback["initialised"]:
        return

    from bdh_graph_harness.graph import build_graph
    from bdh_graph_harness.retrieval import compute_all_embeddings, BM25Index
    from bdh_graph_harness.memory import load_state

    config = load_config(config_path)
    _fallback["config"] = config

    vault_root = os.path.expanduser(config["vault_path"])
    if not os.path.isdir(vault_root):
        raise FileNotFoundError(f"Vault path '{vault_root}' not found")

    _fallback["vault_root"] = vault_root

    nodes, edges = build_graph(vault_root, use_cache=True)
    _fallback["nodes"] = nodes
    _fallback["edges"] = edges

    collection = compute_all_embeddings(nodes, vault_root)
    _fallback["collection"] = collection

    if config.get("hybrid_search", False):
        _fallback["bm25_index"] = BM25Index(
            nodes,
            k1=config.get("bm25_k1", 1.5),
            b=config.get("bm25_b", 0.75),
        )

    _fallback["hebbian_state"] = load_state(vault_root)
    _fallback["initialised"] = True

    logger.info(
        f"BDH MCP fallback initialised: {len(nodes)} neurons, "
        f"{sum(len(e) for e in edges.values())} synapses, "
        f"{len(_fallback['hebbian_state']['synapses'])} Hebbian connections"
    )


def _fallback_query(question: str) -> str:
    """Run a query directly against the in-process pipeline."""
    from bdh_graph_harness.retrieval.attention import attention
    from bdh_graph_harness.memory import hebbian_update, save_state
    from bdh_graph_harness.llm import llm_respond
    from bdh_graph_harness.neurogenesis import extract_new_concepts, create_note

    _init_fallback()
    cfg = _fallback["config"]
    nodes = _fallback["nodes"]
    edges = _fallback["edges"]
    collection = _fallback["collection"]
    bm25_idx = _fallback["bm25_index"]
    state = _fallback["hebbian_state"]
    vault_root = _fallback["vault_root"]

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
        _fallback["hebbian_state"] = state

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


def _fallback_stats() -> str:
    """Get stats from the in-process pipeline."""
    _init_fallback()
    nodes = _fallback["nodes"]
    edges = _fallback["edges"]
    state = _fallback["hebbian_state"]

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


def _fallback_hebbian() -> str:
    """Get Hebbian state from the in-process pipeline."""
    _init_fallback()
    state = _fallback["hebbian_state"]
    nodes = _fallback["nodes"]

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


def _fallback_graph() -> str:
    """Get graph structure from the in-process pipeline."""
    _init_fallback()
    nodes = _fallback["nodes"]
    edges = _fallback["edges"]

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


def _fallback_refresh() -> str:
    """Force refresh via in-process pipeline."""
    from bdh_graph_harness.graph import build_graph
    from bdh_graph_harness.retrieval import compute_all_embeddings, BM25Index
    from bdh_graph_harness.memory import load_state

    _init_fallback()
    cfg = _fallback["config"]
    vault_root = _fallback["vault_root"]

    nodes, edges = build_graph(vault_root, use_cache=False)
    _fallback["nodes"] = nodes
    _fallback["edges"] = edges

    collection = compute_all_embeddings(nodes, vault_root, force_refresh=True)
    _fallback["collection"] = collection

    if cfg.get("hybrid_search", False):
        _fallback["bm25_index"] = BM25Index(
            nodes,
            k1=cfg.get("bm25_k1", 1.5),
            b=cfg.get("bm25_b", 0.75),
        )

    _fallback["hebbian_state"] = load_state(vault_root)

    total_edges = sum(len(e) for e in edges.values())
    result = {
        "status": "refreshed",
        "neurons": len(nodes),
        "synapses": total_edges,
        "embeddings": collection.count(),
        "hebbian_synapses": len(_fallback["hebbian_state"]["synapses"]),
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
def query(question: str) -> str:
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

    Returns:
        A JSON string with: response (LLM answer), activated_notes (sources
        with similarity scores), new_concepts (if any were created),
        hebbian_updates (synaptic changes from this query).
    """
    # Try the web server first (thin client)
    result = _http_post(_api_url("/api/query"), {"query": question})
    if result is not None:
        return json.dumps(result, ensure_ascii=False, indent=2)

    # Fallback: in-process pipeline
    logger.warning("Web server not reachable, using in-process fallback pipeline")
    return _fallback_query(question)


@mcp.tool()
def stats() -> str:
    """Get graph statistics: neuron count, synapse count, Hebbian state, top hubs.

    Returns:
        JSON string with graph statistics.
    """
    result = _http_get(_api_url("/api/stats"))
    if result is not None:
        # Web API already returns the right keys, just pass through
        return json.dumps(result, ensure_ascii=False, indent=2)

    logger.warning("Web server not reachable, using in-process fallback pipeline")
    return _fallback_stats()


@mcp.tool()
def hebbian() -> str:
    """Get the full Hebbian synaptic state — all learned connections between notes.

    Hebbian synapses are connections discovered through co-activation during
    queries, not from wikilinks. They represent semantic relationships the
    graph learned from usage patterns.

    Returns:
        JSON string with all synaptic connections sorted by weight.
    """
    result = _http_get(_api_url("/api/hebbian"))
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
    return _fallback_hebbian()


@mcp.tool()
def graph() -> str:
    """Get the full graph structure as nodes and edges.

    Returns all notes (neurons) with their metadata and all wikilink connections
    (synapses). Useful for visualisation or analysis.

    Returns:
        JSON string with nodes array and edges array.
    """
    result = _http_get(_api_url("/api/graph"))
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
    return _fallback_graph()


@mcp.tool()
def refresh() -> str:
    """Force refresh all embeddings in ChromaDB.

    Use this after adding new notes to the vault or when the graph seems stale.
    Rebuilds the graph from scratch (skips cache) and re-computes all embeddings.

    Returns:
        JSON string with refresh results.
    """
    result = _http_post(_api_url("/api/refresh"), {})
    if result is not None:
        # Enrich with hebbian count if not present
        if "hebbian_synapses" not in result:
            stats = _http_get(_api_url("/api/stats"))
            if stats and "hebbian_synapses" in stats:
                result["hebbian_synapses"] = stats["hebbian_synapses"]
        return json.dumps(result, ensure_ascii=False, indent=2)

    logger.warning("Web server not reachable, using in-process fallback pipeline")
    return _fallback_refresh()


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