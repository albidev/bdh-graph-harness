"""MCP (Model Context Protocol) server for BDH Graph Harness.

Exposes the Hebbian graph retrieval system as MCP tools so any MCP-compatible
client (Claude Desktop, Cursor, Windsurf, Continue, etc.) can query the vault
knowledge graph directly.

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

from mcp.server.fastmcp import FastMCP

from bdh_graph_harness.config import load_config
from bdh_graph_harness.graph import build_graph
from bdh_graph_harness.retrieval import compute_all_embeddings, BM25Index
from bdh_graph_harness.retrieval.attention import attention
from bdh_graph_harness.memory import load_state, save_state, hebbian_update
from bdh_graph_harness.llm import llm_respond
from bdh_graph_harness.neurogenesis import extract_new_concepts, create_note

logger = logging.getLogger("bdh-mcp")

# ---------------------------------------------------------------------------
# Global state — initialised once at startup, reused across tool calls
# ---------------------------------------------------------------------------

_state = {
    "config": None,
    "vault_root": None,
    "nodes": None,
    "edges": None,
    "collection": None,
    "hebbian_state": None,
    "bm25_index": None,
    "initialised": False,
}


def _init():
    """Initialise graph, embeddings, BM25, and Hebbian state. Called once."""
    if _state["initialised"]:
        return

    config = load_config()
    _state["config"] = config

    vault_root = os.path.expanduser(config["vault_path"])
    if not os.path.isdir(vault_root):
        raise FileNotFoundError(f"Vault path '{vault_root}' not found")

    _state["vault_root"] = vault_root

    # Build graph (with cache)
    nodes, edges = build_graph(vault_root, use_cache=True)
    _state["nodes"] = nodes
    _state["edges"] = edges

    # Compute embeddings
    collection = compute_all_embeddings(nodes, vault_root)
    _state["collection"] = collection

    # BM25 index for hybrid search
    if config.get("hybrid_search", False):
        _state["bm25_index"] = BM25Index(
            nodes,
            k1=config.get("bm25_k1", 1.5),
            b=config.get("bm25_b", 0.75),
        )

    # Load Hebbian state
    _state["hebbian_state"] = load_state(vault_root)
    _state["initialised"] = True

    logger.info(
        f"BDH MCP server initialised: {len(nodes)} neurons, "
        f"{sum(len(e) for e in edges.values())} synapses, "
        f"{len(_state['hebbian_state']['synapses'])} Hebbian connections"
    )


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

    Args:
        question: The question to ask the knowledge graph.

    Returns:
        A JSON string with: response (LLM answer), activated_notes (sources
        with similarity scores), new_concepts (if any were created),
        hebbian_updates (synaptic changes from this query).
    """
    _init()
    cfg = _state["config"]
    nodes = _state["nodes"]
    edges = _state["edges"]
    collection = _state["collection"]
    bm25_idx = _state["bm25_index"]
    state = _state["hebbian_state"]
    vault_root = _state["vault_root"]

    # Attention: hybrid search + k-hop traversal + adaptive threshold
    active = attention(question, nodes, edges, collection, bm25_index=bm25_idx)

    if not active:
        return json.dumps({
            "response": "No notes activated above threshold for this query.",
            "activated_notes": [],
            "new_concepts": [],
            "hebbian_updates": [],
        })

    # Online plasticity: Hebbian update before LLM
    pre_synapse_count = len(state["synapses"])
    if cfg.get("online_plasticity", True):
        state = hebbian_update(active, state)
        save_state(vault_root, state)
        _state["hebbian_state"] = state

    # Track which synapses were updated
    hebbian_updates = []
    for key, syn in state["synapses"].items():
        if key not in _state.get("_pre_synapses", set()):
            a, b = key.split("|")
            hebbian_updates.append({
                "pair": f"{a} ↔ {b}",
                "weight": round(syn["weight"], 4),
                "frequency": syn["frequency"],
            })
    _state["_pre_synapses"] = set(state["synapses"].keys())

    # LLM response
    response_text = llm_respond(question, active, nodes)

    # Neurogenesis: extract and create new concept notes
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

    # Build activated notes list with scores
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
        "hebbian_updates": hebbian_updates,
        "hebbian_synapse_count": len(state["synapses"]),
    }
    return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool()
def stats() -> str:
    """Get graph statistics: neuron count, synapse count, Hebbian state, top hubs.

    Returns:
        JSON string with graph statistics.
    """
    _init()
    nodes = _state["nodes"]
    edges = _state["edges"]
    state = _state["hebbian_state"]

    total_edges = sum(len(e) for e in edges.values())
    degrees = [len(e) for e in edges.values()]
    avg_degree = total_edges / max(len(edges), 1)

    # Top hubs by degree
    sorted_by_degree = sorted(edges.items(), key=lambda x: -len(x[1]))[:10]
    top_hubs = [
        {"note_id": nid, "title": nodes.get(nid, {}).get("title", nid), "degree": len(links)}
        for nid, links in sorted_by_degree
    ]

    # Top Hebbian connections
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


@mcp.tool()
def hebbian() -> str:
    """Get the full Hebbian synaptic state — all learned connections between notes.

    Hebbian synapses are connections discovered through co-activation during
    queries, not from wikilinks. They represent semantic relationships the
    graph learned from usage patterns.

    Returns:
        JSON string with all synaptic connections sorted by weight.
    """
    _init()
    state = _state["hebbian_state"]
    nodes = _state["nodes"]

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


@mcp.tool()
def graph() -> str:
    """Get the full graph structure as nodes and edges.

    Returns all notes (neurons) with their metadata and all wikilink connections
    (synapses). Useful for visualisation or analysis.

    Returns:
        JSON string with nodes array and edges array.
    """
    _init()
    nodes = _state["nodes"]
    edges = _state["edges"]

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


@mcp.tool()
def refresh() -> str:
    """Force refresh all embeddings in ChromaDB.

    Use this after adding new notes to the vault or when the graph seems stale.
    Rebuilds the graph from scratch (skips cache) and re-computes all embeddings.

    Returns:
        JSON string with refresh results.
    """
    _init()
    cfg = _state["config"]
    vault_root = _state["vault_root"]

    # Rebuild graph without cache
    nodes, edges = build_graph(vault_root, use_cache=False)
    _state["nodes"] = nodes
    _state["edges"] = edges

    # Re-compute embeddings
    collection = compute_all_embeddings(nodes, vault_root, force_refresh=True)
    _state["collection"] = collection

    # Rebuild BM25
    if cfg.get("hybrid_search", False):
        _state["bm25_index"] = BM25Index(
            nodes,
            k1=cfg.get("bm25_k1", 1.5),
            b=cfg.get("bm25_b", 0.75),
        )

    # Reload Hebbian state
    _state["hebbian_state"] = load_state(vault_root)

    total_edges = sum(len(e) for e in edges.values())
    result = {
        "status": "refreshed",
        "neurons": len(nodes),
        "synapses": total_edges,
        "embeddings": collection.count(),
        "hebbian_synapses": len(_state["hebbian_state"]["synapses"]),
    }
    return json.dumps(result, ensure_ascii=False, indent=2)


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

    # Initialise global state with explicit config path
    config = load_config(config_path)
    _state["config"] = config

    vault_root = os.path.expanduser(config["vault_path"])
    if not os.path.isdir(vault_root):
        raise FileNotFoundError(f"Vault path '{vault_root}' not found")

    _state["vault_root"] = vault_root

    # Build graph (with cache)
    nodes, edges = build_graph(vault_root, use_cache=True)
    _state["nodes"] = nodes
    _state["edges"] = edges

    # Compute embeddings
    collection = compute_all_embeddings(nodes, vault_root)
    _state["collection"] = collection

    # BM25 index for hybrid search
    if config.get("hybrid_search", False):
        _state["bm25_index"] = BM25Index(
            nodes,
            k1=config.get("bm25_k1", 1.5),
            b=config.get("bm25_b", 0.75),
        )

    # Load Hebbian state
    _state["hebbian_state"] = load_state(vault_root)
    _state["initialised"] = True

    logger.info(
        f"BDH MCP server initialised: {len(nodes)} neurons, "
        f"{sum(len(e) for e in edges.values())} synapses, "
        f"{len(_state['hebbian_state']['synapses'])} Hebbian connections"
    )

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