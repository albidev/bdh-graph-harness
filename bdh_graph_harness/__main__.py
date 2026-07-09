"""CLI entry point for the BDH Graph Harness.

Run with::

    python -m bdh_graph_harness [args]
    python -m bdh_graph_harness --serve
    python -m bdh_graph_harness --stats
    python -m bdh_graph_harness --interactive

Contains the ``main()`` argparse dispatcher plus terminal-output helpers
``show_stats()``, ``show_hebbian()``, and ``interactive_mode()``.
"""

import os
import sys
import argparse

from bdh_graph_harness.config import CONFIG, load_config
from bdh_graph_harness.graph import build_graph
from bdh_graph_harness.retrieval.attention import attention
from bdh_graph_harness.retrieval import compute_all_embeddings, BM25Index
from bdh_graph_harness.memory import load_state, save_state, hebbian_update
from bdh_graph_harness.llm import llm_respond
from bdh_graph_harness.neurogenesis import extract_new_concepts, create_note
from bdh_graph_harness.api import start_api_server

__all__ = ["main", "show_stats", "show_hebbian", "interactive_mode"]


# ---------------------------------------------------------------------------
# Terminal output helpers
# ---------------------------------------------------------------------------

def show_stats(nodes, edges, state):
    """Print graph statistics."""
    print(f"\n📊 BDH Graph Stats")
    print(f"  Neurons (notes): {len(nodes)}")
    print(f"  Synapses (links): {sum(len(e) for e in edges.values())}")
    print(f"  Avg degree: {sum(len(e) for e in edges.values()) / max(len(edges), 1):.1f}")

    # Degree distribution
    degrees = [len(e) for e in edges.values()]
    if degrees:
        degrees.sort(reverse=True)
        print(f"  Max degree: {degrees[0]}")
        print(f"  Top-5 hubs:")
        sorted_by_degree = sorted(edges.items(), key=lambda x: -len(x[1]))
        for note_id, links in sorted_by_degree[:5]:
            print(f"    {note_id} ({len(links)} links)")

    print(f"\n  Hebbian synapses: {len(state['synapses'])}")
    print(f"  Queries processed: {state['queries']}")

    # Show top Hebbian connections
    if state['synapses']:
        sorted_syn = sorted(state['synapses'].items(), key=lambda x: -x[1]['weight'])
        print(f"\n  Top Hebbian connections:")
        for key, syn in sorted_syn[:5]:
            a, b = key.split('|')
            print(f"    {a} ↔ {b} (w={syn['weight']:.3f}, freq={syn['frequency']})")


def show_hebbian(state):
    """Show all Hebbian synaptic connections."""
    if not state['synapses']:
        print("\n🔌 No Hebbian synapses yet. Run some queries first.")
        return

    print(f"\n🔌 Hebbian Synaptic State ({len(state['synapses'])} connections)")
    sorted_syn = sorted(state['synapses'].items(), key=lambda x: -x[1]['weight'])
    for key, syn in sorted_syn:
        a, b = key.split('|')
        print(f"  {a} ↔ {b}")
        print(f"    weight: {syn['weight']:.3f} | freq: {syn['frequency']} | last: {syn['last_coactivated']}")


def interactive_mode(vault_root, nodes, edges, collection, state, bm25_index=None):
    """Interactive REPL for querying the vault."""
    print(f"\n🐉 BDH Graph Harness — Interactive Mode")
    print(f"   Vault: {vault_root}")
    print(f"   Neurons: {len(nodes)} | Synapses: {sum(len(e) for e in edges.values())}")
    print(f"   Type 'exit' to quit, 'stats' for graph stats, 'hebbian' for synaptic state\n")

    while True:
        try:
            query = input("❯ ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not query:
            continue
        if query.lower() == 'exit':
            break
        if query.lower() == 'stats':
            show_stats(nodes, edges, state)
            continue
        if query.lower() == 'hebbian':
            show_hebbian(state)
            continue

        # Attention (with hybrid search)
        active = attention(query, nodes, edges, collection, bm25_index=bm25_index)

        if not active:
            print("  No notes activated above threshold.\n")
            continue

        # Online plasticity: Hebbian update right after attention (Phase 3.2)
        if CONFIG.get('online_plasticity', True):
            state, _updated_keys, _pruned = hebbian_update(active, state)
            save_state(vault_root, state)

        # Show results
        sorted_active = sorted(active.items(), key=lambda x: -x[1])
        print(f"\n  🧠 Activated {len(active)} neurons:")
        for note_id, score in sorted_active:
            node = nodes.get(note_id)
            title = node['title'] if node else note_id
            print(f"    [{score:.3f}] {title}")

        # LLM response
        print(f"\n  🤖 LLM response:")
        print("  " + "-" * 60)
        response = llm_respond(query, active, nodes)
        for line in response.split('\n'):
            print(f"  {line}")
        print("  " + "-" * 60)

        # Neurogenesis
        active_titles = [nodes[nid]['title'] for nid in active if nid in nodes]
        new_concepts = extract_new_concepts(response, query, active, nodes)

        if new_concepts:
            print(f"\n  🧬 Neurogenesis: {len(new_concepts)} new concept(s) detected")
            for concept in new_concepts:
                title = concept.get('title', '').strip()
                definition = concept.get('definition', '').strip()
                if not title or not definition:
                    continue
                new_note_id = create_note(vault_root, title, definition, active_titles, query)
                if new_note_id:
                    print(f"    ✨ Created: {new_note_id} — {title}")
                else:
                    print(f"    ⊘ Skipped (already exists): {title}")
        else:
            print(f"\n  🧬 Neurogenesis: no new concepts")

        # Hebbian update (post-response, with nodes for quality pruning)
        state, _updated_keys, _pruned = hebbian_update(active, state, nodes=nodes)
        save_state(vault_root, state)
        print(f"  🔌 Hebbian update: {len(state['synapses'])} synapses\n")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='BDH Graph Harness — neural vault knowledge engine',
    )
    parser.add_argument('query', nargs='*', help='Query text (positional)')
    parser.add_argument('--config', default=None, help='Path to config YAML file')
    parser.add_argument('--vault', default=None, help='Vault root path (overrides config)')
    parser.add_argument('--vault-id', default=None,
                        help='Select a configured vault by ID (for multi-vault configs)')
    parser.add_argument('--serve', action='store_true', help='Start the API server')
    parser.add_argument('--stats', action='store_true', help='Show graph statistics')
    parser.add_argument('--hebbian-show', action='store_true', help='Show Hebbian synaptic state')
    parser.add_argument('--refresh-embeddings', action='store_true',
                        help='Force refresh all embeddings in ChromaDB')
    parser.add_argument('--interactive', action='store_true', help='Interactive REPL mode')
    parser.add_argument('--no-cache', action='store_true',
                        help='Force full graph rebuild (skip cache)')
    parser.add_argument('--mcp', action='store_true',
                        help='Start the MCP (Model Context Protocol) server')
    parser.add_argument('--mcp-transport', choices=['stdio', 'http'], default='stdio',
                        help='MCP transport mode (default: stdio)')
    parser.add_argument('--mcp-port', type=int, default=8644,
                        help='MCP HTTP server port (default: 8644)')
    parser.add_argument('--list-vaults', action='store_true',
                        help='List configured vaults and exit')
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # --list-vaults: show configured vaults and exit
    if args.list_vaults:
        from bdh_graph_harness.vaults import normalize_vault_configs
        import chromadb
        vault_cfgs = normalize_vault_configs(config)
        print(f"\n{'id':<12} {'path':<40} {'collection':<30} neurons embeddings")
        print("-" * 100)
        for vc in vault_cfgs:
            neurons = 0
            embeddings = 0
            try:
                from bdh_graph_harness.graph.builder import build_graph
                nodes, _ = build_graph(vc.path, use_cache=True)
                neurons = len(nodes)
            except Exception:
                pass
            try:
                client = chromadb.PersistentClient(path=vc.chroma_path)
                coll = client.get_collection(vc.chroma_collection)
                embeddings = coll.count()
            except Exception:
                pass
            print(f"{vc.id:<12} {vc.path:<40} {vc.chroma_collection:<30} {neurons:<7} {embeddings}")
        print()
        return

    # Resolve vault (--vault-id takes precedence over --vault)
    if args.vault_id:
        from bdh_graph_harness.vaults import normalize_vault_configs
        vault_cfgs = normalize_vault_configs(config)
        matched = [v for v in vault_cfgs if v.id == args.vault_id]
        if not matched:
            available = [v.id for v in vault_cfgs]
            print(f"Error: vault-id '{args.vault_id}' not found. Available: {available}")
            sys.exit(1)
        vault_root = matched[0].path
    else:
        vault_root = args.vault or config['vault_path']
        vault_root = os.path.expanduser(vault_root)

    if not os.path.isdir(vault_root):
        print(f"Error: vault path '{vault_root}' not found")
        sys.exit(1)

    print(f"🐉 BDH Graph Harness — Building graph from vault...")
    print(f"   Vault: {vault_root}")

    nodes, edges = build_graph(vault_root, use_cache=not args.no_cache)
    print(f"   ✓ {len(nodes)} neurons, {sum(len(e) for e in edges.values())} synapses")

    # Compute embeddings
    collection = compute_all_embeddings(nodes, vault_root)

    # Build BM25 index for hybrid search (Phase 3.1)
    # Skip if --serve: the server builds its own index
    bm25_idx = None
    if config.get('hybrid_search', False) and not args.serve:
        print("📊 Building BM25 index...")
        bm25_idx = BM25Index(nodes, k1=config.get('bm25_k1', 1.5), b=config.get('bm25_b', 0.75))
        print(f"   ✓ BM25: {bm25_idx.N} docs, {len(bm25_idx.df)} terms")

    # Load state
    state = load_state(vault_root)

    # --- Mode dispatch ---

    if args.mcp:
        from bdh_graph_harness.mcp_server import run_mcp_server
        print(f"🐉 BDH Graph Harness — MCP Server ({args.mcp_transport})")
        run_mcp_server(transport=args.mcp_transport, port=args.mcp_port, config_path=args.config)
        return

    if args.serve:
        start_api_server(config, nodes, edges, collection, state)
        return

    if args.stats:
        show_stats(nodes, edges, state)
        return

    if args.hebbian_show:
        show_hebbian(state)
        return

    if args.refresh_embeddings:
        print("🔄 Force-refreshing all embeddings...")
        collection = compute_all_embeddings(nodes, vault_root, force_refresh=True)
        print(f"  ✓ Refreshed {collection.count()} embeddings")
        return

    if args.interactive:
        interactive_mode(vault_root, nodes, edges, collection, state, bm25_index=bm25_idx)
        return

    # Single query mode
    query = ' '.join(args.query).strip()
    if not query:
        print("No query provided. Use --interactive for REPL mode or --serve for API server.")
        return

    print(f"\n🔍 Query: '{query}'")

    active = attention(query, nodes, edges, collection, bm25_index=bm25_idx)

    if not active:
        print("  No notes activated above threshold.")
        return

    sorted_active = sorted(active.items(), key=lambda x: -x[1])
    print(f"\n  🧠 Activated {len(active)} neurons:")
    for note_id, score in sorted_active:
        node = nodes.get(note_id)
        title = node['title'] if node else note_id
        print(f"    [{score:.3f}] {title}")

    # Online plasticity: Hebbian update right after attention (Phase 3.2)
    if config.get('online_plasticity', True):
        state, _updated_keys, _pruned = hebbian_update(active, state, nodes=nodes)
        save_state(vault_root, state)
        print(f"  🔌 Hebbian update (online): {len(state['synapses'])} synapses")

    # LLM response
    print(f"\n  🤖 LLM response:")
    print("  " + "-" * 60)
    response = llm_respond(query, active, nodes)
    # Indent response
    for line in response.split('\n'):
        print(f"  {line}")
    print("  " + "-" * 60)

    # Neurogenesis: extract new concepts from LLM response
    active_titles = [nodes[nid]['title'] for nid in active if nid in nodes]
    new_concepts = extract_new_concepts(response, query, active, nodes)

    if new_concepts:
        print(f"\n  🧬 Neurogenesis: {len(new_concepts)} new concept(s) detected")
        for concept in new_concepts:
            title = concept.get('title', '').strip()
            definition = concept.get('definition', '').strip()
            if not title or not definition:
                continue
            new_note_id = create_note(vault_root, title, definition, active_titles, query)
            if new_note_id:
                print(f"    ✨ Created: {new_note_id} — {title}")
            else:
                print(f"    ⊘ Skipped (already exists): {title}")
    else:
        print(f"\n  🧬 Neurogenesis: no new concepts")


if __name__ == '__main__':
    main()