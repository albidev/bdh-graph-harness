"""
Phantom links — semantic similarity connections.

Creates synthetic edges between notes that are semantically similar
but not connected via wikilinks. This mirrors how biological brains
form associations between concepts that co-occur in experience,
even without a direct structural connection.

Uses ChromaDB embeddings for nearest-neighbor search. Phantom links
are stored in state['phantom_links'] and persist across restarts.
"""

import os
import chromadb

from bdh_graph_harness.config import CONFIG, logger

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_SIMILARITY_THRESHOLD = 0.65   # minimum cosine similarity to create a phantom link
DEFAULT_MAX_PHANTOM_PER_NODE = 3      # max phantom links per node (prevents hub creation)
DEFAULT_MAX_TOTAL_PHANTOM = 100       # hard cap on total phantom links per cycle


def find_phantom_links(
    nodes: dict,
    edges: dict,
    vault_root: str,
    *,
    config: dict | None = None,
    collection=None,
) -> list[dict]:
    """Find semantically similar node pairs via ChromaDB nearest-neighbor search.

    For each node, query ChromaDB for the nearest neighbors. Filter to pairs
    that:
    - Are not already connected by a real wikilink
    - Exceed the similarity threshold
    - Don't exceed the per-node cap

    Parameters
    ----------
    nodes : dict
        Graph nodes (note_id -> node data).
    edges : dict
        Existing graph edges (note_id -> list of links).
    vault_root : str
        Path to the Obsidian vault (used when *collection* is not provided
        and *config* requires opening ChromaDB from disk).
    config : dict, optional
        Per-vault settings dict.  Falls back to global ``CONFIG``.
    collection : chromadb.Collection, optional
        Pre-opened ChromaDB collection.  When provided, *vault_root* and
        *config* are not used to open a new client, eliminating global
        ``CONFIG`` dependency.

    Returns
    -------
    list[dict]
        List of phantom link dicts: ``[{source, target, similarity}]``.
    """
    cfg = config or CONFIG
    threshold = cfg.get('phantom_similarity_threshold', DEFAULT_SIMILARITY_THRESHOLD)
    max_per_node = cfg.get('phantom_max_per_node', DEFAULT_MAX_PHANTOM_PER_NODE)
    max_total = cfg.get('phantom_max_total', DEFAULT_MAX_TOTAL_PHANTOM)

    # Open ChromaDB collection if not pre-supplied
    if collection is None:
        raw_cp = cfg.get('chroma_path', '.bdh-chroma')
        if os.path.isabs(raw_cp):
            chroma_path = raw_cp
        else:
            chroma_path = os.path.join(vault_root, raw_cp)
        if not os.path.isdir(chroma_path):
            logger.warning(f"Phantom links: ChromaDB not found at {chroma_path}")
            return []

        client = chromadb.PersistentClient(path=chroma_path)
        try:
            collection = client.get_collection(cfg.get('chroma_collection', 'notes'))
        except Exception:
            logger.warning("Phantom links: ChromaDB collection not found")
            return []

    if collection.count() == 0:
        return []

    # Build set of existing edges for fast lookup
    existing_pairs = set()
    for src, links in edges.items():
        for link in links:
            target = link['target'] if isinstance(link, dict) else link
            existing_pairs.add((src, target))
            existing_pairs.add((target, src))

    # Also exclude self-links
    node_ids = list(nodes.keys())

    # Query nearest neighbors for each node
    phantom_links = []
    node_phantom_count = {}  # track per-node count

    # Cache ChromaDB IDs once (avoids repeated full fetches)
    chroma_ids = set(collection.get()['ids'])

    # Batch query: get all embeddings and find nearest neighbors via ChromaDB
    # ChromaDB's query supports n_results, but we need to query per-node
    # since each node has a different embedding. For efficiency, query in batches.
    batch_size = 50
    for i in range(0, len(node_ids), batch_size):
        batch_ids = node_ids[i:i + batch_size]
        for nid in batch_ids:
            if nid not in chroma_ids:
                continue

            # Skip nodes that already have too many phantom links
            if node_phantom_count.get(nid, 0) >= max_per_node:
                continue

            try:
                nid_data = collection.get(ids=[nid], include=['embeddings'])
                embs = nid_data.get('embeddings') if nid_data else None
                if embs is None or len(embs) == 0:
                    continue
                results = collection.query(
                    query_embeddings=[embs[0]],
                    n_results=max_per_node + 5,  # extra to account for self + existing links
                    include=['distances'],
                )
            except Exception as e:
                logger.debug(f"Phantom query failed for {nid}: {e}")
                continue

            if not results or not results['ids'] or not results['ids'][0]:
                continue

            neighbors = results['ids'][0]
            distances = results['distances'][0] if results.get('distances') else []

            for j, neighbor_id in enumerate(neighbors):
                # Skip self
                if neighbor_id == nid:
                    continue

                # Skip if already connected by real wikilink
                if (nid, neighbor_id) in existing_pairs:
                    continue

                # Skip if neighbor doesn't exist in current graph
                if neighbor_id not in nodes:
                    continue

                # ChromaDB returns L2 distance; convert to cosine similarity
                # For normalized embeddings: cosine_sim = 1 - (l2_dist^2 / 2)
                l2_dist = distances[j] if j < len(distances) else 1.0
                similarity = max(0.0, 1.0 - (l2_dist ** 2) / 2.0)

                if similarity < threshold:
                    continue

                # Check per-node cap
                if node_phantom_count.get(nid, 0) >= max_per_node:
                    break
                if node_phantom_count.get(neighbor_id, 0) >= max_per_node:
                    continue

                # Check total cap
                if len(phantom_links) >= max_total:
                    return phantom_links

                phantom_links.append({
                    'source': nid,
                    'target': neighbor_id,
                    'similarity': round(similarity, 4),
                })
                node_phantom_count[nid] = node_phantom_count.get(nid, 0) + 1
                node_phantom_count[neighbor_id] = node_phantom_count.get(neighbor_id, 0) + 1

    # Sort by similarity (strongest first)
    phantom_links.sort(key=lambda x: -x['similarity'])

    return phantom_links[:max_total]


def update_phantom_links(
    state: dict,
    nodes: dict,
    edges: dict,
    vault_root: str,
    *,
    config: dict | None = None,
    collection=None,
) -> dict:
    """Recompute phantom links and update state.

    Called during consolidation to refresh semantic connections.
    Phantom links replace previous ones each cycle (full recomputation).

    Parameters
    ----------
    state : dict
        The harness state dict (mutated in place).
    nodes : dict
        Graph nodes.
    edges : dict
        Graph edges.
    vault_root : str
        Vault path for ChromaDB access (when *collection* is not provided).
    config : dict, optional
        Per-vault settings dict.  Falls back to global ``CONFIG``.
    collection : chromadb.Collection, optional
        Pre-opened ChromaDB collection.

    Returns
    -------
    dict
        The mutated state.
    """
    cfg = config or CONFIG
    phantom = find_phantom_links(nodes, edges, vault_root, config=cfg, collection=collection)
    state['phantom_links'] = phantom

    logger.info(
        f"Phantom links: {len(phantom)} semantic connections "
        f"(threshold={cfg.get('phantom_similarity_threshold', DEFAULT_SIMILARITY_THRESHOLD)})"
    )

    return state


def get_phantom_edges(state: dict) -> list[dict]:
    """Get phantom links formatted for the visualization API.

    Returns
    -------
    list[dict]
        List of ``{source, target, similarity}`` dicts.
    """
    return state.get('phantom_links', [])
