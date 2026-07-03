"""
BDH Graph Harness — Graph cache helpers.

Thin wrappers around the builder's cache save/load functionality.
"""

import os
import json

from bdh_graph_harness.config import logger
from bdh_graph_harness.graph.builder import GRAPH_CACHE_FILE


def load_graph_cache(vault_root):
    """Load cached graph from disk. Returns (nodes, edges) or (None, None) if missing/corrupt."""
    cache_path = os.path.join(vault_root, GRAPH_CACHE_FILE)
    if not os.path.isfile(cache_path):
        return None, None
    try:
        with open(cache_path, 'r', encoding='utf-8') as f:
            cached = json.load(f)
        return cached.get('nodes', {}), cached.get('edges', {})
    except (json.JSONDecodeError, IOError) as e:
        logger.warning(f"Graph cache corrupt: {e}")
        return None, None


def save_graph_cache(vault_root, nodes, edges):
    """Save graph + mtimes to cache file for incremental rebuilds."""
    from datetime import datetime
    cache_path = os.path.join(vault_root, GRAPH_CACHE_FILE)
    cache = {
        'nodes': nodes,
        'edges': edges,
        'cached_at': datetime.now().isoformat(),
        'vault_path': vault_root,
    }
    try:
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(cache, f)
    except IOError as e:
        logger.warning(f"Failed to save graph cache: {e}")