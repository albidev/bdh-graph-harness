"""
BDH Graph Harness — Graph subpackage.

Re-exports main functions for easy importing.
"""

from bdh_graph_harness.graph.parser import (
    WIKILINK_RE,
    FRONTMATTER_RE,
    extract_note_id,
    find_note_by_id,
    parse_frontmatter,
    extract_wikilinks,
    extract_text,
)
from bdh_graph_harness.graph.builder import (
    GRAPH_CACHE_FILE,
    build_graph,
    _full_graph_build,
    _incremental_graph_update,
    _save_graph_cache,
    _resolve_target,
)
from bdh_graph_harness.graph.cache import (
    load_graph_cache,
    save_graph_cache,
)

__all__ = [
    'WIKILINK_RE',
    'FRONTMATTER_RE',
    'extract_note_id',
    'find_note_by_id',
    'parse_frontmatter',
    'extract_wikilinks',
    'extract_text',
    'GRAPH_CACHE_FILE',
    'build_graph',
    '_full_graph_build',
    '_incremental_graph_update',
    '_save_graph_cache',
    '_resolve_target',
    'load_graph_cache',
    'save_graph_cache',
]