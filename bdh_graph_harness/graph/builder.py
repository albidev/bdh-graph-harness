"""
BDH Graph Harness — Graph builder module.

Builds the note graph from an Obsidian vault with caching support.
"""
import os
import json
import fnmatch
from collections import defaultdict
from datetime import datetime

from bdh_graph_harness.config import CONFIG, logger
from bdh_graph_harness.graph.parser import (
    extract_note_id,
    parse_frontmatter,
    extract_text,
    extract_wikilinks,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GRAPH_CACHE_FILE = ".bdh-graph-cache.json"


# ---------------------------------------------------------------------------
# Ignore filter
# ---------------------------------------------------------------------------

def _is_ignored(note_id: str, ignore_list=None) -> bool:
    """Check if a note_id should be excluded from the graph."""
    ignore_list = CONFIG.get('graph_ignore', []) if ignore_list is None else ignore_list
    for pattern in ignore_list:
        if fnmatch.fnmatch(note_id, pattern):
            return True
    return False


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph(vault_root, use_cache=True, graph_ignore=None):
    """Build the note graph: nodes with text, edges from wikilinks.

    With use_cache=True, loads from .bdh-graph-cache.json and only re-reads
    files whose mtime has changed since last cache. Falls back to full rebuild
    if cache is missing, corrupt, or file count has changed drastically.
    """
    cache_path = os.path.join(vault_root, GRAPH_CACHE_FILE)

    # Try loading cached graph
    cached = None
    if use_cache and os.path.isfile(cache_path):
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cached = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Graph cache corrupt, full rebuild: {e}")
            cached = None

    if cached:
        return _incremental_graph_update(vault_root, cached, cache_path, graph_ignore)
    else:
        nodes, edges = _full_graph_build(vault_root, graph_ignore)
        _save_graph_cache(vault_root, nodes, edges, cache_path)
        return nodes, edges


def _full_graph_build(vault_root, ignore_list=None):
    """Full graph build by walking the entire vault."""
    nodes = {}  # note_id -> {text, title, tags, path, mtime}
    edges = defaultdict(list)  # note_id -> [(target_id, ...), ...]
    ignored = set()

    for root, dirs, files in os.walk(vault_root):
        # Skip hidden dirs, .obsidian, raw/
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '.obsidian']
        for f in files:
            if not f.endswith('.md'):
                continue
            filepath = os.path.join(root, f)
            note_id = extract_note_id(filepath, vault_root)

            # Skip ignored notes
            if _is_ignored(note_id, ignore_list):
                ignored.add(note_id)
                continue

            mtime = os.path.getmtime(filepath)
            with open(filepath, 'r', encoding='utf-8') as fh:
                content = fh.read()

            fm = parse_frontmatter(content)
            text = extract_text(content)
            links = extract_wikilinks(content)

            nodes[note_id] = {
                'id': note_id,
                'title': fm.get('title', os.path.basename(f)[:-3]),
                'tags': fm.get('tags', ''),
                'text': text,
                'path': filepath,
                'mtime': mtime,
            }

            for target, display in links:
                edges[note_id].append({
                    'target': target,
                    'display': display,
                })

    # Filter edges pointing to ignored nodes
    if ignored:
        for src in list(edges.keys()):
            edges[src] = [l for l in edges[src]
                         if _resolve_target(l['target'], nodes) is not None]
        logger.info(f"Graph ignore: excluded {len(ignored)} nodes")

    return nodes, dict(edges)


def _incremental_graph_update(vault_root, cached, cache_path, ignore_list=None):
    """Update cached graph by only re-reading changed files.

    Compares mtimes: if a file's mtime changed, re-read it. If a file
    disappeared, remove it. If new files appeared, add them. Also rebuild
    edges for any changed node (since wikilinks may have changed).
    """
    cached_nodes = cached.get('nodes', {})
    cached_edges = cached.get('edges', {})
    cached_mtimes = {nid: n.get('mtime', 0) for nid, n in cached_nodes.items()}

    # Walk vault to find current files
    current_files = {}  # note_id -> (filepath, mtime)
    for root, dirs, files in os.walk(vault_root):
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '.obsidian']
        for f in files:
            if not f.endswith('.md'):
                continue
            filepath = os.path.join(root, f)
            note_id = extract_note_id(filepath, vault_root)
            # Skip ignored notes
            if _is_ignored(note_id, ignore_list):
                continue
            mtime = os.path.getmtime(filepath)
            current_files[note_id] = (filepath, mtime)

    # Determine what changed
    current_ids = set(current_files.keys())
    cached_ids = set(cached_nodes.keys())

    new_notes = current_ids - cached_ids
    deleted_notes = cached_ids - current_ids
    potentially_changed = current_ids & cached_ids

    changed_notes = set()
    for nid in potentially_changed:
        if current_files[nid][1] != cached_mtimes.get(nid, 0):
            changed_notes.add(nid)

    total_changes = len(new_notes) + len(deleted_notes) + len(changed_notes)

    if total_changes == 0:
        # Nothing changed — return cached graph as-is
        logger.info(f"Graph cache hit (0 changes, {len(cached_nodes)} nodes)")
        print(f"   ✓ Graph cache hit — 0 changes, {len(cached_nodes)} neurons")
        return cached_nodes, cached_edges

    # If >50% of notes changed, do a full rebuild (more efficient)
    if total_changes > len(current_ids) * 0.5:
        logger.info(f"Too many changes ({total_changes}/{len(current_ids)}), full rebuild")
        print(f"   🔄 {total_changes} changes (>50%), full rebuild")
        nodes, edges = _full_graph_build(vault_root, ignore_list)
        _save_graph_cache(vault_root, nodes, edges, cache_path)
        return nodes, edges

    # Incremental update
    nodes = dict(cached_nodes)
    edges = dict(cached_edges)

    # Remove deleted notes (also remove notes that are now ignored)
    for nid in deleted_notes:
        nodes.pop(nid, None)
        edges.pop(nid, None)
        # Also remove edges pointing to deleted notes
        for src, links in edges.items():
            edges[src] = [l for l in links if _resolve_target(l['target'], nodes) != nid
                         and l['target'] != nid]

    # Re-read changed and new notes
    to_read = new_notes | changed_notes
    for nid in to_read:
        filepath, mtime = current_files[nid]
        with open(filepath, 'r', encoding='utf-8') as fh:
            content = fh.read()

        fm = parse_frontmatter(content)
        text = extract_text(content)
        links = extract_wikilinks(content)

        nodes[nid] = {
            'id': nid,
            'title': fm.get('title', os.path.basename(filepath)[:-3]),
            'tags': fm.get('tags', ''),
            'text': text,
            'path': filepath,
            'mtime': mtime,
        }

        edges[nid] = [{'target': t, 'display': d} for t, d in links]

    print(f"   🔄 Incremental update: {len(new_notes)} new, {len(changed_notes)} changed, {len(deleted_notes)} deleted")
    logger.info(f"Graph incremental: +{len(new_notes)} ~{len(changed_notes)} -{len(deleted_notes)}")

    _save_graph_cache(vault_root, nodes, edges, cache_path)
    return nodes, edges


def _save_graph_cache(vault_root, nodes, edges, cache_path):
    """Save graph + mtimes to cache file for incremental rebuilds."""
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


def _resolve_target(target, nodes):
    """Resolve a wikilink target to an actual note ID in the graph."""
    # Direct match
    if target in nodes:
        return target
    # Try with wiki/ prefix or without
    for prefix in ['', 'wiki/', 'concepts/', 'entities/', 'comparisons/', 'queries/']:
        candidate = f"{prefix}{target}" if prefix else target
        if candidate in nodes:
            return candidate
    # Try matching by basename
    basename = os.path.basename(target)
    for note_id in nodes:
        if os.path.basename(note_id) == basename:
            return note_id
    return None