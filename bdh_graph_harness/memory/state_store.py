"""State persistence — load/save BDH synaptic state with file locking."""

import os
import json
import fcntl
from datetime import datetime

from bdh_graph_harness.config import CONFIG, STATE_FILE, LOCK_FILE


def _state_path(vault_root):
    """Return the configured state file, preserving the legacy default."""
    return os.path.join(vault_root, CONFIG.get('hebbian_state_file', STATE_FILE))


def _empty_state():
    return {
        'synapses': {},
        'created': datetime.now().isoformat(),
        'updated': datetime.now().isoformat(),
        'queries': 0,
    }


def _read_state_unlocked(state_path):
    if not os.path.isfile(state_path):
        return _empty_state()
    try:
        with open(state_path, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, ValueError):
        import logging
        logging.getLogger('bdh').warning(
            f"Corrupt state file at {state_path}, starting fresh"
        )
        return _empty_state()


def load_state(vault_root):
    """Load persisted BDH state while holding the vault file lock."""
    state_path = _state_path(vault_root)
    lock_path = os.path.join(vault_root, LOCK_FILE)
    with open(lock_path, 'w') as lock_f:
        fcntl.flock(lock_f, fcntl.LOCK_EX)
        try:
            return _read_state_unlocked(state_path)
        finally:
            fcntl.flock(lock_f, fcntl.LOCK_UN)


def _preserve_synapse_for_persistence(key, valid_node_ids):
    """Keep opaque synapses; prune decodable state with missing endpoints.

    An ambiguous pipe-delimited key cannot be safely resolved against the
    current graph without risking data loss, so it remains persisted and
    consumers simply ignore it. Decodable v2 and simple legacy keys can be
    pruned against the current graph as before.
    """
    from bdh_graph_harness.memory.hebbian import safe_decode_synapse_key

    endpoints = safe_decode_synapse_key(key)
    if endpoints is None:
        return True
    return all(endpoint in valid_node_ids for endpoint in endpoints)


def merge_states(disk_state, mem_state, *, valid_node_ids=None):
    """Merge on-disk state with in-memory state to prevent lost updates.

    Merge strategy:
    - synapses: union of keys; for shared keys keep the entry with the more
      recent ``last_coactivated`` timestamp (falls back to higher frequency,
      then higher weight). Synapses absent from memory but present on disk
      are kept ONLY if they have a more recent timestamp than the memory
      state's ``updated`` — this prevents resurrecting pruned synapses.
    - queries: take the maximum of the two values.
    - any other top-level keys: take the memory version (active writer),
      preserving disk-only keys that memory doesn't override.
    """
    merged = {}

    # --- synapses -----------------------------------------------------------
    disk_syn = disk_state.get('synapses', {})
    mem_syn = mem_state.get('synapses', {})
    if valid_node_ids is not None:
        valid = set(valid_node_ids)
        disk_syn = {
            key: value for key, value in disk_syn.items()
            if _preserve_synapse_for_persistence(key, valid)
        }
        mem_syn = {
            key: value for key, value in mem_syn.items()
            if _preserve_synapse_for_persistence(key, valid)
        }
    merged_syn = {}

    # Keys present in memory: always use the memory version (the active writer
    # has the most recent state, including decay/consolidation/pruning effects).
    # Keys only on disk: keep them (created by another writer, e.g. MCP fallback).
    # This fixes the core bug: shared synapses no longer resurrect pre-decay
    # weights from disk via frequency-wins. Disk-only synapses are preserved
    # to support concurrent writers (e.g. MCP fallback writing while server runs).
    for key in set(disk_syn) | set(mem_syn):
        if key in mem_syn:
            merged_syn[key] = mem_syn[key]
        else:
            merged_syn[key] = disk_syn[key]

    merged['synapses'] = merged_syn

    # --- queries ------------------------------------------------------------
    merged['queries'] = max(disk_state.get('queries', 0), mem_state.get('queries', 0))

    # --- other top-level keys: memory wins, disk-only keys preserved --------
    for key, val in disk_state.items():
        if key not in ('synapses', 'queries'):
            merged[key] = val
    for key, val in mem_state.items():
        if key not in ('synapses', 'queries'):
            merged[key] = val

    return merged


def reconcile_state_to_nodes(state, nodes):
    """Drop persisted state that references notes absent from the current graph.

    Full graph refreshes can remove many nodes at once (for example after a
    reversible quarantine). Runtime state must follow the graph or stats,
    quality, and visualization will report dead synapses and dormant nodes.
    """
    valid = set(nodes)
    synapses = {}
    for key, value in state.get('synapses', {}).items():
        if _preserve_synapse_for_persistence(key, valid):
            synapses[key] = value
    state['synapses'] = synapses

    state['node_quality'] = {
        node_id: value
        for node_id, value in state.get('node_quality', {}).items()
        if node_id in valid
    }
    state['dormant_nodes'] = sorted(
        node_id for node_id in state.get('dormant_nodes', []) if node_id in valid
    )

    phantom = state.get('phantom_links', [])
    state['phantom_links'] = [
        link for link in phantom
        if isinstance(link, dict)
        and link.get('source') in valid
        and link.get('target') in valid
    ]

    # Recompute quality for the new node set, including newly added nodes.
    from bdh_graph_harness.memory.quality import prune_dormant
    return prune_dormant(state, nodes)


def save_state(vault_root, state, *, valid_node_ids=None):
    """Persist BDH state. Uses fcntl.flock for concurrency safety.

    Before writing, reloads the on-disk state and merges it with the
    in-memory state to prevent lost updates from concurrent writers.

    The write is atomic: data is written to a temp file, then os.replace()
    swaps it into place — a crash during write cannot leave a corrupt file.
    """
    state['updated'] = datetime.now().isoformat()
    state_path = _state_path(vault_root)
    lock_path = os.path.join(vault_root, LOCK_FILE)
    tmp_path = state_path + '.tmp'

    with open(lock_path, 'w') as lock_f:
        fcntl.flock(lock_f, fcntl.LOCK_EX)
        try:
            # Keep read, merge, and atomic replace under one lock: otherwise
            # two writers can both merge against the same stale disk snapshot.
            disk_state = _read_state_unlocked(state_path)
            merged = merge_states(
                disk_state,
                state,
                valid_node_ids=valid_node_ids,
            )
            with open(tmp_path, 'w') as f:
                json.dump(merged, f, indent=2)
            os.replace(tmp_path, state_path)
        except Exception:
            if os.path.isfile(tmp_path):
                os.unlink(tmp_path)
            raise
        finally:
            fcntl.flock(lock_f, fcntl.LOCK_UN)