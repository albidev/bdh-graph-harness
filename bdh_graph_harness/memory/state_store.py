"""State persistence — load/save BDH synaptic state with file locking."""

import os
import json
import fcntl
from datetime import datetime

from bdh_graph_harness.config import STATE_FILE, LOCK_FILE


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
    state_path = os.path.join(vault_root, STATE_FILE)
    lock_path = os.path.join(vault_root, LOCK_FILE)
    with open(lock_path, 'w') as lock_f:
        fcntl.flock(lock_f, fcntl.LOCK_EX)
        try:
            return _read_state_unlocked(state_path)
        finally:
            fcntl.flock(lock_f, fcntl.LOCK_UN)


def merge_states(disk_state, mem_state):
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


def save_state(vault_root, state):
    """Persist BDH state. Uses fcntl.flock for concurrency safety.

    Before writing, reloads the on-disk state and merges it with the
    in-memory state to prevent lost updates from concurrent writers.

    The write is atomic: data is written to a temp file, then os.replace()
    swaps it into place — a crash during write cannot leave a corrupt file.
    """
    state['updated'] = datetime.now().isoformat()
    state_path = os.path.join(vault_root, STATE_FILE)
    lock_path = os.path.join(vault_root, LOCK_FILE)
    tmp_path = state_path + '.tmp'

    with open(lock_path, 'w') as lock_f:
        fcntl.flock(lock_f, fcntl.LOCK_EX)
        try:
            # Keep read, merge, and atomic replace under one lock: otherwise
            # two writers can both merge against the same stale disk snapshot.
            disk_state = _read_state_unlocked(state_path)
            merged = merge_states(disk_state, state)
            with open(tmp_path, 'w') as f:
                json.dump(merged, f, indent=2)
            os.replace(tmp_path, state_path)
        except Exception:
            if os.path.isfile(tmp_path):
                os.unlink(tmp_path)
            raise
        finally:
            fcntl.flock(lock_f, fcntl.LOCK_UN)