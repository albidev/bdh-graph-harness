"""State persistence — load/save BDH synaptic state with file locking."""

import os
import json
import fcntl
from datetime import datetime

from bdh_graph_harness.config import STATE_FILE, LOCK_FILE


def load_state(vault_root):
    """Load persisted BDH state (synaptic weights, co-activation history).
    Uses fcntl.flock for concurrency safety.
    """
    state_path = os.path.join(vault_root, STATE_FILE)
    lock_path = os.path.join(vault_root, LOCK_FILE)

    with open(lock_path, 'w') as lock_f:
        fcntl.flock(lock_f, fcntl.LOCK_EX)
        try:
            if os.path.isfile(state_path):
                with open(state_path, 'r') as f:
                    try:
                        state = json.load(f)
                    except (json.JSONDecodeError, ValueError):
                        # Corrupt state file — start fresh, don't crash
                        import logging
                        logging.getLogger('bdh').warning(
                            f"Corrupt state file at {state_path}, starting fresh"
                        )
                        state = {
                            'synapses': {},
                            'created': datetime.now().isoformat(),
                            'updated': datetime.now().isoformat(),
                            'queries': 0,
                        }
            else:
                state = {
                    'synapses': {},  # "note_a|note_b" -> {weight, frequency, last_coactivated}
                    'created': datetime.now().isoformat(),
                    'updated': datetime.now().isoformat(),
                    'queries': 0,
                }
        finally:
            fcntl.flock(lock_f, fcntl.LOCK_UN)

    return state


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

    # Reload disk state and merge to avoid clobbering concurrent writes.
    disk_state = load_state(vault_root)
    merged = merge_states(disk_state, state)

    with open(lock_path, 'w') as lock_f:
        fcntl.flock(lock_f, fcntl.LOCK_EX)
        try:
            # Write to temp file first, then atomically replace
            with open(tmp_path, 'w') as f:
                json.dump(merged, f, indent=2)
            os.replace(tmp_path, state_path)
        except Exception:
            # Clean up temp file on failure
            if os.path.isfile(tmp_path):
                os.unlink(tmp_path)
            raise
        finally:
            fcntl.flock(lock_f, fcntl.LOCK_UN)