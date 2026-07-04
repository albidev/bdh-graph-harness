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
                    state = json.load(f)
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
    - synapses: union of keys; for shared keys keep the entry with higher
      ``frequency`` (higher ``weight`` as tiebreaker).
    - queries: take the maximum of the two values.
    - any other top-level keys: take the memory version (active writer),
      preserving disk-only keys that memory doesn't override.
    """
    merged = {}

    # --- synapses -----------------------------------------------------------
    disk_syn = disk_state.get('synapses', {})
    mem_syn = mem_state.get('synapses', {})
    merged_syn = {}
    for key in set(disk_syn) | set(mem_syn):
        if key not in mem_syn:
            merged_syn[key] = disk_syn[key]
        elif key not in disk_syn:
            merged_syn[key] = mem_syn[key]
        else:
            d = disk_syn[key]
            m = mem_syn[key]
            d_freq = d.get('frequency', 0)
            m_freq = m.get('frequency', 0)
            if m_freq > d_freq:
                merged_syn[key] = m
            elif d_freq > m_freq:
                merged_syn[key] = d
            else:
                # tie on frequency → higher weight wins
                d_w = d.get('weight', 0)
                m_w = m.get('weight', 0)
                merged_syn[key] = m if m_w >= d_w else d
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
    """
    state['updated'] = datetime.now().isoformat()
    state_path = os.path.join(vault_root, STATE_FILE)
    lock_path = os.path.join(vault_root, LOCK_FILE)

    # Reload disk state and merge to avoid clobbering concurrent writes.
    disk_state = load_state(vault_root)
    merged = merge_states(disk_state, state)

    with open(lock_path, 'w') as lock_f:
        fcntl.flock(lock_f, fcntl.LOCK_EX)
        try:
            with open(state_path, 'w') as f:
                json.dump(merged, f, indent=2)
        finally:
            fcntl.flock(lock_f, fcntl.LOCK_UN)