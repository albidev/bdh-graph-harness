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


def save_state(vault_root, state):
    """Persist BDH state. Uses fcntl.flock for concurrency safety."""
    state['updated'] = datetime.now().isoformat()
    state_path = os.path.join(vault_root, STATE_FILE)
    lock_path = os.path.join(vault_root, LOCK_FILE)

    with open(lock_path, 'w') as lock_f:
        fcntl.flock(lock_f, fcntl.LOCK_EX)
        try:
            with open(state_path, 'w') as f:
                json.dump(state, f, indent=2)
        finally:
            fcntl.flock(lock_f, fcntl.LOCK_UN)