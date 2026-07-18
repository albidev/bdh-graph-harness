"""Memory consolidation — sleep-cycle graph maintenance.

Biological brains consolidate memories during sleep via three mechanisms:

1. **Synaptic homeostasis** — global downscaling of all synaptic weights,
   preventing saturation. Strong (frequently-used) synapses survive; weak
   ones fall below pruning threshold naturally.

2. **Structural pruning** — synapses below a floor weight are deleted
   outright. Nodes that have been dormant for multiple consolidation
   cycles are removed from the quality map entirely (not just hidden).

3. **Quality re-evaluation** — after downscaling + pruning, node quality
   scores are recomputed from the surviving synapse set, and dormancy
   status is updated.

This module is called by the ``/api/consolidate`` endpoint, which is
designed to be triggered via cron (e.g. nightly) or manually. It is NOT
called during the query cycle — consolidation is a separate process,
mirroring the biological separation between waking (retrieval/plasticity)
and sleep (consolidation).

Concurrency
-----------
``consolidate()`` operates on the ``app_state['state']`` dict in-place.
The caller (API endpoint) is responsible for acquiring an ``asyncio.Lock``
if concurrent queries are expected. For cron-based nightly runs when the
system is idle, no lock is needed.
"""

from datetime import datetime, timedelta
from copy import deepcopy

from bdh_graph_harness.config import CONFIG, logger
from bdh_graph_harness.memory.quality import compute_all_qualities


# ---------------------------------------------------------------------------
# Defaults (override via config)
# ---------------------------------------------------------------------------
DEFAULT_DOWNSCALE_FACTOR = 0.90       # multiply all weights by this
DEFAULT_PRUNE_WEIGHT_FLOOR = 0.02     # delete synapses below this weight
DEFAULT_WEAK_WEIGHT_THRESHOLD = 0.15
DEFAULT_WEAK_MAX_FREQUENCY = 1.0
DEFAULT_WEAK_MIN_AGE_HOURS = 48
DEFAULT_DORMANT_PERSIST_CYCLES = 3    # remove nodes dormant for N+ cycles
DEFAULT_PRUNE_DORMANT_NODES = True    # actually remove stale dormant nodes


# ---------------------------------------------------------------------------
# Core consolidation steps
# ---------------------------------------------------------------------------

def synaptic_downscaling(state: dict, factor: float | None = None) -> dict:
    """Global synaptic homeostasis — scale all weights by ``factor``.

    This mirrors the sleep-phase downscaling observed in biological brains.
    Strong synapses (weight near 1.0) lose a small fraction; weak synapses
    (weight near 0.05) drop toward zero and become candidates for pruning.

    Parameters
    ----------
    state : dict
        The harness state dict (mutated in place).
    factor : float, optional
        Downscaling factor. Defaults to ``CONFIG['consolidation_downscale_factor']``
        or ``0.90``.

    Returns
    -------
    dict
        The mutated state (same reference).
    """
    if factor is None:
        factor = CONFIG.get('consolidation_downscale_factor', DEFAULT_DOWNSCALE_FACTOR)

    synapses = state.get('synapses', {})
    for syn in synapses.values():
        syn['weight'] = round(syn.get('weight', 0.0) * factor, 6)

    return state


def structural_pruning(state: dict, weight_floor: float | None = None) -> dict:
    """Delete synapses with weight below ``weight_floor``.

    Called after downscaling, this removes the synapses that were pushed
    below the floor by the global scaling step.

    Parameters
    ----------
    state : dict
        The harness state dict (mutated in place).
    weight_floor : float, optional
        Minimum weight to survive. Defaults to
        ``CONFIG['consolidation_prune_weight_floor']`` or ``0.02``.

    Returns
    -------
    dict
        The mutated state (same reference).
    """
    if weight_floor is None:
        weight_floor = CONFIG.get('consolidation_prune_weight_floor', DEFAULT_PRUNE_WEIGHT_FLOOR)

    synapses = state.get('synapses', {})
    to_delete = [
        key for key, syn in synapses.items()
        if syn.get('weight', 0.0) < weight_floor
    ]
    for key in to_delete:
        del synapses[key]

    return state


def _coactivation_timestamp(value):
    """Parse an ISO timestamp, returning ``None`` for malformed values."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    except (TypeError, ValueError):
        logger.warning("Ignoring malformed synapse timestamp: %r", value)
        return None


def is_stale_weak_synapse(
    synapse: dict,
    *,
    now: datetime | None = None,
    config: dict | None = None,
) -> bool:
    """Return whether a weak, low-frequency synapse is old enough to prune."""
    cfg = config or CONFIG
    threshold = cfg.get(
        'consolidation_weak_weight_threshold', DEFAULT_WEAK_WEIGHT_THRESHOLD
    )
    max_frequency = cfg.get(
        'consolidation_weak_max_frequency', DEFAULT_WEAK_MAX_FREQUENCY
    )
    min_age_hours = cfg.get(
        'consolidation_weak_min_age_hours', DEFAULT_WEAK_MIN_AGE_HOURS
    )

    try:
        weight = float(synapse.get('weight', 0.0))
        frequency = float(synapse.get('frequency', 0.0))
    except (TypeError, ValueError):
        return False
    if weight >= threshold or frequency > max_frequency:
        return False

    last_coactivated = _coactivation_timestamp(synapse.get('last_coactivated'))
    if last_coactivated is None:
        return False
    effective_now = now or datetime.now(last_coactivated.tzinfo)
    if last_coactivated.tzinfo is None and effective_now.tzinfo is not None:
        last_coactivated = last_coactivated.replace(tzinfo=effective_now.tzinfo)
    elif last_coactivated.tzinfo is not None and effective_now.tzinfo is None:
        effective_now = effective_now.replace(tzinfo=last_coactivated.tzinfo)
    return last_coactivated <= effective_now - timedelta(hours=float(min_age_hours))


def prune_stale_weak(
    state: dict,
    *,
    now: datetime | None = None,
    config: dict | None = None,
) -> int:
    """Delete only stale weak/low-frequency synapses and return the count."""
    synapses = state.get('synapses', {})
    to_delete = [
        key for key, synapse in synapses.items()
        if is_stale_weak_synapse(synapse, now=now, config=config)
    ]
    for key in to_delete:
        del synapses[key]
    return len(to_delete)


def hebbian_tail_stats(state: dict, *, config: dict | None = None) -> dict:
    """Classify Hebbian synapses without mutating state."""
    cfg = config or CONFIG
    threshold = cfg.get(
        'consolidation_weak_weight_threshold', DEFAULT_WEAK_WEIGHT_THRESHOLD
    )
    synapses = state.get('synapses', {}).values()
    strong = 0
    weak = 0
    stale_weak = 0
    for synapse in synapses:
        try:
            is_weak = float(synapse.get('weight', 0.0)) < threshold
        except (TypeError, ValueError):
            is_weak = False
        if is_weak:
            weak += 1
            if is_stale_weak_synapse(synapse, config=cfg):
                stale_weak += 1
        else:
            strong += 1
    return {
        'hebbian_strong_synapses': strong,
        'hebbian_weak_synapses': weak,
        'hebbian_stale_weak_synapses': stale_weak,
    }


def prune_stale_dormant(state: dict, nodes: dict, persist_cycles: int | None = None) -> dict:
    """Remove nodes that have been dormant for too many consolidation cycles.

    Unlike the periodic quality pruning in ``hebbian_update()`` which only
    *marks* nodes as dormant (hiding them), this function *removes* stale
    dormant nodes from the quality map entirely. They can be re-evaluated
    from scratch if a future query activates them.

    A node is removed if:
    - It is currently dormant.
    - Its ``dormant_cycles`` counter (tracked in node_quality) is >=
      ``persist_cycles``.

    The counter is incremented each time ``consolidate()`` runs and the
    node is still dormant. It is reset to 0 when a node is re-activated.

    Parameters
    ----------
    state : dict
        The harness state dict (mutated in place).
    nodes : dict
        The graph nodes dict (used to skip nodes that no longer exist).
    persist_cycles : int, optional
        Number of dormant cycles before removal. Defaults to
        ``CONFIG['consolidation_dormant_persist_cycles']`` or ``3``.

    Returns
    -------
    dict
        The mutated state (same reference).
    """
    if persist_cycles is None:
        persist_cycles = CONFIG.get(
            'consolidation_dormant_persist_cycles',
            DEFAULT_DORMANT_PERSIST_CYCLES,
        )

    nq = state.get('node_quality', {})
    dormant_list = state.get('dormant_nodes', [])
    now = datetime.now().isoformat()

    removed = []
    for nid in list(nq.keys()):
        # Skip nodes that no longer exist in the graph (already deleted from vault)
        if nid not in nodes:
            removed.append(nid)
            del nq[nid]
            continue

        entry = nq[nid]
        if entry.get('dormant', False):
            entry['dormant_cycles'] = entry.get('dormant_cycles', 0) + 1
            entry['evaluated_at'] = now
            if entry['dormant_cycles'] >= persist_cycles:
                removed.append(nid)
                del nq[nid]
        else:
            # Reset counter for active nodes
            entry['dormant_cycles'] = 0

    # Clean up dormant_nodes lookup list
    surviving_dormant = sorted(
        nid for nid, q in nq.items() if q.get('dormant', False)
    )
    state['dormant_nodes'] = surviving_dormant

    # Clean up synapses that reference removed nodes
    synapses = state.get('synapses', {})
    syn_to_delete = []
    for key in synapses:
        a, b = key.split('|')
        if a in removed or b in removed:
            syn_to_delete.append(key)
    for key in syn_to_delete:
        del synapses[key]

    return state


# ---------------------------------------------------------------------------
# Full consolidation pass
# ---------------------------------------------------------------------------

def consolidate(
    state: dict,
    nodes: dict,
    edges: dict | None = None,
    *,
    config: dict | None = None,
    collection=None,
) -> dict:
    """Run a full consolidation cycle (sleep phase).

    Steps:
      1. Synaptic downscaling (global weight reduction).
      2. Structural pruning (delete synapses below floor).
      3. Quality re-evaluation (recompute scores from surviving synapses).
      4. Stale dormant pruning (remove nodes dormant for too many cycles).

    The state is mutated in place and also returned. The caller should
    persist it via ``save_state()`` after this function returns.

    Parameters
    ----------
    state : dict
        The harness state dict.
    nodes : dict
        The graph nodes dict (for quality re-evaluation and stale pruning).

    Returns
    -------
    dict
        A results dict with statistics about the consolidation:

        ``{
            'downscale_factor': float,
            'synapses_before': int,
            'synapses_after': int,
            'synapses_pruned': int,
            'nodes_before': int,
            'nodes_after': int,
            'nodes_removed': int,
            'dormant_before': int,
            'dormant_after': int,
            'cycles': int,
            'timestamp': str,
        }``
    """
    from bdh_graph_harness.memory.quality import prune_dormant

    cfg = config or CONFIG

    synapses_before = len(state.get('synapses', {}))
    nq_before = state.get('node_quality', {})
    dormant_before = len(state.get('dormant_nodes', []))

    # Track consolidation cycles
    state['consolidation_cycles'] = state.get('consolidation_cycles', 0) + 1
    cycles = state['consolidation_cycles']

    # Step 1: Global downscaling
    factor = cfg.get('consolidation_downscale_factor', DEFAULT_DOWNSCALE_FACTOR)
    state = synaptic_downscaling(state, factor)

    # Step 2: Remove only stale weak traces; fresh/frequent weak memory survives.
    stale_weak_pruned = prune_stale_weak(state, config=cfg)

    # Step 3: Structural pruning
    floor = cfg.get('consolidation_prune_weight_floor', DEFAULT_PRUNE_WEIGHT_FLOOR)
    state = structural_pruning(state, floor)

    # Step 4: Quality re-evaluation from surviving synapses
    # Save dormant_cycles so prune_dormant's full rebuild doesn't lose them
    old_dormant_cycles = {
        nid: q.get('dormant_cycles', 0)
        for nid, q in nq_before.items()
    }
    state = prune_dormant(state, nodes)
    # Restore dormant_cycles into the freshly computed quality map
    for nid, q in state.get('node_quality', {}).items():
        if nid in old_dormant_cycles:
            q['dormant_cycles'] = old_dormant_cycles[nid]

    # Step 5: Remove stale dormant nodes
    if cfg.get('consolidation_prune_dormant_nodes', DEFAULT_PRUNE_DORMANT_NODES):
        persist = cfg.get(
            'consolidation_dormant_persist_cycles',
            DEFAULT_DORMANT_PERSIST_CYCLES,
        )
        state = prune_stale_dormant(state, nodes, persist)

    # Step 6: Phantom links — semantic similarity connections
    if cfg.get('phantom_links_enabled', True):
        from bdh_graph_harness.memory.phantom import update_phantom_links
        vault_root = cfg.get('vault_path', '')
        if vault_root and edges:
            state = update_phantom_links(
                state, nodes, edges or {}, vault_root,
                config=cfg, collection=collection,
            )

    # Recompute dormant count after stale pruning
    dormant_after = len(state.get('dormant_nodes', []))
    synapses_after = len(state.get('synapses', {}))
    nq_after = state.get('node_quality', {})

    results = {
        'downscale_factor': factor,
        'weight_floor': floor,
        'synapses_before': synapses_before,
        'synapses_after': synapses_after,
        'synapses_pruned': synapses_before - synapses_after,
        'stale_weak_pruned': stale_weak_pruned,
        'nodes_before': len(nq_before),
        'nodes_after': len(nq_after),
        'nodes_removed': len(nq_before) - len(nq_after),
        'dormant_before': dormant_before,
        'dormant_after': dormant_after,
        'cycles': cycles,
        'timestamp': datetime.now().isoformat(),
    }

    logger.info(
        f"Consolidation #{cycles}: downscale={factor}, "
        f"synapses {synapses_before}→{synapses_after} ({results['synapses_pruned']} pruned), "
        f"nodes {len(nq_before)}→{len(nq_after)} ({results['nodes_removed']} removed), "
        f"dormant {dormant_before}→{dormant_after}"
    )

    return results


# ---------------------------------------------------------------------------
# Stats helper
# ---------------------------------------------------------------------------

def consolidation_stats(state: dict) -> dict:
    """Return summary statistics about consolidation history."""
    return {
        'cycles': state.get('consolidation_cycles', 0),
        'downscale_factor': CONFIG.get('consolidation_downscale_factor', DEFAULT_DOWNSCALE_FACTOR),
        'weight_floor': CONFIG.get('consolidation_prune_weight_floor', DEFAULT_PRUNE_WEIGHT_FLOOR),
        'dormant_persist_cycles': CONFIG.get(
            'consolidation_dormant_persist_cycles',
            DEFAULT_DORMANT_PERSIST_CYCLES,
        ),
    }