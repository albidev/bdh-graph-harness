"""Memory consolidation — sleep-cycle graph maintenance.

Biological brains consolidate memories during sleep via three mechanisms:

1. **Synaptic homeostasis** — global downscaling of all synaptic weights,
   preventing saturation. Strong (frequently-used) synapses survive; weak
   ones fall below pruning threshold naturally.

2. **Guarded structural pruning** — candidates are confirmed across multiple
   cycles, structural/bridge links and recent coactivations are protected,
   and anomalous prune ratios are quarantined instead of committed.

3. **Quality re-evaluation** — after a safe prune, node quality scores are
   recomputed from the surviving synapse set and dormancy status is updated.

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
from bdh_graph_harness.memory.hebbian import encode_synapse_key, safe_decode_synapse_key
from bdh_graph_harness.memory.quality import compute_all_qualities


# ---------------------------------------------------------------------------
# Defaults (override via config)
# ---------------------------------------------------------------------------
DEFAULT_DOWNSCALE_FACTOR = 0.90       # multiply all weights by this
DEFAULT_PRUNE_WEIGHT_FLOOR = 0.02     # delete synapses below this weight
DEFAULT_WEAK_WEIGHT_THRESHOLD = 0.15
DEFAULT_WEAK_MAX_FREQUENCY = 1.0
DEFAULT_WEAK_MIN_AGE_HOURS = 48
DEFAULT_PRUNE_CONFIRM_CYCLES = 2
DEFAULT_MAX_PRUNE_RATIO = 0.35
DEFAULT_MAX_PRUNE_PER_CYCLE = 0.15
DEFAULT_PROTECT_BACKBONE = True
DEFAULT_PROTECT_RECENT_HOURS = 72
DEFAULT_DORMANT_PERSIST_CYCLES = 3    # remove nodes dormant for N+ cycles
DEFAULT_PRUNE_DORMANT_NODES = True    # actually remove stale dormant nodes
# Clean-room shadow state path may explicitly opt into a non-1.0 downscale factor.
# When absent, the clean-room path uses an effective factor of 1.0 (no extra
# persisted downscale) regardless of the inherited global default.
DEFAULT_CLEAN_ROOM_DOWNSCALE_FACTOR = None


def _clean_room_basename(state_file: str | None) -> bool:
    """Return whether the state file resolves to the clean-room shadow path."""
    if not state_file:
        return False
    return 'primary-seeds-v2' in str(state_file).replace('\\', '/')


def state_mode(state_file: str | None) -> str:
    """Classify the state-file basename into a known mode string."""
    if not state_file:
        return 'legacy_active'
    basename = str(state_file).replace('\\', '/').split('/')[-1]
    if basename == '.bdh-state.json':
        return 'legacy_active'
    if 'primary-seeds-v2' in basename:
        return 'clean_room_shadow'
    if 'legacy-curated' in basename:
        return 'curated_experimental'
    return 'custom'



def effective_downscale_factor(config: dict | None = None) -> float:
    """Return the downscale factor to apply for a consolidation cycle.

    The clean-room shadow state path (``.bdh-state-primary-seeds-v2.json``)
    inherits ``consolidation_downscale_factor`` from the global default when the
    live config omits it. That 0.90 per-cycle downscale would compound on top of
    the online Hebbian decay. To prevent this double decay, the clean-room path
    uses an effective factor of 1.0 (no extra persisted downscale) unless the
    profile explicitly opts into a different factor via
    ``consolidation_clean_room_downscale_factor``.

    Parameters
    ----------
    config : dict, optional
        The effective config dict (per-profile merged settings). Defaults to the
        global ``CONFIG``.

    Returns
    -------
    float
        The downscale factor to apply.
    """
    cfg = config if config is not None else CONFIG
    state_file = cfg.get('hebbian_state_file')
    if not _clean_room_basename(state_file):
        return float(cfg.get('consolidation_downscale_factor', DEFAULT_DOWNSCALE_FACTOR))
    seam = cfg.get('consolidation_clean_room_downscale_factor', DEFAULT_CLEAN_ROOM_DOWNSCALE_FACTOR)
    if seam is not None:
        return float(seam)
    return 1.0



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


def _normal_synapse_key(source: str, target: str) -> str:
    """Canonicalize an undirected synapse key for protection lookups."""
    return encode_synapse_key(source, target)


def _structural_backbone_keys(edges: dict | None) -> set[str]:
    """Return structural links and bridges represented by the vault graph."""
    if not edges:
        return set()

    structural: set[str] = set()
    adjacency: dict[str, set[str]] = {}
    for source, links in edges.items():
        for link in links or []:
            target = link.get('target') if isinstance(link, dict) else link
            if not target or target == source:
                continue
            key = _normal_synapse_key(source, target)
            structural.add(key)
            adjacency.setdefault(str(source), set()).add(str(target))
            adjacency.setdefault(str(target), set()).add(str(source))

    # Tarjan bridge detection preserves single points of graph connectivity.
    discovery: dict[str, int] = {}
    low: dict[str, int] = {}
    bridges: set[str] = set()
    clock = 0

    def visit(node: str, parent: str | None = None) -> None:
        nonlocal clock
        clock += 1
        discovery[node] = low[node] = clock
        for neighbour in adjacency.get(node, set()):
            if neighbour == parent:
                continue
            if neighbour not in discovery:
                visit(neighbour, node)
                low[node] = min(low[node], low[neighbour])
                if low[neighbour] > discovery[node]:
                    bridges.add(_normal_synapse_key(node, neighbour))
            else:
                low[node] = min(low[node], discovery[neighbour])

    for node in adjacency:
        if node not in discovery:
            visit(node)
    return structural | bridges


def protected_synapses(
    state: dict,
    edges: dict | None = None,
    *,
    now: datetime | None = None,
    config: dict | None = None,
) -> dict[str, str]:
    """Return synapses that must not be pruned in this consolidation pass."""
    cfg = config or CONFIG
    protected: dict[str, str] = {}
    backbone = set()
    if cfg.get('consolidation_protect_backbone', DEFAULT_PROTECT_BACKBONE):
        backbone = _structural_backbone_keys(edges)
        for key in state.get('synapses', {}):
            endpoints = safe_decode_synapse_key(key)
            if endpoints is not None and _normal_synapse_key(*endpoints) in backbone:
                protected[key] = 'structural_or_bridge'

    recent_hours = float(cfg.get(
        'consolidation_protect_recent_hours', DEFAULT_PROTECT_RECENT_HOURS,
    ))
    effective_now = now or datetime.now()
    for key, synapse in state.get('synapses', {}).items():
        timestamp = _coactivation_timestamp(synapse.get('last_coactivated'))
        if timestamp is None:
            continue
        compare_now = effective_now
        if timestamp.tzinfo is not None and compare_now.tzinfo is None:
            compare_now = compare_now.replace(tzinfo=timestamp.tzinfo)
        elif timestamp.tzinfo is None and compare_now.tzinfo is not None:
            timestamp = timestamp.replace(tzinfo=compare_now.tzinfo)
        if timestamp >= compare_now - timedelta(hours=recent_hours):
            protected.setdefault(key, 'recently_coactivated')
    return protected


def _prune_candidate_reason(
    synapse: dict,
    *,
    floor: float,
    now: datetime,
    config: dict,
) -> str | None:
    """Return the reason a synapse is eligible for hysteresis tracking."""
    try:
        if float(synapse.get('weight', 0.0)) < float(floor):
            return 'weight_floor'
    except (TypeError, ValueError):
        return None
    if is_stale_weak_synapse(synapse, now=now, config=config):
        return 'stale_weak'
    return None


def _scan_prune_candidates(
    state: dict,
    *,
    now: datetime,
    config: dict,
    protected: dict[str, str],
    include_floor: bool = True,
) -> tuple[list[str], int, dict[str, str]]:
    """Mark candidates and return confirmed candidates, pending count, reasons."""
    floor = config.get('consolidation_prune_weight_floor', DEFAULT_PRUNE_WEIGHT_FLOOR)
    confirm_cycles = max(1, int(config.get(
        'consolidation_prune_confirm_cycles', DEFAULT_PRUNE_CONFIRM_CYCLES,
    )))
    candidates: list[str] = []
    reasons: dict[str, str] = {}
    pending = 0
    for key, synapse in state.get('synapses', {}).items():
        if key in protected:
            synapse.pop('consolidation_candidate_cycles', None)
            continue
        reason = _prune_candidate_reason(
            synapse, floor=float(floor), now=now, config=config,
        ) if include_floor else (
            'stale_weak'
            if is_stale_weak_synapse(synapse, now=now, config=config)
            else None
        )
        if reason is None:
            synapse.pop('consolidation_candidate_cycles', None)
            continue
        reasons[key] = reason
        cycles = int(synapse.get('consolidation_candidate_cycles', 0)) + 1
        synapse['consolidation_candidate_cycles'] = cycles
        if cycles >= confirm_cycles:
            candidates.append(key)
        else:
            pending += 1
    return candidates, pending, reasons


def _apply_prune_candidates(state: dict, keys: list[str]) -> int:
    """Delete selected candidates and return the number actually removed."""
    for key in keys:
        state.get('synapses', {}).pop(key, None)
    return len(keys)


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
    protected_keys: set[str] | None = None,
    max_prune: int | None = None,
) -> int:
    """Delete confirmed stale weak synapses and return the count.

    A candidate must remain stale for ``consolidation_prune_confirm_cycles``
    calls. This hysteresis prevents a link from oscillating around the
    threshold. ``protected_keys`` and ``max_prune`` are used by the full
    consolidation safety gate.
    """
    cfg = config or CONFIG
    effective_now = now or datetime.now()
    protected = protected_keys or set()
    candidates, _pending, _reasons = _scan_prune_candidates(
        state,
        now=effective_now,
        config=cfg,
        protected={key: 'protected' for key in protected},
        include_floor=False,
    )
    if max_prune is not None:
        candidates = candidates[:max(0, int(max_prune))]
    return _apply_prune_candidates(state, candidates)


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
        pair = safe_decode_synapse_key(key)
        if pair is None:
            continue
        a, b = pair
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
    dry_run: bool = False,
) -> dict:
    """Run a guarded consolidation cycle.

    The cycle is evaluated in-place so existing callers retain object identity,
    but an anomalous candidate set is rolled back atomically. ``dry_run`` runs
    the complete analysis against the state and restores it before returning.
    """
    from bdh_graph_harness.memory.quality import prune_dormant

    cfg = config or CONFIG
    snapshot = deepcopy(state)
    synapses_before = len(state.get('synapses', {}))
    nq_before = deepcopy(state.get('node_quality', {}))
    dormant_before = len(state.get('dormant_nodes', []))
    cycles = state.get('consolidation_cycles', 0) + 1
    configured_factor = float(cfg.get(
        'consolidation_downscale_factor', DEFAULT_DOWNSCALE_FACTOR,
    ))
    floor = cfg.get('consolidation_prune_weight_floor', DEFAULT_PRUNE_WEIGHT_FLOOR)
    effective_now = datetime.now()
    phase_order = [
        'snapshot', 'downscale', 'protection', 'candidate_scan',
        'hysteresis', 'safety_gate', 'apply_prune', 'quality', 'dormant', 'phantom',
    ]

    state['consolidation_cycles'] = cycles
    # P1: the clean-room shadow path neutralises the inherited downscale factor
    # (effective 1.0) so consolidation does not compound on top of online decay.
    factor = effective_downscale_factor(cfg)
    state = synaptic_downscaling(state, factor)
    protected = protected_synapses(state, edges, now=effective_now, config=cfg)
    candidates, pending_confirmation, reasons = _scan_prune_candidates(
        state,
        now=effective_now,
        config=cfg,
        protected=protected,
    )
    candidates.sort(key=lambda key: float(state['synapses'][key].get('weight', 0.0)))
    candidate_count = len(candidates)
    candidate_ratio = candidate_count / synapses_before if synapses_before else 0.0
    max_ratio = float(cfg.get('consolidation_max_prune_ratio', DEFAULT_MAX_PRUNE_RATIO))
    max_per_cycle = float(cfg.get(
        'consolidation_max_prune_per_cycle', DEFAULT_MAX_PRUNE_PER_CYCLE,
    ))
    state_target = cfg.get('hebbian_state_file', '.bdh-state.json')
    result_base = {
        # configured_downscale_factor = inherited/configured value; downscale_factor
        # is the effective factor actually applied (compatibility alias retained).
        'configured_downscale_factor': configured_factor,
        'downscale_factor': factor,
        'weight_floor': floor,
        'synapses_before': synapses_before,
        'nodes_before': len(nq_before),
        'dormant_before': dormant_before,
        'cycles': cycles,
        # Raw candidate set (before any budget cap) — observability of a spike.
        'candidate_synapses': candidate_count,
        'candidate_prune_ratio': round(candidate_ratio, 6),
        'candidate_ratio_anomaly': bool(candidate_ratio > max_ratio),
        # Planned mutation = raw candidates after the per-cycle budget cap.
        'planned_prune_count': 0,
        'planned_prune_ratio': 0.0,
        'candidate_reasons': reasons,
        'protected_synapses': len(protected),
        'pending_confirmation': pending_confirmation,
        'phase_order': phase_order,
        'state_target': state_target,
        'state_mode': state_mode(state_target),
        'dry_run': dry_run,
        'aborted': False,
        'abort_reason': None,
        'would_commit': True,
        'decision': 'dry_run' if dry_run else 'commit',
    }

    # P0: rank raw candidates, cap planned deletion at the per-cycle budget,
    # then apply the hard safety gate to the *planned* mutation. A raw candidate
    # spike is surfaced as an anomaly warning, not an abort — the budget bounds
    # what actually gets deleted.
    prune_budget = max(1, int(synapses_before * max_per_cycle)) if candidate_count else 0
    selected = candidates[:prune_budget]
    planned_count = len(selected)
    planned_ratio = planned_count / synapses_before if synapses_before else 0.0
    planned = {**result_base, 'planned_prune_count': planned_count,
               'planned_prune_ratio': round(planned_ratio, 6)}

    # Kill switch: quarantine the whole cycle when the *planned* (budget-capped)
    # mutation still exceeds the safety ratio. This cannot be bounded further by
    # the budget, so committing it would be a suspicious mass prune. The abort
    # reason keeps the historical value for API/contract compatibility.
    if planned_ratio > max_ratio:
        state.clear()
        state.update(snapshot)
        return {
            **planned,
            'synapses_after': synapses_before,
            'synapses_pruned': 0,
            'stale_weak_pruned': 0,
            'nodes_after': len(nq_before),
            'nodes_removed': 0,
            'dormant_after': dormant_before,
            'prune_ratio': 0.0,
            'capped': False,
            'aborted': True,
            'abort_reason': 'candidate_prune_ratio_exceeded',
            'would_commit': False,
            'decision': 'abort',
            'timestamp': datetime.now().isoformat(),
        }

    pruned_count = _apply_prune_candidates(state, selected)
    stale_weak_pruned = sum(reasons.get(key) == 'stale_weak' for key in selected)
    capped = planned_count < candidate_count

    old_dormant_cycles = {
        nid: q.get('dormant_cycles', 0)
        for nid, q in nq_before.items()
    }
    state = prune_dormant(state, nodes)
    for nid, quality in state.get('node_quality', {}).items():
        if nid in old_dormant_cycles:
            quality['dormant_cycles'] = old_dormant_cycles[nid]

    if cfg.get('consolidation_prune_dormant_nodes', DEFAULT_PRUNE_DORMANT_NODES):
        persist = cfg.get(
            'consolidation_dormant_persist_cycles',
            DEFAULT_DORMANT_PERSIST_CYCLES,
        )
        state = prune_stale_dormant(state, nodes, persist)

    # Phantom links are a post-consolidation enrichment step. Never touch the
    # external collection during dry-run, and never enrich an aborted cycle.
    if not dry_run and cfg.get('phantom_links_enabled', True):
        from bdh_graph_harness.memory.phantom import update_phantom_links
        vault_root = cfg.get('vault_path', '')
        if vault_root and edges:
            state = update_phantom_links(
                state, nodes, edges or {}, vault_root,
                config=cfg, collection=collection,
            )

    dormant_after = len(state.get('dormant_nodes', []))
    synapses_after = len(state.get('synapses', {}))
    nq_after = state.get('node_quality', {})
    results = {
        **planned,
        'synapses_after': synapses_after,
        'synapses_pruned': pruned_count,
        'stale_weak_pruned': stale_weak_pruned,
        'nodes_after': len(nq_after),
        'nodes_removed': len(nq_before) - len(nq_after),
        'dormant_after': dormant_after,
        'prune_ratio': round(pruned_count / synapses_before, 6) if synapses_before else 0.0,
        'capped': capped,
        'timestamp': datetime.now().isoformat(),
    }

    if dry_run:
        state.clear()
        state.update(snapshot)

    logger.info(
        f"Consolidation #{cycles}: downscale={factor}, "
        f"synapses {synapses_before}→{synapses_after} ({pruned_count} pruned), "
        f"nodes {len(nq_before)}→{len(nq_after)} ({results['nodes_removed']} removed), "
        f"dormant {dormant_before}→{dormant_after}, "
        f"candidate_ratio={candidate_ratio:.1%}, capped={capped}, dry_run={dry_run}"
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
        'prune_confirm_cycles': CONFIG.get(
            'consolidation_prune_confirm_cycles', DEFAULT_PRUNE_CONFIRM_CYCLES,
        ),
        'max_prune_ratio': CONFIG.get(
            'consolidation_max_prune_ratio', DEFAULT_MAX_PRUNE_RATIO,
        ),
        'max_prune_per_cycle': CONFIG.get(
            'consolidation_max_prune_per_cycle', DEFAULT_MAX_PRUNE_PER_CYCLE,
        ),
        'protect_backbone': CONFIG.get(
            'consolidation_protect_backbone', DEFAULT_PROTECT_BACKBONE,
        ),
        'protect_recent_hours': CONFIG.get(
            'consolidation_protect_recent_hours', DEFAULT_PROTECT_RECENT_HOURS,
        ),
    }