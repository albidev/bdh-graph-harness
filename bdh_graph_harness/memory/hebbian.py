"""Hebbian update — reinforce links between co-activated notes.

Key design choices
------------------
* Recency is computed from actual Δt, not a constant.
* Weight updates are incremental: decay persists between touches.
* Frequency contribution uses a non-saturating compression so popular
  synapses remain distinguishable.
* Co-activation strength weights the frequency increment by the product of
  the two nodes' activation scores.
* Topology is seeds → activated nodes (directed), not a clique.
* Pruning is deferred to the consolidation cycle; here synapses are only
  decayed and never deleted.

Forgetting
----------
``hebbian_decay_mode`` selects how an unused synapse fades:

* ``query`` (legacy default): every learning query multiplies the weight of
  each untouched synapse by ``decay``. Forgetting then tracks traffic, not
  time: a busy day erases what a quiet month would have kept, and it stacks
  with the recency term and with the nightly downscale.
* ``time``: the weight is a function of the synapse's frequency and of the
  time since its last co-activation, nothing else. Frequency halves every
  ``hebbian_frequency_half_life_days``; weights are recomputed on each
  learning update and each consolidation cycle, and the nightly downscale is
  not applied on top.
"""
import base64
import json
from datetime import datetime, timedelta, timezone
from math import exp, log1p

from bdh_graph_harness.config import CONFIG
from bdh_graph_harness.memory.quality import prune_dormant, try_reactivate
from bdh_graph_harness.memory.source_policy import get_frequency_increment, get_source_policy


# ---------------------------------------------------------------------------
# Synapse key codec — v2 base64url JSON
# ---------------------------------------------------------------------------

_V2_PREFIX = "v2:"


def encode_synapse_key(a: str, b: str) -> str:
    """Encode two note IDs into a canonical v2 synapse key.

    Format: ``v2:`` + URL-safe base64 (no padding) of the compact JSON
    array ``[canonical_a, canonical_b]`` with lexicographic endpoint
    ordering.
    """
    a, b = (a, b) if a < b else (b, a)
    payload = json.dumps([a, b], separators=(",", ":"), ensure_ascii=True)
    b64 = base64.urlsafe_b64encode(payload.encode()).rstrip(b"=").decode()
    return f"{_V2_PREFIX}{b64}"


def decode_synapse_key(key: str) -> tuple[str, str]:
    """Decode a synapse key into its two endpoint note IDs.

    Recognises:
    * ``v2:`` + base64url JSON array — the canonical format.
    * ``a|b`` (exactly one pipe) — the legacy format.
    * More than one pipe — ambiguous; raises ``ValueError``.
    """
    if key.startswith(_V2_PREFIX):
        b64_part = key[len(_V2_PREFIX):]
        # Re-add padding for base64 decoding.
        padding = 4 - len(b64_part) % 4
        if padding != 4:
            b64_part += "=" * padding
        raw = base64.urlsafe_b64decode(b64_part)
        pair = json.loads(raw)
        if (
            not isinstance(pair, list)
            or len(pair) != 2
            or not all(isinstance(endpoint, str) for endpoint in pair)
        ):
            raise ValueError(f"v2 key must decode to a 2-element string array, got {pair!r}")
        return pair[0], pair[1]

    # Legacy: pipe-delimited
    parts = key.split("|")
    if len(parts) == 2:
        return parts[0], parts[1]
    raise ValueError(
        f"ambiguous synapse key with {len(parts) - 1} pipe(s): {key!r}"
    )


def safe_decode_synapse_key(key: str) -> tuple[str, str] | None:
    """Decode a synapse key, returning ``None`` on malformed input."""
    try:
        return decode_synapse_key(key)
    except (ValueError, KeyError):
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ordered_key(a: str, b: str) -> str:
    """Canonical ordering for undirected synapse keys."""
    return encode_synapse_key(a, b)


def _recency_factor(last_coactivated: str | None, tau_hours: float) -> float:
    """Exponential recency: exp(-Δt / τ) with Δt in hours."""
    if not last_coactivated:
        return 0.0
    try:
        last = datetime.fromisoformat(last_coactivated)
        if last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)
        delta_hours = (datetime.now(timezone.utc) - last).total_seconds() / 3600.0
        return exp(-max(delta_hours, 0.0) / max(tau_hours, 1e-6))
    except (ValueError, TypeError):
        return 0.0


def _frequency_component(freq: float, freq_scale: float) -> float:
    """Non-saturating frequency compression in [0, 1]."""
    denom = log1p(freq_scale)
    return log1p(freq) / denom if denom > 0 else 0.0


def _compute_weight(
    syn: dict,
    alpha: float,
    beta: float,
    freq_scale: float,
    tau_recency_hours: float,
) -> float:
    """Recompute synapse weight from stored state."""
    freq = syn.get('frequency', 0.0)
    recency = _recency_factor(syn.get('last_coactivated'), tau_recency_hours)
    return alpha * _frequency_component(freq, freq_scale) + beta * recency


# ---------------------------------------------------------------------------
# Time-based forgetting
# ---------------------------------------------------------------------------

DEFAULT_FREQUENCY_HALF_LIFE_DAYS = 30.0


def is_time_decay(config: dict | None = None) -> bool:
    """Return whether forgetting is driven by elapsed time instead of query count."""
    cfg = config if config is not None else CONFIG
    return cfg.get('hebbian_decay_mode', 'query') == 'time'


def _elapsed_hours(timestamp: str | None, now: datetime) -> float | None:
    """Hours between an ISO timestamp and ``now``; ``None`` when unparsable."""
    if not timestamp:
        return None
    try:
        moment = datetime.fromisoformat(str(timestamp).replace('Z', '+00:00'))
    except (TypeError, ValueError):
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    return max((now - moment).total_seconds() / 3600.0, 0.0)


def _decayed_frequency(syn: dict, now: datetime, half_life_days: float) -> float:
    """Frequency as of ``now``: it halves every ``half_life_days`` without use."""
    frequency = float(syn.get('frequency', 0.0))
    hours = _elapsed_hours(syn.get('last_coactivated') or syn.get('created'), now)
    if hours is None or half_life_days <= 0:
        return frequency
    return frequency * 0.5 ** (hours / 24.0 / half_life_days)


def current_weight(syn: dict, *, now: datetime | None = None, config: dict | None = None) -> float | None:
    """Weight a synapse has at ``now`` under time-based forgetting.

    Returns ``None`` for a synapse with no usable timestamp: its stored weight
    is the only evidence there is, so it is left as it stands.
    """
    cfg = config if config is not None else CONFIG
    now = now or datetime.now(timezone.utc)
    hours = _elapsed_hours(syn.get('last_coactivated') or syn.get('created'), now)
    if hours is None:
        return None
    half_life = float(cfg.get('hebbian_frequency_half_life_days', DEFAULT_FREQUENCY_HALF_LIFE_DAYS))
    frequency = _decayed_frequency(syn, now, half_life)
    recency = (
        exp(-hours / max(float(cfg.get('tau_recency_hours', 24.0)), 1e-6))
        if syn.get('last_coactivated') else 0.0
    )
    return (
        cfg.get('alpha', 0.7) * _frequency_component(frequency, cfg.get('hebbian_frequency_scale', 10.0))
        + cfg.get('beta', 0.3) * recency
    )


def refresh_weights(state: dict, *, now: datetime | None = None, config: dict | None = None,
                    skip=()) -> int:
    """Recompute every synapse weight from its frequency and age.

    Called by the learning update and by consolidation in ``time`` mode, so a
    stored weight is never more than one cycle stale. Returns the number of
    synapses refreshed.
    """
    now = now or datetime.now(timezone.utc)
    refreshed = 0
    for key, syn in state.get('synapses', {}).items():
        if key in skip:
            continue
        weight = current_weight(syn, now=now, config=config)
        if weight is None:
            continue
        syn['weight'] = round(weight, 6)
        refreshed += 1
    return refreshed


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def hebbian_update(active_notes, state, nodes=None, source=None, traversed_edges=None):
    """Reinforce links from strong seeds to activated notes.

    Args:
        active_notes: dict {note_id: activation_score} in descending order.
        state: Hebbian state dict with 'synapses', 'queries', 'dormant_nodes'.
        nodes: optional full node map; used only by the periodic quality prune.
        source: 'assistant_response' or 'nightly_semantic_consolidation' dampen
                the update; anything else uses full strength.
        traversed_edges: optional ``(seed_id, neighbour_id, strength)`` triples
                for the neighbours this query reached from a seed. Each one
                reinforces the edge it travelled, so the links that keep
                being used gain weight and the ones that are not fade.

    Returns:
        (state, updated_keys, pruned_count)
    """
    min_score = CONFIG.get('hebbian_min_score', 0.15)

    # Dampening for derived/secondary signals — delegated to the explicit
    # source policy registry (issue #16).  Unknown sources now raise
    # ValueError instead of silently falling through to full strength.
    freq_increment = get_frequency_increment(source)
    source_policy = get_source_policy(source)

    # Only notes above threshold participate.
    strong = {nid: s for nid, s in active_notes.items() if s >= min_score}
    if not strong:
        return state, set(), 0

    # Learning uses its own conservative seed budget; retrieval can overfetch safely.
    seed_count = CONFIG.get('hebbian_learning_seed_count', CONFIG.get('seed_count', 5))
    seeds = [nid for nid, _ in sorted(strong.items(), key=lambda x: -x[1])[:seed_count]]

    alpha = CONFIG.get('alpha', 0.7)
    beta = CONFIG.get('beta', 0.3)
    freq_scale = CONFIG.get('hebbian_frequency_scale', 10.0)
    tau_recency = CONFIG.get('tau_recency_hours', 24.0)

    now_dt = datetime.now(timezone.utc)
    now = now_dt.isoformat()
    time_decay = is_time_decay()
    half_life = float(CONFIG.get('hebbian_frequency_half_life_days', DEFAULT_FREQUENCY_HALF_LIFE_DAYS))
    updated_keys = set()

    def reinforce(key, increment):
        updated_keys.add(key)
        if key not in state['synapses']:
            syn_entry = {
                'weight': 0.0,
                'frequency': 0.0,
                'last_coactivated': None,
                'created': now,
            }
            # Provenance: store the source label so downstream
            # consumers can trace where a synapse was born.
            if source_policy is not None:
                syn_entry['source'] = source_policy.provenance_label
            state['synapses'][key] = syn_entry

        syn = state['synapses'][key]
        if time_decay:
            # Bring the stored frequency forward to now before adding to it.
            syn['frequency'] = _decayed_frequency(syn, now_dt, half_life)
        syn['frequency'] += increment
        syn['last_coactivated'] = now
        # Incremental: recompute from stored state so decay persists.
        syn['weight'] = _compute_weight(syn, alpha, beta, freq_scale, tau_recency)

    # Directed seed → later seed edges. The key is stored canonical/undirected,
    # but we create each pair exactly once using the score product of the two ends.
    for i, seed_id in enumerate(seeds):
        seed_score = strong[seed_id]
        for target_id in seeds[i + 1:]:
            target_score = strong[target_id]
            # Increment weighted by co-activation strength.
            reinforce(
                _ordered_key(seed_id, target_id),
                freq_increment * seed_score * target_score,
            )

    # Edges actually travelled from a seed to an admitted neighbour.
    for source_id, target_id, strength in traversed_edges or ():
        if source_id == target_id or not strength or strength <= 0:
            continue
        reinforce(_ordered_key(source_id, target_id), freq_increment * float(strength))

    if time_decay:
        # Untouched synapses fade with elapsed time, not with the number of
        # queries the vault happened to receive.
        refresh_weights(state, now=now_dt, skip=updated_keys)
    else:
        # Decay synapses that were not touched by this query.
        decay_rate = CONFIG.get('decay', 0.95)
        touched = updated_keys
        for key, syn in list(state['synapses'].items()):
            if key in touched:
                continue
            syn['weight'] *= decay_rate
            # Floor instead of delete; consolidation owns pruning.
            if syn['weight'] < 0:
                syn['weight'] = 0.0

    # Try to re-activate dormant nodes that received strong activation.
    reactivated = 0
    for nid, score in active_notes.items():
        if score >= min_score and try_reactivate(nid, score, state):
            reactivated += 1

    # Periodic quality pruning.
    pruned_count = 0
    state['queries'] += 1
    prune_interval = CONFIG.get('quality_prune_interval', 50)
    if nodes and state['queries'] % prune_interval == 0:
        old_dormant = set(state.get('dormant_nodes', []))
        state = prune_dormant(state, nodes)
        new_dormant = set(state.get('dormant_nodes', []))
        pruned_count = len(new_dormant - old_dormant)

    return state, updated_keys, pruned_count
