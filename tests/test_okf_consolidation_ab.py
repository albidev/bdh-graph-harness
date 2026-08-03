"""A/B evidence: does OKF retrieval inflate Hebbian consolidation candidates?

Deterministic offline fixture. Runs the exact retrieval -> plasticity path
(``attention`` -> ``hebbian_update``) with the OKF retrieval policy ON vs OFF,
using the same query, same node graph, same vector scores, and the same state
seed. Never touches ``consolidation.py`` and never mutates a live vault/state
file.

Metrics recorded per arm and per query:
  - activated (seed) node ids and scores
  - learning seeds actually fed to ``hebbian_update`` (role == 'seed', >=
    ``hebbian_min_score``, capped at ``hebbian_learning_seed_count``)
  - new synapse keys created in that query
  - accumulated synapse count, distinct endpoints, weight distribution
  - offline consolidation-candidate proxy (stale-weak / weight-floor), i.e.
    what a consolidation pass would target -- computed read-only, mirroring
    ``consolidation._prune_candidate_reason`` without importing it.

The central causal claim under test: OKF is a ranking adjustment that can
change WHICH notes become learning seeds (composition), but it cannot inflate
the per-query Hebbian write budget, which is bounded by
``hebbian_learning_seed_count`` (default 2 => exactly one new pair per query).
Consolidation candidate count is therefore bounded at the creation boundary
and OKF is *contributory to composition, not causal of inflation*.
"""

from copy import deepcopy

from bdh_graph_harness import config
from bdh_graph_harness.retrieval.attention import attention
from bdh_graph_harness.memory.hebbian import hebbian_update, safe_decode_synapse_key


# ---------------------------------------------------------------------------
# Deterministic fixture
# ---------------------------------------------------------------------------

def _node(title, **okf):
    node = {"id": title, "title": title, "text": f"body of {title}", "okf": {}}
    node["okf"] = okf or None  # None => legacy/neutral (no okf block)
    if not okf:
        node.pop("okf")
    return node


# Vector-similarity ranking (higher raw score = more similar to the query).
# Beta scores highest on raw vectors; Alpha is second; Gamma/Delta lower.
# OKF metadata is chosen so that, when the policy is ON, the deprecated+stale
# Beta is demoted far below Alpha, changing the learning-seed composition.
FIXED_VECTOR_SCORES = {
    "beta": 1.00,   # sim = 1 - dist ; dist 0.00
    "alpha": 0.90,  # dist 0.10
    "gamma": 0.85,  # dist 0.15
    "delta": 0.80,  # dist 0.20
}

DISTANCES = {
    "beta": 1.0 - FIXED_VECTOR_SCORES["beta"],
    "alpha": 1.0 - FIXED_VECTOR_SCORES["alpha"],
    "gamma": 1.0 - FIXED_VECTOR_SCORES["gamma"],
    "delta": 1.0 - FIXED_VECTOR_SCORES["delta"],
}
_ORDER = ["beta", "alpha", "gamma", "delta"]


def make_nodes():
    return {
        "alpha": _node("alpha", status="stable", verified=True,
                       sources=[{"resource": "https://example.com/spec"}],
                       stale_after="2999-09-01"),
        "beta": _node("beta", status="deprecated", verified=False,
                      stale_after="2026-07-01"),
        "gamma": _node("gamma"),
        "delta": _node("delta"),
    }


class _Collection:
    """Stub Chroma collection returning the fixed vector ranking."""

    def count(self):
        return len(_ORDER)

    def query(self, **_kwargs):
        return {
            "ids": [_ORDER],
            "distances": [[DISTANCES[nid] for nid in _ORDER]],
        }

    def get(self, ids=None, include=None):
        return {"ids": list(ids or []), "embeddings": []}


def _run_arm(monkeypatch, *, okf_on, queries=8):
    """Run the retrieval->plasticity loop with a fixed state seed.

    Returns the accumulated state plus a per-query trace.
    """
    # Fixed config: pure vector, no hybrid/BM25, no adaptive threshold, no
    # abstention gate, no IaF -- isolates the OKF ranking effect.
    monkeypatch.setitem(config.CONFIG, "hybrid_search", False)
    monkeypatch.setitem(config.CONFIG, "retrieval_abstention_enabled", False)
    monkeypatch.setitem(config.CONFIG, "adaptive_threshold", False)
    monkeypatch.setitem(config.CONFIG, "experimental_integrate_fire", False)
    monkeypatch.setitem(config.CONFIG, "hub_dampening", False)
    monkeypatch.setitem(config.CONFIG, "okf_mode", "read" if okf_on else False)
    monkeypatch.setitem(config.CONFIG, "okf_retrieval_policy_enabled", okf_on)
    monkeypatch.setitem(config.CONFIG, "okf_policy_deprecated_multiplier", 0.50)
    monkeypatch.setitem(config.CONFIG, "okf_policy_stale_multiplier", 0.60)
    monkeypatch.setitem(config.CONFIG, "okf_policy_verified_bonus", 1.08)
    monkeypatch.setitem(config.CONFIG, "okf_policy_provenance_bonus", 1.03)
    monkeypatch.setitem(config.CONFIG, "okf_policy_min_multiplier", 0.35)
    monkeypatch.setitem(config.CONFIG, "okf_policy_max_multiplier", 1.20)
    monkeypatch.setitem(config.CONFIG, "seed_count", 2)
    monkeypatch.setitem(config.CONFIG, "hebbian_learning_seed_count", 2)
    monkeypatch.setitem(config.CONFIG, "hebbian_min_score", 0.15)
    monkeypatch.setitem(config.CONFIG, "alpha", 0.7)
    monkeypatch.setitem(config.CONFIG, "beta", 0.3)
    monkeypatch.setitem(config.CONFIG, "decay", 0.95)
    monkeypatch.setitem(config.CONFIG, "hebbian_frequency_scale", 10.0)
    monkeypatch.setitem(config.CONFIG, "tau_recency_hours", 24.0)
    monkeypatch.setitem(config.CONFIG, "quality_prune_interval", 50)
    import bdh_graph_harness.retrieval.attention as _attn_mod
    monkeypatch.setattr(
        _attn_mod, "get_embeddings", lambda _texts: [[1.0, 0.0, 0.0]]
    )

    nodes = make_nodes()
    edges = {nid: [] for nid in nodes}  # no static links -> no traversal
    collection = _Collection()
    state = {"queries": 0, "synapses": {}, "node_quality": {}, "dormant_nodes": []}

    trace = []
    for _ in range(queries):
        routing = {}
        active = attention(
            "deterministic query",
            nodes,
            edges,
            collection,
            k=config.CONFIG["seed_count"],
            max_hop=0,
            hebbian_state=state,
            routing_meta=routing,
        )
        # Mirror routes._run_attention_and_plasticity_unlocked: learn only from
        # role == 'seed' notes.
        learning_active = {
            item["id"]: active[item["id"]]
            for item in routing.get("activation_details", [])
            if item.get("role") == "seed" and item["id"] in active
        }
        synapses_before = set(state["synapses"].keys())
        state, updated_keys, _pruned = hebbian_update(
            learning_active, state, nodes
        )
        new_keys = updated_keys - synapses_before
        trace.append({
            "active_seeds": {nid: round(s, 4) for nid, s in active.items()},
            "learning_seeds": sorted(learning_active, key=lambda k: -learning_active[k]),
            "new_synapses": sorted(new_keys),
        })

    return state, trace


def _weight_distribution(state):
    weights = [s.get("weight", 0.0) for s in state["synapses"].values()]
    if not weights:
        return {"count": 0, "min": None, "max": None, "mean": None}
    return {
        "count": len(weights),
        "min": round(min(weights), 4),
        "max": round(max(weights), 4),
        "mean": round(sum(weights) / len(weights), 4),
    }


def _consolidation_candidate_proxy(state, now=None):
    """Offline mirror of consolidation._prune_candidate_reason (read-only).

    A synapse is a consolidation candidate if it is stale-weak (weight below
    ``consolidation_weak_weight_threshold``, frequency at most
    ``consolidation_weak_max_frequency``, and last coactivated older than
    ``consolidation_weak_min_age_hours``) OR weight below the prune floor.
    Since every synapse in this fixture was just created ``now``, none are
    stale yet -- so this returns the *structural* upper bound the candidate
    set could reach as the state ages, which is what matters for the
    "inflation" question.
    """
    threshold = config.CONFIG.get(
        "consolidation_weak_weight_threshold",
        config.CONFIG.get("consolidation_weak_weight_threshold", 0.15),
    )
    return {
        "weak_below_threshold": sum(
            1 for s in state["synapses"].values()
            if float(s.get("weight", 0.0)) < threshold
        ),
        "total_synapses": len(state["synapses"]),
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_okf_changes_learning_seed_composition_but_not_per_query_synapse_count(
    monkeypatch,
):
    off_state, off_trace = _run_arm(monkeypatch, okf_on=False)
    on_state, on_trace = _run_arm(monkeypatch, okf_on=True)

    # --- Composition differs: which notes become learning seeds. ---
    off_first_seeds = off_trace[0]["learning_seeds"]
    on_first_seeds = on_trace[0]["learning_seeds"]
    assert off_first_seeds != on_first_seeds, (
        "OKF should change which notes rank as learning seeds"
    )
    # With OKF off, raw-vector top-2 = beta, alpha. With OKF on, beta is
    # demoted (deprecated*stale) below alpha, so the top-2 swap.
    assert "beta" in off_first_seeds
    assert "beta" not in on_first_seeds

    # --- New-synapse creation is bounded by the learning budget: exactly ONE
    # new pair on the first query, zero on subsequent reinforces. OKF never
    # changes this bound. ---
    off_first_new = off_trace[0]["new_synapses"]
    on_first_new = on_trace[0]["new_synapses"]
    assert len(off_first_new) == len(on_first_new) == 1, (
        "hebbian_learning_seed_count=2 bounds new synapses to one per query "
        "regardless of OKF"
    )
    # The synapse that got created DIFFERS (composition), even though count is same.
    assert off_first_new != on_first_new


def test_okf_does_not_inflate_accumulated_synapse_count(monkeypatch):
    off_state, _ = _run_arm(monkeypatch, okf_on=False)
    on_state, _ = _run_arm(monkeypatch, okf_on=True)

    # Re-querying the same query reinforces the SAME single pair, so both arms
    # accumulate exactly one distinct synapse. No inflation.
    assert len(off_state["synapses"]) == 1
    assert len(on_state["synapses"]) == 1
    assert set(off_state["synapses"].keys()) != set(on_state["synapses"].keys())

    # Consolidation-candidate proxy is equal (both arms have a single synapse).
    assert _consolidation_candidate_proxy(off_state) == _consolidation_candidate_proxy(
        on_state
    )


def test_okf_affects_retrieval_ranking_only_not_candidate_creation_bound(monkeypatch):
    off_state, off_trace = _run_arm(monkeypatch, okf_on=False, queries=20)
    on_state, on_trace = _run_arm(monkeypatch, okf_on=True, queries=20)

    # Across 20 queries, the accumulated synapse count stays bounded at 1 in
    # both arms (same query always reinforces the same single pair). The
    # bound is OKF-independent -- no inflation in COUNT.
    assert len(off_state["synapses"]) == len(on_state["synapses"]) == 1

    # Both arms produce a single synapse carrying a healthy weight (above the
    # consolidation weak threshold), so neither arm creates candidate pressure
    # from this query loop. The exact weight VALUE differs between arms only
    # because OKF changes WHICH pair is reinforced (composition), not because
    # it changes how many synapses exist.
    off_dist = _weight_distribution(off_state)
    on_dist = _weight_distribution(on_state)
    assert off_dist["count"] == on_dist["count"] == 1
    weak_threshold = config.CONFIG.get(
        "consolidation_weak_weight_threshold", 0.15
    )
    assert off_dist["min"] > weak_threshold
    assert on_dist["min"] > weak_threshold
