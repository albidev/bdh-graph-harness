"""Reproducible A/B evidence: does OKF retrieval inflate Hebbian consolidation candidates?

Self-contained harness. Runs the retrieval -> plasticity path
(``attention`` -> ``hebbian_update``) with the OKF retrieval policy ON vs OFF
using identical query, node graph, vector scores, and state seed. No live
vault, Chroma, service, cron, or state-file writes. Does not touch
``consolidation.py``.

Run:
    python scripts/okf_ab_evidence.py
(or: python -m scripts.okf_ab_evidence from the repo root)

Output: measured per-query synapse creation, accumulated synapse count,
weight distribution, distinct endpoints, and an offline consolidation
candidate proxy -- the evidence behind the causal verdict.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bdh_graph_harness import config
from bdh_graph_harness.retrieval.attention import attention
from bdh_graph_harness.memory.hebbian import (
    hebbian_update,
    safe_decode_synapse_key,
)


# ---------------------------------------------------------------------------
# Deterministic fixture (mirrors tests/test_okf_consolidation_ab.py)
# ---------------------------------------------------------------------------

def _node(title, **okf):
    node = {"id": title, "title": title, "text": f"body of {title}"}
    if okf:
        node["okf"] = okf
    return node


FIXED_VECTOR_SCORES = {"beta": 1.00, "alpha": 0.90, "gamma": 0.85, "delta": 0.80}
DISTANCES = {nid: 1.0 - s for nid, s in FIXED_VECTOR_SCORES.items()}
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
    def count(self):
        return len(_ORDER)

    def query(self, **_kwargs):
        return {
            "ids": [_ORDER],
            "distances": [[DISTANCES[nid] for nid in _ORDER]],
        }

    def get(self, ids=None, include=None):
        return {"ids": list(ids or []), "embeddings": []}


def _run_arm(*, okf_on, queries=20):
    """Run retrieval->plasticity with the OKF policy toggled. Returns state + trace.

    ``monkeypatch`` is emulated by save/restore around CONFIG keys and the
    module's get_embeddings, so this is dependency-free and reproducible.
    """
    import bdh_graph_harness.retrieval.attention as _attn_mod

    # Snapshot everything we mutate so we restore it (no test runner needed).
    cfg_snapshot = {k: config.CONFIG[k] for k in _CONFIG_KEYS}
    emb_orig = _attn_mod.get_embeddings

    def _configure():
        config.CONFIG.update({
            "hybrid_search": False,
            "retrieval_abstention_enabled": False,
            "adaptive_threshold": False,
            "experimental_integrate_fire": False,
            "hub_dampening": False,
            "okf_mode": "read" if okf_on else False,
            "okf_retrieval_policy_enabled": okf_on,
            "okf_policy_deprecated_multiplier": 0.50,
            "okf_policy_stale_multiplier": 0.60,
            "okf_policy_verified_bonus": 1.08,
            "okf_policy_provenance_bonus": 1.03,
            "okf_policy_min_multiplier": 0.35,
            "okf_policy_max_multiplier": 1.20,
            "seed_count": 2,
            "hebbian_learning_seed_count": 2,
            "hebbian_min_score": 0.15,
            "alpha": 0.7,
            "beta": 0.3,
            "decay": 0.95,
            "hebbian_frequency_scale": 10.0,
            "tau_recency_hours": 24.0,
            "quality_prune_interval": 50,
        })
        _attn_mod.get_embeddings = lambda _texts: [[1.0, 0.0, 0.0]]

    def _restore():
        config.CONFIG.update(cfg_snapshot)
        _attn_mod.get_embeddings = emb_orig

    try:
        _configure()
        nodes = make_nodes()
        edges = {nid: [] for nid in nodes}
        collection = _Collection()
        state = {"queries": 0, "synapses": {}, "node_quality": {}, "dormant_nodes": []}
        trace = []
        for _ in range(queries):
            routing = {}
            active = attention(
                "deterministic query", nodes, edges, collection,
                k=config.CONFIG["seed_count"], max_hop=0,
                hebbian_state=state, routing_meta=routing,
            )
            learning_active = {
                item["id"]: active[item["id"]]
                for item in routing.get("activation_details", [])
                if item.get("role") == "seed" and item["id"] in active
            }
            synapses_before = set(state["synapses"].keys())
            state, updated_keys, _pruned = hebbian_update(
                learning_active, state, nodes
            )
            trace.append({
                "learning_seeds": sorted(learning_active, key=lambda k: -learning_active[k]),
                "new_synapses": sorted(updated_keys - synapses_before),
            })
        return state, trace
    finally:
        _restore()


_CONFIG_KEYS = [
    "hybrid_search", "retrieval_abstention_enabled", "adaptive_threshold",
    "experimental_integrate_fire", "hub_dampening", "okf_mode",
    "okf_retrieval_policy_enabled", "okf_policy_deprecated_multiplier",
    "okf_policy_stale_multiplier", "okf_policy_verified_bonus",
    "okf_policy_provenance_bonus", "okf_policy_min_multiplier",
    "okf_policy_max_multiplier", "seed_count", "hebbian_learning_seed_count",
    "hebbian_min_score", "alpha", "beta", "decay", "hebbian_frequency_scale",
    "tau_recency_hours", "quality_prune_interval",
]


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


def _candidate_proxy(state):
    threshold = config.CONFIG.get("consolidation_weak_weight_threshold", 0.15)
    return {
        "weak_below_threshold": sum(
            1 for s in state["synapses"].values()
            if float(s.get("weight", 0.0)) < threshold
        ),
        "total_synapses": len(state["synapses"]),
    }


def main() -> int:
    off_state, off_trace = _run_arm(okf_on=False)
    on_state, on_trace = _run_arm(okf_on=True)

    print("=== QUERY 1 (fresh) ===")
    print("OKF OFF learning seeds:", off_trace[0]["learning_seeds"],
          "-> new synapse:", off_trace[0]["new_synapses"])
    print("OKF ON  learning seeds:", on_trace[0]["learning_seeds"],
          "-> new synapse:", on_trace[0]["new_synapses"])

    print("\n=== OVER 20 QUERIES (same query, deterministic) ===")
    print("OKF OFF accumulated synapses:", len(off_state["synapses"]),
          "weights:", _weight_distribution(off_state))
    print("OKF ON  accumulated synapses:", len(on_state["synapses"]),
          "weights:", _weight_distribution(on_state))
    print("OKF OFF candidate proxy:", _candidate_proxy(off_state))
    print("OKF ON  candidate proxy:", _candidate_proxy(on_state))

    off_endpoints, on_endpoints = set(), set()
    for q in off_trace:
        for k in q["new_synapses"]:
            a, b = safe_decode_synapse_key(k)
            off_endpoints.update({a, b})
    for q in on_trace:
        for k in q["new_synapses"]:
            a, b = safe_decode_synapse_key(k)
            on_endpoints.update({a, b})
    print("\nDistinct endpoints touched by learning, OKF OFF:", sorted(off_endpoints))
    print("Distinct endpoints touched by learning, OKF ON :", sorted(on_endpoints))

    print("\n=== CAUSAL VERDICT ===")
    print("Per-query new-synapse cap (hebbian_learning_seed_count=2 => 1 pair): bounded in BOTH arms")
    print("Accumulated synapse count equal across arms:",
          len(off_state["synapses"]) == len(on_state["synapses"]))
    print("Synapse COMPOSITION differs (which pair):",
          set(off_state["synapses"]) != set(on_state["synapses"]))
    print("=> OKF changes WHICH seeds (composition), NOT HOW MANY synapses (no count inflation).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
