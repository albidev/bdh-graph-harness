"""Evaluate utility of real learned non-static edges without title leakage."""
from datetime import datetime, timezone
from pathlib import Path
import json
import math
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.ablation import (
    CONFIG, BM25Index, _build_materialized_graph, compute_all_embeddings,
    config_overlay, load_config,
)
from bdh_graph_harness.graph.builder import _resolve_target
from bdh_graph_harness.memory.state_store import load_state
from bdh_graph_harness.retrieval.attention import attention, _resolve_hebbian_node_id

CONFIG_PATH = "/Users/albi/Projects/bdh-graph-harness/.worktrees/develop-live/bdh-config.local.yaml"
RESULTS = ROOT / "benchmarks" / "review"


def norm(value):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", value.lower())).strip()


def leakage_free_query(note, source_title, target_title):
    blocked = (norm(source_title), norm(target_title))
    text = note.get("text", "")
    text = re.sub(r"(?m)^---.*?^---\s*", "", text, flags=re.S)
    sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
    usable = []
    for sentence in sentences:
        sentence = re.sub(r"^#+\s*", "", sentence).strip()
        compact = norm(sentence)
        if len(compact) >= 45 and not any(title and title in compact for title in blocked):
            usable.append(sentence)
        if len(" ".join(usable)) >= 180:
            break
    return " ".join(usable)[:420]


def route(query, nodes, edges, collection, bm25, state, dynamic, tuning=None):
    metadata = {}
    settings = {"hebbian_dynamic_edges_enabled": dynamic, "hebbian_associative_context_enabled": dynamic}
    if tuning:
        settings.update(tuning)
    with config_overlay(settings):
        active = attention(query, nodes, edges, collection, bm25_index=bm25,
                           hebbian_state=state, routing_meta=metadata)
    return active, metadata


def static_pairs(edges, nodes):
    result = set()
    for source, outgoing in edges.items():
        for edge in outgoing:
            target = _resolve_target(edge.get("target", ""), nodes)
            if target:
                result.add(tuple(sorted((source, target))))
    return result


def rank(target, active):
    ordered = [node_id for node_id, _ in sorted(active.items(), key=lambda item: item[1], reverse=True)]
    try:
        return ordered.index(target) + 1
    except ValueError:
        return None


def metrics(ranks):
    count = len(ranks)
    return {
        "queries": count,
        "mrr": round(sum(1 / value if value else 0 for value in ranks) / count, 4),
        "recall_at_5": round(sum(value is not None and value <= 5 for value in ranks) / count, 4),
        "precision_at_5": round(sum(0.2 if value is not None and value <= 5 else 0 for value in ranks) / count, 4),
        "ndcg_at_5": round(sum(1 / math.log2(value + 1) if value is not None and value <= 5 else 0 for value in ranks) / count, 4),
    }


load_config(CONFIG_PATH)
with config_overlay({"chroma_path": str(ROOT / "benchmarks" / ".bdh-chroma-edge-utility")}):
    nodes, edges, _ = _build_materialized_graph()
    collection = compute_all_embeddings(nodes, CONFIG["vault_path"])
    bm25 = BM25Index(nodes)
    state = load_state(CONFIG["vault_path"])
    static = static_pairs(edges, nodes)
    candidates = []
    valid = set(nodes)
    for key, synapse in state.get("synapses", {}).items():
        try:
            left, right = key.split("|", 1)
        except ValueError:
            continue
        left = _resolve_hebbian_node_id(left, valid)
        right = _resolve_hebbian_node_id(right, valid)
        weight = float(synapse.get("weight", 0.0))
        if left and right and weight >= CONFIG.get("hebbian_dynamic_min_weight", 0.15):
            if tuple(sorted((left, right))) not in static:
                candidates.append((weight, left, right))
    candidates.sort(reverse=True)

    records = []
    for weight, source, target in candidates:
        query = leakage_free_query(nodes[source], nodes[source].get("title", source), nodes[target].get("title", target))
        if not query:
            continue
        static_active, static_details = route(query, nodes, edges, collection, bm25, state, False)
        if {item["id"]: item for item in static_details.get("activation_details", [])}.get(source, {}).get("role") != "seed":
            continue
        dynamic_active, dynamic_details = route(query, nodes, edges, collection, bm25, state, True)
        records.append({
            "source_id": source,
            "target_id": target,
            "weight": weight,
            "query": query,
            "source_seeded": True,
            "target_static_rank": rank(target, static_active),
            "target_dynamic_rank": rank(target, dynamic_active),
            "target_dynamic_provenance": {item["id"]: item for item in dynamic_details.get("activation_details", [])}.get(target, {}).get("matched_by"),
            "associative_context": dynamic_details.get("associative_context", []),
        })

static_ranks = [record["target_static_rank"] for record in records]
dynamic_ranks = [record["target_dynamic_rank"] for record in records]


def evaluate_tuning(tuning):
    evaluated = []
    for record in records:
        active, details = route(
            record["query"], nodes, edges, collection, bm25, state, True, tuning
        )
        target = record["target_id"]
        evaluated.append({
            "rank": rank(target, active),
            "provenance": details.get(target, {}).get("matched_by"),
        })
    ranks = [item["rank"] for item in evaluated]
    return {
        "tuning": tuning,
        **metrics(ranks),
        "dynamic_only_rate": round(sum(
            static_rank is None and item["rank"] is not None
            for static_rank, item in zip(static_ranks, evaluated)
        ) / len(evaluated), 4),
        "hebbian_provenance_rate": round(sum(
            item["provenance"] == "hebbian_edge" for item in evaluated
        ) / len(evaluated), 4),
    }


sweep = []
for gain in (1.0, 1.25, 1.5, 2.0):
    for decay in (0.5, 0.6, 0.7):
        sweep.append(evaluate_tuning({
            "hebbian_dynamic_gain": gain,
            "hebbian_dynamic_hop_decay": decay,
        }))
best = max(sweep, key=lambda item: (item["recall_at_5"], item["mrr"], item["ndcg_at_5"]))
for min_weight in (0.10, 0.15, 0.20):
    for top_n in (3, 5):
        sweep.append(evaluate_tuning({
            **best["tuning"],
            "hebbian_dynamic_min_weight": min_weight,
            "hebbian_dynamic_top_n": top_n,
        }))

result = {
    "created_at": datetime.now(timezone.utc).isoformat(),
    "protocol": "blind-associative-review-pack-v1",
    "state_synapses": len(state.get("synapses", {})),
    "eligible_non_static_edges": len(candidates),
    "source_seeded_queries": len(records),
    "static_only": metrics(static_ranks),
    "dynamic_hebbian": metrics(dynamic_ranks),
    "dynamic_only_target_retrieval_rate": round(sum(
        record["target_static_rank"] is None and record["target_dynamic_rank"] is not None
        for record in records) / len(records), 4) if records else 0.0,
    "hebbian_provenance_rate": round(sum(
        record["target_dynamic_provenance"] == "hebbian_edge" for record in records
        ) / len(records), 4) if records else 0.0,
    "sweep": sweep,
    "associative_lane": {
        "queries_with_context": sum(bool(r["associative_context"]) for r in records),
        "context_coverage": round(sum(bool(r["associative_context"]) for r in records) / len(records), 4) if records else 0.0,
    },
    "review_schema": {"labels": ["useful", "redundant", "noise", "misleading"], "instruction": "Judge each candidate only against the query; do not infer its lane."},
    "records": records,
}
RESULTS.mkdir(parents=True, exist_ok=True)
path = RESULTS / f"hebbian-edge-utility-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
print(path)
print(json.dumps({key: value for key, value in result.items() if key != "records"}, indent=2))
