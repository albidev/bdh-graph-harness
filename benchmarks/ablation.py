"""Parametric ablation runner for BDH Graph Harness.

Runs the golden-set queries under a temporary config overlay and a fresh,
temporary Hebbian state. Does not mutate the on-disk config or the persisted
state file.

Example:
    from benchmarks.ablation import run_eval
    metrics = run_eval({"max_hop": 1}, load_dataset())
"""
from __future__ import annotations

import json
import math
import os
import re
import tempfile
import time
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bdh_graph_harness.config import CONFIG, config_overlay, load_config
from bdh_graph_harness.graph.builder import build_graph
from bdh_graph_harness.memory import load_state, save_state
from bdh_graph_harness.memory.hebbian import hebbian_update
from bdh_graph_harness.retrieval.attention import attention
from bdh_graph_harness.retrieval.bm25 import BM25Index
from bdh_graph_harness.retrieval.chroma_store import compute_all_embeddings

from benchmarks.metrics import compute_all_metrics, aggregate_metrics


@dataclass
class Metrics:
    """Container for ablation results."""

    mrr: float = 0.0
    recall_at_5: float = 0.0
    precision_at_5: float = 0.0
    ndcg_at_5: float = 0.0
    mean_latency_ms: float = 0.0
    per_query: list[dict] = field(default_factory=list)
    config_hash: str | None = None
    hop_histogram: dict[int, int] = field(default_factory=dict)
    synapse_counts: list[int] = field(default_factory=list)


def _config_hash(overrides: dict) -> str:
    """Short deterministic identifier for a config overlay."""
    import hashlib

    canonical = json.dumps(overrides, sort_keys=True, default=str)
    return hashlib.md5(canonical.encode()).hexdigest()[:8]


def _fresh_state() -> dict:
    """Return an empty Hebbian state backed by a temp file."""
    return {
        "synapses": {},
        "queries": 0,
        "dormant_nodes": [],
        "state_file": None,
    }


def _load_golden_set(path: Path | str | None = None) -> list[dict]:
    """Load golden-set queries from YAML or fall back to dataset.py."""
    if path is None:
        path = Path(__file__).resolve().parent / "golden_set.yaml"
    else:
        path = Path(path)

    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data.get("queries", [])

    # Fallback to the older inline dataset while the YAML is being reviewed.
    from benchmarks.dataset import load_dataset

    return load_dataset()


def _normalized_text(value: str) -> str:
    """Normalize text for case- and punctuation-insensitive title checks."""
    value = unicodedata.normalize("NFKD", value).casefold()
    value = "".join(char for char in value if not unicodedata.combining(char))
    return re.sub(r"[^\w]+", " ", value).strip()


def validate_golden_set(queries: list[dict], nodes: dict) -> None:
    """Fail fast when benchmark ground truth cannot be evaluated honestly."""
    errors = []
    for index, entry in enumerate(queries, start=1):
        query = entry.get("query", "")
        relevant_ids = entry.get("relevant_note_ids")
        if not isinstance(query, str) or not query.strip():
            errors.append(f"query {index}: query must be a non-empty string")
            continue
        if not isinstance(relevant_ids, list) or not 1 <= len(relevant_ids) <= 4:
            errors.append(f"query {index}: relevant_note_ids must contain 1-4 IDs")
            continue
        normalized_query = _normalized_text(query)
        for note_id in relevant_ids:
            node = nodes.get(note_id)
            if node is None:
                errors.append(f"query {index}: relevant_note_id does not exist in graph: {note_id}")
                continue
            title = _normalized_text(str(node.get("title", "")))
            if title and title in normalized_query:
                errors.append(f"query {index}: title leakage for target {note_id}: {node.get('title', '')}")
    if errors:
        raise ValueError("Invalid golden set:\n- " + "\n- ".join(errors))


def _run_single_query(
    query: str,
    expected: set[str],
    nodes: dict,
    edges: dict,
    collection,
    bm25_index: BM25Index,
    state: dict,
    category: str,
    cold: bool = True,
    collect_hops: bool = True,
) -> tuple[dict, dict]:
    """Run one query under current CONFIG and return (metrics, metadata)."""
    t0 = time.perf_counter()
    routing_meta = {} if collect_hops else None
    active = attention(
        query,
        nodes,
        edges,
        collection,
        bm25_index=bm25_index,
        hebbian_state=state,
        routing_meta=routing_meta,
    )
    latency_ms = (time.perf_counter() - t0) * 1000

    activated_ids = [nid for nid, _ in sorted(active.items(), key=lambda x: -x[1])]
    metrics = compute_all_metrics(activated_ids, expected, k_values=(1, 3, 5, 10))
    metrics["latency_ms"] = round(latency_ms, 2)
    metrics["query"] = query
    metrics["category"] = category
    metrics["n_activated"] = len(activated_ids)

    metadata = {"latency_ms": latency_ms}

    # Online plasticity: update Hebbian state in warm mode unless disabled.
    if not cold and CONFIG.get("online_plasticity", True):
        state, _updated_keys, _pruned = hebbian_update(active, state)

    if collect_hops:
        hop_counts = defaultdict(int)
        for detail in (routing_meta or {}).get("activation_details", []):
            hop = detail.get("hop", 0)
            hop_counts[hop] += 1
        metadata["hop_histogram"] = dict(sorted(hop_counts.items()))

    metadata["synapse_count"] = len(state["synapses"])
    return metrics, metadata


def run_eval(
    config_overrides: dict,
    queries: list[dict] | None = None,
    *,
    config_path: str | None = None,
    cold: bool = True,
    warm: bool = False,
    collect_hops: bool = True,
    golden_set_path: Path | str | None = None,
) -> dict:
    """Run the golden set under a temporary config overlay.

    Args:
        config_overrides: dict of CONFIG keys to override for this run only.
        queries: optional list of query dicts; defaults to golden_set.yaml.
        config_path: optional path to a YAML config to load before overlaying.
        cold: run once from a fresh Hebbian state.
        warm: also run a second pass over the same queries to measure plasticity.
        collect_hops: instrument traversal to build a hop histogram.
        golden_set_path: optional path to a YAML golden set.

    Returns:
        dict with cold Metrics, optional warm Metrics, plus metadata.
    """
    queries = queries if queries is not None else _load_golden_set(golden_set_path)
    if not queries:
        raise ValueError("No queries in golden set")

    # Ensure CONFIG reflects the caller's on-disk config before we overlay.
    load_config(config_path)
    resolved_config_path = str(Path(config_path).resolve()) if config_path else None

    # Ablations must not mutate the production ChromaDB or state files.
    # Use a dedicated, reusable embedding cache under benchmarks/.
    ablation_chroma = Path(__file__).resolve().parent / ".bdh-chroma-ablation"

    overrides = dict(config_overrides)
    overrides.setdefault("chroma_path", str(ablation_chroma))

    with config_overlay(overrides):
        nodes, edges, _state_path = _build_materialized_graph()
        validate_golden_set(queries, nodes)
        collection = compute_all_embeddings(nodes, CONFIG["vault_path"])
        bm25_index = BM25Index(nodes)

        results: dict[str, Any] = {
            "config_overrides": config_overrides,
            "config_hash": _config_hash(config_overrides),
            "config_path": resolved_config_path,
            "vault_size": len(nodes),
            "query_count": len(queries),
            "query_version": _query_version(queries),
            "cold": None,
        }

        cold_state = _fresh_state()
        if cold:
            cold_metrics, cold_meta = _run_pass(
                queries, nodes, edges, collection, bm25_index, cold_state, cold=True, collect_hops=collect_hops
            )
            results["cold"] = cold_metrics
            results["cold_hop_histogram"] = cold_meta["hop_histogram"]
            results["cold_final_synapses"] = len(cold_state["synapses"])

        if warm:
            warm_state = cold_state
            warm_metrics, warm_meta = _run_pass(
                queries, nodes, edges, collection, bm25_index, warm_state, cold=False, collect_hops=collect_hops
            )
            results["warm"] = warm_metrics
            results["warm_hop_histogram"] = warm_meta["hop_histogram"]
            results["warm_final_synapses"] = len(warm_state["synapses"])

        return results


def _build_materialized_graph():
    """Build graph using the current CONFIG. Returns (nodes, edges, state_path)."""
    vault_path = CONFIG["vault_path"]
    nodes, edges = build_graph(vault_path, use_cache=True)
    return nodes, edges, None


def _run_pass(queries, nodes, edges, collection, bm25_index, state, *, cold: bool, collect_hops: bool):
    per_query = []
    aggregated_meta = {"hop_histogram": defaultdict(int), "latencies": [], "synapse_counts": []}

    for entry in queries:
        metrics, metadata = _run_single_query(
            query=entry["query"],
            expected=set(entry["relevant_note_ids"]),
            nodes=nodes,
            edges=edges,
            collection=collection,
            bm25_index=bm25_index,
            state=state,
            category=entry.get("category", "unknown"),
            cold=cold,
            collect_hops=collect_hops,
        )
        per_query.append(metrics)
        aggregated_meta["latencies"].append(metadata["latency_ms"])
        aggregated_meta["synapse_counts"].append(metadata["synapse_count"])
        for hop, count in metadata.get("hop_histogram", {}).items():
            aggregated_meta["hop_histogram"][int(hop)] += int(count)

    agg = aggregate_metrics(per_query)
    metrics = Metrics(
        mrr=agg.get("mrr", {}).get("mean", 0.0),
        recall_at_5=agg.get("recall@5", {}).get("mean", 0.0),
        precision_at_5=agg.get("precision@5", {}).get("mean", 0.0),
        ndcg_at_5=agg.get("ndcg@5", {}).get("mean", 0.0),
        mean_latency_ms=sum(aggregated_meta["latencies"]) / len(aggregated_meta["latencies"])
        if aggregated_meta["latencies"]
        else 0.0,
        per_query=per_query,
        hop_histogram=dict(sorted(aggregated_meta["hop_histogram"].items())),
        synapse_counts=aggregated_meta["synapse_counts"],
    )
    return metrics, aggregated_meta


def _evaluate_hebbian_trajectory(
    train_queries: list[dict],
    holdout_queries: list[dict],
    *,
    nodes: dict,
    edges: list,
    collection,
    bm25_index,
    collect_hops: bool,
) -> dict:
    """Measure holdout ranking before and after a separate Hebbian trajectory."""
    baseline_state = _fresh_state()
    cold_metrics, _ = _run_pass(
        holdout_queries,
        nodes,
        edges,
        collection,
        bm25_index,
        baseline_state,
        cold=True,
        collect_hops=collect_hops,
    )

    trained_state = _fresh_state()
    train_metrics, _ = _run_pass(
        train_queries,
        nodes,
        edges,
        collection,
        bm25_index,
        trained_state,
        cold=False,
        collect_hops=collect_hops,
    )
    after_training_metrics, _ = _run_pass(
        holdout_queries,
        nodes,
        edges,
        collection,
        bm25_index,
        trained_state,
        cold=True,
        collect_hops=collect_hops,
    )
    return {
        "cold": cold_metrics,
        "train": train_metrics,
        "after_training": after_training_metrics,
        "cold_final_synapses": len(baseline_state["synapses"]),
        "trained_final_synapses": len(trained_state["synapses"]),
    }


def _query_version(queries: list[dict]) -> str:
    import hashlib

    canonical = json.dumps(
        [{"q": q.get("query"), "c": q.get("category"), "r": q.get("relevant_note_ids")} for q in queries],
        sort_keys=True,
        default=str,
    )
    return hashlib.md5(canonical.encode()).hexdigest()[:8]


def _format_table(rows: list[dict], baseline: dict | None = None) -> str:
    """Render an ablation results table as markdown."""
    header = ["config", "MRR", "R@5", "P@5", "NDCG@5", "lat(ms)", "ΔMRR"]
    lines = ["| " + " | ".join(header) + " |"]
    lines.append("|" + "|".join(["---"] * len(header)) + "|")

    base_mrr = baseline["cold"].mrr if baseline else 0.0
    for row in rows:
        cold = row["cold"]
        delta = cold.mrr - base_mrr if baseline else 0.0
        cells = [
            row["label"],
            f"{cold.mrr:.4f}",
            f"{cold.recall_at_5:.4f}",
            f"{cold.precision_at_5:.4f}",
            f"{cold.ndcg_at_5:.4f}",
            f"{cold.mean_latency_ms:.0f}",
            f"{delta:+.4f}" if baseline else "—",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _save_artifact(results: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    filename = f"ablation-{results['config_hash']}-{ts}.json"
    path = out_dir / filename
    # Convert Metrics dataclasses to plain dicts for JSON.
    serializable = _serialize(results)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    return path


def _serialize(obj: Any) -> Any:
    if isinstance(obj, Metrics):
        return {
            "mrr": obj.mrr,
            "recall_at_5": obj.recall_at_5,
            "precision_at_5": obj.precision_at_5,
            "ndcg_at_5": obj.ndcg_at_5,
            "mean_latency_ms": obj.mean_latency_ms,
            "per_query": obj.per_query,
            "hop_histogram": obj.hop_histogram,
            "synapse_counts": obj.synapse_counts,
        }
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialize(v) for v in obj]
    return obj


def run_ablation_matrix(
    overrides_list: list[dict],
    labels: list[str],
    queries: list[dict] | None = None,
    *,
    warm: bool = False,
    out_dir: Path | str | None = None,
) -> list[dict]:
    """Run baseline + several overrides and print a comparative table."""
    if out_dir is None:
        out_dir = Path(__file__).resolve().parent / "results"
    else:
        out_dir = Path(out_dir)

    queries = queries if queries is not None else _load_golden_set()
    rows = []
    for label, overrides in zip(labels, overrides_list):
        results = run_eval(overrides, queries, warm=warm)
        rows.append({"label": label, **results})
        artifact = _save_artifact(results, out_dir)
        print(f"  saved artifact: {artifact.name}")

    baseline = rows[0] if rows else None
    print("\n" + _format_table(rows, baseline=baseline))
    return rows


def main():
    """CLI entry point for a quick baseline run."""
    config_path = 'bdh-config.local.yaml' if Path('bdh-config.local.yaml').exists() else None
    load_config(config_path)
    queries = _load_golden_set()
    print(f"Loaded {len(queries)} queries from golden set")

    baseline = run_eval({}, queries, config_path=config_path)
    artifact = _save_artifact(baseline, Path(__file__).resolve().parent / "results")
    print(f"\nBaseline saved to {artifact}")
    print("\n" + _format_table([{"label": "baseline", **baseline}]))


if __name__ == "__main__":
    main()
