"""Benchmark runner — executes queries against the real vault and measures quality.

Usage:
    python -m benchmarks.run_all [--vault PATH] [--output results.json]
"""
import json
import os
import sys
import time
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bdh_graph_harness.config import load_config, CONFIG
from bdh_graph_harness.graph.builder import build_graph
from bdh_graph_harness.retrieval.chroma_store import compute_all_embeddings
from bdh_graph_harness.retrieval.attention import (
    compute_adaptive_threshold,
)
from bdh_graph_harness.retrieval.bm25 import BM25Index
from bdh_graph_harness.retrieval.hybrid import hybrid_score
from bdh_graph_harness.retrieval.embeddings import get_embeddings

from benchmarks.dataset import load_dataset


def setup_system(vault_root, config=None):
    """Load graph, embeddings, and BM25 index from the real vault."""
    if config is None:
        config = load_config()

    print(f"Loading graph from {vault_root}...")
    nodes, edges = build_graph(vault_root, use_cache=True)
    print(f"  {len(nodes)} nodes, {sum(len(v) for v in edges.values())} edges")

    print("Loading ChromaDB embeddings...")
    collection = compute_all_embeddings(nodes, vault_root)
    print(f"  {collection.count()} vectors")

    print("Building BM25 index...")
    bm25 = BM25Index(nodes)
    print(f"  {len(bm25.df)} terms across {bm25.N} docs")

    return nodes, edges, collection, bm25


def run_single_query(query, nodes, edges, collection, bm25, method="hybrid"):
    """Run a single query and return activated notes with scores and latency.

    Returns:
        (activated_list, latency_ms) where activated_list is [(note_id, score), ...]
    """
    t0 = time.perf_counter()

    # Get embeddings
    q_emb = get_embeddings([query])[0]

    # Vector search via ChromaDB
    overfetch = min(CONFIG.get("seed_count", 5) * 10, collection.count())
    results = collection.query(
        query_embeddings=[q_emb],
        n_results=overfetch,
        include=["documents", "metadatas"],
    )

    raw_vector_scores = {}
    if results and results["ids"] and results["ids"][0]:
        ids = results["ids"][0]
        distances = results["distances"][0] if results.get("distances") else [0.0] * len(ids)
        for nid, dist in zip(ids, distances):
            sim = max(0.0, 1.0 - dist)
            raw_vector_scores[nid] = sim

    # Score by method
    scored = {}
    if method == "hybrid":
        # Compute BM25 ONCE for all candidates, normalized [0,1]
        bm25_scores = bm25.score_batch(query, list(raw_vector_scores.keys()))
        alpha = CONFIG.get("hybrid_alpha", 0.7)
        beta = CONFIG.get("hybrid_beta", 0.3)
        for nid in raw_vector_scores:
            vec_s = raw_vector_scores.get(nid, 0.0)
            bm_s = bm25_scores.get(nid, 0.0)
            scored[nid] = alpha * vec_s + beta * bm_s
    elif method == "vector":
        for nid in raw_vector_scores:
            scored[nid] = raw_vector_scores.get(nid, 0.0)
    elif method == "bm25":
        # Use batch normalization
        bm25_scores = bm25.score_batch(query, list(raw_vector_scores.keys()))
        for nid in raw_vector_scores:
            scored[nid] = bm25_scores.get(nid, 0.0)

    # Sort by score
    sorted_notes = sorted(scored.items(), key=lambda x: -x[1])

    # Apply adaptive threshold
    if CONFIG.get("adaptive_threshold", True):
        scores = [s for _, s in sorted_notes]
        threshold = compute_adaptive_threshold(scores)
        sorted_notes = [(nid, s) for nid, s in sorted_notes if s >= threshold]

    t1 = time.perf_counter()
    latency_ms = (t1 - t0) * 1000

    return sorted_notes, latency_ms


def run_benchmark(vault_root=None, output_path=None):
    """Run the full benchmark suite."""
    from benchmarks.metrics import compute_all_metrics, aggregate_metrics

    if vault_root is None:
        vault_root = os.environ.get("VAULT_ROOT", os.path.expanduser("~/Documents/Hermes"))

    config = load_config()
    nodes, edges, collection, bm25 = setup_system(vault_root, config)
    dataset = load_dataset()

    results = {
        "vault": vault_root,
        "vault_size": len(nodes),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "methods": {},
    }

    methods = ["hybrid", "vector", "bm25"]

    for method in methods:
        print(f"\n{'='*60}")
        print(f"Running benchmark: {method}")
        print(f"{'='*60}")

        per_query = []
        for i, entry in enumerate(dataset):
            query = entry["query"]
            expected = entry["expected_notes"]
            category = entry["category"]

            activated, latency = run_single_query(
                query, nodes, edges, collection, bm25, method=method
            )
            activated_ids = [nid for nid, _ in activated]

            metrics = compute_all_metrics(activated_ids, expected, k_values=(1, 3, 5, 10))
            metrics["latency_ms"] = round(latency, 2)
            metrics["query"] = query
            metrics["category"] = category
            metrics["n_activated"] = len(activated_ids)
            metrics["expected_count"] = len(expected)

            per_query.append(metrics)

            hit = "✅" if metrics["first_hit_position"] > 0 else "❌"
            print(f"  {hit} [{category:>8}] {query[:50]:<50} "
                  f"mrr={metrics['mrr']:.2f} p@5={metrics['precision@5']:.2f} "
                  f"lat={metrics['latency_ms']:.0f}ms")

        # Aggregate
        agg = aggregate_metrics(per_query)
        results["methods"][method] = {
            "aggregate": agg,
            "per_query": per_query,
        }

        # Print summary
        print(f"\n--- {method.upper()} Summary ---")
        for metric_name, stats in agg.items():
            if metric_name in ("precision@5", "recall@5", "f1@5", "ndcg@5", "mrr", "latency_ms"):
                print(f"  {metric_name:>20}: {stats['mean']:.4f} ± {stats['std']:.4f} "
                      f"[{stats['min']:.4f} – {stats['max']:.4f}]")

    # Save results
    if output_path is None:
        output_path = str(Path(__file__).resolve().parent / "results.json")

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_path}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="BDH Benchmark Suite")
    parser.add_argument("--vault", help="Path to vault root")
    parser.add_argument("--output", help="Output JSON path")
    args = parser.parse_args()

    run_benchmark(vault_root=args.vault, output_path=args.output)
