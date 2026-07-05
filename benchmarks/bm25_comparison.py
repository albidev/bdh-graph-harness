"""Comparative BM25 benchmark — no stemming vs Italian Snowball stemming.

Tests whether Italian stemming improves BM25 retrieval quality.
Measures: MRR, Recall@5, Precision@5, NDCG@5, latency.
"""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bdh_graph_harness.config import load_config, CONFIG
from bdh_graph_harness.graph.builder import build_graph
from bdh_graph_harness.retrieval.chroma_store import compute_all_embeddings
from bdh_graph_harness.retrieval.bm25 import BM25Index
from bdh_graph_harness.retrieval.bm25_stemmed import BM25StemmedIndex
from bdh_graph_harness.retrieval.embeddings import get_embeddings
from bdh_graph_harness.retrieval.attention import compute_adaptive_threshold

from benchmarks.dataset import load_dataset
from benchmarks.metrics import compute_all_metrics, aggregate_metrics


def setup(vault_root):
    config = load_config()
    print(f"Loading graph from {vault_root}...")
    nodes, edges = build_graph(vault_root, use_cache=True)
    print(f"  {len(nodes)} nodes, {sum(len(v) for v in edges.values())} edges")

    print("Loading ChromaDB embeddings...")
    collection = compute_all_embeddings(nodes, vault_root)
    print(f"  {collection.count()} vectors")

    print("Building BM25 index (no stemming)...")
    bm25_plain = BM25Index(nodes)
    print(f"  {len(bm25_plain.df)} terms across {bm25_plain.N} docs")

    print("Building BM25 index (with Italian stemming)...")
    bm25_stemmed = BM25StemmedIndex(nodes)
    print(f"  {len(bm25_stemmed.df)} unique stems across {bm25_stemmed.N} docs")

    return nodes, edges, collection, bm25_plain, bm25_stemmed


def run_query(query, nodes, collection, bm25, method, alpha=0.7, beta=0.3):
    """Run a single query and return (activated_list, latency_ms)."""
    t0 = time.perf_counter()

    q_emb = get_embeddings([query])[0]

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

    scored = {}
    if method == "vector":
        scored = dict(raw_vector_scores)
    elif method == "bm25":
        bm25_scores = bm25.score_batch(query, list(raw_vector_scores.keys()))
        scored = {nid: bm25_scores.get(nid, 0.0) for nid in raw_vector_scores}
    elif method == "hybrid":
        bm25_scores = bm25.score_batch(query, list(raw_vector_scores.keys()))
        for nid in raw_vector_scores:
            vec_s = raw_vector_scores.get(nid, 0.0)
            bm_s = bm25_scores.get(nid, 0.0)
            scored[nid] = alpha * vec_s + beta * bm_s

    sorted_notes = sorted(scored.items(), key=lambda x: -x[1])

    if CONFIG.get("adaptive_threshold", True):
        scores = [s for _, s in sorted_notes]
        threshold = compute_adaptive_threshold(scores)
        sorted_notes = [(nid, s) for nid, s in sorted_notes if s >= threshold]

    t1 = time.perf_counter()
    latency_ms = (t1 - t0) * 1000
    return sorted_notes, latency_ms


def main():
    vault_root = os.environ.get("VAULT_ROOT", os.path.expanduser("~/Documents/Hermes"))
    nodes, edges, collection, bm25_plain, bm25_stemmed = setup(vault_root)
    dataset = load_dataset()

    # Configurations to test
    configs = [
        ("vector-only", None, None, None),
        ("bm25-plain", bm25_plain, "bm25", None),
        ("bm25-stemmed", bm25_stemmed, "bm25", None),
        ("hybrid-plain-70/30", bm25_plain, "hybrid", (0.7, 0.3)),
        ("hybrid-stemmed-70/30", bm25_stemmed, "hybrid", (0.7, 0.3)),
        ("hybrid-plain-85/15", bm25_plain, "hybrid", (0.85, 0.15)),
        ("hybrid-stemmed-85/15", bm25_stemmed, "hybrid", (0.85, 0.15)),
    ]

    all_results = {}

    for name, bm25, method, weights in configs:
        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}")

        alpha, beta = weights if weights else (0.7, 0.3)
        per_query = []

        for i, entry in enumerate(dataset):
            query = entry["query"]
            expected = entry["expected_notes"]
            category = entry["category"]

            if method is None:
                activated, latency = run_query(query, nodes, collection, None, "vector")
            else:
                activated, latency = run_query(query, nodes, collection, bm25, method, alpha, beta)

            activated_ids = [nid for nid, _ in activated]
            metrics = compute_all_metrics(activated_ids, expected, k_values=(1, 3, 5, 10))
            metrics["latency_ms"] = round(latency, 2)
            metrics["query"] = query
            metrics["category"] = category
            metrics["n_activated"] = len(activated_ids)
            per_query.append(metrics)

            hit = "✅" if metrics["first_hit_position"] > 0 else "❌"
            print(f"  {hit} [{category:>8}] {query[:48]:<48} "
                  f"mrr={metrics['mrr']:.2f} p@5={metrics['precision@5']:.2f} "
                  f"lat={metrics['latency_ms']:.0f}ms act={metrics['n_activated']}")

        agg = aggregate_metrics(per_query)
        all_results[name] = {
            "aggregate": agg,
            "per_query": per_query,
        }

        print(f"\n  --- {name} Summary ---")
        for metric_name in ("precision@5", "recall@5", "f1@5", "ndcg@5", "mrr", "latency_ms"):
            if metric_name in agg:
                s = agg[metric_name]
                print(f"    {metric_name:>15}: {s['mean']:.4f} ± {s['std']:.4f} "
                      f"[{s['min']:.4f} – {s['max']:.4f}]")

    # Comparative table
    print(f"\n\n{'='*72}")
    print("  COMPARATIVE SUMMARY")
    print(f"{'='*72}")
    header = f"{'Method':<25} {'MRR':>8} {'R@5':>8} {'P@5':>8} {'NDCG@5':>8} {'Lat':>7}"
    print(header)
    print("-" * 72)
    for name in all_results:
        agg = all_results[name]["aggregate"]
        mrr = agg["mrr"]["mean"]
        r5 = agg["recall@5"]["mean"]
        p5 = agg["precision@5"]["mean"]
        ndcg = agg["ndcg@5"]["mean"]
        lat = agg["latency_ms"]["mean"]
        print(f"{name:<25} {mrr:>8.4f} {r5:>8.4f} {p5:>8.4f} {ndcg:>8.4f} {lat:>6.0f}ms")

    # Save
    output_path = str(Path(__file__).resolve().parent / "bm25_comparison.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_path}")

    # Winner analysis
    vector_mrr = all_results["vector-only"]["aggregate"]["mrr"]["mean"]
    best_name = "vector-only"
    best_mrr = vector_mrr
    for name in all_results:
        mrr = all_results[name]["aggregate"]["mrr"]["mean"]
        if mrr > best_mrr:
            best_mrr = mrr
            best_name = name

    print(f"\n🏆 Winner: {best_name} (MRR={best_mrr:.4f})")
    if best_name == "vector-only":
        print("   → BM25 does not help for this vault.")
    else:
        improvement = (best_mrr - vector_mrr) / vector_mrr * 100
        print(f"   → {improvement:+.1f}% improvement over vector-only")


if __name__ == "__main__":
    main()
