"""Benchmark metrics — retrieval quality, ranking, and end-to-end measures."""

import math
from collections import defaultdict


def precision_at_k(activated, expected, k):
    """Fraction of top-K activated notes that are in expected set."""
    top_k = activated[:k]
    hits = sum(1 for n in top_k if n in expected)
    return hits / k if k > 0 else 0.0


def recall_at_k(activated, expected, k):
    """Fraction of expected notes found in top-K activated."""
    if not expected:
        return 1.0
    top_k = activated[:k]
    hits = sum(1 for n in top_k if n in expected)
    return hits / len(expected)


def f1_at_k(activated, expected, k):
    """Harmonic mean of precision@K and recall@K."""
    p = precision_at_k(activated, expected, k)
    r = recall_at_k(activated, expected, k)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def mean_reciprocal_rank(activated, expected):
    """Reciprocal rank of the first relevant note in activated list."""
    for i, note in enumerate(activated):
        if note in expected:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(activated, expected, k):
    """Normalized Discounted Cumulative Gain at K."""
    dcg = 0.0
    for i, note in enumerate(activated[:k]):
        rel = 1.0 if note in expected else 0.0
        dcg += rel / math.log2(i + 2)  # i+2 because log2(1) = 0

    # Ideal DCG
    ideal = min(len(expected), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal))

    return dcg / idcg if idcg > 0 else 0.0


def activation_sparsity(activated, total_notes):
    """Fraction of notes that are activated (lower = sparser)."""
    return len(activated) / total_notes if total_notes > 0 else 0.0


def compute_all_metrics(activated_notes, expected_notes, k_values=(1, 3, 5, 10)):
    """Compute all retrieval metrics for a single query.

    Args:
        activated_notes: list of note IDs in activation order (highest score first)
        expected_notes: set of note IDs that are ground truth
        k_values: which K values to compute precision/recall/F1/NDCG for

    Returns:
        dict with metric_name -> value
    """
    results = {}

    for k in k_values:
        results[f"precision@{k}"] = precision_at_k(activated_notes, expected_notes, k)
        results[f"recall@{k}"] = recall_at_k(activated_notes, expected_notes, k)
        results[f"f1@{k}"] = f1_at_k(activated_notes, expected_notes, k)
        results[f"ndcg@{k}"] = ndcg_at_k(activated_notes, expected_notes, k)

    results["mrr"] = mean_reciprocal_rank(activated_notes, expected_notes)
    results["first_hit_position"] = _first_hit_position(activated_notes, expected_notes)

    return results


def _first_hit_position(activated, expected):
    """1-indexed position of first relevant note, or -1 if none found."""
    for i, note in enumerate(activated):
        if note in expected:
            return i + 1
    return -1


def aggregate_metrics(per_query_results):
    """Aggregate per-query metrics into mean ± std.

    Args:
        per_query_results: list of metric dicts (one per query)

    Returns:
        dict with metric_name -> {mean, std, min, max}
    """
    if not per_query_results:
        return {}

    keys = per_query_results[0].keys()
    agg = {}

    # Skip non-numeric fields
    skip = {"query", "category"}

    for key in keys:
        if key in skip:
            continue
        values = [r[key] for r in per_query_results if key in r and isinstance(r[key], (int, float))]
        if not values:
            continue
        n = len(values)
        mean = sum(values) / n
        variance = sum((v - mean) ** 2 for v in values) / n if n > 1 else 0
        std = math.sqrt(variance)
        agg[key] = {
            "mean": round(mean, 4),
            "std": round(std, 4),
            "min": round(min(values), 4),
            "max": round(max(values), 4),
            "n": n,
        }

    return agg
