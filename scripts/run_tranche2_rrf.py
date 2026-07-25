"""Compare vector-only vs BM25-only vs weighted-sum hybrid vs RRF."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.ablation import run_eval, _load_golden_set


def main():
    queries = _load_golden_set()

    configs = [
        ("vector-only", {"hybrid_search": False}),
        ("bm25-only", {"hybrid_search": True, "hybrid_fusion": "weighted", "hybrid_alpha": 0.0, "hybrid_beta": 1.0}),
        ("weighted-70/30", {"hybrid_search": True, "hybrid_fusion": "weighted", "hybrid_alpha": 0.7, "hybrid_beta": 0.3}),
        ("rrf", {"hybrid_search": True, "hybrid_fusion": "rrf", "rrf_k": 60}),
    ]

    rows = []
    for label, overrides in configs:
        print(f"\n>>> Running {label}...")
        result = run_eval(overrides, queries, config_path='bdh-config.local.yaml')
        rows.append({"label": label, **result})
        c = result["cold"]
        print(f"MRR={c.mrr:.4f} R@5={c.recall_at_5:.4f} P@5={c.precision_at_5:.4f} NDCG@5={c.ndcg_at_5:.4f}")

    vector_mrr = rows[0]["cold"].mrr
    print("\n\n## Hybrid fusion comparison")
    print("| method | MRR | Δ vs vector | R@5 | P@5 | NDCG@5 |")
    print("|---|---|---|---|---|---|")
    for row in rows:
        c = row["cold"]
        delta = c.mrr - vector_mrr
        print(f"| {row['label']} | {c.mrr:.4f} | {delta:+.4f} | {c.recall_at_5:.4f} | {c.precision_at_5:.4f} | {c.ndcg_at_5:.4f} |")


if __name__ == "__main__":
    main()
