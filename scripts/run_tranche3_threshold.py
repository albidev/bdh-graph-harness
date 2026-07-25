"""Ablation for unified threshold policy: adaptive vs static."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.ablation import run_eval, _load_golden_set


def main():
    queries = _load_golden_set()

    configs = [
        ("adaptive unified", {"adaptive_threshold": True}),
        ("static 0.25", {"adaptive_threshold": False, "active_threshold": 0.25}),
        ("static 0.15", {"adaptive_threshold": False, "active_threshold": 0.15}),
        ("static 0.35", {"adaptive_threshold": False, "active_threshold": 0.35}),
    ]

    rows = []
    for label, overrides in configs:
        print(f"\n>>> Running {label}...")
        result = run_eval(overrides, queries, config_path='bdh-config.local.yaml')
        rows.append({"label": label, **result})
        c = result["cold"]
        print(f"MRR={c.mrr:.4f} R@5={c.recall_at_5:.4f} NDCG={c.ndcg_at_5:.4f} hops={result['cold_hop_histogram']}")

    base = rows[0]["cold"].mrr
    print("\n\n## Threshold policy ablation")
    print("| policy | MRR | Δ | R@5 | NDCG@5 | hop histogram |")
    print("|---|---|---|---|---|---|")
    for row in rows:
        c = row["cold"]
        print(f"| {row['label']} | {c.mrr:.4f} | {c.mrr-base:+.4f} | {c.recall_at_5:.4f} | {c.ndcg_at_5:.4f} | {row['cold_hop_histogram']} |")


if __name__ == "__main__":
    main()
