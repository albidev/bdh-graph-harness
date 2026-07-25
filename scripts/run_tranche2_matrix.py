"""Run the full Tranche 2 ablation matrix and print the report."""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.ablation import run_eval, _load_golden_set


def main():
    queries = _load_golden_set()

    configs = [
        ("baseline", {}),
        ("hebbian_seed_boost off", {"hebbian_seed_boost": False}),
        ("max_hop 1", {"max_hop": 1}),
        ("max_hop 2", {"max_hop": 2}),
        ("max_hop 3", {"max_hop": 3}),
        ("phantom off", {"phantom_links_enabled": False}),
        ("adaptive off", {"adaptive_threshold": False}),
        ("hub_dampening off", {"hub_dampening": False}),
        ("iaf on", {"experimental_integrate_fire": True}),
    ]

    rows = []
    for label, overrides in configs:
        print(f"\n>>> Running {label}...")
        result = run_eval(overrides, queries, config_path='bdh-config.local.yaml')
        rows.append({"label": label, **result})
        print(f"MRR={result['cold'].mrr:.4f} R@5={result['cold'].recall_at_5:.4f} "
              f"NDCG@5={result['cold'].ndcg_at_5:.4f} hops={result['cold_hop_histogram']}")

    # Noise floor: baseline run 3x with shuffled query order
    print("\n>>> Estimating noise floor (baseline x3, shuffled)...")
    base_mrrs = []
    for i in range(3):
        shuffled = list(queries)
        random.seed(1000 + i)
        random.shuffle(shuffled)
        r = run_eval({}, shuffled, config_path='bdh-config.local.yaml')
        base_mrrs.append(r["cold"].mrr)
        print(f"  shuffle {i+1}: MRR={r['cold'].mrr:.4f}")
    noise_floor = max(base_mrrs) - min(base_mrrs)
    print(f"\nNoise floor (max MRR spread across 3 shuffled runs): {noise_floor:.4f}")

    # Print comparative markdown table
    base_mrr = rows[0]["cold"].mrr
    print("\n\n## Tranche 2.1 — Ablation matrix")
    print("| config | MRR | ΔMRR | R@5 | P@5 | NDCG@5 | lat | hop histogram |")
    print("|---|---|---|---|---|---|---|---|")
    for row in rows:
        c = row["cold"]
        delta = c.mrr - base_mrr
        hops = row.get("cold_hop_histogram", {})
        print(f"| {row['label']} | {c.mrr:.4f} | {delta:+.4f} | "
              f"{c.recall_at_5:.4f} | {c.precision_at_5:.4f} | {c.ndcg_at_5:.4f} | "
              f"{c.mean_latency_ms:.0f}ms | {hops} |")
    print(f"\nNoise floor: {noise_floor:.4f} MRR")


if __name__ == "__main__":
    main()
