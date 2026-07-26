"""Evaluate cold vs warm Hebbian effects with/without propagation gain."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.ablation import run_eval, _load_golden_set


def main():
    queries = _load_golden_set()

    configs = [
        ("baseline hebbian_gain=0", {"hebbian_gain": 0.0}),
        ("hebbian_gain=0.5", {"hebbian_gain": 0.5}),
        ("hebbian_gain=1.0", {"hebbian_gain": 1.0}),
        ("hebbian_gain=2.0", {"hebbian_gain": 2.0}),
    ]

    rows = []
    for label, overrides in configs:
        print(f"\n>>> Running {label}...")
        result = run_eval(overrides, queries, config_path='bdh-config.local.yaml', cold=True, warm=True)
        rows.append({"label": label, **result})
        c = result["cold"]; w = result["warm"]
        print(f"  cold: MRR={c.mrr:.4f} R@5={c.recall_at_5:.4f} NDCG={c.ndcg_at_5:.4f} syns={result['cold_final_synapses']}")
        print(f"  warm: MRR={w.mrr:.4f} R@5={w.recall_at_5:.4f} NDCG={w.ndcg_at_5:.4f} syns={result['warm_final_synapses']}")
        print(f"  Δwarm-cold: MRR={w.mrr-c.mrr:+.4f}")

    print("\n\n## Tranche 3.2 — Cold vs Warm with Hebbian propagation gain")
    print("| config | cold MRR | warm MRR | Δ | cold R@5 | warm R@5 | cold NDCG | warm NDCG | syns |")
    print("|---|---|---|---|---|---|---|---|---|")
    for row in rows:
        c = row["cold"]; w = row["warm"]
        print(f"| {row['label']} | {c.mrr:.4f} | {w.mrr:.4f} | {w.mrr-c.mrr:+.4f} | "
              f"{c.recall_at_5:.4f} | {w.recall_at_5:.4f} | {c.ndcg_at_5:.4f} | {w.ndcg_at_5:.4f} | "
              f"{row['warm_final_synapses']} |")


if __name__ == "__main__":
    main()
