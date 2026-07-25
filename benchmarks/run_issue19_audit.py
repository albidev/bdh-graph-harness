"""Issue #19 compatibility and mixed-language audit runner.

This script evaluates the BDH retrieval contract without touching core
retrieval or frontend code.  It can run in two modes:

1.  **Baseline mode** — measures current single-query retrieval on the
    Issue #19 golden set and reports where the contract is currently satisfied
    (for example, original-query preservation is already correct because the
    backend receives the original query as the authoritative ``query`` field).

2.  **Contract audit mode** — validates request/response schema assumptions
    such as bounds, deduplication, and provenance shape by exercising the
    helper functions that will be used by the future multi-query path.

The runner intentionally does NOT implement multi-query fusion itself; that
belongs to the retrieval backend once the contract is accepted.

Usage:
    python benchmarks/run_issue19_audit.py
    python benchmarks/run_issue19_audit.py --mode contract
    python benchmarks/run_issue19_audit.py --mode baseline --golden benchmarks/golden_set_issue19.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.ablation import _load_golden_set, run_eval
from benchmarks.metrics import aggregate_metrics, compute_all_metrics


def _note_variants(query_entry: dict) -> list[dict]:
    """Return the explicit variant list or a single synthetic variant."""
    variants = query_entry.get("variants") or []
    if not variants:
        return [{"query": query_entry["query"], "language": "original", "weight": 1.0}]
    # Ignore the synthetic original here; contract tests inspect the explicit list.
    return [v for v in variants if v.get("query")]


def _valid_variants(query_entry: dict, max_variants: int = 3) -> list[dict]:
    """Normalize variants the way the future backend contract will:

    - strip empty / whitespace-only queries
    - remove exact duplicates (keep first occurrence)
    - cap at ``max_variants``
    - keep ``language`` and ``weight`` if present
    """
    seen: set[str] = set()
    result: list[dict] = []
    for v in query_entry.get("variants", []):
        q = str(v.get("query", "")).strip()
        if not q:
            continue
        if q in seen:
            continue
        seen.add(q)
        result.append({
            "query": q,
            "language": v.get("language", "unknown"),
            "weight": float(v.get("weight", 1.0)),
        })
        if len(result) >= max_variants:
            break
    return result


def audit_contract(golden_path: Path | str, max_variants: int = 3) -> dict:
    """Run contract-level checks on the Issue #19 golden set.

    Returns a dict with counts and any failing checks per query.
    """
    queries = _load_golden_set(golden_path)
    categories = Counter(q.get("category", "unknown") for q in queries)

    checks = []
    for i, entry in enumerate(queries):
        original = entry.get("query", "").strip()
        variants = _valid_variants(entry, max_variants=max_variants)
        raw_count = len(entry.get("variants", []))

        ok = True
        notes: list[str] = []

        if not original:
            ok = False
            notes.append("missing original query")

        # Bounds check
        if raw_count > max_variants:
            if len(variants) > max_variants:
                ok = False
                notes.append(f"variant cap exceeded: {len(variants)} > {max_variants}")
            else:
                notes.append(f"variant list truncated from {raw_count} to {max_variants}")

        # Deduplication / empty check
        raw_nonempty = [v for v in entry.get("variants", []) if str(v.get("query", "")).strip()]
        if raw_count != len(raw_nonempty):
            notes.append(f"removed {raw_count - len(raw_nonempty)} empty variant(s)")
        if len(raw_nonempty) != len({v.get("query", "").strip() for v in raw_nonempty}):
            notes.append("removed duplicate variant(s)")

        # Mixed-language heuristic
        category = entry.get("category", "")
        if category == "mixed-language":
            # Use the original query AND explicit variants when checking labels.
            check_texts = [original] + [v.get("query", "") for v in entry.get("variants", [])]
            has_it = any("it" == v.get("language") for v in entry.get("variants", [])) or any(_looks_italian(t) for t in check_texts)
            has_en = any("en" == v.get("language") for v in entry.get("variants", [])) or any(_looks_english(t) for t in check_texts)
            if not (has_it and has_en):
                ok = False
                notes.append("mixed-language entry must expose both Italian and English signals")

        # Write-semantics check
        if category == "write-semantics":
            if not original:
                ok = False
                notes.append("write-semantics entry must keep original query")

        checks.append({
            "index": i,
            "query": original,
            "category": category,
            "valid_variant_count": len(variants),
            "raw_variant_count": raw_count,
            "ok": ok,
            "notes": notes,
        })

    return {
        "mode": "contract",
        "golden_set": str(golden_path),
        "query_count": len(queries),
        "categories": dict(categories),
        "max_variants": max_variants,
        "checks": checks,
        "all_ok": all(c["ok"] for c in checks),
    }


def _looks_italian(text: str) -> bool:
    """Very cheap heuristic: common Italian function words and markers."""
    markers = {
        "il", "la", "lo", "nel", "del", "con", "per", "che", "cosa", "come",
        "sono", "è", "spiegato", "italiano", "modello", "soglia", "adattiva",
    }
    lowered = text.lower()
    return any(m in lowered for m in markers)


def _looks_english(text: str) -> bool:
    """Very cheap heuristic: common English function words and markers."""
    markers = {
        "the", "what", "how", "does", "is", "are", "explain", "overview",
        "gating", "sparse", "activation", "baby", "dragon", "hatchling",
        "transformer", "comparison",
    }
    lowered = text.lower()
    return any(m in lowered for m in markers)


def run_baseline_audit(golden_path: Path, config_path: str | None = None) -> dict:
    """Run the Issue #19 golden set through the existing ablation runner.

    This measures the current single-query baseline on the new categories
    and records which contract requirements are already satisfied.
    """
    queries = _load_golden_set(golden_path)
    result = run_eval({}, queries, config_path=config_path, golden_set_path=golden_path)

    # Add contract metadata on top of the cold metrics
    cold = result.get("cold")
    per_query = cold.per_query if cold else []
    categories = defaultdict(list)
    for q in per_query:
        categories[q.get("category", "unknown")].append(q)

    category_table = []
    for cat in sorted(categories):
        qs = categories[cat]
        agg = aggregate_metrics(qs)
        category_table.append({
            "category": cat,
            "queries": len(qs),
            "mrr": agg.get("mrr", {}).get("mean", 0.0),
            "recall@5": agg.get("recall@5", {}).get("mean", 0.0),
            "precision@5": agg.get("precision@5", {}).get("mean", 0.0),
            "ndcg@5": agg.get("ndcg@5", {}).get("mean", 0.0),
        })

    result["issue19"] = {
        "mode": "baseline",
        "golden_set": str(golden_path),
        "category_breakdown": category_table,
    }
    return result


def _print_contract_report(report: dict) -> None:
    print(f"\nIssue #19 contract audit — {report['query_count']} queries")
    print(f"Categories: {report['categories']}")
    print(f"Max variants: {report['max_variants']}")
    print(f"All checks OK: {report['all_ok']}")
    print("")
    failed = [c for c in report["checks"] if not c["ok"]]
    if failed:
        print("Failed checks:")
        for c in failed:
            print(f"  [{c['category']}] {c['query'][:60]!r}")
            for n in c["notes"]:
                print(f"    - {n}")
    else:
        print("No contract violations found.")


def _print_baseline_report(report: dict) -> None:
    cold = report.get("cold")
    print(f"\nIssue #19 baseline retrieval — {report['query_count']} queries")
    if cold:
        print(
            f"MRR={cold.mrr:.4f}  R@5={cold.recall_at_5:.4f}  "
            f"P@5={cold.precision_at_5:.4f}  NDCG@5={cold.ndcg_at_5:.4f}  "
            f"lat={cold.mean_latency_ms:.0f}ms"
        )
    print("\nPer-category:")
    for row in report["issue19"]["category_breakdown"]:
        print(
            f"  {row['category']:24} n={row['queries']:2}  "
            f"MRR={row['mrr']:.4f}  R@5={row['recall@5']:.4f}  "
            f"P@5={row['precision@5']:.4f}  NDCG@5={row['ndcg@5']:.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Issue #19 compatibility audit")
    parser.add_argument(
        "--mode",
        choices=["contract", "baseline", "both"],
        default="both",
        help="Run contract audit, baseline retrieval, or both",
    )
    parser.add_argument(
        "--golden",
        default="benchmarks/golden_set_issue19.yaml",
        help="Path to the Issue #19 golden set",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Config path to load (defaults to bdh-config.local.yaml if present)",
    )
    parser.add_argument(
        "--max-variants",
        type=int,
        default=3,
        help="Maximum number of query variants the contract allows",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON file to write the full report",
    )
    args = parser.parse_args()

    golden_path = Path(args.golden)
    if not golden_path.is_absolute():
        golden_path = Path.cwd() / golden_path

    config_path = args.config
    if config_path is None and Path("bdh-config.local.yaml").exists():
        config_path = "bdh-config.local.yaml"

    full_report: dict[str, Any] = {"golden_set": str(golden_path)}

    if args.mode in ("contract", "both"):
        contract_report = audit_contract(golden_path, max_variants=args.max_variants)
        _print_contract_report(contract_report)
        full_report["contract"] = contract_report

    if args.mode in ("baseline", "both"):
        baseline_report = run_baseline_audit(golden_path, config_path=config_path)
        _print_baseline_report(baseline_report)
        # run_eval returns Metrics dataclasses that are not JSON-serializable.
        # Convert the cold/warm metrics before saving.
        from benchmarks.ablation import _serialize
        full_report["baseline"] = _serialize(baseline_report)

    if args.output:
        out_path = Path(args.output)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(full_report, f, indent=2, ensure_ascii=False)
        print(f"\nFull report written to {out_path}")


if __name__ == "__main__":
    main()
