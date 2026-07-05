"""BDH Benchmark Suite — entry point.

Runs retrieval benchmarks (hybrid vs vector vs BM25), generates results.json,
and produces BENCHMARK_REPORT.md.

Usage:
    python -m benchmarks.run_all
    python -m benchmarks.run_all --vault /path/to/vault
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.runner import run_benchmark
from benchmarks.report import generate_report


def main():
    import argparse
    parser = argparse.ArgumentParser(description="BDH Benchmark Suite")
    parser.add_argument("--vault", help="Path to vault root")
    parser.add_argument("--output-dir", default="benchmarks", help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    results_path = str(output_dir / "results.json")
    report_path = str(output_dir / "BENCHMARK_REPORT.md")

    # Run
    results = run_benchmark(vault_root=args.vault, output_path=results_path)

    # Report
    generate_report(results_path, report_path)

    print("\n✅ Benchmark complete!")
    print(f"   Results: {results_path}")
    print(f"   Report:  {report_path}")


if __name__ == "__main__":
    main()
