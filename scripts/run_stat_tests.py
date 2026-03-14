#!/usr/bin/env python3
"""
scripts/run_stat_tests.py
──────────────────────────
Run paired statistical tests comparing model performance across seeds.

Usage
-----
    # Auto-detect all valid pairs from best_results.csv:
    python scripts/run_stat_tests.py --config configs/baseline.yaml --auto

    # Explicit comparison — two models on the same mode:
    python scripts/run_stat_tests.py \\
        --config configs/baseline.yaml \\
        --model-a resnet50 \\
        --model-b efficientnet_b0 \\
        --mode linearprobe

    # Multiple explicit comparisons with seeds:
    python scripts/run_stat_tests.py \\
        --config configs/baseline.yaml \\
        --model-a resnet50 --model-b convnext_tiny \\
        --mode linearprobe --seeds 42 43 44

Output
------
    outputs/stats/stat_tests.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils.config import load_config
from src.visualization.analysis_stats import (
    StatTestResult,
    load_best_results,
    paired_comparison,
    run_all_comparisons,
    save_stat_csv,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Paired statistical tests for model comparisons."
    )
    p.add_argument("--config",   required=True)
    p.add_argument("--auto",     action="store_true",
                   help="Auto-detect all valid model pairs and run all comparisons.")
    p.add_argument("--model-a",  default=None, help="First model name.")
    p.add_argument("--model-b",  default=None, help="Second model name.")
    p.add_argument("--mode",     default="linearprobe",
                   help="Experiment mode to compare (default: linearprobe).")
    p.add_argument("--seeds",    nargs="*", type=int, default=None,
                   help="Specific seeds to compare (auto-detected if omitted).")
    p.add_argument("--metric",   default="best_val_acc",
                   help="Metric column to compare (default: best_val_acc).")
    p.add_argument("--outdir",   default="outputs",
                   help="Root output directory.")
    p.add_argument("--append",   action="store_true",
                   help="Append to existing stat_tests.csv instead of overwriting.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args    = parse_args()
    config  = load_config(args.config)
    outdir  = _PROJECT_ROOT / args.outdir
    csv_path = outdir / "stats" / "stat_tests.csv"

    results_csv = outdir / "best_results.csv"
    print("=" * 60)
    print("  Statistical Tests — AID Transfer Learning")
    print("=" * 60)

    try:
        exp_results = load_best_results(results_csv)
    except FileNotFoundError as exc:
        print(f"\n  ERROR: {exc}")
        sys.exit(1)

    print(f"  Loaded {len(exp_results)} experiment rows from {results_csv.name}")

    stat_results: list[StatTestResult] = []

    if args.auto:
        print("\n  Auto mode: detecting all valid model pairs …")
        stat_results = run_all_comparisons(exp_results, metric=args.metric)
        print(f"  Found {len(stat_results)} valid comparisons.")
    elif args.model_a and args.model_b:
        print(f"\n  Comparing {args.model_a} vs {args.model_b} on {args.mode} …")
        sr = paired_comparison(
            results  = exp_results,
            model_a  = args.model_a,
            model_b  = args.model_b,
            mode     = args.mode,
            seeds    = args.seeds,
            metric   = args.metric,
        )
        if sr is None:
            print("  Not enough matched pairs for a valid test.")
        else:
            stat_results = [sr]
    else:
        print(
            "\n  Provide --auto or --model-a + --model-b to run comparisons.\n"
            "  Use --help for usage details."
        )
        sys.exit(0)

    if stat_results:
        saved = save_stat_csv(stat_results, csv_path, append=args.append)
        print(f"\n  Results saved → {saved}")
        _print_table(stat_results)
    else:
        print("\n  No results to save.")

    print("=" * 60)


def _print_table(results: list[StatTestResult]) -> None:
    """Print a compact summary table to stdout."""
    print()
    hdr = (f"{'A vs B':<40} {'mode':<14} {'Δmean':>8} "
           f"{'p_t':>8} {'p_w':>8} {'d':>6} {'sig':>5}")
    print(hdr)
    print("─" * len(hdr))
    for r in results:
        pair = f"{r.model_a} vs {r.model_b}"
        sig  = "✓" if r.significant_005 else ""
        print(
            f"  {pair:<38} {r.mode:<14} {r.mean_diff:+8.3f} "
            f"{r.p_value_ttest:8.4f} {r.p_value_wilcoxon:8.4f} "
            f"{r.cohens_d:6.3f} {sig:>5}"
        )
    print()


if __name__ == "__main__":
    main()
