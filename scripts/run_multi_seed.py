#!/usr/bin/env python3
"""
scripts/run_multi_seed.py
──────────────────────────
Run the same experiment (linear probe or fine-tune scheme) across multiple
random seeds and aggregate results.

Optionally, compare two models/modes using a paired statistical test
(Wilcoxon signed-rank or paired t-test).

Usage
-----
    # Linear probe, 3 seeds:
    python scripts/run_multi_seed.py \\
        --config configs/baseline.yaml \\
        --model resnet50 \\
        --mode linearprobe \\
        --seeds 42 43 44

    # Compare two models over 5 seeds:
    python scripts/run_multi_seed.py \\
        --config configs/baseline.yaml \\
        --model resnet50 efficientnet_b0 \\
        --mode linearprobe \\
        --seeds 42 43 44 0 1 \\
        --compare

Outputs (in outputs/logs/)
--------------------------
    <model>_<mode>_multi_seed_summary.csv
    comparison_<model_a>_vs_<model_b>_<mode>.csv  (if --compare)
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run multi-seed experiments and aggregate results."
    )
    p.add_argument("--config",    required=True,   help="Path to YAML config.")
    p.add_argument("--model",     nargs="+",        required=True,
                   help="One or two model names (for --compare provide two).")
    p.add_argument("--mode",      default="linearprobe",
                   choices=["linearprobe", "lastblock", "selective", "finetune"],
                   help="Experiment mode.")
    p.add_argument("--seeds",     nargs="+", type=int, required=True,
                   help="List of random seeds.")
    p.add_argument("--epochs",    type=int,   default=None)
    p.add_argument("--lr",        type=float, default=None)
    p.add_argument("--batch-size",type=int,   default=None)
    p.add_argument("--amp",       action="store_true")
    p.add_argument("--compare",   action="store_true",
                   help="Run paired statistical test when 2 models provided.")
    p.add_argument("--scheme",    default="last-block",
                   choices=["last-block", "selective", "full"],
                   help="Unfreeze scheme (only used for finetune modes).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Single-seed dispatcher
# ---------------------------------------------------------------------------

_MODE_SCRIPT = {
    "linearprobe": "scripts/train_linear_probe.py",
    "lastblock":   "scripts/finetune_ablation.py",
    "selective":   "scripts/finetune_ablation.py",
    "finetune":    "scripts/finetune_ablation.py",
}

_MODE_SCHEME = {
    "lastblock":  "last-block",
    "selective":  "selective",
    "finetune":   "full",
}


def _build_cmd(
    model:      str,
    mode:       str,
    seed:       int,
    config:     str,
    extra_args: List[str],
) -> List[str]:
    """Construct the subprocess command for one seed."""
    script = str(_PROJECT_ROOT / _MODE_SCRIPT[mode])
    cmd    = [sys.executable, script, "--config", config,
              "--model", model, "--seed", str(seed)]
    if mode != "linearprobe":
        cmd += ["--scheme", _MODE_SCHEME[mode]]
    cmd += extra_args
    return cmd


def run_one_seed(
    model:      str,
    mode:       str,
    seed:       int,
    config:     str,
    extra_args: List[str],
) -> Optional[float]:
    """
    Invoke the appropriate training script for one seed via subprocess.

    Returns the best_val_acc parsed from outputs/best_results.csv, or
    ``None`` if the run failed.
    """
    cmd = _build_cmd(model, mode, seed, config, extra_args)
    print(f"\n  → seed={seed}: {' '.join(cmd)}")
    ret = subprocess.run(cmd, capture_output=False)
    if ret.returncode != 0:
        print(f"  [WARN] seed={seed} exited with code {ret.returncode}")
        return None

    # Parse best_val_acc from outputs/best_results.csv
    results_csv = _PROJECT_ROOT / "outputs" / "best_results.csv"
    if not results_csv.exists():
        return None

    best_acc: Optional[float] = None
    with results_csv.open() as fh:
        for row in csv.DictReader(fh):
            if (row.get("model") == model
                    and row.get("mode", "").replace("linearprobe", "linearprobe") == mode.replace("-", "")
                    and int(row.get("seed", -1)) == seed):
                best_acc = float(row["best_val_acc"])
    return best_acc


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def summarise(accs: List[Optional[float]]) -> Dict[str, float]:
    valid = [a for a in accs if a is not None]
    if not valid:
        return {"mean": float("nan"), "std": float("nan"),
                "min": float("nan"), "max": float("nan"), "n": 0}
    arr = np.array(valid)
    return {
        "mean": float(arr.mean()),
        "std":  float(arr.std(ddof=1) if len(arr) > 1 else 0.0),
        "min":  float(arr.min()),
        "max":  float(arr.max()),
        "n":    len(valid),
    }


def save_summary_csv(
    output_path: Path,
    model:       str,
    mode:        str,
    seeds:       List[int],
    accs:        List[Optional[float]],
) -> None:
    """Write per-seed values + aggregate stats to a CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats = summarise(accs)

    with output_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["model", "mode", "seed", "best_val_acc"])
        for s, a in zip(seeds, accs):
            writer.writerow([model, mode, s, f"{a:.6f}" if a is not None else "FAILED"])

        writer.writerow([])
        writer.writerow(["model", "mode", "metric", "value"])
        writer.writerow([model, mode, "mean",  f"{stats['mean']:.4f}"])
        writer.writerow([model, mode, "std",   f"{stats['std']:.4f}"])
        writer.writerow([model, mode, "min",   f"{stats['min']:.4f}"])
        writer.writerow([model, mode, "max",   f"{stats['max']:.4f}"])
        writer.writerow([model, mode, "n_runs",stats["n"]])

    print(f"\n  Summary saved → {output_path}")
    print(f"  {model} | {mode}: "
          f"mean={stats['mean']:.2f}% ± {stats['std']:.2f}% "
          f"(n={stats['n']})")


# ---------------------------------------------------------------------------
# Statistical comparison
# ---------------------------------------------------------------------------

def compare_two_models(
    model_a:   str,
    model_b:   str,
    mode:      str,
    accs_a:    List[Optional[float]],
    accs_b:    List[Optional[float]],
    output_path: Path,
) -> None:
    """Paired Wilcoxon / t-test between two sets of seed results."""
    valid_pairs = [
        (a, b) for a, b in zip(accs_a, accs_b)
        if a is not None and b is not None
    ]
    if len(valid_pairs) < 2:
        print("  [WARN] Not enough valid pairs for statistical test.")
        return

    arr_a = np.array([p[0] for p in valid_pairs])
    arr_b = np.array([p[1] for p in valid_pairs])
    diff  = arr_a - arr_b

    try:
        from scipy.stats import wilcoxon, ttest_rel
        _, p_wilcoxon = wilcoxon(diff)
        _, p_ttest    = ttest_rel(arr_a, arr_b)
    except ImportError:
        print("  [info] scipy not installed — skipping statistical tests.")
        p_wilcoxon = p_ttest = float("nan")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["model_a", "model_b", "mode", "n_pairs",
                         "mean_a", "mean_b", "mean_diff",
                         "p_wilcoxon", "p_ttest"])
        writer.writerow([
            model_a, model_b, mode, len(valid_pairs),
            f"{arr_a.mean():.4f}", f"{arr_b.mean():.4f}",
            f"{diff.mean():.4f}",
            f"{p_wilcoxon:.6f}", f"{p_ttest:.6f}",
        ])

    print(f"\n  Comparison ({model_a} vs {model_b}, {mode}):")
    print(f"    mean diff  = {diff.mean():.2f}%")
    print(f"    p_wilcoxon = {p_wilcoxon:.4f}")
    print(f"    p_ttest    = {p_ttest:.4f}")
    print(f"    Saved → {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    extra_args: List[str] = []
    if args.epochs:
        extra_args += ["--epochs", str(args.epochs)]
    if args.lr:
        extra_args += ["--lr", str(args.lr)]
    if args.batch_size:
        extra_args += ["--batch-size", str(args.batch_size)]
    if args.amp:
        extra_args += ["--amp"]

    log_dir = _PROJECT_ROOT / "outputs" / "logs"

    all_results: Dict[str, List[Optional[float]]] = {}

    for model_name in args.model:
        print(f"\n{'='*65}")
        print(f"  Model: {model_name} | Mode: {args.mode} | Seeds: {args.seeds}")
        print(f"{'='*65}")

        accs: List[Optional[float]] = []
        for seed in args.seeds:
            acc = run_one_seed(
                model      = model_name,
                mode       = args.mode,
                seed       = seed,
                config     = args.config,
                extra_args = extra_args,
            )
            accs.append(acc)

        all_results[model_name] = accs

        summary_path = log_dir / f"{model_name}_{args.mode}_multi_seed_summary.csv"
        save_summary_csv(summary_path, model_name, args.mode, args.seeds, accs)

    # ── Optional comparison ────────────────────────────────────────────
    if args.compare and len(args.model) == 2:
        m_a, m_b = args.model
        cmp_path = log_dir / f"comparison_{m_a}_vs_{m_b}_{args.mode}.csv"
        compare_two_models(
            model_a     = m_a,
            model_b     = m_b,
            mode        = args.mode,
            accs_a      = all_results[m_a],
            accs_b      = all_results[m_b],
            output_path = cmp_path,
        )
    elif args.compare:
        print("  [info] --compare requires exactly 2 models.")

    print("\nMulti-seed run complete.")


if __name__ == "__main__":
    main()
