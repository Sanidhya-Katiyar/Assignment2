"""
src/visualization/analysis_stats.py
─────────────────────────────────────
Paired statistical comparisons for transfer-learning experiment results.

For each valid (model_a, model_b, mode) triple the module computes:

* Paired t-test (t-statistic, p-value)
* Wilcoxon signed-rank test (W-statistic, p-value)
* Cohen's d effect size
* Bootstrap 95 % CI on the mean difference (1,000 resamples)

Results are written to ``outputs/stats/stat_tests.csv``.

Public API
----------
``load_best_results``       – read ``outputs/best_results.csv`` into a tidy DataFrame-like list.
``find_valid_comparisons``  – detect pairs with matching seeds.
``paired_comparison``       – run both tests + effect size for one pair.
``run_all_comparisons``     – auto-detect and run all valid pairs.
``save_stat_csv``           – persist results to CSV.
``bootstrap_ci``            – non-parametric 95 % CI for mean difference.
``cohens_d``                – Cohen's d for paired samples.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ExperimentResult:
    """One row from ``outputs/best_results.csv``."""
    model:            str
    mode:             str
    seed:             int
    best_val_acc:     float
    epoch_of_best:    int
    runtime_seconds:  float
    checkpoint_path:  str


@dataclass
class StatTestResult:
    """Results of a paired statistical comparison between two models."""
    metric:           str
    model_a:          str
    model_b:          str
    mode:             str
    n_pairs:          int
    mean_a:           float
    mean_b:           float
    mean_diff:        float
    std_diff:         float
    t_stat:           float
    p_value_ttest:    float
    w_stat:           float
    p_value_wilcoxon: float
    cohens_d:         float
    ci_low:           float
    ci_high:          float
    significant_005:  bool = field(init=False)

    def __post_init__(self):
        self.significant_005 = min(self.p_value_ttest, self.p_value_wilcoxon) < 0.05


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_best_results(csv_path: Path | str) -> List[ExperimentResult]:
    """
    Load ``outputs/best_results.csv`` produced by
    :func:`~src.train.utils_checkpoint.append_best_results`.

    Args:
        csv_path: Path to the CSV.

    Returns:
        List of :class:`ExperimentResult`.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(
            f"best_results.csv not found at '{p}'.  "
            "Run training experiments first."
        )
    results: List[ExperimentResult] = []
    with p.open() as fh:
        for row in csv.DictReader(fh):
            try:
                results.append(ExperimentResult(
                    model           = row["model"],
                    mode            = row["mode"],
                    seed            = int(row["seed"]),
                    best_val_acc    = float(row["best_val_acc"]),
                    epoch_of_best   = int(row["epoch_of_best"]),
                    runtime_seconds = float(row["runtime_seconds"]),
                    checkpoint_path = row["checkpoint_path"],
                ))
            except (KeyError, ValueError):
                continue   # skip malformed rows
    return results


def load_robustness_results(csv_path: Path | str) -> List[Dict[str, Any]]:
    """
    Load a robustness results CSV (if present) and return as a list of dicts.

    Returns an empty list if the file does not exist (robustness eval optional).
    """
    p = Path(csv_path)
    if not p.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with p.open() as fh:
        for row in csv.DictReader(fh):
            rows.append(dict(row))
    return rows


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute Cohen's d for two paired samples.

    Uses the pooled standard deviation of the differences.

    Args:
        a: Scores for condition A.
        b: Scores for condition B.

    Returns:
        Cohen's d (signed — positive means A > B).
    """
    diff = a - b
    sd   = diff.std(ddof=1)
    return float(diff.mean() / sd) if sd > 0 else 0.0


def bootstrap_ci(
    a:         np.ndarray,
    b:         np.ndarray,
    n_boot:    int   = 1000,
    ci:        float = 0.95,
    seed:      int   = 0,
) -> Tuple[float, float]:
    """
    Non-parametric bootstrap confidence interval for the mean of (a − b).

    Args:
        a:      Scores for condition A.
        b:      Scores for condition B.
        n_boot: Number of bootstrap resamples (default 1,000).
        ci:     Confidence level (default 0.95).
        seed:   Random seed.

    Returns:
        ``(ci_low, ci_high)`` percentile bootstrap interval.
    """
    rng   = np.random.default_rng(seed)
    diffs = a - b
    n     = len(diffs)
    boot_means = np.array(
        [rng.choice(diffs, size=n, replace=True).mean() for _ in range(n_boot)]
    )
    alpha    = 1.0 - ci
    ci_low   = float(np.percentile(boot_means, 100 * alpha / 2))
    ci_high  = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return ci_low, ci_high


def _ttest_paired(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    """Paired t-test — returns (t_stat, p_value)."""
    try:
        from scipy.stats import ttest_rel  # type: ignore
        t, p = ttest_rel(a, b)
        return float(t), float(p)
    except ImportError:
        pass
    # Manual implementation (no scipy)
    diff  = a - b
    n     = len(diff)
    mean  = diff.mean()
    se    = diff.std(ddof=1) / math.sqrt(n)
    if se == 0:
        return 0.0, 1.0
    t_stat = mean / se
    # Approximate two-tailed p using normal approximation for large n
    from math import erfc, sqrt
    z = abs(t_stat) / sqrt(1 + 1 / max(n - 1, 1))
    p_approx = float(erfc(abs(t_stat) / sqrt(2)))
    return t_stat, p_approx


def _wilcoxon(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    """Wilcoxon signed-rank test — returns (W_stat, p_value)."""
    try:
        from scipy.stats import wilcoxon  # type: ignore
        diff = a - b
        if np.all(diff == 0):
            return 0.0, 1.0
        w, p = wilcoxon(diff)
        return float(w), float(p)
    except ImportError:
        pass
    # Fallback: return NaN when scipy absent
    return float("nan"), float("nan")


# ---------------------------------------------------------------------------
# Comparison logic
# ---------------------------------------------------------------------------

def find_valid_comparisons(
    results: List[ExperimentResult],
) -> List[Tuple[str, str, str, List[int]]]:
    """
    Auto-detect model pairs that share the same (mode, seeds).

    Returns:
        List of ``(model_a, model_b, mode, shared_seeds)`` tuples.
    """
    # Group: mode → model → list of seeds
    groups: Dict[str, Dict[str, List[int]]] = {}
    for r in results:
        groups.setdefault(r.mode, {}).setdefault(r.model, []).append(r.seed)

    comparisons: List[Tuple[str, str, str, List[int]]] = []
    for mode, model_seeds in groups.items():
        models = sorted(model_seeds.keys())
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                ma, mb = models[i], models[j]
                shared = sorted(set(model_seeds[ma]) & set(model_seeds[mb]))
                if len(shared) >= 2:
                    comparisons.append((ma, mb, mode, shared))

    return comparisons


def paired_comparison(
    results:  List[ExperimentResult],
    model_a:  str,
    model_b:  str,
    mode:     str,
    seeds:    Optional[List[int]] = None,
    metric:   str = "best_val_acc",
) -> Optional[StatTestResult]:
    """
    Run paired statistical tests comparing *model_a* vs *model_b* on *mode*.

    Args:
        results:  All experiment results loaded from best_results.csv.
        model_a:  First model name.
        model_b:  Second model name.
        mode:     Experiment mode (e.g. ``"linearprobe"``).
        seeds:    Seeds to compare; if ``None``, all shared seeds are used.
        metric:   Column name to compare (default ``"best_val_acc"``).

    Returns:
        :class:`StatTestResult` or ``None`` if insufficient data.
    """
    def _get_acc(model: str, seed: int) -> Optional[float]:
        for r in results:
            if r.model == model and r.mode == mode and r.seed == seed:
                return getattr(r, metric, None)
        return None

    if seeds is None:
        # Detect shared seeds
        seeds_a = {r.seed for r in results if r.model == model_a and r.mode == mode}
        seeds_b = {r.seed for r in results if r.model == model_b and r.mode == mode}
        seeds   = sorted(seeds_a & seeds_b)

    if len(seeds) < 2:
        return None

    vals_a = np.array([_get_acc(model_a, s) for s in seeds], dtype=float)
    vals_b = np.array([_get_acc(model_b, s) for s in seeds], dtype=float)

    # Drop pairs where either value is missing
    valid = ~(np.isnan(vals_a) | np.isnan(vals_b))
    vals_a, vals_b = vals_a[valid], vals_b[valid]
    if len(vals_a) < 2:
        return None

    t_stat, p_ttest   = _ttest_paired(vals_a, vals_b)
    w_stat, p_wilcoxon= _wilcoxon(vals_a, vals_b)
    cd                = cohens_d(vals_a, vals_b)
    ci_low, ci_high   = bootstrap_ci(vals_a, vals_b)
    diff              = vals_a - vals_b

    return StatTestResult(
        metric           = metric,
        model_a          = model_a,
        model_b          = model_b,
        mode             = mode,
        n_pairs          = len(vals_a),
        mean_a           = float(vals_a.mean()),
        mean_b           = float(vals_b.mean()),
        mean_diff        = float(diff.mean()),
        std_diff         = float(diff.std(ddof=1)),
        t_stat           = t_stat,
        p_value_ttest    = p_ttest,
        w_stat           = w_stat,
        p_value_wilcoxon = p_wilcoxon,
        cohens_d         = cd,
        ci_low           = ci_low,
        ci_high          = ci_high,
    )


def run_all_comparisons(
    results: List[ExperimentResult],
    metric:  str = "best_val_acc",
) -> List[StatTestResult]:
    """
    Auto-detect valid pairs and run all paired comparisons.

    Args:
        results: All experiment results from best_results.csv.
        metric:  Metric to compare.

    Returns:
        List of :class:`StatTestResult`.
    """
    comparisons = find_valid_comparisons(results)
    stat_results: List[StatTestResult] = []
    for ma, mb, mode, seeds in comparisons:
        sr = paired_comparison(results, ma, mb, mode, seeds=seeds, metric=metric)
        if sr is not None:
            stat_results.append(sr)
    return stat_results


# ---------------------------------------------------------------------------
# CSV persistence
# ---------------------------------------------------------------------------

_STAT_CSV_COLUMNS = [
    "metric", "model_a", "model_b", "mode", "n_pairs",
    "mean_a", "mean_b", "mean_diff", "std_diff",
    "t_stat", "p_value_ttest",
    "w_stat", "p_value_wilcoxon",
    "cohens_d", "ci_low_95", "ci_high_95", "significant_005",
]


def save_stat_csv(
    stat_results: List[StatTestResult],
    output_path:  Path | str,
    append:       bool = False,
) -> Path:
    """
    Write statistical test results to a CSV file.

    Args:
        stat_results: List of :class:`StatTestResult`.
        output_path:  Destination path.
        append:       If ``True``, append rows (default: overwrite).

    Returns:
        Path of the written file.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    write_header = (not out.exists()) or (not append)
    mode_str     = "a" if append else "w"

    with out.open(mode_str, newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_STAT_CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        for r in stat_results:
            writer.writerow({
                "metric":           r.metric,
                "model_a":          r.model_a,
                "model_b":          r.model_b,
                "mode":             r.mode,
                "n_pairs":          r.n_pairs,
                "mean_a":           f"{r.mean_a:.4f}",
                "mean_b":           f"{r.mean_b:.4f}",
                "mean_diff":        f"{r.mean_diff:.4f}",
                "std_diff":         f"{r.std_diff:.4f}",
                "t_stat":           f"{r.t_stat:.4f}",
                "p_value_ttest":    f"{r.p_value_ttest:.6f}",
                "w_stat":           f"{r.w_stat:.4f}",
                "p_value_wilcoxon": f"{r.p_value_wilcoxon:.6f}",
                "cohens_d":         f"{r.cohens_d:.4f}",
                "ci_low_95":        f"{r.ci_low:.4f}",
                "ci_high_95":       f"{r.ci_high:.4f}",
                "significant_005":  r.significant_005,
            })
    return out
