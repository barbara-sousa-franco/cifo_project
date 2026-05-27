"""Statistical tests on the checkpoint JSONs produced by _run_experiment.py.

Design rationale
----------------
The experimental design is One-Factor-At-A-Time (OFAT): each configuration
is an independent treatment. The primary analysis therefore uses
unpaired non-parametric tests:

    - Mann-Whitney U (pairwise)
    - Kruskal-Wallis (omnibus)

However, runs across configurations share the same RNG seed structure
(`random.seed(run * SEED)` in _run_experiment.py:_run_one), so observation
i in config A and observation i in config B start from the same random
initial population. This correlation lets us additionally report paired
non-parametric tests as a secondary, more powerful analysis:

    - Wilcoxon signed-rank (pairwise paired)
    - Friedman chi-square (omnibus paired)

Both analyses are reported; the paired versions tend to give smaller
p-values when the seed-induced correlation is positive. If the two
analyses lead to the same conclusions, the result is robust to the
choice of test.

Usage:
    python _stats_tests.py              # runs both mutation and crossover
    python _stats_tests.py mutation     # only mutation
    python _stats_tests.py crossover    # only crossover
    python _stats_tests.py probabilities size alpha diversity
"""
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats

ART = Path("run_artifacts")


def _stars(p: float) -> str:
    """Map a p-value to the conventional significance star annotation."""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def _build_paired_matrix(raw: dict) -> tuple[list[str], np.ndarray] | None:
    """Build a (n_runs, n_configs) matrix aligned by run number.

    Returns (config_names, matrix) where matrix[i, j] is the fitness of
    run (i+1) in config_names[j]. Returns None if configs do not share
    the same run-id set (cannot be paired).
    """
    cfg_runs = {
        name: {r["run"]: r["fitness"] for r in runs}
        for name, runs in raw.items() if runs
    }
    if not cfg_runs:
        return None
    common = set.intersection(*(set(d.keys()) for d in cfg_runs.values()))
    if len(common) < 3:
        return None
    runs_sorted = sorted(common)
    names = sorted(cfg_runs.keys())
    matrix = np.array(
        [[cfg_runs[n][r] for n in names] for r in runs_sorted],
        dtype=np.float64,
    )
    return names, matrix


def analyse(checkpoint_name: str, title: str) -> None:
    """Run the primary and secondary statistical analyses for one phase.

    Reads ``run_artifacts/{checkpoint_name}_checkpoint.json``, prints a
    per-config summary table, the pairwise Mann-Whitney U + Kruskal-Wallis
    (primary, unpaired) analysis, and the pairwise Wilcoxon signed-rank
    + Friedman (secondary, paired by seed) analysis. No return value;
    everything is printed to stdout.
    """
    path = ART / f"{checkpoint_name}_checkpoint.json"
    if not path.exists():
        print(f"[skip] {path} not found.")
        return

    raw = json.loads(path.read_text(encoding="utf-8"))
    data = {name: [r["fitness"] for r in runs] for name, runs in raw.items() if runs}

    if not data:
        print(f"[skip] {checkpoint_name}: no completed runs in checkpoint.")
        return

    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")

    # ---------------- Summary ----------------
    print("\nSummary (sorted by avg, lower = better):")
    names_sorted = sorted(data.keys(), key=lambda n: np.mean(data[n]))
    for n in names_sorted:
        f = data[n]
        print(f"  {n:25s}  avg={np.mean(f):7.3f}  std={np.std(f):5.3f}  "
              f"min={np.min(f):6.3f}  max={np.max(f):6.3f}  n={len(f)}")

    # ---------------- PRIMARY: unpaired tests (OFAT) ----------------
    print("\n--- PRIMARY ANALYSIS: unpaired (OFAT independent treatments) ---")

    print("\nPairwise Mann-Whitney U (two-sided):")
    print(f"  {'config_A':25s} vs {'config_B':25s}  {'U':>8s}  {'p-value':>10s}  sig")
    for a, b in combinations(names_sorted, 2):
        u, p = stats.mannwhitneyu(data[a], data[b], alternative="two-sided")
        print(f"  {a:25s} vs {b:25s}  {u:8.1f}  {p:10.4g}  {_stars(p)}")

    groups = [data[n] for n in names_sorted]
    h, p = stats.kruskal(*groups)
    verdict = "significant" if p < 0.05 else "NOT significant"
    print(f"\nKruskal-Wallis (all configs): H = {h:.3f}   p = {p:.4g}   ({verdict})")

    # ---------------- SECONDARY: paired tests (shared seed) ----------------
    paired = _build_paired_matrix(raw)
    if paired is None:
        print("\n[paired tests skipped: configs do not share a common run-id set]")
        return

    names_p, matrix = paired
    # Re-order matrix columns to match names_sorted
    name_to_col = {n: i for i, n in enumerate(names_p)}
    cols = [name_to_col[n] for n in names_sorted if n in name_to_col]
    matrix_sorted = matrix[:, cols]
    names_for_paired = [n for n in names_sorted if n in name_to_col]

    print(f"\n--- SECONDARY ANALYSIS: paired by seed (n={matrix_sorted.shape[0]} matched runs) ---")
    print("\nPairwise Wilcoxon signed-rank (two-sided):")
    print(f"  {'config_A':25s} vs {'config_B':25s}  {'W':>8s}  {'p-value':>10s}  sig")
    for (i, a), (j, b) in combinations(enumerate(names_for_paired), 2):
        diff = matrix_sorted[:, i] - matrix_sorted[:, j]
        if np.allclose(diff, 0):
            print(f"  {a:25s} vs {b:25s}  {'---':>8s}  {'identical':>10s}  ns")
            continue
        try:
            w, pw = stats.wilcoxon(matrix_sorted[:, i], matrix_sorted[:, j],
                                   alternative="two-sided", zero_method="wilcox")
        except ValueError as e:
            print(f"  {a:25s} vs {b:25s}  {'---':>8s}  {'n/a':>10s}  ({e})")
            continue
        print(f"  {a:25s} vs {b:25s}  {w:8.1f}  {pw:10.4g}  {_stars(pw)}")

    chi2, pf = stats.friedmanchisquare(*[matrix_sorted[:, i] for i in range(matrix_sorted.shape[1])])
    verdict_f = "significant" if pf < 0.05 else "NOT significant"
    print(f"\nFriedman chi-square (all configs, paired): chi2 = {chi2:.3f}   "
          f"p = {pf:.4g}   ({verdict_f})")


def main() -> None:
    """CLI entry point: dispatch ``analyse`` over the phases named on argv."""
    known = {
        "mutation":      "MUTATION TESTS (sec 5)",
        "crossover":     "CROSSOVER TESTS (sec 6)",
        "probabilities": "PROBABILITIES TESTS (sec 9)",
        "size":          "TRIANGLE SIZE TESTS (sec 7)",
        "alpha":         "ALPHA WINDOW TESTS (sec 8)",
        "diversity":     "DIVERSITY MECHANISM TESTS (sec 11)",
        "all_diversity": "ALL-DIVERSITY-COMBINED TEST (sec 11 bonus)",
        "validate_top3": "VALIDATE TOP-3 TESTS (refinement)",
    }
    targets = sys.argv[1:] if len(sys.argv) > 1 else ["mutation", "crossover"]
    for t in targets:
        if t in known:
            analyse(t, known[t])
        else:
            print(f"[skip] Unknown target: {t} (known: {sorted(known)})")


if __name__ == "__main__":
    main()
