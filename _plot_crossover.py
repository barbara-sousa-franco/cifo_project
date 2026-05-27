"""Generate crossover plots using the SAME utils.py functions Catarina
used for the mutation section, so the visual style matches exactly.

Outputs three plots (same as the mutation ones):
  run_artifacts/crossover_summary_plot.png       (all 5 crossovers, conv + boxplot)
  run_artifacts/crossover_best_grid.png          (target + 5 best individuals)
  run_artifacts/crossover_uniform_vs_adaptive.png (head-to-head of the two winners)
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from operators import (
    uniform_crossover,
    kpoint_crossover,
    reduced_surrogate_crossover,
    shuffle_crossover,
    adaptive_crossover_schedule,
)
from utils import plot_experiment_summary, plot_best_individuals

ART = Path("run_artifacts")
CHECKPOINT = ART / "crossover_checkpoint.json"

# Same config naming/order as _run_crossover.py and main.ipynb
CROSSOVER_CONFIGS = [
    {"name": "Uniform",          "fn": uniform_crossover},
    {"name": "KPoint",           "fn": kpoint_crossover},
    {"name": "ReducedSurrogate", "fn": reduced_surrogate_crossover},
    {"name": "Shuffle",          "fn": shuffle_crossover},
    {"name": "AdaptiveXO",       "fn": adaptive_crossover_schedule},
]


class _LoadedInd:
    """Mirror Catarina's wrapper in main.ipynb cell 14 so the loaded PNGs
    quack like Individual objects (.render() and .fitness())."""
    def __init__(self, png_path, fitness_val):
        self._img = Image.open(png_path).convert("RGB")
        self._fit = float(fitness_val)

    def render(self):
        return self._img

    def fitness(self):
        return self._fit


def main() -> None:
    ckpt = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    names = [c["name"] for c in CROSSOVER_CONFIGS]

    # Build run_experiment()-compatible structures (same as Catarina's cell 14).
    all_results_xo = []
    for name in names:
        for r in ckpt[name]:
            all_results_xo.append({
                "crossover_type": name,
                "run":            r["run"],
                "best_fitness":   r["fitness"],
                "time_seconds":   r["time_seconds"],
            })

    all_curves_xo = {}
    for name in names:
        curves = []
        for r in sorted(ckpt[name], key=lambda x: x["run"]):
            curves.append(np.load(ART / r["curve_file"]).tolist())
        all_curves_xo[name] = curves

    best_inds_xo = {
        name: _LoadedInd(
            ART / f"crossover_best_{name}.png",
            min(ckpt[name], key=lambda r: r["fitness"])["fitness"],
        )
        for name in names
    }

    df_xo = pd.DataFrame(all_results_xo)

    # ---------- Plot 1: 5 crossovers (convergence + boxplot) ----------
    plot_experiment_summary(
        all_curves   = all_curves_xo,
        df           = df_xo,
        configs      = CROSSOVER_CONFIGS,
        config_key   = "crossover_type",
        title_prefix = "Crossover",
    )
    plt.savefig(ART / "crossover_summary_plot.png", dpi=150, bbox_inches="tight")
    print("Saved crossover_summary_plot.png")
    plt.close("all")

    # ---------- Plot 2: best individuals grid ----------
    target_img = Image.open("data/girl_pearl_earing.png").convert("RGB")
    plot_best_individuals(
        best_inds    = best_inds_xo,
        configs      = CROSSOVER_CONFIGS,
        target_img   = target_img,
        title_prefix = "Crossover",
    )
    plt.savefig(ART / "crossover_best_grid.png", dpi=150, bbox_inches="tight")
    print("Saved crossover_best_grid.png")
    plt.close("all")

    # ---------- Plot 3: Uniform vs AdaptiveXO head-to-head ----------
    # Mirror of Catarina's cell 21 (Gaussian vs Adaptive comparison).
    all_curves_xo_comparison = {
        "Uniform":  all_curves_xo["Uniform"],
        "Adaptive": all_curves_xo["AdaptiveXO"],
    }
    all_results_xo_comparison = (
        [r for r in all_results_xo if r["crossover_type"] == "Uniform"] +
        [{**r, "crossover_type": "Adaptive"}
         for r in all_results_xo if r["crossover_type"] == "AdaptiveXO"]
    )
    df_xo_comparison = pd.DataFrame(all_results_xo_comparison)

    COMPARISON_XO_CONFIGS = [
        {"name": "Uniform",  "fn": uniform_crossover},
        {"name": "Adaptive", "fn": adaptive_crossover_schedule},
    ]

    plot_experiment_summary(
        all_curves   = all_curves_xo_comparison,
        df           = df_xo_comparison,
        configs      = COMPARISON_XO_CONFIGS,
        config_key   = "crossover_type",
        title_prefix = "Uniform vs Adaptive Crossover",
    )
    plt.savefig(ART / "crossover_uniform_vs_adaptive.png", dpi=150, bbox_inches="tight")
    print("Saved crossover_uniform_vs_adaptive.png")
    plt.close("all")


if __name__ == "__main__":
    main()
