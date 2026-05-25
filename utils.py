# Utility functions 

import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time
from pathlib import Path

from itertools import combinations
from scipy import stats

from PIL import Image

from ga import genetic_algorithm
from solution import Individual


# FUNCTION FOR EXPERIMENTS OVER DIFFERENT CONFIGURATIONS

def run_experiment(configs, target_array, n_runs, max_gens, pop_size, seed, xo_prob, mut_prob, elitism, 
                   selection_algorithm, fixed_xo_fn=None, fixed_mut_fn=None, config_key="config", maximization=False, 
                   fitness_metric="rmse", individual_kwargs=None, **ga_kwargs):
    """
    Run a GA experiment over a list of operator configurations or probability pairs.

    Args:
        - configs (list[dict] | list[tuple]): Operator configs (each with 'name' and 'fn')
          or (mut_prob, xo_prob) tuples for probability grid experiments.
        - target_array (np.ndarray): Target image as an (H, W, 3) float32 array.
        - n_runs (int): Number of independent runs per configuration.
        - max_gens (int): Maximum number of generations per run.
        - pop_size (int): Number of individuals in the population.
        - seed (int): Base seed; run i uses i * seed for reproducibility.
        - xo_prob (float): Crossover probability (default when not varied).
        - mut_prob (float): Mutation probability (default when not varied).
        - elitism (bool): Whether to carry the best individual unchanged.
        - selection_algorithm (Callable): Selection function.
        - fixed_xo_fn (Callable | None): Fixed crossover; if None, taken from each config.
        - fixed_mut_fn (Callable | None): Fixed mutation; if None, taken from each config.
        - config_key (str): Key used in results dicts (e.g. 'mutation_type').
        - individual_kwargs (dict, optional): Extra kwargs for Individual constructor.
        - **ga_kwargs: Extra kwargs forwarded to genetic_algorithm.

    Returns:
        - all_results (list[dict]): One dict per run with config_key, 'run', 'best_fitness',
          'time_seconds'.
        - all_curves (dict): Maps config key → list of per-run fitness curves.
        - best_inds (dict): Maps config key → best Individual across all runs.
    """
    individual_kwargs = individual_kwargs or {}
    all_results = []
    all_curves  = {}
    best_inds   = {}

    # Normalise configs into (key, xo_fn, mut_fn, run_xo_prob, run_mut_prob)
    if configs and isinstance(configs[0], dict):
        normalized = [
            (
                c["name"],
                c["fn"] if fixed_xo_fn  is None else fixed_xo_fn,
                c["fn"] if fixed_mut_fn is None else fixed_mut_fn,
                xo_prob,
                mut_prob,
            )
            for c in configs
        ]
    else:
        normalized = [
            (
                (mp, xp),
                fixed_xo_fn,
                fixed_mut_fn,
                xp,
                mp,
            )
            for mp, xp in configs
        ]

    for key, xo_fn, mut_fn, run_xo_prob, run_mut_prob in normalized:
        curves = []
        label  = key if isinstance(key, str) else f"mut={key[0]} xo={key[1]}"

        print(f"\n{'='*60}")
        print(f"  {config_key}: {label}")
        print(f"{'='*60}")

        for run in range(1, n_runs + 1):
            print(f"  Run {run}/{n_runs}", end="  ")

            random.seed(run * seed)
            np.random.seed(run * seed)

            initial_pop = [
                Individual(
                    target=target_array,
                    fitness_metric=fitness_metric,
                    **individual_kwargs,
                )
                for _ in range(pop_size)
            ]

            start = time()
            best_ind, fitness_curve = genetic_algorithm(
                initial_population  = initial_pop,
                max_generations     = max_gens,
                selection_algorithm = selection_algorithm,
                xo_method           = xo_fn,
                mut_method          = mut_fn,
                maximization        = maximization,
                xo_prob             = run_xo_prob,
                mut_prob            = run_mut_prob,
                elitism             = elitism,
                verbose             = False,
                **ga_kwargs,
            )
            elapsed = time() - start

            if (key not in best_inds
                    or (not maximization and best_ind.fitness() < best_inds[key].fitness())
                    or (    maximization and best_ind.fitness() > best_inds[key].fitness())):
                best_inds[key] = best_ind

            curves.append(fitness_curve)
            all_results.append({
                config_key    : key,
                "run"         : run,
                "best_fitness": best_ind.fitness(),
                "time_seconds": round(elapsed, 2),
            })
            print(f"best fitness: {best_ind.fitness():.4f}")

        all_curves[key] = curves
        avg = np.mean([r["best_fitness"] for r in all_results if r[config_key] == key])
        print(f"  Avg best fitness: {avg:.4f}")

    return all_results, all_curves, best_inds






# FUNCTION TO RUN A SINGLE CONFIGURATION  FOR MULTIPLE RUNS

def run_single_experiment(target_array, n_runs, max_gens, pop_size, seed, xo_prob, mut_prob, elitism,
                          selection_algorithm, xo_fn, mut_fn, maximization=False, fitness_metric="rmse",
                          individual_kwargs=None, **ga_kwargs):
    """
    Run a single GA configuration for multiple independent runs.

    Args:
        - target_array (np.ndarray): Target image as an (H, W, 3) float32 array.
        - n_runs (int): Number of independent runs.
        - max_gens (int): Maximum number of generations per run.
        - pop_size (int): Number of individuals in the population.
        - seed (int): Base seed; run i uses i * seed for reproducibility.
        - xo_prob (float): Crossover probability.
        - mut_prob (float): Mutation probability.
        - elitism (bool): Whether to carry the best individual unchanged.
        - selection_algorithm (Callable): Selection function.
        - xo_fn (Callable): Crossover function.
        - mut_fn (Callable): Mutation function.
        - individual_kwargs (dict, optional): Extra kwargs for Individual constructor.
        - **ga_kwargs: Extra kwargs forwarded to genetic_algorithm.

    Returns:
        - all_results (list[dict]): One dict per run: 'run', 'best_fitness', 'time_seconds'.
        - all_curves (list[list[float]]): Per-generation best fitness for each run.
        - best_ind (Individual): Best individual found across all runs.
    """
    individual_kwargs = individual_kwargs or {}
    all_results = []
    all_curves  = []
    best_ind    = None

    for run in range(1, n_runs + 1):
        print(f"  Run {run}/{n_runs}", end="  ")

        random.seed(run * seed)
        np.random.seed(run * seed)

        initial_pop = [
            Individual(
                target=target_array,
                fitness_metric=fitness_metric,
                **individual_kwargs,
            )
            for _ in range(pop_size)
        ]

        start = time()
        ind, fitness_curve = genetic_algorithm(
            initial_population  = initial_pop,
            max_generations     = max_gens,
            selection_algorithm = selection_algorithm,
            xo_method           = xo_fn,
            mut_method          = mut_fn,
            maximization        = maximization,
            xo_prob             = xo_prob,
            mut_prob            = mut_prob,
            elitism             = elitism,
            verbose             = False,
            **ga_kwargs,
        )
        elapsed = time() - start

        if (best_ind is None
                or (not maximization and ind.fitness() < best_ind.fitness())
                or (    maximization and ind.fitness() > best_ind.fitness())):
            best_ind = ind

        all_curves.append(fitness_curve)
        all_results.append({
            "run"         : run,
            "best_fitness": ind.fitness(),
            "time_seconds": round(elapsed, 2),
        })
        print(f"best fitness: {ind.fitness():.4f}")

    avg_fitness = np.mean([r["best_fitness"] for r in all_results])
    print(f"\n  Average fitness over {n_runs} runs: {avg_fitness:.4f}")

    return all_results, all_curves, best_ind



# COMPARE TWO EXPERIMENTS
def compare_two_experiments(results_a, curves_a, label_a, results_b, curves_b, label_b,
    maximization=False, alpha=0.05):
    """
    Statistically compare two configurations using Wilcoxon signed-rank test.

    Args:
        - results_a / results_b: Output of run_single_experiment.
        - curves_a / curves_b: Corresponding per-run fitness curves.
        - label_a / label_b: Config names.
        - maximization (bool): Direction of optimisation.
        - alpha (float): Significance level (default 0.05).

    Returns:
        - avg_curve_a, avg_curve_b (np.ndarray): Mean fitness per generation.
        - p_value (float)
        - significant (bool)
    """
    finals_a = np.array([r["best_fitness"] for r in results_a])
    finals_b = np.array([r["best_fitness"] for r in results_b])

    avg_curve_a = np.mean(curves_a, axis=0)
    avg_curve_b = np.mean(curves_b, axis=0)

    if len(finals_a) == len(finals_b):
        stat, p_value = stats.wilcoxon(finals_a, finals_b)
        test_name = "Wilcoxon signed-rank"
    else:
        stat, p_value = stats.mannwhitneyu(finals_a, finals_b, alternative="two-sided")
        test_name = "Mann-Whitney U (unpaired)"

    significant = p_value < alpha
    if significant:
        if not maximization:
            winner = label_a if finals_a.mean() < finals_b.mean() else label_b
        else:
            winner = label_a if finals_a.mean() > finals_b.mean() else label_b
    else:
        winner = None

    print(f"\n{'='*60}")
    print(f"  Comparison: '{label_a}'  vs  '{label_b}'")
    print(f"{'='*60}")
    print(f"  {'':30}  {'Mean':>8}  {'Std':>8}  {'Best':>8}  {'Worst':>8}")
    print(f"  {'-'*30}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")
    for label, finals in [(label_a, finals_a), (label_b, finals_b)]:
        best  = finals.min() if not maximization else finals.max()
        worst = finals.max() if not maximization else finals.min()
        print(f"  {label:<30}  {finals.mean():>8.4f}  {finals.std():>8.4f}  {best:>8.4f}  {worst:>8.4f}")
    print(f"\n  {test_name}: stat={stat:.4f}, p={p_value:.4f}")
    if significant:
        print(f"  Statistically significant (p < {alpha}). Winner: '{winner}'.")
    else:
        print(f"  No statistically significant difference (p ≥ {alpha}).")

    return avg_curve_a, avg_curve_b, p_value, significant


# COMPARE MORE THAN ONE CONFIGURATION
def compare_all_configs(all_results, all_curves, config_key, maximization=False, alpha=0.05):
    """
    Compare all configurations at once using the Kruskal-Wallis test
    (non-parametric equivalent of one-way ANOVA), then produce a clean
    summary table with per-config statistics.

    If Kruskal-Wallis is significant, runs post-hoc pairwise Mann-Whitney U
    tests with Bonferroni correction to identify which configs differ.

    Args:
        - all_results (list[dict]): Full results from run_experiment.
        - all_curves (dict): Full curves dict from run_experiment.
        - config_key (str): Key used to identify configs (e.g. 'mutation_type').
        - maximization (bool): Direction of optimisation.
        - alpha (float): Significance level (default 0.05).

    Returns:
        - summary_df (pd.DataFrame): Per-config statistics table.
        - avg_curves (dict): Maps config name -> mean fitness curve (np.ndarray).
    """

    config_names = list(all_curves.keys())

    # Per-config statistics
    rows = []
    finals = {}
    avg_curves = {}
    for name in config_names:
        runs = [r["best_fitness"] for r in all_results if r[config_key] == name]
        times = [r["time_seconds"] for r in all_results if r[config_key] == name]
        arr = np.array(runs)
        finals[name] = arr
        avg_curves[name] = np.mean(all_curves[name], axis=0)
        rows.append({
            "config"   : name,
            "avg"      : arr.mean(),
            "std"      : arr.std(),
            "best"     : arr.min() if not maximization else arr.max(),
            "worst"    : arr.max() if not maximization else arr.min(),
            "avg_time" : np.mean(times),
        })

    summary_df = pd.DataFrame(rows).set_index("config")

    # Global test: Kruskal-Wallis 
    # Tests whether at least one config is drawn from a different distribution.
    # Non-parametric, no normality assumption needed.
    groups = [finals[name] for name in config_names]
    stat_kw, p_kw = stats.kruskal(*groups)

    print(f"\n{'='*65}")
    print(f"  Global comparison — {config_key}")
    print(f"  Kruskal-Wallis: H={stat_kw:.4f}, p={p_kw:.4f}  "
          f"→ {'significant' if p_kw < alpha else 'not significant'}")
    print(f"{'='*65}")
    print(summary_df.to_string(float_format=lambda x: f"{x:.4f}"))

    # Post-hoc: pairwise Mann-Whitney U with Bonferroni correction 
    # Only run if global test is significant.
    if p_kw < alpha:
        pairs = list(combinations(config_names, 2))
        bonferroni_alpha = alpha / len(pairs)
        print(f"\n  Post-hoc pairwise Mann-Whitney U (Bonferroni α={bonferroni_alpha:.4f})")
        print(f"  {'Pair':<45}  {'p-value':>9}  {'Winner'}")
        print(f"  {'-'*45}  {'-'*9}  {'-'*20}")
        for a, b in pairs:
            _, p = stats.mannwhitneyu(finals[a], finals[b], alternative="two-sided")
            significant = p < bonferroni_alpha
            if significant:
                winner = a if (
                    (not maximization and finals[a].mean() < finals[b].mean()) or
                    (    maximization and finals[a].mean() > finals[b].mean())
                ) else b
            else:
                winner = "— (ns)"
            print(f"  {a+' vs '+b:<45}  {p:>9.4f}  {winner}")
    else:
        print("\n  No post-hoc tests run (global test not significant).")

    return summary_df, avg_curves





# FUNCTION TO PLOT THE CONVERGENCE CURVE OF A SINGLE RUN

def plot_convergence_curve(fitness_curve, baseline_rmse, MAX_GENS):

    '''
    Args:
        - fitness_curve (list[float]): The fitness of the best individual at each generation.
        - baseline_rmse (float): The RMSE of a random baseline for comparison.
        - MAX_GENS (int): The maximum number of generations, used for the x-axis range.
    '''

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(range(1, MAX_GENS + 1), fitness_curve, color="#2196F3", linewidth=1.5)
    ax.axhline(y=baseline_rmse, color="#F44336", linestyle="--", linewidth=1.5, label=f"Random baseline (RMSE={baseline_rmse:.2f})")
    ax.set_xlabel("Generation", fontsize=12)
    ax.set_ylabel("Best Fitness (RMSE)", fontsize=12)
    ax.set_title("Convergence Curve", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()




# FUNCTION FOR PLOTTING EXPERIMENT RESULTS - CONVERGE CURVES + FINAL FITNESS

def plot_experiment_summary(all_curves, df, configs, config_key, title_prefix, colors=None, errorbar_every=None):
    """
    Plot per-generation average fitness with std error bars (as shown in class)
    and a boxplot of final fitness distributions.

    Args:
        - all_curves (dict): Maps config name -> list of per-run fitness curves.
        - df (pd.DataFrame): Results dataframe with config_key and 'best_fitness'.
        - configs (list[dict] | list[tuple]): Operator configs or (mut_prob, xo_prob) tuples.
        - config_key (str): Column in df to group by.
        - title_prefix (str): Figure suptitle prefix.
        - colors (list[str], optional): One hex color per config.
        - errorbar_every (int, optional): Plot error bars every N generations to avoid clutter.
          Defaults to max(1, max_gens // 20).
    """
    if colors is None:
        colors = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0"]

    if configs and isinstance(configs[0], dict):
        keys   = [c["name"] for c in configs]
        labels = keys
    else:
        keys   = configs
        labels = [f"mut={k[0]} xo={k[1]}" for k in keys]

    max_gens = len(next(iter(all_curves.values()))[0])
    if errorbar_every is None:
        errorbar_every = max(1, max_gens // 20)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Plot 1: mean ± std error bars per generation (as shown in class) 
    ax = axes[0]
    for idx, (key, label) in enumerate(zip(keys, labels)):
        curves = np.array(all_curves[key])   # shape: (n_runs, max_gens)
        mean   = curves.mean(axis=0)
        std    = curves.std(axis=0)
        gens   = np.arange(1, max_gens + 1)
        color  = colors[idx % len(colors)]

        # Plot the mean line
        ax.plot(gens, mean, label=label, color=color, linewidth=2)

        # Error bars only every `errorbar_every` generations to keep it readable
        eb_idx = np.arange(0, max_gens, errorbar_every)
        ax.errorbar(
            gens[eb_idx], mean[eb_idx], yerr=std[eb_idx],
            fmt="none",           # no marker — just the caps
            ecolor=color,
            elinewidth=1.2,
            capsize=4,
            capthick=1.2,
            alpha=0.8,
        )

    ax.set_xlabel("Generation", fontsize=12)
    ax.set_ylabel("Avg Best Fitness (RMSE)", fontsize=12)
    ax.set_title(f"Convergence by {title_prefix}\n(mean ± std)", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Plot 2: boxplot of final fitness 
    ax = axes[1]
    groups = [df[df[config_key] == key]["best_fitness"].values for key in keys]
    bp = ax.boxplot(groups, labels=labels, patch_artist=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_ylabel("Final Best Fitness (RMSE)", fontsize=12)
    ax.set_title(
        f"Final Fitness Distribution\n({len(next(iter(all_curves.values())))} runs per configuration)",
        fontsize=13,
    )
    ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(f"{title_prefix} Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()
 



# FUNCTION FOR PLOTTING EXPERIMENT RESULTS - BEST INDIVIDUALS VS TARGET
def plot_best_individuals(best_inds, configs, target_img, title_prefix, ncols=6):
    """
    Display the target image alongside the best evolved individual(s). This function supports two modes:
    1. Single individual mode:
       - `best_inds` is a single Individual object.
       - Displays the target image and the best individual side by side.

    2. Multiple configuration mode:
       - `best_inds` is a dictionary mapping configuration identifiers
         to Individuals.
       - Displays the target image alongside the best individual for
         each configuration.

    Args:
        - best_inds (Individual or dict):
            Either:
                * A single best Individual object.
                * A dictionary mapping configuration keys to Individuals.
        - configs (list[dict] or list[tuple] or None): 
            Configuration descriptors associated with `best_inds`.
            Supported formats:
                * list[dict] with key "name"
                * list[(mut_prob, xo_prob)] tuples
                * None when plotting a single individual
        - target_img (PIL.Image or np.ndarray):
            The target image displayed as reference.
        - title_prefix (str):
            Figure title prefix.
        - ncols (int):
            Max panels per row in multi-config mode (default 6,
          which includes the Target panel — so up to 5 configs on the first row).
    """

    # Single individual mode
    if not isinstance(best_inds, dict):

        fig, axes = plt.subplots(1, 2, figsize=(8, 4))

        axes[0].imshow(target_img)
        axes[0].set_title("Target", fontsize=12)
        axes[0].axis("off")

        axes[1].imshow(best_inds.render())
        axes[1].set_title(
            f"Best Individual\nRMSE={best_inds.fitness():.2f}",
            fontsize=10
        )
        axes[1].axis("off")

        fig.suptitle(
            title_prefix,
            fontsize=13,
            fontweight="bold"
        )

        plt.tight_layout()
        plt.show()

        return

    # Multiple configuration mode
    if configs and isinstance(configs[0], dict):
        keys   = [c["name"] for c in configs]
        labels = keys

    else:
        keys   = configs
        labels = [f"mut={k[0]} xo={k[1]}" for k in keys]

    # Panel 0 = Target, then one per config.
    n_panels = len(keys) + 1
    ncols    = min(ncols, n_panels)
    nrows    = (n_panels + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(3 * ncols, 4 * nrows),
        squeeze=False,
    )
    axes_flat = axes.ravel()

    axes_flat[0].imshow(target_img)
    axes_flat[0].set_title("Target", fontsize=12)
    axes_flat[0].axis("off")

    for ax, key, label in zip(axes_flat[1:], keys, labels):

        ind = best_inds[key]

        ax.imshow(ind.render())

        ax.set_title(
            f"{label}\nRMSE={ind.fitness():.2f}",
            fontsize=10
        )

        ax.axis("off")
    
    # Hide any unused panels at the end of the grid.
    for ax in axes_flat[n_panels:]:
        ax.axis("off")


    fig.suptitle(
        f"{title_prefix} — Best Individual per Configuration",
        fontsize=13,
        fontweight="bold"
    )

    plt.tight_layout()
    plt.show()


# ONE-CALL EVALUATION PIPELINE FOR A MULTI-CONFIG EXPERIMENT
def evaluate_experiment(all_results, all_curves, best_inds, configs, config_key,
                        target_img, title_prefix, maximization=False, alpha=0.05,
                        plot_curves=True, plot_images=True, ncols=6):
    """
    Run the full post-experiment reporting pipeline in a single call.

    Bundles the four steps used across every test section so that each section
    in the notebook needs only ONE evaluation cell instead of four:
      1. Build a tidy DataFrame from per-run results.
      2. Run the statistical comparison (Kruskal-Wallis + post-hoc Mann-Whitney
         with Bonferroni correction).
      3. Plot mean ± std convergence curves and the boxplot of final fitnesses.
      4. Plot the best individual per config against the target image.

    Args:
        - all_results (list[dict]): Output of run_experiment, or aggregated
          from per-config run_single_experiment results (one dict per run).
        - all_curves (dict): Maps config key -> list of per-run fitness curves.
        - best_inds (dict): Maps config key -> best Individual across runs.
        - configs (list[dict] | list[tuple]): Operator configs (each with
          "name", optionally "fn") or (mut_prob, xo_prob) tuples.
        - config_key (str): Column / key identifying each config in all_results.
        - target_img (PIL.Image or np.ndarray): Reference image for the
          best-individuals plot.
        - title_prefix (str): Used in plot titles and section headers.
        - maximization (bool): Direction of optimisation (default False).
        - alpha (float): Significance level for statistical tests.
        - plot_curves (bool): If False, skip the convergence + boxplot figure.
        - plot_images (bool): If False, skip the best-individuals figure.

    Returns:
        - df (pd.DataFrame): Tidy per-run results.
        - summary (pd.DataFrame): Per-config statistics + global test outcome.
        - avg_curves (dict): Maps config -> mean fitness curve (np.ndarray).
    """
    # 1. DataFrame
    df = pd.DataFrame(all_results)

    # 2. Statistical comparison (prints the Kruskal-Wallis table + post-hoc)
    summary, avg_curves = compare_all_configs(
        all_results = all_results,
        all_curves  = all_curves,
        config_key  = config_key,
        maximization= maximization,
        alpha       = alpha,
    )

    # 3. Convergence (mean ± std) + final-fitness boxplot
    if plot_curves:
        plot_experiment_summary(
            all_curves   = all_curves,
            df           = df,
            configs      = configs,
            config_key   = config_key,
            title_prefix = title_prefix,
        )

    # 4. Best individual per config against the target
    if plot_images:
        plot_best_individuals(
            best_inds    = best_inds,
            configs      = configs,
            target_img   = target_img,
            title_prefix = title_prefix,
            ncols         = ncols
        )

    return df, summary, avg_curves

# FUNCTIONS FOR HORIZONTAL BOX PLOT FOR BETTER VISUALISATION OF MULTIPLE CONFIGURATIONS
def plot_experiment_summary_h(all_curves, df, configs, config_key, title_prefix,
                              colors=None, errorbar_every=None):
    """
    Same as plot_experiment_summary, but the final-fitness boxplot is
    horizontal so long config names (e.g. 'mut0.01_xo0.85') stay readable.
    """
    if colors is None:
        colors = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0",
                  "#00BCD4", "#FFC107", "#795548", "#607D8B"]

    if configs and isinstance(configs[0], dict):
        keys   = [c["name"] for c in configs]
        labels = keys
    else:
        keys   = configs
        labels = [f"mut={k[0]} xo={k[1]}" for k in keys]

    max_gens = len(next(iter(all_curves.values()))[0])
    if errorbar_every is None:
        errorbar_every = max(1, max_gens // 20)

    # Slightly wider/taller than the vertical version so 9 horizontal
    # boxes don't get cramped when there are many configs.
    fig, axes = plt.subplots(
        1, 2,
        figsize=(16, max(5, 0.5 * len(keys) + 2)),
        gridspec_kw={"width_ratios": [1.4, 1]},
    )

    # --- Plot 1: mean ± std convergence (unchanged) ---
    ax = axes[0]
    for idx, (key, label) in enumerate(zip(keys, labels)):
        curves = np.array(all_curves[key])
        mean   = curves.mean(axis=0)
        std    = curves.std(axis=0)
        gens   = np.arange(1, max_gens + 1)
        color  = colors[idx % len(colors)]

        ax.plot(gens, mean, label=label, color=color, linewidth=2)

        eb_idx = np.arange(0, max_gens, errorbar_every)
        ax.errorbar(
            gens[eb_idx], mean[eb_idx], yerr=std[eb_idx],
            fmt="none", ecolor=color,
            elinewidth=1.2, capsize=4, capthick=1.2, alpha=0.8,
        )

    ax.set_xlabel("Generation", fontsize=12)
    ax.set_ylabel("Avg Best Fitness (RMSE)", fontsize=12)
    ax.set_title(f"Convergence by {title_prefix}\n(mean ± std)", fontsize=13)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.3)

    # --- Plot 2: HORIZONTAL boxplot of final fitness ---
    ax = axes[1]
    groups = [df[df[config_key] == key]["best_fitness"].values for key in keys]

    bp = ax.boxplot(
        groups,
        vert=False,                # <-- horizontal
        labels=labels,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=1.5),
        flierprops=dict(marker="o", markersize=4, markerfacecolor="white"),
    )

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_xlabel("Final Best Fitness (RMSE)", fontsize=12)
    ax.set_title(
        f"Final Fitness Distribution\n"
        f"({len(next(iter(all_curves.values())))} runs per configuration)",
        fontsize=13,
    )
    ax.invert_yaxis()   # first config on top (more natural reading order)
    ax.grid(True, axis="x", alpha=0.3)

    fig.suptitle(f"{title_prefix} Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()


def evaluate_experiment_h(all_results, all_curves, best_inds, configs, config_key,
                          target_img, title_prefix, maximization=False, alpha=0.05,
                          plot_curves=True, plot_images=True, ncols=6):
    """
    Drop-in replacement for evaluate_experiment that uses the horizontal-
    boxplot version of the summary plot (plot_experiment_summary_h).
    Everything else (DataFrame, Kruskal-Wallis + post-hoc, best-individuals
    figure) is identical to the original.
    """
    # 1. DataFrame
    df = pd.DataFrame(all_results)

    # 2. Statistical comparison (Kruskal-Wallis + post-hoc Mann-Whitney)
    summary, avg_curves = compare_all_configs(
        all_results = all_results,
        all_curves  = all_curves,
        config_key  = config_key,
        maximization= maximization,
        alpha       = alpha,
    )

    # 3. Convergence (mean ± std) + HORIZONTAL final-fitness boxplot
    if plot_curves:
        plot_experiment_summary_h(
            all_curves   = all_curves,
            df           = df,
            configs      = configs,
            config_key   = config_key,
            title_prefix = title_prefix,
        )

    # 4. Best individual per config against the target
    if plot_images:
        plot_best_individuals(
            best_inds    = best_inds,
            configs      = configs,
            target_img   = target_img,
            title_prefix = title_prefix,
            ncols         = ncols
        )

    return df, summary, avg_curves

# LOAD PRE-COMPUTED EXPERIMENT ARTIFACTS FROM DISK
 
class _LoadedIndividual:
    """Lightweight stand-in for an Individual, built from saved artifacts.
 
    Implements the minimal interface that ``plot_best_individuals`` and
    ``evaluate_experiment`` rely on: ``.render()`` returns a PIL image of
    the best individual, and ``.fitness()`` returns its scalar fitness.
    """
    def __init__(self, png_path, fitness_val):
        self._img = Image.open(png_path).convert("RGB")
        self._fit = float(fitness_val)
 
    def render(self):
        return self._img
 
    def fitness(self):
        return self._fit
    
def load_experiment_artifacts(checkpoint_path, config_names, config_key,
                              best_png_template=None):
    """
    Load a pre-computed experiment from disk into the same (all_results,
    all_curves, best_inds) triple that ``run_experiment`` returns, so it
    can be fed directly into ``evaluate_experiment``.
 
    Useful for sections whose experiment was run by a separate script
    (e.g. ``_run_mutation.py``) because a full sweep is too slow to do
    inline in the notebook.
 
    Expected on-disk layout (paths relative to ``checkpoint_path.parent``):
        - ``checkpoint_path``: JSON file structured as
          ``{config_name: [{"run", "fitness", "time_seconds", "curve_file"}, ...]}``.
        - ``curve_file``: ``.npy`` fitness curve per run (path stored in the checkpoint).
        - best-individual PNGs named via ``best_png_template``.
 
    Args:
        - checkpoint_path (str | Path): Path to the JSON checkpoint.
        - config_names (list[str]): Configs to load (must be keys in the checkpoint).
        - config_key (str): Field name used in each result dict
          (e.g. "mutation_type") — must match what ``evaluate_experiment`` expects.
        - best_png_template (str, optional): Filename template for the best
          PNG, with ``{name}`` placeholder. Defaults to
          ``"{stem}_best_{name}.png"`` where ``{stem}`` is the checkpoint
          filename with ``_checkpoint`` stripped (e.g. ``mutation_checkpoint.json``
          -> ``mutation_best_{name}.png``).
 
    Returns:
        - all_results (list[dict]): One dict per run.
        - all_curves (dict[str, list[list[float]]]): Per-config fitness curves.
        - best_inds (dict[str, _LoadedIndividual]): Per-config best individual.
    """
    checkpoint_path = Path(checkpoint_path)
    artifacts_dir = checkpoint_path.parent
 
    with open(checkpoint_path, encoding="utf-8") as f:
        checkpoint = json.load(f)
 
    if best_png_template is None:
        stem = checkpoint_path.stem.replace("_checkpoint", "")
        best_png_template = f"{stem}_best_{{name}}.png"
 
    all_results = []
    all_curves  = {}
    best_inds   = {}
 
    for name in config_names:
        runs = checkpoint[name]
 
        # Flat per-run records — same format as run_experiment outputs.
        for r in runs:
            all_results.append({
                config_key    : name,
                "run"         : r["run"],
                "best_fitness": r["fitness"],
                "time_seconds": r["time_seconds"],
            })
 
        # Per-run fitness curves, ordered by run number.
        all_curves[name] = [
            np.load(artifacts_dir / r["curve_file"]).tolist()
            for r in sorted(runs, key=lambda x: x["run"])
        ]
 
        # Best individual: lowest-fitness run's PNG.
        best_run = min(runs, key=lambda r: r["fitness"])
        best_inds[name] = _LoadedIndividual(
            artifacts_dir / best_png_template.format(name=name),
            best_run["fitness"],
        )
 
    return all_results, all_curves, best_inds