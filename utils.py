# Utility functions 

import random
import numpy as np
import matplotlib.pyplot as plt
from time import time

from ga import genetic_algorithm
from solution import Individual


# FUNCTION FOR EXPERIMENTS OVER DIFFERENT CONFIGURATIONS

def run_experiment(
    configs,
    target_array,
    n_runs, max_gens, pop_size, seed,
    xo_prob, mut_prob, elitism,
    selection_algorithm,
    fixed_xo_fn=None,
    fixed_mut_fn=None,
    config_key="config",
    maximization=False,
    fitness_metric="rmse",
    antialiased=False,
    **ga_kwargs,
):
    """
    Run a genetic algorithm experiment over a list of operator configurations
    (mutations, crossovers) or a grid of (mut_prob, xo_prob) probability pairs.

    Args:
        - configs (list[dict] or list[tuple]): Either a list of operator configs (each with keys 'name' and 'fn') 
        or a list of (mut_prob, xo_prob)tuples for probability grid experiments.

        - target_array (np.ndarray): Target image as an (H, W, 3) float32 array.

        - n_runs (int): Number of independent runs per configuration.

        - max_gens (int): Maximum number of generations per run.

        - pop_size (int): Number of individuals in the population.

        - seed (int): Base seed for reproducibility; each run uses run * seed.

        - xo_prob (float): Crossover probability (used as default when not varied).

        - mut_prob (float): Mutation probability (used as default when not varied).

        - elitism (bool): Whether to carry the best individual to the next generation.

        - selection_algorithm (Callable): Selection function used by the GA.

        - fixed_xo_fn (Callable or None): Crossover function to use across all runs. If None, the function is 
        taken from each config (i.e. we are testing crossover operators).

        - fixed_mut_fn (Callable or None): Mutation function to use across all runs. If None, the function is 
        taken from each config (i.e. we are testing mutation operators).

        - config_key (str): Key name used in the results dicts, e.g. 'mutation_type', 'crossover_type', or 
        'config' for probability experiments.

    Returns:
        - all_results (list[dict]): One dict per run with keys config_key, 'run', and 'best_fitness'.

        - all_curves (dict): Maps config key to a list of fitness curves (one list of floats per run).

        - best_inds (dict): Maps config key to the best Individual found across all runs for that configuration.

    """

    all_results = []
    all_curves  = {}
    best_inds   = {}

    # Normalize configs into (key, label, xo_fn, mut_fn, xo_prob, mut_prob)
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
                (mp, xp),   # key is the tuple itself
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
                    antialiased=antialiased,
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

            if (key not in best_inds or best_ind.fitness() < best_inds[key].fitness()) and not maximization:
                best_inds[key] = best_ind

            elif (key not in best_inds or best_ind.fitness() > best_inds[key].fitness()) and maximization:
                best_inds[key] = best_ind

            curves.append(fitness_curve)
            all_results.append({
                config_key    : key,
                "run"         : run,
                "best_fitness": best_ind.fitness(),
                "time_seconds"  : round(elapsed, 2),
            })
            print(f"best fitness: {best_ind.fitness():.4f}")

        all_curves[key] = curves
        avg = np.mean([r["best_fitness"] for r in all_results if r[config_key] == key])
        print(f" Avg: {avg:.4f}")

    return all_results, all_curves, best_inds






# FUNCTION TO RUN A SINGLE CONFIGURATION  FOR MULTIPLE RUNS

def run_single_experiment(
    target_array,
    n_runs, max_gens, pop_size, seed,
    xo_prob, mut_prob, elitism,
    selection_algorithm,
    xo_fn,
    mut_fn,
    maximization=False,
    fitness_metric="rmse",
    antialiased=False,
    **ga_kwargs,
):
    """
    Run a single configuration of the genetic algorithm for multiple runs.

    Args:
        - target_array (np.ndarray): Target image as an (H, W, 3) float32 array.

        - n_runs (int): Number of independent runs.

        - max_gens (int): Maximum number of generations per run.

        - pop_size (int): Number of individuals in the population.

        - seed (int): Base seed for reproducibility; each run uses run * seed.

        - xo_prob (float): Crossover probability.

        - mut_prob (float): Mutation probability.

        - elitism (bool): Whether to carry the best individual to the next generation.

        - selection_algorithm (Callable): Selection function used by the GA.

        - xo_fn (Callable): Crossover function to use across all runs.

        - mut_fn (Callable): Mutation function to use across all runs.

    Returns:
        - all_results (list[dict]): One dict per run with keys 'run' and 'best_fitness'.

        - all_curves (list[list[float]]): List of fitness curves (one list of floats per run).

        - best_ind (Individual): The best Individual found across all runs.

    """

    all_results = []
    all_curves  = []
    best_ind    = None

    for run in range(1, n_runs + 1):
        print(f"Run {run}/{n_runs}", end="  ")

        random.seed(run * seed)
        np.random.seed(run * seed)

        initial_pop = [
            Individual(
                target=target_array,
                fitness_metric=fitness_metric,
                antialiased=antialiased,
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

        if (best_ind is None or ind.fitness() < best_ind.fitness()) and not maximization:
            best_ind = ind

        elif (best_ind is None or ind.fitness() > best_ind.fitness()) and maximization:
            best_ind = ind

        all_curves.append(fitness_curve)
        all_results.append({
            "run"         : run,
            "best_fitness": ind.fitness(),
            "time_seconds"  : round(elapsed, 2),
        })
        print(f"best fitness: {ind.fitness():.4f}")

    avg_fitness = np.mean([r["best_fitness"] for r in all_results])
    print(f"\nAverage fitness over {n_runs} runs: {avg_fitness:.4f}")

    return all_results, all_curves, best_ind






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

def plot_experiment_summary(
    all_curves,
    df,
    configs,
    config_key,
    title_prefix,
    colors=None,
):
    """
    Plot convergence curves (mean ± std) and a boxplot of final fitness
    for a set of operator or probability configurations.

    Args:
        - all_curves (dict): Maps config name/key to a list of fitness curves (one list of floats per run), 
        as returned by run_operator_experiment or run_probability_experiment.

        - df (pd.DataFrame): Results dataframe with columns config_key and 'best_fitness', as built from 
        all_results.

        - configs (list[dict] or list[tuple]): Operator configurations (each with key 'name') or list of 
        (mut_prob, xo_prob) tuples.

        - config_key (str): Column name in df to group by, e.g. 'mutation_type', 'crossover_type', or a 
        composite key for probability experiments.

        - title_prefix (str): Prefix for the figure suptitle, e.g. 'Mutation', 'Crossover', or 'Probability Grid'.

        - colors (list[str], optional): List of hex color strings, one per config. Defaults to a built-in 
        palette if None.

    """

    if colors is None:
        colors = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0"]

    # Normalize configs into (key, label) pairs regardless of input type
    if configs and isinstance(configs[0], dict):
        keys   = [c["name"] for c in configs]
        labels = keys
    else:
        keys   = configs  # list of (mut_prob, xo_prob) tuples
        labels = [f"mut={k[0]} xo={k[1]}" for k in keys]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Plot 1 — Convergence curves (mean ± std)
    ax = axes[0]
    for idx, (key, label) in enumerate(zip(keys, labels)):
        curves = np.array(all_curves[key])
        mean   = curves.mean(axis=0)
        std    = curves.std(axis=0)
        gens   = np.arange(1, len(mean) + 1)
        color  = colors[idx % len(colors)]
        ax.plot(gens, mean, label=label, color=color, linewidth=2)
        ax.fill_between(gens, mean - std, mean + std, alpha=0.2, color=color)

    ax.set_xlabel("Generation", fontsize=12)
    ax.set_ylabel("Best Fitness (RMSE)", fontsize=12)
    ax.set_title(f"Convergence by {title_prefix}\n(mean ± std)", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Plot 2 — Boxplot of final fitness
    ax = axes[1]
    groups = [df[df[config_key] == key]["best_fitness"].values for key in keys]
    bp = ax.boxplot(groups, labels=labels, patch_artist=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_ylabel("Final Best Fitness (RMSE)", fontsize=12)
    ax.set_title(f"Final Fitness Distribution\n({len(next(iter(all_curves.values())))} runs per configuration)", fontsize=13)
    ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(f"{title_prefix} Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()

 






# FUNCTION FOR PLOTTING EXPERIMENT RESULTS - BEST INDIVIDUALS VS TARGET

# FUNCTION FOR PLOTTING EXPERIMENT RESULTS - BEST INDIVIDUALS VS TARGET
def plot_best_individuals(
    best_inds,
    configs,
    target_img,
    title_prefix,
):
    """
    Display the target image alongside the best evolved individual(s).

    This function supports two modes:

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


    """

    # ---------------------------------------------------------
    # Single individual mode
    # ---------------------------------------------------------
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

    # ---------------------------------------------------------
    # Multiple configuration mode
    # ---------------------------------------------------------
    if configs and isinstance(configs[0], dict):
        keys   = [c["name"] for c in configs]
        labels = keys

    else:
        keys   = configs
        labels = [f"mut={k[0]} xo={k[1]}" for k in keys]

    fig, axes = plt.subplots(1, len(keys) + 1, figsize=(4 * (len(keys) + 1), 4))

    axes[0].imshow(target_img)
    axes[0].set_title("Target", fontsize=12)
    axes[0].axis("off")

    for ax, key, label in zip(axes[1:], keys, labels):

        ind = best_inds[key]

        ax.imshow(ind.render())

        ax.set_title(
            f"{label}\nRMSE={ind.fitness():.2f}",
            fontsize=10
        )

        ax.axis("off")

    fig.suptitle(
        f"{title_prefix} — Best Individual per Configuration",
        fontsize=13,
        fontweight="bold"
    )

    plt.tight_layout()
    plt.show()