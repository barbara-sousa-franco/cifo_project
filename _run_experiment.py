"""Unified runner for the full CIFO project experiment pipeline.

Phases (in dependency order):
  mutation        sec 5 -- 5 mutations
  crossover       sec 6 -- 5 crossovers, fixed mut=AdaptiveMut
  probabilities   sec 9 -- 3x3 grid of (mut_prob, xo_prob), fixed mut+xo
  size            sec 7 -- 4 values of MAX_TRIANGLE_SIZE
  alpha           sec 8 -- 5 (alpha_min, alpha_max) windows
  diversity       sec 11 -- 7 anti-convergence mechanisms (incl. fitness sharing,
                  restricted mating)
  random_search   refinement -- random sampling around the winning probs +
                  size + alpha to look for nearby optima
  validate_top3   refinement -- 15-run validation of sample_02/05/11
  final_run       sec 10 -- THE final run: pop=500, gens=15000, all winners
                  plugged in. Long run, single best image of the project.
  ciede2000       sec 12 -- Challenge: one CIEDE2000 run with the same final
                  setup/budget, then visual comparison against final_run.

Every phase writes per-run results to run_artifacts/<phase>_checkpoint.json
as soon as each run finishes, so a crash/restart just resumes.

==========================================================================
HOW TO RUN
==========================================================================

Run ALL phases in default order:
    python _run_experiment.py

Run a subset:
    python _run_experiment.py mutation crossover
    python _run_experiment.py probabilities
    python _run_experiment.py random_search validate_top3 final_run ciede2000

Filter configs within the chosen phases (case-insensitive):
    python _run_experiment.py mutation --only Gaussian AdaptiveMut
    python _run_experiment.py probabilities --skip mut0.15_xo0.85
    python _run_experiment.py random_search --only sample_03 sample_07

Outputs (per phase, under run_artifacts/):
  <phase>_checkpoint.json      Per-run results, restart-safe
  <phase>_results.json         Final aggregated summary (avg/std/min/max)
  <phase>_<config>_run<NN>_curve.npy   Per-generation fitness curve
  <phase>_best_<config>.png    Best-of-config rendered image
  final_vs_challenge.png       Side-by-side comparison once both final images exist
==========================================================================

Winners assumed (used as fixed values in later phases):
  mutation  : adaptive_mutation_schedule
  crossover : uniform_crossover
  prob      : (mut_prob=0.01, xo_prob=0.95)

These are encoded in the WINNERS dict at the top of the file so they can
be updated in one place if the real winners differ.
The post-random-search setup used by Challenge/final_run is encoded in
FINAL_SETUP so it can be swapped after validate_top3 finishes.
"""

# Import libraries
from __future__ import annotations

import argparse
from cmath import phase
import functools
import json
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image, ImageDraw

# Silence per-generation "Generation X/Y" prints from ga.py.
import builtins as _b


# Suppress verbose per-generation output from the genetic_algorithm() loop
# while preserving all other print statements (summaries, run headers, etc.).

# Save a reference to the original print so the wrapper can still forward
# all non-generation messages to the real print function.
_orig_print = _b.print

def _quiet_print(*a, **k):
    """Wrapper around print that suppresses generation-header lines.
    
    Forwards all messages to the original print except those containing
    'Generation:', which are the per-generation progress lines printed
    by genetic_algorithm() on every iteration.

    Args:
        *a: Positional arguments forwarded to the original print.
        **k: Keyword arguments forwarded to the original print.
    """
    msg = " ".join(str(x) for x in a)
    if "Generation:" not in msg:
        _orig_print(*a, **k)

# Replace print in the ga module's namespace with the filtered version.
# From this point on, any call to print() inside ga.py will go through
# _quiet_print instead of the built-in.
_b.print = _quiet_print

from solution import Individual
from operators import (
    tournament_selection,
    fitness_sharing_tournament,
    restricted_mating_selection,
    # mutation
    triangle_mutation_vcf,
    triangle_mutation_full,
    gaussian_gene_mutation,
    color_creep_mutation,
    adaptive_mutation_schedule,
    # crossover
    triangle_crossover,
    uniform_crossover,
    kpoint_crossover,
    reduced_surrogate_crossover,
    shuffle_crossover,
    adaptive_crossover_schedule,
)
from ga import genetic_algorithm

# Global constants
SEED = 23
POP = 100
GENS = 500
N_RUNS = 15
FINAL_POP = 500
FINAL_GENS = 15000
FINAL_N_RUNS = 1

# The winners from each phase, plugged in as fixed values for later phases. Before the Random Search phase.
#If we later discover a different winner we only edit this dict instead of hunting through the file.
WINNERS = {
    "mut_fn":     adaptive_mutation_schedule,   # sec 5 winner
    "xo_fn":      uniform_crossover,            # sec 6 winner
    "mut_prob":   0.01,                         # sec 9 winner
    "xo_prob":    0.95,                         # sec 9 winner (31.48 RMSE)
    "max_triangle_size": 1.00,                 # sec 7 winner (31.48 RMSE)
    "alpha_min": 0.10,                         # sec 8 winner (30.18 RMSE)
    "alpha_max": 0.40,                         # sec 8 winner (30.18 RMSE)
    # Sec 11 winner: restricted_mating beat baseline by 2.59 RMSE points
    # (27.59 vs 30.18). fitness_sharing on its own adds nothing; combined
    # with restricted_mating it ties (27.64 vs 27.59), so we keep just the
    # mate selector for simplicity.
    "diversity_kwargs": {
        "selection_algorithm": tournament_selection,
        "mate_selection_algorithm": restricted_mating_selection,
    },
}

# Create the artifacts directory if it doesn't exist. All phases write their outputs here, so it needs to be shared
# and created upfront.
ART = Path("run_artifacts")
ART.mkdir(exist_ok=True)





# ==========================================================================
# Phase dataclass.
#
# Each phase varies one or more dimensions and holds all the others fixed.
# To keep things uniform we encode every per-config knob as an override
# inside the config dict itself: "fn", "mut_prob", "xo_prob", "ga_kwargs",
# "individual_kwargs", "fitness_metric", "selection_algorithm". Anything
# absent falls back to the Phase-level default.


@dataclass
class Phase:
    name: str
    configs: list[dict]
    # GA defaults at the phase level (each config can override).
    xo_prob: float = 0.9
    mut_prob: float = 0.05
    # Fixed operators at the phase level (None = take from the config's "fn").
    xo_fn: Callable | None = None
    mut_fn: Callable | None = None
    selection_algorithm: Callable = tournament_selection
    elitism: bool = True
    # Per-phase GA budget (defaults to the shared POP/GENS/N_RUNS).
    pop: int = POP
    gens: int = GENS
    n_runs: int = N_RUNS
    # Extra Individual / GA kwargs applied to EVERY config in this phase.
    individual_kwargs: dict[str, Any] = field(default_factory=dict)
    ga_kwargs: dict[str, Any] = field(default_factory=dict)
    fitness_metric: str = "rmse"


def _make_tournament_selection(tournament_size: int) -> Callable:
    """
    Create a tournament selection function with a fixed tournament size.

    This is a small factory/helper function that pre-configures
    `tournament_selection` and returns a callable that can be used later
    without repeatedly passing `tournament_size`.

    Example:
        selection_fn = _make_tournament_selection(3)

        # Later:
        parent = selection_fn(population, fitness_scores)

    Parameters:
        - tournament_size (int): Number of individuals competing in each tournament.

    Returns:
        - A partially configured tournament selection function.
    """

    return functools.partial(tournament_selection, tournament_size=tournament_size)




def _make_restricted_mating_selection(
    min_distance: float,
    max_distance: float,
    *,
    tournament_size: int | None = None,
) -> Callable:
    
    """
    Create a restricted mating selection function with fixed distance bounds.

    This helper pre-configures `restricted_mating_selection` with the given
    distance constraints.

    If `tournament_size` is provided, tournament selection is used as the
    base selection strategy inside restricted mating.

    Parameters:
        - min_distance (float): Minimum allowed distance between selected mates.

        - max_distance (float): Maximum allowed distance between selected mates.

        - tournament_size (int | None): Optional tournament size for the internal base selection method.

    Returns:
        - A partially configured restricted mating selection function.
    """

    kwargs: dict[str, Any] = {
        "min_distance": min_distance,
        "max_distance": max_distance,
    }
    if tournament_size is not None:
        kwargs["base_selection"] = _make_tournament_selection(tournament_size)

    return functools.partial(restricted_mating_selection, **kwargs)



# ==========================================================================
# Helper: build the random_search configs deterministically (seeded RNG so
# every machine produces the same list).

def _build_random_search_configs(n_samples: int = 12) -> list[dict]:
    """Random sampling around the winners found in earlier phases.

    Eight dimensions varied. The first five centre on the winners from
    sec 5/6/7/8/9; the last three explore parameters that were never
    tuned anywhere else (tournament size and the restricted mating
    distance window, since restricted_mating won the diversity phase).

        mut_prob          : U(0.005, 0.03)    centred near 0.01 (sec 9 winner)
        xo_prob           : U(0.90, 1.00)     centred near 0.95 (sec 9 winner)
        max_triangle_size : U(0.40, 1.00)     covers winner (1.00, sec 7) and
                                              next-best (0.40)
        alpha_min         : U(0.05, 0.20)     winner = 0.10 (sec 8), explore
                                              both sides
        alpha_max         : alpha_min + U(0.20, 0.50)   winner window was 0.30
                                              wide ([0.10, 0.40]); try both
                                              narrower and slightly wider
        tournament_size   : choice(2, 3, 5)   never tuned (always 2)
        mating_min_dist   : U(0.005, 0.05)    restricted_mating default = 0.012
        mating_max_dist   : U(0.20, 0.50)     restricted_mating default = 0.30

    Parameters:
        - n_samples (int): Number of random configs to generate.

    Returns:
        - List of config dicts with random hyperparameters for the random search phase.
    """
    rng = random.Random(SEED + 999)
    configs = []
    for i in range(n_samples):
        # Generate random values for each hyperparameter according to the specified distributions and ranges.
        mut_p     = round(rng.uniform(0.005, 0.03), 4)
        xo_p      = round(rng.uniform(0.90, 1.00), 4)
        size      = round(rng.uniform(0.40, 1.00), 3)
        a_min     = round(rng.uniform(0.05, 0.20), 3)
        a_max     = round(min(0.95, a_min + rng.uniform(0.20, 0.50)), 3)
        tour_size = rng.choice([2, 3, 5])
        mate_min  = round(rng.uniform(0.005, 0.05), 4)
        mate_max  = round(rng.uniform(0.20, 0.50), 3)

        # Build configured selection + mate-selection functions via partial,
        # so the GA can keep calling them with (population, maximization)
        # while the per-sample hyperparameters are baked in.
        selection_fn = _make_tournament_selection(tour_size)
        mate_fn      = _make_restricted_mating_selection(mate_min, mate_max)

        # Build the config dict for this sample, including the random hyperparameters and the resolved functions.
        #  Also stash the raw hyperparameter values under "_params"
        configs.append({
            "name": f"sample_{i:02d}",
            "mut_prob": mut_p,
            "xo_prob":  xo_p,
            "individual_kwargs": {
                "max_triangle_size": size,
                "alpha_min": a_min,
                "alpha_max": a_max,
            },
            "selection_algorithm": selection_fn,
            "ga_kwargs": {"mate_selection_algorithm": mate_fn},
            # Stash the raw values too for the final summary table.
            "_params": {
                "mut_prob":          mut_p,
                "xo_prob":           xo_p,
                "max_triangle_size": size,
                "alpha_min":         a_min,
                "alpha_max":         a_max,
                "tournament_size":   tour_size,
                "mating_min_dist":   mate_min,
                "mating_max_dist":   mate_max,
            },
        })
    return configs


# =========================================================================
# Provisional post-random-search setup used by the final run and Challenge.
#
# validate_top3 selected sample_11 by average RMSE across 15 runs. Keep the
# final/challenge setup here so it can still be swapped in one place.

FINAL_SETUP: dict[str, Any] = {
    "source": "sample_11",
    "mut_prob": 0.0123,
    "xo_prob": 0.9891,
    "max_triangle_size": 0.573,
    "alpha_min": 0.178,
    "alpha_max": 0.521,
    "tournament_size": 5,
    "mating_min_dist": 0.0104,
    "mating_max_dist": 0.348,
}


def _final_individual_kwargs() -> dict[str, Any]:
    """
    Build the keyword arguments used to initialize individuals
    in the final GA experiments.

    These values are taken from `FINAL_SETUP`, which stores the
    best-performing hyperparameters found during previous tuning phases.

    Returns:
        Dictionary containing the individual-level parameters:
        - max_triangle_size
        - alpha_min
        - alpha_max
    """
    return {
        "max_triangle_size": FINAL_SETUP["max_triangle_size"],
        "alpha_min": FINAL_SETUP["alpha_min"],
        "alpha_max": FINAL_SETUP["alpha_max"],
    }


def _final_selection_algorithm() -> Callable:
    """
    Build the parent selection function used in the final GA runs.

    The tournament size is taken from `FINAL_SETUP` and baked into
    the returned callable using `functools.partial`, so the GA can
    call it directly without passing extra parameters.

    Returns:
        Configured tournament selection callable.
    """
    return _make_tournament_selection(FINAL_SETUP["tournament_size"])


def _final_ga_kwargs() -> dict[str, Any]:
    """
    Build GA-level keyword arguments for the final experiment setup.

    Currently this configures the mating-selection strategy using
    the best distance bounds found during the restricted mating phase.

    Returns:
        Dictionary of keyword arguments to be passed into the GA,
        including the configured mate selection algorithm.
    """
    return {
        "mate_selection_algorithm": _make_restricted_mating_selection(
            FINAL_SETUP["mating_min_dist"],
            FINAL_SETUP["mating_max_dist"],
        )
    }


def _final_config(name: str, *, fitness_metric: str | None = None) -> dict[str, Any]:
    """
    Build the base configuration dictionary for a final experiment.

    This helper creates the shared metadata for a final run and stores
    a copy of the selected hyperparameters under `_params` for logging,
    tracking, and summary reporting.

    Args:
        name:
            Name/label of the experiment run.

        fitness_metric:
            Optional fitness metric override to evaluate the run with.
            If omitted, the default metric is used.

    Returns:
        Configuration dictionary describing the final experiment.
    """
    # Store experiment name and a copy of all final hyperparameters
    cfg: dict[str, Any] = {"name": name, "_params": dict(FINAL_SETUP)}
    if fitness_metric is not None:
        cfg["fitness_metric"] = fitness_metric
    return cfg






# ==========================================================================
# The complete pipeline. Order matters -- the default run goes top to bottom.
# ==========================================================================
PHASES: dict[str, Phase] = {
    # ----- sec 5 -----
    "mutation": Phase(
        name="mutation",
        configs=[
            {"name": "VCF",         "fn": triangle_mutation_vcf},
            {"name": "Full",        "fn": triangle_mutation_full},
            {"name": "Gaussian",    "fn": gaussian_gene_mutation},
            {"name": "ColorCreep",  "fn": color_creep_mutation},
            {"name": "AdaptiveMut", "fn": adaptive_mutation_schedule},
        ],
        xo_prob=0,
        mut_prob=0.1,
        xo_fn=uniform_crossover,
        mut_fn=None,
    ),

    # ----- sec 6 -----
    "crossover": Phase(
        name="crossover",
        configs=[
            {"name": "Uniform",          "fn": uniform_crossover},
            {"name": "KPoint",           "fn": kpoint_crossover},
            {"name": "ReducedSurrogate", "fn": reduced_surrogate_crossover},
            {"name": "Shuffle",          "fn": shuffle_crossover},
            {"name": "AdaptiveXO",       "fn": adaptive_crossover_schedule},
        ],
        xo_prob=0.9,
        mut_prob=0,
        xo_fn=None,
        mut_fn=adaptive_mutation_schedule,
    ),

    # ----- sec 9 -----
    "probabilities": Phase(
        name="probabilities",
        configs=[
            {"name": f"mut{mp}_xo{xp}", "mut_prob": mp, "xo_prob": xp}
            for mp in [0.01, 0.05, 0.15]
            for xp in [0.85, 0.90, 0.95]
        ],
        xo_fn=uniform_crossover,
        mut_fn=adaptive_mutation_schedule,
    ),

    # ----- sec 7 -----
    "size": Phase(
        name="size",
        configs=[
            {"name": f"size_{s}", "individual_kwargs": {"max_triangle_size": s}}
            for s in [1.00, 0.40, 0.25, 0.15]
        ],
        xo_prob=WINNERS["xo_prob"],
        mut_prob=WINNERS["mut_prob"],
        xo_fn=WINNERS["xo_fn"],
        mut_fn=WINNERS["mut_fn"],
    ),

    # ----- sec 8 -----
    "alpha": Phase(
        name="alpha",
        configs=[
            {"name": f"alpha_{lo}_{hi}",
             "individual_kwargs": {"alpha_min": lo, "alpha_max": hi}}
            for (lo, hi) in [(0.00, 1.00), (0.10, 0.40),
                             (0.20, 0.90), (0.30, 0.80), (0.50, 0.70)]
        ],
        xo_prob=WINNERS["xo_prob"],
        mut_prob=WINNERS["mut_prob"],
        xo_fn=WINNERS["xo_fn"],
        mut_fn=WINNERS["mut_fn"],
        # Use the size winner from sec 7 as the fixed value here.
        individual_kwargs={"max_triangle_size": WINNERS["max_triangle_size"]},
    ),

    # ----- sec 11 -----
    "diversity": Phase(
        name="diversity",
        configs=[
            {"name": "baseline",
             "ga_kwargs": {}},
            {"name": "adaptive_mutation_rule",
             "ga_kwargs": {"adaptive_mutation": True}},
            {"name": "diversity_injection",
             "ga_kwargs": {"diversity_injection": True}},
            {"name": "adaptive_and_injection",
             "ga_kwargs": {"adaptive_mutation": True, "diversity_injection": True}},
            {"name": "fitness_sharing",
             "ga_kwargs": {},
             "selection_algorithm": fitness_sharing_tournament},
            {"name": "restricted_mating",
             "ga_kwargs": {"mate_selection_algorithm": restricted_mating_selection}},
            {"name": "sharing_and_restricted",
             "ga_kwargs": {"mate_selection_algorithm": restricted_mating_selection},
             "selection_algorithm": fitness_sharing_tournament},
        ],
        xo_prob=WINNERS["xo_prob"],
        mut_prob=WINNERS["mut_prob"],
        xo_fn=WINNERS["xo_fn"],
        mut_fn=WINNERS["mut_fn"],
        individual_kwargs={
            "max_triangle_size": WINNERS["max_triangle_size"],
            "alpha_min":         WINNERS["alpha_min"],
            "alpha_max":         WINNERS["alpha_max"],
        },
    ),

    # ----- sec 11 bonus: ALL diversity mechanisms combined -----
    # Sanity check: even though the individual mechanisms (except restricted_mating) underperformed in the main
    # diversity phase, we wanted to test whether all four together produce
    # any positive interaction. Expectation: similar to or worse than
    # restricted_mating alone (27.59), because adaptive+injection already
    # hurt and fitness_sharing didn't add anything on top of restricted.
    "all_diversity": Phase(
        name="all_diversity",
        configs=[
            {"name": "all_combined",
             "ga_kwargs": {
                 "adaptive_mutation":         True,
                 "diversity_injection":       True,
                 "mate_selection_algorithm":  restricted_mating_selection,
             },
             "selection_algorithm": fitness_sharing_tournament},
        ],
        xo_prob=WINNERS["xo_prob"],
        mut_prob=WINNERS["mut_prob"],
        xo_fn=WINNERS["xo_fn"],
        mut_fn=WINNERS["mut_fn"],
        individual_kwargs={
            "max_triangle_size": WINNERS["max_triangle_size"],
            "alpha_min":         WINNERS["alpha_min"],
            "alpha_max":         WINNERS["alpha_max"],
        },
    ),

    # ----- random search refinement (before the final run) -----
    "random_search": Phase(
        name="random_search",
        configs=_build_random_search_configs(n_samples=12),
        xo_fn=WINNERS["xo_fn"],
        mut_fn=WINNERS["mut_fn"],
        n_runs=5,   # fewer runs per sample - random search is exploratory
    ),

    # ----- validate top configs from random_search with 15 runs each -----
    # The exploratory random_search used only 5 runs per sample, so the
    # ranking between close samples is noisy. We re-run the two best
    # tournament_size=5 samples (sample_05 and sample_11) plus the best
    # tournament_size=2 sample (sample_02 -- the strongest config that
    # respects the historical default tournament size) with 15 runs each
    # to get a statistically solid winner.
    "validate_top3": Phase(
        name="validate_top3",
        configs=[c for c in _build_random_search_configs(n_samples=12)
                 if c["name"] in {"sample_05", "sample_11", "sample_02"}],
        xo_fn=WINNERS["xo_fn"],
        mut_fn=WINNERS["mut_fn"],
        n_runs=15,
    ),

    # ----- sec 10: THE final run -----
    "final_run": Phase(
        name="final_run",
        configs=[_final_config("final")],
        xo_prob=FINAL_SETUP["xo_prob"],
        mut_prob=FINAL_SETUP["mut_prob"],
        xo_fn=WINNERS["xo_fn"],
        mut_fn=WINNERS["mut_fn"],
        selection_algorithm=_final_selection_algorithm(),
        individual_kwargs=_final_individual_kwargs(),
        ga_kwargs=_final_ga_kwargs(),
        pop=FINAL_POP,
        gens=FINAL_GENS,
        n_runs=FINAL_N_RUNS,
    ),

    # ----- sec 12 (Challenge): same final setup/budget, but CIEDE2000 fitness -----
    # Did not use this one since it is estimated that it would run for at least 72h
    "ciede2000": Phase(
        name="ciede2000",
        configs=[_final_config("challenge", fitness_metric="ciede2000")],
        xo_prob=FINAL_SETUP["xo_prob"],
        mut_prob=FINAL_SETUP["mut_prob"],
        xo_fn=WINNERS["xo_fn"],
        mut_fn=WINNERS["mut_fn"],
        selection_algorithm=_final_selection_algorithm(),
        individual_kwargs=_final_individual_kwargs(),
        ga_kwargs=_final_ga_kwargs(),
        pop=FINAL_POP,
        gens=FINAL_GENS,
        n_runs=FINAL_N_RUNS,
    ),
    
    "ciede2000_short": Phase(
    name="ciede2000_short",
    configs=[_final_config("challenge_short", fitness_metric="ciede2000")],
    xo_prob=FINAL_SETUP["xo_prob"],
    mut_prob=FINAL_SETUP["mut_prob"],
    xo_fn=WINNERS["xo_fn"],
    mut_fn=WINNERS["mut_fn"],
    selection_algorithm=_final_selection_algorithm(),
    individual_kwargs=_final_individual_kwargs(),
    ga_kwargs=_final_ga_kwargs(),
    pop=200,
    gens=5000,
    n_runs=1,
),

    # A short-run version of the final run for quick visual comparison with the challenge's result
    "final_run_short": Phase(
    name="final_run_short",
    configs=[_final_config("final_short", fitness_metric="rmse")],
    xo_prob=FINAL_SETUP["xo_prob"],
    mut_prob=FINAL_SETUP["mut_prob"],
    xo_fn=WINNERS["xo_fn"],
    mut_fn=WINNERS["mut_fn"],
    selection_algorithm=_final_selection_algorithm(),
    individual_kwargs=_final_individual_kwargs(),
    ga_kwargs=_final_ga_kwargs(),
    pop=200,
    gens=5000,
    n_runs=1,
),
}


# --------------------------------------------------------------------------
# Plumbing.
# --------------------------------------------------------------------------
def _stamp(msg: str) -> None:
    """
    Print a timestamped log message to stdout.

    Used to make experiment logs easier to read while running long jobs,
    especially when multiple configurations are evaluated sequentially.

    Example:
        [14:32:18] Starting phase 3...

    Parameters:
        - msg: Message to print.
    """
    _orig_print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _load_checkpoint(path: Path) -> dict:
    """
    Load a checkpoint file from disk.

    If the checkpoint file does not exist yet, return an empty dictionary.

    This is useful for resuming experiment runs without repeating already
    completed work.

    Parameters:
        - path: Path to the checkpoint JSON file.

    Returns:
        - Parsed checkpoint dictionary, or {} if the file does not exist.
    """
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_checkpoint(path: Path, state: dict) -> None:
    """
    Save the current experiment state to a checkpoint file.

    Used so interrupted runs can later be resumed from the saved state.

    Parameters:
        - path: Destination path of the checkpoint file.
        - state: Dictionary containing the state/results to persist.
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def _filter_configs(configs: list[dict], only: list[str], skip: list[str]) -> list[dict]:
    """
    Filter experiment configurations by name.

    Supports:
        - selecting only specific config names
        - excluding specific config names

    Matching is case-insensitive.

    Parameters:
        - configs: Full list of experiment configuration dictionaries.
        - only: Optional list of config names to include.
        - skip: Optional list of config names to exclude.

    Returns:
        - Filtered list of configs.
    """
    out = configs
    if only:
        wanted = {n.lower() for n in only}
        out = [c for c in out if c["name"].lower() in wanted]
    if skip:
        unwanted = {n.lower() for n in skip}
        out = [c for c in out if c["name"].lower() not in unwanted]
    return out


def _merge_dicts(*dicts: dict) -> dict:
    """
    Merge multiple dictionaries into one.

    If the same key appears multiple times, the right-most dictionary wins.

    Commonly used to combine phase-level default kwargs with config-specific override kwargs

    Example:
        {"mut_prob": 0.01}
        merged with
        {"mut_prob": 0.02, "xo_prob": 0.95}

        becomes:
        {"mut_prob": 0.02, "xo_prob": 0.95}

    Parameters:
        - *dicts: Any number of dictionaries.

    Returns:
        - Single merged dictionary.
    """
    out: dict = {}
    for d in dicts:
        if d:
            out.update(d)
    return out


def _run_one(
    phase: Phase,
    cfg: dict,
    target_array: np.ndarray,
    run: int,
) -> tuple[Individual, list[float]]:
    """
    Execute a single GA run for one (phase, config, run) combination.

    Parameters:
        - phase (Phase): Phase definition containing defaults such as population size,
        generations, operators and probabilities.
        - cfg (dict): Configuration dictionary for this experiment variant.
        - target_array (np.ndarray): Target image as a NumPy array.
        - run (int): Run index (used to seed randomness for reproducibility).

    Returns:
        - Tuple: (best_individual, fitness_curve)
    """
    random.seed(run * SEED)
    np.random.seed(run * SEED)

    # Resolve operators and probabilities (config wins over phase default).
    xo_method = phase.xo_fn if phase.xo_fn is not None else cfg.get("fn")
    mut_method = phase.mut_fn if phase.mut_fn is not None else cfg.get("fn")
    xo_prob = cfg.get("xo_prob", phase.xo_prob)
    mut_prob = cfg.get("mut_prob", phase.mut_prob)

    # Individual constructor kwargs (max_triangle_size, alpha_min, alpha_max,
    # fitness_metric). Phase default + config override.
    ind_kwargs = _merge_dicts(phase.individual_kwargs, cfg.get("individual_kwargs", {}))
    fitness_metric = cfg.get("fitness_metric", phase.fitness_metric)

    # GA kwargs (adaptive_mutation, diversity_injection, mate_selection_*).
    ga_kwargs = _merge_dicts(phase.ga_kwargs, cfg.get("ga_kwargs", {}))

    # Selection algorithm (phase default; config can override -- needed for
    # the diversity phase where fitness_sharing replaces tournament).
    # Allow ga_kwargs to carry it too (e.g. when WINNERS["diversity_kwargs"]
    # bundles the winning selection): pop it out so it's not passed twice.
    selection_fn = ga_kwargs.pop("selection_algorithm",
                                 cfg.get("selection_algorithm", phase.selection_algorithm))

    # Enable per-generation printing only for the final run.
    if phase.name in ["final_run", "ciede2000", "ciede2000_short", "final_run_short"]:
        _b.print = _orig_print
    else:
        _b.print = _quiet_print

    initial_pop = [
        Individual(target=target_array, fitness_metric=fitness_metric, **ind_kwargs)
        for _ in range(phase.pop)
    ]

    best_ind, fitness_curve = genetic_algorithm(
        initial_population=initial_pop,
        max_generations=phase.gens,
        selection_algorithm=selection_fn,
        xo_method=xo_method,
        mut_method=mut_method,
        xo_prob=xo_prob,
        mut_prob=mut_prob,
        elitism=phase.elitism,
        verbose=False,
        **ga_kwargs,
    )

    # Restore quiet mode after the run so subsequent phases stay clean.
    _b.print = _quiet_print
    
    return best_ind, fitness_curve


def run_phase(phase: Phase, only: list[str], skip: list[str]) -> None:
    """
    Run a complete experiment phase across all selected configurations.

    Handles:
        - config filtering (--only / --skip)
        - checkpoint loading/resume
        - repeated runs per config
        - fitness curve saving
        - best-image saving
        - phase summary statistics

    Results are written to disk under the artifacts directory.

    Parameters:
        - phase (Phase): Phase definition to execute.
        - only (list[str]): Optional config-name whitelist.
        - skip (list[str]): Optional config-name blacklist.
    """
    # Apply CLI filters to configs.
    configs = _filter_configs(phase.configs, only, skip)
    if not configs:
        _stamp(f"[{phase.name}] no configs left after --only/--skip; skipping.")
        return

    # Resume progress if checkpoint already exists.
    checkpoint = ART / f"{phase.name}_checkpoint.json"
    state = _load_checkpoint(checkpoint)
    if state:
        already = sum(len(v) for v in state.values())
        _stamp(f"[{phase.name}] resuming from checkpoint ({already} runs already done).")

    # Load target image once for the whole phase.
    target_img = Image.open("data/girl_pearl_earing.png").convert("RGB")
    target_array = np.array(target_img, dtype=np.float32)

    # Print a timestamped log message with the phase name, GA budget, and configs to be run.
    _stamp(f"[{phase.name}] POP={phase.pop} GENS={phase.gens} N_RUNS={phase.n_runs}  "
           f"configs={[c['name'] for c in configs]}")

    best_inds: dict[str, Individual] = {}
    phase_start = time.time()

    # Run every configuration
    for cfg in configs:
        name = cfg["name"]
        completed = state.get(name, [])
        completed_runs = {r["run"] for r in completed}
        _stamp(f"  === {name} ===  ({len(completed_runs)}/{phase.n_runs} already done)")

        # Run repeated trials
        for run in range(1, phase.n_runs + 1):

            # Skip already-completed runs
            if run in completed_runs:
                continue

            t0 = time.time()
            best_ind, fitness_curve = _run_one(phase, cfg, target_array, run)
            elapsed = time.time() - t0
            fit = best_ind.fitness()

            # Save fitness evolution curve
            curve_path = ART / f"{phase.name}_{name}_run{run:02d}_curve.npy"
            np.save(curve_path, np.asarray(fitness_curve, dtype=np.float32))

            # Track best image for this config across all runs, and save it if it's the best so far.
            if name not in best_inds or fit < best_inds[name].fitness():
                best_inds[name] = best_ind
                best_ind.render().save(ART / f"{phase.name}_best_{name}.png")

            # Store run result in the checkpoint state and save it to disk.
            entry = {
                "run": run,
                "fitness": float(fit),
                "time_seconds": round(elapsed, 2),
                "curve_file": curve_path.name,
            }
            # For random_search we also store the per-sample parameters so
            # the post-hoc analysis can correlate them with fitness.
            if "_params" in cfg:
                entry["params"] = cfg["_params"]

            # Append this run's result to the list of completed runs for this config in the checkpoint state.
            state.setdefault(name, []).append(entry)

            # Save checkpoint after every run
            _save_checkpoint(checkpoint, state)
            _stamp(f"    Run {run:2d}/{phase.n_runs}: fitness {fit:.3f} in {elapsed:.1f}s")

        # After all runs for this config are done, print a summary of the fitness scores.
        fits = [r["fitness"] for r in state[name]]
        if fits:
            _orig_print(f"    summary: avg={np.mean(fits):.3f} std={np.std(fits):.3f} "
                        f"min={np.min(fits):.3f} max={np.max(fits):.3f}")

    _stamp(f"[{phase.name}] phase done in {(time.time() - phase_start) / 3600:.2f}h")

    # Final summary written for plotting / stats scripts.
    summary = {}
    for cfg in configs:
        name = cfg["name"]
        fits = [r["fitness"] for r in state.get(name, [])]
        if not fits:
            continue
        summary[name] = {
            "n_runs": len(fits),
            "avg":    float(np.mean(fits)),
            "std":    float(np.std(fits)),
            "min":    float(np.min(fits)),
            "max":    float(np.max(fits)),
        }
        if "_params" in cfg:
            summary[name]["params"] = cfg["_params"]

    # Save machine-readable results summary for this phase to a JSON file in the artifacts directory.
    with open(ART / f"{phase.name}_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Print ranking ordered by average fitness
    _orig_print(f"\n[{phase.name}] final summary (sorted by avg):")
    for n, s in sorted(summary.items(), key=lambda kv: kv[1]["avg"]):
        _orig_print(f"  {n:25s}  avg={s['avg']:7.3f}  std={s['std']:5.3f}  "
                    f"min={s['min']:6.3f}  (n={s['n_runs']})")







def _maybe_save_final_visual_comparison() -> None:
    """
    Create a side-by-side comparison image between the best final-run
    result and the best challenge result (both with short runs).

    If either image does not exist, nothing is done.

    Output:
        rmse_vs_ciede2000.png
    """
    # Paths to the precomputed best images
    final_path = ART / "final_run_short_best_final_short.png"
    challenge_path = ART / "ciede2000_short_best_challenge_short.png"

    # If either image is missing, skip compariso
    if not final_path.exists() or not challenge_path.exists():
        return

    # Load images
    final_img = Image.open(final_path).convert("RGB")
    challenge_img = Image.open(challenge_path).convert("RGB")

    # Compute canvas size (align both images vertically)
    width = max(final_img.width, challenge_img.width)
    height = max(final_img.height, challenge_img.height)

    label_h = 28 # space for titles
    gap = 12 # space between images

    # Create blank canvas
    canvas = Image.new("RGB", (width * 2 + gap, height + label_h), "white")

    # Paste both images centered in their halves
    canvas.paste(final_img, ((width - final_img.width) // 2, label_h))
    canvas.paste(challenge_img, (width + gap + (width - challenge_img.width) // 2, label_h))

    # Add labels above each image
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), "Final run (RMSE)", fill=(0, 0, 0))
    draw.text((width + gap + 8, 8), "Challenge (CIEDE2000)", fill=(0, 0, 0))

    # Save the comparison image to disk.
    out_path = ART / "rmse_vs_ciede2000.png"
    canvas.save(out_path)
    _stamp(f"Saved visual comparison: {out_path}")









def main() -> None:

    """
    Entry point for the full experiment pipeline.

    Responsibilities:
        - Parse command-line arguments
        - Select which phases to run
        - Apply optional config filters (--only / --skip)
        - Execute phases sequentially
        - Generate final comparison image
        - Handle interruptions and errors safely

    The system is checkpoint-aware:
        If execution is interrupted, re-running the script resumes progress
        instead of restarting from scratch.
    """

    # An ArgumentParser reads the command-line arguments. For example, if the user runs `
    # 'python _run_experiment.py mutation --only mut0.01_xo0.95' then the parser will separate into:
    # script -> _run_experiment.py
    # phase -> mutation
    # option -> --only
    # config -> mut0.01_xo0.95
    # discription is the help message that is printed when the user runs `python _run_experiment.py --help`.
    # epilog is the block of examples printed at the end of the help message.
    parser = argparse.ArgumentParser(
        description="Run the CIFO experiment pipeline with checkpoints.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join([
            "Examples:",
            "  python _run_experiment.py                            # all phases",
            "  python _run_experiment.py mutation                   # only mutation",
            "  python _run_experiment.py size alpha                 # secs 7 + 8",
            "  python _run_experiment.py random_search validate_top3",
            "  python _run_experiment.py final_run ciede2000",
            "  python _run_experiment.py probabilities --only mut0.01_xo0.95",
        ]),
    )

    # Positional argument: list of phases to execute
    parser.add_argument(
        "phases",
        nargs="*", # zero or more phases can be specified; if none, defaults to all
        choices=list(PHASES.keys()), # the user can only choose from the defined phases
        default=None,
        help="Which phase(s) to run. Default: all, in the order shown by --help.",
    )

    # Filter: run only selected configs
    parser.add_argument(
        "--only", nargs="+", default=[],
        help="Run only these configs (case-insensitive). Applies to every chosen phase.",
    )

    # Filter: skip selected configs
    parser.add_argument(
        "--skip", nargs="+", default=[],
        help="Skip these configs (case-insensitive). Applies to every chosen phase.",
    )

    # Parser reads the command-line arguments and converts to an object
    args = parser.parse_args()

    # If no phases are explicitly provided, run all
    selected = args.phases or list(PHASES.keys())

    _stamp(f"Phases requested: {selected}")
    if args.only:
        _stamp(f"--only:  {args.only}")
    if args.skip:
        _stamp(f"--skip:  {args.skip}")

    overall_start = time.time()
    try:
        for p in selected:
            run_phase(PHASES[p], args.only, args.skip)

        # Generate final comparison image if possible
        _maybe_save_final_visual_comparison()
    except KeyboardInterrupt:
        _stamp("Interrupted by user. Checkpoint is up-to-date; rerun to resume.")
        sys.exit(130)
    except Exception as e:
        _stamp(f"FATAL: {e!r}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Final runtime report
    _stamp(f"All done in {(time.time() - overall_start) / 3600:.2f}h")

# Standard Python entry point
if __name__ == "__main__":
    main()
