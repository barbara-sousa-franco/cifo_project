<<<<<<< HEAD
"""Unified runner for the GA experiments: mutation (sec 5), crossover (sec 6)
and probabilities (sec 9), with checkpoints and per-config filters.

Each phase keeps everything else fixed and only varies the variable it is
testing. The probabilities phase uses the winners of the previous two
phases as the fixed mutation/crossover operators
(adaptive_mutation_schedule + uniform_crossover).
=======
"""Unified runner for the full CIFO project experiment pipeline.

Phases (in dependency order):
  mutation        sec 5 -- 5 mutations
  crossover       sec 6 -- 5 crossovers, fixed mut=AdaptiveMut
  probabilities   sec 9 -- 3x3 grid of (mut_prob, xo_prob), fixed mut+xo
  size            sec 7 -- 4 values of MAX_TRIANGLE_SIZE
  alpha           sec 8 -- 5 (alpha_min, alpha_max) windows
  diversity       sec 11 -- 7 anti-convergence mechanisms (incl. fitness sharing,
                  restricted mating)
  ciede2000       sec 12 -- challenge: RMSE vs CIEDE2000 fitness
  random_search   refinement -- random sampling around the winning probs +
                  size + alpha to look for nearby optima
  final_run       sec 10 -- THE final run: pop=500, gens=15000, all winners
                  plugged in. Long run, single best image of the project.

Every phase writes per-run results to run_artifacts/<phase>_checkpoint.json
as soon as each run finishes, so a crash/restart just resumes.
>>>>>>> f3fb8154d7120382f685c708f578f6776a2316c7

==========================================================================
HOW TO RUN
==========================================================================

<<<<<<< HEAD
Run ALL phases (mutation -> crossover -> probabilities), with checkpoints:
    python _run_experiment.py

Run ONLY one phase:
    python _run_experiment.py mutation
    python _run_experiment.py crossover
    python _run_experiment.py probabilities

Run a subset explicitly:
    python _run_experiment.py crossover probabilities

Restrict to specific configs (case-insensitive). Probability configs are
named like "mut0.01_xo0.9" so the filter still works:
    python _run_experiment.py mutation --only Gaussian AdaptiveMut
    python _run_experiment.py crossover --skip ReducedSurrogate Shuffle
    python _run_experiment.py probabilities --skip mut0.15_xo0.85

Each completed run is saved immediately to a checkpoint file in
run_artifacts/. If the PC reboots or the script crashes, just re-run it
with the same arguments -- completed runs are recovered from disk and only
the missing ones execute.

Outputs (per phase, under run_artifacts/):
  <phase>_checkpoint.json      Per-run results, restart-safe
  <phase>_results.json         Final aggregated summary
  <phase>_<config>_run<NN>_curve.npy   Per-generation fitness curve
  <phase>_best_<config>.png    Best-of-config rendered image
==========================================================================
=======
Run ALL phases in default order:
    python _run_experiment.py

Run a subset:
    python _run_experiment.py mutation crossover
    python _run_experiment.py probabilities
    python _run_experiment.py random_search final_run

Filter configs within the chosen phases (case-insensitive):
    python _run_experiment.py mutation --only Gaussian AdaptiveMut
    python _run_experiment.py probabilities --skip mut0.15_xo0.85
    python _run_experiment.py random_search --only sample_03 sample_07

Outputs (per phase, under run_artifacts/):
  <phase>_checkpoint.json      Per-run results, restart-safe
  <phase>_results.json         Final aggregated summary (avg/std/min/max)
  <phase>_<config>_run<NN>_curve.npy   Per-generation fitness curve
  <phase>_best_<config>.png    Best-of-config rendered image
==========================================================================

Winners assumed (used as fixed values in later phases):
  mutation  : adaptive_mutation_schedule
  crossover : uniform_crossover
  prob      : (mut_prob=0.01, xo_prob=0.95)

These are encoded in the WINNERS dict at the top of the file so they can
be updated in one place if the real winners differ.
>>>>>>> f3fb8154d7120382f685c708f578f6776a2316c7
"""
from __future__ import annotations

import argparse
import json
<<<<<<< HEAD
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
=======
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
>>>>>>> f3fb8154d7120382f685c708f578f6776a2316c7

import numpy as np
from PIL import Image

<<<<<<< HEAD
# Silence the per-generation "Generation X/Y" prints from ga.py
=======
# Silence per-generation "Generation X/Y" prints from ga.py.
>>>>>>> f3fb8154d7120382f685c708f578f6776a2316c7
import builtins as _b
_orig_print = _b.print
def _quiet_print(*a, **k):
    msg = " ".join(str(x) for x in a)
    if "Generation:" not in msg:
        _orig_print(*a, **k)
_b.print = _quiet_print

from solution import Individual
from operators import (
    tournament_selection,
<<<<<<< HEAD
=======
    fitness_sharing_tournament,
    restricted_mating_selection,
>>>>>>> f3fb8154d7120382f685c708f578f6776a2316c7
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

<<<<<<< HEAD
import random

# --------------------------------------------------------------------------
# Shared experiment hyperparameters (chosen to match the notebook).
=======

# --------------------------------------------------------------------------
# Shared hyperparameters and known winners.
#
# WINNERS is the single source of truth for the fixed values that later
# phases plug into the GA. If we later discover a different winner we only
# edit this dict instead of hunting through the file.
>>>>>>> f3fb8154d7120382f685c708f578f6776a2316c7
# --------------------------------------------------------------------------
SEED = 23
POP = 100
GENS = 500
N_RUNS = 15
<<<<<<< HEAD
=======

WINNERS = {
    "mut_fn":     adaptive_mutation_schedule,   # sec 5 winner
    "xo_fn":      uniform_crossover,            # sec 6 winner
    "mut_prob":   0.01,                         # sec 9 winner
    "xo_prob":    0.95,                         # sec 9 winner
    # The size and alpha winners are placeholders until secs 7/8 finish.
    # The default values match the Triangle.__init__ defaults so leaving
    # them None means "use whatever the constructor defaults to".
    "max_triangle_size": 0.25,
    "alpha_min": 0.30,
    "alpha_max": 0.80,
    # Sec 11 winner -- updated once the diversity phase runs.
    "diversity_kwargs": {
        "selection_algorithm": tournament_selection,
    },
}

>>>>>>> f3fb8154d7120382f685c708f578f6776a2316c7
ART = Path("run_artifacts")
ART.mkdir(exist_ok=True)


<<<<<<< HEAD
@dataclass
class Phase:
    """Static description of one experiment phase.

    name        : short identifier used in CLI and filenames ("mutation",
                  "crossover" or "probabilities")
    configs     : list of dicts describing each tested variant. Required key
                  is "name". For mutation/crossover phases each config also
                  provides "fn" (the operator). For the probabilities phase
                  each config provides "mut_prob" and "xo_prob" overrides
                  instead.
    xo_prob     : crossover probability used by the GA. A config may
                  override this via {"xo_prob": ...}.
    mut_prob    : mutation probability used by the GA. A config may
                  override this via {"mut_prob": ...}.
    xo_fn       : fixed crossover function. None means "take from the
                  config's 'fn' field", which is what the crossover phase
                  needs because it varies the crossover.
    mut_fn      : fixed mutation function. Same convention as xo_fn.
    """
    name: str
    configs: list[dict]
    xo_prob: float
    mut_prob: float
    xo_fn: Callable | None
    mut_fn: Callable | None


PHASES: dict[str, Phase] = {
=======
# --------------------------------------------------------------------------
# Phase dataclass.
#
# Each phase varies one or more dimensions and holds all the others fixed.
# To keep things uniform we encode every per-config knob as an override
# inside the config dict itself: "fn", "mut_prob", "xo_prob", "ga_kwargs",
# "individual_kwargs", "fitness_metric", "selection_algorithm". Anything
# absent falls back to the Phase-level default.
# --------------------------------------------------------------------------
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


# --------------------------------------------------------------------------
# Helper: build the random_search configs deterministically (seeded RNG so
# every machine produces the same list).
# --------------------------------------------------------------------------
def _build_random_search_configs(n_samples: int = 12) -> list[dict]:
    """Random sampling near the current winners for mut_prob, xo_prob,
    max_triangle_size and the alpha window.

    Centred on the current WINNERS but exploring small neighbourhoods:
        mut_prob          : U(0.005, 0.04)
        xo_prob           : U(0.85, 1.00)
        max_triangle_size : U(0.15, 0.40)
        alpha_min         : U(0.10, 0.40)
        alpha_max         : alpha_min + U(0.30, 0.60)   (always > alpha_min)
    """
    rng = random.Random(SEED + 999)
    configs = []
    for i in range(n_samples):
        mut_p = round(rng.uniform(0.005, 0.04), 4)
        xo_p  = round(rng.uniform(0.85, 1.00), 4)
        size  = round(rng.uniform(0.15, 0.40), 3)
        a_min = round(rng.uniform(0.10, 0.40), 3)
        a_max = round(min(0.95, a_min + rng.uniform(0.30, 0.60)), 3)
        configs.append({
            "name": f"sample_{i:02d}",
            "mut_prob": mut_p,
            "xo_prob":  xo_p,
            "individual_kwargs": {
                "max_triangle_size": size,
                "alpha_min": a_min,
                "alpha_max": a_max,
            },
            # Stash the raw values too for the final summary table.
            "_params": {
                "mut_prob": mut_p,
                "xo_prob":  xo_p,
                "max_triangle_size": size,
                "alpha_min": a_min,
                "alpha_max": a_max,
            },
        })
    return configs


# --------------------------------------------------------------------------
# The complete pipeline. Order matters -- the default run goes top to bottom.
# --------------------------------------------------------------------------
PHASES: dict[str, Phase] = {
    # ----- sec 5 -----
>>>>>>> f3fb8154d7120382f685c708f578f6776a2316c7
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
<<<<<<< HEAD
        # In the mutation phase the crossover is held fixed (vanilla 1-point)
        # and the mutation function varies per config.
        xo_fn=triangle_crossover,
        mut_fn=None,
    ),
=======
        xo_fn=triangle_crossover,
        mut_fn=None,
    ),

    # ----- sec 6 -----
>>>>>>> f3fb8154d7120382f685c708f578f6776a2316c7
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
<<<<<<< HEAD
        # In the crossover phase the mutation is held fixed (the winner of
        # the mutation phase) and the crossover varies per config.
        xo_fn=None,
        mut_fn=adaptive_mutation_schedule,
    ),
    # ----------------------------------------------------------------
    # Probabilities phase (sec 9).
    #
    # Holds operators fixed to the winners of mutation (adaptive schedule)
    # and crossover (uniform; tied with AdaptiveXO statistically, but
    # Uniform is simpler and avoids the late-run Reduced-Surrogate phase
    # of AdaptiveXO which we already saw underperforms).
    #
    # Varies (mut_prob, xo_prob) in a grid. Each config below carries the
    # per-config override for these probabilities; xo_prob/mut_prob at the
    # Phase level act only as fallbacks (we set them to None to flag that
    # they MUST come from the config).
    # ----------------------------------------------------------------
=======
        xo_fn=None,
        mut_fn=adaptive_mutation_schedule,
    ),

    # ----- sec 9 -----
>>>>>>> f3fb8154d7120382f685c708f578f6776a2316c7
    "probabilities": Phase(
        name="probabilities",
        configs=[
            {"name": f"mut{mp}_xo{xp}", "mut_prob": mp, "xo_prob": xp}
            for mp in [0.01, 0.05, 0.15]
            for xp in [0.85, 0.90, 0.95]
        ],
<<<<<<< HEAD
        # Sentinel values; the actual numbers come from each config dict.
        xo_prob=0.9,
        mut_prob=0.05,
        # Both operators are held fixed (no per-config "fn" in this phase).
        xo_fn=uniform_crossover,
        mut_fn=adaptive_mutation_schedule,
    ),
}


=======
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

    # ----- sec 12 (challenge) -----
    "ciede2000": Phase(
        name="ciede2000",
        configs=[
            {"name": "rmse",      "fitness_metric": "rmse"},
            {"name": "ciede2000", "fitness_metric": "ciede2000"},
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
        n_runs=5,   # fewer runs per sample -- random search is exploratory
    ),

    # ----- sec 10: THE final run -----
    "final_run": Phase(
        name="final_run",
        configs=[
            {"name": "final",
             "mut_prob": WINNERS["mut_prob"],
             "xo_prob":  WINNERS["xo_prob"],
             "individual_kwargs": {
                 "max_triangle_size": WINNERS["max_triangle_size"],
                 "alpha_min":         WINNERS["alpha_min"],
                 "alpha_max":         WINNERS["alpha_max"],
             },
             "ga_kwargs": dict(WINNERS["diversity_kwargs"])},
        ],
        xo_fn=WINNERS["xo_fn"],
        mut_fn=WINNERS["mut_fn"],
        pop=500,
        gens=15000,
        n_runs=1,
    ),
}


# --------------------------------------------------------------------------
# Plumbing.
# --------------------------------------------------------------------------
>>>>>>> f3fb8154d7120382f685c708f578f6776a2316c7
def _stamp(msg: str) -> None:
    _orig_print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _load_checkpoint(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
<<<<<<< HEAD
        data = json.load(f)
    return data
=======
        return json.load(f)
>>>>>>> f3fb8154d7120382f685c708f578f6776a2316c7


def _save_checkpoint(path: Path, state: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def _filter_configs(configs: list[dict], only: list[str], skip: list[str]) -> list[dict]:
<<<<<<< HEAD
    """Apply --only / --skip filters. Names are matched case-insensitively."""
=======
>>>>>>> f3fb8154d7120382f685c708f578f6776a2316c7
    out = configs
    if only:
        wanted = {n.lower() for n in only}
        out = [c for c in out if c["name"].lower() in wanted]
    if skip:
        unwanted = {n.lower() for n in skip}
        out = [c for c in out if c["name"].lower() not in unwanted]
    return out


<<<<<<< HEAD
def run_phase(phase: Phase, only: list[str], skip: list[str]) -> None:
    """Execute one phase: iterate over its configs and runs, with checkpoints."""
    configs = _filter_configs(phase.configs, only, skip)
    if not configs:
        _stamp(f"[{phase.name}] no configs left after --only/--skip filters; skipping.")
=======
def _merge_dicts(*dicts: dict) -> dict:
    """Right-most wins. Used to combine phase-level + config-level kwargs."""
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
    """Execute a single GA run for a (phase, cfg, run) triple."""
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
    return best_ind, fitness_curve


def run_phase(phase: Phase, only: list[str], skip: list[str]) -> None:
    configs = _filter_configs(phase.configs, only, skip)
    if not configs:
        _stamp(f"[{phase.name}] no configs left after --only/--skip; skipping.")
>>>>>>> f3fb8154d7120382f685c708f578f6776a2316c7
        return

    checkpoint = ART / f"{phase.name}_checkpoint.json"
    state = _load_checkpoint(checkpoint)
    if state:
        already = sum(len(v) for v in state.values())
        _stamp(f"[{phase.name}] resuming from checkpoint ({already} runs already done).")

    target_img = Image.open("data/girl_pearl_earing.png").convert("RGB")
    target_array = np.array(target_img, dtype=np.float32)

<<<<<<< HEAD
    _stamp(f"[{phase.name}] POP={POP} GENS={GENS} N_RUNS={N_RUNS}  "
=======
    _stamp(f"[{phase.name}] POP={phase.pop} GENS={phase.gens} N_RUNS={phase.n_runs}  "
>>>>>>> f3fb8154d7120382f685c708f578f6776a2316c7
           f"configs={[c['name'] for c in configs]}")

    best_inds: dict[str, Individual] = {}
    phase_start = time.time()

    for cfg in configs:
        name = cfg["name"]
        completed = state.get(name, [])
        completed_runs = {r["run"] for r in completed}
<<<<<<< HEAD
        _stamp(f"  === {name} ===  ({len(completed_runs)}/{N_RUNS} already done)")

        for run in range(1, N_RUNS + 1):
            if run in completed_runs:
                continue

            random.seed(run * SEED)
            np.random.seed(run * SEED)

            initial_pop = [Individual(target=target_array) for _ in range(POP)]

            # Decide which operator is fixed vs varying for this phase.
            xo_method = phase.xo_fn if phase.xo_fn is not None else cfg["fn"]
            mut_method = phase.mut_fn if phase.mut_fn is not None else cfg["fn"]

            # Per-config probability overrides (used by the probabilities
            # phase to sweep mut_prob x xo_prob without changing operators).
            xo_prob = cfg.get("xo_prob", phase.xo_prob)
            mut_prob = cfg.get("mut_prob", phase.mut_prob)

            t0 = time.time()
            best_ind, fitness_curve = genetic_algorithm(
                initial_population=initial_pop,
                max_generations=GENS,
                selection_algorithm=tournament_selection,
                xo_method=xo_method,
                mut_method=mut_method,
                xo_prob=xo_prob,
                mut_prob=mut_prob,
                elitism=True,
                verbose=False,
            )
            elapsed = time.time() - t0
            fit = best_ind.fitness()

            # Per-run curve saved separately to avoid bloating the JSON.
            curve_path = ART / f"{phase.name}_{name}_run{run:02d}_curve.npy"
            np.save(curve_path, np.asarray(fitness_curve, dtype=np.float32))

            # Keep the best image per config (overwrites only on improvement).
=======
        _stamp(f"  === {name} ===  ({len(completed_runs)}/{phase.n_runs} already done)")

        for run in range(1, phase.n_runs + 1):
            if run in completed_runs:
                continue

            t0 = time.time()
            best_ind, fitness_curve = _run_one(phase, cfg, target_array, run)
            elapsed = time.time() - t0
            fit = best_ind.fitness()

            curve_path = ART / f"{phase.name}_{name}_run{run:02d}_curve.npy"
            np.save(curve_path, np.asarray(fitness_curve, dtype=np.float32))

>>>>>>> f3fb8154d7120382f685c708f578f6776a2316c7
            if name not in best_inds or fit < best_inds[name].fitness():
                best_inds[name] = best_ind
                best_ind.render().save(ART / f"{phase.name}_best_{name}.png")

<<<<<<< HEAD
            state.setdefault(name, []).append({
=======
            entry = {
>>>>>>> f3fb8154d7120382f685c708f578f6776a2316c7
                "run": run,
                "fitness": float(fit),
                "time_seconds": round(elapsed, 2),
                "curve_file": curve_path.name,
<<<<<<< HEAD
            })
            _save_checkpoint(checkpoint, state)
            _stamp(f"    Run {run:2d}/{N_RUNS}: fitness {fit:.3f} in {elapsed:.1f}s")
=======
            }
            # For random_search we also store the per-sample parameters so
            # the post-hoc analysis can correlate them with fitness.
            if "_params" in cfg:
                entry["params"] = cfg["_params"]

            state.setdefault(name, []).append(entry)
            _save_checkpoint(checkpoint, state)
            _stamp(f"    Run {run:2d}/{phase.n_runs}: fitness {fit:.3f} in {elapsed:.1f}s")
>>>>>>> f3fb8154d7120382f685c708f578f6776a2316c7

        fits = [r["fitness"] for r in state[name]]
        if fits:
            _orig_print(f"    summary: avg={np.mean(fits):.3f} std={np.std(fits):.3f} "
                        f"min={np.min(fits):.3f} max={np.max(fits):.3f}")

    _stamp(f"[{phase.name}] phase done in {(time.time() - phase_start) / 3600:.2f}h")

<<<<<<< HEAD
    # Final aggregated summary written for plotting / stats scripts.
=======
    # Final summary written for plotting / stats scripts.
>>>>>>> f3fb8154d7120382f685c708f578f6776a2316c7
    summary = {}
    for cfg in configs:
        name = cfg["name"]
        fits = [r["fitness"] for r in state.get(name, [])]
<<<<<<< HEAD
        if fits:
            summary[name] = {
                "n_runs": len(fits),
                "avg": float(np.mean(fits)),
                "std": float(np.std(fits)),
                "min": float(np.min(fits)),
                "max": float(np.max(fits)),
            }
=======
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
>>>>>>> f3fb8154d7120382f685c708f578f6776a2316c7
    with open(ART / f"{phase.name}_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    _orig_print(f"\n[{phase.name}] final summary (sorted by avg):")
    for n, s in sorted(summary.items(), key=lambda kv: kv[1]["avg"]):
<<<<<<< HEAD
        _orig_print(f"  {n:18s}  avg={s['avg']:7.3f}  std={s['std']:5.3f}  "
=======
        _orig_print(f"  {n:25s}  avg={s['avg']:7.3f}  std={s['std']:5.3f}  "
>>>>>>> f3fb8154d7120382f685c708f578f6776a2316c7
                    f"min={s['min']:6.3f}  (n={s['n_runs']})")


def main() -> None:
    parser = argparse.ArgumentParser(
<<<<<<< HEAD
        description="Run mutation and/or crossover experiments with checkpoints.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join([
            "Examples:",
            "  python _run_experiment.py                    # all phases, all configs",
            "  python _run_experiment.py mutation           # only the mutation phase",
            "  python _run_experiment.py crossover --only Uniform AdaptiveXO",
            "  python _run_experiment.py mutation --skip Full Gaussian",
=======
        description="Run the CIFO experiment pipeline with checkpoints.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join([
            "Examples:",
            "  python _run_experiment.py                            # all phases",
            "  python _run_experiment.py mutation                   # only mutation",
            "  python _run_experiment.py size alpha                 # secs 7 + 8",
            "  python _run_experiment.py diversity ciede2000",
            "  python _run_experiment.py random_search final_run",
>>>>>>> f3fb8154d7120382f685c708f578f6776a2316c7
            "  python _run_experiment.py probabilities --only mut0.01_xo0.95",
        ]),
    )
    parser.add_argument(
        "phases",
        nargs="*",
        choices=list(PHASES.keys()),
        default=None,
<<<<<<< HEAD
        help=("Which phase(s) to run. Default: all, in the order "
              "mutation -> crossover -> probabilities."),
=======
        help="Which phase(s) to run. Default: all, in the order shown by --help.",
>>>>>>> f3fb8154d7120382f685c708f578f6776a2316c7
    )
    parser.add_argument(
        "--only", nargs="+", default=[],
        help="Run only these configs (case-insensitive). Applies to every chosen phase.",
    )
    parser.add_argument(
        "--skip", nargs="+", default=[],
        help="Skip these configs (case-insensitive). Applies to every chosen phase.",
    )

    args = parser.parse_args()
<<<<<<< HEAD
    selected = args.phases or ["mutation", "crossover", "probabilities"]
=======
    selected = args.phases or list(PHASES.keys())
>>>>>>> f3fb8154d7120382f685c708f578f6776a2316c7

    _stamp(f"Phases requested: {selected}")
    if args.only:
        _stamp(f"--only:  {args.only}")
    if args.skip:
        _stamp(f"--skip:  {args.skip}")

    overall_start = time.time()
    try:
        for p in selected:
            run_phase(PHASES[p], args.only, args.skip)
    except KeyboardInterrupt:
        _stamp("Interrupted by user. Checkpoint is up-to-date; rerun to resume.")
        sys.exit(130)
    except Exception as e:
        _stamp(f"FATAL: {e!r}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    _stamp(f"All done in {(time.time() - overall_start) / 3600:.2f}h")


if __name__ == "__main__":
    main()
