"""Unified runner for the GA experiments: mutation (sec 5), crossover (sec 6)
and probabilities (sec 9), with checkpoints and per-config filters.

Each phase keeps everything else fixed and only varies the variable it is
testing. The probabilities phase uses the winners of the previous two
phases as the fixed mutation/crossover operators
(adaptive_mutation_schedule + uniform_crossover).

==========================================================================
HOW TO RUN
==========================================================================

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
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

# Silence the per-generation "Generation X/Y" prints from ga.py
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

import random

# --------------------------------------------------------------------------
# Shared experiment hyperparameters (chosen to match the notebook).
# --------------------------------------------------------------------------
SEED = 23
POP = 100
GENS = 500
N_RUNS = 15
ART = Path("run_artifacts")
ART.mkdir(exist_ok=True)


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
        # In the mutation phase the crossover is held fixed (vanilla 1-point)
        # and the mutation function varies per config.
        xo_fn=triangle_crossover,
        mut_fn=None,
    ),
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
    "probabilities": Phase(
        name="probabilities",
        configs=[
            {"name": f"mut{mp}_xo{xp}", "mut_prob": mp, "xo_prob": xp}
            for mp in [0.01, 0.05, 0.15]
            for xp in [0.85, 0.90, 0.95]
        ],
        # Sentinel values; the actual numbers come from each config dict.
        xo_prob=0.9,
        mut_prob=0.05,
        # Both operators are held fixed (no per-config "fn" in this phase).
        xo_fn=uniform_crossover,
        mut_fn=adaptive_mutation_schedule,
    ),
}


def _stamp(msg: str) -> None:
    _orig_print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _load_checkpoint(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def _save_checkpoint(path: Path, state: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def _filter_configs(configs: list[dict], only: list[str], skip: list[str]) -> list[dict]:
    """Apply --only / --skip filters. Names are matched case-insensitively."""
    out = configs
    if only:
        wanted = {n.lower() for n in only}
        out = [c for c in out if c["name"].lower() in wanted]
    if skip:
        unwanted = {n.lower() for n in skip}
        out = [c for c in out if c["name"].lower() not in unwanted]
    return out


def run_phase(phase: Phase, only: list[str], skip: list[str]) -> None:
    """Execute one phase: iterate over its configs and runs, with checkpoints."""
    configs = _filter_configs(phase.configs, only, skip)
    if not configs:
        _stamp(f"[{phase.name}] no configs left after --only/--skip filters; skipping.")
        return

    checkpoint = ART / f"{phase.name}_checkpoint.json"
    state = _load_checkpoint(checkpoint)
    if state:
        already = sum(len(v) for v in state.values())
        _stamp(f"[{phase.name}] resuming from checkpoint ({already} runs already done).")

    target_img = Image.open("data/girl_pearl_earing.png").convert("RGB")
    target_array = np.array(target_img, dtype=np.float32)

    _stamp(f"[{phase.name}] POP={POP} GENS={GENS} N_RUNS={N_RUNS}  "
           f"configs={[c['name'] for c in configs]}")

    best_inds: dict[str, Individual] = {}
    phase_start = time.time()

    for cfg in configs:
        name = cfg["name"]
        completed = state.get(name, [])
        completed_runs = {r["run"] for r in completed}
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
            if name not in best_inds or fit < best_inds[name].fitness():
                best_inds[name] = best_ind
                best_ind.render().save(ART / f"{phase.name}_best_{name}.png")

            state.setdefault(name, []).append({
                "run": run,
                "fitness": float(fit),
                "time_seconds": round(elapsed, 2),
                "curve_file": curve_path.name,
            })
            _save_checkpoint(checkpoint, state)
            _stamp(f"    Run {run:2d}/{N_RUNS}: fitness {fit:.3f} in {elapsed:.1f}s")

        fits = [r["fitness"] for r in state[name]]
        if fits:
            _orig_print(f"    summary: avg={np.mean(fits):.3f} std={np.std(fits):.3f} "
                        f"min={np.min(fits):.3f} max={np.max(fits):.3f}")

    _stamp(f"[{phase.name}] phase done in {(time.time() - phase_start) / 3600:.2f}h")

    # Final aggregated summary written for plotting / stats scripts.
    summary = {}
    for cfg in configs:
        name = cfg["name"]
        fits = [r["fitness"] for r in state.get(name, [])]
        if fits:
            summary[name] = {
                "n_runs": len(fits),
                "avg": float(np.mean(fits)),
                "std": float(np.std(fits)),
                "min": float(np.min(fits)),
                "max": float(np.max(fits)),
            }
    with open(ART / f"{phase.name}_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    _orig_print(f"\n[{phase.name}] final summary (sorted by avg):")
    for n, s in sorted(summary.items(), key=lambda kv: kv[1]["avg"]):
        _orig_print(f"  {n:18s}  avg={s['avg']:7.3f}  std={s['std']:5.3f}  "
                    f"min={s['min']:6.3f}  (n={s['n_runs']})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run mutation and/or crossover experiments with checkpoints.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join([
            "Examples:",
            "  python _run_experiment.py                    # all phases, all configs",
            "  python _run_experiment.py mutation           # only the mutation phase",
            "  python _run_experiment.py crossover --only Uniform AdaptiveXO",
            "  python _run_experiment.py mutation --skip Full Gaussian",
            "  python _run_experiment.py probabilities --only mut0.01_xo0.95",
        ]),
    )
    parser.add_argument(
        "phases",
        nargs="*",
        choices=list(PHASES.keys()),
        default=None,
        help=("Which phase(s) to run. Default: all, in the order "
              "mutation -> crossover -> probabilities."),
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
    selected = args.phases or ["mutation", "crossover", "probabilities"]

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
