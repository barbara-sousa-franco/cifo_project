"""Mutation tests (sec 5) with checkpoints.

==========================================================================
HOW TO RUN (replaces section 5 in main.ipynb):

  1. Open a terminal in the project folder
  2. Run:    python _run_mutation.py
  3. Leave it running ~30h (5 configs x 15 runs of pop=100 gens=500)

The script auto-saves a checkpoint after EVERY completed run. If the PC
restarts or the script crashes, just re-run "python _run_mutation.py" --
it picks up exactly where it left off (skips already-done runs).

All results land in run_artifacts/ as JSON, .npy and .png. After it
finishes, do "git add run_artifacts/" + "git push" so the team can use
the results.
==========================================================================

Runs 15 independent runs of 4 mutation operators + 1 adaptive mutation.
Total: 5 x 15 = 75 runs. Each run pop=100, gens=500.

Checkpoints: after each completed run, saves the partial results to
run_artifacts/mutation_checkpoint.json so a crash/reboot only loses the
current run (not the previous ones).

Restart-safe: if mutation_checkpoint.json exists, it loads completed runs
and skips them; only the missing runs execute.

Output:
  run_artifacts/mutation_checkpoint.json   (per-run results, restart-safe)
  run_artifacts/mutation_results.json      (final aggregated)
  run_artifacts/mutation_curves.npz        (per-generation fitness curves)
  run_artifacts/mutation_best_<config>.png (best image per config)
"""
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

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
    triangle_crossover,
    triangle_mutation_vcf,
    triangle_mutation_full,
    gaussian_gene_mutation,
    color_creep_mutation,
    adaptive_mutation_schedule,
)
from ga import genetic_algorithm

SEED = 23
ART = Path("run_artifacts")
ART.mkdir(exist_ok=True)
CHECKPOINT = ART / "mutation_checkpoint.json"

POP, GENS = 100, 500
N_RUNS = 15
XO_PROB = 0
MUT_PROB = 0.1

# 4 + 1 adaptive = 5 configs
CONFIGS = [
    {"name": "VCF",          "fn": triangle_mutation_vcf},
    {"name": "Full",         "fn": triangle_mutation_full},
    {"name": "Gaussian",     "fn": gaussian_gene_mutation},
    {"name": "ColorCreep",   "fn": color_creep_mutation},
    {"name": "AdaptiveMut",  "fn": adaptive_mutation_schedule},
]


def _stamp(msg):
    _orig_print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_checkpoint():
    if not CHECKPOINT.exists():
        return {}
    with open(CHECKPOINT, "r", encoding="utf-8") as f:
        data = json.load(f)
    _stamp(f"Loaded checkpoint with {sum(len(v) for v in data.values())} runs completed.")
    return data


def save_checkpoint(state):
    with open(CHECKPOINT, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def main():
    target_img = Image.open("data/girl_pearl_earing.png").convert("RGB")
    target_array = np.array(target_img, dtype=np.float32)
    _stamp(f"Mutation experiment | POP={POP} GENS={GENS} N_RUNS={N_RUNS}")
    _stamp(f"Total runs to do: {len(CONFIGS) * N_RUNS}")

    # state[config_name] = list of {run, fitness, time_seconds, curve}
    state = load_checkpoint()

    all_curves = {c["name"]: [] for c in CONFIGS}
    best_inds  = {}
    overall_start = time.time()

    for cfg in CONFIGS:
        name = cfg["name"]
        mut_fn = cfg["fn"]
        completed = state.get(name, [])
        completed_runs = {r["run"] for r in completed}
        _stamp(f"\n=== {name} ===  ({len(completed_runs)}/{N_RUNS} already done)")

        for run in range(1, N_RUNS + 1):
            if run in completed_runs:
                # Recover the curve from disk if available
                continue

            random.seed(run * SEED)
            np.random.seed(run * SEED)

            initial_pop = [Individual(target=target_array) for _ in range(POP)]

            t0 = time.time()
            best_ind, fitness_curve = genetic_algorithm(
                initial_population=initial_pop,
                max_generations=GENS,
                selection_algorithm=tournament_selection,
                xo_method=triangle_crossover,
                mut_method=mut_fn,
                xo_prob=XO_PROB,
                mut_prob=MUT_PROB,
                elitism=True,
                verbose=False,
            )
            elapsed = time.time() - t0
            fit = best_ind.fitness()

            # Save curve to disk separately (lots of data)
            curve_path = ART / f"mutation_{name}_run{run:02d}_curve.npy"
            np.save(curve_path, np.asarray(fitness_curve, dtype=np.float32))

            # Save best image if it improves
            if name not in best_inds or fit < best_inds[name].fitness():
                best_inds[name] = best_ind
                best_ind.render().save(ART / f"mutation_best_{name}.png")

            # Append to state and checkpoint immediately
            state.setdefault(name, []).append({
                "run": run,
                "fitness": float(fit),
                "time_seconds": round(elapsed, 2),
                "curve_file": curve_path.name,
            })
            save_checkpoint(state)
            _stamp(f"  Run {run:2d}/{N_RUNS}: fitness {fit:.3f} in {elapsed:.1f}s")

        # Per-config summary
        fits = [r["fitness"] for r in state[name]]
        if fits:
            _orig_print(f"  {name} summary: avg={np.mean(fits):.3f} std={np.std(fits):.3f} "
                        f"min={np.min(fits):.3f} max={np.max(fits):.3f}")

    total = time.time() - overall_start
    _stamp(f"\n=== DONE in {total/3600:.2f}h ===")

    # Final summary
    summary = {}
    for name in [c["name"] for c in CONFIGS]:
        fits = [r["fitness"] for r in state.get(name, [])]
        if fits:
            summary[name] = {
                "n_runs": len(fits),
                "avg": float(np.mean(fits)),
                "std": float(np.std(fits)),
                "min": float(np.min(fits)),
                "max": float(np.max(fits)),
            }
    with open(ART / "mutation_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    _orig_print("\nFinal summary (sorted by avg):")
    for name, s in sorted(summary.items(), key=lambda kv: kv[1]["avg"]):
        _orig_print(f"  {name:15s}  avg={s['avg']:.3f}  std={s['std']:.3f}  "
                    f"min={s['min']:.3f}  (n={s['n_runs']})")


if __name__ == "__main__":
    main()
