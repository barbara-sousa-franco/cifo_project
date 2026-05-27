"""Smoke test for the post-audit code changes (ga.py elite cache fix).

Runs a tiny GA (POP=20, GENS=5, 1 seed) for both fitness metrics:
- RMSE
- CIEDE2000

Verifies:
1. GA completes without error.
2. Final fitness is a finite, positive float (not NaN, not inf, not None).
3. Fitness curve has the expected length.
"""
import math
import random
import sys

import numpy as np
from PIL import Image

from solution import Individual
from operators import tournament_selection, uniform_crossover, gaussian_gene_mutation
from ga import genetic_algorithm


SEED = 23
POP = 20
GENS = 5


def smoke(metric: str) -> bool:
    """Run a tiny GA (POP=20, GENS=5) under ``metric`` and check sanity.

    Returns True iff the GA completed, the best fitness is a finite
    positive float, and the per-generation curve has the expected length.
    """
    print(f"\n=== Smoke test: fitness_metric={metric!r} ===")
    random.seed(SEED)
    np.random.seed(SEED)

    target = np.array(Image.open("data/girl_pearl_earing.png").convert("RGB"),
                      dtype=np.float32)

    pop = [Individual(target=target, fitness_metric=metric) for _ in range(POP)]

    best, curve = genetic_algorithm(
        initial_population=pop,
        max_generations=GENS,
        selection_algorithm=tournament_selection,
        xo_method=uniform_crossover,
        mut_method=gaussian_gene_mutation,
        xo_prob=0.9,
        mut_prob=0.05,
        elitism=True,
        verbose=False,
    )

    fit = best.fitness()
    ok = (isinstance(fit, float)
          and math.isfinite(fit)
          and fit > 0
          and len(curve) == GENS)

    print(f"  best fitness  : {fit:.4f}")
    print(f"  curve length  : {len(curve)}  (expected {GENS})")
    print(f"  curve first/last: {curve[0]:.4f} / {curve[-1]:.4f}")
    print(f"  PASS" if ok else f"  FAIL")
    return ok


if __name__ == "__main__":
    rmse_ok = smoke("rmse")
    ciede_ok = smoke("ciede2000")
    if rmse_ok and ciede_ok:
        print("\n[OK] Both smoke tests passed.")
        sys.exit(0)
    print("\n[FAIL] One or both smoke tests failed.")
    sys.exit(1)
