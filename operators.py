# DEFINE OPERATORS - SELECTION, CROSSOVER, MUTATION
#
# All operators work on the normalized [0, 1] genome and rely on
# Triangle.__init__ to enforce the domain constraints
# (MAX_TRIANGLE_SIZE, ALPHA_MIN/MAX, non-degenerate vertices). Whenever a
# mutation/crossover produces a new gene block, it must rebuild the Triangle
# via Triangle(repr=...) so the constraint repair runs on the result.

from copy import deepcopy
import random

import numpy as np

from solution import (
    Individual,
    Triangle,
    IMG_WIDTH,
    IMG_HEIGHT,
    GENES_PER_TRIANGLE,
    ALPHA_MIN,
    ALPHA_MAX,
    clip_alpha,
    shrink_to_max_size,
)


# Helper used by fitness sharing and restricted mating: flatten an
# Individual's genome to a single 1-D NumPy array so we can measure
# genotypic distance with np.linalg.norm.
def _flat_genome(individual: Individual) -> np.ndarray:
    """Return a flat (N_TRIANGLES * 10,) array of the individual's genes."""
    return np.fromiter(
        (g for tri in individual.repr for g in tri.repr),
        dtype=np.float64,
        count=len(individual.repr) * GENES_PER_TRIANGLE,
    )


def _genome_distance(a: Individual, b: Individual) -> float:
    """Euclidean distance between two individuals' flattened genomes,
    normalised by sqrt(genome length) so the scale is independent of the
    number of triangles."""
    ga = _flat_genome(a)
    gb = _flat_genome(b)
    return float(np.linalg.norm(ga - gb) / np.sqrt(len(ga)))


# SELECTION: Tournament Selection
def tournament_selection(population: list[Individual], maximization: bool = False, tournament_size: int = 2):
    """Selects an individual from the population using tournament selection.
    Parameters:
        - population (list[Individual]): The list of individuals in the population.
        - maximization (bool): If True, selects the individual with the highest fitness; otherwise, selects the one with the lowest fitness.
        - tournament_size (int): The number of individuals to participate in the tournament.
    Returns:
        - Individual: A copy of the selected individual.
    """
    # Select a random subset of individuals for the tournament
    tournament = random.choices(population, k=tournament_size)
    if maximization:
        best_individual = max(tournament, key=lambda ind: ind.fitness())
    else:
        best_individual = min(tournament, key=lambda ind: ind.fitness())

    return best_individual.with_repr(best_individual.repr)


# =====================================
# SELECTION VARIANTS FOR DIVERSITY:
# - Fitness sharing tournament (Goldberg & Richardson, 1987): penalises
#   individuals that are surrounded by similar individuals in genotype
#   space. The penalty grows with crowding within radius sigma_share.
# - Restricted mating (Goldberg, 1989; Eshelman & Schaffer, 1991):
#   parent 2 is constrained to be neither too close nor too far from
#   parent 1, so the GA recombines compatible-but-different individuals.
# Both are standard techniques covered in the CIFO course.
# =====================================

def fitness_sharing_tournament(
    population: list[Individual],
    maximization: bool = False,
    tournament_size: int = 2,
    sigma_share: float = 0.22,
    strength: float = 0.18,
):
    """Tournament selection on fitness penalised by local crowding.

    For each individual, we count how many neighbours fall inside a niche
    of radius ``sigma_share`` (genotypic Euclidean distance). The raw
    fitness is then multiplied by ``(1 + strength * crowding)`` for
    minimisation (so crowded individuals look worse) or divided by the
    same factor for maximisation.

    Reference: Goldberg & Richardson, "Genetic algorithms with sharing
    for multimodal function optimization", ICGA 1987.

    Parameters
    ----------
    population : list[Individual]
        Current population.
    maximization : bool
        If True, larger fitness is better.
    tournament_size : int
        Number of contenders in the tournament.
    sigma_share : float
        Niche radius in normalised genotype space (0 .. 1 typical).
        Smaller -> more niches, more diversity pressure.
    strength : float
        How aggressively crowded individuals are penalised. 0 disables.
    """
    if not population:
        raise ValueError("Cannot select from an empty population.")

    n = len(population)
    raw = np.array([ind.fitness() for ind in population], dtype=np.float64)
    genomes = np.stack([_flat_genome(ind) for ind in population], axis=0)
    # Pairwise distances scaled to keep sigma_share independent of genome length.
    norm = np.sqrt(genomes.shape[1])
    crowding = np.zeros(n, dtype=np.float64)
    for i in range(n):
        distances = np.linalg.norm(genomes - genomes[i], axis=1) / norm
        sharing = np.maximum(0.0, 1.0 - distances / max(1e-9, sigma_share))
        crowding[i] = float(np.mean(sharing))

    if maximization:
        adjusted = raw / (1.0 + strength * crowding)
    else:
        adjusted = raw * (1.0 + strength * crowding)

    # Standard tournament on the adjusted scores.
    contenders_idx = random.choices(range(n), k=max(1, tournament_size))
    if maximization:
        best_idx = max(contenders_idx, key=lambda i: adjusted[i])
    else:
        best_idx = min(contenders_idx, key=lambda i: adjusted[i])
    best = population[best_idx]
    return best.with_repr(best.repr)


def restricted_mating_selection(
    population: list[Individual],
    parent1: Individual,
    maximization: bool = False,
    pool_size: int = 10,
    min_distance: float = 0.012,
    max_distance: float = 0.30,
    base_selection=None,
):
    """Pick parent 2 from candidates that are not too close to parent 1.

    First samples ``pool_size`` candidates via ``base_selection`` (defaults
    to tournament). Then keeps only those whose genotypic distance to
    ``parent1`` falls inside [min_distance, max_distance]. From that
    surviving pool returns the best one. If no candidate fits the window,
    returns the candidate whose distance is closest to the midpoint of the
    allowed range -- a graceful fallback that still respects the intent.

    Reference: Goldberg, "Genetic Algorithms in Search, Optimization and
    Machine Learning", 1989, §5.5; Eshelman & Schaffer, "Preventing
    premature convergence by preventing incest", ICGA 1991.

    Parameters
    ----------
    population : list[Individual]
        Current population.
    parent1 : Individual
        The already-selected first parent.
    pool_size : int
        How many candidates to draw before filtering.
    min_distance, max_distance : float
        The allowed niche width in normalised genotype space.
    base_selection : callable, optional
        Selection function used to sample candidates. Defaults to
        tournament_selection.
    """
    if base_selection is None:
        base_selection = tournament_selection

    candidates = [base_selection(population, maximization) for _ in range(max(1, pool_size))]
    distances = [_genome_distance(parent1, c) for c in candidates]
    valid = [(c, d) for c, d in zip(candidates, distances) if min_distance <= d <= max_distance]

    if valid:
        pool = [c for c, _ in valid]
        if maximization:
            chosen = max(pool, key=lambda ind: ind.fitness())
        else:
            chosen = min(pool, key=lambda ind: ind.fitness())
        return chosen.with_repr(chosen.repr)

    # Fallback: closest candidate to the centre of the allowed window.
    target = 0.5 * (min_distance + max_distance)
    chosen = min(zip(candidates, distances), key=lambda item: abs(item[1] - target))[0]
    return chosen.with_repr(chosen.repr)


# =====================================
# CROSSOVER:
# =====================================

def triangle_crossover(parent1, parent2, crossover_prob, verbose=False, **kwargs):
    """Performs single-point crossover between two parent individuals.
    A random crossover point is selected, and the segments after this point are swapped between
    the parents to create two children.
    Parameters:
        - parent1 (Individual): The first parent individual.
        - parent2 (Individual): The second parent individual.
        - crossover_prob (float): The probability of performing crossover.
    Returns:
        - tuple: A tuple containing the two child individuals resulting from crossover.
    """
    if random.random() <= crossover_prob:
        # Select a random crossover point (between 1 and the number of triangles - 1)
        point = random.randint(1, len(parent1.repr) - 1)

        # Single-point crossover - swap the segments after the crossover point
        child1_triangles = parent1.repr[:point] + parent2.repr[point:]
        child2_triangles = parent2.repr[:point] + parent1.repr[point:]

        # Create new child individuals with the new triangle lists
        child1 = parent1.with_repr(child1_triangles)
        child2 = parent2.with_repr(child2_triangles)

    else: # If crossover does not occur, return deep copies of the parents
        child1 = parent1.with_repr(parent1.repr)
        child2 = parent2.with_repr(parent2.repr)

    return child1, child2


# --- Helper: cria offspring com novo repr ---
def _new_ind(parent, new_repr):
    return parent.with_repr(new_repr)


# 1. UNIFORM CROSSOVER
def uniform_crossover(p1, p2, xo_prob, verbose=False, p=0.5, **kwargs):
    """Cada gene é herdado independentemente de p1 (prob p) ou p2 (prob 1-p)."""
    if random.random() > xo_prob:
        return deepcopy(p1), deepcopy(p2)
    r1, r2 = p1.repr, p2.repr
    size = len(r1)
    mask = [random.random() < p for _ in range(size)]
    c1 = [deepcopy(r1[i]) if mask[i] else deepcopy(r2[i]) for i in range(size)]
    c2 = [deepcopy(r2[i]) if mask[i] else deepcopy(r1[i]) for i in range(size)]
    if verbose:
        print(f"Uniform Crossover: {c1} | {c2}")
    return _new_ind(p1, c1), _new_ind(p2, c2)


# 2. K-POINT CROSSOVER  (K variável entre k_min e k_max)
def kpoint_crossover(p1, p2, xo_prob, verbose=False, k_min=3, k_max=7, **kwargs):
    """K pontos de corte, K amostrado aleatoriamente em [k_min, k_max] a cada chamada."""
    if random.random() > xo_prob:
        return deepcopy(p1), deepcopy(p2)
    r1, r2 = p1.repr, p2.repr
    size = len(r1)
    k = random.randint(k_min, k_max)
    cuts = sorted(random.sample(range(1, size), min(k, size - 1)))
    c1, c2 = [], []
    prev = 0
    for seg_idx, cut in enumerate(cuts + [size]):
        if seg_idx % 2 == 0:
            c1.extend(deepcopy(r1[prev:cut]))
            c2.extend(deepcopy(r2[prev:cut]))
        else:
            c1.extend(deepcopy(r2[prev:cut]))
            c2.extend(deepcopy(r1[prev:cut]))
        prev = cut
    if verbose:
        print(f"K-Point Crossover: {c1} | {c2}")
    return _new_ind(p1, c1), _new_ind(p2, c2)


# 3. REDUCED SURROGATE CROSSOVER
def reduced_surrogate_crossover(p1, p2, xo_prob, verbose=False, **kwargs):
    """
    Corta apenas numa posição onde os dois pais diferem.
    Evita crossovers inúteis — mais eficiente quando a população converge.
    """
    if random.random() > xo_prob:
        return deepcopy(p1), deepcopy(p2)
    r1, r2 = p1.repr, p2.repr
    size = len(r1)
    diff = [i for i in range(size) if r1[i] != r2[i]]
    if len(diff) < 2:  # pais quase idênticos: devolve clones
        return deepcopy(p1), deepcopy(p2)
    cut = random.choice(diff[:-1])
    c1 = deepcopy(r1[:cut]) + deepcopy(r2[cut:])
    c2 = deepcopy(r2[:cut]) + deepcopy(r1[cut:])
    if verbose:
        print(f"Reduced Surrogate Crossover: {c1} | {c2}")
    return _new_ind(p1, c1), _new_ind(p2, c2)


# 4. SHUFFLE CROSSOVER
def shuffle_crossover(p1, p2, xo_prob, verbose=False, **kwargs):
    """
    Aplica o mesmo shuffle aleatório a ambos os pais, faz single-point crossover,
    e depois inverte o shuffle — elimina o viés posicional.
    """
    if random.random() > xo_prob:
        return deepcopy(p1), deepcopy(p2)
    r1, r2 = p1.repr, p2.repr
    size = len(r1)
    indices = list(range(size))
    random.shuffle(indices)
    s1 = [r1[i] for i in indices]
    s2 = [r2[i] for i in indices]
    cut = random.randint(1, size - 1)
    cs1 = s1[:cut] + s2[cut:]
    cs2 = s2[:cut] + s1[cut:]
    inv = [0] * size
    for new_pos, orig_pos in enumerate(indices):
        inv[orig_pos] = new_pos
    c1 = [deepcopy(cs1[inv[i]]) for i in range(size)]
    c2 = [deepcopy(cs2[inv[i]]) for i in range(size)]
    if verbose:
        print(f"Shuffle Crossover: {c1} | {c2}")
    return _new_ind(p1, c1), _new_ind(p2, c2)


# 5. ADAPTIVE CROSSOVER SCHEDULE
def adaptive_crossover_schedule(p1, p2, xo_prob, verbose=False,
                                 current_gen=0, max_gen=100, **kwargs):
    """
    Changes the operation of crossover based on the phase of evolution:
      - Initial phase  (< 50% of generations) : Uniform      → maximum exploration
      - Mid phase      (50–85%)              : K-Point      → structured mixing
      - Final phase    (> 85% of generations) : Red.Surrogate → focus on real differences

    Args:
        - p1, p2: Parent individuals.
        - xo_prob: Crossover probability.
        - verbose: If True, prints the chosen crossover type and resulting children.
        - current_gen: The current generation number (starting from 0).
        - max_gen: The total number of generations planned for the evolution process.
    """
    phase = current_gen / max_gen

    if phase < 0.5:
        if verbose: print(f"Gen {current_gen}: Uniform")
        return uniform_crossover(p1, p2, xo_prob, verbose=verbose, **kwargs)
    elif phase < 0.85:
        if verbose: print(f"Gen {current_gen}: K-Point")
        return kpoint_crossover(p1, p2, xo_prob, verbose=verbose, **kwargs)
    else:
        if verbose: print(f"Gen {current_gen}: Reduced Surrogate")
        return reduced_surrogate_crossover(p1, p2, xo_prob, verbose=verbose, **kwargs)

# =====================================
# MUTATION:
# =====================================
#
# All mutation operators that touch genes in place must respect the domain
# constraints (alpha clip, max triangle size, non-degenerate). The simplest
# way is to rebuild the mutated Triangle via Triangle(repr=...) at the end,
# which runs the constraint repairs from solution.py. For very hot inner
# loops we instead clip alpha + shrink in place to avoid the repair cost.

def triangles_overlap(t1, t2):
    """ Checks if two triangles overlap by comparing their bounding boxes.
    Parameters:
        - t1 (Triangle): The first triangle.
        - t2 (Triangle): The second triangle.
    Returns:
        - bool: True if the triangles overlap, False otherwise.
    """
    # Bounding box of the triangle 1
    x1_min, x1_max = min(t1.repr[0], t1.repr[2], t1.repr[4]), max(t1.repr[0], t1.repr[2], t1.repr[4])
    y1_min, y1_max = min(t1.repr[1], t1.repr[3], t1.repr[5]), max(t1.repr[1], t1.repr[3], t1.repr[5])
    # Bounding box of the triangle 2
    x2_min, x2_max = min(t2.repr[0], t2.repr[2], t2.repr[4]), max(t2.repr[0], t2.repr[2], t2.repr[4])
    y2_min, y2_max = min(t2.repr[1], t2.repr[3], t2.repr[5]), max(t2.repr[1], t2.repr[3], t2.repr[5])

    # Verifies if the bounding boxes intersect
    return (x1_min < x2_max and x1_max > x2_min and
            y1_min < y2_max and y1_max > y2_min)


def triangle_mutation_vcf(individual, mutation_prob):
    """ Performs mutation on an individual by applying vertex, color, and order mutations to its triangles.
    Each triangle in the individual's representation has a chance to mutate based on the given mutation probability.
    The type of mutation applied to each triangle is randomly selected from vertex, color, full, and order mutations.

    Domain constraints are enforced after each gene-level perturbation:
      - alpha (gene 9) is clipped to [ALPHA_MIN, ALPHA_MAX] after a color mutation
      - the triangle is rebuilt via Triangle(repr=...) so the bounding box is
        shrunk to MAX_TRIANGLE_SIZE and any degenerate vertices are repaired

    Parameters:
        - individual (Individual): The individual to be mutated.
        - mutation_prob (float): The probability of mutating each triangle.
    Returns:
        - Individual: A new individual resulting from mutation.
    """

    # Initial defensive copy of the individual's representation to ensure that we do not modify the original individual directly.
    new_repr = [t.copy() for t in individual.repr]

    for i, triangle in enumerate(new_repr):
        if random.random() > mutation_prob:
            continue

        # Choose between 1 to 4 mutations to apply to this triangle
        mutations = random.sample(["vertices", "color", "full", "order"],
                                   k=random.randint(1, 4))

        # If "full" was chosen, it doesn't make sense to apply the other mutations, so we can skip them
        if "full" in mutations:
            new_repr[i] = Triangle()
            continue

        touched = False
        if "vertices" in mutations:
            idx = random.choice([0, 2, 4])
            triangle.repr[idx] = max(0.0, min(1.0, triangle.repr[idx] + random.gauss(0, 0.05)))
            triangle.repr[idx + 1] = max(0.0, min(1.0, triangle.repr[idx + 1] + random.gauss(0, 0.05)))
            touched = True

        if "color" in mutations:
            for j in range(6, 10):
                triangle.repr[j] = max(0.0, min(1.0, triangle.repr[j] + random.gauss(0, 0.05)))
            # Alpha must stay within the visibility window
            triangle.repr[9] = clip_alpha(triangle.repr[9])
            touched = True

        # Re-apply the domain repairs if vertices changed (size + degenerate).
        # We don't rebuild for color-only changes because those cannot break
        # the geometric constraints.
        if touched and "vertices" in mutations:
            new_repr[i] = Triangle(repr=triangle.repr)

        if "order" in mutations:
            # Find triangles that overlap with this one
            overlapping = [j for j, t in enumerate(new_repr)
                           if j != i and triangles_overlap(new_repr[i], t)]
            if overlapping:
                j = random.choice(overlapping)
                new_repr[i], new_repr[j] = new_repr[j], new_repr[i]

    return individual.with_repr(new_repr)


def triangle_mutation_full(individual, mutation_prob):
    """ Performs mutation on an individual by replacing entire triangles with new random triangles. Each triangle in the individual's representation has a chance to mutate based on the given mutation probability.
    Parameters:
        - individual (Individual): The individual to be mutated.
        - mutation_prob (float): The probability of mutating each triangle.
    Returns:
        - Individual: A new individual resulting from mutation.
    """
    new_repr = [t.copy() for t in individual.repr]

    for i in range(len(new_repr)):
        if random.random() > mutation_prob:
            continue # the triangle does not mutate

        # Replace the triangle with a completely new random triangle.
        # Triangle() applies all domain constraints automatically.
        new_repr[i] = Triangle()

    return individual.with_repr(new_repr)


def gaussian_gene_mutation(individual, mutation_prob, sigma=0.05):
    """Standard real-coded GA mutation: per-gene Gaussian perturbation.

    Each of the 10 floats in each triangle mutates independently with
    probability ``mutation_prob`` by adding a Gaussian(0, sigma) and
    clipping to [0, 1]. After the gene-level perturbation, the Triangle is
    rebuilt via Triangle(repr=...) so the domain constraints
    (alpha clip, size shrink, degenerate repair) are applied.

    This is the "vanilla" mutation that pairs naturally with single-point
    or uniform crossover. It explores the continuous genome more uniformly
    than the VCF mutation, at the cost of weaker domain awareness.
    """
    new_repr = []
    for triangle in individual.repr:
        if random.random() > mutation_prob:
            new_repr.append(triangle.copy())
            continue
        # Perturb every gene with the per-gene probability mutation_prob.
        # Empirically this matches the behaviour of standard real-coded GAs
        # on continuous chromosomes (Eiben & Smith, §4.4).
        new_genes = list(triangle.repr)
        for k in range(GENES_PER_TRIANGLE):
            if random.random() <= mutation_prob:
                new_genes[k] = max(0.0, min(1.0, new_genes[k] + random.gauss(0.0, sigma)))
        new_repr.append(Triangle(repr=new_genes))
    return individual.with_repr(new_repr)


def color_creep_mutation(individual, mutation_prob, color_sigma=0.04):
    """Mutate only the RGBA channels of each triangle (genes 6..9).

    Useful late in the run, when the triangle geometry is roughly correct
    but the colors still need fine-tuning. Decouples geometric exploration
    from colour exploitation - similar in spirit to the staged operators of
    Mantere & Koljonen, "Image Optimization by Genetic Algorithms"
    (Soft Computing 2008).
    """
    new_repr = []
    for triangle in individual.repr:
        new_genes = list(triangle.repr)
        changed = False
        for k in range(6, 10):
            if random.random() <= mutation_prob:
                new_genes[k] = max(0.0, min(1.0, new_genes[k] + random.gauss(0.0, color_sigma)))
                changed = True
        if changed:
            # Color-only changes: alpha must stay in window; geometry untouched
            # so we can avoid the (expensive) full Triangle rebuild.
            new_genes[9] = clip_alpha(new_genes[9])
            new_tri = Triangle.__new__(Triangle)
            new_tri.repr = new_genes
            new_repr.append(new_tri)
        else:
            new_repr.append(triangle.copy())
    return individual.with_repr(new_repr)
