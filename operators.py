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

from solution import Individual, Triangle, IMG_WIDTH, IMG_HEIGHT, GENES_PER_TRIANGLE, clip_alpha, shrink_to_max_size


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


# =====================================
# SELECTION:
# =====================================


# TOURNAMENT SELECTION:
def tournament_selection(population: list[Individual], maximization: bool = False, tournament_size: int = 2):
    """
    Selects an individual from the population using tournament selection.

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

    # Genome is identical => carry the cached fitness so the caller does
    # not have to rerender + reevaluate when it reads .fitness() later.
    copy = best_individual.with_repr(best_individual.repr)
    copy._fitness = best_individual._fitness
    return copy


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


# FITNESS SHARING TOURNAMENT

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

    Parameters:
    - population : list[Individual]
        Current population.
    - maximization : bool
        If True, larger fitness is better.
    - tournament_size : int
        Number of contenders in the tournament.
    - sigma_share : float
        Niche radius in normalised genotype space (0 .. 1 typical).
        Smaller -> more niches, more diversity pressure.
    - strength : float
        How aggressively crowded individuals are penalised. 0 disables.

    Returns:
    - Individual: A selected individual from the population, with fitness adjusted for sharing.

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
        # sharing indicates the similarity between each individual in the pop and ind i
        # sigma share is the radius of the niche: individuals within this distance contribute to crowding
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
    # Carry the cached raw fitness across the copy (selection adjusts the
    # score but the underlying fitness value is unchanged).
    copy = best.with_repr(best.repr)
    copy._fitness = best._fitness
    return copy



# RESTRICTED MATING

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

    Parameters:
    - population : list[Individual]
        Current population.
    - parent1 : Individual
        The already-selected first parent.
    - pool_size : int
        How many candidates to draw before filtering.
    - min_distance : float
        The minimum allowed distance.
    - max_distance : float
        The maximum allowed distance.
    - base_selection : callable, optional
        Selection function used to sample candidates. Defaults to
        tournament_selection.

    Returns:
    - Individual: A selected parent 2 that is neither too close nor too far from parent 1.

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
        # Carry cached fitness across (identical genome).
        copy = chosen.with_repr(chosen.repr)
        copy._fitness = chosen._fitness
        return copy

    # Fallback: closest candidate to the centre of the allowed window.
    target = 0.5 * (min_distance + max_distance)
    chosen = min(zip(candidates, distances), key=lambda item: abs(item[1] - target))[0]
    copy = chosen.with_repr(chosen.repr)
    copy._fitness = chosen._fitness
    return copy














# =====================================
# CROSSOVER:
# =====================================


# --- Helper: creates offspring with new representation ---
def _new_ind(parent, new_repr):
    return parent.with_repr(new_repr)


# 1. UNIFORM CROSSOVER
def uniform_crossover(p1, p2, xo_prob, verbose=False, p=0.5, **kwargs):

    """
    Each gene is inherited independently from p1 (prob p) or p2 (prob 1-p).
    
    Parameters:
    - p1, p2: Parent individuals.
    - xo_prob: Crossover probability.
    - verbose: If True, prints the resulting children.
    - p: Probability of inheriting each gene from p1 (default 0.5).
    
    Returns:
    - Tuple of two child individuals resulting from crossover.
    
    """
    if random.random() > xo_prob:
        return deepcopy(p1), deepcopy(p2)
    r1, r2 = p1.repr, p2.repr
    size = len(r1)
    # For each gene position, randomly decide whether to take from p1 or p2 based on probability p.
    mask = [random.random() < p for _ in range(size)]
    c1 = [deepcopy(r1[i]) if mask[i] else deepcopy(r2[i]) for i in range(size)]
    c2 = [deepcopy(r2[i]) if mask[i] else deepcopy(r1[i]) for i in range(size)]
    if verbose:
        print(f"Uniform Crossover: {c1} | {c2}")
    return _new_ind(p1, c1), _new_ind(p2, c2)


# 2. K-POINT CROSSOVER  (K between k_min and k_max)
def kpoint_crossover(p1, p2, xo_prob, verbose=False, k_min=3, k_max=7, **kwargs):

    """
    K random crossover points are selected, and the segments between these points are alternately 
    swapped between the two parents to create two children.

    Parameters:
    - p1, p2: Parent individuals.
    - xo_prob: Crossover probability.
    - verbose: If True, prints the resulting children.
    - k_min: Minimum number of crossover points (default 3).
    - k_max: Maximum number of crossover points (default 7).

    Returns:
    - Tuple of two child individuals resulting from crossover.
    
    """
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
    Cuts the genome at a position where the two parents differ, ensuring that the crossover is meaningful.

    Parameters:
    - p1, p2: Parent individuals.
    - xo_prob: Crossover probability.
    - verbose: If True, prints the resulting children.

    Returns:
    - Tuple of two child individuals resulting from crossover.

    """
    if random.random() > xo_prob:
        return deepcopy(p1), deepcopy(p2)
    r1, r2 = p1.repr, p2.repr
    size = len(r1)
    diff = [i for i in range(size) if r1[i] != r2[i]]
    if len(diff) < 2:  # parents almost identical - no meaningful crossover points
        return deepcopy(p1), deepcopy(p2)
    # we exclude the last differing point to ensure we have a lot of difference after the cut
    cut = random.choice(diff[:-1])
    c1 = deepcopy(r1[:cut]) + deepcopy(r2[cut:])
    c2 = deepcopy(r2[:cut]) + deepcopy(r1[cut:])
    if verbose:
        print(f"Reduced Surrogate Crossover: {c1} | {c2}")
    return _new_ind(p1, c1), _new_ind(p2, c2)


# 4. SHUFFLE CROSSOVER
def shuffle_crossover(p1, p2, xo_prob, verbose=False, **kwargs):

    """
    Apply the same random shuffle to both parents, perform single-point crossover, and then invert the shuffle
     — eliminates positional bias.

    Parameters:
    - p1, p2: Parent individuals.
    - xo_prob: Crossover probability.
    - verbose: If True, prints the resulting children.

    Returns:
    - Tuple of two child individuals resulting from crossover.

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

    Parameters:
    - p1, p2: Parent individuals.
    - xo_prob: Crossover probability.
    - verbose: If True, prints the chosen crossover type and resulting children.
    - current_gen: The current generation number (starting from 0).
    - max_gen: The total number of generations planned for the evolution process.

    Returns:
    - Tuple of two child individuals resulting from crossover.

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


def triangle_mutation_vcf(individual, mutation_prob, **kwargs):
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
            new_repr[i] = Triangle(alpha_min=individual.alpha_min, alpha_max=individual.alpha_max, max_triangle_size=individual.max_triangle_size)
            continue

        touched = False
        if "vertices" in mutations:
            # mutate one vertex (x, y) pair by adding Gaussian noise and clipping to [0, 1]
            idx = random.choice([0, 2, 4])
            triangle.repr[idx] = max(0.0, min(1.0, triangle.repr[idx] + random.gauss(0, 0.05)))
            triangle.repr[idx + 1] = max(0.0, min(1.0, triangle.repr[idx + 1] + random.gauss(0, 0.05)))
            touched = True

        if "color" in mutations:
            for j in range(6, 10):
                triangle.repr[j] = max(0.0, min(1.0, triangle.repr[j] + random.gauss(0, 0.05)))
            # Alpha must stay within the visibility window
            triangle.repr[9] = clip_alpha(triangle.repr[9], alpha_min=individual.alpha_min, alpha_max=individual.alpha_max)
            touched = True

        # Re-apply the domain repairs if vertices changed (size + degenerate).
        # We don't rebuild for color-only changes because those cannot break
        # the geometric constraints.
        if touched and "vertices" in mutations:
            new_repr[i] = Triangle(repr=triangle.repr, alpha_min=individual.alpha_min, alpha_max=individual.alpha_max, max_triangle_size=individual.max_triangle_size)

        if "order" in mutations:
            # Find triangles that overlap with this one
            overlapping = [j for j, t in enumerate(new_repr)
                           if j != i and triangles_overlap(new_repr[i], t)]
            if overlapping:
                j = random.choice(overlapping)
                new_repr[i], new_repr[j] = new_repr[j], new_repr[i]

    return individual.with_repr(new_repr)


def triangle_mutation_full(individual, mutation_prob, **kwargs):
    """ Performs mutation on an individual by replacing entire triangles with new random triangles. 
    Each triangle in the individual's representation has a chance to mutate based on the given mutation 
    probability.

    Parameters:
        - individual (Individual): The individual to be mutated.
        - mutation_prob (float): The probability of mutating each triangle.
        - alpha_min (float): The minimum alpha value.
        - alpha_max (float): The maximum alpha value.

    Returns:
        - Individual: A new individual resulting from mutation.

    """
    new_repr = [t.copy() for t in individual.repr]

    for i in range(len(new_repr)):
        if random.random() > mutation_prob:
            continue # the triangle does not mutate

        # Replace the triangle with a completely new random triangle.
        # Triangle() applies all domain constraints automatically.
        new_repr[i] = Triangle(alpha_min=individual.alpha_min, alpha_max=individual.alpha_max, max_triangle_size=individual.max_triangle_size)

    return individual.with_repr(new_repr)


def gaussian_gene_mutation(individual, mutation_prob, sigma=0.05, **kwargs):
    """
    Standard real-coded GA mutation: per-gene Gaussian perturbation.

    Each Traingle of the Individual mutates with probability mutation_prob. If selected, then
    each of the 10 floats in each triangle mutates independently by adding a Gaussian(0, sigma) and
    clipping to [0, 1]. After the gene-level perturbation, the Triangle is
    rebuilt via Triangle(repr=...) so the domain constraints
    (alpha clip, size shrink, degenerate repair) are applied.

    This is the "vanilla" mutation that pairs naturally with single-point
    or uniform crossover. It explores the continuous genome more uniformly
    than the VCF mutation, at the cost of weaker domain awareness.

    Parameters:
    - individual (Individual): The individual to be mutated.
    - mutation_prob (float): The probability of mutating each triangle.
    - sigma (float): The standard deviation of the Gaussian perturbation.

    Returns:
    - Individual: A new individual resulting from mutation.

    """
    new_repr = []
    for triangle in individual.repr:
        if random.random() > mutation_prob:
            new_repr.append(triangle.copy())
            continue

        new_genes = list(triangle.repr)
        for k in range(GENES_PER_TRIANGLE):
            new_genes[k] = max(0.0, min(1.0, new_genes[k] + random.gauss(0.0, sigma)))
        new_repr.append(Triangle(repr=new_genes, alpha_min=triangle.alpha_min, alpha_max=triangle.alpha_max, max_triangle_size=triangle.max_triangle_size))
    return individual.with_repr(new_repr)


def color_creep_mutation(individual, mutation_prob, color_sigma=0.04, **kwargs):
    """
    Mutate only the RGBA channels of each triangle (genes 6..9).

    Useful late in the run, when the triangle geometry is roughly correct
    but the colors still need fine-tuning. Decouples geometric exploration
    from colour exploitation - similar in spirit to the staged operators of
    Mantere & Koljonen, "Image Optimization by Genetic Algorithms"
    (Soft Computing 2008).

    Parameters:
    - individual (Individual): The individual to be mutated.
    - mutation_prob (float): The probability of mutating each triangle.
    - color_sigma (float): The standard deviation of the Gaussian perturbation for color genes.

    Returns:
    - Individual: A new individual resulting from mutation.

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
            new_genes[9] = clip_alpha(new_genes[9], alpha_min=individual.alpha_min, alpha_max=individual.alpha_max)
            new_tri = Triangle.__new__(Triangle)
            new_tri.repr = new_genes
            new_tri.alpha_min = triangle.alpha_min
            new_tri.alpha_max = triangle.alpha_max
            new_tri.max_triangle_size = triangle.max_triangle_size
            new_repr.append(new_tri)
        else:
            new_repr.append(triangle.copy())
    return individual.with_repr(new_repr)




def adaptive_mutation_schedule(individual, mut_prob, current_gen=0, max_gens=100, verbose=False, **kwargs):
    """
    Switches mutation operator based on the phase of evolution:
      - Initial phase  (< 40%) : Full replacement  → maximum exploration
      - Mid phase      (40-75%): Gaussian           → uniform fine-tuning (best overall)
      - Final phase    (> 75%) : Color creep        → color refinement only

    The VCF mutation is not used here because Gaussian outperformed it
    consistently across runs, and color creep is more targeted for late
    convergence than VCF's order mutation.

    Parameters:
    - individual (Individual): The individual to be mutated.
    - mut_prob (float): Mutation probability.
    - current_gen (int): Current generation number.
    - max_gens (int): Total number of generations.
    - verbose (bool): If True, prints the chosen mutation type.

    Returns:
    - Individual: A new individual resulting from mutation.
    """
    phase = current_gen / max_gens

    if phase < 0.40:
        if verbose: print(f"Gen {current_gen}: Full replacement")
        return triangle_mutation_full(individual, mut_prob)
    elif phase < 0.85:
        if verbose: print(f"Gen {current_gen}: Gaussian")
        return gaussian_gene_mutation(individual, mut_prob)
    else:
        if verbose: print(f"Gen {current_gen}: Color creep")
        return color_creep_mutation(individual, mut_prob)