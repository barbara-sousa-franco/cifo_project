# DEFINE OPERATORS - SELECTION, CROSSOVER, MUTATION

from copy import deepcopy
import random
from solution import Individual, Triangle, IMG_WIDTH, IMG_HEIGHT


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


# def triangle_crossover_double_cut(parent1, parent2, crossover_prob, **kwargs):
#     """Performs double-point crossover between two parent individuals.
#     Two crossover points are randomly selected, and the segments between these points are swapped 
#     between the parents to create two children.
#     Parameters:
#         - parent1 (Individual): The first parent individual.
#         - parent2 (Individual): The second parent individual.
#         - crossover_prob (float): The probability of performing crossover.
#     Returns:
#         - tuple: A tuple containing the two child individuals resulting from crossover.
#     """
#     if random.random() <= crossover_prob: # verify if crossover should occur based on the given probability
#         n = len(parent1.repr) # number of triangles in the individual's representation. It will always be 100 in our case, but we can keep it general for flexibility.

#         # Select two random crossover points (point1 < point2)
#         point1 = random.randint(1, n - 2)
#         point2 = random.randint(point1 + 1, n - 1)
        
#         # Two-point crossover: swap the segments between point1 and point2
#         child1_triangles = (parent1.repr[:point1] + 
#                            parent2.repr[point1:point2] + 
#                            parent1.repr[point2:])
#         child2_triangles = (parent2.repr[:point1] + 
#                            parent1.repr[point1:point2] + 
#                            parent2.repr[point2:])
        
#         # Create new child individuals with the new triangle lists
#         child1 = Individual(child1_triangles)
#         child2 = Individual(child2_triangles)
#     else: # If crossover does not occur, return deep copies of the parents
#         child1 = deepcopy(parent1)
#         child2 = deepcopy(parent2)
    
#     return child1, child2


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
    Muda o operador de crossover com base na fase da evolução:
      - Fase inicial  (< 50% das gerações) : Uniform      → máxima exploração
      - Fase intermédia (50–85%)           : K-Point      → mistura estruturada
      - Fase final    (> 85% das gerações) : Red.Surrogate → foco nas diferenças reais
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

        if "vertices" in mutations:
            idx = random.choice([0, 2, 4])
            triangle.repr[idx] = max(0.0, min(1.0, triangle.repr[idx] + random.gauss(0, 0.05)))
            triangle.repr[idx + 1] = max(0.0, min(1.0, triangle.repr[idx + 1] + random.gauss(0, 0.05)))

        if "color" in mutations:
            for j in range(6, 10):
                triangle.repr[j] = max(0.0, min(1.0, triangle.repr[j] + random.gauss(0, 0.05)))

        if "order" in mutations:
            # Find triangles that overlap with this one 
            overlapping = [j for j, t in enumerate(new_repr) 
                           if j != i and triangles_overlap(triangle, t)]
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

        # Replace the triangle with a completely new random triangle
        new_repr[i] = Triangle()

    return individual.with_repr(new_repr)

