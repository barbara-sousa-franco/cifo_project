# DEFINE OPERATORS - SELECTION, CROSSOVER, MUTATION

from copy import deepcopy
import random
from solution import Individual, Triangle, IMG_WIDTH, IMG_HEIGHT


# SELECTION: Tournament Selection
def tournament_selection(target, population: list[Individual], tournament_size: int = 2):
    """Selects an individual from the population using tournament selection.
    Parameters:
        - population (list[Individual]): The list of individuals in the population.
        - tournament_size (int): The number of individuals to participate in the tournament.
        - target (Image): The target image used to evaluate fitness during selection.
    Returns:
        - Individual: A copy of the selected individual.
    """
    # Select a random subset of individuals for the tournament
    tournament = random.choices(population, k=tournament_size)
    best_individual = min(tournament, key=lambda ind: ind.fitness(target))

    return deepcopy(best_individual)

# CROSSOVER:
def triangle_crossover(parent1, parent2, crossover_prob):
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
        point = random.randint(1, len(parent1.triangles) - 1)

        # Single-point crossover - swap the segments after the crossover point
        child1_triangles = parent1.repr[:point] + parent2.repr[point:]
        child2_triangles = parent2.repr[:point] + parent1.repr[point:]

        # Create new child individuals with the new triangle lists
        child1 = Individual(child1_triangles)
        child2 = Individual(child2_triangles)

    else: # If crossover does not occur, return deep copies of the parents
        child1 = deepcopy(parent1)
        child2 = deepcopy(parent2)

    return child1, child2


# def triangle_crossover_double_cut(parent1, parent2, crossover_prob):
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

# MUTATION:

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
    individual = deepcopy(individual)
    
    for i, triangle in enumerate(individual.repr):
        if random.random() > mutation_prob:
            continue
        
        # Choose between 1 to 4 mutations to apply to this triangle
        mutations = random.sample(["vertices", "color", "full", "order"], 
                                   k=random.randint(1, 4))
        
        # If "full" was chosen, it doesn't make sense to apply the other mutations, so we can skip them
        if "full" in mutations:
            triangle.repr = triangle.random_initial_representation()
            continue
        
        if "vertices" in mutations:
            idx = random.choice([0, 2, 4])
            triangle.repr[idx] = int(max(0, min(IMG_WIDTH, triangle.repr[idx] + random.gauss(0, 15))))
            triangle.repr[idx + 1] = int(max(0, min(IMG_HEIGHT, triangle.repr[idx + 1] + random.gauss(0, 15))))
        
        if "color" in mutations:
            for j in range(6, 9):
                triangle.repr[j] = int(max(0, min(255, triangle.repr[j] + random.gauss(0, 20))))
            triangle.repr[9] = max(0.0, min(1.0, triangle.repr[9] + random.gauss(0, 0.05)))
        
        if "order" in mutations:
            # Find triangles that overlap with this one 
            overlapping = [j for j, t in enumerate(individual.repr) 
                          if j != i and triangles_overlap(triangle, t)]
            
            if overlapping:
                j = random.choice(overlapping)
                individual.repr[i], individual.repr[j] = individual.repr[j], individual.repr[i]
    
    return individual


def triangle_mutation_full(individual, mutation_prob):
    """ Performs mutation on an individual by replacing entire triangles with new random triangles. Each triangle in the individual's representation has a chance to mutate based on the given mutation probability.
    Parameters:
        - individual (Individual): The individual to be mutated.
        - mutation_prob (float): The probability of mutating each triangle.
    Returns:
        - Individual: A new individual resulting from mutation.
    """
    individual = deepcopy(individual)

    for triangle in individual.repr:
        if random.random() > mutation_prob:
            continue  # the triangle does not mutate

        else:
            # Replace the triangle with a completely new random triangle
            triangle.repr = triangle.random_initial_representation()

    return individual