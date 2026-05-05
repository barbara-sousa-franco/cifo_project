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
def triangle_mutation(individual, mutation_prob):
    """ Performs mutation on an individual by randomly perturbing the vertices or color of its triangles, 
    or replacing a triangle entirely. Each triangle in the individual's representation has a chance to mutate 
    based on the given mutation probability.
    Parameters:
        - individual (Individual): The individual to be mutated.
        - mutation_prob (float): The probability of mutating each triangle.
    Returns:
        - Individual: A new individual resulting from mutation.
    """
    individual = deepcopy(individual)

    for triangle in individual.repr:
        if random.random() > mutation_prob:
            continue  # este triângulo não muta

        mutation_type = random.choice(["vertices", "color", "full"])

        if mutation_type == "vertices":
            # Escolhe um vértice aleatório (0, 2, ou 4 = índices de x) e perturba-o
            idx = random.choice([0, 2, 4])  # x do vértice 1, 2 ou 3
            triangle.repr[idx]   = max(0, min(IMG_WIDTH,  triangle.repr[idx]   + random.randint(-30, 30)))
            triangle.repr[idx+1] = max(0, min(IMG_HEIGHT, triangle.repr[idx+1] + random.randint(-30, 30)))

        elif mutation_type == "color":
            # Perturba r, g, b (índices 6, 7, 8) e alpha (índice 9)
            for i in range(6, 9):
                triangle.repr[i] = max(0, min(255, triangle.repr[i] + random.randint(-30, 30)))
            triangle.repr[9] = max(0.0, min(1.0, triangle.repr[9] + random.uniform(-0.1, 0.1)))

        elif mutation_type == "full":
            # Substitui o triângulo por um completamente novo
            triangle.repr = triangle.random_initial_representation()

    return individual
