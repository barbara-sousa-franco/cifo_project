# DEFINE OPERATORS - SELECTION, CROSSOVER, MUTATION

from copy import deepcopy
from random import random


# SELECTION: Tournament Selection
def tournament_selection(population: list[Solution], tournament_size: int = 2):
    """Selects an individual from the population using tournament selection.
    Parameters:
        - population (list[Solution]): The list of individuals in the population.
        - tournament_size (int): The number of individuals to participate in the tournament.
    Returns:
        - Solution: A copy of the selected individual.
    """
    # Select a random subset of individuals for the tournament
    tournament = random.choices(population, k=tournament_size)
    best_individual = min(tournament, key=lambda ind: ind.fitness())

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
        child1_triangles = parent1.triangles[:point] + parent2.triangles[point:]
        child2_triangles = parent2.triangles[:point] + parent1.triangles[point:]

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
#         n = len(parent1.triangles) # number of triangles in the individual's representation. It will always be 100 in our case, but we can keep it general for flexibility.

#         # Select two random crossover points (point1 < point2)
#         point1 = random.randint(1, n - 2)
#         point2 = random.randint(point1 + 1, n - 1)
        
#         # Two-point crossover: swap the segments between point1 and point2
#         child1_triangles = (parent1.triangles[:point1] + 
#                            parent2.triangles[point1:point2] + 
#                            parent1.triangles[point2:])
#         child2_triangles = (parent2.triangles[:point1] + 
#                            parent1.triangles[point1:point2] + 
#                            parent2.triangles[point2:])
        
#         # Create new child individuals with the new triangle lists
#         child1 = Individual(child1_triangles)
#         child2 = Individual(child2_triangles)
#     else: # If crossover does not occur, return deep copies of the parents
#         child1 = deepcopy(parent1)
#         child2 = deepcopy(parent2)
    
#     return child1, child2

# MUTATION:
def mutate(self, img_width=300, img_height=400):
    """Mutates the individual's triangles by randomly modifying their vertices, colors, or replacing them entirely.
    The mutation type is randomly selected for each triangle, and the modifications are constrained to ensure valid triangle properties.
    Parameters:
        - img_width (int): The width of the image, used to constrain vertex positions.
        - img_height (int): The height of the image, used to constrain vertex positions.
    Returns:    
        - None: The function modifies the individual's triangles in place.
    """
    # Randomly select a mutation type for each triangle: "vertices", "color", or "full"
    mutation_type = random.choice(["vertices", "color", "full"])
    
    if mutation_type == "vertices":
        # Slightly modify the vertices of a randomly selected triangle
        idx = random.randint(0, 2)
        self.vertices[idx] = (
            max(0, min(img_width,  self.vertices[idx][0] + random.randint(-30, 30))),
            max(0, min(img_height, self.vertices[idx][1] + random.randint(-30, 30)))
        )
    
    elif mutation_type == "color":
        # Slightly modify the color of a randomly selected triangle
        self.color = tuple(
            max(0, min(255, c + random.randint(-30, 30)))
            for c in self.color
        )
    
    elif mutation_type == "full":
        # Completely replace a randomly selected triangle 
        self.__init__(img_width, img_height)