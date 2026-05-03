# DEFINE GENETIC ALGORITHM

from operators import tournament_selection
from operators import triangle_crossover
from operators import triangle_mutation


def genetic_algorithm(population_size, num_generations, crossover_prob, mutation_prob, tournament_size=2):
    """Implements a genetic algorithm to evolve a population of individuals (solutions) over a specified number of generations.
    Parameters:
        - population_size (int): The number of individuals in the population.
        - num_generations (int): The number of generations to run the algorithm.
        - crossover_prob (float): The probability of performing crossover between two parents.
        - mutation_prob (float): The probability of mutating an individual.
        - tournament_size (int): The number of individuals to participate in the tournament selection.
    Returns:
        - Individual: The best individual found after running the genetic algorithm.
    """

    # 1. INITIALIZATION — create an initial population of individuals
    population = [Individual() for _ in range(population_size)]

    # Record the best individual from the initial population
    best = min(population, key=lambda ind: ind.fitness())

    # 2. PRINCIPAL LOOP — run for N generations
    for generation in range(num_generations):
        new_population = []

        # 3. SELECTION + CROSSOVER — generate new individuals through selection and crossover
        while len(new_population) < population_size:
            parent1 = tournament_selection(population, tournament_size)
            parent2 = tournament_selection(population, tournament_size)
            child1, child2 = triangle_crossover(parent1, parent2, crossover_prob)
            new_population.extend([child1, child2])

        # 4. MUTATION — apply mutation to the entire new population
        new_population = [triangle_mutation(ind, mutation_prob) for ind in new_population]

        # 5. SUBSTITUTION — the new population replaces the old one
        population = new_population[:population_size] # Ensure population size is maintained

        # 6. BEST INDIVIDUAL — record the progress
        generation_best = min(population, key=lambda ind: ind.fitness()) # Find the best individual in the current generation

        # Update the overall best individual if the current generation's best is better
        if generation_best.fitness() < best.fitness():
            best = generation_best

        print(f"Geração {generation+1} | Melhor RMSE: {best.fitness():.4f}")

    return best