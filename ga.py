# DEFINE GENETIC ALGORITHM
from collections.abc import Callable

from solution import *
from operators import *
from operators import *

from copy import deepcopy
from time import time

def get_best_ind(population: list, maximization: bool = False):
    '''
    Returns the best individual in the population based on fitness.
    If maximization is True, returns the individual with the highest fitness; otherwise, returns the individual with the lowest fitness.
    This corresponds to returning the Elite=1.

        Args:
        population (list): The population of solutions.
        maximization (bool): If True (the default), considers that higher values of fitness are better; otherwise, the opposite.
    '''
    fitness_list = [ind.fitness() for ind in population]
    if maximization:
        return population[fitness_list.index(max(fitness_list))]
    else:
        return population[fitness_list.index(min(fitness_list))]
    

def genetic_algorithm(initial_population: list, max_generations: int, selection_algorithm: Callable, xo_method: Callable, mut_method: Callable,
    maximization: bool = False, xo_prob: float = 0.9, mut_prob: float = 0.05, elitism: bool = True, verbose: bool = False,):
    """
    Executes a genetic algorithm to optimize a population of solutions.

    Args:
        initial_population (list[Solution]): The starting population of solutions.
        max_generations (int): The maximum number of generations to evolve.
        selection_algorithm (Callable): Function used for selecting individuals.
        xo_method (Callable): Function used for crossover between two individuals.
        mut_method (Callable): Function used for mutating an individual.
        maximization (bool, optional): If True, maximizes the fitness function; otherwise, minimizes. Defaults to False.
        xo_prob (float, optional): Probability of applying crossover. Defaults to 0.9.
        mut_prob (float, optional): Probability of applying mutation. Defaults to 0.05.
        elitism (bool, optional): If True, carries the best individual to the next generation. Defaults to True.
        verbose (bool, optional): If True, prints detailed logs for debugging. Defaults to False.

    Returns:
        Solution: The best solution found on the last population after evolving for max_gen generations.
        list[float]: The fitness of the best individual over the generations
    """
    starttime = time()
    best_fitness_over_gens = []
    best_ind = None

    # 1. Initial population P - passed as argument
    population = initial_population

    # 2. Repeat until termination condition (number of generations) is reached
    for gen in range(1, max_generations + 1):
        gen_starttime = time()

        # Even if verbose is False, we will still print the generation number so that we're sure we're progressing, even if slowly.
        # Feel free to only print this information if verbose is True.
        print(f'-------------- Generation: {gen}/{max_generations}, duration: {gen_starttime - starttime:.2f}s --------------')

        # 2.1. Create an empty population P'
        new_population = []

        # 2.2. If using elitism, insert best individual from P into P'
        if elitism:
            best = get_best_ind(population, maximization)
            new_population.append(best.with_repr(best.repr))
        
        # 2.3. Repeat until P' contains N individuals
        while len(new_population) < len(population):
            # 2.3.1. Choose 2 individuals from P using a selection algorithm
            first_ind = selection_algorithm(population, maximization)
            second_ind = selection_algorithm(population, maximization)

            
            # 2.3.2. Choose an operator between crossover and replication
            # 2.3.3. Apply the operator to generate the offspring
            # Our binary standard crossover function takes care of both the crossover and replication cases, since if the crossover
            # probability is not met, it simply returns the original individuals as offspring.

            offspring1, offspring2 = xo_method(first_ind, second_ind, xo_prob, verbose, current_gen=gen)
            
            # 2.3.4. Apply mutation to offspring1
            first_new_ind = mut_method(offspring1, mut_prob)
            # 2.3.5. Insert the mutated individual into P'
            new_population.append(first_new_ind)

            # Check if we can add the second offspring or if we've reached the population limit
            if len(new_population) < len(population):
                second_new_ind = mut_method(offspring2, mut_prob)
                new_population.append(second_new_ind)
                if verbose:
                    print(f'Mutated individuals added: {first_new_ind} and {second_new_ind}')
            else:
                if verbose:
                    print(f'Mutated individuals added: {first_new_ind}. Population limit reached, so second offspring not added.')
        
        # 2.4. Replace P with P'
        population = new_population

        # Record the best fitness in the current population and append it to list best_fitness_over_gens
        best_ind = get_best_ind(population, maximization)
        best_fitness_over_gens.append(best_ind.fitness())

        if verbose:
            print(f"Best individual/fitness in generation: {best_ind}, f={best_ind.fitness()}")

    # 3. Return the best individual in P + the best individual fitness at each generation
    return best_ind, best_fitness_over_gens
