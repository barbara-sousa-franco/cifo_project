# DEFINE GENETIC ALGORITHM
#
# Standard generational GA with optional EA enhancements:
#   - elitism (always carries the best individual through, unchanged)
#   - adaptive mutation (Rechenberg's 1/5 success rule, 1973)
#   - diversity injection (replace worst x% with random individuals when
#     phenotypic diversity collapses) - addresses premature convergence,
#     one of the failure modes discussed in Eiben & Smith §3.10.


from collections.abc import Callable
import numpy as np
from solution import Individual, Triangle
from time import time



def get_best_ind(population: list, maximization: bool = False):
    '''
    Returns the best individual in the population based on fitness.
    If maximization is True, returns the individual with the highest fitness; otherwise, returns the individual with the lowest fitness.
    This corresponds to returning the Elite=1.

    Parameters:
        - population (list): The population of solutions.
        - maximization (bool): If True (the default), considers that higher values of fitness are better; otherwise,
        the opposite.

    Returns:
        - The best individual in the population based on fitness.
    '''
    fitness_list = [ind.fitness() for ind in population]
    if maximization:
        return population[fitness_list.index(max(fitness_list))]
    else:
        return population[fitness_list.index(min(fitness_list))]


def population_fitness_std(population: list[Individual]) -> float:
    """Std-dev of fitness across the population — a cheap phenotypic
    diversity proxy. Low values mean every individual is roughly as good
    (or bad) as the others, i.e. the GA has converged.
    
    Parameters:
        - population (list[Individual]): The population of solutions.

    Returns:
        - float: The standard deviation of fitness across the population.
    
    """
    if not population:
        return 0.0
    fits = np.array([ind.fitness() for ind in population], dtype=np.float64)
    return float(fits.std())


def genetic_algorithm(
    initial_population: list,
    max_generations: int,
    selection_algorithm: Callable,
    xo_method: Callable,
    mut_method: Callable,
    maximization: bool = False,
    xo_prob: float = 0.9,
    mut_prob: float = 0.05,
    elitism: bool = True,
    verbose: bool = False,
    # --- Optional EA enhancements (default off, backwards compatible) ---
    adaptive_mutation: bool = False,
    adaptive_window: int = 10,
    adaptive_min: float = 0.001,
    adaptive_max: float = 0.5,
    diversity_injection: bool = False,
    diversity_threshold: float = 0.5,
    diversity_replace_frac: float = 0.2,
    # When provided, parent 2 is chosen by this function instead of by the
    # base ``selection_algorithm``. The function must accept
    # ``(population, parent1, maximization)`` and is used to implement
    # restricted mating (Goldberg 1989; Eshelman & Schaffer 1991).
    mate_selection_algorithm: Callable | None = None,
):
    """Generational GA with optional adaptive mutation and diversity injection.

    Parameters:
        - initial_population (list[Individual]): The starting population.
        - max_generations (int): Maximum number of generations to evolve.
        - selection_algorithm (Callable): Parent selection function. Called
            as ``selection_algorithm(population, maximization)``.
        - xo_method (Callable): Crossover operator. Returns two children.
        - mut_method (Callable): Mutation operator. Returns one individual.
        - maximization (bool): If True maximises fitness; else minimises.
        - xo_prob (float): Crossover probability.
        - mut_prob (float): Initial mutation probability. May change over
            generations if ``adaptive_mutation`` is True.
        - elitism (bool): If True, carries the best individual unchanged.
        - verbose (bool): If True, prints extra per-generation diagnostics
            (success rate, mut_prob, diversity injection events). The
            "Generation X/Y" header is always printed so progress is
            visible on long runs.
        - adaptive_mutation (bool): If True, apply Rechenberg's 1/5 success
            rule: keep a sliding window of how often offspring beat their
            parents, then scale mut_prob up if success rate > 0.2, down if
            success rate < 0.2. Clipped to [adaptive_min, adaptive_max].
        - adaptive_window (int): Number of generations in the success window.
        - adaptive_min, adaptive_max (float): Clipping bounds for mut_prob
            under the 1/5 rule.
        - diversity_injection (bool): If True, when phenotypic std-dev drops
            below ``diversity_threshold`` * initial std-dev, replace the
            worst ``diversity_replace_frac`` of the population with new
            random individuals. Helps escape premature convergence.
        - diversity_threshold, diversity_replace_frac (float): Trigger and
            replacement fraction for diversity injection.
        - mate_selection_algorithm (Callable | None): If provided, parent 2
            is selected by this function instead of ``selection_algorithm``
            (used to implement restricted mating).

    Returns:
        - Individual: Best individual found across all generations.
        - list[float]: Best fitness per generation (length == max_generations).
    """
    starttime = time()
    best_fitness_over_gens: list[float] = []
    best_ind = None

    # Initial population P (caller-provided)
    population = initial_population

    # Baseline diversity for the diversity-injection trigger.
    initial_std = population_fitness_std(population) if diversity_injection else 0.0

    # Sliding window for the 1/5 success rule.
    success_window: list[float] = []

    # Repeat until termination condition (number of generations) is reached
    for gen in range(1, max_generations + 1):
        gen_starttime = time()

        # Even if verbose is False, print the generation header so we know
        # we're progressing on slow runs.
        print(f'-------------- Generation: {gen}/{max_generations}, duration: {gen_starttime - starttime:.2f}s --------------')

        new_population = []


        # Elitism: copy the best individual into the new population.
        if elitism:
            best = get_best_ind(population, maximization)
            elite = best.with_repr(best.repr)
            elite._fitness = best._fitness
            new_population.append(elite)

        # Track how many children beat their parents this generation
        # (used for the 1/5 rule when adaptive_mutation is on).
        n_better = 0
        n_offspring = 0

        # Fill P' with offspring until it reaches the original size.
        while len(new_population) < len(population):

            first_ind = selection_algorithm(population, maximization)

            if mate_selection_algorithm is not None:

                # Restricted mating: parent 2 is constrained by its distance to parent 1.
                second_ind = mate_selection_algorithm(population, first_ind, maximization)

            else:
                second_ind = selection_algorithm(population, maximization)

            # Perform crossover
            offspring1, offspring2 = xo_method(
                first_ind, second_ind, xo_prob, verbose, current_gen=gen, max_gens=max_generations)

            # Determine the better parent (for tracking 1/5 success rate if adaptive_mutation is on).
            parent_best = (
                max(first_ind.fitness(), second_ind.fitness())
                if maximization else min(first_ind.fitness(), second_ind.fitness()))

            # Apply mutation to the first child, add it to the new population
            first_new_ind = mut_method(offspring1, mut_prob, verbose=verbose, current_gen=gen, max_gens=max_generations)
            new_population.append(first_new_ind)
            n_offspring += 1

            # Track if the first child is better than the better parent (for the 1/5 rule if adaptive_mutation is on).
            if (maximization and first_new_ind.fitness() > parent_best) or (
                not maximization and first_new_ind.fitness() < parent_best):

                n_better += 1

            # Apply mutation to the second child, add it to the new population if there's room
            #  (we might have already hit the target population size with the first child).
            if len(new_population) < len(population):
                second_new_ind = mut_method(offspring2, mut_prob, verbose=verbose, current_gen=gen, max_gens=max_generations)
                new_population.append(second_new_ind)
                n_offspring += 1

                # Again, track if the second child is better than the better parent
                #  (for the 1/5 rule if adaptive_mutation is on).
                if (maximization and second_new_ind.fitness() > parent_best) or (
                    not maximization and second_new_ind.fitness() < parent_best):

                    n_better += 1

        # Replace the old population with the new one.
        population = new_population

        # --- Adaptive mutation (Rechenberg 1/5 rule) ---
        # If too many children beat their parents, we are exploring too
        # gently: raise mut_prob. If too few, the perturbation is too
        # disruptive: lower mut_prob. Target success rate 0.2.
        if adaptive_mutation and n_offspring > 0:

            # Record the success rate for this generation
            success_window.append(n_better / n_offspring)

            # If we have enough data in the success window, adjust mut_prob based on the average success rate.
            if len(success_window) >= adaptive_window:
                rate = float(np.mean(success_window[-adaptive_window:])) # average of the last adaptive_window generations
                if rate > 0.2:
                    mut_prob = min(adaptive_max, mut_prob * 1.1)
                elif rate < 0.2:
                    mut_prob = max(adaptive_min, mut_prob * 0.9)

        # --- Diversity injection ---
        # If phenotypic diversity collapses below the trigger, replace the
        # worst fraction of the population with brand-new random individuals
        # so the GA can keep exploring.
        if diversity_injection and initial_std > 0:

            # Calculate the current phenotypic diversity (std-dev of fitness across the population).
            current_std = population_fitness_std(population)

            # If diversity is too low, inject new random individuals by replacing the worst performers.
            if current_std < diversity_threshold * initial_std:

                n_replace = max(1, int(diversity_replace_frac * len(population)))
                # Find the worst n_replace individuals (preserve elite).
                
                ordered = sorted(
                    range(len(population)),
                    key=lambda i: population[i].fitness(),
                    reverse=maximization,
                )
                victim_indices = ordered[-n_replace:]
                template = population[0]

                # Replace the worst individuals with new random ones.
                for idx in victim_indices:
                    new_ind = Individual(
                        target=template.target,
                        fitness_metric=template.fitness_metric,
                        target_lab=template.target_lab,
                        max_triangle_size=template.max_triangle_size,
                        alpha_min=template.alpha_min,
                        alpha_max=template.alpha_max,
                    )
                    population[idx] = new_ind

                if verbose:
                    print(f'  Diversity injection: replaced {n_replace} individuals '
                          f'(std {current_std:.3f} < {diversity_threshold:.2f}*{initial_std:.3f})')

        # Record the best fitness in the current population.
        best_ind = get_best_ind(population, maximization)
        best_fitness_over_gens.append(best_ind.fitness())

        if verbose:
            print(f"Best individual fitness in generation: {best_ind.fitness():.4f} | mut_prob={mut_prob:.4f}")

    # Return the best individual + per-generation curve.
    return best_ind, best_fitness_over_gens
