import random


def tournament_selection(population: list, n_to_select: int, tournament_size: int, **kw_args):
    """Selects n individuals from the population according  to the  tournament  selection approach used  in genetic
    algorithms.

    Args:
        population (list): Contains all individuals of a population.
        n_to_select (int): The number of individuals to select from the population.
        tournament_size (int): Size of the subset used in a tournament.
    """
    winners = [perform_tournament(population, tournament_size, **kw_args) for _ in range(n_to_select)]
    winners.sort(key=lambda x: x.calculate_fitness(**kw_args), reverse=True)
    return winners


def perform_tournament(population: list, tournament_size: int, **kwargs):
    """Randomly selects a subset from a population and returns the best individual.

    Args:
        population (list): Contains all individuals of a population.
        tournament_size (int): Number of individuals in the subset (tournament)
    """
    subset = random.sample(population, k=tournament_size)
    subset.sort(key=lambda x: x.calculate_fitness(**kwargs), reverse=True)
    return subset[0]


def roulette_wheel_selection(population: list, size: int, **kw_args):
    """Returns a population according to the roulette wheel selection
    approach used in genetic algorithms.

    Args:
        population (list): Contains all individuals of a population.
        size (int): Size of the returned population
    """
    new_population = []
    sum_fitness = sum([ind.calculate_fitness(**kw_args) for ind in population])
    while len(new_population) < size:
        for ind in population:
            selection_prob = ind.calculate_fitness(**kw_args) / sum_fitness
            if random.random() < selection_prob:
                new_population.append(ind)
                if len(new_population) == size:
                    break
    return new_population


def rejuvenate_population(population: list, old_population: list, children: list, n_youngest: list, **kw_args):
    """Keeps the n_youngest individuals in the generation.

    Args:
        population (list): Contains all individuals of the current population.
        old_population (list): Contains all individuals of the last population.
        children (list): Contains all individuals that are children of the current population.
        n_youngest (int): Number of youngest individuals that should be kept
    """
    youngest = population[:-1-n_youngest] + old_population + children
    ids = []

    # remove duplicates
    for i in list(youngest):
        if i.id in ids:
            youngest.remove(i)
        else:
            ids.append(i.id)

    youngest.sort(key=lambda x: x.generated_in_generation, reverse=True)
    np = population[:len(population) - n_youngest] + youngest[:n_youngest]
    np.sort(key=lambda x: x.calculate_fitness(**kw_args), reverse=True)
    return np

