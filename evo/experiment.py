from evo import params, custom_fitness
from evo.individual import Individual
from evo.population import divide_population_into_islands
from evo.evolution import Evolution


class Experiment:

    def __init__(self, fitness, **kwargs):
        self.kwargs = kwargs
        custom_fitness.set_custom_fitness(fitness)
        self.population = self.generate_initial_population()
        self.evolution = Evolution(self.population)

    def generate_initial_population(self, log_parameters=True):
        population = [Individual(params=params) for _ in range(params.population_size)]
        population = divide_population_into_islands(population, params.n_sub_populations)
        return population

    def run_experiments(self):
        return self.evolution.begin_evolution(**self.kwargs)
