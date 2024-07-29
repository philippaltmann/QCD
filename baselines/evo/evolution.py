import uuid

from evo.individual import Individual
from evo.crossover import *
from evo.mutation import *
from evo.selection import tournament_selection, roulette_wheel_selection, rejuvenate_population
from evo.population import migration
import numpy as np

from typing import List
from evo.population import update_population_statistics

class Evolution:

  def __init__(self, population: List[Individual]):
    self.population = population
    self.n_children = int(int(params.population_size / params.n_sub_populations) * params.offspring_rate)

  def begin_evolution(self, metrics={}, **kw_args):
    stats = {k:[] for k in metrics}; m = int(100 / params.n_generations); assert m < params.population_size
    best_fitness_per_gen, avg_fitness_per_gen, avg_age_per_gen = [[] for _ in range(params.n_sub_populations)], [[] for _ in range(params.n_sub_populations)], [[] for _ in range(params.n_sub_populations)]
    diversity_per_gen = [[] for _ in range(params.n_sub_populations)] if params.calculate_diversity else None

    for current_gen in range(params.n_generations):
      if params.migrate_every_n_generations != 0 and current_gen % params.migrate_every_n_generations == 0:
        migration(self.population, migration_rate=params.migration_rate)

      for sub_pop_indx, population in enumerate(self.population):

        # Sort population by fitness
        population.sort(key=lambda x: x.calculate_fitness(**kw_args), reverse=True)
        for k,f in metrics.items(): 
          #  stats[k].append(f(population[0]))
           stats[k].extend([f(i) for i in population[:m]])

        if diversity_per_gen is not None:
          update_population_statistics(population, current_gen, best_fitness_per_gen[sub_pop_indx],
                                       avg_fitness_per_gen[sub_pop_indx], avg_age_per_gen[sub_pop_indx],
                                       diversity_per_gen[sub_pop_indx])
        else: update_population_statistics(population, current_gen, best_fitness_per_gen[sub_pop_indx],
                                           avg_fitness_per_gen[sub_pop_indx], avg_age_per_gen[sub_pop_indx],
                                           None)

        # Produce new individuals
        children = self.produce_offspring(population=population, current_gen=current_gen, **kw_args) 
        
        if params.youngest_ratio > 0: old_population = self.population.copy()

        # Determine individuals that will be part of the next generation
        population = self.replacement(population, children, **kw_args)
        # keep the youngest individuals if parameter is set
        if params.youngest_ratio > 0:
          n_youngest = int(params.youngest_ratio * params.population_size)
          population = rejuvenate_population(population, old_population, children, n_youngest, **kw_args)
    
    # Sort last population by fitness and return best solution
    population = self.population[-1]
    population.sort(key=lambda x: x.calculate_fitness(**kw_args), reverse=True)
    return stats, population
    # mean_fitness = np.array([i.fitness for p in self.population for i in p]).mean()
    # return population, mean_fitness
    
  def produce_offspring(self, population: List[Individual], current_gen: int = 0, **kw_args) -> \
          List[Individual]:
      children = []
      while len(children) < self.n_children:
          parents = self.select_parents(population, **kw_args)
          children_created = self.create_children(parents)
          for child in children_created: children.append(child)
      for child in children:
          apply_mutations(child)
          child.solution = child.solution  # to sync also with qc
          child.set_generated_in_generation(current_gen + 1)
          child.id = uuid.uuid4()

      return children

  def select_parents(self, population, **kw_args) -> List[Individual]:
      """Select parents following the selection method specified in evolution parameters.
          """
      if params.parent_selection_method == "random":
          parents = [random.choice(population), random.choice(population)]
      elif params.parent_selection_method == "tournament":
          parents = tournament_selection(population, n_to_select=2,
                                          tournament_size=params.tournament_size,
                                          **kw_args)
      elif params.parent_selection_method == "roulette_wheel":
          parents = roulette_wheel_selection(population, size=2, **kw_args)
      else:
          raise ValueError("Invalid parent selection method!")
      return parents

  @staticmethod
  def create_children(parents: List[Individual]) -> List[Individual]:
      """Create a new child by recombining the genome of two parent individuals

      Args:
          parents (:obj:`list` of :obj:`Individual`): parent individuals

      Returns:
          :obj:`Individual`: Returns the generated child individual.
      """
      children, crossover_applied = apply_crossover(parents)
      if not crossover_applied:
          if random.random() < params.single_parent:
              child = Individual()
              random_parent = random.choice(parents)
              child.n_gates_per_qubit = random_parent.n_gates_per_qubit
              child.n_qubits = random_parent.n_qubits
              child.solution = deepcopy(random_parent.solution)
              child.gateset = random_parent.gateset
          else:
              child = Individual(max_qubits=parents[0].n_qubits)
          children.append(child)
      return children

  def replacement(self, population, children, **kw_args):
      if params.survivor_selection_method == "strongest":
          for idx in range(self.n_children):
              population[-idx - 1] = children[idx]
      elif params.survivor_selection_method == "tournament":
          population = tournament_selection(population + children,
                                            n_to_select=params.population_size,
                                            tournament_size=params.tournament_size,
                                            **kw_args)
      else:
          raise ValueError("Invalid survivor selection method!")
      return population

