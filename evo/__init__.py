"""Genetic Algorithm Baseline from https://arxiv.org/pdf/2302.01303.pdf"""

import datetime

class CustomFitness:
    def __init__(self):
        self.custom_fitness = None

    def set_custom_fitness(self, fitness):
        self.custom_fitness = fitness

class Config:
  def __init__(self, config):
    for k, v in config.items(): setattr(self, k, v)

custom_fitness = CustomFitness()

params = Config(dict(
    single_gate_flip_mutation_rate=0.3,
    swap_control_qubit_mutation_rate=0.3,
    mutate_n_qubit_mutation_rate=0.3,         # rate for which the number of qbits gets mutated
    mutate_n_gates_mutation_rate=0.3,         # rate for which the number of gates gets mutated
    swap_columns_mutation_rate=0.3,           # mutation rate to swap columns
    gate_parameters_mutation_rate=0.3,        # rate for which gate parameters are mutated
    single_point_crossover_rate=0.3,
    multi_point_crossover_rate=0.3,
    blockwise_crossover_rate=0,
    n_generations=50,                         # number of generations the GA process is running
    population_size=20,                       # size of the population for each generation
    n_sub_populations=1,
    offspring_rate=0.3,                       # the rate for which individuals an offspring gets produced
    migration_rate=0.1,
    migrate_every_n_generations=20,
    fitness_function_name="custom", 
    parent_selection_method="random",         # random (each individual is equally likely to be selected)  / tournament 
    survivor_selection_method="strongest",    # strongest (best n individuals always survive)  / tournament
    tournament_size=10,                       # number of individuals in a tournament
    youngest_ratio=0,                         # percentage of the youngest individuals that should be kept per generation
    single_parent=True,
    calculate_diversity=False,
    constant_n_qubits=None,                   # Only required if n_qubits must be constant
    parameter_mutation='gaussian',            # decides how the gate parameters are mutated, either 'uniform' or 'gaussian'
    init_min_gates=0,                         # minimal gates that are used for each individual
    init_max_gates=0,                         # maximal gates that are used for each individual
    parameter_init=None,                      # constant for initialization of every parameter, if set to None then all are random
    data_dim=0,                               # dimension of the data input
    gatesets=[['cx', 'rx', 'cp', 'p']],
    data_gates=[],                            # the parametrized gates that can be data (reuploading) gates, if empty then no reupload will happen
    experiment_date=datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
    log_every_n_generation=20
))
