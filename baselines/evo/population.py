import random

def divide_population_into_islands(initial_population, n_sub_populations):
    full_population = []
    sub_population_size = int(len(initial_population) / n_sub_populations)
    population_index = 0
    for idx in range(n_sub_populations):
        sub_population = []
        for jdx in range(population_index, population_index + sub_population_size):
            sub_population.append(initial_population[jdx])
        population_index += sub_population_size
        full_population.append(sub_population)
    return full_population


def migration(population: list, migration_rate):
    n_individuals_to_migrate = int(len(population[0]) * migration_rate)
    random.shuffle(population)
    for idx in range(0, len(population)-1):
        swap_individual(population[idx], population[idx+1], n_individuals_to_migrate)


def swap_individual(sub_population_a, sub_population_b, n_to_migrate):
    random_ind_a, random_ind_b = random.sample([idx for idx in range(len(sub_population_a))], k=n_to_migrate),\
                                 random.sample([idx for idx in range(len(sub_population_b))], k=n_to_migrate)
    temp = [sub_population_a[random_ind_a[idx]] for idx in range(n_to_migrate)]
    for idx in range(n_to_migrate):
        sub_population_a[random_ind_a[idx]] = sub_population_b[random_ind_b[idx]]
        sub_population_b[random_ind_b[idx]] = temp[idx]



def update_population_statistics(population, current_gen, best_fitness_per_gen, avg_fitness_per_gen, avg_age_per_gen,
                                 diversity_per_gen=None):
    best_fitness_per_gen.append(population[0].fitness)
    avg_fitness_per_gen.append(sum([x.fitness for x in population]) / len(population))
    avg_age = sum([current_gen - x.generated_in_generation for x in population]) / len(population)
    avg_age_per_gen.append(avg_age)
    if diversity_per_gen is not None:
        diversity = calculate_diversity_of_population(population)
        diversity_per_gen.append(diversity)

def calculate_diversity_of_population(population):
    distances = []
    for ind in population:
        for other_ind in population:
            if ind == other_ind:
                continue
            distances.append(compare_circuits(ind, other_ind))
    return sum(distances) / len(distances)


# Placeholder: find better way to compare circuits
def compare_circuits(ind_a, ind_b):
    distance = 0
    for row_a, row_b in zip(ind_a.solution, ind_b.solution):
        for gate_a, gate_b in zip(row_a, row_b):
            if gate_a != gate_b:
                distance += 1
        distance += abs(ind_a.n_gates_per_qubit - ind_b.n_gates_per_qubit)
    distance += abs(ind_a.n_qubits - ind_b.n_qubits) * ind_a.n_gates_per_qubit if ind_a.n_qubits > ind_b.n_qubits else abs(ind_a.n_qubits - ind_b.n_qubits) * ind_b.n_gates_per_qubit
    return distance
