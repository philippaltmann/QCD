import math
import random

from evo import params
from evo.gate import create_random_gate
from evo.individual import Individual


def apply_mutations(child: Individual):
    """Applies each mutation to an individual according to the probability specified in params.
    Each mutation can have a different probability assigned. Probabilities do not need to sum up to 1.

    Args:
        child (Individual): The individual on which the mutation should be applied.
    """
    if child.n_qubits == 0 or child.n_gates_per_qubit==0: return
    if random.random() < params.single_gate_flip_mutation_rate:
        single_gate_flip(child)
    if random.random() < params.swap_control_qubit_mutation_rate:
        swap_control_qubit(child)
    if params.constant_n_qubits is None and random.random() < params.mutate_n_qubit_mutation_rate:
        mutate_n_qubits(child)
    if random.random() < params.mutate_n_gates_mutation_rate:
        mutate_n_gates(child)
    if random.random() < params.swap_columns_mutation_rate:
        swap_columns(child)
    if random.random() < params.gate_parameters_mutation_rate:
        mutate_gate_parameters(child)


def single_gate_flip(child: Individual):
    """Randomly selects a gate and replaces it with a random gate. Also replaces gates of affected qubits with random
    gates.

    Args:
        child (Individual): The individual on which the mutation should be applied.
    """

    random_qubit, random_column = random.randint(0, child.n_qubits - 1), random.randint(0, child.n_gates_per_qubit - 1)
    affected_qubits = child.solution[random_qubit][random_column].affected_qubits

    # Determine if it's a gate acting on more than one qubits. If so, also mutate that gate
    if len(affected_qubits) > 1:
        for i in affected_qubits:
            child.solution[i][random_column] = create_random_gate(n_qubits=child.n_qubits, qubit_id=i,
                                                                  gateset=child.gateset,
                                                                  max_affected_qubits=True)
    else:
        child.solution[random_qubit][random_column] = create_random_gate(n_qubits=child.n_qubits, qubit_id=random_qubit,
                                                                         gateset=child.gateset,
                                                                         max_affected_qubits=True)


def swap_control_qubit(child: Individual, max_loop_iterations=10):
    """Randomly searches for a controlled-gate and swaps control and target qubit (if one exists).

    Args:
        child (Individual): The individual on which the mutation should be applied.
        max_loop_iterations (int): Specifies for how many iterations maximally to search for a controlled-gate
    """
    for _ in range(max_loop_iterations):
        # Choose random qubit and column
        qubit_id, column_id = random.randint(0, child.n_qubits - 1), random.randint(0, child.n_gates_per_qubit - 1)
        # Determine if circuit contains a controlled gate
        if len(child.solution[qubit_id][column_id].control_qubits) > 0:
            # Found a controlled gate, now swap control and target qubits
            control_qubits = child.solution[qubit_id][column_id].control_qubits
            target_qubits = child.solution[qubit_id][column_id].target_qubits
            random_control_qubit = random.choice(range(len(control_qubits)))

            temp = child.solution[qubit_id][column_id].control_qubits[random_control_qubit]
            control_qubits[random_control_qubit] = child.solution[qubit_id][column_id].target_qubits[0]
            target_qubits[0] = temp

            for idx in control_qubits + target_qubits:
                child.solution[idx][column_id].control_qubits = control_qubits
                child.solution[idx][column_id].target_qubits = target_qubits

            break


def mutate_n_qubits(child: Individual):
    """Adjusts the number of qubits in a circuit.

    Args:
        child (Individual): The individual on which the mutation should be applied.
    """
    current_qubits = child.n_qubits
    if current_qubits == 2:
        child.n_qubits += random.randint(1, 2)
    elif current_qubits == 3:
        child.n_qubits += random.choice([-1, 1])
    elif random.random() < 0.5:
        child.n_qubits += random.randint(1, 2)
    else:
        child.n_qubits -= random.randint(1, 2)

    # Adjust circuit
    if child.n_qubits < current_qubits:
        # Remove qubits
        for _ in range(abs(current_qubits - child.n_qubits)):
            child.solution.pop()
        repair_affected_qubits(child)
    else:
        # Add new qubits and gates (currently only adds single qubit gates)
        for qubit_id in range(current_qubits, child.n_qubits):
            child.solution.append([])
            for _ in range(child.n_gates_per_qubit):
                child.solution[qubit_id].append(
                    create_random_gate(n_qubits=child.n_qubits, qubit_id=qubit_id, gateset=child.gateset,
                                       max_affected_qubits=1))


def mutate_n_gates(child: Individual):
    """Adjusts the number of gates in a circuit.

    Args:
        child (Individual): The individual on which the mutation should be applied.
    """
    current_gates = child.n_gates_per_qubit

    if current_gates < 2:
        return child

    if current_gates == 2:
        child.n_gates_per_qubit += random.randint(1, 2)
    elif current_gates == 3:
        child.n_gates_per_qubit += random.choice([-1, 1])
    elif random.random() < 0.5:
        child.n_gates_per_qubit += random.randint(1, 2)
    else:
        child.n_gates_per_qubit -= random.randint(1, 2)

    # check max_gates and min_gates constraints
    if child.n_gates_per_qubit < params.init_min_gates:
        child.n_gates_per_qubit = params.init_min_gates
    elif child.n_gates_per_qubit > params.init_max_gates:
        child.n_gates_per_qubit = params.init_max_gates

    # Adjust circuit
    if child.n_gates_per_qubit < current_gates:
        # Remove gates (a column)
        for _ in range(abs(current_gates - child.n_gates_per_qubit)):
            for qubit_id in range(child.n_qubits):
                child.solution[qubit_id].pop()

    else:
        # Add new gates
        for qubit_id in range(child.n_qubits):
            for idx in range(current_gates, child.n_gates_per_qubit):
                # Could be adjusted such that not only single gates can be added
                child.solution[qubit_id].append(
                    create_random_gate(n_qubits=child.n_qubits, qubit_id=qubit_id, gateset=child.gateset,
                                       max_affected_qubits=1))


def repair_affected_qubits(child: Individual):
    """Used as part of mutation functions in order to adjust and fix solutions that have  been corrupted by a mutation.

    Args:
        child (Individual): The individual on which the mutation should be applied.
    """
    for column_id in range(child.n_gates_per_qubit):
        for qubit_id in range(child.n_qubits):
            affected_qubits = child.solution[qubit_id][column_id].affected_qubits
            for qubit in affected_qubits:
                if qubit >= child.n_qubits:
                    child.solution[qubit_id][column_id] = create_random_gate(n_qubits=child.n_qubits, qubit_id=qubit_id,
                                                                             gateset=child.gateset,
                                                                             max_affected_qubits=1)


def swap_columns(child: Individual):
    """Mutation that exchanges all gates from two randomly chosen columns of a circuit.

    Args:
        child (Individual): The individual on which the mutation should be applied.
    """
    if child.n_gates_per_qubit < 2:
        return child

    # Randomly choose two columns to swap
    column_1, column_2 = random.sample(range(child.n_gates_per_qubit), 2)
    # print('Column 1: {} Column 2: {}'.format(column_1, column_2))
    for qubit_id in range(child.n_qubits):
        temp = child.solution[qubit_id][column_1]
        child.solution[qubit_id][column_1] = child.solution[qubit_id][column_2]
        child.solution[qubit_id][column_2] = temp


def mutate_gate_parameters(child: Individual, max_loop_iterations=10):
    """Randomly selects a parameterised gate and adjusts its parameter (if such a gate is found).

    Args:
        child (Individual): The individual on which the mutation should be applied.
        max_loop_iterations (int): Specifies for how many iterations maximally to search for a controlled-gate
    """
    # Determine if circuit contains a parameterised gate
    for _ in range(max_loop_iterations):
        qubit_id, column_id = random.randint(0, child.n_qubits - 1), random.randint(0, child.n_gates_per_qubit - 1)
        if child.solution[qubit_id][column_id].parameters is not None \
                and len(child.solution[qubit_id][column_id].parameters) > 0:
            # print('Mutating qubit {} gate {}'.format(qubit_id, column_id))
            for idx in range(len(child.solution[qubit_id][column_id].parameters)):
                if params.parameter_mutation == 'uniform':
                    child.solution[qubit_id][column_id].parameters[idx] = random.uniform(-math.pi, math.pi)
                elif params.parameter_mutation == 'gaussian':
                    mu = child.solution[qubit_id][column_id].parameters[idx]
                    sigma = 0.25 * math.pi
                    child.solution[qubit_id][column_id].parameters[idx] = random.gauss(mu, sigma)

            return
