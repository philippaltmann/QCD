from __future__ import annotations

import random
from copy import deepcopy
import numpy as np

from evo import params
from evo.individual import Individual
from typing import List

from evo.gate import *
from evo.individual import Individual

def apply_crossover(original_parents: List[Individual]) -> tuple[List[Individual], bool]:
    crossover_applied, children_solutions = False, []
    parents = get_crossover_parents(original_parents)
    children = [Individual(is_empty=True), Individual(is_empty=True)]
    min_gates, max_gates = get_min_max_gates_in_parents(parents)
    if min_gates < 2:
        return children_solutions, crossover_applied
    min_qubits, max_qubits = get_min_max_qubits_in_parents(parents)
    if random.random() < params.single_point_crossover_rate:
        single_point_crossover_children = single_point_crossover(deepcopy(parents), deepcopy(children),
                                                                 [parent.solution for parent in parents],
                                                                 min_gates, min_qubits, max_qubits)
        for child in single_point_crossover_children:
            children_solutions.append(child)
            validate_solution_metadata(child.solution)

        crossover_applied = True
    if random.random() < params.multi_point_crossover_rate:
        if min_gates < 3:
            return children_solutions, crossover_applied
        else:
            multi_point_crossover_children = multi_point_crossover(deepcopy(parents), deepcopy(children),
                                                                   [parent.solution for parent in parents],
                                                                   min_gates, min_qubits, max_qubits)
            for child in multi_point_crossover_children:
                children_solutions.append(child)
                validate_solution_metadata(child.solution)
            crossover_applied = True
    if random.random() < params.blockwise_crossover_rate:
        blockwise_crossover_children = blockwise_crossover(deepcopy(parents), deepcopy(children),
                                                           [parent.solution for parent in parents],
                                                           min_gates, min_qubits, max_qubits)
        for child in blockwise_crossover_children:
            children_solutions.append(child)
            for idx_qubit, q in enumerate(child.solution):
                for idx_gate, g in enumerate(q):
                    if g.qubit_id != idx_qubit:
                        print(
                            f"Error in blockwise_crossover_children: gate {g.name} on the qubit {idx_qubit} has inconsistent metadata . ")
            for child in blockwise_crossover_children:
                for idx_qubit, q in enumerate(child.solution):
                    for idx_gate, g in enumerate(q):
                        if len(g.affected_qubits) > 1:
                            affected = g.affected_qubits
                            for a in affected:
                                if child.solution[a][idx_gate].name == "id":
                                    print("Inconsistent Circuit! after the blockwise crossover")

        crossover_applied = True
    # choose gateset for child from random parent
    for child in children_solutions:
        child.gateset = parents[random.randrange(len(parents))].gateset
    return children_solutions, crossover_applied


def get_crossover_parents(original_parents: list[Individual]):
    parents = deepcopy(original_parents)
    return parents


# def validate_solution_metadata(solution):
#    for idx_qubit, q in enumerate(solution):
#        for idx_gate, g in enumerate(q):
#            validate_qubit_id(g, idx_qubit)
#            validate_affected_qubits(solution, g, idx_gate)


# def validate_qubit_id(g, idx_qubit):
#    if g.qubit_id != idx_qubit:
#        raise AttributeError(
#            f"Error during the crossover has been occurred: gate {g.name} on the qubit {idx_qubit} "
#            f"has inconsistent metadata . ")


# def validate_affected_qubits(solution, g, idx_gate):
#    if len(g.affected_qubits) > 1:
#        affected = g.affected_qubits
#        for a in affected:
#            if solution[a][idx_gate].name == "id":
#                print(solution[a][idx_gate].name)
#                raise ValueError("Error in generating multi-qubit gates during the crossover!")


def single_point_crossover(parents: List[Individual], children: List[Individual], parent_solutions: list[list],
                           min_gates: int, min_qubits: int, max_qubits: int) -> list[Individual]:
    """
    Crossover on a single point
    p1: ------------|
    p2:             |---------------
    """
    children_solutions, smaller_parent_idx, offset = init_crossover_variables(parents, min_qubits, max_qubits)
    # find crossover point
    splitting_point = random.randint(1, min_gates)
    adjust_gates_of_the_smaller_parent(parents, parent_solutions, smaller_parent_idx, offset)
    pad_smaller_circuit_with_additional_qubits(parent_solutions, smaller_parent_idx, parents, offset, min_qubits,
                                               max_qubits)
    # crossover
    for qubit in range(max_qubits):
        children_solutions[0].append(deepcopy(
            parent_solutions[0][qubit][:splitting_point]) + deepcopy(parent_solutions[1][qubit][splitting_point:]))
        children_solutions[1].append(deepcopy(
            parent_solutions[1][qubit][:splitting_point]) + deepcopy(parent_solutions[0][qubit][splitting_point:]))

    final_adjustments_in_each_child(children, children_solutions, max_qubits, blockwise_cross=False)
    return children


def init_crossover_variables(parents: List[Individual], min_qubits: int, max_qubits: int):
    children_solutions = [[], []]
    smaller_parent_idx = get_smaller_parent_idx(parents, min_qubits)
    offset = random.randint(0, max_qubits - min_qubits)
    return children_solutions, smaller_parent_idx, offset


def multi_point_crossover(parents: List[Individual], children: List[Individual], parent_solutions: list[list],
                          min_gates: int, min_qubits: int, max_qubits: int) -> list[Individual]:
    """
    Crossover on multiple points
    p1:   |---------|         |------
    p2: --|         |---------|
    """
    children_solutions, smaller_parent_idx, offset = init_crossover_variables(parents, min_qubits, max_qubits)
    pad_smaller_parent_with_dummy_qubits(parents, parent_solutions, smaller_parent_idx, offset)
    pad_smaller_circuit_with_additional_qubits(parent_solutions, smaller_parent_idx, parents, offset, min_qubits,
                                               max_qubits)

    indices = list(range(1, min_gates))
    num_splitting_points = random.randint(2, len(indices))
    choices = random.choices(indices, k=num_splitting_points)
    splitting_points = list(set(choices))

    # crossover
    for qubit in range(max_qubits):
        p_s_0 = parent_solutions[0][qubit]
        p_s_1 = parent_solutions[1][qubit]
        qubit_splits = [_split_list(p_s_0, splitting_points),
                        _split_list(p_s_1, splitting_points)]
        children_solutions[0].append([])
        children_solutions[1].append([])

        for i, splits in enumerate(zip(qubit_splits[0], qubit_splits[1])):
            children_solutions[0][-1].extend(splits[i % 2])
            children_solutions[1][-1].extend(splits[(i + 1) % 2])

    final_adjustments_in_each_child(children, children_solutions, max_qubits, blockwise_cross=False)

    return children


def blockwise_crossover(parents: List[Individual], children: List[Individual], parent_solutions: list[list],
                        min_gates: int, min_qubits: int, max_qubits: int) -> Individual | list[Individual]:
    """
    Block includes multiple qubits and multiple gates; exact number is dynamic
    """
    children_solutions, smaller_parent_idx, offset = init_crossover_variables(parents, min_qubits, max_qubits)
    pad_smaller_parent_with_dummy_qubits(parents, parent_solutions, smaller_parent_idx, offset)
    pad_smaller_circuit_with_additional_qubits(parent_solutions, smaller_parent_idx,
                                               parents, offset, min_qubits, max_qubits)

    # pad qubits with identity gate to meet max_gates
    max_gates = max([len(q) for solution in parent_solutions for q in solution])
    for solution in parent_solutions:
        for i, qubit in enumerate(solution):
            if len(qubit) < max_gates:
                for _ in range(max_gates - len(qubit)):
                    qubit.append(get_identity_gate(i))

    # find crossover points
    # TODO: find good parameter to find a reasonable number of splits
    indices = list(range(1, max_gates))  # max_gates since the smallest parent is padded
    num_splitting_points = random.randint(1, len(indices))
    num_blocks = num_splitting_points + 1
    choices = random.choices(indices, k=num_splitting_points)
    splitting_points = list(set(choices))
    qubits_in_blocks = []
    for b in range(num_blocks):
        nr_of_concerned_qubits = random.randint(0, max_qubits)
        qubits_in_block = np.random.choice(max_qubits, nr_of_concerned_qubits, replace=False)
        qubits_in_blocks.append(qubits_in_block)

    # crossover
    for qubit in range(max_qubits):
        qubit_splits = [_split_list(deepcopy(parent_solutions[0][qubit]), splitting_points),
                        _split_list(deepcopy(parent_solutions[1][qubit]), splitting_points)]

        children_solutions[0].append([])
        children_solutions[1].append([])

        for i, splits in enumerate(zip(qubit_splits[0], qubit_splits[1])):
            if qubit in qubits_in_blocks[i]:
                children_solutions[0][-1].extend(deepcopy(splits[1]))
                children_solutions[1][-1].extend(deepcopy(splits[0]))
            else:
                children_solutions[0][-1].extend(deepcopy(splits[0]))
                children_solutions[1][-1].extend(deepcopy(splits[1]))

    # fix multi-qubit gates
    for c, solution in enumerate(children_solutions):
        for q in range(max_qubits):
            for g in range(len(solution[q]) - 1, -1, -1):
                gate = deepcopy(solution[q][g])
                if len(gate.affected_qubits) <= 1:
                    continue  # if this gate is not a multi-qubit gate

                modified_qubits = []
                for a in gate.affected_qubits:
                    if a != q:
                        # check if gate on qubit is another multi-qubit gate
                        # if yes: remove that
                        gate_on_affected_qubit = deepcopy(solution[a][g])
                        if gate.name != gate_on_affected_qubit.name and len(gate_on_affected_qubit.affected_qubits) > 1:
                            for other_gate_affected_qubit in gate_on_affected_qubit.affected_qubits:
                                if solution[other_gate_affected_qubit][g] == gate_on_affected_qubit:
                                    solution[other_gate_affected_qubit].insert(g + 1,
                                                                               get_identity_gate(
                                                                                   other_gate_affected_qubit))
                                    solution[other_gate_affected_qubit].remove(gate_on_affected_qubit)
                        # TODO: need validation
                        elif gate.name != gate_on_affected_qubit.name:
                            print("TODO Blockwise crossover: need validation")
                        elif gate.affected_qubits != gate_on_affected_qubit.affected_qubits:
                            print("TODO Blockwise crossover: need validation")
                        elif gate.control_qubits != gate_on_affected_qubit.control_qubits:
                            print("TODO Blockwise crossover: need validation")
                        elif gate.target_qubits != gate_on_affected_qubit.target_qubits:
                            continue
                if len(modified_qubits) > 0:
                    for i in range(max_qubits):
                        if i not in modified_qubits:
                            solution[i].insert(g + 1, get_identity_gate(i))
    final_adjustments_in_each_child(children, children_solutions, max_qubits, blockwise_cross=True)
    return children


def get_identity_gate(qubit_id: int) -> Gate:
    return Gate(name="id", qubit_id=qubit_id, affected_qubits=[qubit_id], target_qubits=[], control_qubits=[],
                parameters=[])


def get_max_gates_for_solutions(solutions: list):
    return max([len(q) for q in solutions])


def get_min_max_gates_in_parents(parents: List[Individual]) -> tuple[int, int]:
    return min([p.n_gates_per_qubit for p in parents]), max([p.n_gates_per_qubit for p in parents])


def get_min_max_qubits_in_parents(parents: List[Individual]):
    return min([p.n_qubits for p in parents]), max([p.n_qubits for p in parents])


def get_smaller_parent_idx(parents: List[Individual], min_qubits: int) -> int:
    return 0 if parents[0].n_qubits == min_qubits else 1


def set_n_qubits_and_n_gates(child: Individual, max_qubits: int, max_gates: int):
    child.n_qubits = max_qubits
    child.n_gates_per_qubit = max_gates


def prune_identity_gates(solution: List[List[Gate]]) -> List[List[Gate]]:
    n_gates = max([len(qubit) for qubit in solution])
    prunable_columns = []

    for column in range(0, n_gates):
        all_ids_in_column = True

        for qubit in solution:
            if not qubit[column].name == "id":
                all_ids_in_column = False

        if all_ids_in_column:
            prunable_columns.append(column)

    for qubit in solution:
        for column in range(len(qubit) - 1, -1, -1):
            if column in prunable_columns:
                qubit.remove(qubit[column])

    return solution


def adjust_gate_metadata(gate: Gate, q: int, offset: int):
    gate.qubit_id = q + offset
    gate.target_qubits = [t + offset for t in gate.target_qubits]
    gate.control_qubits = [c + offset for c in gate.control_qubits]
    gate.affected_qubits = [a + offset for a in gate.affected_qubits]


def final_adjustments_in_each_child(children: List[Individual], children_solutions: list[list],
                                    max_qubits: int, blockwise_cross: bool = False):
    for c, child in enumerate(children):
        max_gates = get_max_gates_for_solutions(children_solutions[c])
        set_n_qubits_and_n_gates(child, max_qubits, max_gates)
        pad_qubits_with_id_gate(children_solutions, max_gates, c)  # to meet max_gates
        if blockwise_cross:
            children_solutions[c] = prune_identity_gates(children_solutions[c])
            max_gates = get_max_gates_for_solutions(children_solutions[c])
            child.n_gates_per_qubit = max_gates
        fix_gates_metadata(children_solutions[c])
        child.solution = children_solutions[c]


def fix_gates_metadata(child):
    for idx_qubit, q in enumerate(child):
        for idx_gate, g in enumerate(q):
            if g.qubit_id != idx_qubit:
                g.qubit_id = idx_qubit


def adjust_gates_of_the_smaller_parent(parents: List[Individual], parent_solutions: list[list],
                                       smaller_parent_idx: int, offset: int):
    for column in range(parents[smaller_parent_idx].n_gates_per_qubit):
        visited_gates = []

        for q, qubit in enumerate(parent_solutions[smaller_parent_idx]):
            gate = parent_solutions[smaller_parent_idx][q][column]

            if gate in visited_gates:  # if this gate has been processed already: skip
                continue

            adjust_gate_metadata(gate, q, offset)
            visited_gates.append(gate)


def pad_smaller_parent_with_dummy_qubits(parents: List[Individual], parent_solutions: list[list],
                                         smaller_parent_idx: int, offset: int):
    for column in range(parents[smaller_parent_idx].n_gates_per_qubit):
        visited_gates = []

        for q, qubit in enumerate(parent_solutions[smaller_parent_idx]):
            gate = qubit[column]
            if gate in visited_gates:  # if this gate has been processed already: skip
                continue
            adjust_gate_metadata(gate, q, offset)
            visited_gates.append(gate)


def pad_smaller_circuit_with_additional_qubits(parent_solutions: list[list], smaller_parent_idx: int,
                                               parents: List[Individual], offset: int,
                                               min_qubits: int, max_qubits: int):
    parent_solutions[smaller_parent_idx] = [[get_identity_gate(i) for _ in
                                             range(parents[smaller_parent_idx].n_gates_per_qubit)] for i in
                                            range(offset)] + parent_solutions[smaller_parent_idx] + [
                                               [get_identity_gate(i) for _ in
                                                range(parents[smaller_parent_idx].n_gates_per_qubit)]
                                               for i in range(offset + min_qubits, max_qubits)]


def pad_qubits_with_id_gate(children_solutions: list, max_gates: int, c: int):
    for i, qubit in enumerate(children_solutions[c]):
        for _ in range(max_gates - len(qubit)):
            qubit.append(
                get_identity_gate(i))


def check_qubit_and_gate_validity(child: Individual, qubit_id: int, column_id: int, parent: Individual) -> bool:
    if parent.n_qubits <= qubit_id or parent.n_gates_per_qubit <= column_id:
        return False
    if parent.solution[qubit_id][column_id].target_id is not None and parent.solution[qubit_id][
        column_id].target_id >= child.n_qubits:
        return False
    return True


def _split_list(split_list: List, idx: List[int]):
    """
    :param split_list: list to be splitted
    :param idx: list of indeces where to split at
    :return: list of splits

    Splits a given list at points defined in idx
    """

    l = [el for el in split_list]

    result = []
    for i in reversed(idx):
        split = l[i:]
        if len(split) > 0:
            result.append(split)
        l = l[:i]
    if len(l) > 0:
        result.append(l)

    return [el for el in reversed(result)]
