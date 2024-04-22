import random
import math
import pandas as pd

from evo import params

g_d = {'cx': [2, 0, 1], 'rx': [1, 1, 0],  'cp': [2, 1, 0], 'p': [1, 1, 0]}
gates_metadata = pd.DataFrame.from_dict(g_d, orient='index', columns=['qubits', 'parameters', 'controls'])
gate_names = g_d.keys()



class Gate:

    def __init__(self, name: str, qubit_id: int, affected_qubits: list, target_qubits: list = [],
                 parameters: list = None,
                 control_qubits: list = [], data_ids=None):
        self._name = name
        self._qubit_id = qubit_id
        self._affected_qubits = affected_qubits
        self._target_qubits = target_qubits
        self._parameters = parameters
        self._control_qubits = control_qubits
        self._data_ids = data_ids

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.__str__()

    @property
    def name(self) -> str:
        """ Gate name. """
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    @property
    def qubit_id(self) -> int:
        """ IDs of the qubit the gate originally acts on. """
        return self._qubit_id

    @qubit_id.setter
    def qubit_id(self, value):
        self._qubit_id = value

    @property
    def is_data_gate(self):
        if self._data_ids is None:
            return False
        else:
            return True

    @property
    def data_ids(self):
        return self._data_ids

    @property
    def affected_qubits(self) -> list:
        """ IDs of the qubits the gate acts on.

        Example: `swap` gate applied on the qubits `0` and `1` -> `affected_qubits = [0, 1]`
        """
        return self._affected_qubits

    @affected_qubits.setter
    def affected_qubits(self, value):
        self._affected_qubits = value

    @property
    def target_qubits(self) -> list:
        """ IDs of target qubits. Specified only by controlled gates otherwise an empty list."""
        return self._target_qubits

    @target_qubits.setter
    def target_qubits(self, value):
        self._target_qubits = value

    @property
    def parameters(self) -> list:
        """ List of parameters for the gate. (e.g. rotation parameters).
        Specified only by parametrized gates otherwise `None`."""
        return self._parameters

    @parameters.setter
    def parameters(self, value):
        self._parameters = value

    @property
    def control_qubits(self) -> list:
        """ IDs of the control qubits for the gate.
        Specified only by controlled gates otherwise an empty list."""
        return self._control_qubits

    @control_qubits.setter
    def control_qubits(self, value):
        self._control_qubits = value



def get_gates_metadata(gateset):
    unavailable_gates = list(set(gate_names) - set(params.gatesets[gateset]))
    return gates_metadata.drop(unavailable_gates)


def create_random_gate(n_qubits, qubit_id, gateset, max_affected_qubits=1, excluded_qubits=None, max_random_iterations=20):
    """
    :qubit_id - index of the qubit in the circuit
    :max_affected_qubits - only single qubit gates if 1, else if n - random choice of the 1...n-qubits gate
    :excluded_qubits - ids (idxes) of the qubits that can not be used additional to the qubit with qubit_id for
    the gate generation, e.g. as controlled qubits
    :max_random_iteration - to prevent infinite or long random loops
    """
    data_id = None
    if excluded_qubits is None:
        excluded_qubits = []
    max_n_qubits = get_max_nr_qubits_in_gate(gateset)
    nr_qubits_for_gate = 1 if max_affected_qubits == 1 else random.randint(1,
                                                                           min(max_affected_qubits,
                                                                               max_n_qubits))
    gate_name, nr_qubits, nr_parameters, nr_controls = choose_random_gate(nr_qubits_for_gate, gateset)
    affected_qubits, target_qubits, control_qubits, parameters = [qubit_id], [], [], [None] * nr_parameters
    if nr_qubits > 1:
        affected_qubits = specify_affected_qubits(n_qubits_circuit=n_qubits, affected_qubits=affected_qubits,
                                                  nr_qubits_gate=nr_qubits, excluded_qubits=excluded_qubits,
                                                  max_random_iterations=max_random_iterations)
    if nr_controls > 0:
        target_qubits = [random.choice(affected_qubits)]  # currently only single target qubit
        for tq in target_qubits:
            excluded_qubits.append(tq)
        control_qubits = specify_control_qubits(affected_qubits=affected_qubits, excluded_qubits=excluded_qubits,
                                                control_qubits=control_qubits, nr_controls=nr_controls,
                                                max_random_iterations=max_random_iterations)
    elif nr_parameters > 0:
        parameters = specify_parameters(parameters=parameters, nr_parameters=nr_parameters)
        #TODO probability for data gates
        if len(params.data_gates) > 0 \
                and (gate_name in params.data_gates) and random.random() > 0.5:
            data_id = [random.randint(0, params.data_dim-1)]
            print(params.data_dim)
            assert False
    return Gate(name=gate_name, qubit_id=qubit_id, affected_qubits=affected_qubits,
                target_qubits=target_qubits, parameters=parameters, control_qubits=control_qubits, data_ids=data_id)


def get_max_nr_qubits_in_gate(gateset):
    gates_metadata = get_gates_metadata(gateset)
    max_q = gates_metadata.iloc[gates_metadata.qubits.argmax(), 0]
    return max_q


def get_gate_metadata_by_name(name, gateset):
    gates_metadata = get_gates_metadata(gateset)
    nr_qubits, nr_parameters, nr_controls = gates_metadata.loc[[name], :].values[0]
    return nr_qubits, nr_parameters, nr_controls


def get_set_of_gates_by_nr_of_qubits(nr_qubits, gateset):
    gates_metadata = get_gates_metadata(gateset)
    gates_by_nr_of_qubits = gates_metadata.index[gates_metadata['qubits'] == nr_qubits].tolist()
    available_gates = list(set(params.gatesets[gateset]).intersection(gates_by_nr_of_qubits))
    return available_gates


def choose_random_gate(nr_qubits, gateset):
    gates_ = get_set_of_gates_by_nr_of_qubits(nr_qubits, gateset)
    gate_name = random.choice(sorted(gates_))
    nr_qubits, nr_parameters, nr_controls = get_gate_metadata_by_name(gate_name, gateset)
    return gate_name, nr_qubits, nr_parameters, nr_controls


def specify_affected_qubits(n_qubits_circuit, affected_qubits, nr_qubits_gate, excluded_qubits, max_random_iterations):
    it = 0
    while len(affected_qubits) != nr_qubits_gate:
        it += 1
        q = random.choice([idx for idx in range(n_qubits_circuit)])
        if q not in affected_qubits and q not in excluded_qubits:
            affected_qubits.append(q)
        # prevent long loop
        if it == max_random_iterations:
            for q in range(0, n_qubits_circuit):
                if q not in affected_qubits and q not in excluded_qubits:
                    affected_qubits.append(q)
                # validate
                if len(affected_qubits) != nr_qubits_gate:
                    pass
                else:
                    break
            if len(affected_qubits) != nr_qubits_gate:
                raise Exception(
                    f'ERROR: The length of the specified affected qubits is unequal to required number of qubits for'
                    f' the gate. Affected qubits: {affected_qubits}, nr of qubits in gate: {nr_qubits_gate}')
            break
    return affected_qubits


def specify_control_qubits(affected_qubits, excluded_qubits, control_qubits, nr_controls, max_random_iterations):
    it = 0
    while len(control_qubits) != nr_controls:  # multi-controlled operations are allowed
        it += 1
        q = random.choice(affected_qubits)
        if q not in control_qubits and q not in excluded_qubits:
            control_qubits.append(q)
        # prevent long loop
        if it == max_random_iterations:
            for q in affected_qubits:
                if q not in control_qubits and q not in excluded_qubits:
                    control_qubits.append(q)
    # validate
    if len(control_qubits) != nr_controls:
        raise Exception(
            f'ERROR: The length of the specified controlled qubits is unequal to the number of the required '
            f'controlled qubits.  Control qubits: {control_qubits}, number of required control qubits: {nr_controls}')
    return control_qubits


def specify_parameters(parameters, nr_parameters, constant=params.parameter_init):
    if len(parameters) == 0:
        parameters = [None] * nr_parameters
    if len(parameters) < nr_parameters:
        for nr_p in range(len(parameters), nr_parameters):
            parameters.append(None)
    for p in range(0, nr_parameters):
        if constant is None:
            parameters[p] = random.uniform(-math.pi, math.pi)
        else:
            parameters[p] = constant
    return parameters


def exclude_already_handled_qubits_in_column(circuit, qubit_id, column_id):
    excluded_qubits = []
    # restriction 1: exclude previous qubits in this column, it means all (qubit_id -n) qubits, where n = 1..qubit_id.
    for q in range(0, qubit_id):
        excluded_qubits.append(q)
    # restriction 2: exclude all following qubits that are already occupied, e.g. used for multiple-qubit gates
    for qb in range(qubit_id, len(circuit)):
        if not circuit[qb][column_id] is None:
            excluded_qubits.append(qb)
    return excluded_qubits


def get_gates_for_qubit(circuit, qubit_id):
    gates = []
    for circuit_data in circuit.data:
        affected_qubits = []
        for q in circuit_data[1]:
            affected_qubits.append(q._index)
        if circuit_data[1][0]._index == qubit_id or qubit_id in affected_qubits:
            target_qubits = [] if circuit_data[0].num_qubits == 1 else [affected_qubits[-1]]
            control_qubits = [] if circuit_data[0].num_qubits == 1 else [affected_qubits[idx] for idx in
                                                                         range(len(circuit_data[1]) - 1)]
            gate_data = {'name': circuit_data[0].name, 'type': circuit_data[0].num_qubits - 1,
                         'qubit_id': qubit_id, 'affected_qubits': affected_qubits, 'target_qubits': target_qubits,
                         'parameters': circuit_data[0].params, 'control_qubits': control_qubits}
            gates.append(gate_data)
    return gates


def validate_solution_metadata(solution):
    for idx_qubit, q in enumerate(solution):
        for idx_gate, g in enumerate(q):
            validate_qubit_id(g, idx_qubit)
            validate_affected_qubits(solution, g, idx_gate)
            validate_control_qubits_vs_name(g)


def validate_control_qubits_vs_name(g):
    if g.name == "id" and len(g.control_qubits) > 0:
        raise AttributeError(
            f'Error in Solutions Setter (Individual): {g.name}, {g.control_qubits}, {g.target_qubits}')


def validate_qubit_id(g, idx_qubit):
    if g.qubit_id != idx_qubit:
        raise AttributeError(
            f"Error in individual solution: gate {g.name} on the qubit {idx_qubit} "
            f"has inconsistent metadata . ")


def validate_affected_qubits(solution, g, idx_gate):
    if len(g.affected_qubits) > 1:
        for affected_qubit in g.affected_qubits:
            validate_id_gate_is_not_in_affected_qubits(solution, affected_qubit, g, idx_gate)
            validate_names_of_affected(solution, affected_qubit, g, idx_gate)
            if len(g.control_qubits) > 0:
                validate_affected_vs_control_and_target_qubits(solution, affected_qubit, len(g.affected_qubits),
                                                               idx_gate)


def validate_id_gate_is_not_in_affected_qubits(solution, affected_qubit, g, idx_gate):
    if solution[affected_qubit][idx_gate].name == "id":
        raise ValueError(f'Error in generating multi-qubit gates! '
                         f'Expected {g.name}, got {solution[affected_qubit][idx_gate].name}')


def validate_affected_vs_control_and_target_qubits(solution, affected_qubit, len_affected, idx_gate):
    len_control_and_target = len(solution[affected_qubit][idx_gate].control_qubits) + len(
        solution[affected_qubit][idx_gate].target_qubits)
    if len_affected != len_control_and_target:
        raise ValueError(f'Error in generating multi-qubit gates! '
                         f'Number of the affected qubits is not equal the sum of the control and target qubits: '
                         f'expected {len_affected} got {len_control_and_target}')


def validate_names_of_affected(solution, affected_qubit, g, idx_gate):
    if g.name != solution[affected_qubit][idx_gate].name:
        raise ValueError(
            f'Error in generating multi-qubit gates! Gate names are inconsistent: expected {g} got '
            f'{solution[affected_qubit][idx_gate].name} ')
