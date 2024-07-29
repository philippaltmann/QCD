import random
import uuid
from copy import deepcopy
from typing import Callable
from qiskit import QuantumCircuit

from evo import params as evo_params, custom_fitness
from evo.gate import *


class Individual:

    def __init__(self, max_qubits: int = 5, params: dict = None,
                 generated_in_generation: int = 0, is_empty=False, initial_solution: list = None, gateset: int = None):
        self._id = uuid.uuid4()
        params = params or evo_params
        if gateset is None:
            self._gateset = random.randrange(len(params.gatesets))
        else: self._gateset = gateset
        self.fitness = None
        assert custom_fitness.custom_fitness
        self.fitness_function = custom_fitness.custom_fitness
        self.generated_in_generation = generated_in_generation
        if is_empty: pass
        else:
            self.n_qubits = params.constant_n_qubits if params.constant_n_qubits is not None else \
                random.randint(1, max_qubits)
            self.n_gates_per_qubit = random.randint(params.init_min_gates, params.init_max_gates)
            if initial_solution is not None:
                self.n_qubits = len(initial_solution)
                self.n_gates_per_qubit = len(initial_solution[0])
                self.solution = initial_solution
            else:
                self.solution = self.create_solution()
            self._qc = self.transform_to_executable_circuit(self.solution)

    def set_fitness_function(self, fitness_function: Callable):
        self.fitness_function = fitness_function

    def get_fitness_function(self):
        return self.fitness_function

    def get_data_gates(self):
        data_gates = []
        for qubit_id, gates in enumerate(self.solution):
            for column_id, gate in enumerate(gates):
                if gate.is_data_gate:
                    data_gates.append({'qubit_id': qubit_id, 'column_id': column_id, 'data_ids': gate.data_ids})
        return data_gates

    def calculate_fitness(self, **kw_args):
        if self.fitness is None:
            self.fitness = self.fitness_function(self, **kw_args)
        return self.fitness

    def generated_in_generation(self) -> int:
        return self.generated_in_generation

    def set_generated_in_generation(self, gen):
        self.generated_in_generation = gen

    @property
    def id(self):
        """Individual id.
        """
        return self._id

    @id.setter
    def id(self, value):
        self._id = value

    @property
    def solution(self) -> list:
        """Individual (quantum circuit) as nested `n x m` list  (`n` qubits, `m` gates).
        """
        return self._solution

    @solution.setter
    def solution(self, value: list):
        validate_solution_metadata(value)
        self._solution = value
        self.qc = self.transform_to_executable_circuit(self._solution)

    @property
    def qc(self) -> QuantumCircuit:
        """An executable individual (qiskit circuit)."""
        return self._qc

    @qc.setter
    def qc(self, value: QuantumCircuit):
        self._qc = value

    @property
    def depth(self) -> int:
        """Depth of the executable individual.
        """
        return self._qc.depth()

    @property
    def size(self) -> int:
        """Total number of instructions in the executable individual.
        In OpenQASM: +1 for each instruction from registry definition till measurements.
        """
        return self._qc.size()

    @property
    def gateset(self) -> int:
        return self._gateset

    @gateset.setter
    def gateset(self, gateset: int):
        self._gateset = gateset

    def create_solution(self) -> list:
        """
        Function for the individual generation (represented as list). The gates are generated column-wise. For the each
        column:

        1. Identify the number of qubits that in can be used for a gate in this column (max_affected_qubits,
        min: 1 qubit, max: 3 qubits). The qubits that are already occupied (=> can not be used) are stored in the
        excluded_qubits array.

         2. Using this info, create a gate at random and assign it to the corresponding qubit
        and column of the circuit. If the generated gate uses more than one qubit -> assign also the same gate with
        its metadata to the other affected qubits in this column.

        Returns:
              A quantum circuit as nested `n x m` list  (`n` qubits, `m` gates).

        Raises:
              ValueError:  If specification of the qubits a gate acts on (affected qubits) is not valid.
        """
        circuit = [[None for _ in range(self.n_gates_per_qubit)] for _ in range(self.n_qubits)]
        for column_id in range(self.n_gates_per_qubit):
            for qubit_id in range(self.n_qubits):
                if circuit[qubit_id][column_id] is None:
                    excluded_qubits = exclude_already_handled_qubits_in_column(circuit=circuit, qubit_id=qubit_id,
                                                                               column_id=column_id)
                    max_affected_qubits = min(3, self.n_qubits - len(excluded_qubits))
                    if max_affected_qubits < 1:
                        max_affected_qubits = 1
                    gate = create_random_gate(n_qubits=self.n_qubits, qubit_id=qubit_id,
                                              gateset=self._gateset,
                                              max_affected_qubits=max_affected_qubits,
                                              excluded_qubits=excluded_qubits)
                    circuit[qubit_id][column_id] = gate
                    if len(gate.affected_qubits) > 1:
                        for q in gate.affected_qubits[1:]:
                            if q > qubit_id:
                                g_copy = deepcopy(gate)
                                g_copy.qubit_id = q
                                circuit[q][column_id] = g_copy
                            else:
                                raise ValueError(
                                    "Id of the helper qubit for multiple qubits gate should greater than qubit_id.")
        return circuit

    def transform_to_executable_circuit(self, circuit_solution: list):
        """Transforms circuit_solution into executable instance of qiskit.QuantumCircuit.

          Args:
              circuit_solution (list): Generated individual as list of `n x m` elements, whereby `n` is the number of
               qubits, `m` is the number of gates acting on each qubit.

          Returns:
              QuantumCircuit: Instance of QuantumCircuit, is executable in Qiskit.

          Raises:
              CircuitError:  if the circuit is not valid.
              TypeError:  If circuit parameters are not valid.
              IndexError: If the number of qubits and gates in the circuit are not valid.
          """
        qc = QuantumCircuit(self.n_qubits)
        running_solution = deepcopy(circuit_solution)  # need for saving info about already handled gates
        for gate_idx in range(self.n_gates_per_qubit):
            for qb_idx in range(self.n_qubits):
                gate = circuit_solution[qb_idx][gate_idx]
                if running_solution[qb_idx][gate_idx] is not None:
                    if gate.control_qubits and gate.control_qubits.__len__() > 1:
                        p = []
                        for cq in gate.control_qubits:
                            p.append(cq)
                        for tq in gate.target_qubits:
                            p.append(tq)
                        getattr(qc, gate.name)(*p)
                    else:
                        if not gate.parameters or gate.parameters.__len__() == 0:
                            getattr(qc, gate.name)(*gate.affected_qubits)
                        elif gate.parameters.__len__() >= 1:
                            p = []
                            for gp in gate.parameters:
                                p.append(gp)
                            for q in gate.affected_qubits:
                                p.append(q)
                            getattr(qc, gate.name)(*p)
                    if gate.affected_qubits and len(gate.affected_qubits) > 1:
                        for q in gate.affected_qubits:
                            if q != qb_idx:
                                running_solution[q][gate_idx] = None
            # qc.measure_all()
        return qc

    def insert_datapoint_to_solution(self, data_point: list):
        """ Inserting data point into circuit via the data gates.
            Args:
                  data_point (list):
        """
        data_gates = self.get_data_gates()
        for data_gate in data_gates:
            qubit_id = data_gate['qubit_id']
            column_id = data_gate['column_id']
            parameters = data_gate['data_ids']
            for idx in range(len(self.solution[qubit_id][column_id].parameters)):
                self.solution[qubit_id][column_id].parameters[idx] = data_point[parameters[idx]]
                self.qc, _ = self.transform_to_executable_circuit(self._solution)

    def __repr__(self):
        # Draw individual in the console and print inforation about the number of qubits, number of gates per qubit and
        # the value of the fitness function.
        return "Individual: Qubits={}, Gates={}, Fitness={}, Circuit: \n{}".format(self.n_qubits,
                                                                                   self.n_gates_per_qubit,
                                                                                   self.fitness, self.qc.draw())

    def __str__(self):
        return self.__repr__()
