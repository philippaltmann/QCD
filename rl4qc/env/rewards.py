import math

import numpy as np
import re
import pennylane as qml
from scipy.stats import unitary_group
from math import erf

class Reward:
    """ Reward class for CircuitDesigner environment:
    computes the reward for all available challenges.

    Attributes
    ----------
    qubits : int
        number of available qubits for quantum circuit
    depth : int
        maximum depth desired for quantum circuit

    """

    # list of available challenges:
    challenges = ['SP', 'UC']
    states = ['random', 'bell', 'ghzN #N:number of qubits']
    unitaries = ['random', 'hadamard', 'toffoli']

    def __init__(self, max_qubit, max_depth):
        self.depth = max_depth
        self.qubits = max_qubit

        # draw random unitary matrix
        self.random_op = unitary_group.rvs(2**self.qubits)
        # draw random Haar state
        self.random_state = np.random.normal(size=(2**self.qubits,)) + 1.j * np.random.normal(size=(2**self.qubits,))
        self.random_state /= np.linalg.norm(self.random_state)

    def compute_reward(self, circuit, challenge, punish):
        """ Wrapper function mapping challenge to corresponding reward function. """
        task, param = re.split("-", challenge)
        if task == 'SP':  # StatePreparation
            reward = self._state_preparation(circuit, param)
        elif task == 'UC':
            reward = self._unitary_composition(circuit, param)
        # and more to come...
        if punish:
            reward -= 0.1 * qml.specs(circuit)()['depth']/self.depth
        return reward

    # REWARD FUNCTIONS:
    def _state_preparation(self, circuit, param):
        """ Compute Reward for State Preparation (SP) task
            = fidelity of the state produced by circuit compared to a given target state defined by param. """

        # compute output state of designed circuit
        state = np.array(circuit())
        # define target state based on param-string
        if param == 'random': # random state
            target = self.random_state
        elif param == 'bell':  # 2-qubit Bell State
            target = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], dtype=np.complex128)
        elif param[:3] == 'ghz':  # n-qubit GHZ State
            n = int(param[3:])
            assert n >= 2, "GHZ entangled state must have at least 2 qubits. " \
                           "\n For N=2: GHZ state is equal to Bell state."
            assert n <= self.qubits, "Target GHZ state cannot consist of more qubits " \
                                     "than are available within the circuit environment."
            target = np.zeros(shape=(2**n,), dtype=np.complex128)
            target[0] = target[-1] = 1/np.sqrt(2)
        else:
            raise ValueError(f'desired target state {param} is not defined in this reward function.'
                             f'See attribute "states" for a list of available states.')

        # make up for possibly unused qubits (transform to basis of output state)
        target = np.array(qml.QNode(self._state_transform, circuit.device)(target))
        # compute fidelity between target and output state within [0,1]
        fidelity = abs(np.vdot(state, target))**2
        return fidelity

    def _unitary_composition(self, circuit, param):
        """ Compute Reward for Unitary Composition (UC) task
            = 1 - erf(norm(U_composed - U_target)) with U_target defined by param. """
        # compute matrix representation of designed circuit
        order = list(range(self.qubits))
        matrix = qml.matrix(circuit(), wire_order=order)
        # compute Frobenius norm of difference between target and output matrix
        if param == 'random':
            norm = np.linalg.norm(self.random_op - matrix)
        elif param == 'hadamard':
            target = qml.matrix(qml.Hadamard(0), wire_order=order)
            norm = np.np.linalg.norm(target - matrix)
        elif param == 'toffoli':
            assert self.qubits >= 3, "to build Toffoli gate you need at least three wires/qubits."
            target = qml.matrix(qml.Toffoli([0, 1, 2]), wire_order=order)
            norm = np.np.linalg.norm(target - matrix)
        else:
            raise ValueError(f'desired target unitary {param} is not defined in this reward function.'
                             f'See attribute "unitaries" for a list of available operations.')

        return 1 - erf(norm)


    # UTILITY FUNCTIONS:
    @staticmethod
    def _state_transform(state):
        n = int(np.log2(state.shape[0]))
        qml.QubitStateVector(state, wires=range(n))
        return qml.state()
