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
        # print(qml.draw(circuit)())
        if task == 'SP': reward = self._state_preparation(circuit, param)     # StatePreparation
        elif task == 'UC': reward = self._unitary_composition(circuit, param) # Unitary Composition
        if punish: reward -= (qml.specs(circuit)()["resources"].depth - self.depth/3) / (self.depth / 2 * 3)  # 1/3 deph overhead to solution

        # and more to come...
        return reward

    # REWARD FUNCTIONS:
    def _state_preparation(self, circuit, param):
        """ Compute Reward for State Preparation (SP) task
            = fidelity of the state produced by circuit compared to a given target state defined by param. """

        # compute output state of designed circuit
        state = np.array(circuit())
        # state = state[:int(state.shape[0]/2)]
        # define target state based on param-string
        if param == 'random': target = self.random_state
        elif param == 'bell': target = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], dtype=np.complex128)
        elif param[:3] == 'ghz':  # n-qubit GHZ State
            n = int(param[-1:])
            target = np.zeros(shape=(2**n,), dtype=np.complex128)
            target[0] = target[-1] = 1/np.sqrt(2)
        else: raise ValueError(f'desired target state {param} is not defined in this reward function.'
                               f'See attribute "states" for a list of available states.')

        # make up for possibly unused qubits (transform to basis of output state)
        # target = np.array(qml.QNode(self._state_transform, circuit.device)(target))
        # compute fidelity between target and output state within [0,1]
        fidelity = abs(np.vdot(state, target))**2
        return fidelity

    def _unitary_composition(self, circuit, param):
        """ Compute Reward for Unitary Composition (UC) task
            = 1 - 2* arctan(norm(U_composed - U_target)) / pi with U_target defined by param. """
        # compute matrix representation of designed circuit
        if qml.specs(circuit)()["resources"].num_gates == 0: return 0
        order = list(range(self.qubits))
        matrix = qml.matrix(circuit, wire_order=order)().astype(np.complex128)
        # compute Frobenius norm of difference between target and output matrix
        if param == 'random': target = self.random_op
        elif param == 'hadamard': target = qml.matrix(qml.Hadamard(0), wire_order=order)
        elif param == 'toffoli': target = qml.matrix(qml.Toffoli([0, 1, 2]), wire_order=order)
        else: raise ValueError(f'desired target unitary {param} is not defined in this reward function.'
                               f'See attribute "unitaries" for a list of available operations.')
        norm = np.linalg.norm(target - matrix)
        return 1 - 2*np.arctan(norm)/np.pi

    # UTILITY FUNCTIONS:
    @staticmethod
    def _state_transform(state):
        n = int(np.log2(state.shape[0]))
        qml.QubitStateVector(state, wires=range(n))
        return qml.state()
