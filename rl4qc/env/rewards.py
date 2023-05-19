import numpy as np
import re
import pennylane as qml


class Reward:
    def __init__(self, circuit, max_depth):
        self.circuit = circuit
        self.depth = max_depth
        # list available challenges:
        self.challenges = ['SP']
        self.states = ['bell', 'ghzN']

    def compute_reward(self, challenge):
        task, param = re.split("-", challenge)
        if task == 'SP':  # StatePreparation
            return self._state_preparation(param)
        else:
            raise ValueError(f'desired challenge {task} is not defined in this class.'
                             f'See attribute "challenges" for a list of available challenges')

    # REWARD FUNCTIONS:
    def _state_preparation(self, param):
        # compute output state of designed circuit
        state = np.array(self.circuit())
        # define target state based on param-string
        if param == 'bell':  # 2-qubit Bell State
            target = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], dtype=np.complex128)
        elif param[:3] == 'ghz':  # n-qubit GHZ State
            n = int(param[3:])
            assert n >= 2, "GHZ entangled state must have at least 2 qubits. " \
                           "\n For N=2: GHZ state is equal to Bell state."
            target = np.zeros(shape=(2**n,), dtype=np.complex128)
            target[0] = target[-1] = 1/np.sqrt(2)
        else:
            raise ValueError(f'desired target state {param} is not defined in this reward function.'
                             f'See attribute "states" for a list of available states')
        # make up for possibly unused qubits (transform to basis of output state)
        target = np.array(qml.QNode(self._state_transform, self.circuit.device)(target))
        # compute fidelity between target and output state within [0,1]
        fidelity = abs(np.vdot(state, target))**2
        # compute punishment due to circuit depth (should be < 0.1)
        punish = qml.specs(self.circuit)()['depth'] * 0.1/self.depth
        return fidelity - punish

    # UTILITY FUNCTIONS:
    @staticmethod
    def _state_transform(state):
        n = int(np.log2(state.shape[0]))
        qml.QubitStateVector(state, wires=range(n))
        return qml.state()
