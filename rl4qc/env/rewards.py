import numpy as np
import re
import pennylane as qml


def compute_reward(circuit, challenge, max_depth):
    task, param = re.split("-", challenge)
    if task == 'SP':  # StatePreparation
        return state_preparation(circuit, param, max_depth)
    else:
        raise ValueError(f'desired challenge {task} is not defined in this class.')
        # TODO: at some point refer to class argument, where to find all possible challenges...


def state_preparation(circuit, param, max_depth):
    # compute output state of designed circuit
    state = np.array(circuit())
    # define target state based on param-string
    if param == 'bell':  # 2-qubit Bell State
        target = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], dtype=np.complex128)
    elif param[:3] == 'ghz':  # n-qubit GHZ State
        n = int(param[3:])
        assert n >= 2, "GHZ entangled state must have at least 2 qubits " \
                       "\n for n=2: GHZ state = Bell state."
        target = np.zeros(shape=(2**n,), dtype=np.complex128)
        target[0] = target[-1] = 1/np.sqrt(2)
    else:
        raise ValueError(f'desired target state {param} is not defined in this reward function.')
        # TODO: at some point refer to class argument, where to find all possible challenges...
    # make up for possibly unused qubits (transform to basis of output state)
    target = qml.QNode(state_transform, circuit.device)(target)
    # compute fidelity between target and output state within [0,1]
    fidelity = abs(np.vdot(state, target))**2
    # compute punishment due to circuit depth (should be < 0.1)
    punish = qml.specs(circuit)()['depth'] * 0.1/max_depth
    return fidelity - punish


# UTILITY FUNCTIONS:
def state_transform(state):
    n = int(np.log2(state.shape[0]))
    qml.QubitStateVector(state, wires=range(n))
    return qml.state()
