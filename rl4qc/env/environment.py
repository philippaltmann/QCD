import gymnasium as gym
from gymnasium.spaces import Tuple, Box, Discrete, Dict

import pennylane as qml
# from pennylane import numpy as np
import numpy as np


class CircuitDesigner(gym.Env):
    """Quantum Circuit Environment. Description will follow..."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, max_qubits, max_depth):
        super().__init__()

        # define parameters
        self.qubits = max_qubits  # the (maximal) number of available qubits
        self.depth = max_depth
        # initialize quantum device to use for QNode
        self.device = qml.device('default.qubit', wires=max_qubits)
        # define action space
        self.action_space = Tuple((Discrete(5), Discrete(max_qubits), Box(low=0, high=2*np.pi, shape=(2,))))
        # define observation space
        self.observation_space = Dict(
            {'real': Box(low=-1, high=+1, shape=(2**max_qubits,)),
             'imag': Box(low=-1, high=+1, shape=(2**max_qubits,))})

    def _action_to_operation(self, action):
        """ Action Converter translating values from action_space into quantum operations """
        wire = action[1]
        if action[0] == 0:  # Z-Rotation
            return qml.RZ(phi=action[2][0], wires=wire)
        elif action[0] == 1:  # Phased-X
            op_z_p = qml.exp(qml.PauliZ(wire), 1j*action[2][1])
            op_x = qml.exp(qml.PauliX(wire), 1j*action[2][0])
            op_z_m = qml.exp(qml.PauliZ(wire), -1j*action[2][1])
            return qml.prod(op_z_p, op_x, op_z_m)
        elif action[0] == 2:  # CNOT (only neighbouring qubits)
            # decide control qubit based on action parameters
            if action[2][0] <= action[2][1]:
                return qml.CNOT(wires=[(wire-1) % (self.qubits-1)+1, wire])
            else:
                return qml.CNOT(wires=[(wire+1) % (self.qubits-1)-1, wire])
        elif action[0] == 3:  # mid-circuit measurement
            return int(wire)

    def _build_circuit(self):
        """ Quantum Circuit Function taking a list of quantum operations and returning state information """
        for op in self._operations:
            if type(op) == int:
                qml.measure(op)
            else:
                qml.apply(op)
        return qml.state()

    def _get_info(self):
        circuit = qml.QNode(self._build_circuit, self.device)
        return qml.specs(circuit)()

    def _draw_circiut(self):
        circuit = qml.QNode(self._build_circuit(), self.device)
        print(qml.draw(circuit)())
        # TODO: fix qml.draw_mpl(self._build_circuit())

    def reset(self, seed=None, options=None):
        # set seed for random number generator
        super().reset(seed=seed)

        # start with an empty trajectory of operations
        self._operations = []
        # calculate zero-state information
        circuit = qml.QNode(self._build_circuit, self.device)
        self._observation = {'real': np.real(np.array(circuit(), np.float32)),
                             'imag': np.imag(np.array(circuit(), np.float32))}
        observation = self._observation

        # evaluate additional information
        info = self._get_info()

        return observation, info

    def step(self, action):

        # check truncation criterion:
        specs = self._get_info()
        if specs['depth'] >= self.depth:
            truncated = True
        else:
            truncated = False

        # determining what action to take
        if action[0] == 4:
            terminated = True
        else:
            terminated = False
            # conduct action
            operation = self._action_to_operation(action)
            # update action trajectory
            self._operations.append(operation)
            # compute state observation
            circuit = qml.QNode(self._build_circuit, self.device)
            self._observation = {'real': np.real(np.array(circuit(), np.float32)),
                                 'imag': np.imag(np.array(circuit(), np.float32))}

        observation = self._observation

        # sparse reward of action
        if not terminated:
            reward = 0  # or -1 to punish step count?
        else:
            self._draw_circiut()  # render circuit only after each episode
            # TODO: figure out how to include reward function into this class
            reward = None

        # evaluate additional information
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def render(self):
        return None

    # def close(self):
    # currently there is no need for a close method
