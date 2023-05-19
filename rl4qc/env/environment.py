import gymnasium as gym
from gymnasium.spaces import Tuple, Box, Discrete

import pennylane as qml
# from pennylane import numpy as np
import numpy as np

from rewards import Reward

# disable warnings
import logging
gym.logger.setLevel(logging.WARNING)


class CircuitDesigner(gym.Env):
    """Quantum Circuit Environment. Description will follow..."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, max_qubits: int, max_depth: int, challenge: str):
        super().__init__()

        # define parameters
        self.qubits = max_qubits  # the (maximal) number of available qubits
        assert max_qubits >= 2, "number of available qubits must be at least 2."
        self.depth = max_depth  # the (maximal) available circuit depth
        self.challenge = challenge  # challenge for reward computation
        # initialize quantum device to use for QNode
        self.device = qml.device('default.qubit', wires=max_qubits)
        # define action space
        self.action_space = Tuple((Discrete(5), Discrete(max_qubits), Box(low=0, high=2*np.pi, shape=(2,))))
        # define observation space
        self.observation_space = Box(low=-1.0, high=+1.0, shape=(2, 2**max_qubits))

    def _action_to_operation(self, action):
        """ Action Converter translating values from action_space into quantum operations """
        wire = action[1]
        # check if wire is already disabled (due to prior measurement)
        if wire in self._disabled:
            return "disabled"
        else:
            if action[0] == 0:  # Z-Rotation
                return qml.RZ(phi=action[2][0], wires=wire)
            elif action[0] == 1:  # Phased-X
                op_z_p = qml.exp(qml.PauliZ(wire), 1j*action[2][1])
                op_x = qml.exp(qml.PauliX(wire), 1j*action[2][0])
                op_z_m = qml.exp(qml.PauliZ(wire), -1j*action[2][1])
                return qml.prod(op_z_p, op_x, op_z_m)
            elif action[0] == 2:  # CNOT (only neighbouring qubits)
                if action[2][0] <= action[2][1]:  # decide control qubit based on parameters
                    if wire == 0:
                        return qml.CNOT(wires=[self.qubits-1, wire])
                    else:
                        return qml.CNOT(wires=[wire-1, wire])
                else:
                    if wire == self.qubits-1:
                        return qml.CNOT(wires=[0, wire])
                    else:
                        return qml.CNOT(wires=[wire+1, wire])
            elif action[0] == 3:  # mid-circuit measurement
                self._disabled.append(wire)
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

    def _draw_circuit(self):
        circuit = qml.QNode(self._build_circuit, self.device)
        print(qml.draw(circuit)())
        # TODO: use instead qml.draw_mpl()...

    def reset(self, seed=None, options=None):
        # set seed for random number generator
        super().reset(seed=seed)

        # start with an empty trajectory of operations
        self._operations = []
        # start with an empty list of disables qubits (due to measurement)
        self._disabled = []

        # calculate zero-state information
        circuit = qml.QNode(self._build_circuit, self.device)
        self._observation = np.vstack((np.real(np.array(circuit(), np.float32)),
                                       np.imag(np.array(circuit(), np.float32))))
        observation = self._observation

        # evaluate additional (circuit) information
        info = self._get_info()

        return observation, info

    def step(self, action):

        # check truncation criterion
        specs = self._get_info()
        if specs['depth'] >= self.depth:
            truncated = True
            terminated = False
        else:
            truncated = False
            # determining what action to take
            if action[0] == 4 or len(self._disabled) == self.qubits:
                terminated = True
            else:
                terminated = False
                # conduct action
                operation = self._action_to_operation(action)
                if operation != "disabled":
                    # update action trajectory
                    self._operations.append(operation)
                # compute state observation
                circuit = qml.QNode(self._build_circuit, self.device)
                self._observation = np.vstack((np.real(np.array(circuit(), np.float32)),
                                               np.imag(np.array(circuit(), np.float32))))

        observation = self._observation

        # sparse reward computation
        if not terminated and not truncated:
            reward = 0
        else:
            self._draw_circuit()  # render circuit only after each episode
            reward = Reward(circuit, self.depth).compute_reward(self.challenge)

        # evaluate additional information
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def render(self):
        return None

    # def close(self):
    # currently there is no need for a close method
