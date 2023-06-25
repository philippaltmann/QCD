import gymnasium as gym
from gymnasium.spaces import Tuple, Box, Dict

from gymnasium.spaces.utils import flatten_space
from gymnasium.spaces.utils import unflatten

import pennylane as qml
import numpy as np
import re

from .rewards import Reward

# disable warnings
import logging
gym.logger.setLevel(logging.ERROR)
logging.getLogger().setLevel(logging.ERROR)
import warnings
warnings.simplefilter(action='ignore', category=np.ComplexWarning)


class CircuitDesigner(gym.Env):
    """ Quantum Circuit Environment:
    build a quantum circuit gate-by-gate for a desired challenge.
    ...

    Attributes
    ----------
    qubits : int
        number of available qubits for quantum circuit
    depth : int
        maximum depth desired for quantum circuit
    challenge : str
        RL challenge for which the circuit is to be built (see Reward class)
    device : qml.device
        quantum device to use (see PennyLane)

    action_space : gymnasium.spaces
        action space consisting of (gate: Box (int), target qubit: Box (int), params: Box (float))
    observation_space : gymnasium.spaces
        complex observation of state in the computational basis as a Box with [real, imag]

    Methods
    -------
    reset():
        resets the circuit to initial state of |0>^n with empty list of operations
    step(action):
        updates environment for given action, returning observation and reward after that action

    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, max_qubits: int, max_depth: int, challenge: str):
        super().__init__()

        # define parameters
        self.qubits = max_qubits  # the (maximal) number of available qubits
        if max_qubits < 2:
            raise ValueError('number of available qubits must be at least 2.')
        self.depth = max_depth  # the (maximal) available circuit depth
        self.challenge = challenge  # challenge for reward computation
        task = re.split("-", self.challenge)[0]
        if task not in Reward.challenges:
            raise ValueError(f'desired challenge {task} is not defined in this class.'
                             f'See attribute "challenges" for a list of available challenges')

        # initialize quantum device to use for QNode
        self.device = qml.device('default.qubit', wires=max_qubits)

        # define action space
        self._action_space = Tuple((Box(low=0, high=5, dtype=np.int_),  # operation type (gate, measurement, terminate)
                                    Box(low=0, high=max_qubits, dtype=np.int_),  # qubit(s) for operation
                                    Box(low=0, high=2*np.pi, shape=(2,))))  # additional continuous parameters
        self.action_space = flatten_space(self._action_space)  # flatten for training purposes

        # define observation space
        self.observation_space = Dict(
            {'real': Box(low=-1.0, high=+1.0, shape=(2**max_qubits, )),
             'imag': Box(low=-1.0, high=+1.0, shape=(2**max_qubits, ))})

    def _action_to_operation(self, action):
        """ Action Converter translating values from action_space into quantum operations """
        wire = action[1][0]
        # check if wire is already disabled (due to prior measurement)
        if wire in self._disabled:
            return "disabled"
        # check if wire is actually available
        elif wire not in range(self.qubits):
            return "disabled"
        # compile action
        else:
            if action[0] == 0:  # Z-Rotation
                return qml.RZ(phi=action[2][0], wires=wire)
            elif action[0] == 1:  # Phased-X
                return self._PX(action[2][0], action[2][1], wire)
            elif action[0] == 2:  # CNOT (only neighbouring qubits)
                if action[2][0] <= action[2][1]:  # decide control qubit based on parameters
                    control = (wire-1) % self.qubits
                else:
                    control = (wire+1) % self.qubits
                if control in self._disabled:  # check if control qubit already disabled
                    return "disabled"
                else:
                    return qml.CNOT(wires=[control, wire])
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
        """ Dictionary of most important circuit properties."""
        circuit = qml.QNode(self._build_circuit, self.device)
        return qml.specs(circuit)()

    def _draw_circuit(self):
        """ Drawing given circuit using matplotlib."""
        circuit = qml.QNode(self._build_circuit, self.device)
        fig, ax = qml.draw_mpl(circuit)()
        fig.show()

    def reset(self, seed=None, options=None):
        # set seed for random number generator
        super().reset(seed=seed)

        # start with an empty trajectory of operations
        self._operations = []
        # start with an empty list of disables qubits (due to measurement)
        self._disabled = []

        # calculate zero-state information
        circuit = qml.QNode(self._build_circuit, self.device)
        observation = {'real': np.real(np.array(circuit(), np.float32)),
                       'imag': np.imag(np.array(circuit(), np.float32))}

        # evaluate additional (circuit) information
        info = self._get_info()

        return observation, info

    def step(self, action):

        # unflatten action for interpretation
        action = unflatten(self._action_space, action)

        # check truncation criterion
        specs = self._get_info()
        if specs['depth'] >= self.depth:
            truncated = True
            terminated = False
        else:
            truncated = False
            # determine what action to take
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
        observation = {'real': np.real(np.array(circuit(), np.float32)),
                       'imag': np.imag(np.array(circuit(), np.float32))}

        # sparse reward computation
        if not terminated and not truncated:
            reward = 0
        else:
            self._draw_circuit()  # render circuit only after each episode
            reward = Reward(circuit, self.qubits, self.depth).compute_reward(self.challenge, punish=True)

        # evaluate additional information
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def render(self):
        return None

    # PHASED-X Operator:
    @staticmethod
    def _PX(phi1, phi2, wire):
        """Wrapper function for the Phased-X operator."""
        op_z_p = qml.exp(qml.PauliZ(wire), 1j * phi2)
        op_x = qml.exp(qml.PauliX(wire), 1j * phi1)
        op_z_m = qml.exp(qml.PauliZ(wire), -1j * phi2)
        return qml.prod(op_z_p, op_x, op_z_m)
