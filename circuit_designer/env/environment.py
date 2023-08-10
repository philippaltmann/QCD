import gymnasium as gym
from gymnasium.spaces import Tuple, Box, Dict

from gymnasium.spaces.utils import flatten_space
from gymnasium.spaces.utils import unflatten
from gymnasium.utils import seeding

import pennylane as qml
import numpy as np
import re

# disable warnings
import warnings
warnings.simplefilter(action='ignore', category=np.ComplexWarning)

from .rewards import Reward


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
    punish: bool
        specifies whether depth of circuit should be punished
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

    metadata = {"render_modes": ["image","text"], "render_fps": 30}

    def __init__(self, max_qubits: int, max_depth: int, challenge: str, punish=True, seed=None, render_mode=None):
        super().__init__()
        if seed is not None: self._np_random, seed = seeding.np_random(seed)
        self.render_mode = render_mode; self.name = f"{challenge}|{max_qubits}-{max_depth}"

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
        self.device = qml.device('lightning.qubit', wires=max_qubits) #default.qubit

        # define action space
        self._action_space = Tuple((Box(low=0, high=5),  # operation type (gate, measurement, terminate)
                                    Box(low=0, high=max_qubits+1),  # qubit(s) for operation
                                    Box(low=0, high=2*np.pi, shape=(2,))))  # additional continuous parameters
        self.action_space = flatten_space(self._action_space)  # flatten for training purposes

        # define observation space
        self.observation_space = Dict(
            {'real': Box(low=-1.0, high=+1.0, shape=(2**max_qubits, )),
             'imag': Box(low=-1.0, high=+1.0, shape=(2**max_qubits, ))})

        # initialize reward class
        self.reward = Reward(self.qubits, self.depth)
        self.punish = punish

    def _action_to_operation(self, action):
        """ Action Converter translating values from action_space into quantum operations """
        gate, wire = int(np.floor(action[0][0])), int(np.floor(action[1][0]))
        if wire in self._disabled: return "disabled" # check if wire is already disabled (due to prior measurement)
        elif wire not in range(self.qubits): return "disabled" # check if wire is actually available
        else: # compile action
            if gate == 0:  # Z-Rotation
                return qml.RZ(phi=action[2][0], wires=wire)
            elif gate == 1:  # Phased-X
                return self._PX(action[2][0], action[2][1], wire)
            elif gate == 2:  # CNOT (only neighbouring qubits)
                if action[2][0] <= action[2][1]:  # decide control qubit based on parameters
                    control = (wire-1) % self.qubits
                else:
                    control = (wire+1) % self.qubits
                if control in self._disabled:  # check if control qubit already disabled
                    return "disabled"
                else:
                    return qml.CNOT(wires=[control, wire])
            elif gate == 4:  # mid-circuit measurement
                " this is currently turned off (by definition of the action space) "
                self._disabled.append(wire)
                return int(wire)

    def _build_circuit(self):
        """ Quantum Circuit Function taking a list of quantum operations and returning state information """
        for op in self._operations:
            if type(op) == int: qml.measure(op)
            else: qml.apply(op)
        return qml.state()

    def _get_info(self):
        """ Dictionary of most important circuit properties."""
        circuit = qml.QNode(self._build_circuit, self.device)
        return qml.specs(circuit)()

    def _draw_circuit(self) -> np.ndarray:
        """ Drawing given circuit using matplotlib."""
        circuit = qml.QNode(self._build_circuit, self.device)
        return qml.draw(circuit)()

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

        # initialize dones
        terminated = False
        truncated = False
        info = {}
        # check truncation criterion
        specs = self._get_info()
        if specs["resources"].depth >= self.depth: #specs['depth']
            truncated = True; info['termination_reason'] = 'DEPTH'
        else:
            # determine what action to take
            if action[0] == 3 or len(self._disabled) == self.qubits:
                # skipping termination actions at the beginning of episode
                if len(self._operations) != 0:
                    terminated = True; info['termination_reason'] = 'DONE'
            else:
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
            # self._draw_circuit()  # render circuit after each episode
            reward = self.reward.compute_reward(circuit, self.challenge, self.punish)

        # evaluate additional information
        info = {**self._get_info(), **info}

        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode is None: return None
        if self.render_mode == 'text': return self._draw_circuit()
        assert False, 'Not Implemented'
        return self._draw_circuit()

    # PHASED-X Operator:
    @staticmethod
    def _PX(phi1, phi2, wire):
        """Wrapper function for the Phased-X operator."""
        op_z_p = qml.exp(qml.PauliZ(wire), 1j * phi2)
        op_x = qml.exp(qml.PauliX(wire), 1j * phi1)
        op_z_m = qml.exp(qml.PauliZ(wire), -1j * phi2)
        return qml.prod(op_z_p, op_x, op_z_m)
