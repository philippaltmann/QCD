import gymnasium as gym
from gymnasium.spaces import Tuple, Box, Discrete

import pennylane as qml
from pennylane import numpy as np

class CircuitDesigner(gym.Env):
    """Quantum Circuit Environment. Description will follow..."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, max_qubits, max_depth):
        super().__init__()

        self.qubits = max_qubits  # the (maximal) number of available qubits
        # define action space
        self.action_space = Tuple((Discrete(7),Discrete(max_qubits), Box(0,2*np.pi)), seed=100)

        # define observation space
        self.observation_space =

    def _action_to_operation(self, action):
        if action[0] == 0:
            return qml.RZ(phi=action[2],wires=action[1])

    def _build_circuit(self, operations):



    def step(self, action):

        # determining what action to take
        if action[0] == 6:
            terminated = True
        else:
            terminated = False
            operation = self._action_to_operation(action)

        # determining reward of action (only if terminated or truncated)
        if not terminated:
            reward = 0
        else:

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        # start with an empty trajectory
        self._operations = []
        return observation, info

    def render(self):
        ...

    def close(self):
        ...

