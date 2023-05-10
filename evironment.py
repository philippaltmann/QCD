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
        self.action_space = Tuple((Discrete(5),Discrete(max_qubits), Box(low=0,high=2*np.pi,shape=(2,))), seed=100)

        # define observation space
        self.observation_space =

    def _action_to_operation(self, action):
        wire = action[1]
        if action[0] == 0: # Z-Rotation
            return qml.RZ(phi=action[2][0],wires=wire)
        elif action[0] == 1: # Phased-X
            op_z_p = qml.exp(qml.PauliZ(wire), 1j*action[2][1])
            op_x = qml.exp(qml.PauliX(wire), 1j*action[2][0])
            op_z_m = qml.exp(qml.PauliZ(wire), -1j*action[2][1])
            return qml.prod(op_z_p,op_x,op_z_m)
        elif action[0] == 2: # CNOT (only neighbouring qubits)
            # decide control qubit based on action parameters
            if action[2][0] <= action[2][1]:
                return qml.CNOT((wire-1)%(self.qubits-1)+1, wire)
            else:
                return qml.CNOT((wire+1)%(self.qubits-1)-1, wire)
        elif action[0] == 3: # mid-circuit measurement
            return qml.measure(wire)


    def _build_circuit(self, operations):



    def step(self, action):

        # determining what action to take
        if action[0] == 4:
            terminated = True
        else:
            terminated = False
            operation = self._action_to_operation(action)

        # determining reward of action (only if terminated or truncated)
        if not terminated:
            reward = 0 # or -1 to punish step count?
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

