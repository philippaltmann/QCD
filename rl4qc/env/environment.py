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
        # initialize quantum device to use
        self.device = qml.device('default.qubit', wires=max_qubits)
        # define action space
        self.action_space = Tuple((Discrete(5),Discrete(max_qubits), Box(low=0,high=2*np.pi,shape=(2,))))
        # define observation space
        self.observation_space = Dict(
            {'real': Box(low=np.NINF, high=np.inf, shape=(2**max_qubits,)),
            'imag': Box(low=np.NINF, high=np.inf, shape=(2**max_qubits,))} )

    def _action_to_operation(self, action):
        """ Converter translating actions from action_space into quantum operations """
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

    #@qml.qnode(self.device)
    def _build_circuit(self):
        """ Quantum Circuit Function taking a list of quantum operations and returning state information """
        for op in self._operations:
            qml.apply(op)
        return qml.state()

    def _get_info(self):
        return qml.specs(self._build_circuit)()

    def _draw_circiut(self):
        fig, ax = qml.draw_mpl(self._build_circuit())
        fig.show()

    def reset(self, seed=None, options=None):

        # start with an empty trajectory of operations
        self._operations = []
        # calculate zero-state information
        obs = np.array(self._build_circuit())
        self._observation = Dict({'real': np.real(obs),
                                  'imag': np.imag(obs)})
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
            obs = np.array(self._build_circuit())
            self._observation = Dict({'real': np.real(obs),
                                      'imag': np.imag(obs)})

        observation = self._observation


        # sparse reward of action
        if not terminated:
            reward = 0 # or -1 to punish step count?
        else:
            self._draw_circiut() # render circuit only after each episode
            # TODO: figure out how to include reward function into this class
            reward = None

        # evaluate additional information
        info = self._get_info()

        return observation, reward, terminated, truncated, info


    def render(self):
        return None

    # def close(self):
    # currently there is no need for a close method


