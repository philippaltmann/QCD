import gymnasium as gym; import pennylane as qml
import numpy as np; import re

# disable warnings
import warnings
warnings.simplefilter(action='ignore', category=np.ComplexWarning)

from .rewards import Reward

# Resolution of the parameter disrectization
GATES = 3
RESOLUTION = 4#32

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

    def __init__(self, max_qubits: int, max_depth: int, challenge: str, punish=True, seed=None, render_mode=None, discrete=True):
        super().__init__()
        if seed is not None: self._np_random, seed = gym.utils.seeding.np_random(seed)
        self.render_mode = render_mode; self.name = f"{challenge}|{max_qubits}-{max_depth}"

        # define parameters
        self.qubits = max_qubits  # the (maximal) number of available qubits
        self.depth = max_depth    # the (maximal) available circuit depth
        self.max_steps = max_depth * max_qubits * 2
        self.challenge = challenge  # challenge for reward computation
        task = re.split("-", self.challenge)[0]
        if task not in Reward.challenges:
            raise ValueError(f'desired challenge {task} is not defined in this class.'
                             f'See attribute "challenges" for a list of available challenges')
        
        if 'toffoli' in challenge: assert self.qubits >= 3, "to build Toffoli gate you need at least three wires/qubits."
        if 'ghz' in challenge:
            n = int(challenge[-1:])
            assert n >= 2, "GHZ entangled state must have at least 2 qubits. " \
                           "\n For N=2: GHZ state is equal to Bell state."
            assert n <= self.qubits, "Target GHZ state cannot consist of more qubits " \
                                     "than are available within the circuit environment."

        # initialize quantum device to use for QNode (add one ancilla)
        self.device = qml.device('lightning.qubit', wires=self.qubits)  # default.qubit
        
        # Action space: Gate, Wire, Control, and Theta
        self.discrete = discrete; m = 1e-5 # prevent gate overflow at bounds due to floor operator 
        if self.discrete: self.action_space = gym.spaces.MultiDiscrete([GATES, self.qubits, self.qubits, RESOLUTION+1])
        else: self.action_space = gym.spaces.Box(np.array([0,0,0,-np.pi]), np.array([GATES-m,self.qubits-m,self.qubits-m,np.pi]))

        # define observation space
        self.observation_space = gym.spaces.Box(low=-1.0, high=+1.0, shape=(2*2**max_qubits,)) #, type=np.float64

        # initialize reward class
        self.reward = Reward(self.qubits, self.depth)
        self.punish = punish

    def _action_to_operation(self, action):
        """ Action Converter translating values from action_space into quantum operations """
        gate, wire, cntrl, theta = action
        if self.discrete: theta = (2 * (theta / RESOLUTION) - 1) * np.pi
        else: gate, wire, cntrl = np.floor([gate, wire, cntrl]).astype(int)
        # print(f"Applying {['NOOP', 'CRZ', 'CRX'][gate]}({theta/np.pi}π) to {gate}•{cntrl}")
        assert wire in range(self.qubits) and cntrl in range(self.qubits), f"{action}"

        if wire in self._disabled:  return None                                 # check if wire is already disabled 
        if gate == 0: self._disabled.append(wire); return int(wire)             # Meassurement 
        if wire == cntrl and gate == 1: return qml.PhaseShift(theta,wire)       # PhaseShift
        if wire == cntrl and gate == 2: return qml.RX(theta,wire)               # RX
        if cntrl in self._disabled: return None                                 # check if control qubit already disabled
        if gate == 1: return qml.ControlledPhaseShift(theta, [cntrl, wire])     # ControlledPhaseShift 
        if gate == 2: return qml.CNOT([cntrl, wire])                            # CNOT
        assert False, 'Unhandled Action on gate ' + gate

    def _build_circuit(self):
        """ Quantum Circuit Function taking a list of quantum operations and returning state information """
        for op in self._operations:
            if op is None: pass
            elif type(op) == int: qml.measure(op); 
            else: qml.apply(op);
        return qml.state()

    def _draw_circuit(self) -> np.ndarray:
        """ Drawing given circuit using matplotlib."""
        circuit = qml.QNode(self._build_circuit, self.device)
        return qml.draw(circuit)()
    
    def _get_state(self) -> tuple[np.ndarray,qml.QNode]:
        # calculate zero-state information
        node = qml.QNode(self._build_circuit, self.device)
        state = node()#[int(2**(self.qubits+1)/2):]
        observation = np.concatenate([state.real, state.imag]).astype(np.float32)
        return observation, node #information

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)         # Set seed for random number generator
        self._operations = []            # Reset trajectory of operations
        self._disabled = []              # Rest list of measured qubits 
        state, node = self._get_state()  # Calculate state get node
        return state, qml.specs(node)()

    def step(self, action):
        terminated = action[0] == 3 or len(self._disabled) >= self.qubits
        if not terminated: # conduct action & update action trajectory
            operation = self._action_to_operation(action)
            self._operations.append(operation)

        state, node = self._get_state(); info = qml.specs(node)()
        terminated = action[0] == 3 or len(self._disabled) >= self.qubits
        truncated = info["resources"].depth >= self.depth or len(self._operations) >= self.max_steps

        if terminated: info['termination_reason'] = 'DONE'
        if truncated: info['termination_reason'] = 'DEPTH'
   
        # sparse reward computation
        if not terminated and not truncated: reward = 0
        else: reward = self.reward.compute_reward(node, self.challenge, self.punish)

        return state, reward, terminated, truncated, info

    def render(self):
        if self.render_mode is None: return None
        if self.render_mode == 'text': return self._draw_circuit()
