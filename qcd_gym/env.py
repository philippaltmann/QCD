import numpy as np; import gymnasium as gym

from qiskit import QuantumCircuit
from qiskit.circuit.library import PhaseGate, RXGate, CPhaseGate, CXGate, CCXGate, HGate
from qiskit.quantum_info import Statevector, Operator, random_statevector, random_unitary
from qiskit.converters import circuit_to_dag

import warnings; warnings.simplefilter(action='ignore', category=np.ComplexWarning)

flat = lambda s: np.concatenate([s.data.real, s.data.imag]).astype(np.float32).flatten()
GATES = 3

class CircuitDesigner(gym.Env):
  """ Quantum Circuit Environment: build a quantum circuit gate-by-gate for a desired objective.

  Attributes
  qubits [int]: number of available qubits for quantum circuit
  depth [int]: maximum depth desired for quantum circuit
  objective [str]: RL objective for which the circuit is to be built (see Reward class)
  punish [bool]: specifies whether depth of circuit should be punished

  Methods
  reset(): resets the circuit to initial state of |0>^n with empty list of operations
  step(action): updates environment for given action, returning observation and reward after that action
  """

  metadata = {"render_modes": ["image","text"], "render_fps": 30}

  def __init__(self, max_qubits: int, max_depth: int, objective: str, 
               punish=True, sparse=True, seed=None, render_mode=None):
    super().__init__()
    if seed is not None: self._np_random, seed = gym.utils.seeding.np_random(seed)
    self.render_mode = render_mode; self.name = f"{objective}|{max_qubits}-{max_depth}"

    # define parameters, the (maximal) number of available qubits and circuit depth
    self.qubits, self.depth = max_qubits, max_depth
    self.max_steps = max_depth * max_qubits * 2
    self.punish = punish; self.sparse = sparse
    self.objective = objective  # objective for reward computation
    self.target = self._target(*objective.split('-'), seed)
    self._qc = QuantumCircuit(self.qubits)

    # Define observation space
    self.observation_space = gym.spaces.Box(low=-1.0, high=+1.0, shape=self._state[0].shape) 

    # Action space: Gate, Wire, Control, and Theta
    m = 1e-5 # prevent gate overflow at bounds due to floor operator 
    self.action_space = gym.spaces.Box(
      np.array([0,0,0,-np.pi]), np.array([GATES-m,self.qubits-m,self.qubits-m,np.pi]))
    

  def _target(self, task, target, seed):
    if task == 'SP':
      if target == 'random': return random_statevector(2**self.qubits, seed)
      if target == 'bell': return Statevector(np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], dtype=np.complex128))
      if 'ghz' in target: 
        n = int(target[-1])
        assert 2 <= n <= self.qubits, f"GHZ entangled state must have at least 2 and at most {self.qubits} qubits." 
        target = np.zeros(shape=(2**n,), dtype=np.complex128)
        target[0] = target[-1] = 1/np.sqrt(2)
        return Statevector(target) 
    if task == 'UC':
      if target == 'random': return random_unitary(2**self.qubits, seed)
      _t = QuantumCircuit(self.qubits)
      if target == 'hadamard': _t.append(HGate(),[0])
      if target == 'toffoli': 
        assert self.qubits >= 3, "to build Toffoli gate you need at least three wires/qubits."
        _t.append(CCXGate(),[0,1,2])
      return Operator(_t)
    assert False, f'{task}-{target} not defined.'


  @property
  def _operations(self): return sum([v for k,v in self._qc.count_ops().items()])

  @property 
  def _used_wires(self): return self.qubits - len(list(circuit_to_dag(self._qc).idle_wires()))


  @property
  def _state(self):
    """ Calculate zero-state information """
    if 'UC' in self.objective: state = flat(Operator(self._qc))
    if 'SP' in self.objective:  state = flat(Statevector.from_instruction(self._qc))
    observation = np.concatenate([state, flat(self.target)])
    info = {'depth': self._qc.depth(), 'operations': self._operations, 'used_wires': self._used_wires}
    return observation, info


  def _operation(self, action):
    """ Action Converter translating values from action_space into quantum operations """
    gate, wire, cntrl, theta = action
    gate, wire, cntrl = np.floor([gate, wire, cntrl]).astype(int)
    assert wire in range(self.qubits) and cntrl in range(self.qubits), f"{action}"
    if wire == cntrl and gate == 0: return PhaseGate(theta), [wire]         # PhaseShift
    if wire == cntrl and gate == 1: return RXGate(theta), [wire]            # RX
    if gate == 0: return CPhaseGate(theta), [cntrl, wire]                   # ControlledPhaseShift 
    if gate == 1: return CXGate(), [cntrl, wire]                            # CNOT
    if gate == 2: return None                                               # Terminate
    assert False, 'Unhandled Action on gate ' + gate

  def _reward_delta(self, reward, cost):
    reward_delta, cost_delta = reward - self.last_reward, cost - self.last_cost
    self.last_reward = reward; self.last_cost = cost; return reward_delta, cost_delta

  @property
  def _reward(self):
    if 'SP' in self.objective: # compute fidelity between target and output state within [0,1]
      reward = abs(np.vdot(Statevector.from_instruction(self._qc), self.target))**2
    if 'UC' in self.objective: # 1 - 2 * arctan(norm(U_composed - U_target)) / pi with U_target defined by param.
      reward = 1 - 2 * np.arctan(np.linalg.norm(self.target - Operator(self._qc)))/np.pi
    cost = (max(0, self._qc.depth() - self.depth/3)) / (self.depth / 2 * 3)  # 1/3 deph overhead to solution
    if not self.sparse: reward, cost = self._reward_delta(reward, cost)
    return reward, cost
  

  def reset(self, seed=None, options=None):
    super().reset(seed=seed)         # Set seed for random number generator
    if not self.sparse: self.last_reward = 0; self.last_cost = 0;
    self._qc.clear()
    return self._state


  def step(self, action):
    operation = self._operation(action)
    terminated = operation is None
    if not terminated: self._qc.append(*operation) 
    state, info = self._state
    truncated = self._qc.depth() >= self.depth or self._operations >= self.max_steps
    if terminated: info['termination_reason'] = 'DONE'
    if truncated: info['termination_reason'] = 'DEPTH'
    reward, cost = self._reward
    info = {**info, 'metric': reward, 'cost': cost}
    if self.sparse and not (terminated or truncated): reward, cost = 0, 0
    if self.punish: reward -= cost
    return state, reward, terminated, truncated, info


  def render(self):
    if self.render_mode is None: return None
    return self._qc.draw(self.render_mode)
