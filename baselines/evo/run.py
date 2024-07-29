import numpy as np
from evo.experiment import Experiment
from evo import params

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, Operator, random_statevector, random_unitary
from qiskit.circuit.library import CCXGate, HGate
from qiskit.converters import circuit_to_dag

import random

def ghz(eta):
  target = np.zeros(shape=(2**eta,), dtype=np.complex128)
  target[0] = target[-1] = 1/np.sqrt(2)
  return target

def SP(individual, target):
  individual.qc.remove_final_measurements()
  state = Statevector.from_instruction(individual.qc)
  return abs(np.vdot(state, target))**2 # Fidelity 

def UC(individual, target):
  matrix = Operator(individual.qc)
  norm = np.linalg.norm(target - matrix)
  return 1 - 2 * np.arctan(norm)/np.pi

def fitness(mode, delta, penalize=True, operation=None):
  def F(individual, target):
    R = eval(mode)(individual, target)
    d = individual.qc.depth()
    C = (max(0, d - delta/3)) / (delta / 2 * 3)
    return R-penalize*C
  return F

def run_evo(config, seed):
  random.seed(seed); np.random.seed(seed)
  mode, goal, eta, delta = config.split('-')
  eta, delta = int(eta[1:]), int(delta[1:])
  params.constant_n_qubits = eta
  params.init_max_gates = delta
  # params.init_max_gates = eta * delta * 2
  target = {
    'hadamard': Operator(HGate()),
    'toffoli': Operator(CCXGate()),
    'bell': Statevector(ghz(2)),
    'ghz3': Statevector(ghz(3)),
    'random': {
      'SP': random_statevector((2**eta,), seed),
      'UC': random_unitary((2**eta,), seed),
    }[mode],
  }[goal]

  experiment = Experiment(target=target, fitness=fitness(mode,delta), metrics={
    'Return': lambda i: fitness(mode, delta, False)(i, target),
    'Metric': lambda i: fitness(mode, delta, True)(i, target),
    'Cost': lambda i: (max(0, i.qc.depth() - delta/3)) / (delta / 2 * 3),
    'Depth': lambda i: i.qc.depth(),
    'Qubits': lambda i: eta - len(list(circuit_to_dag(i.qc).idle_wires()))
  })

  stats, population = experiment.run_experiments()
  mean = {k: np.mean(v) for k,v in stats.items()}
  return mean 
