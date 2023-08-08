# RL4QC

**R**einforcement **L**earning for **Q**uantum **C**ircuit Design.

## **Description**

This repository contains a general **`gymnasium`** environment "`CircuitDesigner-v0`" that builds quantum circuits gate-by-gate using **`pennylane`** for specific challenges in quantum computing. The implemented challenges include

+ state preparation
  (find a gate sequence that turns some initial state into the target quantum state)
+ unitary composition
  (find a gate sequence that constructs an arbitrary quantum operator)

for a finite set of quantum gates (see [Actions](#actions)). A simple routine is set up for training the environment with a reinforcement learning algorithm implemented in **`stable_baselines3`**.

### Special features of this package

+ the action space consists of discrete and continuous actions
  (= discrete set of parametrized gates applied to discrete qubits)
  
+ the agent can decide to terminate on its own as an additional action

+ the reward is sparse and will only be given after termination/truncation based on the full circuit output

+ the state observations correspond to the full complex state vector of the quantum system at the current time step

## **Setup**

The environment can be set up as:

```python
import rl4qc
import gymnasium as gym

env = gym.make("CircuitDesigner-v0", max_qubits, max_depth, challenge)
```

### *Parameters*

The relevant parameters for setting up the environment are:

| Parameter  | Type   | Explanation                                                  |
| :--------- | ------ | ------------------------------------------------------------ |
| max_qubits | `int` | maximal number of qubits available                           |
| max_depth  | `int`  | maximal circuit depth allowed (= truncation criterion)       |
| challenge  | `str`  | RL challenge for which the circuit is to be built (see [Challenges](#challenges)) |
| punish     | `bool` | specifier for turning on multi-objectives (see [Further Objectives](#further-objectives)) |

### *Actions*

The action space of the environment consists of the **universal gate set**[^1]

+ `Z-Rotation`: $$R_Z(\theta) = \exp(-\text{i} \frac{\theta}{2} Z)$$
+ `Phased X`: $$P_X(\theta, \phi) = \exp(\text{i}\theta Z) \cdot \exp(\text{i}\phi X) \cdot \exp(-\text{i}\theta Z)$$
+ `CNOT`: $$\text{CNOT}\ket{a} \ket{b} = \ket{a} \ket{a \oplus b}$$

and the additional action `Terminate` which actively terminates an episode.

Therefore the action space is a flattened Tuple (instance of `gym.spaces`) with `Box` elements:

| Index | Type    | Range             | Description                                |
| ----- | ------- | ----------------- | :----------------------------------------- |
| 0     | `int`   | [0, 3]            | specifying action (gate or terminate)      |
| 1     | `int`   | [0, `max_qubits`) | specifying qubit/wire to apply the gate to |
| 2     | `float` | [0, 2 $\pi$]       | continuous parameter $\phi$                |
| 3     | `float` | [0, 2 $\pi$]       | continuous parameter $\theta$              |

[^1]: ["Quantum circuit optimization with deep reinforcement learning"](http://arxiv.org/pdf/2103.07585v1)

### *Observations*

The state observation returned by the environment is the complex quantum state $\ket{\psi} \in \mathbb{C} ^{n}$ with $n = 2^{\mathrm{qubits}}$ in the computational basis of the quantum system. The initial state of the environment is chosen as $\ket{\psi_{\text{init}}} = \ket{0}^{\otimes n}$.

### *Challenges*

Currently, there are two RL challenges implemented within the environment:

#### State Preparation `'SP'`

The objective of this challenge is to construct a quantum circuit that generates a desired quantum state (e.g. the GHZ state).
For the reward function, the distance metric called ***fidelity*** $$\mathcal{F} = |\ket{\psi_{\text{env}}}\bra{\psi_{\text{target}}}|^2 \in [0,1]$$ is used.

##### currently available states for this challenge

+ `'SP-random'` (a random state over *max_qubits* )
+ `'SP-bell'` (the 2-qubit Bell state)
+ `'SP-ghz**N**'` (the ***N*** qubit GHZ state)

##### Unitary Composition `'UC'`

The objective of this challenge is to construct a quantum circuit with a given finite set of gates that implements a desired unitary transformation/operation. For the reward function, an 1-arctan mapping of the ***Frobenius norm*** $$|U_{\text{env}} - U_{\text{target}}|_2$$ to the interval $[0,1]$ is chosen.

##### currently available unitaries for this challenge

+ `'UC-random'` (a random unitary operation on *max_qubits* )
+ `'UC-hadamard'` (the single qubit Hadamard gate)
+ `'UC-toffoli'` (the 3-qubit Toffoli gate)

See [Outlook](#outlook-and-todos) for more challenges to come...

### *Further Objectives*

The goal of this implementation is to not only construct any circuit that fulfills a specific challenge but to also make this circuit optimal, that is to give the environment further objectives, such as minimizing:

+ Circuit Depth
+ Qubit Count
+ Gate Count (or: 2-qubit Gate Count)
+ Parameter Count
+ Qubit-Connectivity

These circuit optimization objectives can be switched on by the parameter `punish` when initializing a new environment.

Currently, the only further objective implemented in this environment is the **circuit depth**, as this is one of the most important features to restrict for NISQ (noisy, intermediate-scale, quantum) devices. This metric already includes gate count and parameter count to some extent. However, further objectives can easily be added within the `Reward` class of this environment (see [Outlook](#outlook-and-todos)).

## **Example**

```python
import rl4qc
import gymnasium as gym

# specify environmental parameters
max_qubits = 2
max_depth = 10
challenge = 'SP-bell'

# initalize environment
env = gym.make("CircuitDesigner-v0", max_qubits, max_depth, challenge)

# run basic example with random actions
observation, info = env.reset()

for _ in range(100):
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()
        
```

This shows a plot of the built circuit after each episode:

![example_circuit](/models/example_circuit.png)

## **Outlook and ToDos**

-[ ] Implement more challenges for the environment (e.g. ansatz search, optimal control, maximal entanglement, etc.)

-[ ] Implement more states for "SP" and more unitaries for "UC" challenge

-[ ] Investigate reward functions and possibly improve them for more efficient learning

-[ ] Include further objectives (as listed in [Further Objectives](#further-objectives)) and evaluate their effectiveness

-[ ] Implement weighting factors for the rewards and punishments and tune them for optimal trainability

-[ ] Design a sophisticated learning routine and evaluate the different RL algorithms (e.g. PPO, A2C, etc.)
