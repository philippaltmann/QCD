# RL4QC

**R**einforcement **L**earning for **Q**uantum **C**ircuit Design.

## **Description**

This repository contains a general **`gymnasium`** environment "`CircuitDesigner-v0`" that builds quantum circuits gate-by-gate using **`pennylane`** for specific challenges in quantum computing. The implemented challenges include

+ state preparation
  (find a gate sequence that turns some initial state into the target quantum state)
+ unitary composition
  (find a gate sequence that constructs an arbitrary quantum operator)

for a finite set of quantum gates (see [Actions](#actions)).
A simple routine is set up for training the environment with reinforcement learning algorithms implemented in **`stable_baselines3`**.

### Special features of this package

+ the action space consists of discrete and continuous actions
  (= discrete set of continuously parametrized gates applied to discrete qubits)
  
+ the agent can decide to terminate on its own as an additional action

+ the reward is sparse and will only be given after termination/truncation based on the full circuit output

+ the state observations correspond to the full complex state vector of the quantum system at the current time step

## **Setup**

To install all required packages run:

```sh
pip install -r requirements
```

The environment can be set up as:

```python
import circuit_designer
import gymnasium as gym

env = gym.make("CircuitDesigner-v0", max_qubits=eta, max_depth=delta, challenge='UC-hadamard')
```

### *Parameters*

The relevant parameters for setting up the environment are:

| Parameter  | Type   | Explanation                                                  |
| :--------- | ------ | ------------------------------------------------------------ |
| max_qubits | `int`  | maximal number of qubits available                            |
| max_depth  | `int`  | maximal circuit depth allowed (= truncation criterion)       |
| challenge  | `str`  | RL challenge for which the circuit is to be built (see [Challenges](#challenges)) |
| punish     | `bool` | specifier for turning on multi-objectives (see [Further Objectives](#further-objectives)) |

### *Actions*

The action space of the environment consists of the **universal gate set**[^1]

+ `PhaseShift`: $$P(\Phi) =  \exp\left(i\frac{\Phi}{2}\right) \cdot \exp\left(-i\frac{\Phi}{2} Z\right)$$
+ `ControlledPhaseShift`: $$CP(\Phi) = I \otimes \ket{0} \bra{0} + P(\Phi) \otimes \ket{1} \bra{1}$$
+ `X-Rotation`: $$RX(\Phi) = \exp\left(-i \frac{\Phi}{2} X\right)$$
+ `CNOT`: $$CX_{a,b} = \ket{0}\bra{0}\otimes I + \ket{1}\bra{1}\otimes X$$

and the additional actions `Meassure` and `Terminate` which actively terminates an episode.

Therefore the action space is a `Box` with the following elements:

| Index | Name      | Type   | Range             | Description                                |
| ----- | --------- |------- | ----------------- | :----------------------------------------- |
| 0     | Operation |`int`   | [0, 3]            | specifying operation (see next table)      |
| 1     | Qubit     |`int`   | [0, `max_qubits`) | specifying qubit to apply the operation    |
| 2     | Control   |`int`   | [0, `max_qubits`) | specifying a control qubit                 |
| 3     | Parameter |`float` | [- $\pi$, $\pi$]  | continuous parameter $\phi$                |

Operations
| Index | Qubit / Control  | Type                 | Arguments                 | Comments                      |
| ----- | :--------------: | -------------------- | ------------------------- | :---------------------------- |
| 0     |  -               | Meassurement         | Qubit                     | Control and Parameter omitted |
| 1     | qubit == control | PhaseShift           | Qubit, Parameter          | Control omitted               |
| 1     | qubit != control | ControlledPhaseShift | Qubit, Control, Parameter | -                             |
| 2     | qubit == control | X-Rotation           | Qubit, Parameter          | Control omitted               |
| 2     | qubit != control | CNOT                 | Qubit, Control            | Parameter omitted             |
| 3     |  -               | Terminate            | -                         | All agruments omitted         |

### *Observations*

The state observation returned by the environment is the complex quantum state $\ket{\psi} \in \mathbb{C} ^{n}$ with $n = 2^{\mathrm{qubits}}$ in the computational basis of the quantum system. The initial state of the environment is chosen as $\ket{\psi_{\text{init}}} = \ket{0}^{\otimes n}$.

### *Challenges*

Currently, there are two RL challenges implemented within the environment:

#### State Preparation `'SP'`

The objective of this challenge is to construct a quantum circuit that generates a desired quantum state (e.g. the GHZ state).
For the reward function, the distance metric called ***fidelity*** $$\mathcal{F} = |\braket{\psi_{\text{env}}|\psi_{\text{target}}}|^2 \in [0,1]$$ is used.

##### currently available states for this challenge

+ `'SP-random'` (a random state over *max_qubits* )
+ `'SP-bell'` (the 2-qubit Bell state)
+ `'SP-ghz<N>'` (the `<N>` qubit GHZ state)

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

## **Outlook and ToDos**

-[ ] Implement more challenges for the environment (e.g. ansatz search, optimal control, maximal entanglement, etc.)

-[ ] Implement more states for "SP" and more unitaries for "UC" challenge

-[ ] Investigate reward functions and possibly improve them for more efficient learning

-[ ] Include further objectives (as listed in [Further Objectives](#further-objectives)) and evaluate their effectiveness

-[ ] Implement weighting factors for the rewards and punishments and tune them for optimal trainability

-[ ] Design a sophisticated learning routine and evaluate the different RL algorithms (e.g. PPO, A2C, etc.)

## Tests

To test the current challenges, run

```sh
python -m circuit_designer.test
```

## Train

To train a policy, run

```sh
python -m train PPO -e UC-hadamard-q1-d9
```

## Baselines

To train the provided baseline algorithms run

```sh
./train
```

## Plots

To generate plots from the `results` folder, run

```sh
python -m plot results
```
