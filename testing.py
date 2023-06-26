import numpy as np

import rl4qc
import gymnasium as gym

#%% DEFINE CRUCIAL PARAMETERS
max_qubits = 2
max_depth = 8

challenge = 'SP-bell'
# challenge = 'UC-toffoli'

#%% BUILD ENVIRONMENT
from gymnasium.utils.env_checker import check_env

env = gym.make("CircuitDesigner-v0", max_qubits=max_qubits, max_depth=max_depth, challenge=challenge)
check_env(env, skip_render_check=True)

#%% CHECK ACTION SPACE - calculations
actions = []
qubits = []
param1 = []
param2 = []

from gymnasium.spaces import Tuple, Box, Dict
from gymnasium.spaces.utils import flatten_space
from gymnasium.spaces.utils import unflatten

space = Tuple((Box(low=0, high=4, dtype=np.int_),  # operation type (gate, measurement, terminate)
               Box(low=0, high=max_qubits-1, dtype=np.int_),  # qubit(s) for operation
               Box(low=0, high=2*np.pi, shape=(2,))))

for i in range(100000):
    action = unflatten(space, env.action_space.sample())
    actions.append(action[0][0])
    qubits.append(action[1][0])
    param1.append(action[2][0])
    param2.append(action[2][1])

#%% CHECK ACTION SPACE - plot results
import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots(2, 2)

sns.countplot(x=actions, ax=ax[0, 0])
ax[0, 0].set_title('operation')
sns.countplot(x=qubits, ax=ax[0, 1])
ax[0, 1].set_title('qubits')

ax[1, 0].hist(param1)
ax[1, 0].set_title('param1')
ax[1, 0].set_ylabel('count')
ax[1, 1].hist(param2)
ax[1, 1].set_title('param2')
ax[1, 1].set_ylabel('count')

plt.tight_layout()
plt.show()

#%% RUN BASIC EXAMPLE with random actions
observation, info = env.reset()

for _ in range(100):
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)
    # print(reward)

    if terminated or truncated:
        observation, info = env.reset()

#%% INITIALIZE STABLE BASELINES
from stable_baselines3.common.env_checker import check_env
check_env(env)

#%% TRAIN ENVIRONMENT ON BASIC EXAMPLE
from stable_baselines3 import PPO

model = PPO("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=25000, progress_bar=True)
model.save("models/SP-bell-25kfull")

#%% or simply upload trained version
from stable_baselines3 import PPO
model = PPO.load("models/SP-bell-25k")
# model = PPO.load("models/UC-toffoli-25k")

#%% EVALUATE AGENT
from stable_baselines3.common.evaluation import evaluate_policy

mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
print(f'mean reward: {mean_reward}')
print(f'std reward: {std_reward}')

#%% USE TRAINED AGENT
vec_env = model.get_env()
obs = vec_env.reset()

dones = False
while dones == False:
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = vec_env.step(action)


