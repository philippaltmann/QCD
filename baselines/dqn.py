from algorithm import TrainableAlgorithm
from stable_baselines3 import DQN as StableDQN, HerReplayBuffer
from circuit_designer.wrappers import Monitor
import gymnasium as gym

class DQN(TrainableAlgorithm, StableDQN):
  """A Trainable extension to DQN using HER"""
  n_steps = 1 # For unified logging
  def __init__(self, policy="MultiInputPolicy", replay_buffer_class = HerReplayBuffer, **kwargs):
    super().__init__(policy=policy, replay_buffer_class=replay_buffer_class, envkwargs={'discrete':True}, **kwargs)


