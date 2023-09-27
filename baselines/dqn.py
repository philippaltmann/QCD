from algorithm import TrainableAlgorithm
from stable_baselines3 import DQN as StableDQN

class DQN(TrainableAlgorithm, StableDQN):
  """A Trainable extension to DQN using HER"""
  n_steps = 1 # For unified logging
  def __init__(self, policy="MultiInputPolicy", **kwargs): #replay_buffer_class = HerReplayBuffer,
    super().__init__(policy=policy, envkwargs={'discrete':True}, **kwargs)


