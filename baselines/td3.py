from algorithm import TrainableAlgorithm
from stable_baselines3.td3 import TD3 as StableTD3

class TD3(TrainableAlgorithm, StableTD3):
  """A Trainable extension to PPO"""
  def __init__( self, train_freq= (4, "step"), **kwargs):
    self.n_steps = train_freq[0] # For unified logging
    super().__init__(train_freq=train_freq,  **kwargs)

