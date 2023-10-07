from algorithm import TrainableAlgorithm
from stable_baselines3.sac import SAC as StableSAC

class SAC(TrainableAlgorithm, StableSAC):
  """A Trainable extension to SAC"""
  n_steps = 1 # For unified logging
  def __init__(self, **kwargs):
    super().__init__(envkwargs={'discrete': False}, **kwargs)
