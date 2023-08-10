from algorithm import TrainableAlgorithm
from stable_baselines3.ppo import PPO as StablePPO

class PPO(TrainableAlgorithm, StablePPO):
  """A Trainable extension to PPO"""
