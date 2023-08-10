from algorithm import TrainableAlgorithm
from stable_baselines3.a2c import A2C as StableA2C

class A2C(TrainableAlgorithm, StableA2C):
  """A Trainable extension to A2C"""
