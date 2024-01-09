# from circuit_designer.utils import factory
import gymnasium as gym; import numpy as np

def bell():
  # env = factory(['SP-bell-q2-d12'], seed=1, n_train=1) #, **envkwargs
  env = gym.make("CircuitDesigner-v0", max_qubits=2, max_depth=12, challenge='SP-bell')
  env.reset()

  # H
  env.step([1,0,0,np.pi/2])  
  env.step([2,0,0,np.pi/2])
  env.step([1,0,0,np.pi/2])  

  # CX
  env.step([2,1,0,np.pi])

  # M
  env.step([0,0,0,0])
  reward = env.step([0,1,0,0])[1]

  np.testing.assert_almost_equal(reward, 1)
  print("Succeeded bell test")

