import gymnasium as gym; import numpy as np

def bell():
  env = gym.make("CircuitDesigner-v0", max_qubits=2, max_depth=12, challenge='SP-bell')
  env.reset()

  # H
  env.step([0,0,0,np.pi/2])  
  env.step([1,0,0,np.pi/2])
  env.step([0,0,0,np.pi/2])  

  # CX
  env.step([1,1,0,np.pi])

  # T
  reward = env.step([2,0,0,0])[1]

  np.testing.assert_almost_equal(reward, 1)
  print("Succeeded bell test")

