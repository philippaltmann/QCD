import gymnasium as gym; import numpy as np

def ghz():
  env = gym.make("CircuitDesigner-v0", max_qubits=3, max_depth=15, challenge='SP-ghz3')
  env.reset()
  
  # H
  env.step([1,0,0,np.pi/2]); env.step([2,0,0,np.pi/2]); env.step([1,0,0,np.pi/2])  

  # CX
  env.step([2,1,0,np.pi]); env.step([2,2,1,np.pi])

  # M
  env.step([0,0,0,0]); env.step([0,1,0,0])
  reward = env.step([0,2,0,0])[1]

  np.testing.assert_almost_equal(reward, 1)
  print("Succeeded GHZ test")


