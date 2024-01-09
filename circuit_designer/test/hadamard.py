import gymnasium as gym; import numpy as np

def hadamard():
  # Test 1-qubit H
  env = gym.make("CircuitDesigner-v0", max_qubits=1, max_depth=9, challenge='UC-hadamard')
  env.reset()
  env.step([1,0,0,3])  
  env.step([2,0,0,3])
  env.step([1,0,0,3])  
  reward = env.step([0,0,0,0])[1] # Meassure
  # assert reward == 1

  # Test 2-qubit H
  env = gym.make("CircuitDesigner-v0", max_qubits=2, max_depth=9, challenge='UC-hadamard')

  env.reset()
  env.step([1,0,0,np.pi/2])  
  env.step([2,0,0,np.pi/2])
  env.step([1,0,0,np.pi/2])  
  reward = env.step([0,0,0,0])[1] # Meassure
  reward = env.step([0,1,0,0])[1] # Meassure
  np.testing.assert_almost_equal(reward, 1)
  print("Succeeded Hadamard test")
