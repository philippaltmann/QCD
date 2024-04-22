import gymnasium as gym; import numpy as np

def toffoli():
  env = gym.make("CircuitDesigner-v0", max_qubits=3, max_depth=63, challenge='UC-toffoli')
  env.reset()

  # V
  env.step([0,2,2,np.pi/2]); env.step([1,2,2,np.pi/2]); env.step([0,2,2,np.pi/2])  
  env.step([0,2,1,np.pi/2])  
  env.step([0,2,2,np.pi/2]); env.step([1,2,2,np.pi/2]); env.step([0,2,2,np.pi/2])  

  # Cnot
  env.step([1,1,0,np.pi/2])  

  # V-
  env.step([0,2,2,np.pi/2]); env.step([1,2,2,np.pi/2]); env.step([0,2,2,np.pi/2])  
  env.step([0,2,1,-np.pi/2])  
  env.step([0,2,2,np.pi/2]); env.step([1,2,2,np.pi/2]); env.step([0,2,2,np.pi/2])  

  # Cnot
  env.step([1,1,0,np.pi/2])  

  env.step([0,2,2,np.pi/2]); env.step([1,2,2,np.pi/2]); env.step([0,2,2,np.pi/2])  
  env.step([0,2,0,np.pi/2])  
  env.step([0,2,2,np.pi/2]); env.step([1,2,2,np.pi/2]); env.step([0,2,2,np.pi/2])  
  # CZ pi/2

  # Meassure
  reward = env.step([2,0,0,0])[1]
  np.testing.assert_almost_equal(reward, 1)
  print("Succeeded Toffoli test")
