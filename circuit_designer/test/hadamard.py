from circuit_designer import factory; import numpy as np

def hadamard():
  # Test 1-qubit H
  env = factory(['UC-hadamard-q1-d9'], seed=1, n_train=1)
  env['train'].reset()
  env['train'].step([[1,0,0,3]])  
  env['train'].step([[2,0,0,3]])
  env['train'].step([[1,0,0,3]])  
  reward = env['train'].step([[0,0,0,0]])[1] # Meassure
  # assert reward == 1

  # Test 2-qubit H
  env = factory(['UC-hadamard-q2-d9'], seed=1, n_train=1)
  env['train'].reset()
  env['train'].step([[1,0,0,np.pi/2]])  
  env['train'].step([[2,0,0,np.pi/2]])
  env['train'].step([[1,0,0,np.pi/2]])  
  reward = env['train'].step([[0,0,0,0]])[1] # Meassure
  reward = env['train'].step([[0,1,0,0]])[1] # Meassure
  assert reward == 1, f"Reward not one: {reward} "
  print("Succeeded Hadamard test")
