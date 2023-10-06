from circuit_designer.utils import factory

def toffoli():
  env = factory(['UC-toffoli-q3-d63'], seed=1, n_train=1) #, **envkwargs
  env['train'].reset()

  # V
  env['train'].step([[1,2,2,3]]); env['train'].step([[2,2,2,3]]); env['train'].step([[1,2,2,3]])  
  env['train'].step([[1,2,1,3]])  
  env['train'].step([[1,2,2,3]]); env['train'].step([[2,2,2,3]]); env['train'].step([[1,2,2,3]])  

  # Cnot
  env['train'].step([[2,1,0,3]])  

  # V-
  env['train'].step([[1,2,2,3]]); env['train'].step([[2,2,2,3]]); env['train'].step([[1,2,2,3]])  
  env['train'].step([[1,2,1,1]])  
  env['train'].step([[1,2,2,3]]); env['train'].step([[2,2,2,3]]); env['train'].step([[1,2,2,3]])  

  # Cnot
  env['train'].step([[2,1,0,3]])  

  env['train'].step([[1,2,2,3]]); env['train'].step([[2,2,2,3]]); env['train'].step([[1,2,2,3]])  
  env['train'].step([[1,2,0,3]])  
  env['train'].step([[1,2,2,3]]); env['train'].step([[2,2,2,3]]); env['train'].step([[1,2,2,3]])  
  # CZ pi/2

  # Meassure
  env['train'].step([[0,0,0,0]])
  env['train'].step([[0,1,0,0]])
  reward = env['train'].step([[0,2,0,0]])[1]
  assert reward == 1, f"Reward not one: {reward} "
  print("Succeeded Toffoli test")

