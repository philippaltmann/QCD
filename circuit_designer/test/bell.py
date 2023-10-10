from circuit_designer.utils import factory; import numpy as np

def bell():
  env = factory(['SP-bell-q2-d12'], seed=1, n_train=1) #, **envkwargs
  env['train'].reset()

  # H
  env['train'].step([[1,0,0,np.pi/2]])  
  env['train'].step([[2,0,0,np.pi/2]])
  env['train'].step([[1,0,0,np.pi/2]])  

  # CX
  env['train'].step([[2,1,0,np.pi]])

  # M
  env['train'].step([[0,0,0,0]])
  reward = env['train'].step([[0,1,0,0]])[1]

  assert reward == 1, reward
  print("Succeeded bell test")

