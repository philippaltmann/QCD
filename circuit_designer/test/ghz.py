from circuit_designer.utils import factory

def ghz():
  envs = factory(['SP-ghz3-q3-d15'], seed=1, n_train=1) #, **envkwargs
  env = envs['test']['validation']
  env = envs['train']; env.render_mode='text'
  env.reset()
  
  # H
  env.step([[1,0,0,3]]); env.step([[2,0,0,3]]); env.step([[1,0,0,3]])  

  # CX
  env.step([[2,1,0,4]]); env.step([[2,2,1,4]])

  # M
  env.step([[0,0,0,0]]); env.step([[0,1,0,0]])
  reward = env.step([[0,2,0,0]])[1]

  assert reward == 1
  print("Succeeded GHZ test")


