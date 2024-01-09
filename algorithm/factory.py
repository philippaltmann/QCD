import gymnasium as gym; import re
from circuit_designer.wrappers.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

def _make(record_video=False, **spec):
  def _init() -> gym.Env: return Monitor(gym.make(**spec), record_video=record_video) 
  return _init

def named(env):
  max_qubits = int(re.search('-q(\d+)', env).group(1)); env = re.sub('-q(\d+)', '', env)
  max_depth = int(re.search('-d(\d+)', env).group(1)); env = re.sub('-d(\d+)', '', env)
  return {'id': 'CircuitDesigner-v0', 'max_qubits': max_qubits, 'max_depth': max_depth, 'challenge': env}

def make_vec(env, seed=None, n_envs=1, **kwargs):
  spec = lambda rank: {**named(env), 'seed': seed+rank, **kwargs}
  return DummyVecEnv([_make(**spec(i)) for i in range(n_envs)])

def factory(env_spec, n_train=4, **kwargs):
  assert len(env_spec) > 0, 'Please specify at least one environment for training'
  test_names = ['validation', *[f'evaluation-{i}' for i in range(len(env_spec)-1)]]
  return { 'train': make_vec(env_spec[0], n_envs=n_train, **kwargs), 
    'test': {name: make_vec(spec, render_mode='text', record_video=True, **kwargs) for name, spec in zip(test_names, env_spec)}
  }   