import numpy as np; import pandas as pd; import random
import torch as th; import scipy.stats as st; import re
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.save_util import load_from_zip_file, recursive_setattr
from torch.utils.tensorboard.writer import SummaryWriter
from typing import Dict, List, Optional, Type, Union
from tqdm import tqdm; import os
import gymnasium as gym
import platform; import stable_baselines3 as sb3; 

from qcd_gym.wrappers.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

def _make(record_video=False, **spec):
  def _init() -> gym.Env: return Monitor(gym.make(**spec), record_video=record_video) 
  return _init

def named(env):
  max_qubits = int(re.search('-q(\d+)', env).group(1)); env = re.sub('-q(\d+)', '', env)
  max_depth = int(re.search('-d(\d+)', env).group(1)); env = re.sub('-d(\d+)', '', env)
  return {'id': 'CircuitDesigner-v0', 'max_qubits': max_qubits, 'max_depth': max_depth, 'objective': env}

def make_vec(env, seed=None, n_envs=4, **kwargs):
  spec = lambda rank: {**named(env), 'seed': seed, **kwargs}
  return DummyVecEnv([_make(**spec(i)) for i in range(n_envs)])



class TrainableAlgorithm(BaseAlgorithm):
  """ Generic Algorithm Class extending BaseAlgorithm with features needed by the training pipeline """
  def __init__(self, env:str, normalize:bool=False, policy:Union[str,Type[ActorCriticPolicy]]="MlpPolicy", path:Optional[str]=None, seed=None, silent=False, log_name=None, envkwargs={}, **kwargs):
    """ :param env: The environment to learn from (if registered in Gym, can be str)
    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...) defaults to MlpPolicy
    :param normalize: whether to use normalized observations, default: False
    :param log_name: optional custom folder name for logging
    :param path: (str) the log location for tensorboard (if None, no logging) """
    _path = lambda seed: f"{path}/{env}/{log_name or str(self.__class__.__name__)}/{seed}"
    gen_seed = lambda s=random.randint(0, 999): s if not os.path.isdir(_path(s)) else gen_seed()
    if seed is None: seed = gen_seed()
    self.path = _path(seed) if path is not None else None; self.eval_frequency, self.progress_bar = None, None
    env = make_vec(env, seed=seed, **envkwargs)
    self.normalize, self.silent, self.continue_training = normalize, silent, True; 
    super().__init__(policy=policy, seed=seed, verbose=0, env=env, **kwargs)
    
  def _setup_model(self) -> None:
    if self.normalize: self.env = VecNormalize(self.env)
    self._naming = {'l': 'length-100', 'r': 'return-100'}; self._custom_scalars = {}
    super(TrainableAlgorithm, self)._setup_model()
    self.writer, self._registered_ci = SummaryWriter(self.path+"/train") if self.path and not self.silent else None, [] 
    if not self.silent: print("+-------------------------------------------------------+\n"\
      f"| System: {platform.version()}         |\n" \
      f"| GPU: {f'Enabled, version {th.version.cuda} on {th.cuda.get_device_name(0)}' if th.cuda.is_available() else'Disabled'} |\n"\
        f"| Python: {platform.python_version()} | PyTorch: {th.__version__} | Numpy: {np.__version__} |\n" \
      f"| Stable-Baselines3: {sb3.__version__} | Gym: {gym.__version__} | Seed: {self.seed:3d}  |\n"\
        "+-------------------------------------------------------+")

  #Helper functions for writing model or hyperparameters
  def _excluded_save_params(self) -> List[str]:
    """ Returns the names of the parameters that should be excluded from being saved by pickling. 
    E.g. replay buffers are skipped by default as they take up a lot of space.
    PyTorch variables should be excluded with this so they can be stored with ``th.save``.
    :return: List of parameters that should be excluded from being saved with pickle. """
    return super(TrainableAlgorithm, self)._excluded_save_params() + ['_naming', '_custom_scalars', '_registered_ci', 'writer', 'progress_bar', 'silent']

  def should_eval(self) -> bool: return self.eval_frequency is not None and self.num_timesteps % self.eval_frequency == 0  

  def learn(self, total_timesteps: int, eval_frequency=8192, eval_kwargs={}, **kwargs) -> "TrainableAlgorithm":
    """ Learn a policy
    :param total_timesteps: The total number of samples (env steps) to train on
    :param **kwargs: further aguments are passed to the parent classes 
    :return: the trained model """
    total = self.num_timesteps+total_timesteps; stepsize = self.n_steps * self.n_envs;
    if eval_frequency is not None: self.eval_frequency = eval_frequency * self.n_envs // stepsize * stepsize or eval_frequency * self.n_envs
    # total = self.num_timesteps+total_timesteps
    # if eval_frequency is not None: self.eval_frequency = eval_frequency * self.n_envs # **2
    hps = self.get_hparams(); hps.pop('seed'); hps.pop('num_timesteps');  
    # self.progress_bar = tqdm(total=total, unit="steps", postfix=[0,""], bar_format="{desc}[R: {postfix[0]:4.2f}][{bar}]({percentage:3.0f}%)[{n_fmt}/{total_fmt}@{rate_fmt}]") 
    metrics = "M: {postfix[0]:4.2f} | Q: {postfix[1]:4.2f} | D: {postfix[2]:4.2f}"
    self.progress_bar = tqdm(total=total, unit="steps", postfix=[0,0,0,""], bar_format="{desc}["+metrics+"][{bar}]({percentage:3.0f}%)[{n_fmt}/{total_fmt}@{rate_fmt}]") 
    self.progress_bar.update(self.num_timesteps); 
    model = super(TrainableAlgorithm, self).learn(total_timesteps=total_timesteps, **kwargs) #callback=callback, 
    self.progress_bar.close()
    return model

  def train(self, **kwargs) -> None:
    if not self.continue_training: return
    self.progress_bar.postfix[0] = np.mean([ep_info["m"] for ep_info in self.ep_info_buffer])
    self.progress_bar.postfix[1] = np.mean([ep_info["q"] for ep_info in self.ep_info_buffer])
    self.progress_bar.postfix[2] = np.mean([ep_info["d"] for ep_info in self.ep_info_buffer])

    if self.should_eval(): self.progress_bar.update(self.eval_frequency); #n_steps
    summary, step = {}, self.num_timesteps 

    super(TrainableAlgorithm, self).train(**kwargs) # Train PPO & Write Training Stats 
    if self.writer == None or not self.should_eval(): return 

    # Get infos from episodes & record rewards confidence intervals to summary 
    epdata = {name: {ep['t']: ep[key] for ep in self.ep_info_buffer} for key,name in self._naming.items()}
    # TODO: fix termination reason logging
    # termination_reasons = self.env.get_attr('termination_reasons')
    # [self.logger.record(str(r).replace('.','s/'), sum([env[r] for env in termination_reasons])) for r in termination_reasons[0]]
    summary.update(self.prepare_ci(epdata, category="rewards", write_raw=True))

    # Get Metrics from logger, record (float,int) as scalars, (ndarray) as confidence intervals
    metrics = pd.json_normalize(self.logger.name_to_value, sep='/').to_dict(orient='records')[0]
    summary.update({t: {step: v} for t,v in metrics.items() if isinstance(v, (float,int))})
    summary.update(self.prepare_ci({t: {step: v} for t, v in metrics.items() if isinstance(v, np.ndarray)}))
    
    #Write metrcis summary to tensorboard 
    _r = {'DEPTH': 0, 'DONE':1} # TODO: meassure 2?
    resons = np.array([_r[r] for e in self.env.envs for r in e.termination_reasons])
    self.writer.add_histogram('training/termination', resons, step)
    [self.writer.add_scalar(tag, value, step) for tag,item in summary.items() for step,value in item.items()]
  
  def prepare_ci(self, infos: dict, category=None, confidence:float=.95, write_raw=False) -> Dict: 
    """ Computes the confidence interval := x(+/-)t*(s/√n) for a given survey of a data set. ref:
    https://github.com/WangYueFt/rfs/blob/f8c837ba93c62dd0ac68a2f4019c619aa86b8421/eval/meta_eval.py#L19
    :param infos: data dict in form {tag: {step: data, ...}}
    :param categroy: category string to write summary (also custom scalar headline), if None: deduced from info tag
    :param confidence: 
    :param wite_raw: bool flag wether to write single datapoints
    :return: Dict to be writen to tb in form: {tag: {step: value}} """
    summary, s = {}, self.num_timesteps; 
    factory = lambda c, t, m, ci: {f"{c}/{t}-mean":{s:m}, f"raw/{c}_{t}-lower":{s:m-ci}, f"raw/{c}_{t}-upper":{s:m+ci}}
    if write_raw: summary.update({f"raw/{category}_{tag}": info for tag, info in infos.items()})
    for tag, data in infos.items():
      d = np.array(list(data.values())).flatten(); mean = d.mean(); ci = st.t.ppf((1+confidence)/2, len(d)-1) * st.sem(d)
      if category == None: category,tag = tag.split('/')
      c = self._custom_scalars.get(category, {}); t = c.get(tag); update = False
      if not c: self._custom_scalars.update({category:{}}); c = self._custom_scalars.get(category); update = True
      if not t: c.update({tag: ['Margin', list(factory(category, tag, 0, 0).keys())]}); update = True
      if update: self.writer.add_custom_scalars(self._custom_scalars)
      summary.update(factory(category, tag, mean, ci))
    return summary

  def get_hparams(self):
    """ Fetches, filters & flattens own hyperparameters
    :return: Dict of Hyperparameters containing numbers and strings only """ 
    exclude = ['device','verbose','writer','tensorboard_log','start_time','rollout_buffer','eval_env']+\
      ['policy','policy_kwargs','policy_class','lr_schedule','sde_sample_freq','clip_range','clip_range_vf']+\
      ['env','observation_space','action_space','action_noise','ep_info_buffer','ep_success_buffer','target_kl']+\
      ['envs', 'path', 'progress_bar', 'disc_kwargs', 'buffer','log_ent_coef']
    hparams = pd.json_normalize(
      {k: v.__name__ if isinstance(v, type) else v.get_hparams() if hasattr(v, 'get_hparams') else 
          v for k,v in vars(self).items() if not(k in exclude or k.startswith('_'))
      }, sep="_").to_dict(orient='records')[0]
    hp_dis = {k:v for k,v in hparams.items() if isinstance(v, (int, bool))}
    hp_num = {k:v for k,v in hparams.items() if isinstance(v, (float))}
    hp_str = {k:str(v) for k,v in hparams.items() if isinstance(v, (str, list))}
    hp_mdd = {k:v for k,v in hparams.items() if isinstance(v, (dict, th.Tensor))} #list
    assert not len(hp_mdd), "Skipped writing hparams of multi-dimensional data"
    return {**hp_dis, **hp_num, **hp_str, **{'env_name': self.env.envs[0].unwrapped.name}}
  
  def save(self, name="/model/train", **kwargs) -> None: 
    kwargs['path'] = self.path + name; super(TrainableAlgorithm, self).save(**kwargs)

  @classmethod
  def load(cls, load,device: Union[th.device, str] = "auto", **kwargs) -> "TrainableAlgorithm":
    assert os.path.exists(f"{load}/model/train.zip"), f"Attempting to load a model from {load} that does not exist"
    data, params, pytorch_variables = load_from_zip_file(f"{load}/model/train", device=device)
    assert data is not None and params is not None, "No data or params found in the saved file"
    model = cls(device=device, _init_setup_model=False, **kwargs)
    model.__dict__.update(data); model.__dict__.update(kwargs); model._setup_model()
    model.set_parameters(params, exact_match=True, device=device)
    if pytorch_variables is not None: [recursive_setattr(model, f"{name}.data", pytorch_variables[name].data)
        for name in pytorch_variables if pytorch_variables[name] is not None]
    if model.use_sde: model.policy.reset_noise() 
    model.num_timesteps -= model.num_timesteps%(model.n_steps * model.n_envs)
    return model
