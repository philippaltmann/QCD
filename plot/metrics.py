import os; from os import path; import itertools; from tqdm import tqdm
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator as EA
import pandas as pd; import numpy as np; import scipy.stats as st; import re
# from ai_safety_gym import factory, make, env_spec
from circuit_designer.utils import factory
from baselines import *

# TODO: acc pretrain scores 

def extract_model(exp, run):
  return None
  if '-' in exp['algorithm']: explorer,exp['algorithm'] = exp['algorithm'].split('-')
  algorithm, seed = eval(exp['algorithm']), int(run.name)
  # TODO: load explorer if not in ['Random', 'LOAD']
  model = algorithm.load(load=run.path, seed=seed, envs=[exp['env']], path=None, device='cpu')
  return model

def fetch_experiments(base='./results', alg=None, env=None, metrics=[], dump_csv=False, baseline=None):
  """Loads and structures all tb log files Given:
  :param base_path (str):
  :param env (optional): the environment to load 
  :param alg (optional): the algorithm to load 
  :param metrics: list of (Name, Tag) tuples of metrics to load
  :param save_csv: save loaded experiments to csv
  Returns: list with dicts of experiments 
  """
  # Helper to fetch all relevant folders 
  subdirs = lambda dir: [d for d in os.scandir(dir) if d.is_dir()]

  print(f"Scanning for {env if env else 'environments'} in {base}")  # First layer: Environments
  if env: experiments = [{'env': e.name, 'path': e.path} for e in subdirs(base) if env in e.name]
  else: experiments = [{'env': e.name, 'path': e.path} for e in subdirs(base) if e.name != 'plots']

  print(f"Scanning for {alg if alg else 'algorithms'} in {base}")  # Second layer: Algorithms
  if alg: experiments = [{**exp, 'algorithm': alg, 'path': a} for exp in tqdm(experiments) for a in subdirs(exp['path']) if alg == a.name]
  else: experiments = [{**exp, 'algorithm': a.name, 'path': a} for exp in tqdm(experiments) for a in subdirs(exp['path']) if any([n in ALGS for n in a.name.split('-')])]

  # Split explorer:
  # experiments = [{**e, 'algorithm': e['algorithm'].split('-')[-1], 'explorer': e['algorithm'].split('-')[0] if len(e['algorithm'].split('-'))>1 else 'Random'} for e in tqdm(experiments)]


  # Third Layer: Count Runs / fetch tb files
  print(f"Scanning for hyperparameters in {base}")  # Third layer: Hyperparameters & number of runs
  experiments = [{ **e, 'runs': len(subdirs(e['path'])) } for e in tqdm(experiments) if os.path.isdir(e['path'])]
  # experiments = [{ **exp, 'path': e.path,  'method': e.name, 'runs': len(subdirs(e)) } for exp in tqdm(experiments) if os.path.isdir(exp['path']) for e in subdirs(exp['path'])] # With hp

  progressbar = tqdm(total=sum([exp['runs'] for exp in experiments])* len(metrics))
  data_buffer = {}

  def fetch_data(exp, run_path, name, key):
    # Load data from csv if possible
    if path.isfile(f'{run_path}/{name}.csv'): return pd.read_csv(f'{run_path}/{name}.csv').set_index('Step')

    # Use buffered Event Accumulator if already open
    if log := data_buffer.get(run_path):
      extract_args = {'columns': ['Time', 'Step', 'Data'], 'index': 'Step', 'exclude': ['Time']}
      # print(log.scalars.Keys())
      data = pd.DataFrame.from_records([(s.wall_time, s.step, s.value) for s in log.Scalars(key)], **extract_args)
      # data = pd.DataFrame.from_records(log.Scalars(key), **extract_args)
      data = data.loc[~data.index.duplicated(keep='first')] # Remove duplicate indexes
      if dump_csv: data.to_csv(f'{run_path}/{name}.csv')
      return data    
    data_buffer.update({run_path: EA(run_path+"/train/").Reload()})
    return fetch_data(exp, run_path, name, key)
  
  def extract_data(exp, run, name, key):
    progressbar.update()
    if name == 'Model': return key
    data = fetch_data(exp, run.path, name, key)
    if baseline is None: return data
    path = run.path.replace(base, baseline)
    prev = fetch_data(exp, path, name, key)
    data.index = data.index - prev.index[-1] # Norm by last baseline index
    return data
  
  # Process given experiments
  finished = lambda dir: [d for d in subdirs(dir) if len(subdirs(d))] # subdirs(d)[0].name == 'model'
  process_data = lambda exp, name, key: [ extract_data(exp, run, name, key) for run in finished(exp['path']) ] 
  fetch_models = lambda exp: [ extract_model(exp, run) for run in finished(exp['path'])] 
  experiments = [{**exp, 'data': { name: process_data(exp, *scalar) for name, scalar in metrics }, 'models': fetch_models(exp)} for exp in experiments]
  progressbar.close()

  return experiments


def group_experiments(experiments, groupby=['algorithm', 'env'], mergeon=None): #merge=None
  # Graphical helpers for titles, labels
  forms = ['algorithm', 'explorer', 'env']
  def label(exp):
    i = {key: re.sub(r'[0-9]+ ', '', exp[key]) for key in forms if key in exp and key not in groupby}
    check = lambda keys,base,op=all: op([k in base for k in keys])
    return f"{i['algorithm']}-{i['explorer']}" if 'explorer' in i else i['algorithm']
    # return f"{i['algorithm'] if check(['algorithm', 'method'],i) and check(['Full','RAD'],i['method'], any) else ''} {'FO' if check(['Full'],i['method']) else i['method']}"

  title = lambda exp: ' '.join([exp[key] for key in forms if key in exp and key in groupby])
  # hue = lambda alg: [hue for key,hue in {'Object': 150, 'Action': 200, 'Radius': 230, 'A2C':40, 'Full': 350, 'RAD': 70}.items() if key in alg][0]
  hue = lambda alg: [hue for key,hue in {'PPO': 150, 'SAC': 200, 'DQN': 230, 'TD3':40, 'TODO': 350, 'TODO': 70}.items() if key in alg][0]

  # Create product of all occuances of specified groups, zip with group titles & add size and a counter of visited group items
  def merge(experiments, key):
    values = {exp[key]:'' for exp in experiments}.keys(); get = lambda d,k,*r: get(d[k],*r) if len(r) else d[k]
    merge = lambda val, *k: [item for exp in experiments for item in get(exp,*k) if exp[key]==val] 
    extract = lambda val, k, *r, p=[]: {k: extract(val, *r, p=[*p,k])} if len(r) else {k: merge(val, *p, k) } 
    data = lambda val: {k:v for key in experiments[0]['data'] for k,v in extract(val, 'data', key)['data'].items()}
    return[{key:val,  **extract(val, 'models'), 'data': data(val)} for val in values]
  if mergeon is not None: experiments = merge(experiments, mergeon)
  options = list(itertools.product(*[ list(dict.fromkeys([exp[group] for exp in experiments])) for group in groupby ]))
  ingroup = lambda experiment, group: all([experiment[k] == v for k,v in zip(groupby, group)])
  options = list(zip(options, [[ title(exp) for exp in experiments if ingroup(exp,group)] for group in options ]))
  options = [(group, [0, len(titles)], titles[0]) for group, titles in options]
  getgraph = lambda exp, index: { 'label': label(exp), 'data': exp['data'], 'models': exp['models'], 'hue': hue(exp['algorithm'])} 
  return [{'title': title, 'graphs': [ getgraph(exp, index) for exp in experiments if ingroup(exp, group) ] } for group, index, title in options ]


def calculate_metrics(plots, metrics):
  """Given a set of experiments and metrics, applies metric calulation
  :param plots: list of dicts containing plot information and graphs with raw data
  :param metrics: Dicts of metric names and calculation processes 
  """
  def process(metric, proc, plot):
    graphs = [ { **graph, 'data': proc(graph['data'][metric], graph['models']) } for graph in plot['graphs']]
    if metric == 'Heatmap':
      return [ { 'title': f"{plot['title']} | {graph['label']} | {key} ", 'data': data, 'metric': metric} 
        for graph in graphs for key, data in graph['data'].items() ]
    return [{ **plot, 'graphs': graphs, 'metric': metric}]
  return [ result for metric in metrics for plot in plots for result in process(*metric, plot)]


def process_ci(data, models):
  # Helper to claculate confidence interval
  ci = lambda d, confidence=0.95: st.t.ppf((1+confidence)/2, len(d)-1) * st.sem(d)
  # stop = models[0].env.get_attr('reward_threshold')[0]
  # reward_range = models[0].env.get_attr('env')[0].env.reward_range
  reward_range = (0,1)
  # upper = models[0].env.get_attr('env')[0].spec.reward_threshold if not models[0].continue_training else None # Retrieve max from threhsold 
  # Prepare Data (fill until highest index)
  steps = [d.index[-1] for d in data]; maxsteps = np.max(steps)
  for d in data: d.at[maxsteps, 'Data'] = float(d.tail(1)['Data'])
  data = pd.concat(data, axis=1, ignore_index=False, sort=True).bfill()
  
  # Mean 1..n | CI 1..n..1
  mean, h = data.mean(axis=1), data.apply(ci, axis=1)
  ci = pd.concat([mean+h, (mean-h).iloc[::-1]]).clip(*reward_range)
  return (mean, ci, reward_range)
  # return (mean, ci, stop)


def process_steps(data, models): return ([d.index[-1] for d in data], 10e5)


iterate = lambda model, envs, func: [ func(env, k,i) for env in envs for k,i in model.heatmap_iterations.items() ]
heatmap = lambda model, envs: iterate(model, envs, lambda env, k,i: env.envs[0].iterate(i[0]))
# metadata = lambda model, envs: iterate(model, envs, lambda env, k,i: (f'{k.capitalize()} {env_spec(env)._kwargs["level_choice"]}', i[1]))
# make_env = lambda model, spec: [e for e in model.envs['test'].values() if spec in env_spec(e)._env_name][0]

def process_heatmap(specs, models):
  raise(NotImplementedError)
  # TODO: wrapper??
  # setting = list(zip(models, [[make_env(model, s) for s in spec] for model,spec in zip(models,specs)]))
  # return { k:d for (k,a),d in zip(metadata(*setting[0]), np.moveaxis(np.array([heatmap(*s) for s in setting]), 0, -1))} 


def process_eval(specs, models, deterministic=True):
  raise(NotImplementedError)
  # from stable_baselines3.common.evaluation import evaluate_policy
  # # assert False, "Make env not using CROP wrappper"
  # data = [(model, make_env(model, spec)) for model,spec in zip(models,specs)]
  # termination_reasons = data[0][1].get_attr('termination_reasons')[0]
  # def callback(g,l): 
  #   if 'episode' in g['info']: termination_reasons[g['info']['extra_observations']['termination_reason']] += 1
  # if deterministic: eval = [evaluate_policy(*args, n_eval_episodes=1, callback=callback)[0] for args in data ]
  # else: eval = [r for args in data for r in evaluate_policy(*args, n_eval_episodes=100, deterministic=False, return_episode_rewards=True, callback=callback)[0]]

  # return (eval, data[0][1].get_attr('reward_threshold')[0], termination_reasons)
  
