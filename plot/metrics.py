import os; from os import path; import itertools; from tqdm import tqdm
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator as EA
import pandas as pd; import numpy as np; import scipy.stats as st; import re
import gymnasium as gym; from algorithm.algorithm import named
from qcd_gym.wrappers import Monitor
from baselines import *


def extract_model(exp, run):
  return None

def fetch_experiments(base='./results', alg=None, env=None, metrics=[], dump_csv=False, baseline=None, random_baseline=True):
  """Loads and structures all tb log files Given:
  :param base_path (str):
  :param env (optional): the environment to load 
  :param alg (optional): the algorithm to load 
  :param metrics: list of (Name, Tag) tuples of metrics to load
  :param dump_csv: save loaded experiments to csv
  Returns: list with dicts of experiments 
  """
  # Helper to fetch all relevant folders 
  subdirs = lambda dir: [d for d in os.scandir(dir) if d.is_dir()]

  print(f"Scanning for {env if env else 'environments'} in {base}")  # First layer: Environments
  if env: experiments = [{'env': e.name, 'path': e.path} for e in subdirs(base) if env in e.name]
  else: experiments = [{'env': e.name, 'path': e.path} for e in subdirs(base) if e.name != 'plots']

  print(f"Scanning for {alg if alg else 'algorithms'} in {base}")  # Second layer: Algorithms
  if alg: experiments = [{**exp, 'algorithm': alg, 'path': a} for exp in tqdm(experiments) for a in subdirs(exp['path']) if alg == a.name]
  else: experiments = [{**exp, 'algorithm': a.name, 'path': a} for exp in tqdm(experiments) for a in subdirs(exp['path']) if any([n in [*ALGS] for n in a.name.split('-')])]


  # Third Layer: Count Runs / fetch tb files
  print(f"Scanning for hyperparameters in {base}")  # Third layer: Hyperparameters & number of runs
  experiments = [{ **e, 'runs': len(subdirs(e['path'])) } for e in tqdm(experiments) if os.path.isdir(e['path'])]

  progressbar = tqdm(total=sum([exp['runs'] for exp in experiments])* len(metrics))
  data_buffer = {}

  def fetch_data(exp, run_path, name, key):
    # Load data from csv if possible
    if path.isfile(f'{run_path}/{name}.csv'): return pd.read_csv(f'{run_path}/{name}.csv').set_index('Step')

    # Use buffered Event Accumulator if already open
    if log := data_buffer.get(run_path):
      extract_args = {'columns': ['Time', 'Step', 'Data'], 'index': 'Step', 'exclude': ['Time']}
      if key == 'auto':
        d = int(exp['env'].split('-d')[1])
        data = fetch_data(exp, run_path, 'Return', 'rewards/return-100-mean')
        depth = fetch_data(exp, run_path, 'Depth', 'rewards/depth-100-mean')
        pen = depth['Data'].apply(lambda D: max(0,D-d/3)/(d/2*3))
        if name == 'Metric': data['Data'] += pen
        if name == 'Cost': data['Data'] = pen
      else: 
        data = pd.DataFrame.from_records([(s.wall_time, s.step, s.value) for s in log.Scalars(key)], **extract_args)
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


def group_experiments(experiments, groupby=['env'], mergeon=None): #merge=None
  # Graphical helpers for titles, labels
  forms = ['algorithm', 'env']
  def label(exp):
    i = {key: re.sub(r'[0-9]+ ', '', exp[key]) for key in forms if key in exp and key not in groupby}
    check = lambda keys,base,op=all: op([k in base for k in keys])
    return f"{i['algorithm']}-{i['explorer']}" if 'explorer' in i else i['algorithm']

  title = lambda exp: ' '.join([exp[key] for key in forms if key in exp and key in groupby])

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
  getgraph = lambda exp, index: { 'label': label(exp), 'data': exp['data'], 'models': exp['models']} 
  return [{'title': title, 'graphs': [ getgraph(exp, index) for exp in experiments if ingroup(exp, group) ] } for group, index, title in options ]


def calculate_metrics(plots, metrics):
  """Given a set of experiments and metrics, applies metric calulation
  :param plots: list of dicts containing plot information and graphs with raw data
  :param metrics: Dicts of metric names and calculation processes 
  """
  def process(metric, proc, plot):
    graphs = [ { **graph, 'data': proc(graph['data'][metric], graph['models']) } for graph in plot['graphs']]
    return [{ **plot, 'graphs': graphs, 'metric': metric}]
  return [ result for metric in metrics for plot in plots for result in process(*metric, plot)]


def process_ci(data, models):
  # Helper to claculate confidence interval
  ci = lambda d, confidence=0.95: st.t.ppf((1+confidence)/2, len(d)-1) * st.sem(d)
  reward_range = (0,1)
  # Prepare Data (fill until highest index)
  steps = [d.index[-1] for d in data]; maxsteps = np.max(steps)
  for d in data: d.at[maxsteps, 'Data'] = float(d.tail(1)['Data'].iloc[0])
  data = pd.concat(data, axis=1, ignore_index=False, sort=True).bfill()
  
  # Mean 1..n | CI 1..n..1
  mean, h = data.mean(axis=1), data.apply(ci, axis=1)
  ci = pd.concat([mean+h, (mean-h).iloc[::-1]])
  return (mean, ci, reward_range)


def process_steps(data, models): return ([d.index[-1] for d in data], 10e5)


def fetch_evo(base, experiment, EPS=100, dump=False, load=True):
  out = f"{base}/{experiment['title']}/GA"
  if not os.path.exists(out) or not load:
    print(f"Running Evo Baseline for {experiment['title']}")
    data = {m: [] for m in experiment['graphs'][0]['data'].keys()}
    steps = [i for g in experiment['graphs'] for i in g['data'][list(data.keys())[0]][0].index]; 
    index = [np.min(steps), np.max(steps)]
    for s in range(8):
      [d.append([]) for d in data.values()]
      info = run_evo(experiment['title'], s+1)
      for key,val in data.items(): 
        val[-1] = pd.DataFrame([info[key]]*2, index=index, columns=['Data']).rename_axis(index='Step')

    if dump: 
      os.makedirs(f"{out}", exist_ok=True)
      [pd.concat(metric, axis=1).to_csv(f"{out}/{key}.csv") for key, metric in data.items()]

  else: data = { m: [
        c.to_frame('Data') for _,c in pd.read_csv(f"{out}/{m}.csv").set_index('Step').items()
      ] for m in experiment['graphs'][0]['data'].keys()}
    
  experiment['graphs'].append({'label': 'GA', 'models': [None], 'data': data})
  

def fetch_random(base, experiment, EPS=100, dump=False, load=True):
  out = f"{base}/{experiment['title']}/Random"
  if not os.path.exists(out) or not load:
    print(f"Running Random Baseline for {experiment['title']}")
    M = {'Return': 'r', 'Depth': 'd', 'Qubits': 'q', 'Metric': 'm', 'Cost': 'c'}
    env = Monitor(gym.make(**named(experiment['title']), seed=42))  #, discrete=False
    data = {m: [] for m in experiment['graphs'][0]['data'].keys()}
    steps = [i for g in experiment['graphs'] for i in g['data'][list(data.keys())[0]][0].index]; 
    index = [np.min(steps), np.max(steps)]
    for s in range(8):
      env.action_space.seed(s+1); [d.append([]) for d in data.values()]
      while len(data[list(data.keys())[0]][-1]) < EPS:
        env.reset(); terminated = False; truncated = False
        while not (terminated or truncated): _, _, terminated, truncated, info = env.step(env.action_space.sample())
        [val[-1].append(info['episode'][M[key]]) for key,val in data.items()]
      for val in data.values(): val[-1] = pd.DataFrame([sum(val[-1])/EPS]*2, index=index, columns=['Data']).rename_axis(index='Step')
    if dump: 
      os.makedirs(f"{out}", exist_ok=True)
      [pd.concat(metric, axis=1).to_csv(f"{out}/{key}.csv") for key, metric in data.items()]

  else: data = { m: [
        c.to_frame('Data') for _,c in pd.read_csv(f"{out}/{m}.csv").set_index('Step').items()
      ] for m in experiment['graphs'][0]['data'].keys()}

  experiment['graphs'].append({'label': 'Random', 'models': [None], 'data': data})
  