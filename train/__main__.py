import argparse; import time
from baselines import *
start = time.time()

# General Arguments
parser = argparse.ArgumentParser()
parser.add_argument('method', help='The algorithm to use', choices=[*ALGS])
parser.add_argument( '-e', dest='env', metavar="Environment")

parser.add_argument('-s', dest='seed', type=int, help='The random seed. If not specified a free seed [0;999] is randomly chosen')
parser.add_argument('-t', dest='timesteps', type=int, help='The number of timesteps to explore.', default=128*(2048*4)) #~10e5
parser.add_argument('--load', type=str, help='Path to load the model.')
parser.add_argument('--test', help='Run in test mode (dont write log files).', action='store_true')
parser.add_argument('--path', default='results', help='The base path, defaults to `results`')
parser.add_argument('--punish', action='store_true')
parser.add_argument('--sparse', action='store_true')

# Get arguments & extract training parameters & merge model args
args = {key: value for key, value in vars(parser.parse_args()).items() if value is not None}; 
if args.pop('test'): args['path'] = None
args['envkwargs'] = {'punish': args.pop('punish'), 'sparse': args.pop('sparse')}
timesteps = args.pop('timesteps'); presteps = 0
load = args.pop('load', None)

# Init Training Model
trainer = eval(args.pop('method'))
 
model = trainer(**args)
model._naming = {**model._naming , 'd': 'depth-100', 'q': 'qbits-100', 'o': 'ops-100', 'm': 'metric-100', 'c': 'cost-100'}

if load is not None:
  _params = model.get_parameters()['policy'].copy().__str__()
  model.set_parameters({**model.get_parameters(), 'policy': model.policy.load(load).state_dict()}) # v2: load policy from file
  assert _params != model.get_parameters()['policy'].copy().__str__(), "Load failed"

print(f"Training {trainer.__name__ } in {args['env']} for {timesteps-presteps:.0f} steps.") 
model.learn(total_timesteps=timesteps-presteps) 
if model.path: model.save()
print(f"Done in {time.time()-start}")
