import argparse; import time; import os; 
from plot.metrics import *
from plot.plotting import * 
import plotly.graph_objects as go

options = { # Title, Scalar(Name, Tag), process(scalar)->data, display(data)->trace
  'Return': (('Return', 'rewards/return-100-mean'), process_ci, plot_ci),
  'Qubits': (('Qubits', 'rewards/qbits-100-mean'), process_ci, plot_ci),
  'Depth': (('Depth', 'rewards/depth-100-mean'), process_ci, plot_ci),
  'Metric': (('Metric', 'rewards/metric-100-mean'), process_ci, plot_ci),
  'Cost': (('Cost', 'rewards/cost-100-mean'), process_ci, plot_ci),
}

# Process commandline arguments 
parser = argparse.ArgumentParser()
parser.add_argument('base', help='The results root')
parser.add_argument('-a', dest='alg', nargs='+', help='The algorithm to vizualise') #choices=[*ALGS]
parser.add_argument('-b', dest='baseline', action='store_true')
parser.add_argument('-e', dest='env', help='Environment to vizualise.')
parser.add_argument('-g', dest='groupby', nargs='+', default=['env'], metavar="groupby", help='Experiment keys to group plotted data by.')
parser.add_argument('-m', dest='metrics', nargs='+', default=['Qubits', 'Depth', 'Metric'], choices=options.keys(), help='Experiment keys to group plotted data by.')
parser.add_argument('--mergeon', help='Key to merge experiments e.g. algorithm.')
parser.add_argument('--no-dump', dest='dump_csv', action='store_false', help='Skip csv dump')

args = vars(parser.parse_args()); tryint = lambda s: int(s) if s.isdigit() else s
if args['alg']: args['alg'] = ' '.join(args['alg'])
groupby = args.pop('groupby'); mergeon = args.pop('mergeon'); baseline = args.pop('baseline')
mergemetrics,_ = (True, groupby.remove('metrics')) if 'metrics' in groupby else (False,None)

metrics = [(metric, *options[metric]) for metric in args.pop('metrics')]
titles, scalars, procs, plotters = zip(*metrics)

experiments = fetch_experiments(**args, metrics=list(zip(titles, scalars)))
experiments = group_experiments(experiments, groupby, mergeon)
if baseline: 
  [fetch_evo(args['base'], exp, dump=args['dump_csv']) for exp in experiments]
  [fetch_random(args['base'], exp, dump=args['dump_csv']) for exp in experiments]

experiments = calculate_metrics(experiments, list(zip(titles, procs)))
if mergemetrics:
  experiments = [ {'title': t, 'metric': metrics[0][0], 'merge': True, 
    'graphs': [ {**g, 'label': ' '.join([g['label'],m[0]])} for exp, m in zip(
        [exp for exp in experiments if exp['title'] == t] , metrics
      ) for g in exp['graphs'] if exp['title'] == t]
    } for t in list(dict.fromkeys([exp['title'] for exp in experiments]))]
figures = generate_figures(experiments, dict(zip(titles, plotters)))

# Save figures
out = f'{args["base"]}/plots/{"-".join(groupby)}'; os.makedirs(out, exist_ok=True)
print("Done Evaluating. Saving Plots.")
for name, figure in tqdm(figures.items()): 
  go.Figure().write_image(f'{out}/{name}.pdf', format="pdf")
  time.sleep(1) # Await package loading to aviod warning boxes
  figure.write_image(f'{out}/{name}.pdf', format="pdf")