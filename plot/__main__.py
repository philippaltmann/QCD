import argparse; import time; import os; 
from plot.metrics import *
from plot.plotting import * 

options = { # Title, Scalar(Name, Tag), process(scalar)->data, display(data)->trace
  'Return': (('Return', 'rewards/return-100-mean'), process_ci, plot_ci),
  'Validation': (('Validation', 'rewards/validation'), process_ci, plot_ci),
  'Evaluation': (('Evaluation', 'rewards/evaluation-0'), process_ci, plot_ci),
}

# Process commandline arguments 
parser = argparse.ArgumentParser()
parser.add_argument('base', help='The results root')
parser.add_argument('-a', dest='alg', nargs='+', help='The algorithm to vizualise') #choices=[*ALGS]
parser.add_argument('-b', dest='baseline', help='Base path of reloaded model.')
parser.add_argument('-e', dest='env', help='Environment to vizualise.')
parser.add_argument('-g', dest='groupby', nargs='+', default=[], metavar="groupby", help='Experiment keys to group plotted data by.')
parser.add_argument('-m', dest='metrics', nargs='+', default=[], choices=options.keys(), help='Experiment keys to group plotted data by.')
parser.add_argument('--heatmap', nargs='+', default=[], help='Environment to vizualise.')
parser.add_argument('--mergeon', help='Key to merge experiments e.g. algorithm.')
parser.add_argument('--no-dump', dest='dump_csv', action='store_false', help='Skip csv dump')

args = vars(parser.parse_args()); tryint = lambda s: int(s) if s.isdigit() else s
if args['alg']: args['alg'] = ' '.join(args['alg'])
hm = args.pop('heatmap') # [tryint(s) for s in args.pop('heatmap')]
groupby = args.pop('groupby'); mergeon = args.pop('mergeon');
mergemetrics,_ = (True, groupby.remove('metrics')) if 'metrics' in groupby else (False,None)
if len(hm): options['Heatmap'] = (('Model', hm), process_heatmap, get_heatmap(True, True)); args['metrics'].append('Heatmap') 

metrics = [(metric, *options[metric]) for metric in args.pop('metrics')]
titles, scalars, procs, plotters = zip(*metrics)

experiments = fetch_experiments(**args, metrics=list(zip(titles, scalars)))
experiments = group_experiments(experiments, groupby, mergeon)
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
if len(hm): os.makedirs(out+'/Heatmaps', exist_ok=True)
print("Done Evaluating. Saving Plots.")
for name, figure in tqdm(figures.items()): figure.write_image(f'{out}/{name}.svg')