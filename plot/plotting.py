import math; import re; import plotly.graph_objects as go

title = lambda plot, y=None: [y or '' if 'merge' in plot.keys() else plot["metric"], plot["title"]]

qb = lambda name: int(re.search('-q(\d+)', name).group(1))
dp = lambda name: int(re.search('-d(\d+)', name).group(1))

# Helper functions to create scatters/graphs from experiment & metric
def plot_ci(plot):  
  traceorder = lambda key: [i for k,i in {'A2C': 0, 'PPO': 1, 'SAC': 2, 'TD3': 3, '': 4}.items() if k in key][0]
  plot['graphs'].sort(key=lambda g: traceorder(g['label'])) # Sort traces  
  smooth = lambda g: {} if 'Random' in g['label'] else {'shape':  'spline',  'smoothing': 0.4}
  scatter = lambda data, **kwargs: go.Scatter(x=data.index, y=data, **kwargs)
  # dash = lambda g: {'dash': 'dash'} if 'Evaluation' in plot['metric'] else {}
  dash = lambda g: {'dash': 'dash'} if 'Random' in g['label'] else {}
  getmean = lambda g: scatter(g['data'][0], name=g['label'], mode='lines', line={'color': color(g), **smooth(g),  **dash(g)})
  getconf = lambda g: scatter(g['data'][1], fillcolor=color(g, 1), fill='toself', line={'color': 'rgba(255,255,255,0)', **smooth(g)}, showlegend=False)
  # threshold = [go.Scatter(y=[plot['graphs'][0]['data'][2][1]]*2, x=[0,max([g['data'][0].tail(1).index[0] for g in plot['graphs']])],
  #   name='Solved', mode='lines', line={'dash':'dot', 'color':'rgb(64, 64, 64)'})] #Threshold
  data = [getconf(g) for g in plot['graphs']] + [getmean(g) for g in plot['graphs']] #+ threshold
  # if not plot['graphs'][0]['models'][0].continue_training: data += threshold #TODO: check for any graph/model
  figure = go.Figure(layout=layout( y=f'Mean {plot["metric"]}', x='Steps', legend=True, inset=len(data)<18), data=data)
  xmax = int(math.floor(max([g['data'][0].index[-1] for g in plot['graphs']])/10))*10
  ymax = 100;  dtick = 1
  if plot['metric'] == "Return": ymax = 1 ; dtick = 0.1
  if plot['metric'] == "Qubits": ymax = qb(plot['title'])
  if plot['metric'] == "Depth": ymax = dp(plot['title'])
  if ymax > 8: dtick = 2 
  if ymax > 16: dtick = 8
  figure.update_yaxes(range=[0, ymax], tickmode = 'linear', dtick=dtick) 
  figure.update_xaxes(range=[2048*4*4, 128*(2048*4)]) 
  
  # if plot['graphs'][0]['models'][0].continue_training: figure.update_xaxes(range=[0, xmax], tickmode = 'linear', dtick = 50000) 
  return {' '.join(title(plot)): figure}

def get_heatmap(compress=False, deterministic=False,flat=True):
  # def plot_heatmap(plot): return {f'Heatmaps/{plot["title"]}': 
  #     heatmap_3D(plot['data'], compress=compress, deterministic=deterministic, flat=flat)}
  # return plot_heatmap
  raise(NotImplementedError)

def color(graph, dim=0): 
#  TODO grey for random?
  if 'Random' in graph['label']: return 'hsva(0,0%,{}%,{:.2f})'.format(20+dim*60, 1.0-dim*0.8)
  hue = {
    'A2C':     40,   # Orange
    # '':  70,   # Yellow 
    'PPO':    200,   # Light Blue 
    # '': 230,   # Blue
    'SAC':    150,   # Green
    'TD3':    350,   # Red
  }[graph['label']]
  return 'hsva({},{}%,{}%,{:.2f})'.format(hue, 90-dim*20, 80+dim*20, 1.0-dim*0.8)

def layout(title=None, legend=True, wide=False, x='', y='', inset=False): 
  d,m,l = 'rgb(64, 64, 64)', 'rgba(64, 64, 64, 0.32)', 'rgba(64,64,64,0.04)'
  axis = lambda title: {'gridcolor': m, 'linecolor': d, 'title': title, 'mirror':True, 'ticks':'outside', 'showline':True, 'zeroline': True, 'zerolinecolor': m} 

  return go.Layout( title=title, showlegend=legend, font=dict(size=20),  
    legend={'yanchor':'top', 'y':0.935, 'xanchor':'left', 'x':0.01,'bgcolor':l,'bordercolor':d,'borderwidth':1} if inset else {},
    margin=dict(l=8, r=8, t=8+(72 * (title is not None)), b=8), width=600+200*wide+100*legend, height=400, 
    xaxis=axis(x), yaxis=axis(y), plot_bgcolor=l) #, paper_bgcolor='rgba(0,0,0,0)', 
   
def generate_figures(plots, generator): return { k:v for p in plots for k,v in generator[p['metric']](p).items()}
