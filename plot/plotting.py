import math; import plotly.graph_objects as go

title = lambda plot, y=None: [y or '' if 'merge' in plot.keys() else plot["metric"], plot["title"]]

# Helper functions to create scatters/graphs from experiment & metric
def plot_ci(plot):  
  traceorder = lambda key: [i for k,i in {'Radius': 0, 'Action': 1, 'Object': 2, 'RAD': 3, 'PPO': 4, 'A2C': 5, '': 6}.items() if k in key][0]
  plot['graphs'].sort(key=lambda g: traceorder(g['label'])) # Sort traces  
  smooth = {'shape':  'spline',  'smoothing': 0.4}
  scatter = lambda data, **kwargs: go.Scatter(x=data.index, y=data, **kwargs)
  dash = lambda g: {'dash': 'dash'} if 'Evaluation' in plot['metric'] else {}
  getmean = lambda g: scatter(g['data'][0], name=g['label'], mode='lines', line={'color': color(g['hue']), **smooth,  **dash(g)})
  getconf = lambda g: scatter(g['data'][1], fillcolor=color(g['hue'], 1), fill='toself', line={'color': 'rgba(255,255,255,0)', **smooth}, showlegend=False)
  # threshold = [go.Scatter(y=[plot['graphs'][0]['data'][2][1]]*2, x=[0,max([g['data'][0].tail(1).index[0] for g in plot['graphs']])],
  #   name='Solved', mode='lines', line={'dash':'dot', 'color':'rgb(64, 64, 64)'})] #Threshold
  data = [getconf(g) for g in plot['graphs']] + [getmean(g) for g in plot['graphs']] #+ threshold
  # if not plot['graphs'][0]['models'][0].continue_training: data += threshold #TODO: check for any graph/model
  figure = go.Figure(layout=layout( y=f'Mean {plot["metric"]}', x='Steps', legend=True, inset=len(data)<18), data=data)
  xmax = int(math.floor(max([g['data'][0].index[-1] for g in plot['graphs']])/10))*10
  figure.update_yaxes(range=[0, 1], tickmode = 'linear', dtick = 0.2) 
  # if plot['graphs'][0]['models'][0].continue_training: figure.update_xaxes(range=[0, xmax], tickmode = 'linear', dtick = 50000) 
  return {' '.join(title(plot)): figure}

def get_heatmap(compress=False, deterministic=False,flat=True):
  # def plot_heatmap(plot): return {f'Heatmaps/{plot["title"]}': 
  #     heatmap_3D(plot['data'], compress=compress, deterministic=deterministic, flat=flat)}
  # return plot_heatmap
  raise(NotImplementedError)

def color(hue, dim=0): return 'hsva({},{}%,{}%,{:.2f})'.format(hue, 90-dim*20, 80+dim*20, 1.0-dim*0.8)

def layout(title=None, legend=True, wide=False, x='', y='', inset=False): 
  d,m,l = 'rgb(64, 64, 64)', 'rgba(64, 64, 64, 0.32)', 'rgba(64,64,64,0.04)'
  axis = lambda title: {'gridcolor': m, 'linecolor': d, 'title': title, 'mirror':True, 'ticks':'outside', 'showline':True, 'zeroline': True, 'zerolinecolor': m} 

  return go.Layout( title=title, showlegend=legend, font=dict(size=20),  
    legend={'yanchor':'top', 'y':0.935, 'xanchor':'left', 'x':0.01,'bgcolor':l,'bordercolor':d,'borderwidth':1} if inset else {},
    margin=dict(l=8, r=8, t=8+(72 * (title is not None)), b=8), width=600+200*wide+100*legend, height=400, 
    xaxis=axis(x), yaxis=axis(y), plot_bgcolor=l) #, paper_bgcolor='rgba(0,0,0,0)', 
   
def generate_figures(plots, generator): return { k:v for p in plots for k,v in generator[p['metric']](p).items()}
