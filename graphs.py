# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 11:49:19 2025

@author: jlegalla
"""

import plotly as ptl
import plotly.graph_objects as go
import plotly.io as pio
#pio.renderers.default = 'svg'

import numpy as np

pio.renderers.default = 'browser'


def plot(data_points):
    fig = go.Figure(go.Scatter(x=np.arange(len(data_points)), y=data_points))
    fig = go.Figure(data=[go.Scatter(x=np.arange(len(data_points)), y=data_points)])
    fig.show()

X1=(0,0.008,0.009,0.019,0.056,0.073,0.109,0.134,0.16,0.186,0.223,0.26,0.317,0.369,0.449,0.549,0.641,0.642,1)
Y1=(0,0,0.154,0.243,0.318,0.43,0.524,0.488,0.419,0.355,0.282,0.222,0.165,0.114,0.063,0.024,0.004,0,0)
X2=(0,0.087,0.088,0.117,0.191,0.263,0.317,0.355,0.356,1)
Y2=(0,0,0.007,0.266,0.206,0.131,0.063,0.016,0,0)
fig = go.Figure( layout_yaxis_range=[0,1], layout_xaxis_range=[0,1] ,layout=dict(legend={"x": 0.7,"y": 0.8,},))
fig.add_scatter(x=X1, y=Y1, mode='lines',name='EPS')
fig.add_scatter(x=X2, y=Y2, mode='lines',name='control') 
fig.update_xaxes(title_text='COST/LOSS RATIO')
fig.update_yaxes(title_text='VALUE')

