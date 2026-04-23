# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 11:49:19 2025

@author: JoPHOBEA
"""

import plotly.graph_objects as go
import plotly.io as pio
#pio.renderers.default = 'svg'

import numpy as np

pio.renderers.default = 'browser'


def plot(data_points):
    """
    Plots a simple line chart of the provided data points using Plotly.
    
    Args:
        data_points (array-like): Sequence of numerical values to be plotted on the y-axis. 
            The x-axis is automatically generated as a range from 0 to len(data_points)-1.
    
    Returns:
        None: This function displays the plot and does not return any value.
    """
    fig = go.Figure(go.Scatter(x=np.arange(len(data_points)), y=data_points))
    fig = go.Figure(data=[go.Scatter(x=np.arange(len(data_points)), y=data_points)])
    fig.show()

