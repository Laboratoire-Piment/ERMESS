# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 13:42:26 2026

@author: JoPHOBEA
"""

import numpy as np
import openpyxl
import copy  

def timeseries_interpolation(datetime_model,series_datetime,series_yvalue):
    """
    Interpolate a time series to match a target datetime grid.
    
    Args:
        datetime_model (array_like): Target datetime array (numeric or float representation) for interpolation.
        series_datetime (array_like): Original datetime array corresponding to the input series_yvalue.
        series_yvalue (array_like): Original series values to interpolate.
    
    Returns:
        np.ndarray: Interpolated series aligned with datetime_model.
    """
    y_values = np.float64(np.interp(np.array(datetime_model,dtype='float64'),np.array(series_datetime,dtype='float64'),np.array(series_yvalue,dtype='float64')))
    return(y_values)


