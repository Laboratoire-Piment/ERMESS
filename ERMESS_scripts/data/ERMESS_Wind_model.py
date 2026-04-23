# -*- coding:utf-8 -*-
'''
:Created: 2025-07-17 10:09:17
:Author: Mathieu Lafitte and JoPHOBEA
:Description: Tool to simulate the behavior of a single wind turbine.

This script is an adaptation of the ModelChain example of windpowerlib for ERMESS. Model is available here :
https://windpowerlib.readthedocs.io/en/stable/getting_started.html#examples-and-basic-usage
'''
#---------------------
# %%
import pandas as pd
from windpowerlib import ModelChain, WindTurbine

def default_wind_turbines(WT_prod):
    """
    Generate a default wind turbine dictionary for ERMESS.
    
    This function returns a turbine parameter dictionary compatible with
    ``windpowerlib.WindTurbine`` used by ERMESS. It provides predefined default power curves
    for several small and medium wind turbines typically used in micro-grid
    optimisation studies within the ERMESS evolutionary sizing framework.
    
    If the turbine type is not recognized as a predefined default model,
    the function converts the input structure into a dictionary format
    compatible with ``WindTurbine``.
    
    Args:
        WT_prod (pandas.Series or dict-like): Wind turbine configuration data. 
            Must contain at least the keys ``'turbine_type'`` and ``'hub_height'``.
    
    Returns:
        dict: Dictionary containing wind turbine parameters formatted for ``windpowerlib.WindTurbine``.
            Structure example:
                {   "nominal_power": float,  # in W
                    "hub_height": float,     # in m
                    "power_curve": pandas.DataFrame   }
    
    Raises:
        KeyError: If required keys (e.g. ``'turbine_type'`` or ``'hub_height'``) are missing.
    
    Note:
        The predefined turbines are simplified generic models (1 kW, 5 kW,
        12 kW, 60 kW, 100 kW) intended for optimisation benchmarking rather
        than manufacturer-certified performance assessment.
    
    Warning:
        Power curves are hard-coded and may not reflect real turbine
        aerodynamics. Use manufacturer data for high-fidelity simulations.
    """
    
    if (WT_prod['turbine_type']=='1 kW default'):
        turbineDict = {#SWIFT_1kW_2
                "nominal_power": 1000,"hub_height": WT_prod['hub_height'],
                "power_curve": pd.DataFrame(data={"value": [0,0,20,50,80,130,180,250,330,430,530,650,770,900,1020,1140,1250,1330,1390,1430,1410,1340,1210,1090,970,760,650,540,470,400,400,350,370,320,370,360,380,370,400,400], # in W
                "wind_speed": list(x/2 for x in range(9,49,1)) # in m/s
                                ,}),}
    elif (WT_prod['turbine_type']=='5 kW default'):
        turbineDict = {#SD6_5.2kW_5
                "nominal_power": 5000,"hub_height": WT_prod['hub_height'],
                "power_curve": pd.DataFrame(data={"value": [0,12,45,96,172,296,485,739,1020,1326,1733,2165,2674,3121,3546,4033,4428,4870,5164,5464,5626,5851,5960,6000,6026,6064,6059,6119,6099], # in W
                "wind_speed": list(x/2 for x in range(4,33,1)) # in m/s
                                ,}),}
    elif (WT_prod['turbine_type']=='12 kW default'):
        turbineDict = {#Jacobs31-20_12kW_9
                "nominal_power": 12000,"hub_height": WT_prod['hub_height'],
                "power_curve": pd.DataFrame(data={"value": [0,30,170,350,660,1070,1600,2190,2970,3840,4630,5580,6670,7910,9190,10620,12000,13140,14480,15720,16470,17520,18280,18620,19060,19200,19540,19580,19780,19910,20000,20030,19660,19980,19840,20120,20180], # in W
                "wind_speed": list(x/2 for x in range(6,43,1)) # in m/s
                                ,}),}
    elif (WT_prod['turbine_type']=='60 kW default'):
        turbineDict = {#NPS60C-24_60kW_24.4
                "nominal_power": 60000,"hub_height": WT_prod['hub_height'],
                "power_curve": pd.DataFrame(data={"value": [0,1700,6000,13000,24500,38000,52500,58700,59700,59900,59900,59900,59900,59900,57500,55000,52000,49100,46800,45000,43000,42000,41000,40500], # in W
                "wind_speed": list(x for x in range(2,26,1)) # in m/s
                                ,}),}
    elif (WT_prod['turbine_type']=='100 kW default'):
        turbineDict = {
                "nominal_power": 100000,"hub_height": WT_prod['hub_height'],
                "power_curve": pd.DataFrame(data={"value": [0,0,500,4100,10500,19000,29400,41000,54300,66800,77700,86400,92800,97800,100000,99900,99200,98400,97500,96800,96400,96300,96800,98000,99200], # in W
                "wind_speed": list(x for x in range(1,26,1)) # in m/s
                                ,}),}
    else :
        turbineDict=WT_prod.T.to_dict()
        
    return (turbineDict)

def windmodel(weather: pd.DataFrame, MyTurbineDict: dict, ModelChainDict: dict=None) -> ModelChain:
    """
    Simulates wind turbine power production using windpowerlib within ERMESS.
    
    This function initializes a ``windpowerlib.WindTurbine`` object from a
    turbine definition dictionary, executes a ``ModelChain`` simulation using
    provided weather data.
    
    It is used inside the ERMESS evolutionary optimisation loop to evaluate
    candidate wind turbines for micro-grid sizing.
    
    Args:
        weather (pandas.DataFrame): Time-indexed weather data required by windpowerlib (e.g. wind speed, temperature, pressure).
    
        MyTurbineDict (dict): Wind turbine definition. Two formats are supported:
            Reference to OpenEnergy Database (OEDB) turbine OR Custom turbine definition
    
        ModelChainDict (dict, optional): Optional dictionary specifying ModelChain
            calculation models (wind speed, density, temperature, power output, etc.).
            
    Returns:
        numpy.ndarray: Simulated wind power output time series (W), aligned with the
            input weather index.
    
    Raises:
        ValueError: If weather data is incomplete or incompatible with
            windpowerlib requirements.
    
        KeyError: If required turbine parameters are missing.
    
    Notes:
        The function internally assigns the simulated power output to
        ``my_turbine.power_output`` for consistency with windpowerlib
        conventions.
    """
    # WindTurbine object initialization
    # ---------------------------------------------------------------------------------------
    my_turbine = WindTurbine(**MyTurbineDict)

    # Simulation
    # ---------------------------------------------------------------------------------------
    if not ModelChainDict == None:
        my_ModelChain = ModelChain(my_turbine, **ModelChainDict).run_model(weather)
    else:
        my_ModelChain = ModelChain(my_turbine).run_model(weather)
    my_turbine.power_output = my_ModelChain.power_output
   
    return my_ModelChain.power_output.values

