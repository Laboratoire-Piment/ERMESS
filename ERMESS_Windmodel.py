# -*- coding:utf-8 -*-
'''
:Created: 2025-07-17 10:09:17
:Project: virtual PMS for microgrids
:Version: 1.0
:Author: Mathieu Lafitte
:Description: Tool to simulate the behavior of a single wind turbine.
- inputs : wind turbine parameters (power curves, height,...) + weather timeseries (pressure, windspeed, roughness)
- outputs : power output of the wind turbine (AC) over time (csv and/or plot)
There are mainly three steps. First you have to import your weather data, then
you need to specify your wind turbine, and in the last step call the
windpowerlib functions to calculate the feed-in time series.
This script is an adaptation of the ModelChain example of windpowerlib for ERMESS. Model is available here :
https://windpowerlib.readthedocs.io/en/stable/getting_started.html#examples-and-basic-usage
'''
#---------------------
# %%
import pandas as pd
from windpowerlib import ModelChain, WindTurbine

def default_wind_turbines(WT_prod):
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

def windmodel(weather: pd.DataFrame, MyTurbineDict: dict, ModelChainDict: dict=None,
              csv: bool=True, plot: bool=False, pow_TS: bool=True, pow_curv: bool=False) -> ModelChain:
    """
    1. Initializes a WindTurbine object.
    There are 2 different ways to do so. You can either :
        A - use turbine data from the OpenEnergy Database (oedb) turbine library
        that is provided along with the windpowerlib.
        B - specify your own turbine by directly providing a power (coefficient) curve.
    To get a list of all wind turbines for which power and/or power coefficient
    curves are provided, execute ``windpowerlib.wind_turbine.get_turbine_types()``

    2. Calculates power output of wind turbines using the modelchain.ModelChain class.
    modelchain.ModelChain is a class that provides all necessary steps to calculate
    the power output of a wind turbine. You can either use the default methods for
    the calculation steps, or choose different methods. Of course, you can also use
    the default methods while only changing one or two of them.

    3. Saves the power output and power curve to csv files and/or plots them.

    Args:

        MyTurbineDict (dict): references OR data of the WindTurbine. Choose between :
            A - ref_oedb = {"turbine_type": str,  # turbine type as in register
                            "hub_height": float,  # in m}
            B - your_power_curve = {
                    "nominal_power": 3e6,  # in W
                    "hub_height": 105,  # in m
                    "power_curve": pd.DataFrame(data={"value": [power list ], # in W
                                                      "wind_speed": [associated wind speed] # in m/s
                                    ,}),}
        ModelChainDict (dict, optional): dictionary with simulation parameters
            example = {
                "wind_speed_model": "logarithmic",  # 'logarithmic' (default), 'hellman' or 'interpolation_extrapolation'
                "density_model": "ideal_gas",  # 'barometric' (default), 'ideal_gas' or 'interpolation_extrapolation'
                "temperature_model": "linear_gradient",  # 'linear_gradient' (def.) or 'interpolation_extrapolation'
                "power_output_model": "power_coefficient_curve",  # 'power_curve' (default) or 'power_coefficient_curve'
                "density_correction": True,  # False (default) or True
                "obstacle_height": 0,  # default: 0
                "hellman_exp": None, # None (default) or None
            }
        csv (bool, optional): If True, saves the power output and power curve to csv files. Default is True.
        plot (bool, optional): If True, plots the power output and power curve. Default is False.
        pow_TS (bool, optional): If True, saves the power output time series to a csv file and plots it. Default is True.
        pow_curv (bool, optional): If True, saves the power curve to a csv file and plots it. Default is False.
    
    Returns:
        ModelChain: contains input parameters and results of the simulation.
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

    # Results saving
    # ---------------------------------------------------------------------------------------
    #MyTurbineDict['name'] = MyTurbineDict['name'].replace('/','-').replace('\\','-')
    # power output
    if csv and pow_TS:
        csv_name_TS = f"PowerOut_{MyTurbineDict['name']}.csv"
        print(csv_name_TS)
        df_TS = pd.DataFrame({'datetime':weather.index,'power (W)': my_turbine.power_output})
        df_TS.set_index('datetime').to_csv(csv_name_TS)
    if plt and plot and pow_TS:
        my_turbine.power_output.plot(legend=True, label="myTurbine")
        plt.xlabel("Time")
        plt.ylabel("Power in W")
        plt.show()

    # power curve
    if csv and pow_curv:
        csv_name_curv = f"PowerCurve_{MyTurbineDict['name']}.csv"
        print(csv_name_curv)
        my_turbine.power_curve.set_index('wind_speed').to_csv(csv_name_curv)
    if plt and plot and pow_curv:
        if my_turbine.power_curve is not False:
            my_turbine.power_curve.plot(
                x="wind_speed",
                y="value",
                style="*",
                title="myTurbine power curve",
            )
            plt.xlabel("Wind speed in m/s")
            plt.ylabel("Power in W")
            plt.show()
    
    return my_ModelChain.power_output.values

