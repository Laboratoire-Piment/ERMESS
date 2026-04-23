# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 13:40:24 2026

@author: jlegalla
"""

import numpy as np
import pandas as pd
import copy

from ERMESS_scripts.data.indices import *

def find_constraint_levels(Context):
    """
    Compute the maximum achievable constraint level based on production, storage, and load.
    
    This function calculates the maximum self-sufficiency or self-consumption level
    given production capacities, storage characteristics, and non-movable/movable loads.
    
    Args:
        constraint_num (int): Number identifying the type of constraint
            (1 = Self-sufficiency, 2 = Self-consumption, etc.).
        Constraint_level (float): Target level of the constraint.
        D_Non_movable_load (array-like): Non-movable load time series (kW).
        D_Movable_load (array-like): Movable load time series (kW).
        storage_characteristics (np.ndarray): Matrix with storage attributes.
        prod_C (float): Current production at the site (W).
        prods_U (np.ndarray): Unit production matrix (kW).
        Bounds_prod (np.ndarray): Maximum capacity of production units (kW).
    
    Returns:
        tuple[int, float, float]: Tuple of (constraint_num, Constraint_level, D_max_constraint_level).
    """
    KILOS_CONVERSION_FACTOR = 1000
    max_production = (Context.production.current_prod+np.inner(Context.production.unit_prods.T,Context.production.capacities))/KILOS_CONVERSION_FACTOR 
    total_load=sum(Context.loads.non_movable)+sum(Context.loads.Y_movable)+sum(Context.loads.D_movable)        
    
    if Context.optimization.constraint_num == CONS_Self_sufficiency :
        extra_prod = np.where(max_production>Context.loads.non_movable,max_production-Context.loads.non_movable,0) 
        sum_extra_prod = sum(extra_prod)-sum(Context.loads.non_movable)        
        Instantaneous_energy=np.minimum(max_production,Context.loads.non_movable)
        reportable_energy=max(Context.storage.characteristics[STOR_ROUND_TRIP_EFF,:])*sum_extra_prod
        max_self_sufficiency = min(1,(sum(Instantaneous_energy)+reportable_energy)/total_load)
        max_constraint_level = max_self_sufficiency
    elif Context.optimization.constraint_num == CONS_SELF_CONSUMPTION :
        ## The max. self-consumption is always 1 because we can always lose artificially energy using storage
        max_self_consumption = 1
        max_constraint_level = max_self_consumption
    elif Context.optimization.constraint_num == CONS_REN_FRACTION :
        max_REN_fraction = min(1,sum(max_production)/total_load)
        max_constraint_level = max_REN_fraction
    return(max_constraint_level)

def compute_grid_prices(datetime_data,grid_price):
    """
    Generate detailed grid price series for each contract and hour type.
    
    This function maps grid price contracts to hourly time series, taking into account
    weekdays, weekends, peak/off-peak hours, seasonal differences, taxes, and premiums.
    
    Args:
        datetime_data (array-like): Array of datetime values for the simulation horizon.
        grid_price (pd.DataFrame): Contract-specific price structures, seasonal hours, and premiums.
    
    Returns:
        list: List containing:
            - prices_hour_type (np.ndarray): String array of shape (n_time, n_contracts) indicating the hour type for each timestep and contract.
            - prices_num (np.ndarray): Numeric price values (€/kWh) of shape (n_time, n_contracts) for each timestep and contract.
            - fixed_premium (np.ndarray): Array of fixed premiums per contract (€/kW).
            - Overrun (np.ndarray): Array of overrun costs per contract (€/kW).
            - Selling_price (np.ndarray): Array of selling prices per timestep and contract (€/kWh).
    """
    grid_price=grid_price.replace(np.nan, '', regex=True)
    prices = [pd.DataFrame(data={'Datetime':datetime_data,'Hour type':'NONE','Price':None}) for i in range(len(grid_price))]
    prices_hour_type = []
    prices_num = []


    fixed_premium = []
    Overrun = []
    Selling_price = []
    
    for i in range(len(grid_price)):
        prices[i].loc[(prices[i]['Datetime'].apply(lambda x: x.dayofweek).isin([5,6])) & (prices[i]['Datetime'].apply(lambda x: str(x.hour)).isin(grid_price.iloc[i]['Weekend peak hours'].split(' '))),'Hour type']='Peak'
        prices[i].loc[(prices[i]['Datetime'].apply(lambda x: x.dayofweek).isin([5,6])) & (prices[i]['Datetime'].apply(lambda x: str(x.hour)).isin(grid_price.iloc[i]['Weekend full hours'].split(' '))),'Hour type']='WE full'
        prices[i].loc[(prices[i]['Datetime'].apply(lambda x: x.dayofweek).isin([5,6])) & (prices[i]['Datetime'].apply(lambda x: str(x.hour)).isin(grid_price.iloc[i]['Weekend off-peak hours'].split(' '))),'Hour type']='WE off'
        prices[i].loc[(prices[i]['Datetime'].apply(lambda x: x.dayofweek).isin([0,1,2,3,4])) & (prices[i]['Datetime'].apply(lambda x: str(x.hour)).isin(grid_price.iloc[i]['Workday peak hours'].split(' '))),'Hour type']='Peak'
        prices[i].loc[(prices[i]['Datetime'].apply(lambda x: x.dayofweek).isin([0,1,2,3,4])) & (prices[i]['Datetime'].apply(lambda x: str(x.hour)).isin(grid_price.iloc[i]['Workday full hours'].split(' '))),'Hour type']='W full'
        prices[i].loc[(prices[i]['Datetime'].apply(lambda x: x.dayofweek).isin([0,1,2,3,4])) & (prices[i]['Datetime'].apply(lambda x: str(x.hour)).isin(grid_price.iloc[i]['Workday off-peak hours'].split(' '))),'Hour type']='W off'
    
        prices[i].loc[prices[i]['Hour type']=='Peak' ,'Price']=grid_price.iloc[i]['Peak (c€/kWh)']
        prices[i].loc[(prices[i]['Hour type']=='WE full') & (prices[i]['Datetime'].apply(lambda x: str(x.month)).isin(grid_price.iloc[i]['Summer months'].split(' '))) ,'Price']=grid_price.iloc[i]['Summer full hours (c€/kWh)']
        prices[i].loc[(prices[i]['Hour type']=='W full') & (prices[i]['Datetime'].apply(lambda x: str(x.month)).isin(grid_price.iloc[i]['Summer months'].split(' '))) ,'Price']=grid_price.iloc[i]['Summer full hours (c€/kWh)']
        prices[i].loc[(prices[i]['Hour type']=='WE off') & (prices[i]['Datetime'].apply(lambda x: str(x.month)).isin(grid_price.iloc[i]['Summer months'].split(' '))) ,'Price']=grid_price.iloc[i]['Summer off-peak hours (c€/kWh)']
        prices[i].loc[(prices[i]['Hour type']=='W off') & (prices[i]['Datetime'].apply(lambda x: str(x.month)).isin(grid_price.iloc[i]['Summer months'].split(' '))) ,'Price']=grid_price.iloc[i]['Summer off-peak hours (c€/kWh)']
        prices[i].loc[(prices[i]['Hour type']=='WE full') & (prices[i]['Datetime'].apply(lambda x: str(x.month)).isin(grid_price.iloc[i]['Winter months'].split(' '))) ,'Price']=grid_price.iloc[i]['Winter full hours (c€/kWh)']
        prices[i].loc[(prices[i]['Hour type']=='W full') & (prices[i]['Datetime'].apply(lambda x: str(x.month)).isin(grid_price.iloc[i]['Winter months'].split(' '))) ,'Price']=grid_price.iloc[i]['Winter full hours (c€/kWh)']
        prices[i].loc[(prices[i]['Hour type']=='WE off') & (prices[i]['Datetime'].apply(lambda x: str(x.month)).isin(grid_price.iloc[i]['Winter months'].split(' '))) ,'Price']=grid_price.iloc[i]['Winter off-peak hours (c€/kWh)']
        prices[i].loc[(prices[i]['Hour type']=='W off') & (prices[i]['Datetime'].apply(lambda x: str(x.month)).isin(grid_price.iloc[i]['Winter months'].split(' '))) ,'Price']=grid_price.iloc[i]['Winter off-peak hours (c€/kWh)']

        prices[i]['Price']=prices[i]['Price']*(1+grid_price.iloc[i]['TVA load'])
        prices[i]['Price']+=grid_price.iloc[i]['CSPE (c€/kWh)']
        prices[i]['Price']=prices[i]['Price']/100
        prices_hour_type.append(prices[i]['Hour type'])
        prices_num.append(prices[i]['Price'])

        fixed_premium.append( grid_price.iloc[i]['Fixed premium (€/kW)'])
        Overrun.append( grid_price.iloc[i]['Power overrun (€/kW)'])
        Selling_price.append(np.repeat(grid_price.iloc[i]['Selling base price (c€/kWh)'],len(datetime_data)))
        Selling_price[i][(prices[i]['Datetime'].apply(lambda x: str(x.hour)).isin(grid_price.iloc[i]['Selling peak hours'].split(' ')))]=grid_price.iloc[i]['Selling peak price (c€/kWh)']
        Selling_price[i]=Selling_price[i]/100
        
        #Converting into Numpy
    fixed_premium=np.array(fixed_premium,dtype=np.float64)
    Overrun=np.array(Overrun,dtype=np.float64)
    Selling_price=np.array(Selling_price,dtype=np.float64)
    prices_num=np.float64(np.vstack(prices_num))
    prices_hour_type=np.vstack(prices_hour_type).astype('U')

    return([prices_hour_type,prices_num,fixed_premium,Overrun,Selling_price])

def adaptation_hyperparameters_initialisation(operators_parameters):
    """
    Automatically adjusts the values of the hyperparameters matrix in an initialization context (pre-optimization).
    
    Args:
        operators_parameters (np.array): Initial matrix.
    
    Returns:
        np.array: adaptated_hyperparameters_operators, the updated matrix.
    """
    INIT_RESEARCH_STORAGE_PATTERNS = 5
    INIT_RESEARCH_STORAGE_MIX = 1
    INIT_RESEARCH_CONSTRAINT_FORCING = 20
    INIT_RESEARCH_STORAGE_POWER_FACTOR = 0.2
    INIT_RESEARCH_CURVE_SMOOTHING_FACTOR = 0.7
    adaptated_operators_parameters=copy.deepcopy(operators_parameters)
    adaptated_operators_parameters[OPER_PROBABILITY,RESEARCH_STORAGE_POWER]=INIT_RESEARCH_STORAGE_POWER_FACTOR*operators_parameters[OPER_PROBABILITY,RESEARCH_STORAGE_POWER]
    adaptated_operators_parameters[OPER_INV_LENGTH,RESEARCH_STORAGE_PATTERNS]=INIT_RESEARCH_STORAGE_PATTERNS
    adaptated_operators_parameters[OPER_INV_LENGTH,RESEARCH_STORAGE_MIX]=INIT_RESEARCH_STORAGE_MIX
    adaptated_operators_parameters[OPER_INV_LENGTH,RESEARCH_CONSTRAINT_FORCING]=INIT_RESEARCH_CONSTRAINT_FORCING
    adaptated_operators_parameters[OPER_PROBABILITY,RESEARCH_CURVE_SMOOTHING]=INIT_RESEARCH_CURVE_SMOOTHING_FACTOR*operators_parameters[OPER_PROBABILITY,RESEARCH_CURVE_SMOOTHING]

    return (adaptated_operators_parameters)