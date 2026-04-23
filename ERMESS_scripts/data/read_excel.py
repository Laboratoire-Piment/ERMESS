# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 13:44:54 2026

@author: JoPHOBEA
"""

import numpy as np
import pandas as pd
import timezonefinder

from . import data_parsers as Edp
from . import ERMESS_meteo as Eme
from . import ERMESS_PV_model as EPV
from . import ERMESS_Wind_model as EWi

from ERMESS_scripts.utils.constraints import compute_grid_prices
from ERMESS_scripts.utils.helpers import timeseries_interpolation

def load_excel(path):
    """Load ERMESS input Excel file."""
    xl = pd.ExcelFile(path)
    data = {sheet: xl.parse(sheet) for sheet in xl.sheet_names}
    xl.close()
    return data


def read_data(Data) :
    """
    Read, preprocess, and structure input data for the ERMESS energy optimization model.

    This function handles:
    - Meteorological data import and interpolation for PV and wind production
    - Production unit initialization (manual or automatic)
    - Load time series processing
    - Grid prices, contracts, and constraints
    - Storage and dispatching characteristics
    - Hyperparameters for GA and professional GA

    :param : Data : Nested dictionary or pandas DataFrame structure containing:
    - Environment: site info, constraints, optimization criteria, meteo/production settings
    - TimeSeries: loads, production
    - PV_production_specs, WT_production_specs: unit specs
    - Storages: storage characteristics
    - Grid_prices: price contracts
    - Diesel generator and Dispatching information
    - Hyperparameters and Hyperparameters_pro for GA evolution
    :rtype dict
        

    :returns:
        A tuple containing all processed arrays, matrices, and parameters needed for ERMESS GA:
        (datetime_model, Grid_Fossil_fuel_ratio, Main_grid_emissions, Main_grid_ratio, Connexion,
        specs_names, specs_Id, specs_num, prod_C, prods_U, Volums_prod, Y_movable_load,
        D_movable_load, Non_movable_load, time_resolution, constraint_num, Constraint_level,
        criterion_num, storage_techs, storage_characteristics_names, storage_characteristics,
        Bounds_prod, n_UP, duration_years, prices_hour_type, prices_num, fixed_premium, Overrun,
        Selling_price, Contract_Id, hyperparameters_main, hyperparameters_operators_names,
        hyperparameters_operators_num, hyperparameters_main_pro, hyperparameters_operators_names_pro,
        hyperparameters_operators_num_pro, Cost_sequence, type_optim, Dispatching,
        DG_fuel_cost, DG_lifetime, DG_unit_cost, DG_maintenance_cost, DG_fuel_consumption,
        DG_EROI, fuel_CO2eq_emissions, groups)
    :rtype:  tuple
    """
    
    EnR_site = {"longitude":Data['Environment']['Longitude (°)'][0],"latitude":Data['Environment']['Latitude (°)'][0],"altitude":Data['Environment']['Altitude (m)'][0]}
    time_resolution = np.float64( Data['Environment']['time resolution (steps/h)'][0])

    # Meteo data
    datetime_data=Data['TimeSeries']['Datetime']
    datetime_data=pd.to_datetime(datetime_data, format="%d/%m/%Y %H:%M").round('1s')
    
    timezone_str = timezonefinder.TimezoneFinder().certain_timezone_at(lat=EnR_site['latitude'], lng=EnR_site['longitude'])
    EnR_site ["tz"]=timezone_str

    datetime_model = pd.date_range(datetime_data[0],datetime_data[len(datetime_data)-1],freq=str(int(60/time_resolution))+'min',tz=timezone_str)#.tz_localize(timezone_str)
    
    Meteo_computation = Data['Environment']['Meteo'][0]
    
    
    Production_computation = Data['Environment']['Production'][0]
    
    if (Production_computation=='automatic'):
        
        if (Meteo_computation=='manual'):
        
            PV_meteo = Data['Meteo'].loc[:, ['GHI (W/m²)','DNI (W/m²)','DHI (W/m²)','Air temperature (°C)','Wind speed (m/s)']].rename(columns={'GHI (W/m²)':'ghi','DNI (W/m²)':'dni','DHI (W/m²)':'dhi','Air temperature (°C)':'temp_air','Wind speed (m/s)':'wind_speed'}).set_index(pd.to_datetime(Data['Meteo'].loc[:,'Datetime']))
            Wind_heights = Data['Meteo'].loc[np.array((5,3,6)),'Measurement height (m)'].to_numpy()
            Wind_meteo = pd.DataFrame(data=Data['Meteo'].loc[:, ['Wind speed (m/s)','Air temperature (°C)','Pressure (hPa)']].values,index=Data['Meteo']['Datetime'],columns=[np.array(['wind_speed','temperature','pressure']),Wind_heights])
            Wind_meteo['temperature']=Wind_meteo['temperature']+273.15
            Wind_meteo['pressure']=Wind_meteo['pressure']*100 #in Pa
            Datetime_prods = pd.DatetimeIndex(pd.to_datetime(Data['Meteo']['Datetime'],format='%d/%m/%Y %H:%M')).round('1s').tz_localize(timezone_str,nonexistent="shift_forward" ,ambiguous=False)
            
        elif (Meteo_computation == 'automatic' ):
            PV_meteo,Wind_meteo = Eme.import_meteo(EnR_site['latitude'],EnR_site['longitude'],EnR_site['altitude'],datetime_model[0],datetime_model[len(datetime_model)-1],timezone_str)
            Datetime_prods =PV_meteo.index

        Terrain_type =  Data['Environment']['Terrain type'][0]
        Roughness_length = 0 if Terrain_type=='off-shore' else (0.005 if Terrain_type=='no vegetation, no obstacles' else (0.005 if Terrain_type=='off-shore' else (0.03 if Terrain_type=='flat terrain, grass, isloated obstacles' else (0.1 if Terrain_type=='low crops, occasional obstacles' else (0.25 if Terrain_type=='high crops, scattered obstacles' else (0.5 if Terrain_type=='parkland, bushes, numerous obstacles' else (1 if Terrain_type=='regular large obstacles (suburbs, forest)' else (2.5 if Terrain_type=='city centre' else 0.5))))))))
        Wind_meteo[('roughness_length', 0)] = Roughness_length
        
        PV_prod = Data['PV_production_specs'].loc[:, ['Tilt (°)','Azimuth (°)','Module type','Mounting','Inverter','Module','Modules per string','Strings','Surface type']]
        WT_prod = Data['WT_production_specs'].loc[:, ['Model','Hub height (m)']].rename(columns={'Hub height (m)':'hub_height','Model':'turbine_type'})
        WT_prod = [EWi.default_wind_turbines(WT_prod.iloc[i]) for i in range(len(WT_prod))]
        
        PV_TempParam_default = {'a': -3.47,'b': -0.0594,'deltaT': 3} #SAPM coeffs
        PV_FixModMount = [{"type":"Fixed","surface_tilt":PV_prod['Tilt (°)'][i],"surface_azimuth":PV_prod['Azimuth (°)'][i],"racking_model":PV_prod['Mounting'][i]} for i in range(len(PV_prod))]
        PV_arrayParam = [{"surface_type":PV_prod['Surface type'][i],"module_type":PV_prod['Module type'][i],"modules_per_string":PV_prod['Modules per string'][i],"strings":PV_prod['Strings'][i]} for i in range(len(PV_prod))]
        LossesParam_default = {"soiling": 2,"shading": 3,"snow": 0,"mismatch": 2,"wiring": 2,"connections": 0.5,"lid": 1.5,"nameplate_rating": 1,"age": 0,"availability": 3}
        
        prods_U_PV = np.array([EPV.pvmodel(EnR_site, PV_meteo, PV_TempParam_default, PV_prod['Module'][i], PV_prod['Inverter'][i], PV_FixModMount[i], PV_arrayParam[i],LossesParam_default,False,False,False,False)[0] for i in range(len(PV_prod))])
        prods_U_PV = np.where(prods_U_PV<0,0,prods_U_PV)
        prods_U_WT = np.array([EWi.windmodel(weather=Wind_meteo,MyTurbineDict=WT_prod[i],ModelChainDict=None) for i in range(len(WT_prod))])
        prods_U = np.row_stack([x for x in (prods_U_PV, prods_U_WT) if x.size] or [np.array([])])
    elif (Production_computation=='manual'):   
        prods_U = Data['Unit_productions'].drop('Datetime',axis=1)
        prods_U=np.array(prods_U).T
        Datetime_prods = Data['Unit_productions']['Datetime']
        
    prods_U=np.array([timeseries_interpolation(datetime_model,  Datetime_prods, prods_U[i]) for i in range(len(prods_U))])
    Volums_prod = np.sum(prods_U,axis=1)
    
    
    PV_specs = Data['PV_production_specs'].loc[:,['Capital unit cost (€)','Operational unit cost (€/yrs)','Lifetime (years)','Capacity','eqCO2 Emissions (gCO2/kWh)','EROI','Surface group']].to_numpy(dtype=np.float64)
    WT_specs = Data['WT_production_specs'].loc[:,['Capital unit cost (€)','Operational unit cost (€/yrs)','Lifetime (years)','Capacity','eqCO2 Emissions (gCO2/kWh)','EROI','Surface group']].to_numpy(dtype=np.float64)    
    specs_num = np.row_stack((PV_specs,WT_specs))
    
    groups = [np.where(specs_num[:,6]==i)[0] for i in np.unique(specs_num[:,6])]  
    
    specs_Id=np.concatenate((Data['PV_production_specs']['Id'].to_numpy(dtype='U'),Data['WT_production_specs']['Id'].to_numpy(dtype='U')))
    specs_names=np.array(['Capital unit cost','Operational unit cost','Lifetime','Capacity','eqCO2 Emissions','EROI','Surface group'],dtype='U')
    
    series_datetime = pd.to_datetime(Data['TimeSeries']['Datetime'], format="%d/%m/%Y %H:%M").dt.tz_localize(timezone_str,nonexistent="shift_forward" ,ambiguous=False)
    prod_C = timeseries_interpolation(datetime_model, series_datetime, Data['TimeSeries']['Current_production (W)'])
    prod_C[prod_C<0]=0
    
    Non_movable_load = timeseries_interpolation(datetime_model, series_datetime, Data['TimeSeries']['Non-controllable load (kW)'])
    Y_movable_load = timeseries_interpolation(datetime_model, series_datetime, Data['TimeSeries']['Yearly movable load (kW)'])
    D_movable_load = timeseries_interpolation(datetime_model, series_datetime, Data['TimeSeries']['Daily movable load (kW)'])
      
    grid_price = Data['Grid_prices']
    Contract_Id = grid_price['Contract_Id']

    #time_resolution = (datetime_data.iloc[-1].round(freq='min')-datetime_data.iloc[1].round(freq='min'))/(len(datetime_data)-2)
    #time_resolution = np.float64(3600/time_resolution.seconds)
    Constraint = Data['Environment']['Constraint'][0]
    Connexion = Data['Environment']['Connexion'][0]
        
    if Constraint== 'Self-sufficiency':
            constraint_num = np.int64(1) 
    elif Constraint== 'Self-consumption':
            constraint_num = np.int64(2)
    elif Constraint== 'EnR fraction':
            constraint_num = np.int64(3)
    else : 
            print('No proper constraint found !')
            constraint_num = np.int64(0)
                
    Constraint_level = np.float64(Data['Environment']['Constraint level'][0])
    criterion = Data['Environment']['Optimisation criterion'][0]
       
    if criterion== 'LCOE':
            criterion_num = np.int64(1)
    elif criterion== 'Annual net benefits':
            criterion_num = np.int64(2)
    elif criterion== 'NPV':
            criterion_num = np.int64(3)
    elif criterion== 'Self-sufficiency':
            criterion_num = np.int64(4)
    elif criterion== 'Self-consumption':
            criterion_num = np.int64(5)
    elif criterion== 'Autonomy':
            criterion_num = np.int64(6)
    elif criterion== 'eqCO2 Emissions':
            criterion_num = np.int64(7)
    elif criterion== 'fossil fuel consumption':
            criterion_num = np.int64(8)
    elif criterion== 'EROI':
            criterion_num = np.int64(9)
    elif criterion== 'Energy losses':
            criterion_num = np.int64(10)
    elif criterion== 'Maximum power from grid':
            criterion_num = np.int64(11)   
    else : 
            criterion_num = np.int64(0)
            
    type_optim = Data['Environment']['type'][0]
    
    storages = Data['Storages']
    Cost_power = storages['PCS cost (€/kW)']+storages['BOP cost (€/kW)']
    storage_characteristics = storages.T[1:].to_numpy(dtype=np.float64)
    storage_techs = storages['Technology'].to_numpy(dtype='U')
    storage_characteristics = np.vstack((storage_characteristics,Cost_power),dtype=np.float64)
    storage_characteristics_names = np.hstack((storages.columns[1:].to_numpy(),'Cost power'))

    Bounds_prod = np.int64(specs_num[:,np.where(specs_names=='Capacity')[0][0]])
    n_UP=len(Bounds_prod)
    duration_years = np.float64(len(prod_C)/time_resolution/8760)
    
    DG_fuel_cost = Data['Diesel generator']['Value'][0]
    DG_lifetime = Data['Diesel generator']['Value'][1]
    DG_unit_cost = Data['Diesel generator']['Value'][2]
    DG_maintenance_cost = Data['Diesel generator']['Value'][3]
    fuel_CO2eq_emissions = Data['Diesel generator']['Value'][4]
    DG_EROI = Data['Diesel generator']['Value'][5]
    DG_fuel_consumption = np.array(Data['Diesel generator']['DG fuel consumption'][1:11],dtype=np.float64)
    
    
    if (type_optim=='pro'):
        Defined_items = np.array(('Discharge order','D_DSM_levels','Y_DSM_levels','Diesel generator'))[np.where(Data['Dispatching']['User-Defined']=='Yes')[0]]
        Discharge_order = (np.array(Data['Dispatching']['Discharge order'],dtype=np.int64)[~pd.isnull(np.array(Data['Dispatching']['Storages']))])-1
        Taking_over = np.array(Data['Dispatching']['Taking over level (%)'][0:9],dtype='float64')
        Taking_over_ext = np.array(Data['Dispatching']['DG/grid taking over level (%)'][0:9],dtype='float64')
        Taking_overs = np.row_stack((Taking_over,Taking_over_ext))
        energy_use_repartition_DSM = Data['Dispatching']['Repartition coefficients'][0]/(Data['Dispatching']['Repartition coefficients'][0]+Data['Dispatching']['Repartition coefficients'][1])
        D_DSM_minimum_levels = np.array(Data['Dispatching']['D_DSM minimum levels'][1:24],dtype='float64')
        Y_DSM_minimum_levels = np.array(Data['Dispatching']['Y_DSM minimum levels'][1:12],dtype='float64')
        DG_strategy = Data['Dispatching']['Diesel generator'][0]
        DG_min_runtime = Data['Dispatching']['Diesel generator'][1]
        DG_min_production = Data['Dispatching']['Diesel generator'][2]
    else : 
        Defined_items = ()
        Discharge_order, Taking_overs,energy_use_repartition_DSM, D_DSM_minimum_levels, Y_DSM_minimum_levels, DG_strategy, DG_min_runtime,DG_min_production = np.repeat(np.nan,8)
        
    Dispatching = [Defined_items, Discharge_order, Taking_overs, D_DSM_minimum_levels, Y_DSM_minimum_levels, DG_strategy, DG_min_runtime, DG_min_production,energy_use_repartition_DSM]
    
    (Grid_Fossil_fuel_ratio,Main_grid_emissions,Main_grid_ratio) = (np.float64(Data['Environment']['Main grid fossil fuel ratio'][0]),np.float64(Data['Environment']['Main grid emissions (gCO2/kWh)'][0]),np.float64(Data['Environment']['Main grid ratio primary over final energy'][0]))
   
    
    r_cross_init=np.float64(Data['Hyperparameters']['Initialisation values'][0])
    n_iter_init=np.int64(Data['Hyperparameters']['Initialisation values'][1])
    r_cross=np.float64(Data['Hyperparameters']['Evolution values'][0])
    n_iter,nb_ere,n_pop,n_nodes,n_core =np.int64(Data['Hyperparameters']['Evolution values'][1:6])
    Cost_sequence = np.float64(Data['Hyperparameters']['Constraint costs'])
    operator_contract = np.float64(Data['Hyperparameters']['Contract'])
    operator_production = np.float64(Data['Hyperparameters']['Production'])
    operator_storage_volumes = np.float64(Data['Hyperparameters']['Storage volume'])
    operator_storage_use = np.float64(Data['Hyperparameters']['Storage use'])
    operator_storage_power = np.float64(Data['Hyperparameters']['Storage power'])
    operator_scheduling_consistency = np.float64(Data['Hyperparameters']['Scheduling consistency'])
    operator_storage_timeserie = np.float64(Data['Hyperparameters']['Storage timeserie'])
    operator_storage_opposite = np.float64(Data['Hyperparameters']['Storage opposite moves'])
    operator_storage_transfers = np.float64(Data['Hyperparameters']['Storage transfers'])
    operator_storage_specification = np.float64(Data['Hyperparameters']['Storage specification'])
    operator_curve_smoothing = np.float64(Data['Hyperparameters']['Curve smoothing'])
    operator_constraint = np.float64(Data['Hyperparameters']['Constraint'])
    operator_long_term_consistency = np.float64(Data['Hyperparameters']['Long-term consistency'])
    operator_YDSM = np.float64(Data['Hyperparameters']['Yearly demand-side management'])
    operator_DDSM = np.float64(Data['Hyperparameters']['Daily demand-side management'])
    hyperparameters_main = {'Initialisation':(n_core,n_nodes,r_cross_init,n_pop,n_iter_init),'Evolution':(n_core,n_nodes,r_cross,n_pop,n_iter,nb_ere)}
    hyperparameters_operators_names = np.array(('Contract','Production','Storage volume','Storage use','Storage power','Scheduling consistency','Storage timeserie','Storage opposite moves','Storage transfers','Storage specification','Curve smoothing','Constraint','Long-term consistency','Yearly demand-side management','Daily demand-side management'))
    hyperparameters_operators_num = np.array((operator_contract,operator_production,operator_storage_volumes,operator_storage_use,operator_storage_power,operator_scheduling_consistency,operator_storage_timeserie,operator_storage_opposite,operator_storage_transfers,operator_storage_specification,operator_curve_smoothing,operator_constraint,operator_long_term_consistency,operator_YDSM,operator_DDSM)).T

    r_cross_pro = np.float64(Data['Hyperparameters_pro']['Evolution values'][0])
    n_iter_pro,n_pop_pro = np.int64(Data['Hyperparameters_pro']['Evolution values'][1:3])
    operator_contract_pro = np.float64(Data['Hyperparameters_pro']['Contract'])
    operator_production_pro = np.float64(Data['Hyperparameters_pro']['Production'])
    operator_strategy_pro = np.float64(Data['Hyperparameters_pro']['Strategy'])
    operator_discharge_order_pro = np.float64(Data['Hyperparameters_pro']['Discharge order'])
    operator_energy_use_pro = np.float64(Data['Hyperparameters_pro']['Energy use'])
    operator_taking_over_pro = np.float64(Data['Hyperparameters_pro']['Taking over'])
    operator_DSM_min_levels_pro = np.float64(Data['Hyperparameters_pro']['DSM minimum levels'])
    operator_DG_min_runtime_pro = np.float64(Data['Hyperparameters_pro']['DG min runtime'])
    operator_DG_min_production_pro = np.float64(Data['Hyperparameters_pro']['DG min production'])
    operator_storages_capacity_pro = np.float64(Data['Hyperparameters_pro']['storages capacity'])
    operator_storages_power_pro = np.float64(Data['Hyperparameters_pro']['storages power'])
    operator_init_SOC_pro = np.float64(Data['Hyperparameters_pro']['Initial SOC'])
    
    hyperparameters_main_pro = (r_cross_pro,n_pop_pro,n_iter_pro)
    hyperparameters_operators_names_pro = np.array(('Contract','Production','Strategy','Discharge order','Energy use','Taking over','DSM minimum levels','DG min runtime','DG min production','storages capacity','storages power','Initial SOC'))
    hyperparameters_operators_num_pro = np.array((operator_contract_pro,operator_production_pro,operator_strategy_pro,operator_discharge_order_pro,operator_energy_use_pro,operator_taking_over_pro,operator_DSM_min_levels_pro,operator_DG_min_runtime_pro,operator_DG_min_production_pro,operator_storages_capacity_pro,operator_storages_power_pro,operator_init_SOC_pro)).T

    [prices_hour_type,prices_num,fixed_premium,Overrun,Selling_price] = compute_grid_prices(datetime_model,grid_price)
    return (pd.Series(datetime_model),Grid_Fossil_fuel_ratio,Main_grid_emissions,Main_grid_ratio,Connexion,specs_names,specs_Id,specs_num,prod_C,prods_U,Volums_prod,Y_movable_load,D_movable_load,Non_movable_load,time_resolution,constraint_num, Constraint_level ,criterion_num,storage_techs,storage_characteristics_names,storage_characteristics,Bounds_prod,n_UP,duration_years,prices_hour_type,prices_num,fixed_premium,Overrun,Selling_price,Contract_Id,hyperparameters_main,hyperparameters_operators_names,hyperparameters_operators_num,hyperparameters_main_pro,hyperparameters_operators_names_pro,hyperparameters_operators_num_pro,Cost_sequence,type_optim,Dispatching,DG_fuel_cost,DG_lifetime,DG_unit_cost,DG_maintenance_cost,DG_fuel_consumption,DG_EROI,fuel_CO2eq_emissions,groups)    
