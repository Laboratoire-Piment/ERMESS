# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 15:06:18 2026

@author: JoPHOBEA
"""

import pandas as pd
import numpy as np
import timezonefinder
import datetime
from ERMESS_scripts.utils.helpers import timeseries_interpolation
from ERMESS_scripts.utils.constraints import compute_grid_prices

from ERMESS_scripts.data.indices import ConstraintIdx, CriterionIdx
from ERMESS_scripts.cost import ERMESS_cost_functions as Cfc


from . import data_classes as  Dcl
from . import ERMESS_meteo as Eme
from . import ERMESS_PV_model as EPV
from . import ERMESS_Wind_model as EWi

def _build_production_characteristics(data):
    
    """
    Build production units characteristics arrays from raw input data.
    
    This function extracts and aggregates numerical characteristics of
    photovoltaic (PV) and wind turbine (WT) production units from the input data.
    It combines both technologies into unified numpy arrays used by the model.
    
    Args:
        data (dict[str, pandas.DataFrame]): Raw input data containing
            at least the sheets 'PV_production_specs' and 'WT_production_specs'.
    
    Returns:
        tuple:
            - characteristics_num (np.ndarray): 2D array of numerical characteristics
              for all production units (PV + WT).
            - ids (np.ndarray): Array of unique identifiers for each production unit.
            - names (np.ndarray): Array of characteristic names.   
    """

    cols = ['Capital unit cost (€)','Operational unit cost (€/yrs)','Lifetime (years)','eqCO2 Emissions (gCO2/kWh)','EROI']

    PV_specs = data["PV_production_specs"][cols].to_numpy(dtype=float)
    WT_specs = data["WT_production_specs"][cols].to_numpy(dtype=float)

    characteristics_num = np.row_stack((PV_specs, WT_specs))
    
    capacities = np.concatenate((data["PV_production_specs"]["Capacity"].to_numpy(dtype=int),data["WT_production_specs"]["Capacity"].to_numpy(dtype=np.int64)))
    groups_ID = np.concatenate((data["PV_production_specs"]["Surface group"].to_numpy(dtype=int),data["WT_production_specs"]["Surface group"].to_numpy(dtype=np.int64)))
    groups = [np.where(groups_ID==i)[0] for i in np.unique(groups_ID)]  

    ids = np.concatenate((data["PV_production_specs"]["Id"].to_numpy(dtype='U'),data["WT_production_specs"]["Id"].to_numpy(dtype='U')))

    return characteristics_num, ids, capacities, groups

def _compute_current_production(data, datetime_model, timezone):
    
    """
    Compute interpolated current production time series.
    
    This function interpolates the measured or provided current production
    onto the model time grid and ensures non-negative values.
    
    Args:
        data (dict[str, pandas.DataFrame]): Raw input data containing
            the 'TimeSeries' sheet.
        datetime_model (pandas.DatetimeIndex): Target time index for the model.
        timezone (str): Timezone used to localize datetime values.
    
    Returns:
        np.ndarray: Interpolated current production time series.
    """

    series_datetime = pd.to_datetime(data["TimeSeries"]["Datetime"],format="%d/%m/%Y %H:%M").dt.tz_localize(timezone,nonexistent="shift_forward",ambiguous=False)
    prod = timeseries_interpolation(datetime_model,series_datetime,data["TimeSeries"]["Current_production (W)"])
    prod[prod < 0] = 0

    return prod

def _compute_production_manual(data, datetime_model):
    """
    Compute production time series from user-defined unit productions.
    
    This function reads production time series directly provided by the user
    and interpolates them onto the model time grid.
    
    Args:
        data (dict[str, pandas.DataFrame]): Raw input data containing
            the 'Unit_productions' sheet.
        datetime_model (pandas.DatetimeIndex): Target time index for the model.
    
    Returns:
        tuple:
            - unit_prods (np.ndarray): 2D array of interpolated unit productions.
            - datetime_prods (pandas.Series): Original datetime index of input data.
    """

    df = data["Unit_productions"]
    unit_prods = df.drop("Datetime", axis=1).to_numpy().T
    datetime_prods = pd.to_datetime(df["Datetime"],format="%d/%m/%Y %H:%M")

    unit_prods = np.array([timeseries_interpolation(datetime_model,datetime_prods,unit_prods[i])
        for i in range(len(unit_prods)) ])

    return unit_prods, datetime_prods

def _compute_production_automatic(data, site, datetime_model, meteo_mode):
    """
    Compute production time series using physical models for PV and wind.
    
    This function generates production profiles based on meteorological data,
    either provided manually or retrieved automatically. It applies PV and wind
    models to compute unit production.
    
    Args:
        data (dict[str, pandas.DataFrame]): Raw input data.
        site (SiteData): Site information including location and timezone.
        datetime_model (pandas.DatetimeIndex): Target time index.
        meteo_mode (str): Meteorological data mode ('manual' or 'automatic').
    
    Returns:
        tuple:
            - unit_prods (np.ndarray): 2D array of production time series.
            - datetime_prods (pandas.DatetimeIndex): Production datetime index.
    """

    if meteo_mode == "manual":

        PV_meteo = data["Meteo"].loc[:, ['GHI (W/m²)', 'DNI (W/m²)', 'DHI (W/m²)','Air temperature (°C)', 'Wind speed (m/s)'
        ]].rename(columns={'GHI (W/m²)': 'ghi','DNI (W/m²)': 'dni','DHI (W/m²)': 'dhi','Air temperature (°C)': 'temp_air','Wind speed (m/s)': 'wind_speed'
        })

        datetime_prods = pd.to_datetime(data["Meteo"]["Datetime"],format="%d/%m/%Y %H:%M").tz_localize(site.timezone)

    else:
        PV_meteo, Wind_meteo = Eme.import_meteo(site.latitude,site.longitude,site.altitude,datetime_model[0],datetime_model[-1],site.timezone)
        datetime_prods = PV_meteo.index

    # --- PV ---
    PV_specs = data["PV_production_specs"]

    PV_TempParam_default = {'a': -3.47,'b': -0.0594,'deltaT': 3} #SAPM coeffs
    PV_FixModMount = [{"type":"Fixed","surface_tilt":PV_specs['Tilt (°)'][i],"surface_azimuth":PV_specs['Azimuth (°)'][i],"racking_model":PV_specs['Mounting'][i]} for i in range(len(PV_specs))]
    PV_arrayParam = [{"surface_type":PV_specs['Surface type'][i],"module_type":PV_specs['Module type'][i],"modules_per_string":PV_specs['Modules per string'][i],"strings":PV_specs['Strings'][i]} for i in range(len(PV_specs))]
    LossesParam_default = {"soiling": 2,"shading": 3,"snow": 0,"mismatch": 2,"wiring": 2,"connections": 0.5,"lid": 1.5,"nameplate_rating": 1,"age": 0,"availability": 3}   

    prods_U_PV = np.array([EPV.pvmodel(site, PV_meteo, PV_TempParam_default, PV_specs['Module'][i], PV_specs['Inverter'][i], PV_FixModMount[i], PV_arrayParam[i],LossesParam_default,False,False,False,False)[0] for i in range(len(PV_specs))])
    prods_U_PV = np.where(prods_U_PV < 0, 0, prods_U_PV)

    # --- WT ---
    WT_specs = data["WT_production_specs"].loc[:, ['Model','Hub height (m)']].rename(columns={'Hub height (m)':'hub_height','Model':'turbine_type'})
    WT_prod = [EWi.default_wind_turbines(WT_specs.iloc[i]) for i in range(len(WT_specs))]

    Terrain_type =  data['Environment']['Terrain type'][0]
    Roughness_length = 0 if Terrain_type=='off-shore' else (0.005 if Terrain_type=='no vegetation, no obstacles' else (0.005 if Terrain_type=='off-shore' else (0.03 if Terrain_type=='flat terrain, grass, isloated obstacles' else (0.1 if Terrain_type=='low crops, occasional obstacles' else (0.25 if Terrain_type=='high crops, scattered obstacles' else (0.5 if Terrain_type=='parkland, bushes, numerous obstacles' else (1 if Terrain_type=='regular large obstacles (suburbs, forest)' else (2.5 if Terrain_type=='city centre' else 0.5))))))))
    Wind_meteo[('roughness_length', 0)] = Roughness_length

    prods_U_WT = np.array([EWi.windmodel(weather=Wind_meteo, MyTurbineDict=WT_prod[i])
        for i in range(len(WT_prod))])

    # concat
    unit_prods = np.row_stack([x for x in (prods_U_PV, prods_U_WT) if x.size])

    return unit_prods, datetime_prods

def _parse_site(data):
    """
    Parse site geographical and timezone information.

    This function extracts latitude, longitude, and altitude from the input
    data and determines the associated timezone.

    Args:
        data (dict[str, pandas.DataFrame]): Raw input data containing
            the 'Environment' sheet.

    Returns:
        SiteData: Dataclass containing site information.
    """

    lat = data["Environment"]["Latitude (°)"][0]
    lon = data["Environment"]["Longitude (°)"][0]
    alt = data["Environment"]["Altitude (m)"][0]

    tz = timezonefinder.TimezoneFinder().certain_timezone_at(lat=lat,lng=lon )

    return Dcl.SiteData(latitude=lat,longitude=lon,altitude=alt,timezone=tz)

def _parse_output_config(data):
    """
    Parse output configuration data.

    This function extracts preferences for the output from the user.

    Args:
        data (dict[str, pandas.DataFrame]): Raw input data containing
            the 'Outputs' sheet.

    Returns:
        SiteData: Dataclass containing site information.
    """
    type_optim = data['Environment']['type'][0]   
    if type_optim == 'pro':
        evaluation_function = Cfc.KPI_pro
    elif type_optim =='research':
        evaluation_function = Cfc.KPI_research
    evaluation_base = Cfc.cost_baseline
    output_file_name = data["Outputs"]["File name"][0]
    export_type = data["Outputs"]["export type"][0]
    export_charts = data["Outputs"]["export charts"][0]
    include_baseline = data["Outputs"]["include baseline"][0]

    return Dcl.PostProcessConfig(evaluation_function,evaluation_base,output_file_name,export_type,export_charts,include_baseline)


def _parse_datetime(data, site):
    """
    Build the model datetime index and time-related parameters.
    
    This function generates the simulation time grid based on input time series
    and time resolution.
    
    Args:
        data (dict[str, pandas.DataFrame]): Raw input data.
        site (SiteData): Site information including timezone.
    
    Returns:
        TimeData: Dataclass containing:
            - datetime (pandas.DatetimeIndex)
            - time_resolution (float)
            - duration_years (float)
    """

    time_resolution = np.float64(data['Environment']['time resolution (steps/h)'][0])
    datetime_data = pd.to_datetime(data['TimeSeries']['Datetime'], format="%d/%m/%Y %H:%M").round('1s')
    
    datetime_model = pd.date_range(datetime_data.iloc[0],datetime_data.iloc[-1],freq=f"{int(60/time_resolution)}min",tz=site.timezone)
    duration_years = (datetime_data.iloc[-1] - datetime_data.iloc[0])/datetime.timedelta(days=1)/365
    n_bits = len(datetime_model)
    n_days=n_bits/time_resolution/24

    return Dcl.TimeData(n_bits = n_bits, n_days = n_days, datetime=datetime_model, time_resolution=time_resolution, duration_years=duration_years)

def _parse_storage(data):
    """
    Parse storage technologies and characteristics.
    
    This function extracts storage technology types and their numerical
    characteristics, including derived power costs.
    
    Args:
        data (dict[str, pandas.DataFrame]): Raw input data containing
            the 'Storages' sheet.
    
    Returns:
        StorageData: Dataclass containing storage technologies and characteristics.
    """
    
    Cost_power = data['Storages']['PCS cost (€/kW)']+data['Storages']['BOP cost (€/kW)']
    storage_characteristics = data['Storages'].T[1:].to_numpy(dtype=np.float64)
    storage_techs = data['Storages']['Technology'].to_numpy(dtype='U')
    storage_characteristics = np.vstack((storage_characteristics,Cost_power),dtype=np.float64)
    n_store = len(storage_techs)
    
    return Dcl.StorageData(n_store=n_store,techs=storage_techs, characteristics_num=storage_characteristics)
    

def _parse_loads(data, datetime_model, timezone):
    """
    Parse and interpolate load time series.
    
    This function processes non-movable and movable loads and interpolates them
    onto the model time grid.
    
    Args:
        data (dict[str, pandas.DataFrame]): Raw input data.
        datetime_model (pandas.DatetimeIndex): Target time index.
        timezone (str): Timezone for datetime localization.
    
    Returns:
        LoadData: Dataclass containing load time series.
    """

    series_datetime = pd.to_datetime(
        data["TimeSeries"]["Datetime"],
        format="%d/%m/%Y %H:%M"
    ).dt.tz_localize(
        timezone,
        nonexistent="shift_forward",
        ambiguous=False)

    non_movable = timeseries_interpolation(
        datetime_model,
        series_datetime,
        data["TimeSeries"]["Non-controllable load (kW)"])

    yearly = timeseries_interpolation(
        datetime_model,
        series_datetime,
        data["TimeSeries"]["Yearly movable load (kW)"] )

    daily = timeseries_interpolation(
        datetime_model,
        series_datetime,
        data["TimeSeries"]["Daily movable load (kW)"])

    return Dcl.LoadData(non_movable=non_movable,yearly_movable=yearly,daily_movable=daily)

def _parse_genset(data):
    """
    Parse diesel generator (genset) characteristics.
    
    Args:
        data (dict[str, pandas.DataFrame]): Raw input data containing
            the 'Diesel generator' sheet.
    
    Returns:
        GensetData: Dataclass containing generator parameters.
    """

    dg = data["Diesel generator"]
    
    fuel_cost=float(dg["Value"][0])
    lifetime=float(dg["Value"][1])
    unit_cost=float(dg["Value"][2])
    maintenance_cost=float(dg["Value"][3])
    emissions=float(dg["Value"][4])
    eroi=float(dg["Value"][5])
    fuel_consumption=np.array(dg["DG fuel consumption"][1:11],dtype=float)

    return Dcl.GensetData(fuel_cost,lifetime,unit_cost,maintenance_cost,fuel_consumption,emissions,eroi)

def _parse_grid(data, datetime_model):
    """
    Parse grid connection parameters and electricity pricing.
    
    This function extracts grid-related parameters and computes time-dependent
    electricity prices.
    
    Args:
        data (dict[str, pandas.DataFrame]): Raw input data.
        datetime_model (pandas.DatetimeIndex): Model time index.
    
    Returns:
        GridData: Dataclass containing grid characteristics and pricing.
    """

    grid_price = data["Grid_prices"]
    Contract_Ids = grid_price['Contract_Id']
    (Grid_Fossil_fuel_ratio,grid_emissions,grid_ratio) = (np.float64(data['Environment']['Main grid fossil fuel ratio'][0]),np.float64(data['Environment']['Main grid emissions (gCO2/kWh)'][0]),np.float64(data['Environment']['Main grid ratio primary over final energy'][0]))

    (prices_hour_type,prices_num,fixed_premium,overrun,selling_price) = compute_grid_prices(datetime_model, grid_price)
    n_contracts = len(Contract_Ids)

    return Dcl.GridData(n_contracts = n_contracts,
                        fossil_fuel_ratio=Grid_Fossil_fuel_ratio,
                        energy_ratio=grid_ratio,
                        CO2eq_emissions=grid_emissions,
                        price_hour_type=prices_hour_type,
                        prices=prices_num,
                        fixed_premium= fixed_premium,
                        Overrun= overrun,
                        Selling_price= selling_price,
                        Contract_Ids= Contract_Ids)

def _parse_production(data, site, datetime_model):
    """
    Parse and compute production data for all units.
    
    This function handles both manual and automatic production modes,
    computes unit production, and assembles all production-related data.
    
    Args:
        data (dict[str, pandas.DataFrame]): Raw input data.
        site (SiteData): Site information.
        datetime_model (pandas.DatetimeIndex): Model time index.
    
    Returns:
        ProductionData: Dataclass containing all production information.
    """

    env = data["Environment"]

    production_mode = env["Production"][0]
    meteo_mode = env["Meteo"][0]

    # =========================
    # 1. COMPUTE UNIT PRODUCTIONS
    # =========================

    if production_mode == "automatic":

        unit_prods, datetime_prods = _compute_production_automatic(data,site,datetime_model,meteo_mode)

    elif production_mode == "manual":

        unit_prods, datetime_prods = _compute_production_manual(data,datetime_model)

    else:
        raise ValueError(f"Unknown production mode: {production_mode}")

    # =========================
    # 2. CHARACTERISTICS
    # =========================

    characteristics_num, ids, capacities, groups = _build_production_characteristics(data)

    # =========================
    # 3. CURRENT PRODUCTION
    # =========================

    current_prod = _compute_current_production(data,datetime_model,site.timezone)

    # =========================
    # 4. POST PROCESSING
    # =========================

    volumes = np.sum(unit_prods, axis=1)
    characteristics_num = np.column_stack((characteristics_num,volumes))

    numbers = len(characteristics_num)

    return Dcl.ProductionData(
        ids=ids,
        characteristics_num=characteristics_num,
        capacities=capacities,
        groups=groups,
        current_prod=current_prod,
        unit_prods=unit_prods,
        numbers=numbers
    )

def parse_constraint(value) :
    mapping = {
        "Self-sufficiency": ConstraintIdx.Self_sufficiency,
        "Self-consumption": ConstraintIdx.Self_consumption,
        "REN fraction": ConstraintIdx.REN_fraction,
    }
    return mapping.get(value)


def parse_criterion(value) :
    mapping = {
        "LCOE": CriterionIdx.LCOE,
        "Annual net benefits": CriterionIdx.Annual_net_benefits,
        "NPV": CriterionIdx.NPV,
        "Self-sufficiency": CriterionIdx.Self_sufficiency,
        "Self-consumption": CriterionIdx.Self_consumption,
        "Autonomy": CriterionIdx.Autonomy,
        "eqCO2 Emissions": CriterionIdx.EqCO2emissions,
        "fossil fuel consumption": CriterionIdx.Fossil_fuel_consumption,
        "EROI": CriterionIdx.EROI,
        "Energy losses": CriterionIdx.Energy_losses,
        "Maximum power from grid": CriterionIdx.Max_power_from_grid,
    }
    return mapping.get(value)

def _parse_optimization(data):
    """
    Parse optimization configuration and constraints.
    
    This function converts user-defined optimization settings into numerical
    parameters used by the model.
    
    Args:
        data (dict[str, pandas.DataFrame]): Raw input data.
    
    Returns:
        OptimizationData: Dataclass containing optimization parameters.
    """

    constraint_str = data['Environment']['Constraint'][0]
                
    constraint_level = np.float64(data['Environment']['Constraint level'][0])
    criterion = data['Environment']['Optimisation criterion'][0]
    
    constraint_num = parse_constraint(constraint_str)
    criterion_num = parse_criterion(criterion)
            
    type_optim = data['Environment']['type'][0]   
    
    return Dcl.OptimizationData(constraint_num=constraint_num,constraint_level=constraint_level,criterion_num=criterion_num,type_optim=type_optim)



def _parse_hyperparameters(data):
    """
    Parse genetic algorithm hyperparameters.
    
    This function extracts initialization and evolution parameters, as well as
    operator weights for the optimization process.
    
    Args:
        data (dict[str, pandas.DataFrame]): Raw input data.
    
    Returns:
        HyperparametersData: Dataclass containing GA hyperparameters.
    """

    r_cross_init,elitism_probability_init,cost_constraint_init=data['Hyperparameters']['Initialisation values'][np.array([0,1,7])]
    n_iter_init=np.int64(data['Hyperparameters']['Initialisation values'][2])
    r_cross,elitism_probability=np.float64(data['Hyperparameters']['Evolution values'][0:2])
    n_iter,nb_ere,n_pop,n_nodes,n_core,cost_constraint=np.int64(data['Hyperparameters']['Evolution values'][2:8])
    
    hyperparameters_operators_num = np.float64(data['Hyperparameters'][['Contract','Production','Storage volume','Storage_global','Storage_power','Storage_trades_consistency','Storage_patterns','Inter_storages','Storage_mix','Curve_smoothing','Constraint_forcing','Interdaily_consistency','DSM_trades_consistency','DSM_noise']])
         
                    
    return Dcl.HyperparametersData(n_iter=n_iter,n_iter_init=n_iter_init,elitism_probability_init=elitism_probability_init,nb_ere=nb_ere,n_pop=n_pop,r_cross=r_cross,r_cross_init=r_cross_init,n_nodes=n_nodes,n_core=n_core,cost_constraint=cost_constraint,elitism_probability=elitism_probability,operators_num=hyperparameters_operators_num)

def _parse_hyperparametersPro(data):
    """
    Parse advanced (professional mode) hyperparameters.
    
    Args:
        data (dict[str, pandas.DataFrame]): Raw input data.
    
    Returns:
        HyperparametersProData: Dataclass containing advanced parameters.
    """

    r_cross_pro,elitism_probability = np.float64(data['Hyperparameters_pro']['Evolution values'][0:2])
    n_iter_pro,n_pop_pro,cost_constraint_pro = np.int64(data['Hyperparameters_pro']['Evolution values'][2:5])
    
 #   operator_contract_pro = np.float64(data['Hyperparameters_pro']['Contract'])
 #   operator_production_pro = np.float64(data['Hyperparameters_pro']['Production'])
 #   operator_strategy_pro = np.float64(data['Hyperparameters_pro']['Strategy'])
 #   operator_discharge_order_pro = np.float64(data['Hyperparameters_pro']['Discharge order'])
 #   operator_energy_use_pro = np.float64(data['Hyperparameters_pro']['Energy use'])
 #   operator_taking_over_pro = np.float64(data['Hyperparameters_pro']['Taking over'])
 #   operator_DSM_min_levels_pro = np.float64(data['Hyperparameters_pro']['DSM minimum levels'])
 #   operator_DG_min_runtime_pro = np.float64(data['Hyperparameters_pro']['DG min runtime'])
 #   operator_DG_min_production_pro = np.float64(data['Hyperparameters_pro']['DG min production'])
 #   operator_storages_capacity_pro = np.float64(data['Hyperparameters_pro']['storages capacity'])
 #   operator_storages_power_pro = np.float64(data['Hyperparameters_pro']['storages power'])
 #   operator_init_SOC_pro = np.float64(data['Hyperparameters_pro']['Initial SOC'])
    

 #   hyperparameters_operators_names_pro = np.array(('Contract','Production','Strategy','Discharge order','Energy use','Taking over','DSM minimum levels','DG min runtime','DG min production','storages capacity','storages power','Initial SOC'))
 #   hyperparameters_operators_num_pro = np.array((operator_contract_pro,operator_production_pro,operator_strategy_pro,operator_discharge_order_pro,operator_energy_use_pro,operator_taking_over_pro,operator_DSM_min_levels_pro,operator_DG_min_runtime_pro,operator_DG_min_production_pro,operator_storages_capacity_pro,operator_storages_power_pro,operator_init_SOC_pro)).T
    hyperparameters_operators_num_pro = np.float64(data['Hyperparameters_pro'][['Contract','Production','Strategy','Discharge order','Energy use','Overlap','DSM minimum levels','DG control','storages capacity','storages power','Initial SOC']])

    return Dcl.HyperparametersProData(r_cross = r_cross_pro , n_pop = n_pop_pro , n_iter = n_iter_pro ,cost_constraint=cost_constraint_pro,elitism_probability=elitism_probability,operators_num=hyperparameters_operators_num_pro)

def _parse_dispatching(data):
    """
    Parse dispatching strategy and control parameters.
    
    This function extracts user-defined dispatching rules such as storage
    discharge order, DSM constraints, and generator control.
    
    Args:
        data (dict[str, pandas.DataFrame]): Raw input data.
    
    Returns:
        DispatchingData: Dataclass containing dispatching configuration.
    """
    
    defined_items = np.where(data['Dispatching']['User-Defined']=='Yes')[0]
    
    if 'Storages management' in defined_items :
        Discharge_order = (np.array(data['Dispatching']['Discharge order'],dtype=np.int64)[~pd.isnull(np.array(data['Dispatching']['Storages']))])-1 
        Intern_overlap = np.array(data['Dispatching']['Overlap levels (%)'][0:9],dtype='float64')
        Extern_overlap = np.array(data['Dispatching']['DG/grid overlap levels (%)'][0:9],dtype='float64')
        Overlaps = np.row_stack((Intern_overlap,Extern_overlap))
    else :
        Discharge_order = None
        Overlaps = None   
    
    if 'DSM' in defined_items :
        energy_use_repartition_DSM = data['Dispatching']['Repartition coefficients'][0]/(data['Dispatching']['Repartition coefficients'][0]+data['Dispatching']['Repartition coefficients'][1])
        D_DSM_minimum_levels = np.array(data['Dispatching']['D_DSM minimum levels'][1:24],dtype='float64')
        Y_DSM_minimum_levels = np.array(data['Dispatching']['Y_DSM minimum levels'][1:12],dtype='float64')
    else :
        energy_use_repartition_DSM = None
        D_DSM_minimum_levels = None
        Y_DSM_minimum_levels = None
        
    if 'Genset control' in defined_items :
        DG_strategy = data['Dispatching']['Diesel generator'][0]
        DG_min_runtime = data['Dispatching']['Diesel generator'][1]
        DG_min_production = data['Dispatching']['Diesel generator'][2]
    else :
        DG_strategy = None
        DG_min_runtime = None
        DG_min_production = None
   
    return Dcl.DispatchingData(defined_items, Discharge_order, Overlaps, energy_use_repartition_DSM, D_DSM_minimum_levels, Y_DSM_minimum_levels, DG_strategy, DG_min_runtime, DG_min_production)

def _parse_ERMESSInputs(data):
    """
    Parse full ERMESS input data structure.
    
    This is the main parser that orchestrates all sub-parsers to build
    a complete ERMESSInputs object from raw input data.
    
    It handles:
    - Site and time data
    - Loads, storage, and production
    - Grid or genset configuration
    - Optimization and hyperparameters
    - Optional professional mode features
    
    Args:
        data (dict[str, pandas.DataFrame]): Raw input data.
    
    Returns:
        ERMESSInputs: Fully structured input object for the optimization model.
    """

    # --- blocs obligatoires ---
    siteData = _parse_site(data)
    TimeData = _parse_datetime(data, siteData)
    loadsData = _parse_loads(data, TimeData.datetime, siteData.timezone)
    storageData = _parse_storage(data)
    productionData = _parse_production(data, siteData, TimeData.datetime)
    
    optimizationData = _parse_optimization(data)
    postProcessConfigData = _parse_output_config(data)
    
    if optimizationData.type_optim=='pro':
        hyperparametersProData = _parse_hyperparametersPro(data)
        dispatchingData = _parse_dispatching(data)
        hyperparametersData = None
    else :
        hyperparametersProData = _parse_hyperparametersPro(data)
        dispatchingData = _parse_dispatching(data)
        hyperparametersData = _parse_hyperparameters(data)

    connexion = data["Environment"]["Connexion"][0]
    tracking = data["Environment"]["Tracking"][0]

    if connexion == "On-grid":
        gridData = _parse_grid(data, TimeData.datetime)
        gensetData = None

    elif connexion == "Off-grid":
        gensetData = _parse_genset(data)
        gridData = None

    else:
        raise ValueError(f"Unknown connexion type: {connexion}")

    return Dcl.ERMESSInputs(
        time= TimeData,
        production= productionData,
        storage= storageData,
        load= loadsData,
        grid= gridData,
        genset= gensetData,
        optimization= optimizationData,
        hyperparameters= hyperparametersData,
        hyperparameterspro= hyperparametersProData,
        dispatching= dispatchingData,
        connexion= connexion,
        postProcessConfig = postProcessConfigData,
        tracking= tracking
    )