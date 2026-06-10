# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 11:50:01 2026

@author: JoPHOBEA
"""

"""
Stochastic building energy load generation model.

This module generates synthetic electricity and thermal demand profiles
for heterogeneous buildings based on occupancy, behavioral models,
weather data, and end-use stochastic processes.

The model includes:
- occupancy generation with AR(1) temporal correlation
- lighting, IT, shared equipment, EV, DHW, HVAC loads
- base electrical load
- load flexibility decomposition
- microgrid aggregation across multiple buildings

All loads are time-resolved and dependent on meteorological inputs.
"""

import numpy as np
import pandas as pd

from ERMESS_scripts.data.config_profiles import PROFILES


def is_holiday(ts, holidays):
    """
    Check whether a timestamp corresponds to a holiday.
    
    Args:
        ts (datetime-like): Timestamp to evaluate.
        holidays (iterable): Collection of holiday dates (pandas.Timestamp or datetime).
    
    Returns:
        bool: True if the date is a holiday, False otherwise.
    """
    return pd.Timestamp(ts.date()) in holidays

def is_vacation(ts, vacations):
    """
    Check whether a timestamp falls within a vacation period.
    
    Args:
        ts (datetime-like): Timestamp to evaluate.
        vacations (pandas.DataFrame): DataFrame containing:
            - 'vacation start dates'
            - 'vacation end dates'
    
    Returns:
        bool: True if the timestamp is within a vacation period, False otherwise.
    """
    date = pd.Timestamp(ts.date())
    return ((vacations['vacation start dates'] <= date) & (date <= vacations['vacation end dates'])).any()

def generate_ar1(n_steps, rho):
    """
    Generate an AR(1) stochastic process.
    
    This function creates a temporally correlated random process used
    to model occupancy and activity fluctuations.
    
    Args:
        n_steps (int): Number of time steps.
        rho (float): Autocorrelation coefficient (0 < rho < 1).
    
    Returns:
        numpy.ndarray: AR(1) process of shape (n_steps,) with zero mean.
    """
    eps = np.random.normal(0, 1, n_steps)
    z = np.zeros(n_steps)

    for t in range(1, n_steps):
        z[t] = rho * z[t-1] + np.sqrt(1 - rho**2) * eps[t]

    return z

def compute_occupancy(series_datetime, profile, events, n_steps, occupancy_params, n_users, time_resolution):
    """
    Generate stochastic occupancy time series for a building.
    
    Occupancy is computed from:
    - deterministic hourly weekday/weekend profiles
    - holiday and vacation modulation factors
    - AR(1) correlated stochastic variability
    - binomial sampling over users
    
    Args:
        series_datetime (pandas.DatetimeIndex): Time index.
        profile (str): Building archetype used to select parameters.
        events (pandas.DataFrame): Boolean DataFrame with columns:
            - 'holiday'
            - 'vacation'
        n_steps (int): Number of simulation steps.
        occupancy_params (dict): Occupancy model parameters from PROFILES.
        n_users (int): Number of building occupants.
        time_resolution (float): Time step resolution in hours.
    
    Returns:
        pandas.Series: Number of present users at each time step.
    """    
    
    hours = np.array([ts.hour for ts in series_datetime])
    weekday = np.array([ts.dayofweek < 5 for ts in series_datetime])
    
    p_weekday = occupancy_params["occupancy_profile_weekday"]
    p_weekend = occupancy_params["occupancy_profile_weekend"]
    
    p_weekday_arr = np.array([p_weekday[h] for h in range(24)])
    p_weekend_arr = np.array([p_weekend[h] for h in range(24)])

    p_base = np.where(weekday, p_weekday_arr[hours], p_weekend_arr[hours])
        
    holiday_factor = np.where(events['holiday'],occupancy_params["holiday_factor"],1.0)
    vacation_factor = np.where(events['vacation'],occupancy_params["vacation_factor"],1.0)

    p_target = p_base * holiday_factor * vacation_factor
    
    rho_occ = np.exp(-1./(occupancy_params["occupancy_memory"]*time_resolution))
    z_occ = generate_ar1(n_steps, rho_occ)
    
    sigma_occ = occupancy_params["occupancy_correlation"] * np.sqrt(p_target * (1 - p_target))
    p_real = np.clip(p_target + sigma_occ * z_occ,0, 1)
    occupancy = pd.Series(np.random.binomial(n_users, p_real),index = series_datetime)

    return occupancy

def lighting_load(occupancy, ghi, z_act, simultaneity_factor, n_steps, light_params):
    """
    Compute lighting electricity demand.
    
    Lighting usage depends on:
    - solar irradiance (GHI)
    - stochastic behavioral variability
    - occupancy-driven activation
    
    Args:
        occupancy (array-like): Number of present users per timestep.
        ghi (array-like): Global horizontal irradiance (W/m²).
        z_act (numpy.ndarray): AR(1) activity noise process.
        simultaneity_factor (float): Correlation strength between users.
        n_steps (int): Number of time steps.
        light_params (dict): Lighting model parameters.
    
    Returns:
        numpy.ndarray: Total lighting power time series.
    """
    p_light_inst = 1 / (1 + np.exp((ghi - light_params["GHI_threshold"]) / light_params["irradiance_smoothness"]))
    sigma_light = simultaneity_factor*np.sqrt(p_light_inst*(1-p_light_inst))
    p_light_real = np.clip(p_light_inst + sigma_light*z_act,0, 1)
    
    active_users = np.random.binomial(occupancy, p_light_real)
    power_light = np.random.normal(light_params['mean_light_power'],light_params['std_light_power'],size=(n_steps, np.max(occupancy) if np.max(occupancy) > 0 else 1))
    mask = np.zeros_like(power_light, dtype=bool)

    for t in range(n_steps):
        if active_users[t] > 0:
            mask[t, :active_users[t]] = True

    P_light = np.sum(power_light * mask, axis=1)

    return(P_light)

def it_load(occupancy, z_act, mean_activity, simultaneity_factor, n_steps, it_params):
    """
    Compute IT and plug-load demand per building.
    
    Each active occupant contributes stochastic IT power consumption.
    
    Args:
        occupancy (array-like): Number of present users per timestep.
        z_act (numpy.ndarray): AR(1) activity process.
        mean_activity (float): Average probability of appliance usage.
        simultaneity_factor (float): Correlation between users' activities.
        n_steps (int): Number of time steps.
        it_params (dict): IT load parameters.
    
    Returns:
        numpy.ndarray: IT electrical load time series.
    """
    p_it_inst = mean_activity
    sigma_it = simultaneity_factor*np.sqrt(p_it_inst*(1-p_it_inst))
    p_it_real = np.clip(p_it_inst + sigma_it*z_act,0, 1)
    
    active_users = np.random.binomial(occupancy, p_it_real)
    power_it = np.random.normal(it_params['mean_it_power'],it_params['std_it_power'],size=(n_steps, np.max(occupancy) if np.max(occupancy) > 0 else 1))
    mask = np.zeros_like(power_it, dtype=bool)

    for t in range(n_steps):
        if active_users[t] > 0:
            mask[t, :active_users[t]] = True

    p_it = np.sum(power_it * mask, axis=1)

    return(p_it)
      

def shared_load(n_present, n_users, z_act, mean_activity, shared_params, simultaneity_factor):
    """
    Compute shared equipment electrical load.
    
    Shared load scales with occupancy ratio and diversity effects,
    with stochastic variability applied at the aggregate level.
    
    Args:
        n_present (array-like): Number of present users per timestep.
        n_users (int): Total number of users in building.
        z_act (numpy.ndarray): AR(1) stochastic process.
        mean_activity (float): Average activity level.
        shared_params (dict): Shared equipment parameters.
        simultaneity_factor (float): Correlation strength.
    
    Returns:
        numpy.ndarray: Shared equipment power demand time series.
    """
    occ_ratio = n_present / n_users
    
    usage = mean_activity * occ_ratio**shared_params['shared_scaling_exponent']
    usage = np.clip(usage + simultaneity_factor * np.sqrt(usage*(1-usage)) * z_act,0, 1)
    
    p_shared = shared_params["shared_installed_power"] * usage
    
    return(p_shared)

def EV_load(n_users, n_steps, time_resolution, n_present, EV_params, mean_activity):
    """
    Simulate electric vehicle charging demand.
    
    EV charging is modeled as:
    - stochastic session initiation
    - stochastic session termination
    - constant random charging power per session
    
    Args:
        n_users (int): Number of users in building.
        n_steps (int): Number of time steps.
        time_resolution (float): Time resolution in hours.
        n_present (array-like): Number of present users.
        EV_params (dict): EV charging parameters.
        mean_activity (float): Activity scaling factor.
    
    Returns:
        numpy.ndarray: EV charging power time series.
    """
    
    p_start = EV_params['EV_charge_rate'] * (n_present / n_users) * mean_activity / time_resolution 
    p_end = 1/(EV_params['EV_charge_duration_mean']*time_resolution)
    
    n_new_sessions = np.random.binomial(n_present,p_start)
    n_sessions = np.zeros(n_steps)
    for t in range(1, n_steps):
        n_sessions[t] = (n_sessions[t-1]+ n_new_sessions[t]- np.random.binomial(n_sessions[t-1], p_end))
    
    P_EV = n_sessions * np.random.uniform(EV_params['EV_power_min'], EV_params['EV_power_max'], n_steps)

    return P_EV

def dhw_load(n_present, dhw_params, n_steps, n_users, mean_activity, dhw_physical_params, time_resolution):
    
    """
    Simulate domestic hot water (DHW) energy demand.
    
    The model includes:
    - stochastic draw events per occupant
    - thermal energy conversion of water usage
    - tank state-of-charge dynamics
    - hysteresis-based heating control
    
    Args:
        n_present (array-like): Number of present users.
        dhw_params (dict): DHW system parameters.
        n_steps (int): Number of time steps.
        n_users (int): Total number of users.
        mean_activity (float): Activity scaling factor.
        dhw_physical_params (dict): Physical constants (water density, Cp, etc.).
        time_resolution (float): Time resolution in hours.
    
    Returns:
        numpy.ndarray: Electrical power required by DHW heater.
    """

    MIN_PER_HOUR = 60
    E_tank_max = (dhw_params["Tank_capacity"]* dhw_physical_params['rho_water']* dhw_physical_params['cp_water']* (dhw_params["T_hot"] - dhw_params["T_cold"])/ dhw_physical_params['JOULES_CONVERSION_FACTOR'])
    p_draw_start = dhw_params["dhw_draw_rate"]* (n_present / n_users) * mean_activity / time_resolution
    p_draw_end = min(1.0,MIN_PER_HOUR/(dhw_params['mean_duration']*time_resolution))
    
    n_new_draws = np.random.binomial(n_present, p_draw_start)
    P_draw_mean = (dhw_params["dhw_mean_flow"] * dhw_physical_params['rho_water'] * dhw_physical_params['cp_water']* (dhw_params["T_hot"] - dhw_params["T_cold"])) / 60000
    
    P_heater = np.zeros(n_steps)
    soc = np.zeros(n_steps)+0.5

    heater_on = False
    
    n_draws = np.zeros(n_steps)
    for t in range(1, n_steps):
        n_draws[t] = (n_draws[t-1]+ n_new_draws[t]- np.random.binomial(n_draws[t-1], p_draw_end))
        P_draw=n_draws[t]* P_draw_mean
        
        soc[t] = soc[t-1] -  (P_draw / time_resolution) / E_tank_max
        
        if (not heater_on and soc[t] < dhw_params["Charging_SOC_start"]):
            heater_on = True
        elif (heater_on and soc[t] > dhw_params["Charging_SOC_end"]):
            heater_on = False
            
        P_heater[t] = (dhw_params["heater_power"]/dhw_params["thermal_gain_factor"] if heater_on else 0)
        soc[t] += (P_heater[t] / time_resolution) / E_tank_max

        soc[t] = np.clip(soc[t], 0, 1)

    
    return P_heater

def hvac_load(occupancy, n_steps, dt, comfort_params,comfort_temp, meteoData):
    
    """
    Simulate building thermal dynamics and HVAC electricity demand.
    
    The model includes:
    - indoor temperature evolution
    - solar heat gains through windows
    - wind-driven infiltration losses
    - internal gains from HVAC and occupants
    - heating and cooling control with comfort thresholds
    
    Args:
        occupancy (array-like): Number of occupants per timestep.
        n_steps (int): Number of simulation steps.
        dt (float): Time step in seconds.
        comfort_params (dict): HVAC and thermal parameters.
        comfort_temp (dict): Comfort temperature bounds.
        meteoData (pandas.DataFrame): Weather data (temperature, GHI, wind).
    
    Returns:
        tuple[numpy.ndarray, numpy.ndarray]:
            - cooling electrical load (kW)
            - heating electrical load (kW)
    """

    T_in = np.zeros(n_steps)
    T_in[0] = (np.mean(meteoData["temp_air"])*3/4 + meteoData["temp_air"].iloc[0]/4)
    
    Q_solar = comfort_params['solar_heat_gain'] * comfort_params['windows_surface'] *  meteoData["ghi"].values
    Q_heat = np.zeros(n_steps)
    Q_cool = np.zeros(n_steps)
    Q_wind = np.zeros(n_steps)
    wind = meteoData["wind_speed"].values
    T_ext = meteoData["temp_air"].values

    for t in range(1, n_steps):

        error_heat = np.maximum(0, comfort_temp['T_comfort_min'] - T_in[t-1])
        error_cool = np.maximum(0, T_in[t-1] - comfort_temp['T_comfort_max'])

        Q_heat[t] = min(comfort_params['hvac_gain_per_user'] * occupancy[t] * error_heat,comfort_params['heat_saturation'] * comfort_params['hvac_COP'])
        Q_cool[t] = min(comfort_params['hvac_gain_per_user'] * occupancy[t] * error_cool,comfort_params['cold_saturation'] * comfort_params['hvac_EER'])
        Q_wind[t] = (comfort_params['wind_sensitivity'] * wind[t]* (T_ext[t] - T_in[t-1]))


        T_in[t] = T_in[t-1] + dt * ((T_ext[t] - T_in[t-1])/(comfort_params['thermal_resistance'] * comfort_params['thermal_capacity'])+ (Q_solar[t] + Q_wind[t] + Q_heat[t] - Q_cool[t]) / comfort_params['thermal_capacity'])

    P_cold_elec = np.minimum(Q_cool / comfort_params['hvac_EER'],comfort_params['cold_saturation'])/1000.
    P_heat_elec = np.minimum(Q_heat / comfort_params['hvac_COP'],comfort_params['heat_saturation'])/1000.

    return(P_cold_elec,P_heat_elec) 

def base_load(n_steps,n_users,time_resolution,base_load_params):
    
    """
    Compute aggregated base electrical load.
    
    Represents:
    - standby consumption
    - always-on appliances
    - background electrical usage
    
    Args:
        n_steps (int): Number of time steps.
        n_users (int): Number of users in building.
        time_resolution (float): Time resolution in hours.
        base_load_params (dict): Base load model parameters.
    
    Returns:
        numpy.ndarray: Base electrical load time series.
    """

    rho = np.exp(-1 / (base_load_params["noise_memory"]* time_resolution))
    z = generate_ar1(n_steps, rho)
    P_nominal = (base_load_params["power_per_user"] * n_users**base_load_params['scaling_exponent'])
    P_base = (P_nominal* (1+ base_load_params["noise_std"]* z))
    return np.maximum(P_base, 0)

def generate_occupancy_and_activity(building,time_resolution,events,meteoData):
    
    """
    Generate full building load time series.
    
    This function combines:
    - occupancy simulation
    - lighting demand
    - IT loads
    - shared equipment
    - EV charging
    - DHW demand
    - HVAC thermal loads
    - base electrical load
    
    Args:
        building (dict): Building configuration (type, number of users, comfort temps).
        time_resolution (float): Time step resolution in hours.
        events (pandas.DataFrame): Holiday and vacation indicators.
        meteoData (pandas.DataFrame): Weather data.
    
    Returns:
        tuple:
            pandas.Series: Occupancy time series.
            pandas.DataFrame: End-use electrical and thermal loads.
    """
    
    series_datetime = meteoData.index
    n_steps = len(series_datetime)
    profile = building["Type"]
    n_users = building["Number of users"]

    simultaneity_factor = PROFILES[profile]["simultaneity_factor"]
    activity_memory = PROFILES[profile]["activity_memory"]
    mean_activity = PROFILES[profile]["mean_activity"]
    
    occupancy_params = PROFILES[profile]["occupancy_params"] 
    base_load_params = PROFILES[profile]["base_load_params"]
    shared_params = PROFILES[profile]["shared_params"]
    EV_params = PROFILES[profile]["EV_params"]
    dhw_params = PROFILES[profile]["dhw_params"]
    it_params = PROFILES[profile]["it_params"]
    light_params = PROFILES[profile]["light_params"]
    comfort_params = PROFILES[profile]["comfort_params"] 
    comfort_temp = {'T_comfort_min':np.float32(building["Min. comfort temperature (°C)"]) ,
                    'T_comfort_max':np.float32(building["Max. comfort temperature (°C)"])}
    
    #Control the weight of the memory
    rho_act = np.exp(-1./(activity_memory*time_resolution))
    
    #Thresholds & specific powers
    dhw_physical_params ={'rho_water' : 1.0, 'cp_water':4186, 'JOULES_CONVERSION_FACTOR':3600000}
    dt = 3600. / time_resolution

    z_act = generate_ar1(n_steps, rho_act)

    occupancy = compute_occupancy(series_datetime, profile, events, n_steps, occupancy_params, n_users, time_resolution)
    p_light = lighting_load(occupancy, meteoData['ghi'], z_act, simultaneity_factor, n_steps, light_params)
    p_it = it_load(occupancy, z_act, mean_activity, simultaneity_factor, n_steps, it_params)
    p_shared = shared_load(occupancy, n_users, z_act, mean_activity, shared_params, simultaneity_factor)
    p_EV = EV_load(n_users, n_steps, time_resolution, occupancy, EV_params, mean_activity)
    p_dhw = dhw_load(occupancy, dhw_params, n_steps, n_users, mean_activity,dhw_physical_params,time_resolution)
    p_cold,p_heat = hvac_load(occupancy, n_steps, dt, comfort_params,comfort_temp, meteoData)
    p_base = base_load(n_steps,n_users,time_resolution,base_load_params)
        
    occupancy = pd.Series(occupancy, index = series_datetime)
    activity = pd.DataFrame({"base":p_base,"lighting":p_light,"IT and office equipment":p_it,"shared equipment":p_shared,"electric vehicles":p_EV,"domestic hot water":p_dhw,"heating":p_heat,"cooling":p_cold},index=series_datetime)
    
    return occupancy,activity

def separate_loads(activity,activity_memory,time_resolution,n_steps,load_flexibility):
    
    """
    Decompose electrical loads into flexibility categories.
    
    Each load is split into:
    - non-movable loads
    - daily movable loads
    - yearly movable loads
    
    The decomposition is based on:
    - stochastic allocation
    - AR(1)-based temporal correlation
    - predefined flexibility coefficients
    
    Args:
        activity (pandas.DataFrame): End-use load time series.
        activity_memory (float): Temporal memory of load flexibility.
        time_resolution (float): Time resolution in hours.
        n_steps (int): Number of time steps.
        load_flexibility (pandas.DataFrame): Flexibility coefficients.
    
    Returns:
        pandas.DataFrame: Load decomposition by flexibility class.
    """
    
    SIGMA_SEPARATION = 0.05
    rates = load_flexibility.set_index("Usage")[["Non movable rate","daily movable rate","yearly movable rate"]]
    
    rho_separation = np.exp(-1./(activity_memory*time_resolution))
    z = generate_ar1(n_steps, rho_separation)

    real_rates = np.zeros((n_steps, len(activity.keys()), 3))
    for i in range(len(activity.keys())):
        for j in range(3):
            real_rates[:,i,j] = ( rates.iloc[i,j]+ SIGMA_SEPARATION*z)
    real_rates = np.clip(real_rates,0.001,None)
    real_rates /= real_rates.sum(axis=2,keepdims=True)
    
    loads = np.einsum("ti,tij->tj",activity.values,real_rates)
    loads = pd.DataFrame(loads,index = activity.index)
    loads.columns = ["non_movable","daily_movable","yearly_movable"] 
    
    return(loads)

def generate_microgrid_load(building_list,load_flexibility,meteoData,holidays,vacations, time_resolution):
    
    """
    Generate aggregated microgrid load from multiple buildings.
    
    The workflow is:
    1. Generate occupancy and loads per building
    2. Apply flexibility decomposition
    3. Aggregate all buildings
    4. Compute total system load
    
    Args:
        building_list (pandas.DataFrame): List of building definitions.
        load_flexibility (pandas.DataFrame): Flexibility parameters.
        meteoData (pandas.DataFrame): Weather time series.
        holidays (iterable): List of holiday dates.
        vacations (pandas.DataFrame): Vacation periods.
        time_resolution (float): Time step resolution in hours.
    
    Returns:
        pandas.DataFrame: Aggregated microgrid load containing:
            - non_movable
            - daily_movable
            - yearly_movable
            - total_load
    """

    n_building = 0
    buildings = []
    names=[]
    
    series_datetime = meteoData.index
    holiday_factor = pd.Series([True if is_holiday(ts, holidays) else False for ts in series_datetime],index = series_datetime)
    vacation_factor = pd.Series([True if is_vacation(ts, vacations) else False for ts in series_datetime],index = series_datetime)
    events = pd.DataFrame({'holiday': holiday_factor, 'vacation': vacation_factor}, index = series_datetime)
    
    for idx, building in building_list.iterrows():
        
        n_building += building['Number']
        n_steps = len(series_datetime)
        
        for i in range(building["Number"]):
            occupancy,activity = generate_occupancy_and_activity(building,time_resolution,events,meteoData)
            activity_memory = PROFILES[building["Type"]]["activity_memory"]
            loads = separate_loads(activity,activity_memory,time_resolution,n_steps,load_flexibility)
            names.append(building["Building"]+" "+str(i) if building["Number"]>1 else building["Building"])
            buildings.append(loads)

    microgrid_load = pd.concat(buildings).groupby(level=0).sum()
    microgrid_load.columns = ["non_movable","daily_movable","yearly_movable"] 

    microgrid_load["total_load"] = microgrid_load.sum(axis=1)

    return microgrid_load


