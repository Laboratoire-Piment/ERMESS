# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 11:50:01 2026

@author: JoPHOBEA
"""

import numpy as np
import pandas as pd

from ERMESS_scripts.data.config_profiles import PROFILES

def normalize_profile(profile):
    TARGET_DAILY_ENERGY = 1.0
    current = sum(profile.values())

    factor = TARGET_DAILY_ENERGY / current

    return {h: round(v * factor, 3) for h, v in profile.items()}

# -----------------------------
# 2. BUILDING MODEL
# -----------------------------

def is_holiday(ts, holidays):
    return pd.Timestamp(ts.date()) in holidays

def is_vacation(ts, vacations):
    date = pd.Timestamp(ts.date())
    return ((vacations['vacation start dates'] <= date) & (date <= vacations['vacation end dates'])).any()

def f_temp(T,T_comfort_min,T_comfort_max):
    if T < T_comfort_min :
        return (((T_comfort_min-T)/10) ** 1.5)
    elif T > T_comfort_max :
        return (((T-T_comfort_max)/10) ** 1.5)
    else : 
        return 0
        
def f_ghi(GHI, sensitivity, GHI_ref=800):
    saturation = 1 - np.exp(-GHI / GHI_ref)
    return max(0.0, 1 - sensitivity * saturation)

def f_wind(w):
    return np.log(1 + w)

def get_profile(profile, ts,holidays,vacations,T_comfort_min,T_comfort_max, meteo_values, temp_sensitivity, ghi_sensitivity, wind_sensitivity, GHI_ref):

    weekend = PROFILES[profile]["weekend_factor"] if ts.dayofweek>= 5 else 1.0
    holiday = PROFILES[profile]["holiday_factor"] if is_holiday(ts, holidays) else 1.0
    vacation = PROFILES[profile]["vacation_factor"] if is_vacation(ts, vacations) else 1.0 
    hourly_base = PROFILES[profile]["hourly_profile"][ts.hour]
    
    weather = (1 + temp_sensitivity*f_temp(meteo_values['temp_air'],T_comfort_min,T_comfort_max)) * f_ghi(meteo_values['ghi'], ghi_sensitivity, GHI_ref) * (1 + wind_sensitivity * f_wind(meteo_values['wind_speed']))
    
    return hourly_base*weekend*holiday*vacation*weather

def generate_building_load(building,meteoData,holidays,vacations):

    profile = building["Type"]

    load = []
    temp_sensitivity = PROFILES[profile]["temp_sensitivity"]
    ghi_sensitivity = PROFILES[profile]["ghi_sensitivity"]
    wind_sensitivity = PROFILES[profile]["wind_sensitivity"]
    T_comfort_min = building["Min. comfort temperature (°C)"]
    T_comfort_max = building["Max. comfort temperature (°C)"]
    GHI_ref = np.percentile(meteoData['ghi'], 90)
    series_datetime = meteoData.index
    
    for ts in series_datetime:

        meteo_values = meteoData.loc[ts]
        value = get_profile(profile, ts,holidays,vacations,T_comfort_min,T_comfort_max, meteo_values, temp_sensitivity, ghi_sensitivity, wind_sensitivity, GHI_ref)
                
        load.append(value)
        
    return pd.Series(load, index=series_datetime)

def long_memory_noise(prev_noise, new_eps, alpha):
    return alpha * prev_noise + (1 - alpha) * new_eps

def add_ar_noise(load_timeserie, sigma, time_resolution, autocorrelation_memory):
    noise = np.zeros(len(load_timeserie))
    alpha = np.exp(-1/(time_resolution*autocorrelation_memory))

    for t in range(1, len(load_timeserie)):
        series_sigma = sigma * (0.5 + 0.5 * load_timeserie[t])
        eps = np.random.normal(0, series_sigma)
        noise[t] = alpha * noise[t-1] + eps

    return load_timeserie + noise

# -----------------------------
# 4. MULTIBUILDINGS MICROGRID
# -----------------------------

def generate_microgrid_load(building_list,meteoData,holidays,vacations, time_resolution):

    n_buildings = 0
    buildings = []
    names=[]
    
    for idx, building in building_list.iterrows():
        
        n_buildings += building['Number']
        profile = building["Type"]
        PROFILES[profile]["hourly_profile"] = normalize_profile(PROFILES[profile]["hourly_profile"])
        peak_kw = building["Number of users"]*PROFILES[profile]["power_per_user"]*PROFILES[profile]["simultaneity_factor"]
        autocorrelation_memory = PROFILES[profile]["autocorrelation_memory"]

        for i in range(building["Number"]):
            load_timeserie = generate_building_load(building,meteoData,holidays,vacations)
            load_timeserie = load_timeserie/max(load_timeserie)*peak_kw
            sigma_rel = 0.3
            sigma = (sigma_rel* (1 - PROFILES[profile]["simultaneity_factor"])/ np.sqrt(building["Number of users"]))
            load_timeserie = add_ar_noise(load_timeserie, sigma, time_resolution, autocorrelation_memory)
            load_timeserie = load_timeserie.clip(lower=0)
            names.append(building["Building"]+" "+str(i) if building["Number"]>1 else building["Building"])
            buildings.append(load_timeserie)

    df = pd.concat(buildings, axis=1)
    df.columns = names

    df["total_load"] = df.sum(axis=1)

    return df


