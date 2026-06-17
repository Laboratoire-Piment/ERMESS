# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 09:49:53 2026

@author: JoPHOBEA
"""

#The model parameters were selected from typical values reported in residential
#building energy simulation literature (ASHRAE, IEA EBC Annex 66, NREL, 
#IEA Heat Pump Programme) and complemented by empirical calibration parameters
#governing occupant behaviour and load aggregation. Physical parameters 
#(thermal capacity, thermal resistance, DHW characteristics, HVAC efficiencies)
#correspond to commonly reported residential building characteristics, while 
#stochastic occupancy and activity parameters were chosen to reproduce 
#realistic residential load diversity.

PROFILES = {
    "tertiary_retail" : {
        
        "occupancy_params":{        
        "vacation_factor": 0.60, #multiplicative factor applied to occupancy probability during vacation periods
        "holiday_factor": 1.20, #multiplicative factor applied to occupancy probability on public holidays
        "occupancy_correlation": 0.70, #strength of occupancy variability around the average profile (0 = deterministic, 1 = highly variable)
        "occupancy_memory": 2, #temporal persistence of occupancy states (hours)
        "occupancy_profile_weekday": {
        0:0.00,1:0.00,2:0.00,3:0.00,
        4:0.00,5:0.00,6:0.05,7:0.10,
        8:0.30,9:0.60,10:1.00,11:1.00,
        12:1.00,13:1.00,14:1.00,15:1.00,
        16:1.00,17:1.00,18:0.90,19:0.60,
        20:0.20,21:0.05,22:0.00,23:0.00
        }, #hourly probability that a user is present on weekdays
        "occupancy_profile_weekend": {
         0:0.00,1:0.00,2:0.00,3:0.00,
        4:0.00,5:0.00,6:0.05,7:0.10,
        8:0.40,9:0.80,10:1.00,11:1.00,
        12:1.00,13:1.00,14:1.00,15:1.00,
        16:1.00,17:1.00,18:1.00,19:0.80,
        20:0.30,21:0.05,22:0.00,23:0.00
        } #hourly probability that a user is present on weekends
        },
    
        "simultaneity_factor": 0.85, #correlation factor between users' electrical activities
        "mean_activity": 0.95, #average probability that a present user activates an appliance
        "activity_memory": 0.5, #temporal persistence of user activity states (hours)
        
        "base_load_params" : {
        "power_per_user": 0.015,     #average permanent load attributed to each user (kW/user)
        "noise_std": 0.06,       #relative standard deviation ratio of the stochastoc load fluctuations
        "noise_memory": 8,       #correlation time of the base-load fluctuations (hours)
        "scaling_exponent": 0.75 #aggregation exponent accounting for diversity effects (<1 reduces per-user load)
        },
        
        "light_params": {
        "GHI_threshold": 250, #solar irradiance threshold below which lighting use increases (W/m²)
        "irradiance_smoothness": 60, #controls how progressively lighting probability changes around the threshold
        "mean_light_power": 0.05, #average electrical power of a lighting event (kW)
        "std_light_power": 0.015, #standard deviation of lighting power
        },
        
        "it_params":{
        "mean_it_power": 0.15, #average IT/electronic load per active user (kW/user)
        "std_it_power": 0.05, #standard deviation of IT/electronic load per active user
        },

        "comfort_params":{
        "wind_sensitivity": 20, # heat transfer coefficient due to wind-driven infiltration (W/K/(m/s))
        "hvac_gain_per_user": 70, #thermal power demand generated per degree of discomfort and per occupant (W/°C/user)
        "hvac_COP": 3.0, #coefficient of performance of the heating system
        "hvac_EER": 3.5, #coefficient of performance of the cooling system
        "heat_saturation":12000., #maximum electrical heating power available (W)
        "cold_saturation":15000., #maximum electrical cooling power available (W)
        "solar_heat_gain": 0.55, #solar Heat Gain Coefficient (fraction of incident solar irradiance converted to indoor heat gains)
        "windows_surface": 40, #average windows surface incident to solar irradiance (m²)
        "thermal_resistance": 0.0015, #global thermal resistance of the building envelope (K/W)
        "thermal_capacity": 2e8, #effective thermal capacity of the building (J/K)
        },
        
        "shared_params":{
        "shared_installed_power_per_user": 0.40, #total installed power of shared equipment per user (kW)
        "shared_equipment_aggregation": 0.35, #aggregation exponent accounting for mutualisation of shared equipment
        "shared_scaling_exponent": 0.85, # aggregation exponent accounting for diversity in shared equipment usage
        },
        
        "EV_params" : {
        "EV_charge_rate": 0.06, #hourly probability that a user charges an electric vehicle
        "EV_power_min": 3.0, #minimum charging power (kW)
        "EV_power_max": 7.4, #maximum charging power (kW)
        "EV_charge_duration_mean": 1.5 #average charging session duration (hours)
        },
                
        "dhw_params":{
        "Tank_capacity_per_user": 5, #domestic hot water tank volume (L)
        "distribution_factor": 0.5, # aggregation exponent accounting for mutualisation of dhw capacity (0:total mutualisation , 1: each user has one tank)
        "Charging_SOC_start": 0.85, #tank state-of-charge threshold triggering reheating
        "Charging_SOC_end": 0.95, #tank state-of-charge threshold stopping reheating
        "T_cold" : 17, #cold water supply temperature (°C)
        "T_hot" : 55, #hot water setpoint temperature (°C)
        "dhw_draw_rate": 0.01, #hourly probability of a hot water draw event per user
        "dhw_mean_flow": 6.0, # average flow rate during a draw event (L/min)
        "mean_duration": 2.0, #average duration of a draw event (min)
        "heater_power": 1.5, #thermal power of the water heater (kW)
        "thermal_gain_factor": 1.0, #conversion from electrical power to thermal power (~1 for resistance heater, ~2.5-4 for heat pump)
        },
        
        },
    
    "tertiary_hospitality" : {
        
        "occupancy_params":{        
        "vacation_factor": 1.10, #multiplicative factor applied to occupancy probability during vacation periods
        "holiday_factor": 1.20, #multiplicative factor applied to occupancy probability on public holidays
        "occupancy_correlation": 0.40, #strength of occupancy variability around the average profile (0 = deterministic, 1 = highly variable)
        "occupancy_memory": 8, #temporal persistence of occupancy states (hours)
        "occupancy_profile_weekday": {
        0:0.90,1:0.95,2:1.00,3:1.00,
        4:1.00,5:1.00,6:1.00,7:0.80,
        8:0.60,9:0.50,10:0.40,11:0.40,
        12:0.50,13:0.50,14:0.40,15:0.40,
        16:0.50,17:0.60,18:0.80,19:0.90,
        20:1.00,21:1.00,22:1.00,23:0.95
        }, #hourly probability that a user is present on weekdays
        "occupancy_profile_weekend": {
         0:1.00,1:1.00,2:1.00,3:1.00,
        4:1.00,5:1.00,6:1.00,7:0.95,
        8:0.80,9:0.70,10:0.60,11:0.60,
        12:0.60,13:0.60,14:0.60,15:0.60,
        16:0.70,17:0.80,18:0.90,19:1.00,
        20:1.00,21:1.00,22:1.00,23:1.00
        } #hourly probability that a user is present on weekends
        },
        "simultaneity_factor": 0.50, #correlation factor between users' electrical activities
        "mean_activity": 0.60, #average probability that a present user activates an appliance
        "activity_memory": 2, #temporal persistence of user activity states (hours)
        
        "base_load_params" : {
        "power_per_user": 0.020,     #average permanent load attributed to each user (kW/user)
        "noise_std": 0.04,       #relative standard deviation ratio of the stochastoc load fluctuations
        "noise_memory": 12,       #correlation time of the base-load fluctuations (hours)
        "scaling_exponent": 0.65 #aggregation exponent accounting for diversity effects (<1 reduces per-user load)
        },
        
        "light_params": {
        "GHI_threshold": 180, #solar irradiance threshold below which lighting use increases (W/m²)
        "irradiance_smoothness": 40, #controls how progressively lighting probability changes around the threshold
        "mean_light_power": 0.03, #average electrical power of a lighting event (kW)
        "std_light_power": 0.010, #standard deviation of lighting power
        },
        
        "it_params":{
        "mean_it_power": 0.08, #average IT/electronic load per active user (kW/user)
        "std_it_power": 0.03, #standard deviation of IT/electronic load per active user
        },

        "comfort_params":{
        "wind_sensitivity": 15, # heat transfer coefficient due to wind-driven infiltration (W/K/(m/s))
        "hvac_gain_per_user": 60, #thermal power demand generated per degree of discomfort and per occupant (W/°C/user)
        "hvac_COP": 3.2, #coefficient of performance of the heating system
        "hvac_EER": 3.8, #coefficient of performance of the cooling system
        "heat_saturation":10000., #maximum electrical heating power available (W)
        "cold_saturation":12000., #maximum electrical cooling power available (W)
        "solar_heat_gain": 0.45, #solar Heat Gain Coefficient (fraction of incident solar irradiance converted to indoor heat gains)
        "windows_surface": 30, #average windows surface incident to solar irradiance (m²)
        "thermal_resistance": 0.002, #global thermal resistance of the building envelope (K/W)
        "thermal_capacity": 2.5e8, #effective thermal capacity of the building (J/K)
        },
        
        "shared_params":{
        "shared_installed_power_per_user": 0.70, #total installed power of shared equipment per user (kW)
        "shared_equipment_aggregation": 0.35, #aggregation exponent accounting for mutualisation of shared equipment
        "shared_scaling_exponent": 0.75, # aggregation exponent accounting for diversity in shared equipment usage
        },
        
        "EV_params" : {
        "EV_charge_rate": 0.03, #hourly probability that a user charges an electric vehicle
        "EV_power_min": 3.0, #minimum charging power (kW)
        "EV_power_max": 7.4, #maximum charging power (kW)
        "EV_charge_duration_mean": 4.0 #average charging session duration (hours)
        },
                
        "dhw_params":{
        "Tank_capacity_per_user": 100, #domestic hot water tank volume (L)
        "distribution_factor": 0.6, # aggregation exponent accounting for mutualisation of dhw capacity (0:total mutualisation , 1: each user has one tank)
        "Charging_SOC_start": 0.7, #tank state-of-charge threshold triggering reheating
        "Charging_SOC_end": 0.90, #tank state-of-charge threshold stopping reheating
        "T_cold" : 17, #cold water supply temperature (°C)
        "T_hot" : 55, #hot water setpoint temperature (°C)
        "dhw_draw_rate": 0.06, #hourly probability of a hot water draw event per user
        "dhw_mean_flow": 8.0, # average flow rate during a draw event (L/min)
        "mean_duration": 5.0, #average duration of a draw event (min)
        "heater_power": 5.0, #thermal power of the water heater (kW)
        "thermal_gain_factor": 1.0, #conversion from electrical power to thermal power (~1 for resistance heater, ~2.5-4 for heat pump)
        },
        
        },
    
    "tertiary_food_service" : {
        
        "occupancy_params":{        
        "vacation_factor": 0.80, #multiplicative factor applied to occupancy probability during vacation periods
        "holiday_factor": 1.20, #multiplicative factor applied to occupancy probability on public holidays
        "occupancy_correlation": 0.75, #strength of occupancy variability around the average profile (0 = deterministic, 1 = highly variable)
        "occupancy_memory": 2, #temporal persistence of occupancy states (hours)
        "occupancy_profile_weekday": {
        0:0.00,1:0.00,2:0.00,3:0.00,
        4:0.00,5:0.00,6:0.05,7:0.10,
        8:0.20,9:0.30,10:0.50,11:0.90,
        12:1.00,13:1.00,14:0.60,15:0.30,
        16:0.30,17:0.50,18:0.80,19:1.00,
        20:1.00,21:0.80,22:0.40,23:0.10
        }, #hourly probability that a user is present on weekdays
        "occupancy_profile_weekend": {
        0:0.00,1:0.00,2:0.00,3:0.00,
        4:0.00,5:0.00,6:0.05,7:0.10,
        8:0.30,9:0.40,10:0.60,11:0.90,
        12:1.00,13:1.00,14:0.80,15:0.50,
        16:0.50,17:0.70,18:1.00,19:1.00,
        20:1.00,21:1.00,22:0.60,23:0.20
        } #hourly probability that a user is present on weekends
        },
        "simultaneity_factor": 0.90, #correlation factor between users' electrical activities
        "mean_activity": 0.98, #average probability that a present user activates an appliance
        "activity_memory": 0.5, #temporal persistence of user activity states (hours)
        
        "base_load_params" : {
        "power_per_user": 0.050,     #average permanent load attributed to each user (kW/user)
        "noise_std": 0.08,       #relative standard deviation ratio of the stochastoc load fluctuations
        "noise_memory": 6,       #correlation time of the base-load fluctuations (hours)
        "scaling_exponent": 0.85 #aggregation exponent accounting for diversity effects (<1 reduces per-user load)
        },
        
        "light_params": {
        "GHI_threshold": 300, #solar irradiance threshold below which lighting use increases (W/m²)
        "irradiance_smoothness": 70, #controls how progressively lighting probability changes around the threshold
        "mean_light_power": 0.06, #average electrical power of a lighting event (kW)
        "std_light_power": 0.020, #standard deviation of lighting power
        },
        
        "it_params":{
        "mean_it_power": 0.05, #average IT/electronic load per active user (kW/user)
        "std_it_power": 0.02, #standard deviation of IT/electronic load per active user
        },

        "comfort_params":{
        "wind_sensitivity": 10, # heat transfer coefficient due to wind-driven infiltration (W/K/(m/s))
        "hvac_gain_per_user": 90, #thermal power demand generated per degree of discomfort and per occupant (W/°C/user)
        "hvac_COP": 3.0, #coefficient of performance of the heating system
        "hvac_EER": 3.5, #coefficient of performance of the cooling system
        "heat_saturation":20000., #maximum electrical heating power available (W)
        "cold_saturation":25000., #maximum electrical cooling power available (W)
        "solar_heat_gain": 0.6, #solar Heat Gain Coefficient (fraction of incident solar irradiance converted to indoor heat gains)
        "windows_surface": 50, #average windows surface incident to solar irradiance (m²)
        "thermal_resistance": 0.0012, #global thermal resistance of the building envelope (K/W)
        "thermal_capacity": 3e8, #effective thermal capacity of the building (J/K)
        },
        
        "shared_params":{
        "shared_installed_power_per_user": 1.50, #total installed power of shared equipment per user (kW)
        "shared_equipment_aggregation": 0.30, #aggregation exponent accounting for mutualisation of shared equipment
        "shared_scaling_exponent": 0.90, # aggregation exponent accounting for diversity in shared equipment usage
        },
        
        "EV_params" : {
        "EV_charge_rate": 0.01, #hourly probability that a user charges an electric vehicle
        "EV_power_min": 3.0, #minimum charging power (kW)
        "EV_power_max": 7.4, #maximum charging power (kW)
        "EV_charge_duration_mean": 1.5 #average charging session duration (hours)
        },
                
        "dhw_params":{
        "Tank_capacity_per_user": 30, #domestic hot water tank volume (L)
        "distribution_factor": 0.4, # aggregation exponent accounting for mutualisation of dhw capacity (0:total mutualisation , 1: each user has one tank)
        "Charging_SOC_start": 0.6, #tank state-of-charge threshold triggering reheating
        "Charging_SOC_end": 0.85, #tank state-of-charge threshold stopping reheating
        "T_cold" : 17, #cold water supply temperature (°C)
        "T_hot" : 55, #hot water setpoint temperature (°C)
        "dhw_draw_rate": 0.08, #hourly probability of a hot water draw event per user
        "dhw_mean_flow": 10.0, # average flow rate during a draw event (L/min)
        "mean_duration": 6.0, #average duration of a draw event (min)
        "heater_power": 10.0, #thermal power of the water heater (kW)
        "thermal_gain_factor": 1.0, #conversion from electrical power to thermal power (~1 for resistance heater, ~2.5-4 for heat pump)
        },
        
        },
    
    "tertiary_healthcare" : {
        
        "occupancy_params":{        
        "vacation_factor": 1.0, #multiplicative factor applied to occupancy probability during vacation periods
        "holiday_factor": 1.0, #multiplicative factor applied to occupancy probability on public holidays
        "occupancy_correlation": 0.30, #strength of occupancy variability around the average profile (0 = deterministic, 1 = highly variable)
        "occupancy_memory": 12, #temporal persistence of occupancy states (hours)
        "occupancy_profile_weekday": {
        0:0.70,1:0.70,2:0.70,3:0.70,
        4:0.70,5:0.75,6:0.80,7:0.90,
        8:1.00,9:1.00,10:1.00,11:1.00,
        12:1.00,13:1.00,14:1.00,15:1.00,
        16:1.00,17:0.95,18:0.90,19:0.85,
        20:0.80,21:0.80,22:0.75,23:0.75
        }, #hourly probability that a user is present on weekdays
        "occupancy_profile_weekend": {
        0:0.70,1:0.70,2:0.70,3:0.70,
        4:0.70,5:0.75,6:0.80,7:0.85,
        8:0.90,9:0.90,10:0.90,11:0.90,
        12:0.90,13:0.90,14:0.90,15:0.90,
        16:0.90,17:0.90,18:0.85,19:0.85,
        20:0.80,21:0.80,22:0.75,23:0.75
        } #hourly probability that a user is present on weekends
        },
        "simultaneity_factor": 0.70, #correlation factor between users' electrical activities
        "mean_activity": 0.85, #average probability that a present user activates an appliance
        "activity_memory": 6, #temporal persistence of user activity states (hours)
        
        "base_load_params" : {
        "power_per_user": 0.080,     #average permanent load attributed to each user (kW/user)
        "noise_std": 0.03,       #relative standard deviation ratio of the stochastoc load fluctuations
        "noise_memory": 48,       #correlation time of the base-load fluctuations (hours)
        "scaling_exponent": 0.60 #aggregation exponent accounting for diversity effects (<1 reduces per-user load)
        },
        
        "light_params": {
        "GHI_threshold": 150, #solar irradiance threshold below which lighting use increases (W/m²)
        "irradiance_smoothness": 30, #controls how progressively lighting probability changes around the threshold
        "mean_light_power": 0.04, #average electrical power of a lighting event (kW)
        "std_light_power": 0.010, #standard deviation of lighting power
        },
        
        "it_params":{
        "mean_it_power": 0.25, #average IT/electronic load per active user (kW/user)
        "std_it_power": 0.1, #standard deviation of IT/electronic load per active user
        },

        "comfort_params":{
        "wind_sensitivity": 25, # heat transfer coefficient due to wind-driven infiltration (W/K/(m/s))
        "hvac_gain_per_user": 120, #thermal power demand generated per degree of discomfort and per occupant (W/°C/user)
        "hvac_COP": 3.5, #coefficient of performance of the heating system
        "hvac_EER": 4.0, #coefficient of performance of the cooling system
        "heat_saturation":30000., #maximum electrical heating power available (W)
        "cold_saturation":35000., #maximum electrical cooling power available (W)
        "solar_heat_gain": 0.4, #solar Heat Gain Coefficient (fraction of incident solar irradiance converted to indoor heat gains)
        "windows_surface": 20, #average windows surface incident to solar irradiance (m²)
        "thermal_resistance": 0.001, #global thermal resistance of the building envelope (K/W)
        "thermal_capacity": 5e8, #effective thermal capacity of the building (J/K)
        },
        
        "shared_params":{
        "shared_installed_power_per_user": 2.50, #total installed power of shared equipment per user (kW)
        "shared_equipment_aggregation": 0.3, #aggregation exponent accounting for mutualisation of shared equipment
        "shared_scaling_exponent": 0.95, # aggregation exponent accounting for diversity in shared equipment usage
        },
        
        "EV_params" : {
        "EV_charge_rate": 0.02, #hourly probability that a user charges an electric vehicle
        "EV_power_min": 3.0, #minimum charging power (kW)
        "EV_power_max": 7.4, #maximum charging power (kW)
        "EV_charge_duration_mean": 3.0 #average charging session duration (hours)
        },
                
        "dhw_params":{
        "Tank_capacity": 80, #domestic hot water tank volume (L)
        "Charging_SOC_start": 0.4, #tank state-of-charge threshold triggering reheating
        "Charging_SOC_end": 0.95, #tank state-of-charge threshold stopping reheating
        "T_cold" : 17, #cold water supply temperature (°C)
        "T_hot" : 55, #hot water setpoint temperature (°C)
        "dhw_draw_rate": 0.05, #hourly probability of a hot water draw event per user
        "dhw_mean_flow": 8.0, # average flow rate during a draw event (L/min)
        "mean_duration": 5.0, #average duration of a draw event (min)
        "heater_power": 15.0, #thermal power of the water heater (kW)
        "thermal_gain_factor": 1.0, #conversion from electrical power to thermal power (~1 for resistance heater, ~2.5-4 for heat pump)
        },
        
        },
    
    "tertiary_public_services" : {
        
        "occupancy_params":{        
        "vacation_factor": 0.50, #multiplicative factor applied to occupancy probability during vacation periods
        "holiday_factor": 0.10, #multiplicative factor applied to occupancy probability on public holidays
        "occupancy_correlation": 0.60, #strength of occupancy variability around the average profile (0 = deterministic, 1 = highly variable)
        "occupancy_memory": 3, #temporal persistence of occupancy states (hours)
        "occupancy_profile_weekday": {
        0:0.00,1:0.00,2:0.00,3:0.00,
        4:0.00,5:0.00,6:0.05,7:0.20,
        8:0.80,9:1.00,10:1.00,11:1.00,
        12:0.80,13:0.80,14:1.00,15:1.00,
        16:0.80,17:0.30,18:0.05,19:0.00,
        20:0.00,21:0.00,22:0.00,23:0.00
        }, #hourly probability that a user is present on weekdays
        "occupancy_profile_weekend": {
        0:0.00,1:0.00,2:0.00,3:0.00,
        4:0.00,5:0.00,6:0.00,7:0.00,
        8:0.00,9:0.00,10:0.00,11:0.00,
        12:0.00,13:0.00,14:0.00,15:0.00,
        16:0.00,17:0.00,18:0.00,19:0.00,
        20:0.00,21:0.00,22:0.00,23:0.00
        } #hourly probability that a user is present on weekends
        },
        "simultaneity_factor": 0.60, #correlation factor between users' electrical activities
        "mean_activity": 0.70, #average probability that a present user activates an appliance
        "activity_memory": 2, #temporal persistence of user activity states (hours)
        
        "base_load_params" : {
        "power_per_user": 0.010,     #average permanent load attributed to each user (kW/user)
        "noise_std": 0.03,       #relative standard deviation ratio of the stochastoc load fluctuations
        "noise_memory": 24,       #correlation time of the base-load fluctuations (hours)
        "scaling_exponent": 0.70 #aggregation exponent accounting for diversity effects (<1 reduces per-user load)
        },
        
        "light_params": {
        "GHI_threshold": 200, #solar irradiance threshold below which lighting use increases (W/m²)
        "irradiance_smoothness": 50, #controls how progressively lighting probability changes around the threshold
        "mean_light_power": 0.03, #average electrical power of a lighting event (kW)
        "std_light_power": 0.008, #standard deviation of lighting power
        },
        
        "it_params":{
        "mean_it_power": 0.1, #average IT/electronic load per active user (kW/user)
        "std_it_power": 0.04, #standard deviation of IT/electronic load per active user
        },

        "comfort_params":{
        "wind_sensitivity": 20, # heat transfer coefficient due to wind-driven infiltration (W/K/(m/s))
        "hvac_gain_per_user": 70, #thermal power demand generated per degree of discomfort and per occupant (W/°C/user)
        "hvac_COP": 3.0, #coefficient of performance of the heating system
        "hvac_EER": 3.5, #coefficient of performance of the cooling system
        "heat_saturation":12000., #maximum electrical heating power available (W)
        "cold_saturation":15000., #maximum electrical cooling power available (W)
        "solar_heat_gain": 0.5, #solar Heat Gain Coefficient (fraction of incident solar irradiance converted to indoor heat gains)
        "windows_surface": 25, #average windows surface incident to solar irradiance (m²)
        "thermal_resistance": 0.0018, #global thermal resistance of the building envelope (K/W)
        "thermal_capacity": 2e8, #effective thermal capacity of the building (J/K)
        },
        
        "shared_params":{
        "shared_installed_power_per_user": 0.20, #total installed power of shared equipment per user (kW)
        "shared_equipment_aggregation": 0.40, #aggregation exponent accounting for mutualisation of shared equipment
        "shared_scaling_exponent": 0.80, # aggregation exponent accounting for diversity in shared equipment usage
        },
        
        "EV_params" : {
        "EV_charge_rate": 0.02, #hourly probability that a user charges an electric vehicle
        "EV_power_min": 3.0, #minimum charging power (kW)
        "EV_power_max": 7.4, #maximum charging power (kW)
        "EV_charge_duration_mean": 2.0 #average charging session duration (hours)
        },
                
        "dhw_params":{
        "Tank_capacity_per_user": 8, #domestic hot water tank volume (L)
        "distribution_factor": 0.5, # aggregation exponent accounting for mutualisation of dhw capacity (0:total mutualisation , 1: each user has one tank)
        "Charging_SOC_start": 0.8, #tank state-of-charge threshold triggering reheating
        "Charging_SOC_end": 0.95, #tank state-of-charge threshold stopping reheating
        "T_cold" : 17, #cold water supply temperature (°C)
        "T_hot" : 55, #hot water setpoint temperature (°C)
        "dhw_draw_rate": 0.01, #hourly probability of a hot water draw event per user
        "dhw_mean_flow": 6.0, # average flow rate during a draw event (L/min)
        "mean_duration": 3.0, #average duration of a draw event (min)
        "heater_power": 2.0, #thermal power of the water heater (kW)
        "thermal_gain_factor": 1.0, #conversion from electrical power to thermal power (~1 for resistance heater, ~2.5-4 for heat pump)
        },
        
        },
    
    "residential" : {
        
        "occupancy_params":{        
        "vacation_factor": 0.20, #multiplicative factor applied to occupancy probability during vacation periods
        "holiday_factor": 1.05, #multiplicative factor applied to occupancy probability on public holidays
        "occupancy_correlation": 0.8, #strength of occupancy variability around the average profile (0 = deterministic, 1 = highly variable)
        "occupancy_memory": 4, #temporal persistence of occupancy states (hours)
        "occupancy_profile_weekday": {
            0: 0.30, 1: 0.30, 2: 0.30, 3: 0.30,
            4: 0.30, 5: 0.30, 6: 0.30, 7: 1.00,
            8: 1.00, 9: 0.30, 10: 0.30, 11: 0.30,
            12: 0.30, 13: 0.30,
            14: 0.30, 15: 0.30, 16: 0.30, 17: 0.30,
            18: 1.00, 19: 1.00,
            20: 1.00, 21: 1.00, 22: 1.00, 23: 0.30
        }, #hourly probability that a user is present on weekdays
        "occupancy_profile_weekend": {
            0: 0.30, 1: 0.30, 2: 0.30, 3: 0.30,
            4: 0.30, 5: 0.30, 6: 0.30, 7: 1.00,
            8: 1.00, 9: 0.30, 10: 0.30, 11: 0.30,
            12: 0.30, 13: 0.30,
            14: 0.30, 15: 0.30, 16: 0.30, 17: 0.30,
            18: 1.00, 19: 1.00,
            20: 1.00, 21: 1.00, 22: 1.00, 23: 0.30
        } #hourly probability that a user is present on weekends
        },

        "simultaneity_factor": 0.40, #correlation factor between users' electrical activities
        "mean_activity": 0.75, #average probability that a present user activates an appliance
        "activity_memory": 1, #temporal persistence of user activity states (hours)
        
        "base_load_params" : {
        "power_per_user": 0.005,     #average permanent load attributed to each user (kW/user)
        "noise_std": 0.02,       #relative standard deviation ratio of the stochastoc load fluctuations
        "noise_memory": 24,       #correlation time of the base-load fluctuations (hours)
        "scaling_exponent": 0.7 #aggregation exponent accounting for diversity effects (<1 reduces per-user load)
        },
        
        "light_params": {
        "GHI_threshold": 200, #solar irradiance threshold below which lighting use increases (W/m²)
        "irradiance_smoothness": 50, #controls how progressively lighting probability changes around the threshold
        "mean_light_power": 0.02, #average electrical power of a lighting event (kW)
        "std_light_power": 0.005, #standard deviation of lighting power
        },
        
        "it_params":{
        "mean_it_power": 0.15, #average IT/electronic load per active user (kW/user)
        "std_it_power": 0.06, #standard deviation of IT/electronic load per active user
        },

        "comfort_params":{
        "wind_sensitivity": 25, # heat transfer coefficient due to wind-driven infiltration (W/K/(m/s))
        "hvac_gain_per_user": 50, #thermal power demand generated per degree of discomfort and per occupant (W/°C/user)
        "hvac_COP": 3.0, #coefficient of performance of the heating system
        "hvac_EER": 3.5, #coefficient of performance of the cooling system
        "heat_saturation":8000., #maximum electrical heating power available (W)
        "cold_saturation":8000., #maximum electrical cooling power available (W)
        "solar_heat_gain": 0.5, #solar Heat Gain Coefficient (fraction of incident solar irradiance converted to indoor heat gains)
        "windows_surface": 20, #average windows surface incident to solar irradiance (m²)
        "thermal_resistance": 0.002, #global thermal resistance of the building envelope (K/W)
        "thermal_capacity": 1e8, #effective thermal capacity of the building (J/K)
        },
        
        "shared_params":{
        "shared_installed_power_per_user": 0.15, #total installed power of shared equipment per user (kW)
        "shared_equipment_aggregation": 0.55, #aggregation exponent accounting for mutualisation of shared equipment
        "shared_scaling_exponent": 0.7, # aggregation exponent accounting for diversity in shared equipment usage
        },
        
        "EV_params" : {
        "EV_charge_rate": 0.08, #hourly probability that a user charges an electric vehicle
        "EV_power_min": 3.0, #minimum charging power (kW)
        "EV_power_max": 7.0, #maximum charging power (kW)
        "EV_charge_duration_mean": 3.5 #average charging session duration (hours)
        },
                
        "dhw_params":{
        "Tank_capacity_per_user": 60, #domestic hot water tank volume (L)
        "distribution_factor": 0.7, # aggregation exponent accounting for mutualisation of dhw capacity (0:total mutualisation , 1: each user has one tank)
        "Charging_SOC_start": 0.8, #tank state-of-charge threshold triggering reheating
        "Charging_SOC_end": 0.95, #tank state-of-charge threshold stopping reheating
        "T_cold" : 17, #cold water supply temperature (°C)
        "T_hot" : 55, #hot water setpoint temperature (°C)
        "dhw_draw_rate": 0.03, #hourly probability of a hot water draw event per user
        "dhw_mean_flow": 6.0, # average flow rate during a draw event (L/min)
        "mean_duration": 6.0, #average duration of a draw event (min)
        "heater_power": 2.0, #thermal power of the water heater (kW)
        "thermal_gain_factor": 1.0, #conversion from electrical power to thermal power (~1 for resistance heater, ~2.5-4 for heat pump)
        },
        },

    "teaching": {

        "occupancy_params":{        
        "vacation_factor": 0.10, #multiplicative factor applied to occupancy probability during vacation periods
        "holiday_factor": 0.05, #multiplicative factor applied to occupancy probability on public holidays
        "occupancy_correlation": 0.75, #strength of occupancy variability around the average profile (0 = deterministic, 1 = highly variable)
        "occupancy_memory": 2, #temporal persistence of occupancy states (hours)
        "occupancy_profile_weekday": {
         0:0.00,1:0.00,2:0.00,3:0.00,
        4:0.00,5:0.00,6:0.05,7:0.30,
        8:1.00,9:1.00,10:1.00,11:1.00,
        12:0.80,13:0.80,14:1.00,15:1.00,
        16:0.80,17:0.20,18:0.05,19:0.00,
        20:0.00,21:0.00,22:0.00,23:0.00
        }, #hourly probability that a user is present on weekdays
        "occupancy_profile_weekend": {
        0:0.00,1:0.00,2:0.00,3:0.00,
        4:0.00,5:0.00,6:0.00,7:0.00,
        8:0.05,9:0.10,10:0.10,11:0.10,
        12:0.10,13:0.10,14:0.10,15:0.10,
        16:0.05,17:0.00,18:0.00,19:0.00,
        20:0.00,21:0.00,22:0.00,23:0.00
        } #hourly probability that a user is present on weekends
        },
        "simultaneity_factor": 0.85, #correlation factor between users' electrical activities
        "mean_activity": 0.90, #average probability that a present user activates an appliance
        "activity_memory": 2, #temporal persistence of user activity states (hours)
        
        "base_load_params" : {
        "power_per_user": 0.020,     #average permanent load attributed to each user (kW/user)
        "noise_std": 0.05,       #relative standard deviation ratio of the stochastoc load fluctuations
        "noise_memory": 12,       #correlation time of the base-load fluctuations (hours)
        "scaling_exponent": 0.80 #aggregation exponent accounting for diversity effects (<1 reduces per-user load)
        },
        
        "light_params": {
        "GHI_threshold": 220, #solar irradiance threshold below which lighting use increases (W/m²)
        "irradiance_smoothness": 55, #controls how progressively lighting probability changes around the threshold
        "mean_light_power": 0.035, #average electrical power of a lighting event (kW)
        "std_light_power": 0.010, #standard deviation of lighting power
        },
        
        "it_params":{
        "mean_it_power": 0.2, #average IT/electronic load per active user (kW/user)
        "std_it_power": 0.08, #standard deviation of IT/electronic load per active user
        },

        "comfort_params":{
        "wind_sensitivity": 20, # heat transfer coefficient due to wind-driven infiltration (W/K/(m/s))
        "hvac_gain_per_user": 80, #thermal power demand generated per degree of discomfort and per occupant (W/°C/user)
        "hvac_COP": 3.0, #coefficient of performance of the heating system
        "hvac_EER": 3.5, #coefficient of performance of the cooling system
        "heat_saturation":20000., #maximum electrical heating power available (W)
        "cold_saturation":20000., #maximum electrical cooling power available (W)
        "solar_heat_gain": 0.55, #solar Heat Gain Coefficient (fraction of incident solar irradiance converted to indoor heat gains)
        "windows_surface": 40, #average windows surface incident to solar irradiance (m²)
        "thermal_resistance": 0.0015, #global thermal resistance of the building envelope (K/W)
        "thermal_capacity": 3e8, #effective thermal capacity of the building (J/K)
        },
        
        "shared_params":{
        "shared_installed_power_per_user": 0.18, #total installed power of shared equipment per user (kW)
        "shared_equipment_aggregation": 0.40, #aggregation exponent accounting for mutualisation of shared equipment
        "shared_scaling_exponent": 0.80, # aggregation exponent accounting for diversity in shared equipment usage
        },
        
        "EV_params" : {
        "EV_charge_rate": 0.0, #hourly probability that a user charges an electric vehicle
        "EV_power_min": 3.0, #minimum charging power (kW)
        "EV_power_max": 7.4, #maximum charging power (kW)
        "EV_charge_duration_mean": 2.0 #average charging session duration (hours)
        },
                
        "dhw_params":{
        "Tank_capacity_per_user": 5, #domestic hot water tank volume (L)
        "distribution_factor": 0.5, # aggregation exponent accounting for mutualisation of dhw capacity (0:total mutualisation , 1: each user has one tank)
        "Charging_SOC_start": 0.8, #tank state-of-charge threshold triggering reheating
        "Charging_SOC_end": 0.95, #tank state-of-charge threshold stopping reheating
        "T_cold" : 17, #cold water supply temperature (°C)
        "T_hot" : 55, #hot water setpoint temperature (°C)
        "dhw_draw_rate": 0.01, #hourly probability of a hot water draw event per user
        "dhw_mean_flow": 6.0, # average flow rate during a draw event (L/min)
        "mean_duration": 3.0, #average duration of a draw event (min)
        "heater_power": 2.0, #thermal power of the water heater (kW)
        "thermal_gain_factor": 1.0, #conversion from electrical power to thermal power (~1 for resistance heater, ~2.5-4 for heat pump)
        },
        
        },

    "office": {

        "occupancy_params":{        
        "vacation_factor": 0.50, #multiplicative factor applied to occupancy probability during vacation periods
        "holiday_factor": 0.10, #multiplicative factor applied to occupancy probability on public holidays
        "occupancy_correlation": 0.65, #strength of occupancy variability around the average profile (0 = deterministic, 1 = highly variable)
        "occupancy_memory": 3, #temporal persistence of occupancy states (hours)
        "occupancy_profile_weekday": {
        0:0.00,1:0.00,2:0.00,3:0.00,
        4:0.00,5:0.00,6:0.05,7:0.20,
        8:0.80,9:1.00,10:1.00,11:1.00,
        12:0.90,13:0.90,14:1.00,15:1.00,
        16:0.90,17:0.60,18:0.20,19:0.05,
        20:0.00,21:0.00,22:0.00,23:0.00
        }, #hourly probability that a user is present on weekdays
        "occupancy_profile_weekend": {
        0:0.00,1:0.00,2:0.00,3:0.00,
        4:0.00,5:0.00,6:0.00,7:0.00,
        8:0.05,9:0.10,10:0.10,11:0.10,
        12:0.10,13:0.10,14:0.10,15:0.10,
        16:0.05,17:0.00,18:0.00,19:0.00,
        20:0.00,21:0.00,22:0.00,23:0.00
        } #hourly probability that a user is present on weekends
        },
        "simultaneity_factor": 0.65, #correlation factor between users' electrical activities
        "mean_activity": 0.80, #average probability that a present user activates an appliance
        "activity_memory": 2, #temporal persistence of user activity states (hours)
        
        "base_load_params" : {
        "power_per_user": 0.030,     #average permanent load attributed to each user (kW/user)
        "noise_std": 0.04,       #relative standard deviation ratio of the stochastoc load fluctuations
        "noise_memory": 24,       #correlation time of the base-load fluctuations (hours)
        "scaling_exponent": 0.70 #aggregation exponent accounting for diversity effects (<1 reduces per-user load)
        },
        
        "light_params": {
        "GHI_threshold": 200, #solar irradiance threshold below which lighting use increases (W/m²)
        "irradiance_smoothness": 50, #controls how progressively lighting probability changes around the threshold
        "mean_light_power": 0.030, #average electrical power of a lighting event (kW)
        "std_light_power": 0.008, #standard deviation of lighting power
        },
        
        "it_params":{
        "mean_it_power": 0.18, #average IT/electronic load per active user (kW/user)
        "std_it_power": 0.07, #standard deviation of IT/electronic load per active user
        },

        "comfort_params":{
        "wind_sensitivity": 20, # heat transfer coefficient due to wind-driven infiltration (W/K/(m/s))
        "hvac_gain_per_user": 70, #thermal power demand generated per degree of discomfort and per occupant (W/°C/user)
        "hvac_COP": 3.2, #coefficient of performance of the heating system
        "hvac_EER": 3.8, #coefficient of performance of the cooling system
        "heat_saturation":15000., #maximum electrical heating power available (W)
        "cold_saturation":18000., #maximum electrical cooling power available (W)
        "solar_heat_gain": 0.5, #solar Heat Gain Coefficient (fraction of incident solar irradiance converted to indoor heat gains)
        "windows_surface": 30, #average windows surface incident to solar irradiance (m²)
        "thermal_resistance": 0.0018, #global thermal resistance of the building envelope (K/W)
        "thermal_capacity": 2.5e8, #effective thermal capacity of the building (J/K)
        },
        
        "shared_params":{
        "shared_installed_power_per_user": 0.12, #total installed power of shared equipment per user (kW)
        "shared_equipment_aggregation": 0.40, #aggregation exponent accounting for mutualisation of shared equipment
        "shared_scaling_exponent": 0.80, # aggregation exponent accounting for diversity in shared equipment usage
        },
        
        "EV_params" : {
        "EV_charge_rate": 0.04, #hourly probability that a user charges an electric vehicle
        "EV_power_min": 3.0, #minimum charging power (kW)
        "EV_power_max": 7.4, #maximum charging power (kW)
        "EV_charge_duration_mean": 2.0 #average charging session duration (hours)
        },
                
        "dhw_params":{
        "Tank_capacity_per_user": 3, #domestic hot water tank volume (L)
        "distribution_factor": 0.5, # aggregation exponent accounting for mutualisation of dhw capacity (0:total mutualisation , 1: each user has one tank)
        "Charging_SOC_start": 0.8, #tank state-of-charge threshold triggering reheating
        "Charging_SOC_end": 0.95, #tank state-of-charge threshold stopping reheating
        "T_cold" : 17, #cold water supply temperature (°C)
        "T_hot" : 55, #hot water setpoint temperature (°C)
        "dhw_draw_rate": 0.01, #hourly probability of a hot water draw event per user
        "dhw_mean_flow": 6.0, # average flow rate during a draw event (L/min)
        "mean_duration": 3.0, #average duration of a draw event (min)
        "heater_power": 2.0, #thermal power of the water heater (kW)
        "thermal_gain_factor": 1.0, #conversion from electrical power to thermal power (~1 for resistance heater, ~2.5-4 for heat pump)
        },
        
        },

    "lab": {

        "occupancy_params":{        
        "vacation_factor": 0.60, #multiplicative factor applied to occupancy probability during vacation periods
        "holiday_factor": 0.70, #multiplicative factor applied to occupancy probability on public holidays
        "occupancy_correlation": 0.50, #strength of occupancy variability around the average profile (0 = deterministic, 1 = highly variable)
        "occupancy_memory": 6, #temporal persistence of occupancy states (hours)
        "occupancy_profile_weekday": {
         0:0.10,1:0.10,2:0.10,3:0.10,
        4:0.10,5:0.10,6:0.20,7:0.40,
        8:0.80,9:1.00,10:1.00,11:1.00,
        12:0.90,13:0.90,14:1.00,15:1.00,
        16:1.00,17:0.90,18:0.70,19:0.50,
        20:0.40,21:0.30,22:0.20,23:0.10
        }, #hourly probability that a user is present on weekdays
        "occupancy_profile_weekend": {
        0:0.10,1:0.10,2:0.10,3:0.10,
        4:0.10,5:0.10,6:0.10,7:0.10,
        8:0.20,9:0.30,10:0.30,11:0.30,
        12:0.30,13:0.30,14:0.30,15:0.30,
        16:0.30,17:0.20,18:0.20,19:0.20,
        20:0.20,21:0.10,22:0.10,23:0.10
        } #hourly probability that a user is present on weekends
        },
        "simultaneity_factor": 0.80, #correlation factor between users' electrical activities
        "mean_activity": 0.95, #average probability that a present user activates an appliance
        "activity_memory": 8, #temporal persistence of user activity states (hours)
        
        "base_load_params" : {
        "power_per_user": 0.120,     #average permanent load attributed to each user (kW/user)
        "noise_std": 0.10,       #relative standard deviation ratio of the stochastoc load fluctuations
        "noise_memory": 48,       #correlation time of the base-load fluctuations (hours)
        "scaling_exponent": 0.90 #aggregation exponent accounting for diversity effects (<1 reduces per-user load)
        },
        
        "light_params": {
        "GHI_threshold": 150, #solar irradiance threshold below which lighting use increases (W/m²)
        "irradiance_smoothness": 30, #controls how progressively lighting probability changes around the threshold
        "mean_light_power": 0.050, #average electrical power of a lighting event (kW)
        "std_light_power": 0.015, #standard deviation of lighting power
        },
        
        "it_params":{
        "mean_it_power": 0.6, #average IT/electronic load per active user (kW/user)
        "std_it_power": 0.25, #standard deviation of IT/electronic load per active user
        },

        "comfort_params":{
        "wind_sensitivity": 30, # heat transfer coefficient due to wind-driven infiltration (W/K/(m/s))
        "hvac_gain_per_user": 150, #thermal power demand generated per degree of discomfort and per occupant (W/°C/user)
        "hvac_COP": 3.5, #coefficient of performance of the heating system
        "hvac_EER": 4, #coefficient of performance of the cooling system
        "heat_saturation":50000., #maximum electrical heating power available (W)
        "cold_saturation":60000., #maximum electrical cooling power available (W)
        "solar_heat_gain": 0.35, #solar Heat Gain Coefficient (fraction of incident solar irradiance converted to indoor heat gains)
        "windows_surface": 20, #average windows surface incident to solar irradiance (m²)
        "thermal_resistance": 0.001, #global thermal resistance of the building envelope (K/W)
        "thermal_capacity": 6e8, #effective thermal capacity of the building (J/K)
        },
        
        "shared_params":{
        "shared_installed_power_per_user": 1.5, #total installed power of shared equipment per user (kW)
        "shared_equipment_aggregation": 0.50, #aggregation exponent accounting for mutualisation of shared equipment
        "shared_scaling_exponent": 0.95, # aggregation exponent accounting for diversity in shared equipment usage
        },
        
        "EV_params" : {
        "EV_charge_rate": 0.01, #hourly probability that a user charges an electric vehicle
        "EV_power_min": 3.0, #minimum charging power (kW)
        "EV_power_max": 7.4, #maximum charging power (kW)
        "EV_charge_duration_mean": 2.0 #average charging session duration (hours)
        },
                
        "dhw_params":{
        "Tank_capacity_per_user": 10, #domestic hot water tank volume (L)
        "distribution_factor": 0.7, # aggregation exponent accounting for mutualisation of dhw capacity (0:total mutualisation , 1: each user has one tank)
        "Charging_SOC_start": 0.85, #tank state-of-charge threshold triggering reheating
        "Charging_SOC_end": 0.95, #tank state-of-charge threshold stopping reheating
        "T_cold" : 17, #cold water supply temperature (°C)
        "T_hot" : 55, #hot water setpoint temperature (°C)
        "dhw_draw_rate": 0.02, #hourly probability of a hot water draw event per user
        "dhw_mean_flow": 6.0, # average flow rate during a draw event (L/min)
        "mean_duration": 3.0, #average duration of a draw event (min)
        "heater_power": 3.0, #thermal power of the water heater (kW)
        "thermal_gain_factor": 1.0, #conversion from electrical power to thermal power (~1 for resistance heater, ~2.5-4 for heat pump)
        },
        
        
        },

    "dormitory": {

        "occupancy_params":{        
        "vacation_factor": 0.30, #multiplicative factor applied to occupancy probability during vacation periods
        "holiday_factor": 1.10, #multiplicative factor applied to occupancy probability on public holidays
        "occupancy_correlation": 0.70, #strength of occupancy variability around the average profile (0 = deterministic, 1 = highly variable)
        "occupancy_memory": 6, #temporal persistence of occupancy states (hours)
        "occupancy_profile_weekday": {
        0:1.00,1:1.00,2:1.00,3:1.00,
        4:1.00,5:1.00,6:1.00,7:0.80,
        8:0.40,9:0.20,10:0.20,11:0.20,
        12:0.30,13:0.30,14:0.20,15:0.20,
        16:0.30,17:0.50,18:0.80,19:1.00,
        20:1.00,21:1.00,22:1.00,23:1.00
        }, #hourly probability that a user is present on weekdays
        "occupancy_profile_weekend": {
        0:1.00,1:1.00,2:1.00,3:1.00,
        4:1.00,5:1.00,6:1.00,7:1.00,
        8:0.90,9:0.80,10:0.70,11:0.70,
        12:0.70,13:0.70,14:0.70,15:0.70,
        16:0.80,17:0.90,18:1.00,19:1.00,
        20:1.00,21:1.00,22:1.00,23:1.00
        } #hourly probability that a user is present on weekends
        },
        "simultaneity_factor": 0.45, #correlation factor between users' electrical activities
        "mean_activity": 0.70, #average probability that a present user activates an appliance
        "activity_memory": 3, #temporal persistence of user activity states (hours)
    
        "base_load_params" : {
        "power_per_user": 0.250,     #average permanent load attributed to each user (kW/user)
        "noise_std": 0.05,       #relative standard deviation ratio of the stochastoc load fluctuations
        "noise_memory": 6,       #correlation time of the base-load fluctuations (hours)
        "scaling_exponent": 0.60 #aggregation exponent accounting for diversity effects (<1 reduces per-user load)
        },
        
        "light_params": {
        "GHI_threshold": 250, #solar irradiance threshold below which lighting use increases (W/m²)
        "irradiance_smoothness": 60, #controls how progressively lighting probability changes around the threshold
        "mean_light_power": 0.025, #average electrical power of a lighting event (kW)
        "std_light_power": 0.008, #standard deviation of lighting power
        },
        
        "it_params":{
        "mean_it_power": 0.12, #average IT/electronic load per active user (kW/user)
        "std_it_power": 0.05, #standard deviation of IT/electronic load per active user
        },

        "comfort_params":{
        "wind_sensitivity": 15, # heat transfer coefficient due to wind-driven infiltration (W/K/(m/s))
        "hvac_gain_per_user": 50, #thermal power demand generated per degree of discomfort and per occupant (W/°C/user)
        "hvac_COP": 3.0, #coefficient of performance of the heating system
        "hvac_EER": 3.5, #coefficient of performance of the cooling system
        "heat_saturation":10000., #maximum electrical heating power available (W)
        "cold_saturation":12000., #maximum electrical cooling power available (W)
        "solar_heat_gain": 0.45, #solar Heat Gain Coefficient (fraction of incident solar irradiance converted to indoor heat gains)
        "windows_surface": 25, #average windows surface incident to solar irradiance (m²)
        "thermal_resistance": 0.002, #global thermal resistance of the building envelope (K/W)
        "thermal_capacity": 2.5e8, #effective thermal capacity of the building (J/K)
        },
        
        "shared_params":{
        "shared_installed_power_per_user": 0.20, #total installed power of shared equipment per user (kW)
        "shared_equipment_aggregation": 0.45, #aggregation exponent accounting for mutualisation of shared equipment
        "shared_scaling_exponent": 0.75, # aggregation exponent accounting for diversity in shared equipment usage
        },
        
        "EV_params" : {
        "EV_charge_rate": 0.02, #hourly probability that a user charges an electric vehicle
        "EV_power_min": 3.0, #minimum charging power (kW)
        "EV_power_max": 7.0, #maximum charging power (kW)
        "EV_charge_duration_mean": 2.5 #average charging session duration (hours)
        },
                
        "dhw_params":{
        "Tank_capacity_per_user": 70, #domestic hot water tank volume (L)
        "distribution_factor": 1., # aggregation exponent accounting for mutualisation of dhw capacity (0:total mutualisation , 1: each user has one tank)
        "Charging_SOC_start": 0.8, #tank state-of-charge threshold triggering reheating
        "Charging_SOC_end": 0.95, #tank state-of-charge threshold stopping reheating
        "T_cold" : 17, #cold water supply temperature (°C)
        "T_hot" : 55, #hot water setpoint temperature (°C)
        "dhw_draw_rate": 0.05, #hourly probability of a hot water draw event per user
        "dhw_mean_flow": 6.0, # average flow rate during a draw event (L/min)
        "mean_duration": 4.0, #average duration of a draw event (min)
        "heater_power": 3.0, #thermal power of the water heater (kW)
        "thermal_gain_factor": 1.0, #conversion from electrical power to thermal power (~1 for resistance heater, ~2.5-4 for heat pump)
        },
        
        }
        
  }          