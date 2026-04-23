# -*- coding:utf-8 -*-
# Created on 2025-05-23 15:37:52
# Projet: BDD-Labo
## @package virtualPMS\Battery.py
# @brief Single battery definition. 

"""
Created on Tue Jun 10 14:03:31 2025

@author: JoPHOBEA

Routines that simulate charge and discharge of storage systems.

"""


import numpy as np
from numba import jit

@jit(nopython=True)
def battery_charge(params, SOC_eff, taking_over, power: float, time_resolution: float) -> float:
        """
        Computes admissible battery charging power under operational constraints.
        
        This function is JIT-compiled using Numba for high-performance execution
        inside the ERMESS evolutionary optimisation loop.
        
        The charging power is limited by:
            - Available surplus power,
            - Maximum charging power,
            - Remaining storage capacity,
            - SOC-dependent participation factor (taking-over).
        
        Args:
            params (numpy.ndarray): Battery parameter array where:
                - params[0] = capacity (kWh)
                - params[1] = maximum charging power (kW)
                
            SOC_eff (float): Effective state of charge (0–1).
            taking_over (numpy.ndarray): SOC-dependent participation control vector.
            power (float): Available charging power (kW).
            time_resolution (float): Simulation time step (hours).
        
        Returns:
            float: Effective charging power applied to the battery (kW).
        
        Notes:
            - The SOC is discretised into 10 levels to determine the participation factor.
            - Available energy is reduced by the round-trip efficiency to account for conversion losses.
        
        Warning:
            No internal SOC update is performed here. The state of charge must be updated consistently after computing the power flow.
            
        """
        e_needed = (1 - SOC_eff) * params[0] # energy needed to fully charge the battery
        closest_level = min(8,max(0,9-np.int64(10*SOC_eff)))
        participation = 1-taking_over[closest_level]
        P_ch = min(power*participation, params[1], e_needed * time_resolution) # charging power
            
        return P_ch

@jit(nopython=True)
def battery_discharge(params, SOC_eff, RT_efficiency, taking_over, power: float, time_resolution: float) -> float:
        """
        Compute admissible battery discharging power under operational constraints.
        
        This function determines the maximum discharge power that satisfies:
        
            - Requested load demand,
            - Maximum discharge power,
            - Available stored energy,
            - Round-trip efficiency losses,
            - SOC-dependent participation factor (taking-over).
        
        It is JIT-compiled for fast execution during large-scale evolutionary
        optimisation runs in ERMESS.
        
        Args:
            params (numpy.ndarray): Battery parameter array where:
                - params[0] = capacity (kWh)
                - params[2] = maximum discharging power (kW)
                
            SOC_eff (float): Effective state of charge (0–1).
            RT_efficiency (float): Round-trip efficiency (0–1).
            taking_over (numpy.ndarray): SOC-dependent participation control vector.
            power (float): Requested discharge power (kW).
            time_resolution (float): Simulation time step (hours).
        
        Returns:
            float: Effective discharging power delivered by the battery (kW).
        
        Notes:
            - Available energy is reduced by the round-trip efficiency to account for conversion losses.
        
        Warning:
            No internal SOC update is performed here. The state of charge must be updated consistently after computing the power flow.
            
        """
        e_available = (SOC_eff) * params[0]  * RT_efficiency  # energy remaining and usable
        closest_level = min(8,max(0,np.int64(10*SOC_eff)))
        participation = 1-taking_over[closest_level]
        P_disch = min(power*participation, params[2], e_available * time_resolution) # discharging power knowing the constraints
        
        return (P_disch)

