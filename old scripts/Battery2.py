# -*- coding:utf-8 -*-
# Created on 2025-05-23 15:37:52
# Projet: BDD-Labo
## @package virtualPMS\Battery.py
# @brief Single battery definition. Includes charge and discharge routines, and test section.
# @version: 1.0
# @author: Mathieu Lafitte
# @n---------------------

#%%

import numpy as np
from numba import jit

class Battery:
    def __init__(self,capacity, SOCmin,SOCmax,RT_efficiency,  power_in, power_out, SOC_initial, SOC):
        """
        Initializes the Battery object.

        Args:
            capacity (float): Storage capacity (kWh).
            SOCmin (float): Minimum state of charge (0 to 1).
            SOCmax (float): Maximum state of charge (0 to 1).
            RT_efficiency (float): Round trip efficiency (0 to 1).
            power_in (float): Maximum power allowed during charge (kW).
            power_out (float): Maximum power allowed during discharge (kW).
            SOC_initial (float): Initial state of charge (0 to 1).
            SOC (float): Current state of charge (0 to 1).
        """
        self.capacity = capacity
        self.SOCmin = SOCmin
        self.SOCmax = SOCmax
        self.RT_efficiency = RT_efficiency
        self.power_in = power_in
        self.power_out = power_out
        self.SOC_initial = SOC_initial
        self.SOC = SOC
        if not (self.SOCmin <= self.SOC <= self.SOCmax):
            print('!! SOCmin < SOC < SOCmax !!')
            self.SOC = self.SOCmin

@jit(nopython=True)
def battery_charge(params, SOC_eff, taking_over, power: float, time_resolution: float) -> float:
        """
        Simulates the charging process of a battery.

        Args:
            params (array-like): List of battery parameters, where:
                params[0] is capacity (kWh),
                params[1] is maximum charge power (kW).
            SOC_eff (float): Effective state of charge (0 to 1).
            taking_over (array-like): Control signal or participation factor indexed by SOC level.
            power (float): Available charging power (kW).
            time_resolution (float): Time step duration (hours).

        Returns:
            float: The actual power used to charge the battery (kW).
        """
        e_needed = (1 - SOC_eff) * params[0] # energy needed to fully charge the battery
        closest_level = min(8,max(0,9-np.int64(10*SOC_eff)))
        participation = 1-taking_over[closest_level]
        P_ch = min(power*participation, params[1], e_needed * time_resolution) # charging power
            
        return P_ch

@jit(nopython=True)
def battery_discharge(params, SOC_eff, RT_efficiency, taking_over, power: float, time_resolution: float) -> float:
        """
        Simulates the discharging process of a battery.

        Args:
            params (array-like): List of battery parameters, where:
                params[0] is capacity (kWh),
                params[2] is maximum discharge power (kW).
            SOC_eff (float): Effective state of charge (0 to 1).
            RT_efficiency (float): Round trip efficiency (0 to 1).
            taking_over (array-like): Control signal or participation factor indexed by SOC level.
            power (float): Requested discharging power (kW).
            time_resolution (float): Time step duration (hours).

        Returns:
            float: The actual power supplied by the battery (kW).
        """
        e_available = (SOC_eff) * params[0]  * RT_efficiency  # energy remaining and usable
        closest_level = min(8,max(0,np.int64(10*SOC_eff)))
        participation = 1-taking_over[closest_level]
        P_disch = min(power*participation, params[2], e_available * time_resolution) # discharging power knowing the constraints
        
            # print('P_disch_bat',P_disch)
            # print(' - e_lost =', P_disch * dt * (1 - self.eta), 'Wh (because of the energy efficiency eta)')
        return (P_disch)

