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
        """battery definition. NB : the energy efficiency eta is only taken into account during the discharge process.

        Args:
            paramIn (dict): contains the following parameters
                capacity (float): storage capacity (kWh)
                SOC (float): (initial) state of charge (0 to 1)
                SOCmin (float): minimum state of charge (0 to 1)
                SOCmax (float): maximum state of charge (0 to 1)
                eta (float): energy efficiency (0 to 1)
                Pmax_ch (float): maximum power allowed during charge (kW)
                Pmax_disch (float): maximum power allowed during discharge (kW)
                lifetime (float): life expectancy of the battery (kWh) (= lifetime in number of cycles * capacity * (SOCmax - SOCmin))
                repl_cost (float): replacement cost or price (euros)
                maint_cost (float): maintenance cost (euros/kWh)
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
        """simulates the charge of a battery.
        Returns:
            float: the power actually used to charge the battery, in kW
        """
        e_needed = (1 - SOC_eff) * params[0] # energy needed to fully charge the battery
        closest_level = min(8,max(0,9-np.int64(10*SOC_eff)))
        participation = 1-taking_over[closest_level]
        P_ch = min(power*participation, params[1], e_needed * time_resolution) # charging power
            
        return P_ch

@jit(nopython=True)
def battery_discharge(params, SOC_eff, RT_efficiency, taking_over, power: float, time_resolution: float) -> float:
        """simulates the discharge of a battery. SOC is updated within
        Returns:
            float: power supplied by the battery, in kW
        """
        e_available = (SOC_eff) * params[0]  * RT_efficiency  # energy remaining and usable
        closest_level = min(8,max(0,np.int64(10*SOC_eff)))
        participation = 1-taking_over[closest_level]
        P_disch = min(power*participation, params[2], e_available * time_resolution) # discharging power knowing the constraints
        
            # print('P_disch_bat',P_disch)
            # print(' - e_lost =', P_disch * dt * (1 - self.eta), 'Wh (because of the energy efficiency eta)')
        return (P_disch)

