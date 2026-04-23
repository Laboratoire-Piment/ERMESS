# -*- coding: utf-8 -*-
"""
Created on Tue Jun 10 14:03:31 2025

@author: JoPHOBEA
"""

import numpy as np
from numba import jit

from . import ERMESS_storages_model as ESM 
from ERMESS_scripts.data import data_classes as Dcl
from ERMESS_scripts.data.indices import *



@jit(nopython=True)
def LFE_CCE(gene, global_parameters, pro_parameters, production ,RENSystems_parameters) :
    """
    Simulates microgrid non-predictive dispatch using core ERMESS algorithm.
    
    Accepts Load Following and Cycle Charging strategies.
    
    Performs a time-step simulation of:
        - Non-movable load supply
        - Demand Side Management (DSM)
        - Multi-storage dispatch (charge/discharge ordering)
        - Grid or diesel generator exchange
        - SOC evolution with efficiency losses
    
    This function is JIT-compiled (Numba, nopython mode) for high-performance
    evaluation inside the evolutionary optimisation loop.
    
    Dispatch logic:
        1. Compute net production (generation - fixed load).
        2. Allocate energy to DSM (daily and yearly movable loads).
        3. Dispatch batteries according to priority order.
        4. Apply Load Following or Cycle Charging logic.
        5. Enforce diesel generator minimum production and runtime constraints
           (off-grid mode).
        6. Close storage loops at final timestep.
    
    Args:
        gene (numba-compatible structured object): Individual solution of the evolutionary
            algorithm containing operational strategy parameters (DSM levels, storage order,
            DG constraints, etc.).
        Non_movable_load (numpy.ndarray): Time series of fixed (non-flexible) load (kW).
        total_D_Movable_load (numpy.ndarray): Daily flexible load energy requirements.
        total_Y_Movable_load (float): Yearly flexible load energy requirement.
        production (numpy.ndarray): Time series of total generation (kW).
        n_bits (int): Number of simulation time steps.
        n_store (int): Number of storage units.
        time_resolution (float): Simulation time step duration (hours).
        Connexion (str): Grid connection mode ('On-grid' or 'Off-grid').
        storage_characteristics (numpy.ndarray): Storage technical parameters
            (including efficiencies).
    
    Returns:
        tuple: Contains the following elements:
            - P_bat (numpy.ndarray, kW): Battery power matrix (positive = discharge, negative = charge).
            - P_ext (numpy.ndarray, kW): External grid/diesel power (positive = import, negative = export).
            - D_DSM (numpy.ndarray, kW): Daily DSM load allocation.
            - Y_DSM (numpy.ndarray, kW): Yearly DSM load allocation.
            - SOCs_eff (numpy.ndarray, -): Effective state of charge per storage.
            - Losses (numpy.ndarray, kW): Storage conversion losses.
            - P_diff (numpy.ndarray, kW): Global power balance residual.
    
    Notes:
        - Storage dispatch priority is controlled by `gene.PMS_discharge_order`.
        - In Cycle Charging mode ('CC'), the diesel generator may be forced
          to operate at minimum production and minimum runtime.
    
    Warnings:
        - The function assumes all inputs are dimensionally consistent and
          NumPy arrays compatible with Numba nopython mode.
    
    Important:
        - This routine is the computational core of ERMESS and directly
          determines the fitness evaluation of candidate microgrid configurations.
    """
    
    #---------------------------------------------------------------------------------
    # Recurrent variables
    #---------------------------------------------------------------------------------
    n_bits = global_parameters.n_bits
    n_store = RENSystems_parameters.n_store
    time_resolution = global_parameters.time_resolution
    
    #---------------------------------------------------------------------------------
    # time series
    # --------------------------------------------------------------------------------------------

    P_net = production - global_parameters.Non_movable_load  # Production - Load power (kW).

    P_ext = np.repeat(0.,n_bits)         # [kW] Grid OUTPUT power (<0 when selling and >0 when buying)
    P_bat = np.repeat(0.,n_store*n_bits).reshape(n_store,n_bits)        # [kW] Battery OUTPUT power (<0 when charging and >0 when discharging).
    SOCs_eff = np.repeat(np.nan,n_store*n_bits).reshape(n_store,n_bits)   # SOCs effective (i.e. between 0 and 1, whatever the discharge depth)
    Losses = np.repeat(0.,n_store*n_bits).reshape(n_store,n_bits) 
    D_DSM = np.repeat(np.nan,n_bits) 
    Y_DSM = np.repeat(np.nan,n_bits) 
    P_ext_cur_runtime = np.repeat(0,n_bits) 
    
    # debug & details
    # --------------------------------------------------------------------------------------------
    indic = np.repeat(np.nan,n_bits) # indicates in which if/else branch each time step is

    # SIMULATION
    # --------------------------------------------------------------------------------------------
    full_D_DSM_min_levels = np.interp(np.arange(0,24,1/time_resolution),np.arange(0.,24.),np.concatenate((gene.D_DSM_minimum_levels,np.array( [1.]))))
    full_Y_DSM_min_levels = np.interp(np.arange(0,12+1/(n_bits),12/(n_bits)),np.arange(0.,13.),np.concatenate((np.array( [0.]),gene.Y_DSM_minimum_levels,np.array( [1.]))))
    day=-1
    D_DSM_count=0
    Y_DSM_count=0
    for i in range(n_bits-1):
        if((i)%(time_resolution*24)==0):
            day = day+1
            D_DSM_needs = pro_parameters.total_D_Movable_load[day]
            D_DSM_count=0
        SOC_eff = SOCs_eff[:,i-1] if i>0 else gene.storages[3,:]
        P_ext[i] = gene.DG_min_production if((gene.DG_strategy=='CC') & (global_parameters.Connexion=='Off-grid' ) & (P_ext_cur_runtime[i]<gene.DG_min_runtime)) else 0
        D_DSM[i] = max(0,D_DSM_needs*full_D_DSM_min_levels[int((i)%(time_resolution*24))]-D_DSM_count)
        Y_DSM[i] = max(0,pro_parameters.total_Y_Movable_load*full_Y_DSM_min_levels[i]-Y_DSM_count)
        P_net_2 = P_net[i] - D_DSM[i] - Y_DSM[i]
        D_DSM[i] = D_DSM[i] + max(0,min(P_net_2*gene.energy_use_coefficient,D_DSM_needs-D_DSM_count-D_DSM[i]))
        P_net_3 = P_net[i] - D_DSM[i] - Y_DSM[i]
        Y_DSM[i] = Y_DSM[i] + max(0,min(P_net_3*gene.energy_use_coefficient,pro_parameters.total_Y_Movable_load-Y_DSM_count-Y_DSM[i]))
        D_DSM_count = D_DSM_count + D_DSM[i]
        Y_DSM_count = Y_DSM_count + Y_DSM[i]
        P_affordable = P_net[i] - D_DSM[i] - Y_DSM[i]

        if P_affordable >= 0:  # power excess
            index=0
            while ((index<(n_store))) : 
                index = index+1
                store = gene.discharge_order[n_store-index]
                if ((P_affordable>0) & (min(gene.storages[0:3,store])>0)):
                    P_bat[store,i] = -ESM.battery_charge(gene.storages[:,store], SOC_eff[store],gene.overlaps[1,:] if (index==(n_store-1)) else (gene.overlaps[0,:]) , P_affordable, time_resolution) # battery charging 
                    P_affordable = P_affordable+P_bat[store,i]
                    SOCs_eff[store,i] = SOC_eff[store] - P_bat[store,i] /time_resolution / gene.storages[0,store]
                else : 
                   # P_bat[store,i],Losses[store,i]=0,0
                    SOCs_eff[store,i] = SOC_eff[store]
                
            indic[i]=1
            
        else :        # power deficit
            index=-1
            while ((index<(n_store-1))) : 
                index = index+1
                store=gene.discharge_order[index]
                if ((P_affordable<0) & (min(gene.storages[0:3,store])>0)):                                   
                    P_bat[store,i]=ESM.battery_discharge(gene.storages[:,store], SOC_eff[store], RENSystems_parameters.specs_storage[STOR_ROUND_TRIP_EFF,store],gene.overlaps[1,:] if (index==(n_store-2)) else (gene.overlaps[0,:]) , -P_affordable, time_resolution) # battery charging                
                    P_affordable=P_affordable+P_bat[store,i]
                    Losses[store,i]=P_bat[store,i]/RENSystems_parameters.specs_storage[STOR_ROUND_TRIP_EFF,store]-P_bat[store,i]
                    SOCs_eff[store,i] = SOC_eff[store]-(P_bat[store,i]+Losses[store,i]) /time_resolution / gene.storages[0,store]
                else : 
                    P_bat[store,i],Losses[store,i]=0,0
                    SOCs_eff[store,i] = SOC_eff[store]
                

        P_ext[i]=P_ext[i] -P_affordable
        if ((gene.DG_strategy=='CC') & (global_parameters.Connexion=='Off-grid' ) & (0<P_ext[i]<gene.DG_min_production)):
            P_ext[i] = gene.DG_min_production
        if (i<(n_bits-1)):
            if(P_ext[i]<=0) :
                P_ext_cur_runtime[i+1] = 0
            else :                         
                P_ext_cur_runtime[i+1] = P_ext_cur_runtime[i]+1         # grid supplies load
        indic[i] = 5
    
    i=n_bits-1
  #Last timestep, storage loops are closed      
    if((i)%(time_resolution*24)==0):
        day = day+1
        D_DSM_needs = pro_parameters.total_D_Movable_load[day]
        D_DSM_count=0
    P_ext[i] = gene.DG_min_production if((gene.DG_strategy=='CC') & (global_parameters.Connexion=='Off-grid' ) & (P_ext_cur_runtime[i]<gene.DG_min_runtime)) else 0
    D_DSM[i] = max(0,D_DSM_needs*full_D_DSM_min_levels[int((i)%(time_resolution*24))]-D_DSM_count)
    Y_DSM[i] = max(0,pro_parameters.total_Y_Movable_load*full_Y_DSM_min_levels[i]-Y_DSM_count)
    P_net_2 = P_net[i] - D_DSM[i] - Y_DSM[i]
    D_DSM[i] = D_DSM[i] + max(0,min(P_net_2*gene.energy_use_coefficient,D_DSM_needs-D_DSM_count-D_DSM[i]))
    P_net_3 = P_net[i] - D_DSM[i] - Y_DSM[i]
    Y_DSM[i] = Y_DSM[i] + max(0,min(P_net_3*gene.energy_use_coefficient,pro_parameters.total_Y_Movable_load-Y_DSM_count-Y_DSM[i]))
    D_DSM_count = D_DSM_count + D_DSM[i]
    Y_DSM_count = Y_DSM_count + Y_DSM[i]
    P_affordable = P_net[i] - D_DSM[i] - Y_DSM[i]

    store=-1
    while ((store<(n_store-1))) : 
            store = store+1
            if( min(gene.storages[0:3,store])>0):
                Diff = -(gene.storages[3,store] - SOCs_eff[store,i-1])*gene.storages[0,store]*time_resolution
                P_bat[store,i] = Diff*RENSystems_parameters.specs_storage[STOR_ROUND_TRIP_EFF,store] if (Diff>0) else Diff
                Losses[store,i]=P_bat[store,i]/RENSystems_parameters.specs_storage[STOR_ROUND_TRIP_EFF,store]-P_bat[store,i] if (Diff>0) else 0
                P_affordable = P_affordable + P_bat[store,i]
                SOCs_eff[store,i] = SOCs_eff[store,i-1] - (P_bat[store,i]+Losses[store,i])  /time_resolution / gene.storages[0,store]
            else :
                SOCs_eff[store,i] = SOCs_eff[store,i-1]
    
    P_ext[i]=P_ext[i] -P_affordable
    if ((gene.DG_strategy=='CC') & (global_parameters.Connexion=='Off-grid' ) & (0<P_ext[i]<gene.DG_min_production)):
        P_ext[i] = gene.DG_min_production
    
    # OUTPUT
    # --------------------------------------------------------------------------------------------
    P_diff = P_net + sum(np.sum(P_bat,axis=1)) + P_ext 

    return (P_bat,P_ext,D_DSM,Y_DSM,SOCs_eff,Losses,P_diff)