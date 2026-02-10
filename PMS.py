# -*- coding: utf-8 -*-
"""
Created on Tue Jun 10 14:03:31 2025

@author: jlegalla
"""

import numpy as np
import Battery
from numba import jit

@jit(nopython=True)
def LFE_CCE(gene, Non_movable_load, total_D_Movable_load, total_Y_Movable_load, production, n_bits,n_store,time_resolution, Connexion, storage_characteristics) :
    """Load Following and Cycle Charging dispatching routine.
    """
    
    # time series
    # --------------------------------------------------------------------------------------------
    P_net = production - Non_movable_load  # Production - Load power (kW).

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
    full_D_DSM_min_levels = np.interp(np.arange(0,24,1/time_resolution),np.arange(0.,24.),np.concatenate((gene.PMS_D_DSM_min_levels,np.array( [1.]))))
    full_Y_DSM_min_levels = np.interp(np.arange(0,12+1/(n_bits),12/(n_bits)),np.arange(0.,13.),np.concatenate((np.array( [0.]),gene.PMS_Y_DSM_min_levels,np.array( [1.]))))
    day=-1
    D_DSM_count=0
    Y_DSM_count=0
    for i in range(n_bits-1):
        if((i)%(time_resolution*24)==0):
            day = day+1
            D_DSM_needs = total_D_Movable_load[day]
            D_DSM_count=0
        SOC_eff = SOCs_eff[:,i-1] if i>0 else gene.storages[3,:]
        P_ext[i] = gene.PMS_DG_min_production if((gene.PMS_strategy=='CC') & (Connexion=='Off-grid' ) & (P_ext_cur_runtime[i]<gene.PMS_DG_min_runtime)) else 0
        D_DSM[i] = max(0,D_DSM_needs*full_D_DSM_min_levels[int((i)%(time_resolution*24))]-D_DSM_count)
        Y_DSM[i] = max(0,total_Y_Movable_load*full_Y_DSM_min_levels[i]-Y_DSM_count)
        P_net_2 = P_net[i] - D_DSM[i] - Y_DSM[i]
        D_DSM[i] = D_DSM[i] + max(0,min(P_net_2*gene.energy_use_repartition_DSM,D_DSM_needs-D_DSM_count-D_DSM[i]))
        P_net_3 = P_net[i] - D_DSM[i] - Y_DSM[i]
        Y_DSM[i] = Y_DSM[i] + max(0,min(P_net_3*gene.energy_use_repartition_DSM,total_Y_Movable_load-Y_DSM_count-Y_DSM[i]))
        D_DSM_count = D_DSM_count + D_DSM[i]
        Y_DSM_count = Y_DSM_count + Y_DSM[i]
        P_affordable = P_net[i] - D_DSM[i] - Y_DSM[i]

        if P_affordable >= 0:  # power excess
            index=0
            while ((index<(n_store))) : 
                index = index+1
                store = gene.PMS_discharge_order[n_store-index]
                if ((P_affordable>0) & (min(gene.storages[0:3,store])>0)):
                    P_bat[store,i] = -Battery.battery_charge(gene.storages[:,store], SOC_eff[store],gene.PMS_taking_over[1,:] if (index==(n_store-1)) else (gene.PMS_taking_over[0,:]) , P_affordable, time_resolution) # battery charging 
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
                store=gene.PMS_discharge_order[index]
                if ((P_affordable<0) & (min(gene.storages[0:3,store])>0)):                                   
                    P_bat[store,i]=Battery.battery_discharge(gene.storages[:,store], SOC_eff[store], storage_characteristics[4,store],gene.PMS_taking_over[1,:] if (index==(n_store-2)) else (gene.PMS_taking_over[0,:]) , -P_affordable, time_resolution) # battery charging                
                    P_affordable=P_affordable+P_bat[store,i]
                    Losses[store,i]=P_bat[store,i]/storage_characteristics[4,store]-P_bat[store,i]
                    SOCs_eff[store,i] = SOC_eff[store]-(P_bat[store,i]+Losses[store,i]) /time_resolution / gene.storages[0,store]
                else : 
                    P_bat[store,i],Losses[store,i]=0,0
                    SOCs_eff[store,i] = SOC_eff[store]
                

        P_ext[i]=P_ext[i] -P_affordable
        if ((gene.PMS_strategy=='CC') & (Connexion=='Off-grid' ) & (0<P_ext[i]<gene.PMS_DG_min_production)):
            P_ext[i] = gene.PMS_DG_min_production
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
        D_DSM_needs = total_D_Movable_load[day]
        D_DSM_count=0
    P_ext[i] = gene.PMS_DG_min_production if((gene.PMS_strategy=='CC') & (Connexion=='Off-grid' ) & (P_ext_cur_runtime[i]<gene.PMS_DG_min_runtime)) else 0
    D_DSM[i] = max(0,D_DSM_needs*full_D_DSM_min_levels[int((i)%(time_resolution*24))]-D_DSM_count)
    Y_DSM[i] = max(0,total_Y_Movable_load*full_Y_DSM_min_levels[i]-Y_DSM_count)
    P_net_2 = P_net[i] - D_DSM[i] - Y_DSM[i]
    D_DSM[i] = D_DSM[i] + max(0,min(P_net_2*gene.energy_use_repartition_DSM,D_DSM_needs-D_DSM_count-D_DSM[i]))
    P_net_3 = P_net[i] - D_DSM[i] - Y_DSM[i]
    Y_DSM[i] = Y_DSM[i] + max(0,min(P_net_3*gene.energy_use_repartition_DSM,total_Y_Movable_load-Y_DSM_count-Y_DSM[i]))
    D_DSM_count = D_DSM_count + D_DSM[i]
    Y_DSM_count = Y_DSM_count + Y_DSM[i]
    P_affordable = P_net[i] - D_DSM[i] - Y_DSM[i]

    store=-1
    while ((store<(n_store-1))) : 
            store = store+1
            if( min(gene.storages[0:3,store])>0):
                Diff = -(gene.storages[3,store] - SOCs_eff[store,i-1])*gene.storages[0,store]*time_resolution
                P_bat[store,i] = Diff*storage_characteristics[4,store] if (Diff>0) else Diff
                Losses[store,i]=P_bat[store,i]/storage_characteristics[4,store]-P_bat[store,i] if (Diff>0) else 0
                P_affordable = P_affordable + P_bat[store,i]
                SOCs_eff[store,i] = SOCs_eff[store,i-1] - (P_bat[store,i]+Losses[store,i])  /time_resolution / gene.storages[0,store]
            else :
                SOCs_eff[store,i] = SOCs_eff[store,i-1]
    
    P_ext[i]=P_ext[i] -P_affordable
    if ((gene.PMS_strategy=='CC') & (Connexion=='Off-grid' ) & (0<P_ext[i]<gene.PMS_DG_min_production)):
        P_ext[i] = gene.PMS_DG_min_production
    
    # OUTPUT
    # --------------------------------------------------------------------------------------------
    P_diff = P_net + sum(np.sum(P_bat,axis=1)) + P_ext 

    return (P_bat,P_ext,D_DSM,Y_DSM,SOCs_eff,Losses,P_diff)