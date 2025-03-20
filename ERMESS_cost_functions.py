# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 13:11:54 2024

@author: jlegalla
"""
import pandas as pd
import numpy as np
import sys
import time
from numba import jit
import ERMESS_classes as ECl



def cost_scenario_LCOE_detail(gene,storage_characteristics,time_resolution,n_store,duration_years,specs_num,prices_num,fixed_premium,Overrun,Selling_price,Non_movable_load,Main_grid_emissions,prod_C,prods_U,Bounds_prod,constraint_num,constraint_level,cost_constraint):

    production_set=gene.production_set
    production = np.dot(gene.production_set,prods_U)/1000+prod_C/1000

    annual_cost_production = np.inner(specs_num[:,0]/specs_num[:,2]+specs_num[:,1],gene.production_set)
    
    losses = np.array([gene.storage_TS[i]/storage_characteristics[4,:][i]-gene.storage_TS[i] for i in range(n_store)])
    losses[losses<0]=0 
    sum_diff_storages = [np.cumsum(gene.storage_TS[i]/time_resolution+losses[i]/time_resolution) for i in range(n_store)]
    energy_storages = [(max(sum_diff_storages[i]) - min(sum_diff_storages[i]))/storage_characteristics[5,:][i] for i in range(n_store)] 
    powers_out = tuple((max(gene.storage_TS[i]) for i in range(n_store)))
    powers_in = tuple((-min(gene.storage_TS[i]) for i in range(n_store)))
    size_power=max(powers_in,powers_out)
    CAPEX_storage_cost =  tuple((size_power[i]*storage_characteristics[11,:][i] + energy_storages[i]*storage_characteristics[0,:][i] + storage_characteristics[9,:][i]*(size_power[i]>0) for i in range(n_store)))
    Equivalent_cycles =  np.divide(np.sum(abs(gene.storage_TS),axis=1)/(2*time_resolution)  ,[e if e>=1e-15 else 1e-15 for e in energy_storages])/duration_years
    Lifetime = tuple(min(storage_characteristics[7,:][i],storage_characteristics[8,:][i]/max(1e-15,Equivalent_cycles[i])) for i in range(n_store))
    Optimized_Load = Non_movable_load+gene.Y_DSM+gene.D_DSM.flatten()
    importation = Optimized_Load-production-np.sum(gene.storage_TS,axis=0)
    importation[importation<0]=0
    Contract_power=max(importation)
    
    exportation = -(Optimized_Load-production-np.sum(gene.storage_TS,axis=0))
    exportation[exportation<0]=0
    obtained_constraint_level = 1-(sum(importation)/sum(Optimized_Load)) if constraint_num==1 else 1-(sum(exportation)/sum(production))
    
    total_cost=(cost_constraint*(constraint_level-obtained_constraint_level) if obtained_constraint_level<constraint_level else 0) + (annual_cost_production*duration_years+sum(np.divide(CAPEX_storage_cost,Lifetime))*duration_years+sum(np.multiply(storage_characteristics[3,:],size_power))*duration_years+np.multiply(importation,prices_num[gene.contract,:]).sum()/time_resolution+fixed_premium[gene.contract]*Contract_power*duration_years+max(0,(max(importation)-Contract_power)*Overrun[gene.contract])-sum(exportation)*Selling_price[gene.contract])/(sum(Optimized_Load)/time_resolution)  

    return{'LCOE (€/kWh)':total_cost,'CAPEX production (€/kWh)':np.inner(specs_num[:,0]/specs_num[:,2],production_set)*duration_years/(sum(Optimized_Load)/time_resolution),'OPEX production (€/kWh)':np.inner(specs_num[:,1],production_set)*duration_years/(sum(Optimized_Load)/time_resolution),'CAPEX storage (€/kWh)':tuple(CAPEX_storage_cost[i]*duration_years/(sum(Optimized_Load)/time_resolution)/Lifetime[i] for i in range(n_store)),'OPEX storage (€/kWh)':tuple(storage_characteristics[3,:][i]*size_power[i]/(sum(Optimized_Load)/time_resolution)*duration_years for i in range(n_store)),'Energy importation (€/kWh)':(np.multiply(importation,prices_num[gene.contract]).sum()/time_resolution)/(sum(Optimized_Load)/time_resolution),'Fixed premium (€/kWh)':fixed_premium[gene.contract]*Contract_power*duration_years/(sum(Optimized_Load)/time_resolution),'Overrun penalty (€/kWh)':max(0,(max(importation)-Contract_power)*Overrun[gene.contract])/(sum(Optimized_Load)/time_resolution),'Energy exportation (€/kWh)':(sum(exportation)*Selling_price[gene.contract]/time_resolution),'Initial investment (€)':np.inner(specs_num[:,0],production_set)+sum(tuple(CAPEX_storage_cost[i] for i in range(n_store))),'Contract power (kW)':Contract_power}

def cost_base(prod_C,time_resolution,duration_years,prices_num,fixed_premium,Overrun,Selling_price,Non_movable_load,Y_movable_load,D_movable_load):
    Load = Non_movable_load+D_movable_load+Y_movable_load
    importation = Load-prod_C/1000
    importation[importation<0]=0
    Contract_power=max(importation)
    exportation = -(Load-prod_C/1000)
    exportation[exportation<0]=0
    total_cost=((np.multiply(importation,prices_num).sum()/time_resolution)+fixed_premium*Contract_power*duration_years+max(0,(max(importation)-Contract_power)*Overrun)-sum(exportation)*Selling_price)/(sum(Load)/time_resolution)
    return{'LCOE (€/kWh)':total_cost,'Energy importation (€/kWh)':(np.multiply(importation,prices_num).sum()/time_resolution)/(sum(Load)/time_resolution),'Fixed premium (€/kWh)':fixed_premium*Contract_power*duration_years/(sum(Load)/time_resolution),'Overrun penalty (€/kWh)':max(0,(max(importation)-Contract_power)*Overrun)/(sum(Load)/time_resolution),'Contract power (kW)':Contract_power}

### Optimisation criterion LCOE #####
def cost_pre_scenario_LCOE(gene,storage_characteristics,time_resolution,n_store,duration_years,specs_num,prices_num,fixed_premium,Overrun,Selling_price,Non_movable_load,Main_grid_emissions,prod_C,prods_U,Bounds_prod,constraint_num,constraint_level):
    
    production = np.dot(gene.production_set,prods_U)/1000+prod_C/1000

    annual_cost_production = np.inner(specs_num[:,0]/specs_num[:,2]+specs_num[:,1],gene.production_set)
    #annual_production_emissions = sum(sum(np.multiply(specs_prod['eqCO2 Emissions (gCO2/kWh)'].values,np.multiply(gene.production_set,prods_U.T))))/time_resolution/duration_years
    #EROI_production = sum(gene.production_set/sum(gene.production_set)*specs_prod["EROI"]) if sum(gene.production_set)>0 else 0
    
    losses = np.array([gene.storage_TS[i]/storage_characteristics['Round-trip efficiency'][i]-gene.storage_TS[i] for i in range(n_store)])
    losses[losses<0]=0 
    sum_diff_storages = [np.cumsum(gene.storage_TS[i]+losses[i]) for i in range(n_store)]
    energy_storages = [(max(sum_diff_storages[i]) - min(sum_diff_storages[i]))/storage_characteristics['Depth of discharge'][i] for i in range(n_store)] 
    powers_out = tuple((max(gene.storage_TS[i]*time_resolution) for i in range(n_store))) 
    powers_in = tuple((-min(gene.storage_TS[i]*time_resolution) for i in range(n_store)))
    size_power=max(powers_in,powers_out)
    CAPEX_storage_cost =  tuple((size_power[i]*storage_characteristics['Cost power'][i] + energy_storages[i]*storage_characteristics['Energy cost (€/kWh)'][i] + storage_characteristics['Installation cost (€)'][i]*(size_power[i]>0) for i in range(n_store)))
    Equivalent_cycles =  np.divide(np.sum(abs(gene.storage_TS),axis=1)/(2*time_resolution)  ,[e if e>=1e-15 else 1e-15 for e in energy_storages])/duration_years
    Lifetime = tuple(min(storage_characteristics['Lifetime (years)'][i],storage_characteristics['Cycle life'][i]/max(1e-15,Equivalent_cycles[i])) for i in range(n_store))
    Optimized_Load = Non_movable_load+gene.Y_DSM+gene.D_DSM.flatten()
    importation = Optimized_Load-production-np.sum(gene.storage_TS,axis=0)
    importation[importation<0]=0
    Contract_power=max(importation)
    
    exportation = -(Optimized_Load-production-np.sum(gene.storage_TS,axis=0))
    exportation[exportation<0]=0
    obtained_constraint_level = 1-(sum(importation)/sum(Optimized_Load)) if constraint_num==1 else 1-(sum(exportation)/sum(production))
    total_cost=10e10+constraint_level-obtained_constraint_level if obtained_constraint_level<constraint_level else (annual_cost_production*duration_years+sum(tuple(CAPEX_storage_cost[i]/Lifetime[i] for i in range(n_store)))*duration_years+sum(tuple(storage_characteristics['O&M cost (€/kW-yr.)'][i]*size_power[i]*duration_years for i in range(n_store)))+np.multiply(importation,prices_num[gene.contract]).sum()/time_resolution+fixed_premium[gene.contract]*Contract_power*duration_years+max(0,(max(importation)-Contract_power)*Overrun[gene.contract])-sum(exportation)*Selling_price[gene.contract])/(sum(Optimized_Load)/time_resolution)  
    return(total_cost)


@jit(nopython=True)
def cost_scenario_LCOE(gene,storage_characteristics,time_resolution,n_store,duration_years,specs_num,prices_num,fixed_premium,Overrun,Selling_price,Non_movable_load,Main_grid_emissions,prod_C,prods_U,Bounds_prod,constraint_num,constraint_level,cost_constraint):

    n_steps = len(gene.storage_TS[0])
    production = ((prods_U.T*gene.production_set).sum(axis=1)+prod_C)/1000    
    annual_cost_production = np.sum((specs_num[:,0]/specs_num[:,2]+specs_num[:,1])*(gene.production_set))
    losses = np.transpose(np.divide(np.transpose(gene.storage_TS),storage_characteristics[4,:]))-gene.storage_TS 
    losses=np.where(losses>0,losses,0)
    sum_diff_storages = [np.cumsum(gene.storage_TS[i,:]+losses[i,:])/time_resolution for i in range(n_store)]
    energy_storages = np.divide(np.array([np.max(sum_diff_storages[i]) - np.min(sum_diff_storages[i]) for i in range(n_store)]),storage_characteristics[5,:])
    powers_out = [np.max(gene.storage_TS[i,:]) for i in range(n_store)]
    powers_in = [-np.min(gene.storage_TS[i,:]) for i in range(n_store)]
    size_power=np.array([max(powers_in[i],powers_out[i]) for i in range(n_store)])
    CAPEX_storage_cost =  np.multiply(size_power,storage_characteristics[11,:]) + np.multiply(energy_storages,storage_characteristics[0,:]) + np.multiply(storage_characteristics[9,:],(size_power>np.repeat(0,n_store)))
    Equivalent_cycles =  np.array([np.sum(np.abs(gene.storage_TS[i,:]))/(2*time_resolution*max(energy_storages[i],np.float64(1e-15))*duration_years) for i in range(n_store)]) 
    Lifetime = np.array([min(storage_characteristics[7,i],storage_characteristics[8,i]/max(1e-15,Equivalent_cycles[i])) for i in range(n_store)])
    Optimized_Load = Non_movable_load+gene.Y_DSM+gene.D_DSM.flatten()
    results_stores=np.array([np.sum(gene.storage_TS[:,i]) for i in range(n_steps)])
    trades = Optimized_Load-production-results_stores
    importation=np.where(trades>0,trades,0)
    exportation = np.where(trades<0,trades,0)
    Contract_power=max(0,max(trades))
    obtained_constraint_level = 1-(sum(np.where(trades>0,trades,0))/sum(Optimized_Load)) if constraint_num==1 else 1-(sum(np.where(trades<0,trades,0))/sum(production))

    total_cost=(cost_constraint*(constraint_level-obtained_constraint_level) if obtained_constraint_level<constraint_level else 0) + (annual_cost_production*duration_years+sum(np.divide(CAPEX_storage_cost,Lifetime))*duration_years+sum(np.multiply(storage_characteristics[3,:],size_power))*duration_years+np.multiply(importation,prices_num[gene.contract,:]).sum()/time_resolution+fixed_premium[gene.contract]*Contract_power*duration_years+max(0,(max(importation)-Contract_power)*Overrun[gene.contract])-sum(exportation)*Selling_price[gene.contract])/(sum(Optimized_Load)/time_resolution)  

    return(total_cost,trades)


def NONJIT_cost_scenario_LCOE(gene,storage_characteristics,time_resolution,n_store,duration_years,specs_num,prices_num,fixed_premium,Overrun,Selling_price,Non_movable_load,Main_grid_emissions,prod_C,prods_U,Bounds_prod,constraint_num,constraint_level,cost_constraint):
    ##start00=time.time()
    ##production = np.sum(((np.dot(prods_U.T,gene.production_set),prod_C)),axis=0)/1000
    ##start0=time.time()
    ##production = np.sum(((np.dot(gene.production_set,prods_U),prod_C)),axis=0)/1000
    
    ##start=time.time()
    
    ##production = np.sum(np.array([gene.production_set[i]*prods_U[i,:] for i in range(len(gene.production_set))])/1000,axis=0)+prod_C/1000
    #print(gene.production_set)
    ##t1=time.time()
    production = ((prods_U.T*gene.production_set).sum(axis=1)+prod_C)/1000
    ##t2=time.time()
    ##print(start0-start00,start-start0,t1-start,t2-t1)
    
    annual_cost_production = np.inner(specs_num[:,0]/specs_num[:,2]+specs_num[:,1],gene.production_set)
    #t2=time.time()
    #annual_production_emissions = sum(sum(np.multiply(specs_prod['eqCO2 Emissions (gCO2/kWh)'].values,np.multiply(gene.production_set,prods_U.T))))/time_resolution/duration_years
    #t3=time.time()
    #EROI_production = sum(gene.production_set/sum(gene.production_set)*specs_prod["EROI"])  if sum(gene.production_set)>0 else 0
    #t4=time.time()
    losses = np.transpose(np.divide(np.transpose(gene.storage_TS),storage_characteristics[4,:]))-gene.storage_TS 
    losses[losses<0]=0 
    #t5=time.time()
    sum_diff_storages = np.cumsum((gene.storage_TS+losses)/time_resolution,axis=1) 
    #t6=time.time()
    energy_storages = np.divide((np.max(sum_diff_storages,axis=1) - np.min(sum_diff_storages,axis=1)),storage_characteristics[5,:])
    #t7=time.time()
    powers_out = tuple(np.max(gene.storage_TS,axis=1))
    #t8=time.time()
    powers_in = tuple(-np.min(gene.storage_TS,axis=1))
    #t9=time.time()
    size_power=max(powers_in,powers_out)
    #t10=time.time()
    CAPEX_storage_cost =  np.multiply(size_power,storage_characteristics[11,:]) + np.multiply(energy_storages,storage_characteristics[0,:]) + np.multiply(storage_characteristics[9,:],(size_power>np.repeat(0,n_store)))
    #t11=time.time()
    Equivalent_cycles =  np.divide(np.sum(abs(gene.storage_TS),axis=1)/(2*time_resolution) ,[e if e>=1e-15 else 1e-15 for e in energy_storages])/duration_years
    #t12=time.time()
    Lifetime = tuple(np.nanmin(np.row_stack((storage_characteristics[7,:],np.divide(storage_characteristics[8,:],[max(1e-15,Equivalent_cycles[i]) for i in range(n_store)]))),axis=0))
    #t13=time.time()
    Optimized_Load = Non_movable_load+gene.Y_DSM+gene.D_DSM.flatten()
    #t14=time.time()
    trades = Optimized_Load-production-np.sum(gene.storage_TS,axis=0)
    importation = np.where(trades>0,trades,0)
    exportation = np.where(trades<0,trades,0)
    #t15=time.time()
    Contract_power=max(importation)
    #t16=time.time()
    obtained_constraint_level = 1-(sum(importation)/sum(Optimized_Load)) if constraint_num==1 else 1-(sum(exportation)/sum(production))
    #t18=time.time()

    total_cost=(cost_constraint*(constraint_level-obtained_constraint_level) if obtained_constraint_level<constraint_level else 0) + (annual_cost_production*duration_years+sum(np.divide(CAPEX_storage_cost,Lifetime))*duration_years+sum(np.multiply(storage_characteristics[3,:],size_power))*duration_years+np.multiply(importation,prices_num[gene.contract,:]).sum()/time_resolution+fixed_premium[gene.contract]*Contract_power*duration_years+max(0,(max(importation)-Contract_power)*Overrun[gene.contract])-sum(exportation)*Selling_price[gene.contract])/(sum(Optimized_Load)/time_resolution)  

    #t19=time.time()
    #print('t calcul fitness : ',t19-start)
    #print(t1-start,t4-t1,t5-t4,t6-t5,t7-t6,t8-t7,t9-t8,t10-t9,t11-t10,t12-t11,t13-t12,t14-t13,t15-t14,t16-t15,t18-t16,t19-t18)
    #print(t1-start,t9-t1,t19-t9)
    #print('volume',energy_storages,'power',(powers_in,powers_out),'Lifetime',Lifetime,'Cycles',Equivalent_cycles)
    #print('Contrainte',0.5*(constraint_level-obtained_constraint_level) if obtained_constraint_level<constraint_level else 0,'production',annual_cost_production*duration_years/(sum(Optimized_Load))/time_resolution,'CAPEX storage',sum(np.divide(CAPEX_storage_cost,Lifetime))*duration_years/(sum(Optimized_Load))/time_resolution,'OPEX storage',sum(np.multiply(storage_characteristics['O&M cost (€/kW-yr.)'],size_power))*duration_years/(sum(Optimized_Load))/time_resolution,'Importation',np.multiply(importation,grid_prices[gene.contract]['Price']).sum()/time_resolution/(sum(Optimized_Load))/time_resolution,'Fixed premium',fixed_premium[gene.contract]*Contract_power*duration_years/(sum(Optimized_Load))/time_resolution)
    return(total_cost,trades)

### Optimisation criterion Self-consumption #####
def Self_consumption_cost(gene,storage_characteristics,time_resolution,n_store,duration_years,specs_num,prices_num,fixed_premium,Overrun,Selling_price,Non_movable_load,Main_grid_emissions,prod_C,prods_U,Bounds_prod,constraint_num,constraint_level,cost_type):
    
    production = np.dot(gene.production_set,prods_U)/1000+prod_C/1000

    losses = np.transpose(np.divide(np.transpose(gene.storage_TS),storage_characteristics[4,:]))-gene.storage_TS 
    losses[losses<0]=0 
    sum_diff_storages = np.cumsum((gene.storage_TS+losses)/time_resolution,axis=1) 
    energy_storages = np.divide((np.max(sum_diff_storages,axis=1) - np.min(sum_diff_storages,axis=1)),storage_characteristics[5,:])
    powers_out = tuple(np.max(gene.storage_TS,axis=1))
    powers_in = tuple(-np.min(gene.storage_TS,axis=1))
    size_power=max(powers_in,powers_out)
    CAPEX_storage_cost = np.multiply(size_power,storage_characteristics[11,:]) + np.multiply(energy_storages,storage_characteristics[0,:]) + np.multiply(storage_characteristics[9,:],(size_power>np.repeat(0,n_store)))
    Equivalent_cycles =  np.divide(np.sum(abs(gene.storage_TS),axis=1)/2 ,[e if e>=1e-15 else 1e-15 for e in energy_storages])/duration_years
    Lifetime = tuple(np.nanmin(np.row_stack((storage_characteristics[7:],np.divide(storage_characteristics[8,:],[max(1e-15,Equivalent_cycles[i]) for i in range(n_store)]))),axis=0))
    Optimized_Load = Non_movable_load+gene.Y_DSM+gene.D_DSM.flatten()
    trades = Optimized_Load-production-np.sum(gene.storage_TS,axis=0)
    importation = Optimized_Load-production-np.sum(gene.storage_TS,axis=0)
    importation[importation<0]=0
    Contract_power=max(importation)
    
    exportation = -(Optimized_Load-production-np.sum(gene.storage_TS,axis=0))
    exportation[exportation<0]=0
    obtained_constraint_level = 1-(sum(importation)/sum(Optimized_Load)) if constraint_num==1 else 1-(sum(exportation)/sum(production))

    SC=10e10+constraint_level-obtained_constraint_level if obtained_constraint_level<constraint_level else -(sum(exportation)/sum(production))  
    return(SC,trades)

### Optimisation criterion Max. power from grid #####

def Max_power_cost(gene,storage_characteristics,time_resolution,n_store,duration_years,specs_num,prices_num,fixed_premium,Overrun,Selling_price,Non_movable_load,Main_grid_emissions,prod_C,prods_U,Bounds_prod,constraint_num,constraint_level,cost_type):
    production = np.dot(gene.production_set,prods_U)/1000+prod_C/1000
    Optimized_Load = Non_movable_load+gene.Y_DSM+gene.D_DSM.flatten()
    trades = Optimized_Load-production-np.sum(gene.storage_TS,axis=0)
    importation = np.where(trades>0,trades,0)
    exportation = np.where(trades<0,trades,0)
    obtained_constraint_level = 1-(sum(importation)/sum(Optimized_Load)) if constraint_num==1 else 1-(sum(exportation)/sum(production))
    Max_power=10e10+constraint_level-obtained_constraint_level if obtained_constraint_level<constraint_level else max(importation)
    return(Max_power,trades)

### Optimisation criterion Energy losses #####

def Losses_cost(gene,storage_characteristics,time_resolution,n_store,duration_years,specs_num,prices_num,Contract_power,fixed_premium,Overrun,Selling_price,Non_movable_load,Main_grid_emissions,prod_C,prods_U,Bounds_prod,constraint_num,constraint_level,cost_type):
    production = np.dot(gene.production_set,prods_U)/1000+prod_C/1000
    losses = np.transpose(np.divide(np.transpose(gene.storage_TS),storage_characteristics[4,:]))-gene.storage_TS 
    losses[losses<0]=0 
    Optimized_Load = Non_movable_load+gene.Y_DSM+gene.D_DSM.flatten()
    trades = Optimized_Load-production-np.sum(gene.storage_TS,axis=0)
    importation = np.where(trades>0,trades,0)
    exportation = np.where(trades<0,trades,0)
    obtained_constraint_level = 1-(sum(importation)/sum(Optimized_Load)) if constraint_num==1 else 1-(sum(exportation)/sum(production))
    sum_losses = 10e10+constraint_level-obtained_constraint_level if obtained_constraint_level<constraint_level else sum(losses)/time_resolution
    return(sum_losses,trades)

### Optimisation criterion Capacity factor #####

def Capacity_factor_cost(gene,storage_characteristics,time_resolution,n_store,duration_years,specs_num,prices_num,fixed_premium,Overrun,Selling_price,Non_movable_load,Main_grid_emissions,prod_C,prods_U,Bounds_prod,constraint_num,constraint_level,cost_type):
    production = np.dot(gene.production_set,prods_U)/1000+prod_C/1000
    Optimized_Load = Non_movable_load+gene.Y_DSM+gene.D_DSM.flatten()
    trades = Optimized_Load-production-np.sum(gene.storage_TS,axis=0)
    importation = np.where(trades>0,trades,0)
    exportation = np.where(trades<0,trades,0)
    obtained_constraint_level = 1-(sum(importation)/sum(Optimized_Load)) if constraint_num==1 else 1-(sum(exportation)/sum(production))
    Capacity_factor = 10e10+constraint_level-obtained_constraint_level if obtained_constraint_level<constraint_level else sum(production)/(max(production)*len(production))
    return(Capacity_factor,trades)

### Optimisation criterion Autonomy #####

def Autonomy_cost(gene,storage_characteristics,time_resolution,n_store,duration_years,specs_num,prices_num,fixed_premium,Overrun,Selling_price,Non_movable_load,Main_grid_emissions,prod_C,prods_U,Bounds_prod,constraint_num,constraint_level,cost_type):
    production = np.dot(gene.production_set,prods_U)/1000+prod_C/1000
    Optimized_Load = Non_movable_load+gene.Y_DSM+gene.D_DSM.flatten()
    trades = Optimized_Load-production-np.sum(gene.storage_TS,axis=0)
    importation = np.where(trades>0,trades,0)
    exportation = np.where(trades<0,trades,0)
    obtained_constraint_level = 1-(sum(importation)/sum(Optimized_Load)) if constraint_num==1 else 1-(sum(exportation)/sum(production))
    Autonomy= 10e10+constraint_level-obtained_constraint_level if obtained_constraint_level<constraint_level else 1-(sum(importation>0)/len(importation))  
    return(Autonomy,trades)

### Optimisation criterion CO2_Emissions #####

def CO2_emissions_cost(gene,storage_characteristics,time_resolution,n_store,duration_years,specs_num,prices_num,fixed_premium,Selling_price,Overrun,Non_movable_load,Main_grid_emissions,prod_C,prods_U,Bounds_prod,constraint_num,constraint_level,cost_type):
    production = np.dot(gene.production_set,prods_U)/1000+prod_C/1000
    Optimized_Load = Non_movable_load+gene.Y_DSM+gene.D_DSM.flatten()
    trades = Optimized_Load-production-np.sum(gene.storage_TS,axis=0)
    importation = np.where(trades>0,trades,0)
    exportation = np.where(trades<0,trades,0)
    obtained_constraint_level = 1-(sum(importation)/sum(Optimized_Load)) if constraint_num==1 else 1-(sum(exportation)/sum(production))
    CO2e= 10e10+constraint_level-obtained_constraint_level if obtained_constraint_level<constraint_level else sum(sum(np.multiply(np.array([gene.production_set[i]*prods_U[i,:] for i in range(len(gene.production_set))]).T/1000,np.array(specs_num[6,:]))))/1000000/time_resolution+np.inner(np.array([sum(gene.storage_TS[i][gene.storage_TS[i]>0]) for i in range(n_store)]),storage_characteristics[6,:])/1000000/time_resolution+sum(importation)*Main_grid_emissions/1000000/time_resolution
    return(CO2e,trades)

### Optimisation criterion fossil fuel consumption #####

def fossil_consumption_cost(gene,storage_characteristics,time_resolution,n_store,duration_years,specs_num,prices_num,fixed_premium,Selling_price,Overrun,Non_movable_load,Main_grid_emissions,prod_C,prods_U,Bounds_prod,constraint_num,constraint_level,cost_type):
    production = np.dot(gene.production_set,prods_U)/1000+prod_C/1000
    Optimized_Load = Non_movable_load+gene.Y_DSM+gene.D_DSM.flatten()
    trades = Optimized_Load-production-np.sum(gene.storage_TS,axis=0)
    importation = np.where(trades>0,trades,0)
    exportation = np.where(trades<0,trades,0)

    obtained_constraint_level = 1-(sum(importation)/sum(Optimized_Load)) if constraint_num==1 else 1-(sum(exportation)/sum(production))
    FFC= 10e10+constraint_level-obtained_constraint_level if obtained_constraint_level<constraint_level else (sum(importation)/time_resolution) 
    return(FFC,trades)

### Optimisation criterion Energy return of investment #####

def EROI_cost(gene,storage_characteristics,time_resolution,n_store,duration_years,specs_num,prices_num,fixed_premium,Overrun,Selling_price,Non_movable_load,Main_grid_emissions,prod_C,prods_U,Bounds_prod,constraint_num,constraint_level,cost_type):
    production = np.dot(gene.production_set,prods_U)/1000+prod_C/1000
    unit_prod=np.sum(np.multiply(gene.production_set,np.transpose(prods_U)),axis=0)/1000/time_resolution
    Optimized_Load = Non_movable_load+gene.Y_DSM+gene.D_DSM.flatten()
    trades = Optimized_Load-production-np.sum(gene.storage_TS,axis=0)
    importation = np.where(trades>0,trades,0)
    exportation = np.where(trades<0,trades,0)
    obtained_constraint_level = 1-(sum(importation)/sum(Optimized_Load)) if constraint_num==1 else 1-(sum(exportation)/sum(production))
    reported_energy=np.sum((gene.storage_TS>0)*gene.storage_TS,axis=1)/time_resolution
    energy_cost = sum(np.divide(reported_energy,storage_characteristics['ESOEI']))+sum(unit_prod/specs_num[5,:])
    EROI= 10e10+constraint_level-obtained_constraint_level if obtained_constraint_level<constraint_level else (sum(production)/time_resolution/energy_cost) 
    return(EROI,trades)

### Finding the proper criterion

def find_cost_functions(criterion_num):
    if criterion_num==1:
        cost_functions = [cost_pre_scenario_LCOE,cost_scenario_LCOE]
    elif criterion_num==2:
        cost_functions = [Self_consumption_cost,Self_consumption_cost]
    elif criterion_num==3:
        cost_functions = [Max_power_cost,Max_power_cost]
    elif criterion_num==4:
        cost_functions = [Losses_cost,Losses_cost]
    elif criterion_num==5:
        cost_functions = [Capacity_factor_cost,Capacity_factor_cost]
    elif criterion_num==6:
        cost_functions = [Autonomy_cost,Autonomy_cost]
    elif criterion_num==7:
        cost_functions = [CO2_emissions_cost,CO2_emissions_cost]
    elif criterion_num==8:
        cost_functions = [fossil_consumption_cost,fossil_consumption_cost]
    elif criterion_num==9:
        cost_functions = [EROI_cost,EROI_cost]
    else : 
        print("No proper optimisation criterion found !")
        sys.exit()
    
    return(cost_functions)


def fitness_list(inputs):
    (population,Contexte)=tuple(inputs[i] for i in range(2))
    
    jitted_pop = ECl.jitting_pop(population)
                    
    fitness_function=find_cost_functions(Contexte.criterion_num)[1]
    fitness_function_ind=lambda ind: fitness_function(ind,Contexte.storage_characteristics,Contexte.time_resolution,Contexte.n_store,Contexte.duration_years,Contexte.specs_num,Contexte.prices_num,Contexte.fixed_premium,Contexte.Overrun,Contexte.Selling_price,Contexte.Non_movable_load,Contexte.Main_grid_emissions,Contexte.prod_C,Contexte.prods_U,Contexte.Bounds_prod,Contexte.constraint_num,Contexte.constraint_level,Contexte.cost_constraint)
    
    fitnesses=[]
    for j in range(len(jitted_pop)):
        fitnesses.append(fitness_function_ind(jitted_pop[j])[0])
       
    return(fitnesses)  
