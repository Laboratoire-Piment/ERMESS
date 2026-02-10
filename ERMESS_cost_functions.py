# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 13:11:54 2024

@author: jlegalla
"""
import pandas as pd
import numpy as np
import sys
from numba import jit
import ERMESS_classes as ECl
import PMS 

def cost_base(gene,datetime,storage_characteristics,time_resolution,n_store,duration_years,specs_num,prices_num,fixed_premium,Overrun,Selling_price,Non_movable_load,D_movable_load,Y_movable_load ,Grid_fossil_fuel_ratio,Main_grid_PoF_ratio,Main_grid_emissions,prod_C,prods_U,Bounds_prod,constraint_num,constraint_level,cost_constraint,n_bits,Connexion,DG_fuel_consumption,DG_fuel_cost,DG_unit_cost,DG_lifetime,DG_maintenance_cost,DG_EROI,fuel_CO2eq_emissions,storage_techs,n_days):

    Load = Non_movable_load+D_movable_load+Y_movable_load
    production = (prod_C)/1000    
    Annual_REN_production = sum(production)/time_resolution/duration_years
    annual_cost_production = 0
    size_power=np.array([0 for i in range(n_store)])
    losses=np.repeat(0,n_bits*n_store).reshape((n_store,n_bits))
    Annual_sum_losses = np.sum(losses,axis=1)/time_resolution/duration_years      

    D_DSM=np.repeat(0,n_bits).reshape((int(n_bits/(24*time_resolution)),24*int(time_resolution)))
    Y_DSM=np.repeat(0,n_bits)
    trades=Load-production        
    Optimized_Load = Load
    Annual_load = sum(Optimized_Load)/time_resolution/duration_years
    EnR_fraction = Annual_REN_production/Annual_load
    storage_TS=np.repeat(0,n_bits*n_store).reshape((n_store,n_bits))
    sum_diff_storages = np.array([-np.cumsum(storage_TS[i,:]/time_resolution+losses[i,:]/time_resolution) for i in range(n_store)])
    reported_energy = np.sum(np.where(storage_TS>0,storage_TS,0),axis=1)/time_resolution/duration_years
    stored_energy = -np.sum(np.where(storage_TS<0,storage_TS,0),axis=1)/time_resolution/duration_years
    power_storage = np.sum(storage_TS,axis=0)
   
    importation=np.where(trades>0,trades,0)
    exportation = np.where(trades<0,-trades,0)
    signal =  Optimized_Load - production      
    Annual_sum_importation=sum(importation)/time_resolution/duration_years
    Annual_sum_exportation=sum(exportation)/time_resolution/duration_years
        
    logicals_sells = np.where((trades<=0) & (signal<=0))[0]
    logicals_buys = np.where((trades>=0) & (signal>=0))[0]
    illogicals_buys = np.where((trades>0) & (signal<=0))[0]
    illogicals_sells = np.where((trades<0) & (signal>=0))[0]
    use_logical_sells = (sum(Optimized_Load[logicals_sells]),-sum(tuple(sum(np.where(storage_TS[i][logicals_sells]<0,storage_TS[i][logicals_sells],0) for i in range(n_store)))),-sum(trades[logicals_sells]))/time_resolution/duration_years-sum(tuple(sum(np.where(storage_TS[i][logicals_sells]>0,storage_TS[i][logicals_sells],0) for i in range(n_store))))
    use_logical_buys = (sum(production[logicals_buys]),sum(tuple(sum(np.where(storage_TS[i][logicals_buys]>0,storage_TS[i][logicals_buys],0) for i in range(n_store)))),sum(trades[logicals_buys]))/time_resolution/duration_years
    use_illogical_sells = (sum(production[illogicals_sells]),sum(tuple(sum(np.where(storage_TS[i][illogicals_sells]>0,storage_TS[i][illogicals_sells],0) for i in range(n_store))))-sum(trades[illogicals_sells]),-sum(trades[illogicals_sells]))/time_resolution/duration_years
    use_illogical_buys = (sum(Optimized_Load[illogicals_buys]),sum(production[illogicals_buys]-sum(Optimized_Load[illogicals_buys])))/time_resolution/duration_years
    
    when_prod_exceeds = pd.DataFrame(data=((sum(Optimized_Load[logicals_sells])+sum(Optimized_Load[illogicals_buys]))/time_resolution/duration_years,-sum(trades[logicals_sells])/time_resolution,sum(trades[illogicals_buys])/time_resolution) + tuple(-sum(storage_TS[i][(signal<0) & (storage_TS[i]<0)])/time_resolution for i in range(n_store)) + tuple(sum(storage_TS[i][(signal<0) & (storage_TS[i]>0)])/time_resolution for i in range(n_store)),index=['Load (kWh)','Grid export (kWh)','Grid import (kWh)']+['Storage '+storage_techs[i]+' charge (kWh)' for i in range(n_store)]+['Storage '+storage_techs[i]+' discharge (kWh)' for i in range(n_store)]).transpose()
    when_load_exceeds = pd.DataFrame(data=(sum(production[logicals_buys])/time_resolution+sum(production[illogicals_sells])/time_resolution,-sum(trades[(signal>0) & (trades<0) ])/time_resolution,sum(trades[(signal>0) & (trades>0) ])/time_resolution) + tuple(-sum(storage_TS[i][(signal>0) & (storage_TS[i]<0)])/time_resolution for i in range(n_store)) + tuple(sum(storage_TS[i][(signal>0) & (storage_TS[i]>0)])/time_resolution for i in range(n_store)),index=['Production (kWh)','Grid export (kWh)','Grid import (kWh)']+['Storage '+storage_techs[i]+' charge (kWh)' for i in range(n_store)]+['Storage '+storage_techs[i]+' discharge (kWh)' for i in range(n_store)]).transpose()
     
    useprod = pd.DataFrame(data=(use_logical_sells[1]+use_illogical_buys[1],use_logical_buys[0]+use_logical_sells[0]+use_illogical_sells[0]+use_illogical_buys[0],use_illogical_sells[2]+use_logical_sells[2] ,Annual_REN_production),index=['Storage (kWh)' ,'Simultaneous load (kWh)','Grid export (kWh)','Annual REN production (kWh)']).transpose()
    Loadmeet = pd.DataFrame(data=(use_logical_buys[1]+use_illogical_sells[1] ,use_logical_buys[0]+use_logical_sells[0]+use_illogical_sells[0]+use_illogical_buys[0],use_logical_buys[2] ,Annual_load),index=['Storage (kWh)', 'Simultaneous production (kWh)','Grid import (kWh)','Annual load (kWh)']).transpose()
    Grid_use_export = pd.DataFrame(data=(useprod['Grid export (kWh)'], Annual_sum_exportation-useprod['Grid export (kWh)']),index=['Production export (kWh)','Storage discharge(kWh)']).transpose()
    Grid_use_import = pd.DataFrame(data=(Loadmeet['Grid import (kWh)'], Annual_sum_importation-Loadmeet['Grid import (kWh)']),index=['To meet load (kWh)','Storage charge(kWh)']).transpose()

    obtained_constraint_level = 1-(sum(np.where(trades>0,trades,0))/sum(Optimized_Load)) if constraint_num==1 else (1+(sum(np.where(trades<0,trades,0))/sum(production)) if sum(production)>0 else 0) 
    energy_storages = np.repeat(0,n_store)
    powers_out = np.repeat(0,n_store)
    powers_in = np.repeat(0,n_store)
    size_power=np.repeat(0,n_store)
    CAPEX_storage_cost = np.repeat(0,n_store)
    Equivalent_cycles = np.repeat(0.,n_store)
    Lifetime = np.repeat(0,n_store)
    
    minSOCs=np.repeat(np.nan,n_store)
    SOCs=(np.divide((sum_diff_storages.T-np.min(sum_diff_storages,axis=1)),energy_storages) + minSOCs).T
    storage_NULL=tuple(energy_storages[i]==0 for i in range(n_store))
    
    State_SOCs=[np.searchsorted([(i+1)/100 for i in range(99)], SOCs[j,:]) for j in range(n_store)]
    distribution_Depth_of_discharge=np.array([[np.count_nonzero(State_SOCs[i]==j)/len(State_SOCs[i]) for j in range(100)] for i in range(n_store)])
    if sum(storage_NULL)>0:
        distribution_Depth_of_discharge[np.where(storage_NULL)[0][0]]=np.repeat(np.nan,100)
    
    dist_DOD = pd.DataFrame(data={'SOC Percentile':[i/100 for i in range(101)]})
    for i in range(n_store):
        dist_DOD = dist_DOD.join(pd.DataFrame(data={'SOC distribution ' + storage_techs[i]:distribution_Depth_of_discharge[i]}))


    Grid_trading = trades if (Connexion=='On-grid') else np.repeat(0,n_bits)
    Grid_importation = importation if (Connexion=='On-grid') else np.repeat(0,n_bits)
    Grid_exportation = exportation if (Connexion=='On-grid') else np.repeat(0,n_bits)  
    DG_production = importation if (Connexion=='Off-grid') else np.repeat(0,n_bits)
    curtailment = exportation if (Connexion=='Off-grid') else np.repeat(0,n_bits)  
        
    annual_CO2eq_prod = 0
    annual_CO2eq_importation = sum(Grid_importation)*Main_grid_emissions/1000000/time_resolution/duration_years
    annual_CO2eq_storage = 0

    annual_fossil_fuel_consumption_importation =  Grid_fossil_fuel_ratio*sum(Grid_importation)/time_resolution/duration_years
    
    productible_energy = 0
    consumed_energy_production = 0
    consumed_energy_storage = sum(np.nanmin(np.array([storage_characteristics[8,:],storage_characteristics[7,:]*Equivalent_cycles]),axis=0)*energy_storages/storage_characteristics[10,:])

    if (Connexion=='Off-grid'):
        DG_nominal_power = max(trades)
        DG_production=importation
        if (DG_nominal_power>0):
            closest_levels = np.array([int(10*(DG_production[i]/DG_nominal_power)-0.5) for i in range(n_bits)])
            annual_fuel_consumption = sum(DG_production*DG_fuel_consumption[closest_levels]/time_resolution/duration_years)
        else : 
            closest_levels = np.array([np.nan for i in range(n_bits)])
            annual_fuel_consumption = 0
        annual_total_fuel_cost = annual_fuel_consumption*DG_fuel_cost
        DG_lifetime_years = DG_lifetime/(sum(np.where(DG_production>0,1,0))/time_resolution/duration_years)
        DG_CAPEX_cost = DG_nominal_power*DG_unit_cost/DG_lifetime_years
        DG_OPEX_cost = sum(DG_production)*DG_maintenance_cost/time_resolution/duration_years
        Contract_power=0
        economics_exportation = 0
        economics_importation = 0
        economics_overrun = 0
        economics_contract_power = 0
        economics_CAPEX_storage = 0
        economics_OPEX_storage = sum(np.multiply(storage_characteristics[3,:],size_power))        
        annual_CO2eq_DG = annual_fuel_consumption*fuel_CO2eq_emissions /1000000   
        productible_energy_DG = sum(DG_production)/time_resolution/duration_years*DG_lifetime_years
        consumed_energy_DG = productible_energy_DG/DG_EROI

    elif (Connexion=='On-grid'):  
        DG_nominal_power = 0
        DG_production=np.repeat(0,n_bits)
        closest_levels = 0
        annual_fuel_consumption = 0
        annual_total_fuel_cost = 0
        DG_lifetime_years = np.nan
        DG_CAPEX_cost = 0
        DG_OPEX_cost = 0          
        Contract_power=max(0,max(trades))
        economics_exportation = np.multiply(exportation,Selling_price).sum()/time_resolution/duration_years
        economics_importation = np.multiply(importation,prices_num).sum()/time_resolution/duration_years
        economics_overrun = max(0,(max(importation)-Contract_power)*Overrun)
        economics_contract_power = fixed_premium*Contract_power
        economics_CAPEX_storage = 0
        economics_OPEX_storage = sum(np.multiply(storage_characteristics[3,:],size_power))
        annual_CO2eq_DG = 0.
        productible_energy_DG = 0.
        consumed_energy_DG = 0.
    
    obtained_self_sufficiency = (1-sum(Grid_importation)/sum(Optimized_Load)) if sum(Optimized_Load )>0 else np.nan
    EnR_self_sufficiency = (1-sum(importation)/sum(Optimized_Load)) if sum(Optimized_Load )>0 else np.nan
    self_consumption = (1-sum(exportation)/sum(production))
    Autonomy = 1-sum(Grid_importation>0)/n_bits
    EnR_autonomy = 1-sum(importation>0)/n_bits
    Capacity_factor = sum(production)/(max(production)*n_bits)
    Max_power_from_grid = max(Grid_importation) 
    Max_power_from_DG = max(DG_production) 
    Max_power_to_grid = max(Grid_exportation) 
    Max_curtailment = max(curtailment)   
        
    #Demand-side management
    indexes_hour = [[int((i+j*time_resolution*24)) for j in range(int(n_days))] for i in range(int(time_resolution*24))]
    Daily_base_load = [np.mean(Non_movable_load[indexes_hour[j]]) for j in range(int(time_resolution*24))]
    Daily_movable_loads = [np.mean(D_movable_load[indexes_hour[j]]+Y_movable_load[indexes_hour[j]]) for j in range(int(time_resolution*24))]
    Daily_final_loads = [np.mean(Optimized_Load[indexes_hour[j]]) for j in range(int(time_resolution*24))]
    Daily_prod = [np.mean(production[indexes_hour[j]]) for j in range(int(time_resolution*24))]
    
    indexes_days = [[int(i+j*time_resolution*24) for i in range(int(time_resolution*24))] for j in range(int(n_days))]
    Yearly_base_load = [np.mean(Non_movable_load[indexes_days[j]]+D_movable_load[indexes_days[j]]) for j in range(int(n_days))]
    Yearly_movable_loads = [np.mean(Y_movable_load[indexes_days[j]]) for j in range(int(n_days))]
    Yearly_final_loads = [np.mean(Optimized_Load[indexes_days[j]]) for j in range(int(n_days))]
    Yearly_prod = [np.mean(production[indexes_days[j]]) for j in range(int(n_days))]
    
    DSM_daily_strategy = pd.DataFrame(data={'Datetime':datetime[0:int(time_resolution*24)].apply(lambda x: x.strftime('%H:%M')),'Daily base load (kW)':Daily_base_load,'Daily movable load (kW)':Daily_movable_loads,'Daily final load (kW)':Daily_final_loads,'Daily prod (kW)':Daily_prod})
    DSM_yearly_strategy = pd.DataFrame(data={'Datetime':datetime[indexes_hour[0]].apply(lambda x: x.strftime('%Y-%m-%d')),'Yearly base load (kW)':Yearly_base_load,'Yearly movable load (kW)':Yearly_movable_loads,'Yearly final load (kW)':Yearly_final_loads,'Yearly prod (kW)':Yearly_prod})
    Load_strategy = pd.DataFrame(data={'Datetime':datetime,'Non controllable load (kW)':Non_movable_load,'Daily movable load (kW)':D_movable_load,'Yearly movable load (kW)':Y_movable_load, 'Daily optimized load (kW)':D_DSM.flatten(),'Yearly optimized load (kW)':Y_DSM})
  
    daily_prod = [np.mean(production[indexes_hour[j]]) for j in range(int(time_resolution*24))]
    yearly_prod = [np.mean(production[indexes_days[j]]) for j in range(int(n_days))]
    daily_importation = [np.mean(importation[indexes_hour[j]]) for j in range(int(time_resolution*24))]
    yearly_importation = [np.mean(importation[indexes_days[j]]) for j in range(int(n_days))]
    daily_exportation = [-np.mean(exportation[indexes_hour[j]]) for j in range(int(time_resolution*24))]
    yearly_exportation = [-np.mean(exportation[indexes_days[j]]) for j in range(int(n_days))]
    daily_storage = [np.mean(power_storage[indexes_hour[j]]) for j in range(int(time_resolution*24))]
    yearly_storage = [np.mean(power_storage[indexes_days[j]]) for j in range(int(n_days))]   
    daily_time_balancing = pd.DataFrame(data={'Datetime':datetime[0:int(time_resolution*24)].apply(lambda x: x.strftime('%H:%M')),'Daily load (kW)':Daily_final_loads,'Daily prod (kW)':daily_prod,'Daily importation (kW)':daily_importation,'Daily exportation (kW)':daily_exportation,'Daily storage (kW)':daily_storage})
    yearly_time_balancing = pd.DataFrame(data={'Datetime':datetime[indexes_hour[0]].apply(lambda x: x.strftime('%Y-%m-%d')),'Yearly load (kW)':Yearly_final_loads,'Yearly prod (kW)':yearly_prod,'Yearly importation (kW)':yearly_importation,'Yearly exportation (kW)':yearly_exportation,'Yearly storage (kW)':yearly_storage})  
    
    
    EROI = (productible_energy+productible_energy_DG)/(consumed_energy_DG+consumed_energy_production+consumed_energy_storage)
    
    annual_fossil_fuel_consumption = annual_fossil_fuel_consumption_importation+annual_fuel_consumption
    annual_CO2eq_total = annual_CO2eq_DG+annual_CO2eq_prod+annual_CO2eq_importation+annual_CO2eq_storage
    LCOE=(annual_cost_production+economics_CAPEX_storage+economics_OPEX_storage+DG_OPEX_cost+DG_CAPEX_cost+annual_total_fuel_cost+economics_importation+economics_contract_power+economics_overrun-economics_exportation)/(sum(Optimized_Load)/time_resolution/duration_years)   if sum(Optimized_Load )>0 else np.nan

    Annual_net_benefits = (-annual_cost_production-economics_CAPEX_storage-economics_OPEX_storage-DG_OPEX_cost-DG_CAPEX_cost-annual_total_fuel_cost-economics_importation-economics_contract_power-economics_overrun+economics_exportation)
    Initial_investment = sum(CAPEX_storage_cost) +0 +0
    fitness_baseline=np.nan

    return{'TimeSeries':{'Storage_TS (kW)':storage_TS,'Storage cumulative energy (kWh)':sum_diff_storages,'SOCs (%)':SOCs,'Losses (kW)': losses,'D_DSM (kW)':D_DSM,'Y_DSM (kW)':Y_DSM,'trades (kW)':trades,'Optimized load (kW)':Optimized_Load,'Non movable load (kW)':Non_movable_load,'production (kW)':production,'Grid trading (kW)':Grid_trading,'Grid importation (kW)':Grid_importation,'Grid exportation (kW)':Grid_exportation,'DG production (kW)':DG_production,'Curtailment (kW)':curtailment},
           'Technical':{'fitness':fitness_baseline,'Self-sufficiency':obtained_self_sufficiency,'Self-consumption':self_consumption,'Autonomy':Autonomy,'EnR fraction':EnR_fraction,'EnR Self-sufficiency':EnR_self_sufficiency,'EnR Autonomy':EnR_autonomy,'Annual sum losses (kWh)':sum(np.sum(losses,axis=1))/time_resolution/duration_years,'Contract power (kW)':Contract_power,'Capacity factor':Capacity_factor,'Max power from grid (kW)':Max_power_from_grid,'Max power to grid (kW)':Max_power_to_grid,'Max curtailment (kW)':Max_curtailment ,'Installation lifetime (yrs.)':np.nan},
           'economics':{'LCOE (€/kWh)':LCOE,'Cost production (€/kWh)':annual_cost_production/(sum(Optimized_Load)/time_resolution/duration_years) ,'OPEX storage (€/kWh)':economics_OPEX_storage/(sum(Optimized_Load)/time_resolution/duration_years),'CAPEX storage (€/kWh)':economics_CAPEX_storage/(sum(Optimized_Load)/time_resolution/duration_years), 'Contract power (€/kWh)':economics_contract_power/(sum(Optimized_Load)/time_resolution/duration_years),'Overrun penalty (€/kWh)':economics_overrun/(sum(Optimized_Load)/time_resolution/duration_years),'Energy importation (€/kWh)':economics_importation/(sum(Optimized_Load)/time_resolution/duration_years),'Fixed premium (€/kWh)':economics_contract_power/(sum(Optimized_Load)/time_resolution/duration_years),'DG fuel cost (€/kWh)':annual_total_fuel_cost/(sum(Optimized_Load)/time_resolution/duration_years),'DG CAPEX cost (€/kWh)':DG_CAPEX_cost/(sum(Optimized_Load)/time_resolution/duration_years),'DG OPEX cost (€/kWh)':DG_OPEX_cost/(sum(Optimized_Load)/time_resolution/duration_years),'Energy exportation (€/kWh)':economics_exportation/(sum(Optimized_Load)/time_resolution/duration_years),'Annual net benefits (€/yrs.)':Annual_net_benefits,'Initial investment (€)':Initial_investment }, 
           'Storages':{'Capacity (kWh)':energy_storages,'Powers in (kW)':powers_in,'Powers out (kW)':powers_out,'Min SOCs (%)':minSOCs ,'Equivalent cycles':Equivalent_cycles,'Storage lifetime (yrs.)':Lifetime}, 
           'DG':{'nominal_power (kW)':DG_nominal_power, 'Max power from DG (kW)':Max_power_from_DG, 'DG lifetime (yrs.)':DG_lifetime_years,'DG min. production':np.nan, 'DG min. runtime':np.nan},
           'DG power distribution':closest_levels,
           'Environment':{'Annual fossil fuel from grid (kWh)':annual_fossil_fuel_consumption_importation,'Annual fossil fuel consumption from DG (kWh)':annual_fuel_consumption,'Annual fossil fuel consumption (kWh)':annual_fossil_fuel_consumption,'Annual CO2eq total (tCO2)':annual_CO2eq_total,'expected produced REN energy (kWh)':productible_energy,'expected produced energy from DG (kWh)':productible_energy_DG,'consumed energy for DG (kWh)':consumed_energy_DG ,'Consumed energy for production (kWh)':consumed_energy_production ,'Consumed energy for storage (kWh)':consumed_energy_storage,'EROI':EROI } ,
           'Flows':{'Annual load (kWh)':Annual_load,'Annual REN production' : Annual_REN_production ,'Sum importation (kWh)':Annual_sum_importation,'Sum exportation (kWh)':Annual_sum_exportation},
           'Flows storages':{'Annual stored energy (kWh)':stored_energy,'Annual reported energy (kWh)':reported_energy,'Annual losses (kWh)':Annual_sum_losses},
           'Extra_outputs':{'Uses':{'useprod':useprod,'Loadmeet':Loadmeet,'when_prod':when_prod_exceeds,'when_load':when_load_exceeds},'Logics':{'logicals_sells':logicals_sells,'logicals_buys':logicals_buys,'illogicals_sells':illogicals_sells,'illogicals_buys':illogicals_buys},'Grid usage':{'export':Grid_use_export,'import':Grid_use_import},'distribution_DOD': dist_DOD },
           'Balancing':{'daily time balancing':daily_time_balancing,'yearly time balancing':yearly_time_balancing},
           'Demand-side management':{'DSM daily strategy':DSM_daily_strategy,'DSM yearly strategy':DSM_yearly_strategy,'Load strategy':Load_strategy}
}

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
    obtained_constraint_level = 1-(sum(importation)/sum(Optimized_Load)) if constraint_num==1 else (1-(sum(exportation)/sum(production)) if sum(production)>0 else 0) 
    total_cost=10e10+constraint_level-obtained_constraint_level if obtained_constraint_level<constraint_level else (annual_cost_production*duration_years+sum(tuple(CAPEX_storage_cost[i]/Lifetime[i] for i in range(n_store)))*duration_years+sum(tuple(storage_characteristics['O&M cost (€/kW-yr.)'][i]*size_power[i]*duration_years for i in range(n_store)))+np.multiply(importation,prices_num[gene.contract]).sum()/time_resolution+fixed_premium[gene.contract]*Contract_power*duration_years+max(0,(max(importation)-Contract_power)*Overrun[gene.contract])+sum(exportation)*Selling_price[gene.contract])/(sum(Optimized_Load)/time_resolution)  
    return(total_cost)

####  PRO COST FUNCTIONS ##################

def KPI_pro(gene,datetime,storage_characteristics,time_resolution,n_store,duration_years,specs_num,prices_num,fixed_premium,Overrun,Selling_price,Non_movable_load,total_D_Movable_load,D_Movable_load, total_Y_Movable_load ,Y_Movable_load ,Grid_fossil_fuel_ratio,Main_grid_PoF_ratio,Main_grid_emissions,prod_C,prods_U,Bounds_prod,constraint_num,constraint_level,cost_constraint,n_bits,Connexion,DG_fuel_consumption,DG_fuel_cost,DG_unit_cost,DG_lifetime,DG_maintenance_cost,DG_EROI,fuel_CO2eq_emissions,storage_techs,n_days):

    production = ((prods_U.T*gene.production_set).sum(axis=1)+prod_C)/1000    
    Annual_REN_production = sum(production)/time_resolution/duration_years
    annual_cost_production = np.sum((specs_num[:,0]/specs_num[:,2]+specs_num[:,1])*(gene.production_set))
    size_power=np.array([max(gene.storages[1:3,i]) for i in range(n_store)])
    
    (storage_TS,trades,D_DSM,Y_DSM,SOCs_eff,losses,P_diff) = PMS.LFE_CCE.py_func(gene, Non_movable_load, total_D_Movable_load, total_Y_Movable_load, production , n_bits,n_store,time_resolution, Connexion, storage_characteristics)
                   
    sum_diff_storages = np.array([-np.cumsum(storage_TS[i,:]/time_resolution+losses[i,:]/time_resolution) for i in range(n_store)])             
    Annual_sum_losses = np.sum(losses,axis=1)/time_resolution/duration_years      
                                          
    Optimized_Load = Non_movable_load+Y_DSM+D_DSM
    Annual_load = sum(Optimized_Load)/time_resolution/duration_years
    EnR_fraction = Annual_REN_production/Annual_load
    importation=np.where(trades>0,trades,0)
    exportation = np.where(trades<0,-trades,0)
    signal =  Optimized_Load - production
    Annual_sum_importation=sum(importation)/time_resolution/duration_years
    Annual_sum_exportation=sum(exportation)/time_resolution/duration_years
        
    logicals_sells = np.where((trades<=0) & (signal<=0))[0]
    logicals_buys = np.where((trades>=0) & (signal>=0))[0]
    illogicals_buys = np.where((trades>0) & (signal<=0))[0]
    illogicals_sells = np.where((trades<0) & (signal>=0))[0]
    use_logical_sells = (sum(Optimized_Load[logicals_sells]),-sum(tuple(sum(np.where(storage_TS[i][logicals_sells]<0,storage_TS[i][logicals_sells],0) for i in range(n_store)))),-sum(trades[logicals_sells]))/time_resolution/duration_years-sum(tuple(sum(np.where(storage_TS[i][logicals_sells]>0,storage_TS[i][logicals_sells],0) for i in range(n_store))))
    use_logical_buys = (sum(production[logicals_buys]),sum(tuple(sum(np.where(storage_TS[i][logicals_buys]>0,storage_TS[i][logicals_buys],0) for i in range(n_store)))),sum(trades[logicals_buys]))/time_resolution/duration_years
    use_illogical_sells = (sum(production[illogicals_sells]),sum(tuple(sum(np.where(storage_TS[i][illogicals_sells]>0,storage_TS[i][illogicals_sells],0) for i in range(n_store))))-sum(trades[illogicals_sells]),-sum(trades[illogicals_sells]))/time_resolution/duration_years
    use_illogical_buys = (sum(Optimized_Load[illogicals_buys]),sum(production[illogicals_buys]-sum(Optimized_Load[illogicals_buys])))/time_resolution/duration_years
    
    when_prod_exceeds = pd.DataFrame(data=((sum(Optimized_Load[logicals_sells])+sum(Optimized_Load[illogicals_buys]))/time_resolution/duration_years,-sum(trades[logicals_sells])/time_resolution,sum(trades[illogicals_buys])/time_resolution) + tuple(-sum(storage_TS[i][(signal<0) & (storage_TS[i]<0)])/time_resolution for i in range(n_store)) + tuple(sum(storage_TS[i][(signal<0) & (storage_TS[i]>0)])/time_resolution for i in range(n_store)),index=['Load (kWh)','Grid export (kWh)','Grid import (kWh)']+['Storage '+storage_techs[i]+' charge (kWh)' for i in range(n_store)]+['Storage '+storage_techs[i]+' discharge (kWh)' for i in range(n_store)]).transpose()
    when_load_exceeds = pd.DataFrame(data=(sum(production[logicals_buys])/time_resolution+sum(production[illogicals_sells])/time_resolution,-sum(trades[(signal>0) & (trades<0) ])/time_resolution,sum(trades[(signal>0) & (trades>0) ])/time_resolution) + tuple(-sum(storage_TS[i][(signal>0) & (storage_TS[i]<0)])/time_resolution for i in range(n_store)) + tuple(sum(storage_TS[i][(signal>0) & (storage_TS[i]>0)])/time_resolution for i in range(n_store)),index=['Production (kWh)','Grid export (kWh)','Grid import (kWh)']+['Storage '+storage_techs[i]+' charge (kWh)' for i in range(n_store)]+['Storage '+storage_techs[i]+' discharge (kWh)' for i in range(n_store)]).transpose()
     
    useprod = pd.DataFrame(data=(use_logical_sells[1]+use_illogical_buys[1],use_logical_buys[0]+use_logical_sells[0]+use_illogical_sells[0]+use_illogical_buys[0],use_illogical_sells[2]+use_logical_sells[2] ,Annual_REN_production),index=['Storage (kWh)' ,'Simultaneous load (kWh)','Grid export (kWh)','Annual REN production (kWh)']).transpose()
    Loadmeet = pd.DataFrame(data=(use_logical_buys[1]+use_illogical_sells[1] ,use_logical_buys[0]+use_logical_sells[0]+use_illogical_sells[0]+use_illogical_buys[0],use_logical_buys[2] ,Annual_load),index=['Storage (kWh)', 'Simultaneous production (kWh)','Grid import (kWh)','Annual load (kWh)']).transpose()
    Grid_use_export = pd.DataFrame(data=(useprod['Grid export (kWh)'], Annual_sum_exportation-useprod['Grid export (kWh)']),index=['Production export (kWh)','Storage discharge(kWh)']).transpose()
    Grid_use_import = pd.DataFrame(data=(Loadmeet['Grid import (kWh)'], Annual_sum_importation-Loadmeet['Grid import (kWh)']),index=['To meet load (kWh)','Storage charge(kWh)']).transpose()
 
    obtained_constraint_level = 1-(sum(np.where(trades>0,trades,0))/sum(Optimized_Load)) if constraint_num==1 else (1+(sum(np.where(trades<0,trades,0))/sum(production)) if sum(production)>0 else 0) 
    energy_storages = gene.storages[0,:]/storage_characteristics[5,:]
    powers_out = gene.storages[2,:]
    powers_in = gene.storages[1,:]
    reported_energy = np.sum(np.where(storage_TS>0,storage_TS,0),axis=1)/time_resolution/duration_years
    stored_energy = -np.sum(np.where(storage_TS<0,storage_TS,0),axis=1)/time_resolution/duration_years
    
    power_storage = np.sum(storage_TS,axis=0)
    
    size_power=np.array([max(powers_in[i],powers_out[i]) for i in range(n_store)])
    CAPEX_storage_cost =  np.multiply(size_power,storage_characteristics[11,:]) + np.multiply(energy_storages,storage_characteristics[0,:]) + np.multiply(storage_characteristics[9,:],(size_power>np.repeat(0,n_store)))
    Equivalent_cycles =  np.array([np.sum(np.abs(storage_TS[i,:]))/(2*time_resolution*max(energy_storages[i],np.float64(1e-15))*duration_years) for i in range(n_store)]) 
    Lifetime = np.array([min(storage_characteristics[7,i],storage_characteristics[8,i]/max(1e-15,Equivalent_cycles[i])) for i in range(n_store)])
    
    minSOCs=(1-storage_characteristics[5,:])/2 
    SOCs=(np.divide((sum_diff_storages.T-np.min(sum_diff_storages,axis=1)),energy_storages) + minSOCs).T
            
    storage_NULL=tuple(energy_storages[i]==0 for i in range(n_store))
    
    State_SOCs=[np.searchsorted([(i+1)/100 for i in range(99)], SOCs[j,:]) for j in range(n_store)]
    distribution_Depth_of_discharge=np.array([[np.count_nonzero(State_SOCs[i]==j)/len(State_SOCs[i]) for j in range(100)] for i in range(n_store)])
    if sum(storage_NULL)>0:
        distribution_Depth_of_discharge[np.where(storage_NULL)[0][0]]=np.repeat(np.nan,100)
    
    dist_DOD = pd.DataFrame(data={'SOC Percentile':[i/100 for i in range(101)]})
    for i in range(n_store):
        dist_DOD = dist_DOD.join(pd.DataFrame(data={'SOC distribution ' + storage_techs[i]:distribution_Depth_of_discharge[i]}))
    
    Grid_trading = trades if (Connexion=='On-grid') else np.repeat(0,n_bits)
    Grid_importation = importation if (Connexion=='On-grid') else np.repeat(0,n_bits)
    Grid_exportation = exportation if (Connexion=='On-grid') else np.repeat(0,n_bits)  
    DG_production = importation if (Connexion=='Off-grid') else np.repeat(0,n_bits)
    curtailment = exportation if (Connexion=='Off-grid') else np.repeat(0,n_bits)  
        
    annual_CO2eq_prod = sum(sum(np.multiply(np.array([gene.production_set[i]*prods_U[i,:] for i in range(len(gene.production_set))]).T/1000,np.array(specs_num[:,4]))))/1000000/time_resolution/duration_years
    annual_CO2eq_importation = sum(Grid_importation)*Main_grid_emissions/1000000/time_resolution/duration_years
    annual_CO2eq_storage = np.inner(np.array([sum(np.where(storage_TS[i]>0,storage_TS[i],0)) for i in range(n_store)]),storage_characteristics[6,:])/1000000/time_resolution/duration_years

    annual_fossil_fuel_consumption_importation =  Grid_fossil_fuel_ratio*sum(Grid_importation)/time_resolution/duration_years
    
    productible_energy = sum(np.sum(np.multiply(gene.production_set,prods_U.T),axis=0)/time_resolution/duration_years*specs_num[:,2])
    consumed_energy_production = sum(productible_energy/specs_num[:,5])
    consumed_energy_storage = sum(np.nanmin(np.array([storage_characteristics[8,:],storage_characteristics[7,:]*Equivalent_cycles]),axis=0)*energy_storages/storage_characteristics[10,:])

     
    if (Connexion=='Off-grid'):
        DG_nominal_power = max(trades)
        DG_production=importation
        if (DG_nominal_power>0):
            closest_levels = np.array([int(10*(DG_production[i]/DG_nominal_power)-0.5) for i in range(n_bits)])
            annual_fuel_consumption = sum(DG_production*DG_fuel_consumption[closest_levels]/time_resolution/duration_years)
        else : 
            closest_levels = np.array([np.nan for i in range(n_bits)])
            annual_fuel_consumption = 0
        annual_total_fuel_cost = annual_fuel_consumption*DG_fuel_cost
        DG_lifetime_years = DG_lifetime/(sum(np.where(DG_production>0,1,0))/time_resolution/duration_years)
        DG_CAPEX_cost = DG_nominal_power*DG_unit_cost/DG_lifetime_years
        DG_OPEX_cost = sum(DG_production)*DG_maintenance_cost/time_resolution/duration_years
        Contract_power=0
        economics_exportation = 0
        economics_importation = 0
        economics_overrun = 0
        economics_contract_power = 0
        economics_CAPEX_storage = sum(np.divide(CAPEX_storage_cost,Lifetime))
        economics_OPEX_storage = sum(np.multiply(storage_characteristics[3,:],size_power))
        annual_CO2eq_DG = annual_fuel_consumption*fuel_CO2eq_emissions  /1000000  
        productible_energy_DG = sum(DG_production)/time_resolution/duration_years*DG_lifetime_years
        consumed_energy_DG = productible_energy_DG/DG_EROI
        
    elif (Connexion=='On-grid'):          
        DG_nominal_power = 0
        DG_production=np.repeat(0,n_bits)
        closest_levels = 0
        annual_fuel_consumption = 0
        annual_total_fuel_cost = 0
        DG_lifetime_years = np.nan
        DG_CAPEX_cost = 0
        DG_OPEX_cost = 0
        Contract_power=max(0,max(trades))
        economics_exportation = np.multiply(exportation,Selling_price[gene.contract,:]).sum()/time_resolution/duration_years
        economics_importation = np.multiply(importation,prices_num[gene.contract,:]).sum()/time_resolution/duration_years
        economics_overrun = max(0,(max(importation)-Contract_power)*Overrun[gene.contract])
        economics_contract_power = fixed_premium[gene.contract]*Contract_power
        economics_CAPEX_storage = sum(np.divide(CAPEX_storage_cost,Lifetime))
        economics_OPEX_storage = sum(np.multiply(storage_characteristics[3,:],size_power))
        annual_CO2eq_DG = 0.
        productible_energy_DG = 0.
        consumed_energy_DG = 0.

    obtained_self_sufficiency = (1-sum(Grid_importation)/sum(Optimized_Load)) if sum(Optimized_Load )>0 else np.nan
    EnR_self_sufficiency = (1-sum(importation)/sum(Optimized_Load)) if sum(Optimized_Load )>0 else np.nan
    self_consumption = (1-sum(exportation)/sum(production))
    Autonomy = 1-sum(Grid_importation>0)/n_bits
    EnR_autonomy = 1-sum(importation>0)/n_bits
    Capacity_factor = sum(production)/(max(production)*n_bits)
    Max_power_from_grid = max(Grid_importation) 
    Max_power_from_DG = max(DG_production) 
    Max_power_to_grid = max(Grid_exportation) 
    Max_curtailment = max(curtailment) 
    
    
    
    #Demand-side management
    indexes_hour = [[int((i+j*time_resolution*24)) for j in range(int(n_days))] for i in range(int(time_resolution*24))]
    Daily_base_load = [np.mean(Non_movable_load[indexes_hour[j]]) for j in range(int(time_resolution*24))]
    Daily_movable_loads = [np.mean(D_Movable_load[indexes_hour[j]]+Y_Movable_load[indexes_hour[j]]) for j in range(int(time_resolution*24))]
    Daily_final_loads = [np.mean(Optimized_Load[indexes_hour[j]]) for j in range(int(time_resolution*24))]
    Daily_prod = [np.mean(production[indexes_hour[j]]) for j in range(int(time_resolution*24))]
    
    indexes_days = [[int(i+j*time_resolution*24) for i in range(int(time_resolution*24))] for j in range(int(n_days))]
    Yearly_base_load = [np.mean(Non_movable_load[indexes_days[j]]+D_Movable_load[indexes_days[j]]) for j in range(int(n_days))]
    Yearly_movable_loads = [np.mean(Y_Movable_load[indexes_days[j]]) for j in range(int(n_days))]
    Yearly_final_loads = [np.mean(Optimized_Load[indexes_days[j]]) for j in range(int(n_days))]
    Yearly_prod = [np.mean(production[indexes_days[j]]) for j in range(int(n_days))]
    
    DSM_daily_strategy = pd.DataFrame(data={'Datetime':datetime[0:int(time_resolution*24)].apply(lambda x: x.strftime('%H:%M')),'Daily base load (kW)':Daily_base_load,'Daily movable load (kW)':Daily_movable_loads,'Daily final load (kW)':Daily_final_loads,'Daily prod (kW)':Daily_prod})
    DSM_yearly_strategy = pd.DataFrame(data={'Datetime':datetime[indexes_hour[0]].apply(lambda x: x.strftime('%Y-%m-%d')),'Yearly base load (kW)':Yearly_base_load,'Yearly movable load (kW)':Yearly_movable_loads,'Yearly final load (kW)':Yearly_final_loads,'Yearly prod (kW)':Yearly_prod})
    Load_strategy = pd.DataFrame(data={'Datetime':datetime,'Non controllable load (kW)':Non_movable_load,'Daily movable load (kW)':D_Movable_load,'Yearly movable load (kW)':Y_Movable_load, 'Daily optimized load (kW)':D_DSM.flatten(),'Yearly optimized load (kW)':Y_DSM})
  
    daily_prod = [np.mean(production[indexes_hour[j]]) for j in range(int(time_resolution*24))]
    yearly_prod = [np.mean(production[indexes_days[j]]) for j in range(int(n_days))]
    daily_importation = [np.mean(importation[indexes_hour[j]]) for j in range(int(time_resolution*24))]
    yearly_importation = [np.mean(importation[indexes_days[j]]) for j in range(int(n_days))]
    daily_exportation = [-np.mean(exportation[indexes_hour[j]]) for j in range(int(time_resolution*24))]
    yearly_exportation = [-np.mean(exportation[indexes_days[j]]) for j in range(int(n_days))]
    daily_storage = [np.mean(power_storage[indexes_hour[j]]) for j in range(int(time_resolution*24))]
    yearly_storage = [np.mean(power_storage[indexes_days[j]]) for j in range(int(n_days))]   
    daily_time_balancing = pd.DataFrame(data={'Datetime':datetime[0:int(time_resolution*24)].apply(lambda x: x.strftime('%H:%M')),'Daily load (kW)':Daily_final_loads,'Daily prod (kW)':daily_prod,'Daily importation (kW)':daily_importation,'Daily exportation (kW)':daily_exportation,'Daily storage (kW)':daily_storage})
    yearly_time_balancing = pd.DataFrame(data={'Datetime':datetime[indexes_hour[0]].apply(lambda x: x.strftime('%Y-%m-%d')),'Yearly load (kW)':Yearly_final_loads,'Yearly prod (kW)':yearly_prod,'Yearly importation (kW)':yearly_importation,'Yearly exportation (kW)':yearly_exportation,'Yearly storage (kW)':yearly_storage})  
    
    
    
    EROI = (productible_energy+productible_energy_DG)/(consumed_energy_DG+consumed_energy_production+consumed_energy_storage)
    
    annual_fossil_fuel_consumption = annual_fossil_fuel_consumption_importation+annual_fuel_consumption
    annual_CO2eq_total = annual_CO2eq_DG+annual_CO2eq_prod+annual_CO2eq_importation+annual_CO2eq_storage
    LCOE=(annual_cost_production+economics_CAPEX_storage+economics_OPEX_storage+DG_OPEX_cost+DG_CAPEX_cost+annual_total_fuel_cost+economics_importation+economics_contract_power+economics_overrun-economics_exportation)/(sum(Optimized_Load)/time_resolution/duration_years)   if sum(Optimized_Load )>0 else np.nan
    
    Lifetime_installation = np.nanmin(np.array([DG_lifetime,min(Lifetime),np.nanmin(np.where(gene.production_set>0,specs_num[:,2],np.nan))]))
    Annual_net_benefits = (-annual_cost_production-economics_CAPEX_storage-economics_OPEX_storage-DG_OPEX_cost-DG_CAPEX_cost-annual_total_fuel_cost-economics_importation-economics_contract_power-economics_overrun+economics_exportation)
    Value = Annual_net_benefits*Lifetime_installation
    
    
    Initial_investment = sum(CAPEX_storage_cost) +np.nanmin((0.,DG_CAPEX_cost*DG_lifetime_years))+ np.sum(specs_num[:,0]*gene.production_set)

    return{'TimeSeries':{'Storage_TS (kW)':storage_TS,'Storage cumulative energy (kWh)':sum_diff_storages,'SOCs (%)':SOCs,'Losses (kW)': losses,'D_DSM (kW)':D_DSM,'Y_DSM (kW)':Y_DSM,'trades (kW)':trades,'Optimized load (kW)':Optimized_Load,'Non movable load (kW)':Non_movable_load,'production (kW)':production,'Grid trading (kW)':Grid_trading,'Grid importation (kW)':Grid_importation,'Grid exportation (kW)':Grid_exportation,'DG production (kW)':DG_production,'Curtailment (kW)':curtailment},
           'Technical':{'fitness':gene.fitness,'Self-sufficiency':obtained_self_sufficiency,'Self-consumption':self_consumption,'Autonomy':Autonomy,'EnR fraction':EnR_fraction,'EnR Self-sufficiency':EnR_self_sufficiency,'EnR Autonomy':EnR_autonomy,'Annual sum losses (kWh)':sum(np.sum(losses,axis=1))/time_resolution/duration_years,'Contract power (kW)':Contract_power,'Capacity factor':Capacity_factor,'Max power from grid (kW)':Max_power_from_grid,'Max power to grid (kW)':Max_power_to_grid,'Max curtailment (kW)':Max_curtailment,'Installation lifetime (yrs.)':Lifetime_installation },
           'economics':{'LCOE (€/kWh)':LCOE,'Cost production (€/kWh)':annual_cost_production/(sum(Optimized_Load)/time_resolution/duration_years) ,'OPEX storage (€/kWh)':economics_OPEX_storage/(sum(Optimized_Load)/time_resolution/duration_years),'CAPEX storage (€/kWh)':economics_CAPEX_storage/(sum(Optimized_Load)/time_resolution/duration_years), 'Contract power (€/kWh)':economics_contract_power/(sum(Optimized_Load)/time_resolution/duration_years),'Overrun penalty (€/kWh)':economics_overrun/(sum(Optimized_Load)/time_resolution/duration_years),'Energy importation (€/kWh)':economics_importation/(sum(Optimized_Load)/time_resolution/duration_years),'Fixed premium (€/kWh)':economics_contract_power/(sum(Optimized_Load)/time_resolution/duration_years),'DG fuel cost (€/kWh)':annual_total_fuel_cost/(sum(Optimized_Load)/time_resolution/duration_years),'DG CAPEX cost (€/kWh)':DG_CAPEX_cost/(sum(Optimized_Load)/time_resolution/duration_years),'DG OPEX cost (€/kWh)':DG_OPEX_cost/(sum(Optimized_Load)/time_resolution/duration_years),'Energy exportation (€/kWh)':economics_exportation/(sum(Optimized_Load)/time_resolution/duration_years),'Annual net benefits (€/yrs.)':Annual_net_benefits,'Value (€)':Value,'Initial investment (€)':Initial_investment}, 
           'Storages':{'Capacity (kWh)':energy_storages,'Powers in (kW)':powers_in,'Powers out (kW)':powers_out,'Min SOCs (%)':minSOCs ,'Equivalent cycles':Equivalent_cycles,'Storage lifetime (yrs.)':Lifetime}, 
           'DG':{'nominal_power (kW)':DG_nominal_power, 'Max power from DG (kW)':Max_power_from_DG, 'DG lifetime (yrs.)':DG_lifetime_years,'DG min. production':gene.PMS_DG_min_production, 'DG min. runtime':gene.PMS_DG_min_runtime },
           'DG power distribution':closest_levels,
           'PMS' : {'D_DSM min. levels':gene.PMS_D_DSM_min_levels,'Y_DSM min. levels':gene.PMS_Y_DSM_min_levels,'discharge order':gene.PMS_discharge_order,'strategy':gene.PMS_strategy, 'taking over':gene.PMS_taking_over,'surplus repartition coefficient':gene.energy_use_repartition_DSM },
           'Environment':{'Annual fossil fuel from grid (kWh)':annual_fossil_fuel_consumption_importation,'Annual fossil fuel consumption from DG (kWh)':annual_fuel_consumption,'Annual fossil fuel consumption (kWh)':annual_fossil_fuel_consumption,'Annual CO2eq total (tCO2)':annual_CO2eq_total,'expected produced REN energy (kWh)':productible_energy,'expected produced energy from DG (kWh)':productible_energy_DG,'consumed energy for DG (kWh)':consumed_energy_DG ,'Consumed energy for production (kWh)':consumed_energy_production ,'Consumed energy for storage (kWh)':consumed_energy_storage,'EROI':EROI },
           'Flows':{'Annual load (kWh)':Annual_load,'Annual REN production' : Annual_REN_production ,'Sum importation (kWh)':Annual_sum_importation,'Sum exportation (kWh)':Annual_sum_exportation},
           'Flows storages':{'Annual stored energy (kWh)':stored_energy,'Annual reported energy (kWh)':reported_energy,'Annual losses (kWh)':Annual_sum_losses},
           'Extra_outputs':{'Uses':{'useprod':useprod,'Loadmeet':Loadmeet,'when_prod':when_prod_exceeds,'when_load':when_load_exceeds},'Logics':{'logicals_sells':logicals_sells,'logicals_buys':logicals_buys,'illogicals_sells':illogicals_sells,'illogicals_buys':illogicals_buys},'Grid usage':{'export':Grid_use_export,'import':Grid_use_import},'distribution_DOD': dist_DOD },
           'Balancing':{'daily time balancing':daily_time_balancing,'yearly time balancing':yearly_time_balancing},
           'Demand-side management':{'DSM daily strategy':DSM_daily_strategy,'DSM yearly strategy':DSM_yearly_strategy,'Load strategy':Load_strategy}
           }


@jit(nopython=True)
#njit(boundscheck=True)
def LCOE_pro(gene,storage_characteristics,time_resolution,n_store,duration_years,specs_num,prices_num,fixed_premium,Overrun,Selling_price,Non_movable_load,total_D_Movable_load, total_Y_Movable_load ,Main_grid_emissions,prod_C,prods_U,Bounds_prod,constraint_num,constraint_level,cost_constraint,n_bits,Connexion,DG_fuel_consumption,DG_fuel_cost,DG_unit_cost,DG_lifetime,DG_maintenance_cost,DG_EROI,fuel_CO2eq_emissions,Grid_fossil_fuel_ratio):

    production = ((prods_U.T*gene.production_set).sum(axis=1)+prod_C)/1000    
    annual_cost_production = np.sum((specs_num[:,0]/specs_num[:,2]+specs_num[:,1])*(gene.production_set))
     
    (storage_TS,trades,D_DSM,Y_DSM,SOCs_eff,losses,P_diff) = PMS.LFE_CCE(gene, Non_movable_load, total_D_Movable_load, total_Y_Movable_load, production , n_bits,n_store,time_resolution, Connexion, storage_characteristics)
    
    for i in range(n_store):
        gene.storages[1][i]=max(-(storage_TS[i]))   
        gene.storages[2][i]=max(storage_TS[i])         
                                                        
    Optimized_Load = Non_movable_load+Y_DSM+D_DSM.flatten()
    importation=np.where(trades>0,trades,0)
    obtained_constraint_level = 1-(sum(importation)/sum(Optimized_Load)) if constraint_num==1 else (1+(sum(np.where(trades<0,trades,0))/sum(production)) if sum(production)>0 else 0) if (constraint_num==2) else sum(production)/sum(Optimized_Load) if(constraint_num==3) else 0

    energy_storages = gene.storages[0,:]/storage_characteristics[5,:]
    powers_out = gene.storages[2,:]
    powers_in = gene.storages[1,:]
    size_power=np.array([max(powers_in[i],powers_out[i]) for i in range(n_store)])
    CAPEX_storage_cost =  np.multiply(size_power,storage_characteristics[11,:]) + np.multiply(energy_storages,storage_characteristics[0,:]) + np.multiply(storage_characteristics[9,:],(size_power>np.repeat(0,n_store)))
    Equivalent_cycles =  np.array([np.sum(np.abs(storage_TS[i,:]))/(2*time_resolution*max(energy_storages[i],np.float64(1e-15))*duration_years) for i in range(n_store)]) 
    Lifetime = np.array([min(storage_characteristics[7,i],storage_characteristics[8,i]/max(1e-15,Equivalent_cycles[i])) for i in range(n_store)])
      
    if (Connexion=='Off-grid'):
        DG_nominal_power = max(trades)
        DG_production=importation
        if (DG_nominal_power>0):
            closest_levels = np.array([int(10*(DG_production[i]/DG_nominal_power)-0.5) for i in range(n_bits)])
            DG_total_fuel_cost = sum(DG_production*DG_fuel_consumption[closest_levels]/time_resolution/duration_years*DG_fuel_cost)
        else : 
            DG_total_fuel_cost=0
        DG_CAPEX_cost = DG_nominal_power*DG_unit_cost*sum(np.where(DG_production>0,1,0))/time_resolution/DG_lifetime/duration_years
        DG_OPEX_cost = sum(DG_production)*DG_maintenance_cost/time_resolution/duration_years
        economics_CAPEX_storage = sum(np.divide(CAPEX_storage_cost,Lifetime))
        economics_OPEX_storage = sum(np.multiply(storage_characteristics[3,:],size_power))
        gene.fitness=(cost_constraint*(constraint_level-obtained_constraint_level) if obtained_constraint_level<constraint_level else 0) + (annual_cost_production+economics_CAPEX_storage+economics_OPEX_storage+DG_CAPEX_cost+DG_OPEX_cost+DG_total_fuel_cost)/(sum(Optimized_Load)/time_resolution/duration_years)  

    elif (Connexion=='On-grid'):            
        exportation = np.where(trades<0,trades,0)
        Contract_power=max(0,max(trades))
        economics_exportation = np.multiply(exportation,Selling_price[gene.contract,:]).sum()/time_resolution/duration_years
        economics_importation = np.multiply(importation,prices_num[gene.contract,:]).sum()/time_resolution/duration_years
        economics_overrun = max(0,(max(importation)-Contract_power)*Overrun[gene.contract])
        economics_contract_power = fixed_premium[gene.contract]*Contract_power
        economics_CAPEX_storage = sum(np.divide(CAPEX_storage_cost,Lifetime))
        economics_OPEX_storage = sum(np.multiply(storage_characteristics[3,:],size_power))
        gene.fitness=(cost_constraint*(constraint_level-obtained_constraint_level) if obtained_constraint_level<constraint_level else 0) + (annual_cost_production+economics_CAPEX_storage+economics_OPEX_storage+economics_importation+economics_contract_power+economics_overrun+economics_exportation)/(sum(Optimized_Load)/time_resolution/duration_years)  

    return(gene)


@jit(nopython=True)
def Self_consumption_pro(gene,storage_characteristics,time_resolution,n_store,duration_years,specs_num,prices_num,fixed_premium,Overrun,Selling_price,Non_movable_load,total_D_Movable_load, total_Y_Movable_load ,Main_grid_emissions,prod_C,prods_U,Bounds_prod,constraint_num,constraint_level,cost_constraint,n_bits,Connexion,DG_fuel_consumption,DG_fuel_cost,DG_unit_cost,DG_lifetime,DG_maintenance_cost,DG_EROI,fuel_CO2eq_emissions,Grid_fossil_fuel_ratio):

    production = ((prods_U.T*gene.production_set).sum(axis=1)+prod_C)/1000    
     
    (storage_TS,trades,D_DSM,Y_DSM,SOCs_eff,losses,P_diff) = PMS.LFE_CCE(gene, Non_movable_load, total_D_Movable_load, total_Y_Movable_load, production , n_bits,n_store,time_resolution, Connexion, storage_characteristics)
    
    for i in range(n_store):
        gene.storages[1][i]=max(-(storage_TS[i]))   
        gene.storages[2][i]=max(storage_TS[i])         
                                                        
    Optimized_Load = Non_movable_load+Y_DSM+D_DSM.flatten()
    importation=np.where(trades>0,trades,0)
    obtained_constraint_level = 1-(sum(importation)/sum(Optimized_Load)) if constraint_num==1 else (1+(sum(np.where(trades<0,trades,0))/sum(production)) if sum(production)>0 else 0) if (constraint_num==2) else sum(production)/sum(Optimized_Load) if(constraint_num==3) else 0

    Self_consumption = 1+(sum(np.where(trades<0,trades,0))/sum(production))
    gene.fitness=(cost_constraint*(constraint_level-obtained_constraint_level) if obtained_constraint_level<constraint_level else 0) - (Self_consumption)  
    return(gene)

@jit(nopython=True)
def Self_sufficiency_pro(gene,storage_characteristics,time_resolution,n_store,duration_years,specs_num,prices_num,fixed_premium,Overrun,Selling_price,Non_movable_load,total_D_Movable_load, total_Y_Movable_load ,Main_grid_emissions,prod_C,prods_U,Bounds_prod,constraint_num,constraint_level,cost_constraint,n_bits,Connexion,DG_fuel_consumption,DG_fuel_cost,DG_unit_cost,DG_lifetime,DG_maintenance_cost,DG_EROI,fuel_CO2eq_emissions,Grid_fossil_fuel_ratio):

    production = ((prods_U.T*gene.production_set).sum(axis=1)+prod_C)/1000    
     
    (storage_TS,trades,D_DSM,Y_DSM,SOCs_eff,losses,P_diff) = PMS.LFE_CCE(gene, Non_movable_load, total_D_Movable_load, total_Y_Movable_load, production , n_bits,n_store,time_resolution, Connexion, storage_characteristics)
    
    for i in range(n_store):
        gene.storages[1][i]=max(-(storage_TS[i]))   
        gene.storages[2][i]=max(storage_TS[i])         
                                                        
    Optimized_Load = Non_movable_load+Y_DSM+D_DSM.flatten()
    importation=np.where(trades>0,trades,0)
    Grid_importation = importation if (Connexion=='On-grid') else np.repeat(0.,n_bits)
    obtained_constraint_level = 1-(sum(importation)/sum(Optimized_Load)) if constraint_num==1 else (1+(sum(np.where(trades<0,trades,0))/sum(production)) if sum(production)>0 else 0) if (constraint_num==2) else sum(production)/sum(Optimized_Load) if(constraint_num==3) else 0

    obtained_self_sufficiency = (1-sum(Grid_importation)/sum(Optimized_Load)) if sum(Optimized_Load )>0 else np.nan

    gene.fitness=(cost_constraint*(constraint_level-obtained_constraint_level) if obtained_constraint_level<constraint_level else 0) - (obtained_self_sufficiency)  
    return(gene)

@jit(nopython=True)
def Max_import_power_pro(gene,storage_characteristics,time_resolution,n_store,duration_years,specs_num,prices_num,fixed_premium,Overrun,Selling_price,Non_movable_load,total_D_Movable_load, total_Y_Movable_load ,Main_grid_emissions,prod_C,prods_U,Bounds_prod,constraint_num,constraint_level,cost_constraint,n_bits,Connexion,DG_fuel_consumption,DG_fuel_cost,DG_unit_cost,DG_lifetime,DG_maintenance_cost,DG_EROI,fuel_CO2eq_emissions,Grid_fossil_fuel_ratio):

    production = ((prods_U.T*gene.production_set).sum(axis=1)+prod_C)/1000    
     
    (storage_TS,trades,D_DSM,Y_DSM,SOCs_eff,losses,P_diff) = PMS.LFE_CCE(gene, Non_movable_load, total_D_Movable_load, total_Y_Movable_load, production , n_bits,n_store,time_resolution, Connexion, storage_characteristics)
    
    for i in range(n_store):
        gene.storages[1][i]=max(-(storage_TS[i]))   
        gene.storages[2][i]=max(storage_TS[i])         
                                                        
    Optimized_Load = Non_movable_load+Y_DSM+D_DSM.flatten()
    importation=np.where(trades>0,trades,0)
    obtained_constraint_level = 1-(sum(importation)/sum(Optimized_Load)) if constraint_num==1 else (1+(sum(np.where(trades<0,trades,0))/sum(production)) if sum(production)>0 else 0) if (constraint_num==2) else sum(production)/sum(Optimized_Load) if(constraint_num==3) else 0

    Max_import = max(importation)
    gene.fitness=(cost_constraint*(constraint_level-obtained_constraint_level) if obtained_constraint_level<constraint_level else 0) + (Max_import)  
    return(gene)

@jit(nopython=True)
def Losses_pro(gene,storage_characteristics,time_resolution,n_store,duration_years,specs_num,prices_num,fixed_premium,Overrun,Selling_price,Non_movable_load,total_D_Movable_load, total_Y_Movable_load ,Main_grid_emissions,prod_C,prods_U,Bounds_prod,constraint_num,constraint_level,cost_constraint,n_bits,Connexion,DG_fuel_consumption,DG_fuel_cost,DG_unit_cost,DG_lifetime,DG_maintenance_cost,DG_EROI,fuel_CO2eq_emissions,Grid_fossil_fuel_ratio):

    production = ((prods_U.T*gene.production_set).sum(axis=1)+prod_C)/1000    
     
    (storage_TS,trades,D_DSM,Y_DSM,SOCs_eff,losses,P_diff) = PMS.LFE_CCE(gene, Non_movable_load, total_D_Movable_load, total_Y_Movable_load, production , n_bits,n_store,time_resolution, Connexion, storage_characteristics)
    
    for i in range(n_store):
        gene.storages[1][i]=max(-(storage_TS[i]))   
        gene.storages[2][i]=max(storage_TS[i])         
                                                        
    Optimized_Load = Non_movable_load+Y_DSM+D_DSM.flatten()
    importation=np.where(trades>0,trades,0)
    obtained_constraint_level = 1-(sum(importation)/sum(Optimized_Load)) if constraint_num==1 else (1+(sum(np.where(trades<0,trades,0))/sum(production)) if sum(production)>0 else 0) if (constraint_num==2) else sum(production)/sum(Optimized_Load) if(constraint_num==3) else 0

    sum_losses = sum(losses)
    gene.fitness=(cost_constraint*(constraint_level-obtained_constraint_level) if obtained_constraint_level<constraint_level else 0) + (sum_losses)  
    return(gene)

@jit(nopython=True)
def Annual_net_benefits_pro(gene,storage_characteristics,time_resolution,n_store,duration_years,specs_num,prices_num,fixed_premium,Overrun,Selling_price,Non_movable_load,total_D_Movable_load, total_Y_Movable_load ,Main_grid_emissions,prod_C,prods_U,Bounds_prod,constraint_num,constraint_level,cost_constraint,n_bits,Connexion,DG_fuel_consumption,DG_fuel_cost,DG_unit_cost,DG_lifetime,DG_maintenance_cost,DG_EROI,fuel_CO2eq_emissions,Grid_fossil_fuel_ratio):

    production = ((prods_U.T*gene.production_set).sum(axis=1)+prod_C)/1000    
    (storage_TS,trades,D_DSM,Y_DSM,SOCs_eff,losses,P_diff) = PMS.LFE_CCE(gene, Non_movable_load, total_D_Movable_load, total_Y_Movable_load, production , n_bits,n_store,time_resolution, Connexion, storage_characteristics)
    
    for i in range(n_store):
        gene.storages[1][i]=max(-(storage_TS[i]))   
        gene.storages[2][i]=max(storage_TS[i])         
                                                        
    Optimized_Load = Non_movable_load+Y_DSM+D_DSM.flatten()
    importation=np.where(trades>0,trades,0)
    obtained_constraint_level = 1-(sum(importation)/sum(Optimized_Load)) if constraint_num==1 else (1+(sum(np.where(trades<0,trades,0))/sum(production)) if sum(production)>0 else 0) if (constraint_num==2) else sum(production)/sum(Optimized_Load) if(constraint_num==3) else 0
    exportation = np.where(trades<0,-trades,0)
    energy_storages = gene.storages[0,:]/storage_characteristics[5,:]
    powers_out = gene.storages[2,:]
    powers_in = gene.storages[1,:]
    size_power=np.array([max(powers_in[i],powers_out[i]) for i in range(n_store)])
    CAPEX_storage_cost =  np.multiply(size_power,storage_characteristics[11,:]) + np.multiply(energy_storages,storage_characteristics[0,:]) + np.multiply(storage_characteristics[9,:],(size_power>np.repeat(0,n_store)))
    Equivalent_cycles =  np.array([np.sum(np.abs(storage_TS[i,:]))/(2*time_resolution*max(energy_storages[i],np.float64(1e-15))*duration_years) for i in range(n_store)]) 
    Lifetime = np.array([min(storage_characteristics[7,i],storage_characteristics[8,i]/max(1e-15,Equivalent_cycles[i])) for i in range(n_store)])
  
    economics_CAPEX_storage = sum(np.divide(CAPEX_storage_cost,Lifetime))
    economics_OPEX_storage = sum(np.multiply(storage_characteristics[3,:],size_power))
    annual_cost_production = np.sum((specs_num[:,0]/specs_num[:,2]+specs_num[:,1])*(gene.production_set))
    
    if (Connexion=='Off-grid'):
        DG_nominal_power = max(trades)
        DG_production=importation
        if (DG_nominal_power>0):
            closest_levels = np.array([int(10*(DG_production[i]/DG_nominal_power)-0.5) for i in range(n_bits)])
            annual_fuel_consumption = sum(DG_production*DG_fuel_consumption[closest_levels]/time_resolution/duration_years)
        else : 
            closest_levels = np.array([np.nan for i in range(n_bits)])
            annual_fuel_consumption = 0
        annual_total_fuel_cost = annual_fuel_consumption*DG_fuel_cost
        DG_lifetime_years = DG_lifetime/(sum(np.where(DG_production>0,1,0))/time_resolution/duration_years)
        DG_CAPEX_cost = DG_nominal_power*DG_unit_cost/DG_lifetime_years
        DG_OPEX_cost = sum(DG_production)*DG_maintenance_cost/time_resolution/duration_years
        economics_exportation = 0
        economics_importation = 0
        economics_overrun = 0
        economics_contract_power = 0
        economics_CAPEX_storage = sum(np.divide(CAPEX_storage_cost,Lifetime))
        economics_OPEX_storage = sum(np.multiply(storage_characteristics[3,:],size_power))
        
    elif (Connexion=='On-grid'):  
        annual_total_fuel_cost = 0
        DG_CAPEX_cost = 0
        DG_OPEX_cost = 0
        DG_CAPEX_cost = 0
        DG_OPEX_cost = 0
        Contract_power=max(0,max(trades))
        economics_exportation = np.multiply(exportation,Selling_price[gene.contract,:]).sum()/time_resolution/duration_years
        economics_importation = np.multiply(importation,prices_num[gene.contract,:]).sum()/time_resolution/duration_years
        economics_overrun = max(0,(max(importation)-Contract_power)*Overrun[gene.contract])
        economics_contract_power = fixed_premium[gene.contract]*Contract_power
        economics_CAPEX_storage = sum(np.divide(CAPEX_storage_cost,Lifetime))
        economics_OPEX_storage = sum(np.multiply(storage_characteristics[3,:],size_power))
        
    Annual_net_benefits = (-annual_cost_production-economics_CAPEX_storage-economics_OPEX_storage-DG_OPEX_cost-DG_CAPEX_cost-annual_total_fuel_cost-economics_importation-economics_contract_power-economics_overrun+economics_exportation)
    fitness=(cost_constraint*(constraint_level-obtained_constraint_level) if obtained_constraint_level<constraint_level else 0) - Annual_net_benefits
    gene.fitness = fitness
    return(gene)

@jit(nopython=True)
def NPV_pro(gene,storage_characteristics,time_resolution,n_store,duration_years,specs_num,prices_num,fixed_premium,Overrun,Selling_price,Non_movable_load,total_D_Movable_load, total_Y_Movable_load ,Main_grid_emissions,prod_C,prods_U,Bounds_prod,constraint_num,constraint_level,cost_constraint,n_bits,Connexion,DG_fuel_consumption,DG_fuel_cost,DG_unit_cost,DG_lifetime,DG_maintenance_cost,DG_EROI,fuel_CO2eq_emissions,Grid_fossil_fuel_ratio):
    production = ((prods_U.T*gene.production_set).sum(axis=1)+prod_C)/1000    
    (storage_TS,trades,D_DSM,Y_DSM,SOCs_eff,losses,P_diff) = PMS.LFE_CCE(gene, Non_movable_load, total_D_Movable_load, total_Y_Movable_load, production , n_bits,n_store,time_resolution, Connexion, storage_characteristics)
    
    for i in range(n_store):
        gene.storages[1][i]=max(-(storage_TS[i]))   
        gene.storages[2][i]=max(storage_TS[i])         
                                                        
    Optimized_Load = Non_movable_load+Y_DSM+D_DSM.flatten()
    importation=np.where(trades>0,trades,0)
    obtained_constraint_level = 1-(sum(importation)/sum(Optimized_Load)) if constraint_num==1 else (1+(sum(np.where(trades<0,trades,0))/sum(production)) if sum(production)>0 else 0) if (constraint_num==2) else sum(production)/sum(Optimized_Load) if(constraint_num==3) else 0
    exportation = np.where(trades<0,-trades,0)
    energy_storages = gene.storages[0,:]/storage_characteristics[5,:]
    powers_out = gene.storages[2,:]
    powers_in = gene.storages[1,:]
    size_power=np.array([max(powers_in[i],powers_out[i]) for i in range(n_store)])
    CAPEX_storage_cost =  np.multiply(size_power,storage_characteristics[11,:]) + np.multiply(energy_storages,storage_characteristics[0,:]) + np.multiply(storage_characteristics[9,:],(size_power>np.repeat(0,n_store)))
    Equivalent_cycles =  np.array([np.sum(np.abs(storage_TS[i,:]))/(2*time_resolution*max(energy_storages[i],np.float64(1e-15))*duration_years) for i in range(n_store)]) 
    Lifetime = np.array([min(storage_characteristics[7,i],storage_characteristics[8,i]/max(1e-15,Equivalent_cycles[i])) for i in range(n_store)])
  
    economics_CAPEX_storage = sum(np.divide(CAPEX_storage_cost,Lifetime))
    economics_OPEX_storage = sum(np.multiply(storage_characteristics[3,:],size_power))
    annual_cost_production = np.sum((specs_num[:,0]/specs_num[:,2]+specs_num[:,1])*(gene.production_set))
    
    if (Connexion=='Off-grid'):
        DG_nominal_power = max(trades)
        DG_production=importation
        if (DG_nominal_power>0):
            closest_levels = np.array([int(10*(DG_production[i]/DG_nominal_power)-0.5) for i in range(n_bits)])
            annual_fuel_consumption = sum(DG_production*DG_fuel_consumption[closest_levels]/time_resolution/duration_years)
        else : 
            closest_levels = np.array([np.nan for i in range(n_bits)])
            annual_fuel_consumption = 0
        annual_total_fuel_cost = annual_fuel_consumption*DG_fuel_cost
        DG_lifetime_years = DG_lifetime/(sum(np.where(DG_production>0,1,0))/time_resolution/duration_years)
        DG_CAPEX_cost = DG_nominal_power*DG_unit_cost/DG_lifetime_years
        DG_OPEX_cost = sum(DG_production)*DG_maintenance_cost/time_resolution/duration_years
        economics_exportation = 0
        economics_importation = 0
        economics_overrun = 0
        economics_contract_power = 0
        economics_CAPEX_storage = sum(np.divide(CAPEX_storage_cost,Lifetime))
        economics_OPEX_storage = sum(np.multiply(storage_characteristics[3,:],size_power))
        
    elif (Connexion=='On-grid'):  
        annual_total_fuel_cost = 0
        DG_CAPEX_cost = 0
        DG_OPEX_cost = 0
        DG_CAPEX_cost = 0
        DG_OPEX_cost = 0
        Contract_power=max(0,max(trades))
        economics_exportation = np.multiply(exportation,Selling_price[gene.contract,:]).sum()/time_resolution/duration_years
        economics_importation = np.multiply(importation,prices_num[gene.contract,:]).sum()/time_resolution/duration_years
        economics_overrun = max(0,(max(importation)-Contract_power)*Overrun[gene.contract])
        economics_contract_power = fixed_premium[gene.contract]*Contract_power
        economics_CAPEX_storage = sum(np.divide(CAPEX_storage_cost,Lifetime))
        economics_OPEX_storage = sum(np.multiply(storage_characteristics[3,:],size_power))
        
    Annual_net_benefits = (-annual_cost_production-economics_CAPEX_storage-economics_OPEX_storage-DG_OPEX_cost-DG_CAPEX_cost-annual_total_fuel_cost-economics_importation-economics_contract_power-economics_overrun+economics_exportation)
    Lifetime_installation = min(DG_lifetime,min(Lifetime),np.nanmin(np.where(gene.production_set>0,specs_num[:,2],np.nan)))
    NPV = Annual_net_benefits*Lifetime_installation
    fitness=(cost_constraint*(constraint_level-obtained_constraint_level) if obtained_constraint_level<constraint_level else 0) - NPV
    gene.fitness = fitness
    return(gene)

@jit(nopython=True)
def Autonomy_pro(gene,storage_characteristics,time_resolution,n_store,duration_years,specs_num,prices_num,fixed_premium,Overrun,Selling_price,Non_movable_load,total_D_Movable_load, total_Y_Movable_load ,Main_grid_emissions,prod_C,prods_U,Bounds_prod,constraint_num,constraint_level,cost_constraint,n_bits,Connexion,DG_fuel_consumption,DG_fuel_cost,DG_unit_cost,DG_lifetime,DG_maintenance_cost,DG_EROI,fuel_CO2eq_emissions,Grid_fossil_fuel_ratio):
    production = ((prods_U.T*gene.production_set).sum(axis=1)+prod_C)/1000    
    (storage_TS,trades,D_DSM,Y_DSM,SOCs_eff,losses,P_diff) = PMS.LFE_CCE(gene, Non_movable_load, total_D_Movable_load, total_Y_Movable_load, production , n_bits,n_store,time_resolution, Connexion, storage_characteristics)
    
    for i in range(n_store):
        gene.storages[1][i]=max(-(storage_TS[i]))   
        gene.storages[2][i]=max(storage_TS[i])    
        
    Optimized_Load = Non_movable_load+Y_DSM+D_DSM                                               
    importation=np.where(trades>0,trades,0)
    obtained_constraint_level = 1-(sum(importation)/sum(Optimized_Load)) if constraint_num==1 else (1+(sum(np.where(trades<0,trades,0))/sum(production)) if sum(production)>0 else 0) if (constraint_num==2) else sum(production)/sum(Optimized_Load) if(constraint_num==3) else 0

    Grid_importation = importation if (Connexion=='On-grid') else np.repeat(0.,n_bits)
    Autonomy = 1-sum(Grid_importation>0)/n_bits
    gene.fitness = (cost_constraint*(constraint_level-obtained_constraint_level) if obtained_constraint_level<constraint_level else 0) - Autonomy

    return(gene)

@jit(nopython=True)
def eqCO2_emissions_pro(gene,storage_characteristics,time_resolution,n_store,duration_years,specs_num,prices_num,fixed_premium,Overrun,Selling_price,Non_movable_load,total_D_Movable_load, total_Y_Movable_load ,Main_grid_emissions,prod_C,prods_U,Bounds_prod,constraint_num,constraint_level,cost_constraint,n_bits,Connexion,DG_fuel_consumption,DG_fuel_cost,DG_unit_cost,DG_lifetime,DG_maintenance_cost,DG_EROI,fuel_CO2eq_emissions,Grid_fossil_fuel_ratio):
    production = ((prods_U.T*gene.production_set).sum(axis=1)+prod_C)/1000    
    (storage_TS,trades,D_DSM,Y_DSM,SOCs_eff,losses,P_diff) = PMS.LFE_CCE(gene, Non_movable_load, total_D_Movable_load, total_Y_Movable_load, production , n_bits,n_store,time_resolution, Connexion, storage_characteristics)
    
    for i in range(n_store):
        gene.storages[1][i]=max(-(storage_TS[i]))   
        gene.storages[2][i]=max(storage_TS[i])    
        
    Optimized_Load = Non_movable_load+Y_DSM+D_DSM                                               
    importation=np.where(trades>0,trades,0)
    obtained_constraint_level = 1-(sum(importation)/sum(Optimized_Load)) if constraint_num==1 else (1+(sum(np.where(trades<0,trades,0))/sum(production)) if sum(production)>0 else 0) if (constraint_num==2) else sum(production)/sum(Optimized_Load) if(constraint_num==3) else 0

    Grid_importation = importation if (Connexion=='On-grid') else np.repeat(0.,n_bits)
    DG_production = importation if (Connexion=='Off-grid') else np.repeat(0.,n_bits)
        
    annual_CO2eq_prod = sum(np.sum((gene.production_set*prods_U.T)/1000*specs_num[:,4],axis=0))/1000000/time_resolution/duration_years   
    annual_CO2eq_importation = sum(Grid_importation)*Main_grid_emissions/1000000/time_resolution/duration_years
    annual_CO2eq_storage = np.inner(np.array([sum(np.where(storage_TS[i]>0,storage_TS[i],0)) for i in range(n_store)]),storage_characteristics[6,:])/1000000/time_resolution/duration_years


    if (Connexion=='Off-grid'):
        DG_nominal_power = max(trades)
        DG_production=importation
        if (DG_nominal_power>0):
            closest_levels = np.array([int(10*(DG_production[i]/DG_nominal_power)-0.5) for i in range(n_bits)])
            annual_fuel_consumption = sum(DG_production*DG_fuel_consumption[closest_levels]/time_resolution/duration_years)
        else : 
            annual_fuel_consumption = 0
        annual_CO2eq_DG = annual_fuel_consumption*fuel_CO2eq_emissions  /1000000      
    elif (Connexion=='On-grid'):          
        annual_CO2eq_DG = 0.

    annual_eqCO2emissions = annual_CO2eq_DG+annual_CO2eq_prod+annual_CO2eq_importation+annual_CO2eq_storage
    gene.fitness = (cost_constraint*(constraint_level-obtained_constraint_level) if obtained_constraint_level<constraint_level else 0) + annual_eqCO2emissions

    return(gene)

@jit(nopython=True)
def Fossil_fuel_consumption_pro(gene,storage_characteristics,time_resolution,n_store,duration_years,specs_num,prices_num,fixed_premium,Overrun,Selling_price,Non_movable_load,total_D_Movable_load, total_Y_Movable_load ,Main_grid_emissions,prod_C,prods_U,Bounds_prod,constraint_num,constraint_level,cost_constraint,n_bits,Connexion,DG_fuel_consumption,DG_fuel_cost,DG_unit_cost,DG_lifetime,DG_maintenance_cost,DG_EROI,fuel_CO2eq_emissions,Grid_fossil_fuel_ratio):
    production = ((prods_U.T*gene.production_set).sum(axis=1)+prod_C)/1000    
    (storage_TS,trades,D_DSM,Y_DSM,SOCs_eff,losses,P_diff) = PMS.LFE_CCE(gene, Non_movable_load, total_D_Movable_load, total_Y_Movable_load, production , n_bits,n_store,time_resolution, Connexion, storage_characteristics)
    
    for i in range(n_store):
        gene.storages[1][i]=max(-(storage_TS[i]))   
        gene.storages[2][i]=max(storage_TS[i])    
        
    Optimized_Load = Non_movable_load+Y_DSM+D_DSM                                               
    importation=np.where(trades>0,trades,0)
    obtained_constraint_level = 1-(sum(importation)/sum(Optimized_Load)) if constraint_num==1 else (1+(sum(np.where(trades<0,trades,0))/sum(production)) if sum(production)>0 else 0) if (constraint_num==2) else sum(production)/sum(Optimized_Load) if(constraint_num==3) else 0

    Grid_importation = importation if (Connexion=='On-grid') else np.repeat(0.,n_bits)
    annual_fossil_fuel_consumption_importation =  Grid_fossil_fuel_ratio*sum(Grid_importation)/time_resolution/duration_years
    
    if (Connexion=='Off-grid'):
        DG_nominal_power = max(trades)
        DG_production=importation
        if (DG_nominal_power>0):
            closest_levels = np.array([int(10*(DG_production[i]/DG_nominal_power)-0.5) for i in range(n_bits)])
            annual_fuel_consumption = sum(DG_production*DG_fuel_consumption[closest_levels]/time_resolution/duration_years)
        else : 
            annual_fuel_consumption = 0    
    elif (Connexion=='On-grid'):          
        annual_fuel_consumption = 0
    
    annual_fossil_fuel_consumption = annual_fossil_fuel_consumption_importation+annual_fuel_consumption
    gene.fitness = (cost_constraint*(constraint_level-obtained_constraint_level) if obtained_constraint_level<constraint_level else 0) + annual_fossil_fuel_consumption

    return(gene)


@jit(nopython=True)
def EROI_pro(gene,storage_characteristics,time_resolution,n_store,duration_years,specs_num,prices_num,fixed_premium,Overrun,Selling_price,Non_movable_load,total_D_Movable_load, total_Y_Movable_load ,Main_grid_emissions,prod_C,prods_U,Bounds_prod,constraint_num,constraint_level,cost_constraint,n_bits,Connexion,DG_fuel_consumption,DG_fuel_cost,DG_unit_cost,DG_lifetime,DG_maintenance_cost,DG_EROI,fuel_CO2eq_emissions,Grid_fossil_fuel_ratio):
    production = ((prods_U.T*gene.production_set).sum(axis=1)+prod_C)/1000    
    (storage_TS,trades,D_DSM,Y_DSM,SOCs_eff,losses,P_diff) = PMS.LFE_CCE(gene, Non_movable_load, total_D_Movable_load, total_Y_Movable_load, production , n_bits,n_store,time_resolution, Connexion, storage_characteristics)
    
    for i in range(n_store):
        gene.storages[1][i]=max(-(storage_TS[i]))   
        gene.storages[2][i]=max(storage_TS[i])    
        
    Optimized_Load = Non_movable_load+Y_DSM+D_DSM                                               
    importation=np.where(trades>0,trades,0)
    obtained_constraint_level = 1-(sum(importation)/sum(Optimized_Load)) if constraint_num==1 else (1+(sum(np.where(trades<0,trades,0))/sum(production)) if sum(production)>0 else 0) if (constraint_num==2) else sum(production)/sum(Optimized_Load) if(constraint_num==3) else 0

    energy_storages = gene.storages[0,:]/storage_characteristics[5,:]              
    Equivalent_cycles =  np.array([np.sum(np.abs(storage_TS[i,:]))/(2*time_resolution*max(energy_storages[i],np.float64(1e-15))*duration_years) for i in range(n_store)]) 

    productible_energy = sum(np.sum(np.multiply(gene.production_set,prods_U.T),axis=0)/time_resolution/duration_years*specs_num[:,2])
    consumed_energy_production = sum(productible_energy/specs_num[:,5])
    consumed_energy_storage = sum(np.array([np.nanmin((storage_characteristics[8,i],storage_characteristics[7,i]*Equivalent_cycles[i]))*energy_storages[i]/storage_characteristics[10,i] for i in range(n_store)]))

     
    if (Connexion=='Off-grid'):
        DG_production=importation
        DG_lifetime_years = DG_lifetime/(sum(np.where(DG_production>0,1,0))/time_resolution/duration_years)
        productible_energy_DG = sum(DG_production)/time_resolution/duration_years*DG_lifetime_years
        consumed_energy_DG = productible_energy_DG/DG_EROI
        
    elif (Connexion=='On-grid'):          
        productible_energy_DG = 0.
        consumed_energy_DG = 0.
        
    EROI = (productible_energy+productible_energy_DG)/(consumed_energy_DG+consumed_energy_production+consumed_energy_storage)
    gene.fitness = (cost_constraint*(constraint_level-obtained_constraint_level) if obtained_constraint_level<constraint_level else 0) - EROI

    return(gene)

####  RESEARCH COST FUNCTIONS ##################

def KPI_research(gene,datetime,storage_characteristics,time_resolution,n_store,duration_years,specs_num,prices_num,fixed_premium,Overrun,Selling_price,Non_movable_load,total_D_Movable_load,D_Movable_load, total_Y_Movable_load ,Y_Movable_load ,Grid_fossil_fuel_ratio,Main_grid_PoF_ratio,Main_grid_emissions,prod_C,prods_U,Bounds_prod,constraint_num,constraint_level,cost_constraint,n_bits,Connexion,DG_fuel_consumption,DG_fuel_cost,DG_unit_cost,DG_lifetime,DG_maintenance_cost,DG_EROI,fuel_CO2eq_emissions,storage_techs,n_days):

    production = ((prods_U.T*gene.production_set).sum(axis=1)+prod_C)/1000    

    Annual_REN_production = sum(production)/time_resolution/duration_years
    annual_cost_production = np.sum((specs_num[:,0]/specs_num[:,2]+specs_num[:,1])*(gene.production_set))

    powers_out = tuple((max(gene.storage_TS[i]) for i in range(n_store)))
    powers_in = tuple((-min(gene.storage_TS[i]) for i in range(n_store)))
    size_power=max(powers_in,powers_out)
    
    storage_TS=gene.storage_TS
    D_DSM=gene.D_DSM.flatten()
    Y_DSM=gene.Y_DSM
    Optimized_Load = Non_movable_load+Y_DSM+D_DSM
    losses = np.array([gene.storage_TS[i]/storage_characteristics[4,:][i]-gene.storage_TS[i] for i in range(n_store)])
    losses[losses<0]=0 
    
    power_storage = np.sum(storage_TS,axis=0)
    trades = Optimized_Load-production-power_storage
                   
    sum_diff_storages = np.array([-np.cumsum(storage_TS[i,:]/time_resolution+losses[i,:]/time_resolution) for i in range(n_store)])             
    Annual_sum_losses = np.sum(losses,axis=1)/time_resolution/duration_years      
                                          
    Annual_load = sum(Optimized_Load)/time_resolution/duration_years
    EnR_fraction = Annual_REN_production/Annual_load
    importation=np.where(trades>0,trades,0)
    exportation = np.where(trades<0,-trades,0)
    signal =  Optimized_Load - production
    Annual_sum_importation=sum(importation)/time_resolution/duration_years
    Annual_sum_exportation=sum(exportation)/time_resolution/duration_years
    
    logicals_sells = np.where((trades<=0) & (signal<=0))[0]
    logicals_buys = np.where((trades>=0) & (signal>=0))[0]
    illogicals_buys = np.where((trades>0) & (signal<=0))[0]
    illogicals_sells = np.where((trades<0) & (signal>=0))[0]
    use_logical_sells = (sum(Optimized_Load[logicals_sells]),-sum(tuple(sum(np.where(storage_TS[i][logicals_sells]<0,storage_TS[i][logicals_sells],0) for i in range(n_store)))),-sum(trades[logicals_sells]))/time_resolution/duration_years-sum(tuple(sum(np.where(storage_TS[i][logicals_sells]>0,storage_TS[i][logicals_sells],0) for i in range(n_store))))
    use_logical_buys = (sum(production[logicals_buys]),sum(tuple(sum(np.where(storage_TS[i][logicals_buys]>0,storage_TS[i][logicals_buys],0) for i in range(n_store)))),sum(trades[logicals_buys]))/time_resolution/duration_years
    use_illogical_sells = (sum(production[illogicals_sells]),sum(tuple(sum(np.where(storage_TS[i][illogicals_sells]>0,storage_TS[i][illogicals_sells],0) for i in range(n_store))))-sum(trades[illogicals_sells]),-sum(trades[illogicals_sells]))/time_resolution/duration_years
    use_illogical_buys = (sum(Optimized_Load[illogicals_buys]),sum(production[illogicals_buys]-sum(Optimized_Load[illogicals_buys])))/time_resolution/duration_years
    
    when_prod_exceeds = pd.DataFrame(data=((sum(Optimized_Load[logicals_sells])+sum(Optimized_Load[illogicals_buys]))/time_resolution/duration_years,-sum(trades[logicals_sells])/time_resolution,sum(trades[illogicals_buys])/time_resolution) + tuple(-sum(storage_TS[i][(signal<0) & (storage_TS[i]<0)])/time_resolution for i in range(n_store)) + tuple(sum(storage_TS[i][(signal<0) & (storage_TS[i]>0)])/time_resolution for i in range(n_store)),index=['Load (kWh)','Grid export (kWh)','Grid import (kWh)']+['Storage '+storage_techs[i]+' charge (kWh)' for i in range(n_store)]+['Storage '+storage_techs[i]+' discharge (kWh)' for i in range(n_store)]).transpose()
    when_load_exceeds = pd.DataFrame(data=(sum(production[logicals_buys])/time_resolution+sum(production[illogicals_sells])/time_resolution,-sum(trades[(signal>0) & (trades<0) ])/time_resolution,sum(trades[(signal>0) & (trades>0) ])/time_resolution) + tuple(-sum(storage_TS[i][(signal>0) & (storage_TS[i]<0)])/time_resolution for i in range(n_store)) + tuple(sum(storage_TS[i][(signal>0) & (storage_TS[i]>0)])/time_resolution for i in range(n_store)),index=['Production (kWh)','Grid export (kWh)','Grid import (kWh)']+['Storage '+storage_techs[i]+' charge (kWh)' for i in range(n_store)]+['Storage '+storage_techs[i]+' discharge (kWh)' for i in range(n_store)]).transpose()
     
    useprod = pd.DataFrame(data=(use_logical_sells[1]+use_illogical_buys[1],use_logical_buys[0]+use_logical_sells[0]+use_illogical_sells[0]+use_illogical_buys[0],use_illogical_sells[2]+use_logical_sells[2] ,Annual_REN_production),index=['Storage (kWh)' ,'Simultaneous load (kWh)','Grid export (kWh)','Annual REN production (kWh)']).transpose()
    Loadmeet = pd.DataFrame(data=(use_logical_buys[1]+use_illogical_sells[1] ,use_logical_buys[0]+use_logical_sells[0]+use_illogical_sells[0]+use_illogical_buys[0],use_logical_buys[2] ,Annual_load),index=['Storage (kWh)', 'Simultaneous production (kWh)','Grid import (kWh)','Annual load (kWh)']).transpose()
    Grid_use_export = pd.DataFrame(data=(useprod['Grid export (kWh)'], Annual_sum_exportation-useprod['Grid export (kWh)']),index=['Production export (kWh)','Storage discharge(kWh)']).transpose()
    Grid_use_import = pd.DataFrame(data=(Loadmeet['Grid import (kWh)'], Annual_sum_importation-Loadmeet['Grid import (kWh)']),index=['To meet load (kWh)','Storage charge(kWh)']).transpose()
 
    obtained_constraint_level = 1-(sum(np.where(trades>0,trades,0))/sum(Optimized_Load)) if constraint_num==1 else (1+(sum(np.where(trades<0,trades,0))/sum(production)) if sum(production)>0 else 0) 
    energy_storages = np.divide(np.array([np.max(sum_diff_storages[i]) - np.min(sum_diff_storages[i]) for i in range(n_store)]),storage_characteristics[5,:])
    reported_energy = np.sum(np.where(storage_TS>0,storage_TS,0),axis=1)/time_resolution/duration_years
    stored_energy = -np.sum(np.where(storage_TS<0,storage_TS,0),axis=1)/time_resolution/duration_years
    
    size_power=np.array([max(powers_in[i],powers_out[i]) for i in range(n_store)])
    CAPEX_storage_cost =  np.multiply(size_power,storage_characteristics[11,:]) + np.multiply(energy_storages,storage_characteristics[0,:]) + np.multiply(storage_characteristics[9,:],(size_power>np.repeat(0,n_store)))
    Equivalent_cycles =  np.array([np.sum(np.abs(storage_TS[i,:]))/(2*time_resolution*max(energy_storages[i],np.float64(1e-15))*duration_years) for i in range(n_store)]) 
    Lifetime = np.array([min(storage_characteristics[7,i],storage_characteristics[8,i]/max(1e-15,Equivalent_cycles[i])) for i in range(n_store)])
    
    minSOCs=(1-storage_characteristics[5,:])/2 
    SOCs=(np.divide((sum_diff_storages.T-np.min(sum_diff_storages,axis=1)),energy_storages) + minSOCs).T
            
    storage_NULL=tuple(energy_storages[i]==0 for i in range(n_store))
    
    State_SOCs=[np.searchsorted([(i+1)/100 for i in range(99)], SOCs[j,:]) for j in range(n_store)]
    distribution_Depth_of_discharge=np.array([[np.count_nonzero(State_SOCs[i]==j)/len(State_SOCs[i]) for j in range(100)] for i in range(n_store)])
    if sum(storage_NULL)>0:
        distribution_Depth_of_discharge[np.where(storage_NULL)[0][0]]=np.repeat(np.nan,100)
    
    dist_DOD = pd.DataFrame(data={'SOC Percentile':[i/100 for i in range(101)]})
    for i in range(n_store):
        dist_DOD = dist_DOD.join(pd.DataFrame(data={'SOC distribution ' + storage_techs[i]:distribution_Depth_of_discharge[i]}))
    
    Grid_trading = trades if (Connexion=='On-grid') else np.repeat(0,n_bits)
    Grid_importation = importation if (Connexion=='On-grid') else np.repeat(0,n_bits)
    Grid_exportation = exportation if (Connexion=='On-grid') else np.repeat(0,n_bits)  
    DG_production = importation if (Connexion=='Off-grid') else np.repeat(0,n_bits)
    curtailment = exportation if (Connexion=='Off-grid') else np.repeat(0,n_bits)  
        
    annual_CO2eq_prod = sum(sum(np.multiply(np.array([gene.production_set[i]*prods_U[i,:] for i in range(len(gene.production_set))]).T/1000,np.array(specs_num[:,4]))))/1000000/time_resolution/duration_years
    annual_CO2eq_importation = sum(Grid_importation)*Main_grid_emissions/1000000/time_resolution/duration_years
    annual_CO2eq_storage = np.inner(np.array([sum(np.where(storage_TS[i]>0,storage_TS[i],0)) for i in range(n_store)]),storage_characteristics[6,:])/1000000/time_resolution/duration_years

    annual_fossil_fuel_consumption_importation =  Grid_fossil_fuel_ratio*sum(Grid_importation)/time_resolution/duration_years
    
    productible_energy = sum(np.sum(np.multiply(gene.production_set,prods_U.T),axis=0)/time_resolution/duration_years*specs_num[:,2])
    consumed_energy_production = sum(productible_energy/specs_num[:,5])
    consumed_energy_storage = sum(np.nanmin(np.array([storage_characteristics[8,:],storage_characteristics[7,:]*Equivalent_cycles]),axis=0)*energy_storages/storage_characteristics[10,:])

    if (Connexion=='Off-grid'):
        DG_nominal_power = max(trades)
        DG_production=importation
        if (DG_nominal_power>0):
            closest_levels = np.array([int(10*(DG_production[i]/DG_nominal_power)-0.5) for i in range(n_bits)])
            annual_fuel_consumption = sum(DG_production*DG_fuel_consumption[closest_levels]/time_resolution/duration_years)
        else : 
            closest_levels = np.array([np.nan for i in range(n_bits)])
            annual_fuel_consumption = 0
        annual_total_fuel_cost = annual_fuel_consumption*DG_fuel_cost
        DG_lifetime_years = DG_lifetime/(sum(np.where(DG_production>0,1,0))/time_resolution/duration_years)
        DG_CAPEX_cost = DG_nominal_power*DG_unit_cost/DG_lifetime_years
        DG_OPEX_cost = sum(DG_production)*DG_maintenance_cost/time_resolution/duration_years
        Contract_power=0
        economics_exportation = 0
        economics_importation = 0
        economics_overrun = 0
        economics_contract_power = 0
        economics_CAPEX_storage = sum(np.divide(CAPEX_storage_cost,Lifetime))
        economics_OPEX_storage = sum(np.multiply(storage_characteristics[3,:],size_power))
        annual_CO2eq_DG = annual_fuel_consumption*fuel_CO2eq_emissions  /1000000  
        productible_energy_DG = sum(DG_production)/time_resolution/duration_years*DG_lifetime_years
        consumed_energy_DG = productible_energy_DG/DG_EROI
        
    elif (Connexion=='On-grid'):          
        DG_nominal_power = 0
        DG_production=np.repeat(0,n_bits)
        closest_levels = 0
        annual_fuel_consumption = 0
        annual_total_fuel_cost = annual_fuel_consumption*DG_fuel_cost
        DG_lifetime_years = np.nan
        DG_CAPEX_cost = 0
        DG_OPEX_cost = 0
        Contract_power=max(0,max(trades))
        economics_exportation = np.multiply(exportation,Selling_price[gene.contract,:]).sum()/time_resolution/duration_years
        economics_importation = np.multiply(importation,prices_num[gene.contract,:]).sum()/time_resolution/duration_years
        economics_overrun = max(0,(max(importation)-Contract_power)*Overrun[gene.contract])
        economics_contract_power = fixed_premium[gene.contract]*Contract_power
        economics_CAPEX_storage = sum(np.divide(CAPEX_storage_cost,Lifetime))
        economics_OPEX_storage = sum(np.multiply(storage_characteristics[3,:],size_power))
        annual_CO2eq_DG = 0.
        productible_energy_DG = 0.
        consumed_energy_DG = 0.

    obtained_self_sufficiency = (1-sum(Grid_importation)/sum(Optimized_Load)) if sum(Optimized_Load )>0 else np.nan
    EnR_self_sufficiency = (1-sum(importation)/sum(Optimized_Load)) if sum(Optimized_Load )>0 else np.nan
    self_consumption = (1-sum(exportation)/sum(production))
    Autonomy = 1-sum(Grid_importation>0)/n_bits
    EnR_autonomy = 1-sum(importation>0)/n_bits
    Capacity_factor = sum(production)/(max(production)*n_bits)
    Max_power_from_grid = max(Grid_importation) 
    Max_power_from_DG = max(DG_production) 
    Max_power_to_grid = max(Grid_exportation) 
    Max_curtailment = max(curtailment) 
    
    
    
    #Demand-side management
    indexes_hour = [[int((i+j*time_resolution*24)) for j in range(int(n_days))] for i in range(int(time_resolution*24))]
    Daily_base_load = [np.mean(Non_movable_load[indexes_hour[j]]) for j in range(int(time_resolution*24))]
    Daily_movable_loads = [np.mean(D_Movable_load[indexes_hour[j]]+Y_Movable_load[indexes_hour[j]]) for j in range(int(time_resolution*24))]
    Daily_final_loads = [np.mean(Optimized_Load[indexes_hour[j]]) for j in range(int(time_resolution*24))]
    Daily_prod = [np.mean(production[indexes_hour[j]]) for j in range(int(time_resolution*24))]
    
    indexes_days = [[int(i+j*time_resolution*24) for i in range(int(time_resolution*24))] for j in range(int(n_days))]
    Yearly_base_load = [np.mean(Non_movable_load[indexes_days[j]]+D_Movable_load[indexes_days[j]]) for j in range(int(n_days))]
    Yearly_movable_loads = [np.mean(Y_Movable_load[indexes_days[j]]) for j in range(int(n_days))]
    Yearly_final_loads = [np.mean(Optimized_Load[indexes_days[j]]) for j in range(int(n_days))]
    Yearly_prod = [np.mean(production[indexes_days[j]]) for j in range(int(n_days))]
    
    DSM_daily_strategy = pd.DataFrame(data={'Datetime':datetime[0:int(time_resolution*24)].apply(lambda x: x.strftime('%H:%M')),'Daily base load (kW)':Daily_base_load,'Daily movable load (kW)':Daily_movable_loads,'Daily final load (kW)':Daily_final_loads,'Daily prod (kW)':Daily_prod})
    DSM_yearly_strategy = pd.DataFrame(data={'Datetime':datetime[indexes_hour[0]].apply(lambda x: x.strftime('%Y-%m-%d')),'Yearly base load (kW)':Yearly_base_load,'Yearly movable load (kW)':Yearly_movable_loads,'Yearly final load (kW)':Yearly_final_loads,'Yearly prod (kW)':Yearly_prod})
    Load_strategy = pd.DataFrame(data={'Datetime':datetime,'Non controllable load (kW)':Non_movable_load,'Daily movable load (kW)':D_Movable_load,'Yearly movable load (kW)':Y_Movable_load, 'Daily optimized load (kW)':D_DSM.flatten(),'Yearly optimized load (kW)':Y_DSM})
  
    daily_prod = [np.mean(production[indexes_hour[j]]) for j in range(int(time_resolution*24))]
    yearly_prod = [np.mean(production[indexes_days[j]]) for j in range(int(n_days))]
    daily_importation = [np.mean(importation[indexes_hour[j]]) for j in range(int(time_resolution*24))]
    yearly_importation = [np.mean(importation[indexes_days[j]]) for j in range(int(n_days))]
    daily_exportation = [-np.mean(exportation[indexes_hour[j]]) for j in range(int(time_resolution*24))]
    yearly_exportation = [-np.mean(exportation[indexes_days[j]]) for j in range(int(n_days))]
    daily_storage = [np.mean(power_storage[indexes_hour[j]]) for j in range(int(time_resolution*24))]
    yearly_storage = [np.mean(power_storage[indexes_days[j]]) for j in range(int(n_days))]   
    daily_time_balancing = pd.DataFrame(data={'Datetime':datetime[0:int(time_resolution*24)].apply(lambda x: x.strftime('%H:%M')),'Daily load (kW)':Daily_final_loads,'Daily prod (kW)':daily_prod,'Daily importation (kW)':daily_importation,'Daily exportation (kW)':daily_exportation,'Daily storage (kW)':daily_storage})
    yearly_time_balancing = pd.DataFrame(data={'Datetime':datetime[indexes_hour[0]].apply(lambda x: x.strftime('%Y-%m-%d')),'Yearly load (kW)':Yearly_final_loads,'Yearly prod (kW)':yearly_prod,'Yearly importation (kW)':yearly_importation,'Yearly exportation (kW)':yearly_exportation,'Yearly storage (kW)':yearly_storage})  
    
    
    
    EROI = (productible_energy+productible_energy_DG)/(consumed_energy_DG+consumed_energy_production+consumed_energy_storage)
    
    annual_fossil_fuel_consumption = annual_fossil_fuel_consumption_importation+annual_fuel_consumption
    annual_CO2eq_total = annual_CO2eq_DG+annual_CO2eq_prod+annual_CO2eq_importation+annual_CO2eq_storage
    LCOE=(annual_cost_production+economics_CAPEX_storage+economics_OPEX_storage+DG_OPEX_cost+DG_CAPEX_cost+annual_total_fuel_cost+economics_importation+economics_contract_power+economics_overrun-economics_exportation)/(sum(Optimized_Load)/time_resolution/duration_years)   if sum(Optimized_Load )>0 else np.nan
    
    Lifetime_installation = min(DG_lifetime,min(Lifetime),np.nanmin(np.where(gene.production_set>0,specs_num[:,2],np.nan)))
    Annual_net_benefits = (-annual_cost_production-economics_CAPEX_storage-economics_OPEX_storage-DG_OPEX_cost-DG_CAPEX_cost-annual_total_fuel_cost-economics_importation-economics_contract_power-economics_overrun+economics_exportation)
    Value = Annual_net_benefits*Lifetime_installation
    
    
    Initial_investment = sum(CAPEX_storage_cost) +np.nanmin((0.,DG_CAPEX_cost*DG_lifetime_years))+ np.sum(specs_num[:,0]*gene.production_set)

    return{'TimeSeries':{'Storage_TS (kW)':storage_TS,'Storage cumulative energy (kWh)':sum_diff_storages,'SOCs (%)':SOCs,'Losses (kW)': losses,'D_DSM (kW)':D_DSM,'Y_DSM (kW)':Y_DSM,'trades (kW)':trades,'Optimized load (kW)':Optimized_Load,'Non movable load (kW)':Non_movable_load,'production (kW)':production,'Grid trading (kW)':Grid_trading,'Grid importation (kW)':Grid_importation,'Grid exportation (kW)':Grid_exportation,'DG production (kW)':DG_production,'Curtailment (kW)':curtailment},
           'Technical':{'fitness':gene.fitness,'Self-sufficiency':obtained_self_sufficiency,'Self-consumption':self_consumption,'Autonomy':Autonomy,'EnR fraction':EnR_fraction,'EnR Self-sufficiency':EnR_self_sufficiency,'EnR Autonomy':EnR_autonomy,'Annual sum losses (kWh)':sum(np.sum(losses,axis=1))/time_resolution/duration_years,'Contract power (kW)':Contract_power,'Capacity factor':Capacity_factor,'Max power from grid (kW)':Max_power_from_grid,'Max power to grid (kW)':Max_power_to_grid,'Max curtailment (kW)':Max_curtailment,'Installation lifetime (yrs.)':Lifetime_installation },
           'economics':{'LCOE (€/kWh)':LCOE,'Cost production (€/kWh)':annual_cost_production/(sum(Optimized_Load)/time_resolution/duration_years) ,'OPEX storage (€/kWh)':economics_OPEX_storage/(sum(Optimized_Load)/time_resolution/duration_years),'CAPEX storage (€/kWh)':economics_CAPEX_storage/(sum(Optimized_Load)/time_resolution/duration_years), 'Contract power (€/kWh)':economics_contract_power/(sum(Optimized_Load)/time_resolution/duration_years),'Overrun penalty (€/kWh)':economics_overrun/(sum(Optimized_Load)/time_resolution/duration_years),'Energy importation (€/kWh)':economics_importation/(sum(Optimized_Load)/time_resolution/duration_years),'Fixed premium (€/kWh)':economics_contract_power/(sum(Optimized_Load)/time_resolution/duration_years),'DG fuel cost (€/kWh)':annual_total_fuel_cost/(sum(Optimized_Load)/time_resolution/duration_years),'DG CAPEX cost (€/kWh)':DG_CAPEX_cost/(sum(Optimized_Load)/time_resolution/duration_years),'DG OPEX cost (€/kWh)':DG_OPEX_cost/(sum(Optimized_Load)/time_resolution/duration_years),'Energy exportation (€/kWh)':economics_exportation/(sum(Optimized_Load)/time_resolution/duration_years),'Annual net benefits (€/yrs.)':Annual_net_benefits,'Value (€)':Value,'Initial investment (€)':Initial_investment}, 
           'Storages':{'Capacity (kWh)':energy_storages,'Powers in (kW)':powers_in,'Powers out (kW)':powers_out,'Min SOCs (%)':minSOCs ,'Equivalent cycles':Equivalent_cycles,'Storage lifetime (yrs.)':Lifetime}, 
           'DG':{'nominal_power (kW)':DG_nominal_power, 'Max power from DG (kW)':Max_power_from_DG, 'DG lifetime (yrs.)':DG_lifetime_years,'DG min. production':np.nan, 'DG min. runtime':np.nan},
           'DG power distribution':closest_levels,
           'Environment':{'Annual fossil fuel from grid (kWh)':annual_fossil_fuel_consumption_importation,'Annual fossil fuel consumption from DG (kWh)':annual_fuel_consumption,'Annual fossil fuel consumption (kWh)':annual_fossil_fuel_consumption,'Annual CO2eq total (tCO2)':annual_CO2eq_total,'expected produced REN energy (kWh)':productible_energy,'expected produced energy from DG (kWh)':productible_energy_DG,'consumed energy for DG (kWh)':consumed_energy_DG ,'Consumed energy for production (kWh)':consumed_energy_production ,'Consumed energy for storage (kWh)':consumed_energy_storage,'EROI':EROI },
           'Flows':{'Annual load (kWh)':Annual_load,'Annual REN production' : Annual_REN_production ,'Sum importation (kWh)':Annual_sum_importation,'Sum exportation (kWh)':Annual_sum_exportation},
           'Flows storages':{'Annual stored energy (kWh)':stored_energy,'Annual reported energy (kWh)':reported_energy,'Annual losses (kWh)':Annual_sum_losses},
           'Extra_outputs':{'Uses':{'useprod':useprod,'Loadmeet':Loadmeet,'when_prod':when_prod_exceeds,'when_load':when_load_exceeds},'Logics':{'logicals_sells':logicals_sells,'logicals_buys':logicals_buys,'illogicals_sells':illogicals_sells,'illogicals_buys':illogicals_buys},'Grid usage':{'export':Grid_use_export,'import':Grid_use_import},'distribution_DOD': dist_DOD },
           'Balancing':{'daily time balancing':daily_time_balancing,'yearly time balancing':yearly_time_balancing},
           'Demand-side management':{'DSM daily strategy':DSM_daily_strategy,'DSM yearly strategy':DSM_yearly_strategy,'Load strategy':Load_strategy}
           }

@jit(nopython=True)
def LCOE_research(gene,storage_characteristics,time_resolution,n_store,duration_years,specs_num,prices_num,fixed_premium,Overrun,Selling_price,Non_movable_load,Main_grid_emissions,DG_fuel_consumption,DG_fuel_cost,DG_lifetime,DG_unit_cost,DG_maintenance_cost,DG_EROI,prod_C,prods_U,Bounds_prod,constraint_num,constraint_level,cost_constraint,Connexion,fuel_CO2eq_emissions,Grid_fossil_fuel_ratio):
    n_bits = len(gene.storage_TS[0])
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
    results_stores=np.array([np.sum(gene.storage_TS[:,i]) for i in range(n_bits)])
    trades = Optimized_Load-production-results_stores
    importation=np.where(trades>0,trades,0)
    exportation = np.where(trades<0,trades,0)
    Contract_power=max(0,max(trades))
    
    if (Connexion=='Off-grid'):
        DG_nominal_power = max(trades)
        DG_production=importation
        if (DG_nominal_power>0):
            closest_levels = np.array([int(10*(DG_production[i]/DG_nominal_power)-0.5) for i in range(n_bits)])
            annual_fuel_consumption = sum(DG_production*DG_fuel_consumption[closest_levels]/time_resolution/duration_years)
        else : 
            annual_fuel_consumption = 0
        annual_total_fuel_cost = annual_fuel_consumption*DG_fuel_cost
        DG_lifetime_years = DG_lifetime/(sum(np.where(DG_production>0,1,0))/time_resolution/duration_years)
        DG_CAPEX_cost = DG_nominal_power*DG_unit_cost/DG_lifetime_years
        DG_OPEX_cost = sum(DG_production)*DG_maintenance_cost/time_resolution/duration_years
        economics_exportation = 0
        economics_importation = 0
        economics_overrun = 0
        economics_contract_power = 0
        economics_CAPEX_storage = sum(np.divide(CAPEX_storage_cost,Lifetime))
        economics_OPEX_storage = sum(np.multiply(storage_characteristics[3,:],size_power))
        
    elif (Connexion=='On-grid'):          
        annual_total_fuel_cost = 0
        DG_CAPEX_cost = 0
        DG_OPEX_cost = 0
        Contract_power=max(0,max(trades))
        economics_exportation = np.multiply(exportation,Selling_price[gene.contract,:]).sum()/time_resolution/duration_years
        economics_importation = np.multiply(importation,prices_num[gene.contract,:]).sum()/time_resolution/duration_years
        economics_overrun = max(0,(max(importation)-Contract_power)*Overrun[gene.contract])
        economics_contract_power = fixed_premium[gene.contract]*Contract_power
        economics_CAPEX_storage = sum(np.divide(CAPEX_storage_cost,Lifetime))
        economics_OPEX_storage = sum(np.multiply(storage_characteristics[3,:],size_power))
    
    LCOE=(annual_cost_production+economics_CAPEX_storage+economics_OPEX_storage+DG_OPEX_cost+DG_CAPEX_cost+annual_total_fuel_cost+economics_importation+economics_contract_power+economics_overrun+economics_exportation)/(sum(Optimized_Load)/time_resolution/duration_years)   if sum(Optimized_Load )>0 else np.nan
 #   total_cost=(cost_constraint*(constraint_level-obtained_constraint_level) if obtained_constraint_level<constraint_level else 0) + (annual_cost_production+sum(np.divide(CAPEX_storage_cost,Lifetime))+sum(np.multiply(storage_characteristics[3,:],size_power))+np.multiply(importation,prices_num[gene.contract,:]).sum()/time_resolution/duration_years+fixed_premium[gene.contract]*Contract_power+max(0,(max(importation)-Contract_power)*Overrun[gene.contract])+np.multiply(exportation,Selling_price[gene.contract,:]).sum()/time_resolution/duration_years)/(sum(Optimized_Load)/time_resolution)  

    obtained_constraint_level = 1-(sum(importation)/sum(Optimized_Load)) if constraint_num==1 else (1+(sum(np.where(trades<0,trades,0))/sum(production)) if sum(production)>0 else 0) if (constraint_num==2) else sum(production)/sum(Optimized_Load) if(constraint_num==3) else 0
    fitness = (cost_constraint*(constraint_level-obtained_constraint_level) if obtained_constraint_level<constraint_level else 0) - LCOE

    return(fitness,trades)

@jit(nopython=True)
def Annual_net_benefits_research(gene,storage_characteristics,time_resolution,n_store,duration_years,specs_num,prices_num,fixed_premium,Overrun,Selling_price,Non_movable_load,Main_grid_emissions,DG_fuel_consumption,DG_fuel_cost,DG_lifetime,DG_unit_cost,DG_maintenance_cost,DG_EROI,prod_C,prods_U,Bounds_prod,constraint_num,constraint_level,cost_constraint,Connexion,fuel_CO2eq_emissions,Grid_fossil_fuel_ratio):

    n_bits = len(gene.storage_TS[0])
    production = ((prods_U.T*gene.production_set).sum(axis=1)+prod_C)/1000    
    annual_cost_production = np.sum((specs_num[:,0]/specs_num[:,2]+specs_num[:,1])*(gene.production_set))
    powers_out = np.array([max(gene.storage_TS[i]) for i in range(n_store)])
    powers_in = np.array([-min(gene.storage_TS[i]) for i in range(n_store)])
    D_DSM=gene.D_DSM.flatten()
    Y_DSM=gene.Y_DSM
    Optimized_Load = Non_movable_load+Y_DSM+D_DSM
    losses = np.transpose(np.divide(np.transpose(gene.storage_TS),storage_characteristics[4,:]))-gene.storage_TS 
    losses=np.where(losses>0,losses,0)
    power_storage = np.sum(gene.storage_TS,axis=0)
    trades = Optimized_Load-production-power_storage
    sum_diff_storages = [-np.cumsum(gene.storage_TS[i,:]+losses[i,:])/time_resolution for i in range(n_store)]     
    importation=np.where(trades>0,trades,0)
    exportation = np.where(trades<0,-trades,0)
    energy_storages = np.divide(np.array([np.max(sum_diff_storages[i]) - np.min(sum_diff_storages[i]) for i in range(n_store)]),storage_characteristics[5,:])

    size_power=np.array([max(powers_in[i],powers_out[i]) for i in range(n_store)])
    CAPEX_storage_cost =  np.multiply(size_power,storage_characteristics[11,:]) + np.multiply(energy_storages,storage_characteristics[0,:]) + np.multiply(storage_characteristics[9,:],(size_power>np.repeat(0,n_store)))
    Equivalent_cycles =  np.array([np.sum(np.abs(gene.storage_TS[i,:]))/(2*time_resolution*max(energy_storages[i],np.float64(1e-15))*duration_years) for i in range(n_store)]) 
    Lifetime = np.array([min(storage_characteristics[7,i],storage_characteristics[8,i]/max(1e-15,Equivalent_cycles[i])) for i in range(n_store)])

    if (Connexion=='Off-grid'):
        DG_nominal_power = max(trades)
        DG_production=importation
        if (DG_nominal_power>0):
            closest_levels = np.array([int(10*(DG_production[i]/DG_nominal_power)-0.5) for i in range(n_bits)])
            annual_fuel_consumption = sum(DG_production*DG_fuel_consumption[closest_levels]/time_resolution/duration_years)
        else : 
            closest_levels = np.array([np.nan for i in range(n_bits)])
            annual_fuel_consumption = 0
        annual_total_fuel_cost = annual_fuel_consumption*DG_fuel_cost
        DG_lifetime_years = DG_lifetime/(sum(np.where(DG_production>0,1,0))/time_resolution/duration_years)
        DG_CAPEX_cost = DG_nominal_power*DG_unit_cost/DG_lifetime_years
        DG_OPEX_cost = sum(DG_production)*DG_maintenance_cost/time_resolution/duration_years
        economics_exportation = 0
        economics_importation = 0
        economics_overrun = 0
        economics_contract_power = 0
        economics_CAPEX_storage = sum(np.divide(CAPEX_storage_cost,Lifetime))
        economics_OPEX_storage = sum(np.multiply(storage_characteristics[3,:],size_power))
        
    elif (Connexion=='On-grid'):          
        DG_nominal_power = 0
        DG_production=np.repeat(0,n_bits)
        annual_fuel_consumption = 0
        annual_total_fuel_cost = annual_fuel_consumption*DG_fuel_cost
        DG_lifetime_years = np.nan
        DG_CAPEX_cost = 0
        DG_OPEX_cost = 0
        Contract_power=max(0,max(trades))
        economics_exportation = np.multiply(exportation,Selling_price[gene.contract,:]).sum()/time_resolution/duration_years
        economics_importation = np.multiply(importation,prices_num[gene.contract,:]).sum()/time_resolution/duration_years
        economics_overrun = max(0,(max(importation)-Contract_power)*Overrun[gene.contract])
        economics_contract_power = fixed_premium[gene.contract]*Contract_power
        economics_CAPEX_storage = sum(np.divide(CAPEX_storage_cost,Lifetime))
        economics_OPEX_storage = sum(np.multiply(storage_characteristics[3,:],size_power))

    obtained_constraint_level = 1-(sum(importation)/sum(Optimized_Load)) if constraint_num==1 else (1+(sum(np.where(trades<0,trades,0))/sum(production)) if sum(production)>0 else 0) if (constraint_num==2) else sum(production)/sum(Optimized_Load) if(constraint_num==3) else 0


    Annual_net_benefits = (-annual_cost_production-economics_CAPEX_storage-economics_OPEX_storage-DG_OPEX_cost-DG_CAPEX_cost-annual_total_fuel_cost-economics_importation-economics_contract_power-economics_overrun+economics_exportation)
    fitness = (cost_constraint*(constraint_level-obtained_constraint_level) if obtained_constraint_level<constraint_level else 0) - Annual_net_benefits

    return(fitness,trades)

@jit(nopython=True)
def NPV_research(gene,storage_characteristics,time_resolution,n_store,duration_years,specs_num,prices_num,fixed_premium,Overrun,Selling_price,Non_movable_load,Main_grid_emissions,DG_fuel_consumption,DG_fuel_cost,DG_lifetime,DG_unit_cost,DG_maintenance_cost,DG_EROI,prod_C,prods_U,Bounds_prod,constraint_num,constraint_level,cost_constraint,Connexion,fuel_CO2eq_emissions,Grid_fossil_fuel_ratio):

    n_bits = len(gene.storage_TS[0])
    production = ((prods_U.T*gene.production_set).sum(axis=1)+prod_C)/1000    

    annual_cost_production = np.sum((specs_num[:,0]/specs_num[:,2]+specs_num[:,1])*(gene.production_set))
    powers_out = np.array([max(gene.storage_TS[i]) for i in range(n_store)])
    powers_in = np.array([-min(gene.storage_TS[i]) for i in range(n_store)])
    D_DSM=gene.D_DSM.flatten()
    Y_DSM=gene.Y_DSM
    Optimized_Load = Non_movable_load+Y_DSM+D_DSM
    losses = np.transpose(np.divide(np.transpose(gene.storage_TS),storage_characteristics[4,:]))-gene.storage_TS 
    losses=np.where(losses>0,losses,0)
    power_storage = np.sum(gene.storage_TS,axis=0)
    trades = Optimized_Load-production-power_storage
    sum_diff_storages = [-np.cumsum(gene.storage_TS[i,:]+losses[i,:])/time_resolution for i in range(n_store)]     
    importation=np.where(trades>0,trades,0)
    exportation = np.where(trades<0,-trades,0)
    energy_storages = np.divide(np.array([np.max(sum_diff_storages[i]) - np.min(sum_diff_storages[i]) for i in range(n_store)]),storage_characteristics[5,:])

    size_power=np.array([max(powers_in[i],powers_out[i]) for i in range(n_store)])
    CAPEX_storage_cost =  np.multiply(size_power,storage_characteristics[11,:]) + np.multiply(energy_storages,storage_characteristics[0,:]) + np.multiply(storage_characteristics[9,:],(size_power>np.repeat(0,n_store)))
    Equivalent_cycles =  np.array([np.sum(np.abs(gene.storage_TS[i,:]))/(2*time_resolution*max(energy_storages[i],np.float64(1e-15))*duration_years) for i in range(n_store)]) 
    Lifetime = np.array([min(storage_characteristics[7,i],storage_characteristics[8,i]/max(1e-15,Equivalent_cycles[i])) for i in range(n_store)])

    if (Connexion=='Off-grid'):
        DG_nominal_power = max(trades)
        DG_production=importation
        if (DG_nominal_power>0):
            closest_levels = np.array([int(10*(DG_production[i]/DG_nominal_power)-0.5) for i in range(n_bits)])
            annual_fuel_consumption = sum(DG_production*DG_fuel_consumption[closest_levels]/time_resolution/duration_years)
        else : 
            closest_levels = np.array([np.nan for i in range(n_bits)])
            annual_fuel_consumption = 0
        annual_total_fuel_cost = annual_fuel_consumption*DG_fuel_cost
        DG_lifetime_years = DG_lifetime/(sum(np.where(DG_production>0,1,0))/time_resolution/duration_years)
        DG_CAPEX_cost = DG_nominal_power*DG_unit_cost/DG_lifetime_years
        DG_OPEX_cost = sum(DG_production)*DG_maintenance_cost/time_resolution/duration_years
        economics_exportation = 0
        economics_importation = 0
        economics_overrun = 0
        economics_contract_power = 0
        economics_CAPEX_storage = sum(np.divide(CAPEX_storage_cost,Lifetime))
        economics_OPEX_storage = sum(np.multiply(storage_characteristics[3,:],size_power))
        
    elif (Connexion=='On-grid'):          
        DG_nominal_power = 0
        DG_production=np.repeat(0,n_bits)
        annual_fuel_consumption = 0
        annual_total_fuel_cost = annual_fuel_consumption*DG_fuel_cost
        DG_lifetime_years = np.nan
        DG_CAPEX_cost = 0
        DG_OPEX_cost = 0
        Contract_power=max(0,max(trades))
        economics_exportation = np.multiply(exportation,Selling_price[gene.contract,:]).sum()/time_resolution/duration_years
        economics_importation = np.multiply(importation,prices_num[gene.contract,:]).sum()/time_resolution/duration_years
        economics_overrun = max(0,(max(importation)-Contract_power)*Overrun[gene.contract])
        economics_contract_power = fixed_premium[gene.contract]*Contract_power
        economics_CAPEX_storage = sum(np.divide(CAPEX_storage_cost,Lifetime))
        economics_OPEX_storage = sum(np.multiply(storage_characteristics[3,:],size_power))

    obtained_constraint_level = 1-(sum(importation)/sum(Optimized_Load)) if constraint_num==1 else (1+(sum(np.where(trades<0,trades,0))/sum(production)) if sum(production)>0 else 0) if (constraint_num==2) else sum(production)/sum(Optimized_Load) if(constraint_num==3) else 0

    Lifetime_installation = min(DG_lifetime,min(Lifetime),np.nanmin(np.where(gene.production_set>0,specs_num[:,2],np.nan)))
    Annual_net_benefits = (-annual_cost_production-economics_CAPEX_storage-economics_OPEX_storage-DG_OPEX_cost-DG_CAPEX_cost-annual_total_fuel_cost-economics_importation-economics_contract_power-economics_overrun+economics_exportation)
    NPV=Annual_net_benefits*Lifetime_installation
    fitness = (cost_constraint*(constraint_level-obtained_constraint_level) if obtained_constraint_level<constraint_level else 0) - NPV
    return(fitness,trades)

@jit(nopython=True)
def Max_import_power_research(gene,storage_characteristics,time_resolution,n_store,duration_years,specs_num,prices_num,fixed_premium,Overrun,Selling_price,Non_movable_load,Main_grid_emissions,DG_fuel_consumption,DG_fuel_cost,DG_lifetime,DG_unit_cost,DG_maintenance_cost,DG_EROI,prod_C,prods_U,Bounds_prod,constraint_num,constraint_level,cost_constraint,Connexion,fuel_CO2eq_emissions,Grid_fossil_fuel_ratio):
    
    production = ((prods_U.T*gene.production_set).sum(axis=1)+prod_C)/1000    
    storage_TS=gene.storage_TS
    D_DSM=gene.D_DSM.flatten()
    Y_DSM=gene.Y_DSM
    Optimized_Load = Non_movable_load+Y_DSM+D_DSM
    
    losses = np.transpose(np.divide(np.transpose(gene.storage_TS),storage_characteristics[4,:]))-gene.storage_TS 
    losses = np.where(losses>0,losses,0)
    
    power_storage = np.sum(storage_TS,axis=0)
    trades = Optimized_Load-production-power_storage
    importation=np.where(trades>0,trades,0)

    Max_import = max(importation) 
    
    obtained_constraint_level = 1-(sum(importation)/sum(Optimized_Load)) if constraint_num==1 else (1+(sum(np.where(trades<0,trades,0))/sum(production)) if sum(production)>0 else 0) if (constraint_num==2) else sum(production)/sum(Optimized_Load) if(constraint_num==3) else 0
    fitness = (cost_constraint*(constraint_level-obtained_constraint_level) if obtained_constraint_level<constraint_level else 0) + Max_import
    
    return(fitness,trades)


@jit(nopython=True)
def Losses_research(gene,storage_characteristics,time_resolution,n_store,duration_years,specs_num,prices_num,fixed_premium,Overrun,Selling_price,Non_movable_load,Main_grid_emissions,DG_fuel_consumption,DG_fuel_cost,DG_lifetime,DG_unit_cost,DG_maintenance_cost,DG_EROI,prod_C,prods_U,Bounds_prod,constraint_num,constraint_level,cost_constraint,Connexion,fuel_CO2eq_emissions,Grid_fossil_fuel_ratio):
    
    production = ((prods_U.T*gene.production_set).sum(axis=1)+prod_C)/1000    
    storage_TS=gene.storage_TS
    D_DSM=gene.D_DSM.flatten()
    Y_DSM=gene.Y_DSM
    Optimized_Load = Non_movable_load+Y_DSM+D_DSM
    
    losses = np.transpose(np.divide(np.transpose(gene.storage_TS),storage_characteristics[4,:]))-gene.storage_TS 
    losses = np.where(losses>0,losses,0)
    
    power_storage = np.sum(storage_TS,axis=0)
    trades = Optimized_Load-production-power_storage
                   
    Annual_sum_losses = np.sum(losses,axis=1)/time_resolution/duration_years 

    obtained_constraint_level = 1-(sum(np.where(trades>0,trades,0))/sum(Optimized_Load)) if constraint_num==1 else (1+(sum(np.where(trades<0,trades,0))/sum(production)) if sum(production)>0 else 0) if (constraint_num==2) else sum(production)/sum(Optimized_Load) if(constraint_num==3) else 0    
    fitness = (cost_constraint*(constraint_level-obtained_constraint_level) if obtained_constraint_level<constraint_level else 0) + Annual_sum_losses
    
    return(fitness,trades)

@jit(nopython=True)
def eqCO2_emissions_research(gene,storage_characteristics,time_resolution,n_store,duration_years,specs_num,prices_num,fixed_premium,Overrun,Selling_price,Non_movable_load,Main_grid_emissions,DG_fuel_consumption,DG_fuel_cost,DG_lifetime,DG_unit_cost,DG_maintenance_cost,DG_EROI,prod_C,prods_U,Bounds_prod,constraint_num,constraint_level,cost_constraint,Connexion,fuel_CO2eq_emissions,Grid_fossil_fuel_ratio):
    
    n_bits = len(gene.storage_TS[0])
    production = ((prods_U.T*gene.production_set).sum(axis=1)+prod_C)/1000    
    storage_TS=gene.storage_TS
    D_DSM=gene.D_DSM.flatten()
    Y_DSM=gene.Y_DSM
    Optimized_Load = Non_movable_load+Y_DSM+D_DSM
    
    power_storage = np.sum(storage_TS,axis=0)
    trades = Optimized_Load-production-power_storage

    importation=np.where(trades>0,trades,0)    
    Grid_importation = np.where(trades>0,trades,0) if (Connexion=='On-grid') else np.repeat(0.,n_bits)
    
    annual_CO2eq_prod = sum(np.sum((gene.production_set*prods_U.T)/1000*specs_num[:,4],axis=0))/1000000/time_resolution/duration_years   
    annual_CO2eq_importation = sum(Grid_importation)*Main_grid_emissions/1000000/time_resolution/duration_years
    annual_CO2eq_storage = sum(np.array([sum(np.where(storage_TS[i]>0,storage_TS[i],0)) for i in range(n_store)])[:]*storage_characteristics[6,:])/1000000/time_resolution/duration_years
    if (Connexion=='Off-grid'):
        DG_nominal_power = max(trades)
        DG_production=importation
        if (DG_nominal_power>0):
            closest_levels = np.array([int(10*(DG_production[i]/DG_nominal_power)-0.5) for i in range(n_bits)])
            annual_fuel_consumption = sum(DG_production*DG_fuel_consumption[closest_levels]/time_resolution/duration_years)
        else : 
            annual_fuel_consumption = 0
        annual_CO2eq_DG = annual_fuel_consumption*fuel_CO2eq_emissions  /1000000      
    elif (Connexion=='On-grid'):          
        annual_CO2eq_DG = 0.

    obtained_constraint_level = 1-(sum(np.where(trades>0,trades,0))/sum(Optimized_Load)) if constraint_num==1 else (1+(sum(np.where(trades<0,trades,0))/sum(production)) if sum(production)>0 else 0) if (constraint_num==2) else sum(production)/sum(Optimized_Load) if(constraint_num==3) else 0     
    annual_CO2eq_total = annual_CO2eq_DG+annual_CO2eq_prod+annual_CO2eq_importation+annual_CO2eq_storage
    fitness = (cost_constraint*(constraint_level-obtained_constraint_level) if obtained_constraint_level<constraint_level else 0) + annual_CO2eq_total
    return(fitness,trades)

@jit(nopython=True)
def Autonomy_research(gene,storage_characteristics,time_resolution,n_store,duration_years,specs_num,prices_num,fixed_premium,Overrun,Selling_price,Non_movable_load,Main_grid_emissions,DG_fuel_consumption,DG_fuel_cost,DG_lifetime,DG_unit_cost,DG_maintenance_cost,DG_EROI,prod_C,prods_U,Bounds_prod,constraint_num,constraint_level,cost_constraint,Connexion,fuel_CO2eq_emissions,Grid_fossil_fuel_ratio):
    
    n_bits = len(gene.storage_TS[0])
    production = ((prods_U.T*gene.production_set).sum(axis=1)+prod_C)/1000    
    storage_TS=gene.storage_TS
    D_DSM=gene.D_DSM.flatten()
    Y_DSM=gene.Y_DSM
    Optimized_Load = Non_movable_load+Y_DSM+D_DSM
    
    power_storage = np.sum(storage_TS,axis=0)
    trades = Optimized_Load-production-power_storage
    
    Grid_importation = np.where(trades>0,trades,0) if (Connexion=='On-grid') else np.repeat(0.,n_bits)

    obtained_constraint_level = 1-(sum(np.where(trades>0,trades,0))/sum(Optimized_Load)) if constraint_num==1 else (1+(sum(np.where(trades<0,trades,0))/sum(production)) if sum(production)>0 else 0) if (constraint_num==2) else sum(production)/sum(Optimized_Load) if(constraint_num==3) else 0    
    Autonomy = 1-sum(Grid_importation>0)/n_bits
    fitness = (cost_constraint*(constraint_level-obtained_constraint_level) if obtained_constraint_level<constraint_level else 0) - Autonomy
    
    return(fitness,trades)


@jit(nopython=True)
def Fossil_fuel_consumption_research(gene,storage_characteristics,time_resolution,n_store,duration_years,specs_num,prices_num,fixed_premium,Overrun,Selling_price,Non_movable_load,Main_grid_emissions,DG_fuel_consumption,DG_fuel_cost,DG_lifetime,DG_unit_cost,DG_maintenance_cost,DG_EROI,prod_C,prods_U,Bounds_prod,constraint_num,constraint_level,cost_constraint,Connexion,fuel_CO2eq_emissions,Grid_fossil_fuel_ratio):
    
    n_bits = len(gene.storage_TS[0])
    production = ((prods_U.T*gene.production_set).sum(axis=1)+prod_C)/1000    
    storage_TS=gene.storage_TS
    D_DSM=gene.D_DSM.flatten()
    Y_DSM=gene.Y_DSM
    Optimized_Load = Non_movable_load+Y_DSM+D_DSM
    
    power_storage = np.sum(storage_TS,axis=0)
    trades = Optimized_Load-production-power_storage

    importation=np.where(trades>0,trades,0)
    Grid_importation = importation if (Connexion=='On-grid') else np.repeat(0.,n_bits)    
    annual_fossil_fuel_consumption_importation =  Grid_fossil_fuel_ratio*sum(Grid_importation)/time_resolution/duration_years

    if (Connexion=='Off-grid'):
        DG_nominal_power = max(trades)
        DG_production=importation
        if (DG_nominal_power>0):
            closest_levels = np.array([int(10*(DG_production[i]/DG_nominal_power)-0.5) for i in range(n_bits)])
            annual_fuel_consumption = sum(DG_production*DG_fuel_consumption[closest_levels]/time_resolution/duration_years)
        else : 
            annual_fuel_consumption = 0
        
    elif (Connexion=='On-grid'):          
        annual_fuel_consumption = 0

    obtained_constraint_level = 1-(sum(np.where(trades>0,trades,0))/sum(Optimized_Load)) if constraint_num==1 else (1+(sum(np.where(trades<0,trades,0))/sum(production)) if sum(production)>0 else 0)  if (constraint_num==2) else sum(production)/sum(Optimized_Load) if(constraint_num==3) else 0    
    annual_fossil_fuel_consumption = annual_fossil_fuel_consumption_importation+annual_fuel_consumption
    fitness = (cost_constraint*(constraint_level-obtained_constraint_level) if obtained_constraint_level<constraint_level else 0) + annual_fossil_fuel_consumption
    
    return(fitness,trades)

@jit(nopython=True)
def EROI_research(gene,storage_characteristics,time_resolution,n_store,duration_years,specs_num,prices_num,fixed_premium,Overrun,Selling_price,Non_movable_load,Main_grid_emissions,DG_fuel_consumption,DG_fuel_cost,DG_lifetime,DG_unit_cost,DG_maintenance_cost,DG_EROI,prod_C,prods_U,Bounds_prod,constraint_num,constraint_level,cost_constraint,Connexion,fuel_CO2eq_emissions,Grid_fossil_fuel_ratio):
    
    production = ((prods_U.T*gene.production_set).sum(axis=1)+prod_C)/1000    
    storage_TS=gene.storage_TS
    D_DSM=gene.D_DSM.flatten()
    Y_DSM=gene.Y_DSM
    Optimized_Load = Non_movable_load+Y_DSM+D_DSM
    
    losses = np.transpose(np.divide(np.transpose(gene.storage_TS),storage_characteristics[4,:]))-gene.storage_TS 
    losses = np.where(losses>0,losses,0)
    
    sum_diff_storages = [-np.cumsum(gene.storage_TS[i,:]+losses[i,:])/time_resolution for i in range(n_store)]            
    
    power_storage = np.sum(storage_TS,axis=0)
    trades = Optimized_Load-production-power_storage
    importation=np.where(trades>0,trades,0)
    
    energy_storages = np.divide(np.array([np.max(sum_diff_storages[i]) - np.min(sum_diff_storages[i]) for i in range(n_store)]),storage_characteristics[5,:])
    
    Equivalent_cycles =  np.array([np.sum(np.abs(storage_TS[i,:]))/(2*time_resolution*max(energy_storages[i],np.float64(1e-15))*duration_years) for i in range(n_store)]) 

    
    productible_energy = sum(np.sum(np.multiply(gene.production_set,prods_U.T),axis=0)/time_resolution/duration_years*specs_num[:,2])
    consumed_energy_production = sum(productible_energy/specs_num[:,5])
    consumed_energy_storage = sum(np.array([np.nanmin((storage_characteristics[8,i],storage_characteristics[7,i]*Equivalent_cycles[i]))*energy_storages[i]/storage_characteristics[10,i] for i in range(n_store)]))
    
    if (Connexion=='Off-grid'):
        DG_production=importation
        DG_lifetime_years = DG_lifetime/(sum(np.where(DG_production>0,1,0))/time_resolution/duration_years)
        productible_energy_DG = sum(DG_production)/time_resolution/duration_years*DG_lifetime_years
        consumed_energy_DG = productible_energy_DG/DG_EROI
        
    elif (Connexion=='On-grid'):          
        productible_energy_DG = 0.
        consumed_energy_DG = 0.

    obtained_constraint_level = 1-(sum(np.where(trades>0,trades,0))/sum(Optimized_Load)) if constraint_num==1 else (1+(sum(np.where(trades<0,trades,0))/sum(production)) if sum(production)>0 else 0)  if (constraint_num==2) else sum(production)/sum(Optimized_Load) if(constraint_num==3) else 0    
    EROI = (productible_energy+productible_energy_DG)/(consumed_energy_DG+consumed_energy_production+consumed_energy_storage)
    fitness = (cost_constraint*(constraint_level-obtained_constraint_level) if obtained_constraint_level<constraint_level else 0) - EROI
    return(fitness,trades)

@jit(nopython=True)
def Self_consumption_research(gene,storage_characteristics,time_resolution,n_store,duration_years,specs_num,prices_num,fixed_premium,Overrun,Selling_price,Non_movable_load,Main_grid_emissions,DG_fuel_consumption,DG_fuel_cost,DG_lifetime,DG_unit_cost,DG_maintenance_cost,DG_EROI,prod_C,prods_U,Bounds_prod,constraint_num,constraint_level,cost_constraint,Connexion,fuel_CO2eq_emissions,Grid_fossil_fuel_ratio):
    
    production = ((prods_U.T*gene.production_set).sum(axis=1)+prod_C)/1000    
    storage_TS=gene.storage_TS
    D_DSM=gene.D_DSM.flatten()
    Y_DSM=gene.Y_DSM
    Optimized_Load = Non_movable_load+Y_DSM+D_DSM
    
    power_storage = np.sum(storage_TS,axis=0)
    trades = Optimized_Load-production-power_storage

    exportation = np.where(trades<0,-trades,0)

    obtained_constraint_level = 1-(sum(np.where(trades>0,trades,0))/sum(Optimized_Load)) if constraint_num==1 else (1+(sum(np.where(trades<0,trades,0))/sum(production)) if sum(production)>0 else 0) if (constraint_num==2) else sum(production)/sum(Optimized_Load) if(constraint_num==3) else 0  
    obtained_self_consumption = (1-sum(exportation)/sum(production))
    fitness = (cost_constraint*(constraint_level-obtained_constraint_level) if obtained_constraint_level<constraint_level else 0) - obtained_self_consumption
    
    return(fitness,trades)



@jit(nopython=True)
def Self_sufficiency_research(gene,storage_characteristics,time_resolution,n_store,duration_years,specs_num,prices_num,fixed_premium,Overrun,Selling_price,Non_movable_load,Main_grid_emissions,DG_fuel_consumption,DG_fuel_cost,DG_lifetime,DG_unit_cost,DG_maintenance_cost,DG_EROI,prod_C,prods_U,Bounds_prod,constraint_num,constraint_level,cost_constraint,Connexion,fuel_CO2eq_emissions,Grid_fossil_fuel_ratio):
    
    n_bits = len(gene.storage_TS[0])
    production = ((prods_U.T*gene.production_set).sum(axis=1)+prod_C)/1000    
    storage_TS=gene.storage_TS
    D_DSM=gene.D_DSM.flatten()
    Y_DSM=gene.Y_DSM
    Optimized_Load = Non_movable_load+Y_DSM+D_DSM
    
    power_storage = np.sum(storage_TS,axis=0)
    trades = Optimized_Load-production-power_storage

    importation=np.where(trades>0,trades,0)
    
    Grid_importation = importation if (Connexion=='On-grid') else np.repeat(0.,n_bits)

    obtained_constraint_level = 1-(sum(np.where(trades>0,trades,0))/sum(Optimized_Load)) if constraint_num==1 else (1+(sum(np.where(trades<0,trades,0))/sum(production)) if sum(production)>0 else 0) if (constraint_num==2) else sum(production)/sum(Optimized_Load) if(constraint_num==3) else 0    
    obtained_self_sufficiency = (1-sum(Grid_importation)/sum(Optimized_Load)) if sum(Optimized_Load )>0 else np.nan
    fitness = (cost_constraint*(constraint_level-obtained_constraint_level) if obtained_constraint_level<constraint_level else 0) - obtained_self_sufficiency
    
    return(fitness,trades)

### Finding the proper criterion

def find_cost_functions(Contexte):
    if (Contexte.type_optim == 'research') :
        if Contexte.criterion_num==1:
            cost_functions = LCOE_research
        elif Contexte.criterion_num==2:
            cost_functions = Annual_net_benefits_research
        elif Contexte.criterion_num==3:
            cost_functions = NPV_research
        elif Contexte.criterion_num==4:
            cost_functions = Self_sufficiency_research
        elif Contexte.criterion_num==5:
            cost_functions = Self_consumption_research
        elif Contexte.criterion_num==6:
            cost_functions = Autonomy_research
        elif Contexte.criterion_num==7:
            cost_functions = eqCO2_emissions_research
        elif Contexte.criterion_num==8:
            cost_functions = Fossil_fuel_consumption_research
        elif Contexte.criterion_num==9:
            cost_functions = EROI_research
        elif Contexte.criterion_num==10:
            cost_functions = Losses_research
        elif Contexte.criterion_num==11:
            cost_functions = Max_import_power_research
        else : 
            print("No proper optimisation criterion found !")
            sys.exit()
        cost_functions_ind=lambda ind: cost_functions(ind,Contexte.storage_characteristics,Contexte.time_resolution,Contexte.n_store,Contexte.duration_years,Contexte.specs_num,Contexte.prices_num,Contexte.fixed_premium,Contexte.Overrun,Contexte.Selling_price,Contexte.Non_movable_load,Contexte.Main_grid_emissions,Contexte.DG_fuel_consumption,Contexte.DG_fuel_cost,Contexte.DG_lifetime,Contexte.DG_unit_cost,Contexte.DG_maintenance_cost,Contexte.DG_EROI,Contexte.prod_C,Contexte.prods_U,Contexte.Bounds_prod,Contexte.constraint_num,Contexte.constraint_level,Contexte.cost_constraint,Contexte.Connexion,Contexte.fuel_CO2eq_emissions,Contexte.Grid_Fossil_fuel_ratio)                                                            
    elif (Contexte.type_optim == 'pro') :
        if Contexte.criterion_num==1:
            cost_functions = LCOE_pro
        elif Contexte.criterion_num==2:
            cost_functions = Annual_net_benefits_pro
        elif Contexte.criterion_num==3:
            cost_functions = NPV_pro
        elif Contexte.criterion_num==4:
            cost_functions = Self_sufficiency_pro
        elif Contexte.criterion_num==5:
            cost_functions = Self_consumption_pro
        elif Contexte.criterion_num==6:
            cost_functions = Autonomy_pro
        elif Contexte.criterion_num==7:
            cost_functions = eqCO2_emissions_pro
        elif Contexte.criterion_num==8:
            cost_functions = Fossil_fuel_consumption_pro
        elif Contexte.criterion_num==9:
            cost_functions = EROI_pro
        elif Contexte.criterion_num==10:
            cost_functions = Losses_pro
        elif Contexte.criterion_num==11:
            cost_functions = Max_import_power_pro
        else : 
            print("No proper optimisation criterion found !")
            sys.exit()
        cost_functions_ind=lambda ind: cost_functions(ind,Contexte.storage_characteristics,Contexte.time_resolution,Contexte.n_store,Contexte.duration_years,Contexte.specs_num,Contexte.prices_num,Contexte.fixed_premium,Contexte.Overrun,Contexte.Selling_price,Contexte.Non_movable_load,Contexte.total_D_Movable_load,Contexte.total_Y_Movable_load,Contexte.Main_grid_emissions,Contexte.prod_C,Contexte.prods_U,Contexte.Bounds_prod,Contexte.constraint_num,Contexte.constraint_level,Contexte.cost_constraint,Contexte.n_bits,Contexte.Connexion,Contexte.DG_fuel_consumption,Contexte.DG_fuel_cost,Contexte.DG_unit_cost,Contexte.DG_lifetime,Contexte.DG_maintenance_cost,Contexte.DG_EROI,Contexte.fuel_CO2eq_emissions,Contexte.Grid_Fossil_fuel_ratio)
    return(cost_functions_ind)

def fitness_list(inputs):
    (population,Contexte)=tuple(inputs[i] for i in range(2))
    
    if (Contexte.type_optim == 'research' ):   
        jitted_pop = ECl.jitting_pop_res(population)
    elif (Contexte.type_optim == 'pro' ):
        jitted_pop = ECl.jitting_pop_pro(population)
                    
    fitness_function=find_cost_functions(Contexte)
    
    for j in range(len(jitted_pop)):
        jitted_pop[j]=(fitness_function(jitted_pop[j])[0])
       
    return(jitted_pop)  
