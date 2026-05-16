# -*- coding: utf-8 -*-
"""
Created on Mon May 11 16:54:12 2026

@author: JoPHOBEA
"""

import pandas as pd
import numpy as np 

from ERMESS_scripts.energy_model import ERMESS_EMS_models as Eems
from ERMESS_scripts.evolutionnary_core import ERMESS_functions as Ef
from ERMESS_scripts.data.indices import *

def annualize(values, Context):
    if values.ndim == 0:
        return(values/ Context.time.time_resolution/ Context.time.duration_years)
    elif values.ndim == 1:
        return (np.sum(values)/ Context.time.time_resolution/ Context.time.duration_years)
    else:
        return (np.sum(values, axis=1)/ Context.time.time_resolution / Context.time.duration_years)


def _compute_net_load_kpis(Context,solution,production,dispatching_timeseries):
    optimized_load = Context.loads.non_movable+dispatching_timeseries["Y_DSM"]+dispatching_timeseries["D_DSM"]
    net_load = optimized_load - production
    
    return {"production":production,"optimized_load":optimized_load,"net_load": net_load}


def _compute_balancing_core(Context,dispatching_timeseries,KPI_net_load):
    """
    Compute global balancing indicators of the system.
    """
    trades = dispatching_timeseries["trades"]
    
    total_storage_power = np.sum(dispatching_timeseries["storage_TS"], axis=0)

    importation = np.where(trades > 0, trades, 0)
    exportation = np.where(trades < 0, -trades, 0)

    return {"importation": importation,"exportation": exportation,"total_storage_power": total_storage_power}

def _compute_EMS_kpis(solution, Context):
    genset_min_runtime = solution.DG_min_runtime / Context.time.time_resolution
    return {'D_DSM min. levels':solution.D_DSM_minimum_levels,'Y_DSM min. levels':solution.Y_DSM_minimum_levels,'discharge order':solution.discharge_order,'strategy':solution.DG_strategy, 'overlaps':solution.overlaps,'energy repartition coefficient':solution.energy_use_coefficient,'genset min. production (kW)':solution.DG_min_production , 'genset min. runtime (h)':genset_min_runtime }

def _compute_energy_flows(Context,KPI_net_load,KPI_core):
    """
    Compute energy exchanges in the system.
    """
    n_bits = Context.time.n_bits
    annual_load = annualize(KPI_net_load["optimized_load"], Context)
    annual_REN_production = annualize(KPI_net_load["production"], Context)


    if Context.optimization.connexion == "On-grid":
        grid_importation = KPI_core["importation"]
        grid_exportation = KPI_core["exportation"]
        grid_trading = KPI_core["trades (kW)"]

        genset_production = np.zeros(n_bits)
        curtailment = np.zeros(n_bits)

    else:
        grid_importation = np.zeros(n_bits)
        grid_exportation = np.zeros(n_bits)
        grid_trading = np.zeros(n_bits)

        genset_production = KPI_core["importation"]
        curtailment = KPI_core["exportation"]

    return {"global":{"annual load (kWh)": annual_load,"annual REN production (kWh)":annual_REN_production,"annual importation (kWh)": annualize(KPI_core["importation"], Context), "annual exportation (kWh)": annualize(KPI_core["exportation"], Context)},"timeseries":{"grid importation (kW)": grid_importation,"grid exportation (kW)": grid_exportation,"grid trading (kW)": grid_trading,"genset production (kW)": genset_production,"curtailment (kW)": curtailment,}}

def _compute_storage_kpis(Context,dispatching_timeseries):
    """
    Compute storage KPIs.
    """
    storage_TS = dispatching_timeseries["storage_TS"]
    n_store = Context.storage.n_store
    losses = np.array([storage_TS[i]/Context.storage.characteristics[STOR_ROUND_TRIP_EFF,:][i]-storage_TS[i] for i in range(n_store)])
    losses[losses<0]=0 
    annual_total_losses = annualize(sum(np.sum(losses,axis=1)),Context)
    cumulative_energy = np.array([-np.cumsum(storage_TS[i] / Context.time.time_resolution + losses[i] / Context.time.time_resolution) for i in range(n_store)])
    reported_energy = annualize(np.where(storage_TS > 0, storage_TS, 0),Context)
    stored_energy = annualize(-np.where(storage_TS < 0, storage_TS, 0), Context)
    n_store = Context.storage.n_store
    energy_storages = np.divide(np.array([np.max(cumulative_energy[i]) - np.min(cumulative_energy[i]) for i in range(n_store)]),Context.storage.characteristics[STOR_DEPTH_OF_DISCHARGE,:])
    powers_out = tuple((max(storage_TS[i]) for i in range(n_store)))
    powers_in = tuple((-min(storage_TS[i]) for i in range(n_store)))
    minSOCs=(1-Context.storage.characteristics[STOR_DEPTH_OF_DISCHARGE,:])/2 
    power_storages=np.array([max(powers_in[i],powers_out[i]) for i in range(n_store)])
    Equivalent_cycles =  np.array([np.sum(np.abs(storage_TS[i,:]))/(2*Context.time.time_resolution*max(energy_storages[i],np.float64(1e-15))*Context.time.duration_years) for i in range(n_store)]) 
    storage_lifetime = np.array([min(Context.storage.characteristics[STOR_LIFETIME,i],Context.storage.characteristics[STOR_CYCLE_LIFE,i]/max(1e-15,Equivalent_cycles[i])) for i in range(n_store)])

    return{"losses (kW)": losses,"cumulative_energy": cumulative_energy,"technology_indicators":{"Energy capacity (kWh)":energy_storages,"Power in (kW)":powers_in,"Power out (kW)":powers_out, "Min. SOC (%)" : minSOCs, "Equivalent cycles (/yrs.)" : Equivalent_cycles, "Storage lifetime (yrs.)" : storage_lifetime},"technology actions":{"annual reported energy (kWh)": reported_energy,"annual stored energy (kWh)": stored_energy,"annual losses (kWh)": annualize(losses, Context),},"annual total losses (kWh)":annual_total_losses, "Power storages (kW)":power_storages}

def _compute_technical_kpis(Context,KPI_net_load,KPI_flows,KPI_core):
    """
    Compute technical energy indicators.
    """

    total_load = np.sum(KPI_net_load["optimized_load"])
    total_prod = np.sum(KPI_net_load["production"])
    self_sufficiency = (1 - np.sum(KPI_flows["timeseries"]["grid importation (kW)"]) / total_load if total_load > 0 else np.nan)
    self_consumption = ( 1 - np.sum(KPI_core["exportation"]) / total_prod if total_prod > 0 else np.nan)
    REN_fraction = total_prod/total_load if total_load > 0 else np.nan
    autonomy = ( 1- np.sum(KPI_flows["timeseries"]["grid importation (kW)"] > 0)/ Context.time.n_bits)
    REN_autonomy = 1-np.sum(KPI_core["importation"]>0)/Context.time.n_bits
    REN_self_sufficiency = (1-np.sum(KPI_core["importation"])/total_load) if total_load>0 else np.nan
    Capacity_factor = sum(KPI_net_load["production"])/(max(KPI_net_load["production"])*Context.time.n_bits) if max(KPI_net_load["production"]) > 0 else np.nan

    obtained_constraint_level = self_sufficiency if Context.optimization.constraint_num==CONS_Self_sufficiency else self_consumption if (Context.optimization.constraint_num==CONS_Self_consumption) else REN_fraction if(Context.optimization.constraint_num==CONS_REN_fraction) else np.nan

    return {"Self-sufficiency": self_sufficiency,"Self-consumption": self_consumption,"Autonomy": autonomy,"REN self-sufficiency":REN_self_sufficiency,"REN fraction": REN_fraction,"REN autonomy":REN_autonomy,"Capacity_factor":Capacity_factor,"Constraint level" : obtained_constraint_level}

def _compute_grid_kpis(Context, dispatching_timeseries, KPI_flows):
    """
    Compute grid energy indicators.
    """
    Max_power_from_grid = max(KPI_flows["timeseries"]["grid importation (kW)"]) 
    Max_power_to_grid = max(KPI_flows["timeseries"]["grid exportation (kW)"]) 
    Max_curtailment = max(KPI_flows["timeseries"]["curtailment (kW)"])     

    if (Context.optimization.connexion=='Off-grid'):
        Contract_power=0
    elif (Context.optimization.connexion=='On-grid'):          
        Contract_power=max(0,max(dispatching_timeseries["trades"]))
    
    return{"Max. power from grid (kW)":Max_power_from_grid, "Max. power to grid (kW)":Max_power_to_grid,"Max. curtailment (kW)": Max_curtailment,"Contract power (kW)": Contract_power}


def _compute_genset_kpis(Context, KPI_core, dispatching_timeseries):
    """
    Compute genset indicators.
    """

    if (Context.optimization.connexion=='Off-grid'):
        DG_nominal_power = max(dispatching_timeseries["trades"])
        DG_production=KPI_core["importation"]
        if (DG_nominal_power>0):
            closest_levels = np.array([int(10*(DG_production[i]/DG_nominal_power)-0.5) for i in range(Context.time.n_bits)])
            annual_fuel_consumption_DG = annualize(sum(DG_production*Context.genset.fuel_consumption[closest_levels]),Context)
            DG_lifetime_years = Context.genset.lifetime/annualize(sum(np.where(DG_production>0,1,0)),Context)
        else : 
            closest_levels = np.array([np.nan for i in range(Context.time.n_bits)])
            annual_fuel_consumption_DG = 0
            DG_lifetime_years = np.nan
            
    elif (Context.optimization.connexion=='On-grid'):     
        
        DG_nominal_power = 0.
        DG_production=np.zeros(Context.time.n_bitsn_bits)
        closest_levels = 0
        DG_lifetime_years = np.nan
        annual_fuel_consumption_DG = 0.
  
    Max_power_from_DG = max(DG_production) 
    
    return{"sizing":{"nominal_power (kW)":DG_nominal_power, "Max. power from genset (kW)" : Max_power_from_DG ,"genset lifetime (yrs.)" : DG_lifetime_years},"Annual fossil fuel consumption from genset (kWh)" : annual_fuel_consumption_DG, "genset production (kW)":DG_production,"closest_levels": closest_levels}

def _compute_environmental_kpis(Context, solution, dispatching_timeseries, KPI_flows, KPI_genset, KPI_storage, TONS_CONVERSION_FACTOR , KILOS_CONVERSION_FACTOR):
    """
    Compute environmental indicators.
    """

    #CO2 emissions
    if (Context.optimization.connexion=='Off-grid'):
        annual_CO2eq_DG = KPI_genset["Annual fossil fuel consumption from genset (kWh)"]*Context.genset.fuel_CO2eq_emissions/TONS_CONVERSION_FACTOR  
    elif (Context.optimization.connexion=='On-grid'):          
        annual_CO2eq_DG = 0.

    annual_CO2eq_prod = sum(sum(np.multiply(np.array([solution.production_set[i]*Context.production.unit_prods[i,:] for i in range(len(solution.production_set))]).T/KILOS_CONVERSION_FACTOR,np.array(Context.production.specs_num[:,PROD_EMISSIONS]))))/TONS_CONVERSION_FACTOR/Context.time.time_resolution/Context.time.duration_years
    annual_CO2eq_storage = annualize(np.inner(np.array([sum(np.where(dispatching_timeseries["storage_TS"][i]>0,dispatching_timeseries["storage_TS"][i],0)) for i in range(Context.storage.n_store)]),Context.storage.characteristics[STOR_EMISSIONS,:])/TONS_CONVERSION_FACTOR,Context)
    annual_CO2eq_importation = sum(KPI_flows["timeseries"]["grid importation (kW)"])*Context.grid.C02eqemissions/TONS_CONVERSION_FACTOR/Context.time.time_resolution/Context.time.duration_years if (Context.optimization.connexion=='On-grid') else 0
       
    #fossil fuel consumption                                
    annual_fossil_fuel_consumption_importation =  annualize(Context.grid.fossil_fuel_ratio*sum(KPI_flows["timeseries"]["grid importation (kW)"]),Context) if (Context.optimization.connexion=='On-grid') else 0
    
    annual_fossil_fuel_consumption = annual_fossil_fuel_consumption_importation+KPI_genset["Annual fossil fuel consumption from genset (kWh)"]
    annual_CO2eq_total = annual_CO2eq_DG+annual_CO2eq_prod+annual_CO2eq_importation+annual_CO2eq_storage

    #EROI
    if (Context.optimization.connexion=='Off-grid' and KPI_genset["sizing"]["nominal_power (kW)"]>0):
        productible_energy_DG = annualize(sum(KPI_genset["genset production (kW)"]),Context)*KPI_genset["sizing"]["genset lifetime (yrs.)"]
        consumed_energy_DG = productible_energy_DG/Context.genset.EROI
    else : 
        productible_energy_DG = 0.
        consumed_energy_DG = 0.  
    
    productible_energy = sum(np.sum(np.multiply(solution.production_set,Context.production.unit_prods.T),axis=0)/Context.time.time_resolution/Context.time.duration_years*Context.production.specs_num[:,PROD_LIFETIME])
    consumed_energy_production = sum(productible_energy/Context.production.specs_num[:,PROD_EROI])
    consumed_energy_storage = sum(np.nanmin(np.array([Context.storage.characteristics[STOR_CYCLE_LIFE,:],Context.storage.characteristics[STOR_LIFETIME,:]*KPI_storage["technology_indicators"]["Equivalent cycles (/yrs.)"]]),axis=0)*KPI_storage["technology_indicators"]["Energy capacity (kWh)"]/Context.storage.characteristics[STOR_ESOEI,:])

    EROI = (productible_energy+productible_energy_DG)/(consumed_energy_DG+consumed_energy_production+consumed_energy_storage)

    return{"Annual fossil fuel consumption (kWh)" : annual_fossil_fuel_consumption , "Annual CO2eq total (tCO2)" : annual_CO2eq_total, "EROI" : EROI}


def _compute_storage_SOC_kpis(Context,KPI_storage):
    """
    Compute storage state-of-charge indicators.
    """
    n_store = Context.storage.n_store
    SOCs=(np.divide((KPI_storage["cumulative_energy"].T-np.min(KPI_storage["cumulative_energy"],axis=1)),KPI_storage["technology_indicators"]["Energy capacity (kWh)"]) + KPI_storage["technology_indicators"]["Min. SOC (%)"]).T
            
    storage_NULL=tuple(KPI_storage["technology_indicators"]["Energy capacity (kWh)"][i]==0 for i in range(n_store))
    State_SOCs=[np.searchsorted([(i+1)/100 for i in range(99)], SOCs[j,:]) for j in range(n_store)]
    distribution_Depth_of_discharge=np.array([[np.count_nonzero(State_SOCs[i]==j)/len(State_SOCs[i]) for j in range(100)] for i in range(n_store)])
    if sum(storage_NULL)>0:
        distribution_Depth_of_discharge[np.where(storage_NULL)[0][0]]=np.repeat(np.nan,100)
    
    dist_DOD = pd.DataFrame(data={'SOC_Percentile':[i/100 for i in range(101)]})
    for i in range(n_store):
        dist_DOD = dist_DOD.join(pd.DataFrame(data={'SOC_distribution_' + Context.storage.technologies[i]:distribution_Depth_of_discharge[i]}))

    return{"SOC (%)" : SOCs, "dist_DOD" : dist_DOD}


def _compute_installation_lifetime(Context,solution,KPI_genset,KPI_storage):
    return np.nanmin(np.array([KPI_genset["sizing"]["genset lifetime (yrs.)"],min(KPI_storage["technology_indicators"]["Storage lifetime (yrs.)"]),np.nanmin(np.where(solution.production_set>0,Context.production.specs_num[:,PROD_LIFETIME],np.nan))]))    

def _compute_economic_kpis (Context, solution, KPI_storage, KPI_genset, KPI_flows, KPI_grid, installation_lifetime):
    """
    Compute economic indicators.
    """
    annual_cost_production = np.sum((Context.production.specs_num[:,PROD_CAPEX]/Context.production.specs_num[:,PROD_LIFETIME]+Context.production.specs_num[:,PROD_OPEX])*(solution.production_set))
    CAPEX_storage_cost =  np.multiply(KPI_storage["Power storages (kW)"],Context.storage.characteristics[STOR_POWER_COST,:]) + np.multiply(KPI_storage["technology_indicators"]["Energy capacity (kWh)"],Context.storage.characteristics[STOR_ENERGY_COST,:]) + np.multiply(Context.storage.characteristics[STOR_INSTALLATION_COST,:],(KPI_storage["Power storages (kW)"]>np.zeros(Context.storage.n_store)))
    annual_cost_storage_CAPEX = sum(np.divide(CAPEX_storage_cost,KPI_storage['technology_indicators']["Storage lifetime (yrs.)"]))
    annual_cost_storage_OPEX = sum(np.multiply(Context.storage.characteristics[STOR_OM_COST,:],KPI_storage["Power storages (kW)"]))

    if (Context.optimization.connexion=='Off-grid'):
        if KPI_genset['sizing']["nominal_power (kW)"]>0 :
            annual_gain_exportation = 0.
            annual_cost_importation = 0.
            annual_cost_overrun = 0.
            annual_cost_contract_power = 0.
            annual_cost_genset_CAPEX = KPI_genset["sizing"]["nominal_power (kW)"]*Context.genset.unit_cost/KPI_genset["sizing"]["genset lifetime (yrs.)"]
            annual_cost_genset_OPEX = annualize(sum(KPI_genset["genset production (kW)"])*Context.genset.maintenance_cost,Context)
            annual_total_fuel_cost = KPI_genset["Annual fossil fuel consumption from genset (kWh)"]*Context.genset.fuel_cost
        else :
            annual_gain_exportation = 0.
            annual_cost_importation = 0.
            annual_cost_overrun = 0.
            annual_cost_contract_power = 0.
            annual_cost_genset_CAPEX = 0.
            annual_cost_genset_OPEX = 0.
            annual_total_fuel_cost = 0.
 
    elif (Context.optimization.connexion=='On-grid'):   
        annual_gain_exportation = annualize(np.multiply(KPI_flows["timeseries"]["grid exportation (kW)"],Context.grid.selling_price[solution.contract,:]).sum(),Context)
        annual_cost_importation = annualize(np.multiply(KPI_flows["timeseries"]["grid importation (kW)"],Context.grid.prices[solution.contract,:]).sum(),Context)
        annual_cost_overrun = max(0,(max(KPI_flows["timeseries"]["grid exportation (kW)"])-KPI_grid["contract_power"])*Context.grid.overrun[solution.contract])
        annual_cost_contract_power = Context.grid.fixed_premium[solution.contract]*KPI_grid["contract_power"]
        annual_cost_genset_CAPEX = 0
        annual_cost_genset_OPEX = 0
        annual_total_fuel_cost = 0.

    LCOE = (annual_cost_production+annual_cost_storage_CAPEX+annual_cost_storage_OPEX+annual_cost_genset_OPEX+annual_cost_genset_CAPEX+annual_total_fuel_cost+annual_cost_importation+annual_cost_contract_power+annual_cost_overrun-annual_gain_exportation)/(KPI_flows["global"]["annual load (kWh)"])   if KPI_flows["global"]["annual load (kWh)"]>0 else np.nan
    Annual_net_benefits = (-annual_cost_production-annual_cost_storage_CAPEX-annual_cost_storage_OPEX-annual_cost_genset_OPEX-annual_cost_genset_CAPEX-annual_total_fuel_cost-annual_cost_importation-annual_cost_contract_power-annual_cost_overrun+annual_gain_exportation)
    Value = Annual_net_benefits*installation_lifetime
    initial_investment = sum(CAPEX_storage_cost) +np.nanmin((0.,annual_cost_genset_CAPEX*KPI_genset["sizing"]["genset lifetime (yrs.)"]))+ np.sum(Context.production.specs_num[:,PROD_CAPEX]*solution.production_set)
    
    return{"annual_cost_production":annual_cost_production, "annual_cost_importation" : annual_cost_importation, "annual_gain_exportation" : annual_gain_exportation, "annual_cost_contract_power" : annual_cost_contract_power, "annual_cost_overrun" : annual_cost_overrun, "annual_cost_storage_CAPEX" : annual_cost_storage_CAPEX, "annual_cost_storage_OPEX" : annual_cost_storage_OPEX, "annual_cost_genset_CAPEX" : annual_cost_genset_CAPEX, "annual_cost_genset_OPEX": annual_cost_genset_OPEX,"annual_fuel_cost" : annual_total_fuel_cost, "global kpis":{"LCOE (€/kWh)" : LCOE, "Annual net benefits (€/yrs.)":Annual_net_benefits , "Value (€)" : Value, "Initial investment (€)" : initial_investment }}

def _compute_economic_decomposition (Context, KPI_economics, KPI_flows):
    """
    Compute economic decomposition (contributions in €/kWh).
    """
    
    annual_load = KPI_flows["global"]["annual load (kWh)"]

    cost_production_per_kWh  = KPI_economics["annual_cost_production"]/annual_load
    OPEX_storage_per_kWh = KPI_economics["annual_cost_storage_OPEX"]/annual_load
    CAPEX_storage_per_kWh = KPI_economics["annual_cost_storage_CAPEX"]/annual_load
    contract_power_per_kWh = KPI_economics["annual_cost_contract_power"]/annual_load
    overrun_penalty_per_kWh = KPI_economics["annual_cost_overrun"]/annual_load
    energy_importation_per_kWh = KPI_economics["annual_cost_importation"]/annual_load
    DG_fuel_cost_per_kWh = KPI_economics["annual_fuel_cost"]/annual_load
    DG_CAPEX_cost_per_kWh = KPI_economics["annual_cost_genset_CAPEX"]/annual_load
    DG_OPEX_cost_per_kWh = KPI_economics["annual_cost_genset_OPEX"]/annual_load
    energy_exportation_per_kWh = KPI_economics["annual_gain_exportation"]/annual_load
    return{"production (€/kWh)":cost_production_per_kWh,"OPEX storage (€/kWh)": OPEX_storage_per_kWh,
        "CAPEX storage (€/kWh)": CAPEX_storage_per_kWh,"contract power (€/kWh)":contract_power_per_kWh,"overrun penalty (€/kWh)": overrun_penalty_per_kWh,
        "energy importation (€/kWh)":energy_importation_per_kWh,"genset fuel cost (€/kWh)":DG_fuel_cost_per_kWh,
        "genset CAPEX cost (€/kWh)" :DG_CAPEX_cost_per_kWh,"genset OPEX cost (€/kWh)":DG_OPEX_cost_per_kWh,"energy exportation (€/kWh)": energy_exportation_per_kWh}


def _compute_DSM_kpis(Context, dispatching_timeseries, KPI_net_load, datetime, KPI_core, KPI_storage, HOURS_PER_DAY):
    """
    Compute demand-side management indicators.
    """
    indexes_hour = [[int((i+j*Context.time.time_resolution*HOURS_PER_DAY)) for j in range(int(Context.time.n_days))] for i in range(int(Context.time.time_resolution*HOURS_PER_DAY))]
    indexes_days = [[int(i+j*Context.time.time_resolution*HOURS_PER_DAY) for i in range(int(Context.time.time_resolution*HOURS_PER_DAY))] for j in range(int(Context.time.n_days))]

    Yearly_base_load = [np.mean(Context.loads.non_movable[indexes_days[j]]+Context.loads.D_movable[indexes_days[j]]) for j in range(int(Context.time.n_days))]
    Yearly_movable_loads = [np.mean(Context.loads.Y_movable[indexes_days[j]]) for j in range(int(Context.time.n_days))]
    Yearly_final_loads = [np.mean(KPI_net_load["optimized_load"][indexes_days[j]]) for j in range(int(Context.time.n_days))]
    Yearly_prod = [np.mean(KPI_net_load["production"][indexes_days[j]]) for j in range(int(Context.time.n_days))]
    DSM_yearly_strategy = pd.DataFrame(data={'Datetime': datetime[indexes_hour[0]].strftime('%Y-%m-%d'),'Yearly base load (kW)': Yearly_base_load,'Yearly movable load (kW)': Yearly_movable_loads,'Yearly final load (kW)': Yearly_final_loads,'Yearly prod (kW)': Yearly_prod})

    Daily_base_load = [np.mean(Context.loads.non_movable[indexes_hour[j]]) for j in range(int(Context.time.time_resolution*HOURS_PER_DAY))]
    Daily_movable_loads = [np.mean(Context.loads.D_movable[indexes_hour[j]]+Context.loads.Y_movable[indexes_hour[j]]) for j in range(int(Context.time.time_resolution*HOURS_PER_DAY))]
    Daily_final_loads = [np.mean(KPI_net_load["optimized_load"][indexes_hour[j]]) for j in range(int(Context.time.time_resolution*HOURS_PER_DAY))]
    Daily_prod = [np.mean(KPI_net_load["production"][indexes_hour[j]]) for j in range(int(Context.time.time_resolution*HOURS_PER_DAY))]
    
    DSM_daily_strategy = pd.DataFrame(data={'Datetime': datetime[0:int(Context.time.time_resolution * 24)].strftime('%H:%M'),'Daily base load (kW)': Daily_base_load,'Daily movable load (kW)': Daily_movable_loads,'Daily final load (kW)': Daily_final_loads,'Daily prod (kW)': Daily_prod})
    DSM_yearly_strategy = pd.DataFrame(data={'Datetime': datetime[indexes_hour[0]].strftime('%Y-%m-%d'),'Yearly base load (kW)': Yearly_base_load,'Yearly movable load (kW)': Yearly_movable_loads,'Yearly final load (kW)': Yearly_final_loads,'Yearly prod (kW)': Yearly_prod})
    Load_strategy = pd.DataFrame(data={'Datetime': datetime,'Non controllable load (kW)':Context.loads.non_movable,'Daily movable load (kW)': Context.loads.D_movable,'Yearly movable load (kW)': Context.loads.Y_movable,'Daily optimized load (kW)': dispatching_timeseries["D_DSM"],'Yearly optimized load (kW)': dispatching_timeseries["Y_DSM"]})

    daily_time_balancing = pd.DataFrame({
        "Datetime": datetime[:int(Context.time.time_resolution * HOURS_PER_DAY)].strftime("%H:%M"),
        "Daily load (kW)": Daily_final_loads,
        "Daily prod (kW)": Daily_prod,
        "Daily importation (kW)": [np.mean(KPI_core["importation"][idx]) for idx in indexes_hour],
        "Daily exportation (kW)": [np.mean(KPI_core["exportation"][idx]) for idx in indexes_hour],
        "Daily storage (kW)": [np.mean(KPI_core["total_storage_power"][idx]) for idx in indexes_hour],})    
 
    yearly_time_balancing = pd.DataFrame({
        "Datetime": datetime[indexes_hour[0]].strftime('%Y-%m-%d'),
        "Yearly load (kW)": Yearly_final_loads,
        "Yearly prod (kW)": Yearly_prod,
        "Yearly importation (kW)": [np.mean(KPI_core["importation"][indexes_days[j]]) for j in range(Context.time.n_days)],
        "Yearly exportation (kW)": [-np.mean(KPI_core["exportation"][indexes_days[j]]) for j in range(Context.time.n_days)],
        "Yearly storage (kW)": [np.mean(KPI_core["total_storage_power"][indexes_days[j]]) for j in range(Context.time.n_days)],})
    
    return{"DSM strategies" : {"DSM_daily_strategy": DSM_daily_strategy, "DSM_yearly_strategy": DSM_yearly_strategy, "Load_strategy":Load_strategy},"time balancing":{"daily_time_balancing" : daily_time_balancing , "yearly_time_balancing" : yearly_time_balancing}}

def storage_charge_energy(mask,n_store,dispatching_timeseries,Context):
    """
    Annual charged energy into storage (kWh/year).
    """
    total = 0.0

    for i in range(n_store):
        total += np.sum(np.where(dispatching_timeseries["storage_TS"][i][mask] < 0,-dispatching_timeseries["storage_TS"][i][mask], 0))
    return annualize(total,Context)

def storage_discharge_energy(mask,n_store,dispatching_timeseries,Context):
    """
    Annual discharged energy from storage (kWh/year).
    """
    total = 0.0

    for i in range(n_store):
        total += np.sum(np.where(dispatching_timeseries["storage_TS"][i][mask] > 0,dispatching_timeseries["storage_TS"][i][mask], 0))
    return annualize(total,Context)

def _compute_energy_allocation_kpis(Context,dispatching_timeseries,KPI_net_load,KPI_core,KPI_flows):
    """
    Compute global energy allocation indicators.
    
    Storage sign convention:
    storage_TS > 0 : discharge to system
    storage_TS < 0 : charging
    """
    trades = dispatching_timeseries["trades"]
    net_load = KPI_net_load["net_load"]
    n_store = Context.storage.n_store
    storage_technologies = Context.storage.technologies
    storage_TS = dispatching_timeseries["storage_TS"]
    optimized_load = KPI_net_load["optimized_load"]
    
    # ==========================================================
    # OPERATING REGIMES
    # ==========================================================
    
    surplus_export_mask = np.where((trades<=0) & (net_load<=0))[0]
    deficit_import_mask = np.where((trades>=0) & (net_load>=0))[0]
    surplus_import_mask = np.where((trades>0) & (net_load<=0))[0]
    deficit_export_mask = np.where((trades<0) & (net_load>=0))[0]
    
    # ==========================================================
    # ENERGY ALLOCATIONS
    # ==========================================================   
    
    surplus_export_allocation = {"load": annualize(np.sum(optimized_load[surplus_export_mask]),Context),"storage_charge": storage_charge_energy(surplus_export_mask,n_store,dispatching_timeseries,Context ),"grid_export": annualize(-np.sum(trades[surplus_export_mask]),Context), "storage_discharge": storage_discharge_energy(surplus_export_mask,n_store,dispatching_timeseries,Context),}
    deficit_import_allocation = {"production": annualize(np.sum(KPI_net_load["production"][deficit_import_mask]),Context ),"storage_discharge": storage_discharge_energy(deficit_import_mask,n_store,dispatching_timeseries,Context),"grid_import": annualize(np.sum(trades[deficit_import_mask]),Context), "storage_charge": storage_charge_energy(deficit_import_mask,n_store,dispatching_timeseries,Context),} 
    deficit_export_allocation = {"production": annualize(np.sum(KPI_net_load["production"][deficit_export_mask]),Context),"storage_discharge": storage_discharge_energy(deficit_export_mask,n_store,dispatching_timeseries,Context),"grid_export": annualize(-np.sum(trades[deficit_export_mask]),Context),}    
    surplus_import_allocation = {"load": annualize(np.sum(optimized_load[surplus_import_mask]),Context),"excess_production": annualize(np.sum(KPI_net_load["production"][surplus_import_mask]),Context)- annualize(np.sum(optimized_load[surplus_import_mask]),Context),"importation":annualize(np.sum(trades[surplus_import_mask]),Context),}

    
    surplus_energy_actions = pd.DataFrame(data=([annualize(np.sum(optimized_load[surplus_export_mask])+ np.sum(optimized_load[surplus_import_mask]),Context),annualize(-np.sum(trades[surplus_export_mask]),Context),annualize(np.sum(trades[surplus_import_mask]),Context),]+ [storage_charge_energy((net_load < 0) & (storage_TS[i] < 0),n_store,dispatching_timeseries,Context)for i in range(n_store)]+ [annualize(np.sum(np.where(storage_TS[i][(net_load < 0) & (storage_TS[i] > 0)] > 0,storage_TS[i][(net_load < 0) & (storage_TS[i] > 0)],0,)),Context)for i in range(n_store)]),index=(["Load (kWh/year)","Grid export (kWh/year)","Grid import (kWh/year)"]+ [f"Storage {storage_technologies[i]} charge (kWh/year)" for i in range(n_store)]+ [f"Storage {storage_technologies[i]} discharge (kWh/year)"for i in range(n_store)]),).transpose()    
    deficit_energy_actions = pd.DataFrame(data=([annualize(np.sum(KPI_net_load["production"][deficit_import_mask]) + np.sum(KPI_net_load["production"][deficit_export_mask]),Context),annualize(-np.sum(trades[deficit_export_mask]),Context),annualize(np.sum(trades[deficit_import_mask]),Context),]+ [storage_charge_energy((net_load > 0) & (storage_TS[i] < 0),n_store,dispatching_timeseries,Context)for i in range(n_store)]+ [annualize(np.sum(np.where(storage_TS[i][(net_load > 0) & (storage_TS[i] > 0)] > 0,storage_TS[i][(net_load > 0) & (storage_TS[i] > 0)],0,) ),Context )for i in range(n_store)]),index=["Production (kWh/year)","Grid export (kWh/year)","Grid import (kWh/year)"]+ [f"Storage {storage_technologies[i]} charge (kWh/year)"for i in range(n_store)]+ [f"Storage {storage_technologies[i]} discharge (kWh/year)"for i in range(n_store)]).transpose()
     
    production_allocation = pd.DataFrame(data=(surplus_export_allocation["storage_charge"]+surplus_import_allocation["excess_production"],deficit_import_allocation["production"]+surplus_export_allocation["load"]+deficit_export_allocation["production"]+surplus_import_allocation["load"],deficit_export_allocation["grid_export"]+surplus_export_allocation["grid_export"],KPI_flows["global"]["annual REN production (kWh)"]),index=['Storage (kWh)' ,'Simultaneous load (kWh)','annual grid export','Annual REN production (kWh)']).transpose()
    load_coverage = pd.DataFrame(data=(deficit_import_allocation["storage_discharge"]+deficit_export_allocation["storage_discharge"]-deficit_export_allocation["grid_export"] ,deficit_import_allocation["production"]+surplus_export_allocation["load"]+deficit_export_allocation["production"]+surplus_import_allocation["load"],deficit_import_allocation["grid_import"],KPI_flows["global"]["annual load (kWh)"]),index=['Storage (kWh)', 'Simultaneous production (kWh)','annual grid import','Annual load (kWh)']).transpose()
    grid_export_sources = pd.DataFrame(data=(production_allocation["annual grid export"], KPI_flows["global"]["annual exportation (kWh)"]-production_allocation["annual grid export"]),index=["Production export (kWh/year)","Storage discharge (kWh/year)"]).transpose()
    grid_import_usage = pd.DataFrame(data=(load_coverage["annual grid import"], KPI_flows["global"]["annual importation (kWh)"]-load_coverage["annual grid import"]),index=['load supply (kWh)','storage charge(kWh)']).transpose()
 
    return{"allocations":{"surplus_export_allocation":surplus_export_allocation, "deficit_import_allocation":deficit_import_allocation, "deficit_export_allocation":deficit_export_allocation , "surplus_import_allocation":surplus_import_allocation} ,"actions":{ "surplus_energy_actions":surplus_energy_actions , "deficit_energy_actions":deficit_energy_actions}, "flow_attribution": {"production_allocation":production_allocation , "load_coverage":load_coverage , "grid_export_sources":grid_export_sources , "grid_import_usage":grid_import_usage}}


def _build_output_dictionary(Context,solution,all_kpis,dispatching_timeseries):
    return{'timeseries':{"losses (kW)":all_kpis["storage"]["losses (kW)"],"SOC (%)":all_kpis["soc"]["SOC (%)"],'D_DSM (kW)':dispatching_timeseries["D_DSM"],'Y_DSM (kW)':dispatching_timeseries["Y_DSM"],'trades (kW)':dispatching_timeseries["trades"],'optimized load (kW)':all_kpis["net_load"]["optimized_load"],'non movable load (kW)':Context.loads.non_movable,'production (kW)':all_kpis["net_load"]["production"],'storage_TS (kW)':dispatching_timeseries["storage_TS"],**all_kpis["flows"]["timeseries"]},           
           'technical':{'fitness':solution.fitness,**all_kpis["technical"],'Installation lifetime (yrs.)':all_kpis["installation_lifetime"]},           
           'economics':{**all_kpis["economics"]["global kpis"],**all_kpis["cost_decomposition"]}, 
           'storages':all_kpis["storage"]["technology_indicators"], 
           'genset':all_kpis["genset"]["sizing"],           
           'genset power distribution':all_kpis["genset"]["closest_levels"],
           'environment':all_kpis["environment"],
           'flows':{**all_kpis["flows"]["global"],'Annual sum losses (kWh)':all_kpis["storage"]["annual total losses (kWh)"],'grid_flows':all_kpis["grid"]},          
           'storage flows':all_kpis["storage"]["technology actions"],
           'distribution_DOD':all_kpis["soc"]["dist_DOD"],
           'energy_allocations':all_kpis["energy_allocation"],
           'balancing':all_kpis["dsm"]["time balancing"],
           'demand-side management':all_kpis["dsm"]["DSM strategies"],
           'EMS':all_kpis["EMS"]
           }

def compute_KPI(solution,Context,datetime):
    """
    Comprehensive cost and performance evaluation of the final (best) microgrid scenario.
    
    This function computes the annualized technical, economic, environmental,
    and operational metrics of a microgrid system.
    
    Parameters:
        gene (object): Object containing production and storage decisions of the system.
        datetime (array-like): Array of datetime values for the simulation period.
        storage_characteristics (array-like): Storage capacities, efficiencies, lifetime, etc.
        time_resolution (float): Number of timesteps per hour (temporal resolution).
        n_store (int): Number of storage units in the system.
        duration_years (float): Total simulation duration in years.
        specs_num (array-like): Technical specifications of production units.
        prices_num (array-like): Energy purchase prices from the main grid.
        fixed_premium (array-like): Fixed premium costs for contracted grid power.
        Overrun (array-like): Penalty costs for exceeding contracted power.
        Selling_price (array-like): Selling prices for exported electricity.
        Non_movable_load (array-like): Baseline non-controllable load (kW) time series.
        total_D_Movable_load (array-like): Total daily controllable load (kW).
        D_Movable_load (array-like): Daily controllable load distribution (kW).
        total_Y_Movable_load (array-like): Total yearly controllable load (kW).
        Y_Movable_load (array-like): Yearly controllable load distribution (kW).
        Grid_fossil_fuel_ratio (float): Fossil fuel fraction of imported electricity.
        Main_grid_PoF_ratio (float): Probability of failure of the main grid (unused here).
        Main_grid_emissions (float): CO2eq emissions of imported electricity (g/kWh).
        prod_C (array-like): Constant production term (kW).
        prods_U (array-like): Unit production capacities.
        Bounds_prod (array-like): Bounds on production capacities.
        constraint_num (int): Number indicating which constraint to apply.
        constraint_level (float): Level of applied constraint.
        cost_constraint (float): Maximum cost allowed for certain constraints.
        n_bits (int): Number of time steps for discretized decisions.
        Connexion (str): 'On-grid' or 'Off-grid' system connection mode.
        DG_fuel_consumption (array-like): Diesel generator fuel consumption table.
        DG_fuel_cost (float): Cost per unit of DG fuel.
        DG_unit_cost (float): Capital cost of the DG (€/kW).
        DG_lifetime (float): Lifetime of DG in hours.
        DG_maintenance_cost (float): OPEX of DG (€/kWh).
        DG_EROI (float): Energy return on investment of the DG.
        fuel_CO2eq_emissions (float): CO2eq emissions of DG fuel (g/kWh).
        storage_techs (list of str): Names of storage technologies.
        n_days (int): Number of days in the simulation period.
    
    Returns:
        dict: Dictionary containing multiple KPI categories:
            - 'TimeSeries': time series of production, storage, load, and trades
            - 'Technical': KPIs such as self-sufficiency, autonomy, capacity factor
            - 'economics': LCOE, costs, net benefits, value, initial investment
            - 'Storages': storage capacities, SOC, equivalent cycles, lifetime
            - 'DG': DG power, lifetime, distribution levels
            - 'Environment': annual CO2 emissions, fossil fuel use, EROI
            - 'Flows': annual load, REN production, import/export sums
            - 'Flows storages': annual stored/reported energy and losses
            - 'Extra_outputs': detailed load and production allocations, logic masks
            - 'Balancing': daily and yearly time balancing
            - 'Demand-side management': daily and yearly DSM strategies
    """
    # =========================
    # CONSTANTS
    # =========================

    TONS_CONVERSION_FACTOR = 1000000
    KILOS_CONVERSION_FACTOR = 1000
    HOURS_PER_DAY = 24
    
    # =========================
    # LOAD & PRODUCTION
    # =========================
    
    production = ((Context.production.unit_prods.T*solution.production_set).sum(axis=1)+Context.production.current_prod)/KILOS_CONVERSION_FACTOR        
        
    if Context.optimization.type_optim == 'research': 
        
        KPI_EMS = None
        dispatching_timeseries = {"D_DSM" : solution.D_DSM.flatten() , "Y_DSM":solution.Y_DSM, "storage_TS":solution.storage_TS, "trades":solution.trades,"losses" : None}
    
    elif Context.optimization.type_optim == 'pro':
        
        KPI_EMS = _compute_EMS_kpis(solution)
        pro_parameters, global_parameters, grid_parameters, RENSystems_parameters, Genset_parameters, extra_parameters = Ef.build_numba_params(Context)
        (storage_TS,trades,D_DSM,Y_DSM,SOCs_eff,losses,P_diff) = Eems.LFE_CCE(solution, global_parameters, pro_parameters, production ,RENSystems_parameters)
        
        dispatching_timeseries = {"D_DSM" : D_DSM , "Y_DSM":Y_DSM, "storage_TS":storage_TS, "trades":trades,"losses" : losses}

    KPI_net_load = _compute_net_load_kpis(Context,solution,production,dispatching_timeseries)


    # =========================
    # CORE ENERGY BALANCE
    # =========================
        
    KPI_core = _compute_balancing_core(Context,dispatching_timeseries,KPI_net_load)
    
    # =========================
    # GRID FLOWS
    # =========================

    KPI_flows = _compute_energy_flows(Context,KPI_net_load,KPI_core)

    # =========================
    # STORAGE
    # =========================
    
    KPI_storage = _compute_storage_kpis(Context,dispatching_timeseries)
    
    if Context.optimization.type_optim == 'research': 
        dispatching_timeseries["losses"] = KPI_storage["losses (kW)"]
    
    # =========================
    # SELF-SUFFICIENCY / AUTONOMY
    # =========================
    
    KPI_technical = _compute_technical_kpis(Context,KPI_net_load,KPI_flows,KPI_core)
    
    # =========================
    # GRID PARAMETERS
    # =========================
    
    KPI_grid = _compute_grid_kpis(Context, dispatching_timeseries, KPI_flows)

    # =========================
    # GENSET
    # =========================

    KPI_genset = _compute_genset_kpis(Context, KPI_core,dispatching_timeseries)
    
    # =========================
    # LIFETIME
    # =========================   
    
    installation_lifetime = _compute_installation_lifetime(Context,solution,KPI_genset,KPI_storage)
                   
    # =========================
    # ENVIRONMENT
    # =========================   

    KPI_environment = _compute_environmental_kpis(Context, solution, dispatching_timeseries, KPI_flows, KPI_genset, KPI_storage, TONS_CONVERSION_FACTOR , KILOS_CONVERSION_FACTOR)

    # =========================
    # STORAGE SOC ANALYSIS
    # =========================
    
    KPI_SOC = _compute_storage_SOC_kpis(Context,  KPI_storage)

    # =========================
    # ECONOMICS
    # =========================
    
    KPI_economics = _compute_economic_kpis (Context, solution, KPI_storage, KPI_genset, KPI_flows, KPI_grid, installation_lifetime)
    
    # =========================
    # COST DECOMPOSITION
    # =========================

    cost_decomposition = _compute_economic_decomposition (Context, KPI_economics, KPI_flows)

    # =========================
    # DEMAND-SIDE MANAGEMENT
    # =========================

    KPI_DSM = _compute_DSM_kpis(Context, dispatching_timeseries, KPI_net_load, datetime , KPI_core, KPI_storage, HOURS_PER_DAY)

    # =========================
    # ENERGY ALLOCATION
    # =========================    

    KPI_energy_allocation = _compute_energy_allocation_kpis(Context,dispatching_timeseries,KPI_net_load,KPI_core,KPI_flows)

    # =========================
    # OUTPUT
    # =========================        
    
    all_kpis = {"net_load" : KPI_net_load, "EMS" : KPI_EMS, "core": KPI_core,"storage": KPI_storage,"genset": KPI_genset,"environment": KPI_environment,
    "grid": KPI_grid,"soc": KPI_SOC,"energy_allocation": KPI_energy_allocation,"dsm": KPI_DSM,"flows": KPI_flows,
    "technical": KPI_technical,"economics": KPI_economics,"cost_decomposition": cost_decomposition,"installation_lifetime":installation_lifetime}
    
    return _build_output_dictionary(Context,solution,all_kpis,dispatching_timeseries)


def KPI_pro(solution,Context,datetime):
    """
    Comprehensive cost and performance evaluation of the final (best) scenario.
    
    This function computes the annualized technical, economic, environmental,
    and operational metrics of a microgrid system, including:
    
        - Net load and production balance
        - Multi-storage operation and state-of-charge (SOC) evolution
        - Grid interaction (import/export)
        - Diesel generator (DG) operation, fuel consumption, and costs
        - Demand-Side Management (DSM) strategies for daily and yearly loads
        - Energy flows, losses, and curtailment
        - Economic indicators (LCOE, CAPEX, OPEX, net benefits)
        - Environmental indicators (CO2eq emissions, fossil fuel consumption)
        - Storage performance metrics (capacity, cycles, lifetime, SOC distribution)
        - Microgrid self-sufficiency, autonomy, and renewable fraction
        - Detailed time-series outputs for load, production, storage, and trades
    
    Args:
        gene (structured object compatible with Numba): Microgrid operational strategy, design (production and storage), and PMS parameters.
        datetime (pandas.DatetimeIndex or numpy.ndarray): Time series for the simulation period.
        storage_characteristics (numpy.ndarray): Array of storage parameters including capacities, power ratings, efficiencies, and lifetime metrics.
        time_resolution (float): Time resolution.
        n_store (int): Number of storage units in the microgrid.
        duration_years (float): Duration of the simulation in years.
        Non_movable_load (numpy.ndarray): Time series of non-flexible (base) load.
        D_movable_load (numpy.ndarray): Daily totals of daily-flexible load.
        total_D_Movable_load (numpy.ndarray): Daily totals of daily-flexible load.
        Y_movable_load (numpy.ndarray): Time series of yearly-flexible load.
        total_Y_Movable_load (float): Yearly total of yearly-flexible load.
        prod_C (numpy.ndarray): Current on-site renewable energy production time series (kW).
        prods_U (numpy.ndarray): Unit production values.
        Bounds_prod (np.array): Bounds on production set.
        constraint_num (int): Index of the constraint.
        constraint_level (float): Level of constraint.
        cost_constraint (float): Cost associated with the constraint.
        n_bits (int): Number of timesteps in the simulation.
        Connexion (str): Grid connection mode ('On-grid' or 'Off-grid').
        DG_fuel_consumption (np.array): Diesel generator consumption.
        DG_fuel_cost (float): Diesel generator fuel cost.
        DG_unit_cost (int): Diesel generator unit cost (€/kW).
        DG_lifetime (float): Diesel generator lifetime (hrs).
        DG_maintenance_cost (float): Diesel generator maintenance cost (€/kWh/yrs.).
        DG_EROI (float): Diesel generator energy returns on investment.
        fuel_CO2eq_emissions (float): Diesel generator eq CO2 emissions.
        storage_techs (list of str): List of storage technology names.
        n_days (int): Number of days in the simulation period.
    
    Returns:
        dict: Dictionary containing multiple sub-dictionaries with metrics and outputs:
            - TimeSeries: Time series of storage state, losses, trades, production, grid flows, and DSM-adjusted loads.
            - Technical: Technical KPIs such as self-sufficiency, autonomy, REN fraction, capacity factor, and installation lifetime.
            - economics: Economic indicators including LCOE, CAPEX/OPEX per kWh, net benefits, and initial investment.
            - Storages: Storage-specific metrics including capacity, power, SOCs, equivalent cycles, and lifetimes.
            - DG: Diesel generator KPIs including nominal power, lifetime, and min. runtime/production.
            - DG power distribution: Level distribution of diesel generator power output.
            - PMS: Demand-side management strategy and settings.
            - Environment: Annual fossil fuel consumption, CO2 emissions, produced and consumed energy, and EROI.
            - Flows: Annual energy flows including load, production, importation, and exportation.
            - Flows storages: Annual energy stored, reported energy, and storage losses.
            - Extra_outputs: Miscellaneous outputs including usage, logical/illogical trades, grid usage, and SOC distributions.
            - Balancing: Daily and yearly time balancing of load, production, import/export, and storage.
            - Demand-side management: Daily and yearly DSM strategies and load adjustments.
    
    Note:
        The function distinguishes between logical and illogical energy trades, calculates
        energy curtailment, storage losses, and accounts for efficiency of storage and DG.
        It is suitable for both on-grid and off-grid microgrid simulations.
    
    Important:
        This is the final evaluation function used to compute ERMESS outputs.
    """
    
    
    # =========================
    # CONSTANTS
    # =========================
    TONS_CONVERSION_FACTOR = 1000000
    KILOS_CONVERSION_FACTOR = 1000
    HOURS_PER_DAY = 24
    
    # =========================
    # LOAD & PRODUCTION
    # =========================

    production = ((Context.production.unit_prods.T*solution.production_set).sum(axis=1)+Context.production.current_prod)/KILOS_CONVERSION_FACTOR    
    annual_cost_production = np.sum((Context.production.specs_num[:,PROD_CAPEX]/Context.production.specs_num[:,PROD_LIFETIME]+Context.production.specs_num[:,PROD_OPEX])*(solution.production_set))

    # =========================
    # CORE ENERGY BALANCE
    # =========================
        
    pro_parameters, global_parameters, grid_parameters, RENSystems_parameters, Genset_parameters, extra_parameters = Ef.build_numba_params(Context)
    (storage_TS,trades,D_DSM,Y_DSM,SOCs_eff,losses,P_diff) = Eems.LFE_CCE(solution, global_parameters, pro_parameters, production ,RENSystems_parameters)
    
    dispatching_timeseries = {"D_DSM" : D_DSM , "Y_DSM":Y_DSM, "storage_TS":storage_TS, "trades":trades,"losses" : losses}


    
    
    
    Optimized_Load = Context.loads.non_movable+Y_DSM+D_DSM
    Annual_load = sum(Optimized_Load)/Context.time.time_resolution/Context.time.duration_years
    Annual_REN_production = sum(production)/Context.time.time_resolution/Context.time.duration_years
    importation=np.where(trades>0,trades,0)
    exportation = np.where(trades<0,-trades,0)
    signal =  Optimized_Load - production
    Annual_sum_importation=sum(importation)/Context.time.time_resolution/Context.time.duration_years
    Annual_sum_exportation=sum(exportation)/Context.time.time_resolution/Context.time.duration_years

    # =========================
    # STORAGE INITIALIZATION
    # =========================    
    
    n_store = Context.storage.n_store
    n_bits = Context.time.n_bits
    
    sum_diff_storages = np.array([-np.cumsum(storage_TS[i,:]/Context.time.time_resolution+losses[i,:]/Context.time.time_resolution) for i in range(n_store)])             
    reported_energy = np.sum(np.where(storage_TS>0,storage_TS,0),axis=1)/Context.time.time_resolution/Context.time.duration_years
    stored_energy = -np.sum(np.where(storage_TS<0,storage_TS,0),axis=1)/Context.time.time_resolution/Context.time.duration_years
    power_storage = np.sum(storage_TS,axis=0)
    annual_losses = np.sum(losses,axis=1)/Context.time.time_resolution/Context.time.duration_years      

    # =========================
    # GRID FLOWS
    # =========================
    
    Grid_importation = importation if (Context.optimization.connexion=='On-grid') else np.zeros(n_bits)
    Grid_exportation = exportation if (Context.optimization.connexion=='On-grid') else np.zeros(n_bits)  
    Grid_trading = trades if (Context.optimization.connexion=='On-grid') else np.zeros(n_bits)

    DG_production = importation if (Context.optimization.connexion=='Off-grid') else np.zeros(n_bits)
    curtailment = exportation if (Context.optimization.connexion=='Off-grid') else np.zeros(n_bits)
     
    # =========================
    # SELF-SUFFICIENCY / AUTONOMY
    # =========================

    obtained_self_sufficiency = (1-sum(Grid_importation)/sum(Optimized_Load)) if sum(Optimized_Load )>0 else np.nan
    REN_self_sufficiency = (1-sum(importation)/sum(Optimized_Load)) if sum(Optimized_Load )>0 else np.nan
    self_consumption = (1-sum(exportation)/sum(production)) if sum(production )>0 else np.nan
    REN_fraction = sum(production)/Annual_load if Annual_load>0 else np.nan

    Autonomy = 1-sum(Grid_importation>0)/n_bits
    REN_autonomy = 1-sum(importation>0)/n_bits
    Capacity_factor = sum(production)/(max(production)*n_bits) if max(production)>0 else np.nan

    obtained_constraint_level = obtained_self_sufficiency if Context.optimization.constraint_num==CONS_Self_sufficiency else self_consumption if (Context.optimization.constraint_num==CONS_Self_consumption) else REN_fraction if(Context.optimization.constraint_num==CONS_REN_fraction) else np.nan

    # =========================
    # GRID PARAMETERS
    # =========================
    
    Max_power_from_grid = max(Grid_importation) 
    Max_power_to_grid = max(Grid_exportation) 
    Max_curtailment = max(curtailment)   
    
    if (Context.optimization.connexion=='Off-grid'):
        Contract_power = 0           
    elif (Context.optimization.connexion=='On-grid'):          
        Contract_power = max(0, max(trades))

    # =========================
    # GENSET
    # =========================
    
    if (Context.optimization.connexion=='Off-grid'):
        DG_nominal_power = max(trades)
        DG_production=importation
        if (DG_nominal_power>0):
            closest_levels = np.array([int(10*(DG_production[i]/DG_nominal_power)-0.5) for i in range(n_bits)])
            annual_fuel_consumption_DG = sum(DG_production*Context.genset.fuel_consumption[closest_levels]/Context.time.time_resolution/Context.time.duration_years)
            DG_lifetime_years = Context.genset.lifetime/(sum(np.where(DG_production>0,1,0))/Context.time.time_resolution/Context.time.duration_years)
        else : 
            closest_levels = np.array([np.nan for i in range(n_bits)])
            annual_fuel_consumption_DG = 0
            DG_lifetime_years = np.nan
    elif (Context.optimization.connexion=='On-grid'):          
        DG_nominal_power = 0
        DG_production=np.zeros(n_bits)
        closest_levels = 0
        DG_lifetime_years = np.nan
        annual_fuel_consumption_DG = 0
        
    Max_power_from_DG = max(DG_production) 
    DG_min_production = solution.DG_min_production
    DG_min_runtime = solution.DG_min_runtime

    # =========================
    # ENVIRONMENT
    # =========================     
    
    if (Context.optimization.connexion=='Off-grid'):
        annual_CO2eq_DG = annual_fuel_consumption_DG*Context.genset.fuel_CO2eq_emissions/TONS_CONVERSION_FACTOR         
    elif (Context.optimization.connexion=='On-grid'):          
        annual_CO2eq_DG = 0.

    annual_CO2eq_prod = sum(sum(np.multiply(np.array([solution.production_set[i]*Context.production.unit_prods[i,:] for i in range(len(solution.production_set))]).T/KILOS_CONVERSION_FACTOR,np.array(Context.production.specs_num[:,PROD_EMISSIONS]))))/TONS_CONVERSION_FACTOR/Context.time.time_resolution/Context.time.duration_years
    annual_CO2eq_storage = np.inner(np.array([sum(np.where(storage_TS[i]>0,storage_TS[i],0)) for i in range(n_store)]),Context.storage.characteristics[STOR_EMISSIONS,:])/TONS_CONVERSION_FACTOR/Context.time.time_resolution/Context.time.duration_years
    annual_CO2eq_importation = sum(Grid_importation)*Context.grid.C02eqemissions/TONS_CONVERSION_FACTOR/Context.time.time_resolution/Context.time.duration_years if (Context.optimization.connexion=='On-grid') else 0

    annual_fossil_fuel_consumption_importation =  Context.grid.fossil_fuel_ratio*sum(Grid_importation)/Context.time.time_resolution/Context.time.duration_years if (Context.optimization.connexion=='On-grid') else 0
    
    annual_fossil_fuel_consumption = annual_fossil_fuel_consumption_importation+annual_fuel_consumption_DG
    annual_CO2eq_total = annual_CO2eq_DG+annual_CO2eq_prod+annual_CO2eq_importation+annual_CO2eq_storage

    # =========================
    # STORAGE
    # =========================        
    
    energy_storages = solution.storages[INDIV_PRO_VOLUME,:]/Context.storage.characteristics[STOR_DEPTH_OF_DISCHARGE,:]
    powers_out = solution.storages[INDIV_PRO_DISCHARGE_POWER,:]
    powers_in = solution.storages[INDIV_PRO_CHARGE_POWER,:]
    size_power=np.array([max(powers_in[i],powers_out[i]) for i in range(n_store)])
    Equivalent_cycles =  np.array([np.sum(np.abs(storage_TS[i,:]))/(2*Context.time.time_resolution*max(energy_storages[i],np.float64(1e-15))*Context.time.duration_years) for i in range(n_store)]) 
    Storage_lifetime = np.array([min(Context.storage.characteristics[STOR_LIFETIME,i],Context.storage.characteristics[STOR_CYCLE_LIFE,i]/max(1e-15,Equivalent_cycles[i])) for i in range(n_store)])
    annual_total_losses = sum(np.sum(losses,axis=1))/Context.time.time_resolution/Context.time.duration_years
     
    # =========================
    # STORAGE SOC analysis
    # =========================     
    
    minSOCs=(1-Context.storage.characteristics[STOR_DEPTH_OF_DISCHARGE,:])/2 
    SOCs=(np.divide((sum_diff_storages.T-np.min(sum_diff_storages,axis=1)),energy_storages) + minSOCs).T
    storage_NULL=tuple(energy_storages[i]==0 for i in range(n_store))
    
    State_SOCs=[np.searchsorted([(i+1)/100 for i in range(99)], SOCs[j,:]) for j in range(n_store)]
    distribution_Depth_of_discharge=np.array([[np.count_nonzero(State_SOCs[i]==j)/len(State_SOCs[i]) for j in range(100)] for i in range(n_store)])

    if sum(storage_NULL)>0:
        distribution_Depth_of_discharge[np.where(storage_NULL)[0][0]]=np.repeat(np.nan,100)
    
    dist_DOD = pd.DataFrame(data={'SOC Percentile':[i/100 for i in range(101)]})
    for i in range(n_store):
        dist_DOD = dist_DOD.join(pd.DataFrame(data={'SOC distribution ' + Context.storage.technologies[i]:distribution_Depth_of_discharge[i]}))

    # =========================
    # Energy return on investment
    # =========================      
    
    if (Context.optimization.connexion=='Off-grid'):
        productible_energy_DG = sum(DG_production)/Context.time.time_resolution/Context.time.duration_years*DG_lifetime_years
        consumed_energy_DG = productible_energy_DG/Context.genset.EROI
        
    elif (Context.optimization.connexion=='On-grid'):          
        productible_energy_DG = 0.
        consumed_energy_DG = 0.

    productible_energy = sum(np.sum(np.multiply(solution.production_set,Context.production.unit_prods.T),axis=0)/Context.time.time_resolution/Context.time.duration_years*Context.production.specs_num[:,PROD_LIFETIME])
    consumed_energy_production = sum(productible_energy/Context.production.specs_num[:,PROD_EROI])
    consumed_energy_storage = sum(np.nanmin(np.array([Context.storage.characteristics[STOR_CYCLE_LIFE,:],Context.storage.characteristics[STOR_LIFETIME,:]*Equivalent_cycles]),axis=0)*energy_storages/Context.storage.characteristics[STOR_ESOEI,:])

    EROI = (productible_energy+productible_energy_DG)/(consumed_energy_DG+consumed_energy_production+consumed_energy_storage)

    # =========================
    # ECONOMICS
    # =========================      
    
    CAPEX_storage_cost =  np.multiply(size_power,Context.storage.characteristics[STOR_POWER_COST,:]) + np.multiply(energy_storages,Context.storage.characteristics[STOR_ENERGY_COST,:]) + np.multiply(Context.storage.characteristics[STOR_INSTALLATION_COST,:],(size_power>np.zeros(n_store)))   
    economics_CAPEX_storage = sum(np.divide(CAPEX_storage_cost,Storage_lifetime))
    economics_OPEX_storage = sum(np.multiply(Context.storage.characteristics[STOR_OM_COST,:],size_power))

    if (Context.optimization.connexion=='Off-grid'):
        economics_exportation = 0
        economics_importation = 0
        economics_overrun = 0
        economics_contract_power = 0  
        DG_CAPEX_cost = DG_nominal_power*Context.genset.unit_cost/DG_lifetime_years
        DG_OPEX_cost = sum(DG_production)*Context.genset.maintenance_cost/Context.time.time_resolution/Context.time.duration_years
        annual_total_fuel_cost = annual_fuel_consumption*Context.genset.fuel_cost        
    elif (Context.optimization.connexion=='On-grid'): 
        economics_exportation = np.multiply(exportation,Context.grid.selling_price[solution.contract,:]).sum()/Context.time.time_resolution/Context.time.duration_years
        economics_importation = np.multiply(importation,Context.grid.prices[solution.contract,:]).sum()/Context.time.time_resolution/Context.time.duration_years
        economics_overrun = max(0,(max(importation)-Contract_power)*Context.grid.overrun[solution.contract])
        economics_contract_power = Context.grid.fixed_premium[solution.contract]*Contract_power
        DG_CAPEX_cost = 0
        DG_OPEX_cost = 0
        annual_total_fuel_cost = 0
        
    LCOE=(annual_cost_production+economics_CAPEX_storage+economics_OPEX_storage+DG_OPEX_cost+DG_CAPEX_cost+annual_total_fuel_cost+economics_importation+economics_contract_power+economics_overrun-economics_exportation)/(sum(Optimized_Load)/Context.time.time_resolution/Context.time.duration_years)   if sum(Optimized_Load )>0 else np.nan
    Annual_net_benefits = (-annual_cost_production-economics_CAPEX_storage-economics_OPEX_storage-DG_OPEX_cost-DG_CAPEX_cost-annual_total_fuel_cost-economics_importation-economics_contract_power-economics_overrun+economics_exportation)
    Value = Annual_net_benefits*Lifetime_installation
    Initial_investment = sum(CAPEX_storage_cost) +np.nanmin((0.,DG_CAPEX_cost*DG_lifetime_years))+ np.sum(Context.production.specs_num[:,PROD_CAPEX]*solution.production_set)
     
    # =========================
    # COST DECOMPOSITION
    # =========================        

    cost_production_per_kWh  = annual_cost_production/(sum(Optimized_Load)/Context.time.time_resolution/Context.time.duration_years)
    OPEX_storage_per_kWh = economics_OPEX_storage/(sum(Optimized_Load)/Context.time.time_resolution/Context.time.duration_years)
    CAPEX_storage_per_kWh = economics_CAPEX_storage/(sum(Optimized_Load)/Context.time.time_resolution/Context.time.duration_years)
    contract_power_per_kWh = economics_contract_power/(sum(Optimized_Load)/Context.time.time_resolution/Context.time.duration_years)
    overrun_penalty_per_kWh = economics_overrun/(sum(Optimized_Load)/Context.time.time_resolution/Context.time.duration_years)
    energy_importation_per_kWh = economics_importation/(sum(Optimized_Load)/Context.time.time_resolution/Context.time.duration_years)
    fixed_premium_per_kWh = economics_contract_power/(sum(Optimized_Load)/Context.time.time_resolution/Context.time.duration_years)
    DG_fuel_cost_per_kWh = annual_total_fuel_cost/(sum(Optimized_Load)/Context.time.time_resolution/Context.time.duration_years)
    DG_CAPEX_cost_per_kWh = DG_CAPEX_cost/(sum(Optimized_Load)/Context.time.time_resolution/Context.time.duration_years)
    DG_OPEX_cost_per_kWh = DG_OPEX_cost/(sum(Optimized_Load)/Context.time.time_resolution/Context.time.duration_years)
    energy_exportation_per_kWh = economics_exportation/(sum(Optimized_Load)/Context.time.time_resolution/Context.time.duration_years)

    # =========================
    # Demand-side management
    # =========================
    
    indexes_hour = [[int((i+j*Context.time.time_resolution*24)) for j in range(Context.time.n_days)] for i in range(int(Context.time.time_resolution*24))]
    indexes_days = [[int(i+j*Context.time.time_resolution*24) for i in range(int(Context.time.time_resolution*24))] for j in range(Context.time.n_days)]

    Daily_base_load = [np.mean(Context.loads.non_movable[indexes_hour[j]]) for j in range(int(Context.time.time_resolution*24))]
    Daily_movable_loads = [np.mean(Context.loads.D_movable[indexes_hour[j]]+Context.loads.Y_movable[indexes_hour[j]]) for j in range(int(Context.time.time_resolution*24))]
    Daily_final_loads = [np.mean(Optimized_Load[indexes_hour[j]]) for j in range(int(Context.time.time_resolution*24))]
    Daily_prod = [np.mean(production[indexes_hour[j]]) for j in range(int(Context.time.time_resolution*24))]

    # =========================
    # TIME BALANCING
    # =========================
    
    Yearly_base_load = [np.mean(Context.loads.non_movable[indexes_days[j]]+Context.loads.D_movable[indexes_days[j]]) for j in range(Context.time.n_days)]
    Yearly_movable_loads = [np.mean(Context.loads.Y_movable[indexes_days[j]]) for j in range(Context.time.n_days)]
    Yearly_final_loads = [np.mean(Optimized_Load[indexes_days[j]]) for j in range(Context.time.n_days)]
    Yearly_prod = [np.mean(production[indexes_days[j]]) for j in range(Context.time.n_days)]

 
    daily_time_balancing = pd.DataFrame({
        "Datetime": datetime[:int(Context.time.time_resolution * HOURS_PER_DAY)].strftime("%H:%M"),
        "Daily load (kW)": Daily_final_loads,
        "Daily prod (kW)": Daily_prod,
        "Daily importation (kW)": [np.mean(importation[idx]) for idx in indexes_hour],
        "Daily exportation (kW)": [np.mean(exportation[idx]) for idx in indexes_hour],
        "Daily storage (kW)": [np.mean(power_storage[idx]) for idx in indexes_hour],})
    
    yearly_time_balancing = pd.DataFrame({
        "Datetime": datetime[indexes_hour[0]].strftime('%Y-%m-%d'),
        "Yearly load (kW)": Yearly_final_loads,
        "Yearly prod (kW)": Yearly_prod,
        "Yearly importation (kW)": [np.mean(importation[indexes_days[j]]) for j in range(Context.time.n_days)],
        "Yearly exportation (kW)": [-np.mean(exportation[indexes_days[j]]) for j in range(Context.time.n_days)],
        "Yearly storage (kW)": [np.mean(power_storage[indexes_days[j]]) for j in range(Context.time.n_days)],})   

    # =========================
    # Demand-side management strategies
    # =========================

    DSM_daily_strategy = pd.DataFrame(data={'Datetime':datetime[0:int(Context.time.time_resolution*24)].strftime('%H:%M'),'Daily base load (kW)':Daily_base_load,'Daily movable load (kW)':Daily_movable_loads,'Daily final load (kW)':Daily_final_loads,'Daily prod (kW)':Daily_prod})
    DSM_yearly_strategy = pd.DataFrame(data={'Datetime':datetime[indexes_hour[0]].strftime('%Y-%m-%d'),'Yearly base load (kW)':Yearly_base_load,'Yearly movable load (kW)':Yearly_movable_loads,'Yearly final load (kW)':Yearly_final_loads,'Yearly prod (kW)':Yearly_prod})
    Load_strategy = pd.DataFrame(data={'Datetime':datetime,'Non controllable load (kW)':Context.loads.non_movable,'Daily movable load (kW)':Context.loads.D_movable,'Yearly movable load (kW)':Context.loads.Y_movable, 'Daily optimized load (kW)':D_DSM.flatten(),'Yearly optimized load (kW)':Y_DSM})
 
    # =========================
    # OVERVIEW
    # =========================   
    
    logicals_sells = np.where((trades<=0) & (signal<=0))[0]
    logicals_buys = np.where((trades>=0) & (signal>=0))[0]
    illogicals_buys = np.where((trades>0) & (signal<=0))[0]
    illogicals_sells = np.where((trades<0) & (signal>=0))[0]
    use_logical_sells = (sum(Optimized_Load[logicals_sells]),-sum(tuple(sum(np.where(storage_TS[i][logicals_sells]<0,storage_TS[i][logicals_sells],0) for i in range(n_store)))),-sum(trades[logicals_sells]))/Context.time.time_resolution/Context.time.duration_years-sum(tuple(sum(np.where(storage_TS[i][logicals_sells]>0,storage_TS[i][logicals_sells],0) for i in range(n_store))))
    use_logical_buys = (sum(production[logicals_buys]),sum(tuple(sum(np.where(storage_TS[i][logicals_buys]>0,storage_TS[i][logicals_buys],0) for i in range(n_store)))),sum(trades[logicals_buys]))/Context.time.time_resolution/Context.time.duration_years
    use_illogical_sells = (sum(production[illogicals_sells]),sum(tuple(sum(np.where(storage_TS[i][illogicals_sells]>0,storage_TS[i][illogicals_sells],0) for i in range(n_store))))-sum(trades[illogicals_sells]),-sum(trades[illogicals_sells]))/Context.time.time_resolution/Context.time.duration_years
    use_illogical_buys = (sum(Optimized_Load[illogicals_buys]),sum(production[illogicals_buys]-sum(Optimized_Load[illogicals_buys])))/Context.time.time_resolution/Context.time.duration_years
    
    when_prod_exceeds = pd.DataFrame(data=((sum(Optimized_Load[logicals_sells])+sum(Optimized_Load[illogicals_buys]))/Context.time.time_resolution/Context.time.duration_years,-sum(trades[logicals_sells])/Context.time.time_resolution,sum(trades[illogicals_buys])/Context.time.time_resolution) + tuple(-sum(storage_TS[i][(signal<0) & (storage_TS[i]<0)])/Context.time.time_resolution for i in range(n_store)) + tuple(sum(storage_TS[i][(signal<0) & (storage_TS[i]>0)])/Context.time.time_resolution for i in range(n_store)),index=['Load (kWh)','Grid export (kWh)','Grid import (kWh)']+['Storage '+Context.storage.technologies[i]+' charge (kWh)' for i in range(n_store)]+['Storage '+Context.storage.technologies[i]+' discharge (kWh)' for i in range(n_store)]).transpose()
    when_load_exceeds = pd.DataFrame(data=(sum(production[logicals_buys])/Context.time.time_resolution+sum(production[illogicals_sells])/Context.time.time_resolution,-sum(trades[(signal>0) & (trades<0) ])/Context.time.time_resolution,sum(trades[(signal>0) & (trades>0) ])/Context.time.time_resolution) + tuple(-sum(storage_TS[i][(signal>0) & (storage_TS[i]<0)])/Context.time.time_resolution for i in range(n_store)) + tuple(sum(storage_TS[i][(signal>0) & (storage_TS[i]>0)])/Context.time.time_resolution for i in range(n_store)),index=['Production (kWh)','Grid export (kWh)','Grid import (kWh)']+['Storage '+Context.storage.technologies[i]+' charge (kWh)' for i in range(n_store)]+['Storage '+Context.storage.technologies[i]+' discharge (kWh)' for i in range(n_store)]).transpose()
     
    useprod = pd.DataFrame(data=(use_logical_sells[1]+use_illogical_buys[1],use_logical_buys[0]+use_logical_sells[0]+use_illogical_sells[0]+use_illogical_buys[0],use_illogical_sells[2]+use_logical_sells[2] ,Annual_REN_production),index=['Storage (kWh)' ,'Load (kWh)','Exportation (kWh)','Annual REN production (kWh)']).transpose()
    Loadmeet = pd.DataFrame(data=(use_logical_buys[1]+use_illogical_sells[1] ,use_logical_buys[0]+use_logical_sells[0]+use_illogical_sells[0]+use_illogical_buys[0],use_logical_buys[2] ,Annual_load),index=['Storage (kWh)', 'Production (kWh)','Importation (kWh)','Annual load (kWh)']).transpose()
    Grid_use_export = pd.DataFrame(data=(useprod['Exportation (kWh)'], Annual_sum_exportation-useprod['Exportation (kWh)']),index=['Production (kWh)','Storage (kWh)']).transpose()
    Grid_use_import = pd.DataFrame(data=(Loadmeet['Importation (kWh)'], Annual_sum_importation-Loadmeet['Importation (kWh)']),index=['Load (kWh)','Storage (kWh)']).transpose()    

    # =========================
    # FITNESS & TECHNICAL
    # =========================   
    
    installation_lifetime = np.nanmin(np.array([DG_lifetime_years,min(Storage_lifetime),np.nanmin(np.where(solution.production_set>0,Context.production.specs_num[:,PROD_LIFETIME],np.nan))]))
    fitness = solution.fitness  
    
    return{'TimeSeries':{'Storage_TS (kW)':storage_TS,'Storage cumulative energy (kWh)':sum_diff_storages,'SOCs (%)':SOCs,'Losses (kW)': losses,'D_DSM (kW)':D_DSM,'Y_DSM (kW)':Y_DSM,'trades (kW)':trades,'Optimized load (kW)':Optimized_Load,'Non movable load (kW)':Context.loads.non_movable,'production (kW)':production,'Grid trading (kW)':Grid_trading,'Grid importation (kW)':Grid_importation,'Grid exportation (kW)':Grid_exportation,'DG production (kW)':DG_production,'Curtailment (kW)':curtailment},
           'Technical':{'fitness':fitness,'Self-sufficiency':obtained_self_sufficiency,'Self-consumption':self_consumption,'Autonomy':Autonomy,'REN fraction':REN_fraction,'REN Self-sufficiency':REN_self_sufficiency,'REN Autonomy':REN_autonomy,'Annual sum losses (kWh)':annual_total_losses,'Contract power (kW)':Contract_power,'Capacity factor':Capacity_factor,'Max power from grid (kW)':Max_power_from_grid,'Max power to grid (kW)':Max_power_to_grid,'Max curtailment (kW)':Max_curtailment,'Installation lifetime (yrs.)':installation_lifetime },
           'economics':{'LCOE (€/kWh)':LCOE,'Cost production (€/kWh)':cost_production_per_kWh,'OPEX storage (€/kWh)':OPEX_storage_per_kWh,'CAPEX storage (€/kWh)':CAPEX_storage_per_kWh, 'Contract power (€/kWh)':contract_power_per_kWh,'Overrun penalty (€/kWh)':overrun_penalty_per_kWh,'Energy importation (€/kWh)':energy_importation_per_kWh,'Fixed premium (€/kWh)':fixed_premium_per_kWh,'DG fuel cost (€/kWh)':DG_fuel_cost_per_kWh,'DG CAPEX cost (€/kWh)':DG_CAPEX_cost_per_kWh,'DG OPEX cost (€/kWh)':DG_OPEX_cost_per_kWh,'Energy exportation (€/kWh)':energy_exportation_per_kWh,'Annual net benefits (€/yrs.)':Annual_net_benefits,'Value (€)':Value,'Initial investment (€)':Initial_investment}, 
           'Storages':{'Capacity (kWh)':energy_storages,'Powers in (kW)':powers_in,'Powers out (kW)':powers_out,'Min SOCs (%)':minSOCs ,'Equivalent cycles':Equivalent_cycles,'Storage lifetime (yrs.)':Storage_lifetime}, 
           'Genset':{'nominal_power (kW)':DG_nominal_power, 'Max power from DG (kW)':Max_power_from_DG, 'DG lifetime (yrs.)':DG_lifetime_years,'DG min. production':solution.DG_min_production, 'DG min. runtime':solution.DG_min_runtime },
           'Genset power distribution':closest_levels,
           'EMS' : {'D_DSM min. levels':solution.D_DSM_minimum_levels,'Y_DSM min. levels':solution.Y_DSM_minimum_levels,'discharge order':solution.discharge_order,'strategy':solution.DG_strategy, 'overlaps':solution.overlaps,'energy repartition coefficient':solution.energy_use_coefficient },
           'Environment':{'Annual fossil fuel from grid (kWh)':annual_fossil_fuel_consumption_importation,'Annual fossil fuel consumption from DG (kWh)':annual_fuel_consumption_DG,'Annual fossil fuel consumption (kWh)':annual_fossil_fuel_consumption,'Annual CO2eq total (tCO2)':annual_CO2eq_total,'expected produced REN energy (kWh)':productible_energy,'expected produced energy from DG (kWh)':productible_energy_DG,'consumed energy for DG (kWh)':consumed_energy_DG ,'Consumed energy for production (kWh)':consumed_energy_production ,'Consumed energy for storage (kWh)':consumed_energy_storage,'EROI':EROI },
           'Flows':{'Annual load (kWh)':Annual_load,'Annual REN production' : Annual_REN_production ,'Sum importation (kWh)':Annual_sum_importation,'Sum exportation (kWh)':Annual_sum_exportation},
           'Flows storages':{'Annual stored energy (kWh)':stored_energy,'Annual reported energy (kWh)':reported_energy,'Annual losses (kWh)':annual_losses},
           'Extra_outputs':{'Uses':{'useprod':useprod,'Loadmeet':Loadmeet,'when_prod':when_prod_exceeds,'when_load':when_load_exceeds},'Logics':{'logicals_sells':logicals_sells,'logicals_buys':logicals_buys,'illogicals_sells':illogicals_sells,'illogicals_buys':illogicals_buys},'Grid usage':{'export':Grid_use_export,'import':Grid_use_import},'distribution_DOD': dist_DOD },
           'Balancing':{'daily time balancing':daily_time_balancing,'yearly time balancing':yearly_time_balancing},
           'Demand-side management':{'DSM daily strategy':DSM_daily_strategy,'DSM yearly strategy':DSM_yearly_strategy,'Load strategy':Load_strategy}
           }













def KPI_pro_old(solution,Context,datetime):
    """
    Comprehensive cost and performance evaluation of the final (best) scenario.
    
    This function computes the annualized technical, economic, environmental,
    and operational metrics of a microgrid system, including:
    
        - Net load and production balance
        - Multi-storage operation and state-of-charge (SOC) evolution
        - Grid interaction (import/export)
        - Diesel generator (DG) operation, fuel consumption, and costs
        - Demand-Side Management (DSM) strategies for daily and yearly loads
        - Energy flows, losses, and curtailment
        - Economic indicators (LCOE, CAPEX, OPEX, net benefits)
        - Environmental indicators (CO2eq emissions, fossil fuel consumption)
        - Storage performance metrics (capacity, cycles, lifetime, SOC distribution)
        - Microgrid self-sufficiency, autonomy, and renewable fraction
        - Detailed time-series outputs for load, production, storage, and trades
    
    Args:
        gene (structured object compatible with Numba): Microgrid operational strategy, design (production and storage), and PMS parameters.
        datetime (pandas.DatetimeIndex or numpy.ndarray): Time series for the simulation period.
        storage_characteristics (numpy.ndarray): Array of storage parameters including capacities, power ratings, efficiencies, and lifetime metrics.
        time_resolution (float): Time resolution.
        n_store (int): Number of storage units in the microgrid.
        duration_years (float): Duration of the simulation in years.
        Non_movable_load (numpy.ndarray): Time series of non-flexible (base) load.
        D_movable_load (numpy.ndarray): Daily totals of daily-flexible load.
        total_D_Movable_load (numpy.ndarray): Daily totals of daily-flexible load.
        Y_movable_load (numpy.ndarray): Time series of yearly-flexible load.
        total_Y_Movable_load (float): Yearly total of yearly-flexible load.
        prod_C (numpy.ndarray): Current on-site renewable energy production time series (kW).
        prods_U (numpy.ndarray): Unit production values.
        Bounds_prod (np.array): Bounds on production set.
        constraint_num (int): Index of the constraint.
        constraint_level (float): Level of constraint.
        cost_constraint (float): Cost associated with the constraint.
        n_bits (int): Number of timesteps in the simulation.
        Connexion (str): Grid connection mode ('On-grid' or 'Off-grid').
        DG_fuel_consumption (np.array): Diesel generator consumption.
        DG_fuel_cost (float): Diesel generator fuel cost.
        DG_unit_cost (int): Diesel generator unit cost (€/kW).
        DG_lifetime (float): Diesel generator lifetime (hrs).
        DG_maintenance_cost (float): Diesel generator maintenance cost (€/kWh/yrs.).
        DG_EROI (float): Diesel generator energy returns on investment.
        fuel_CO2eq_emissions (float): Diesel generator eq CO2 emissions.
        storage_techs (list of str): List of storage technology names.
        n_days (int): Number of days in the simulation period.
    
    Returns:
        dict: Dictionary containing multiple sub-dictionaries with metrics and outputs:
            - TimeSeries: Time series of storage state, losses, trades, production, grid flows, and DSM-adjusted loads.
            - Technical: Technical KPIs such as self-sufficiency, autonomy, REN fraction, capacity factor, and installation lifetime.
            - economics: Economic indicators including LCOE, CAPEX/OPEX per kWh, net benefits, and initial investment.
            - Storages: Storage-specific metrics including capacity, power, SOCs, equivalent cycles, and lifetimes.
            - DG: Diesel generator KPIs including nominal power, lifetime, and min. runtime/production.
            - DG power distribution: Level distribution of diesel generator power output.
            - PMS: Demand-side management strategy and settings.
            - Environment: Annual fossil fuel consumption, CO2 emissions, produced and consumed energy, and EROI.
            - Flows: Annual energy flows including load, production, importation, and exportation.
            - Flows storages: Annual energy stored, reported energy, and storage losses.
            - Extra_outputs: Miscellaneous outputs including usage, logical/illogical trades, grid usage, and SOC distributions.
            - Balancing: Daily and yearly time balancing of load, production, import/export, and storage.
            - Demand-side management: Daily and yearly DSM strategies and load adjustments.
    
    Note:
        The function distinguishes between logical and illogical energy trades, calculates
        energy curtailment, storage losses, and accounts for efficiency of storage and DG.
        It is suitable for both on-grid and off-grid microgrid simulations.
    
    Important:
        This is the final evaluation function used to compute ERMESS outputs.
    """
    
    from ERMESS_scripts.evolutionnary_core import ERMESS_functions as Ef
    
    # =========================
    # CONSTANTS
    # =========================
    TONS_CONVERSION_FACTOR = 1000000
    KILOS_CONVERSION_FACTOR = 1000
    HOURS_PER_DAY = 24
    
    # =========================
    # LOAD & PRODUCTION
    # =========================

    production = ((Context.production.unit_prods.T*solution.production_set).sum(axis=1)+Context.production.current_prod)/KILOS_CONVERSION_FACTOR    
    annual_cost_production = np.sum((Context.production.specs_num[:,PROD_CAPEX]/Context.production.specs_num[:,PROD_LIFETIME]+Context.production.specs_num[:,PROD_OPEX])*(solution.production_set))

    # =========================
    # CORE ENERGY BALANCE
    # =========================
        
    pro_parameters, global_parameters, grid_parameters, RENSystems_parameters, Genset_parameters, extra_parameters = Ef.build_numba_params(Context)
    (storage_TS,trades,D_DSM,Y_DSM,SOCs_eff,losses,P_diff) = Eems.LFE_CCE(solution, global_parameters, pro_parameters, production ,RENSystems_parameters)
    
    dispatching_timeseries = {"D_DSM" : D_DSM , "Y_DSM":Y_DSM, "storage_TS":storage_TS, "trades":trades,"losses" : losses}


    
    
    
    Optimized_Load = Context.loads.non_movable+Y_DSM+D_DSM
    Annual_load = sum(Optimized_Load)/Context.time.time_resolution/Context.time.duration_years
    Annual_REN_production = sum(production)/Context.time.time_resolution/Context.time.duration_years
    importation=np.where(trades>0,trades,0)
    exportation = np.where(trades<0,-trades,0)
    signal =  Optimized_Load - production
    Annual_sum_importation=sum(importation)/Context.time.time_resolution/Context.time.duration_years
    Annual_sum_exportation=sum(exportation)/Context.time.time_resolution/Context.time.duration_years

    # =========================
    # STORAGE INITIALIZATION
    # =========================    
    
    n_store = Context.storage.n_store
    n_bits = Context.time.n_bits
    
    sum_diff_storages = np.array([-np.cumsum(storage_TS[i,:]/Context.time.time_resolution+losses[i,:]/Context.time.time_resolution) for i in range(n_store)])             
    reported_energy = np.sum(np.where(storage_TS>0,storage_TS,0),axis=1)/Context.time.time_resolution/Context.time.duration_years
    stored_energy = -np.sum(np.where(storage_TS<0,storage_TS,0),axis=1)/Context.time.time_resolution/Context.time.duration_years
    power_storage = np.sum(storage_TS,axis=0)
    annual_losses = np.sum(losses,axis=1)/Context.time.time_resolution/Context.time.duration_years      

    # =========================
    # GRID FLOWS
    # =========================
    
    Grid_importation = importation if (Context.optimization.connexion=='On-grid') else np.zeros(n_bits)
    Grid_exportation = exportation if (Context.optimization.connexion=='On-grid') else np.zeros(n_bits)  
    Grid_trading = trades if (Context.optimization.connexion=='On-grid') else np.zeros(n_bits)

    DG_production = importation if (Context.optimization.connexion=='Off-grid') else np.zeros(n_bits)
    curtailment = exportation if (Context.optimization.connexion=='Off-grid') else np.zeros(n_bits)
     
    # =========================
    # SELF-SUFFICIENCY / AUTONOMY
    # =========================

    obtained_self_sufficiency = (1-sum(Grid_importation)/sum(Optimized_Load)) if sum(Optimized_Load )>0 else np.nan
    REN_self_sufficiency = (1-sum(importation)/sum(Optimized_Load)) if sum(Optimized_Load )>0 else np.nan
    self_consumption = (1-sum(exportation)/sum(production)) if sum(production )>0 else np.nan
    REN_fraction = sum(production)/Annual_load if Annual_load>0 else np.nan

    Autonomy = 1-sum(Grid_importation>0)/n_bits
    REN_autonomy = 1-sum(importation>0)/n_bits
    Capacity_factor = sum(production)/(max(production)*n_bits) if max(production)>0 else np.nan

    obtained_constraint_level = obtained_self_sufficiency if Context.optimization.constraint_num==CONS_Self_sufficiency else self_consumption if (Context.optimization.constraint_num==CONS_Self_consumption) else REN_fraction if(Context.optimization.constraint_num==CONS_REN_fraction) else np.nan

    # =========================
    # GRID PARAMETERS
    # =========================
    
    Max_power_from_grid = max(Grid_importation) 
    Max_power_to_grid = max(Grid_exportation) 
    Max_curtailment = max(curtailment)   
    
    if (Context.optimization.connexion=='Off-grid'):
        Contract_power = 0           
    elif (Context.optimization.connexion=='On-grid'):          
        Contract_power = max(0, max(trades))

    # =========================
    # GENSET
    # =========================
    
    if (Context.optimization.connexion=='Off-grid'):
        DG_nominal_power = max(trades)
        DG_production=importation
        if (DG_nominal_power>0):
            closest_levels = np.array([int(10*(DG_production[i]/DG_nominal_power)-0.5) for i in range(n_bits)])
            annual_fuel_consumption_DG = sum(DG_production*Context.genset.fuel_consumption[closest_levels]/Context.time.time_resolution/Context.time.duration_years)
            DG_lifetime_years = Context.genset.lifetime/(sum(np.where(DG_production>0,1,0))/Context.time.time_resolution/Context.time.duration_years)
        else : 
            closest_levels = np.array([np.nan for i in range(n_bits)])
            annual_fuel_consumption_DG = 0
            DG_lifetime_years = np.nan
    elif (Context.optimization.connexion=='On-grid'):          
        DG_nominal_power = 0
        DG_production=np.zeros(n_bits)
        closest_levels = 0
        DG_lifetime_years = np.nan
        annual_fuel_consumption_DG = 0
        
    Max_power_from_DG = max(DG_production) 
    DG_min_production = solution.DG_min_production
    DG_min_runtime = solution.DG_min_runtime

    # =========================
    # ENVIRONMENT
    # =========================     
    
    if (Context.optimization.connexion=='Off-grid'):
        annual_CO2eq_DG = annual_fuel_consumption_DG*Context.genset.fuel_CO2eq_emissions/TONS_CONVERSION_FACTOR         
    elif (Context.optimization.connexion=='On-grid'):          
        annual_CO2eq_DG = 0.

    annual_CO2eq_prod = sum(sum(np.multiply(np.array([solution.production_set[i]*Context.production.unit_prods[i,:] for i in range(len(solution.production_set))]).T/KILOS_CONVERSION_FACTOR,np.array(Context.production.specs_num[:,PROD_EMISSIONS]))))/TONS_CONVERSION_FACTOR/Context.time.time_resolution/Context.time.duration_years
    annual_CO2eq_storage = np.inner(np.array([sum(np.where(storage_TS[i]>0,storage_TS[i],0)) for i in range(n_store)]),Context.storage.characteristics[STOR_EMISSIONS,:])/TONS_CONVERSION_FACTOR/Context.time.time_resolution/Context.time.duration_years
    annual_CO2eq_importation = sum(Grid_importation)*Context.grid.C02eqemissions/TONS_CONVERSION_FACTOR/Context.time.time_resolution/Context.time.duration_years if (Context.optimization.connexion=='On-grid') else 0

    annual_fossil_fuel_consumption_importation =  Context.grid.fossil_fuel_ratio*sum(Grid_importation)/Context.time.time_resolution/Context.time.duration_years if (Context.optimization.connexion=='On-grid') else 0
    
    annual_fossil_fuel_consumption = annual_fossil_fuel_consumption_importation+annual_fuel_consumption_DG
    annual_CO2eq_total = annual_CO2eq_DG+annual_CO2eq_prod+annual_CO2eq_importation+annual_CO2eq_storage

    # =========================
    # STORAGE
    # =========================        
    
    energy_storages = solution.storages[INDIV_PRO_VOLUME,:]/Context.storage.characteristics[STOR_DEPTH_OF_DISCHARGE,:]
    powers_out = solution.storages[INDIV_PRO_DISCHARGE_POWER,:]
    powers_in = solution.storages[INDIV_PRO_CHARGE_POWER,:]
    size_power=np.array([max(powers_in[i],powers_out[i]) for i in range(n_store)])
    Equivalent_cycles =  np.array([np.sum(np.abs(storage_TS[i,:]))/(2*Context.time.time_resolution*max(energy_storages[i],np.float64(1e-15))*Context.time.duration_years) for i in range(n_store)]) 
    Storage_lifetime = np.array([min(Context.storage.characteristics[STOR_LIFETIME,i],Context.storage.characteristics[STOR_CYCLE_LIFE,i]/max(1e-15,Equivalent_cycles[i])) for i in range(n_store)])
    annual_total_losses = sum(np.sum(losses,axis=1))/Context.time.time_resolution/Context.time.duration_years
     
    # =========================
    # STORAGE SOC analysis
    # =========================     
    
    minSOCs=(1-Context.storage.characteristics[STOR_DEPTH_OF_DISCHARGE,:])/2 
    SOCs=(np.divide((sum_diff_storages.T-np.min(sum_diff_storages,axis=1)),energy_storages) + minSOCs).T
    storage_NULL=tuple(energy_storages[i]==0 for i in range(n_store))
    
    State_SOCs=[np.searchsorted([(i+1)/100 for i in range(99)], SOCs[j,:]) for j in range(n_store)]
    distribution_Depth_of_discharge=np.array([[np.count_nonzero(State_SOCs[i]==j)/len(State_SOCs[i]) for j in range(100)] for i in range(n_store)])

    if sum(storage_NULL)>0:
        distribution_Depth_of_discharge[np.where(storage_NULL)[0][0]]=np.repeat(np.nan,100)
    
    dist_DOD = pd.DataFrame(data={'SOC Percentile':[i/100 for i in range(101)]})
    for i in range(n_store):
        dist_DOD = dist_DOD.join(pd.DataFrame(data={'SOC distribution ' + Context.storage.technologies[i]:distribution_Depth_of_discharge[i]}))

    # =========================
    # Energy return on investment
    # =========================      
    
    if (Context.optimization.connexion=='Off-grid'):
        productible_energy_DG = sum(DG_production)/Context.time.time_resolution/Context.time.duration_years*DG_lifetime_years
        consumed_energy_DG = productible_energy_DG/Context.genset.EROI
        
    elif (Context.optimization.connexion=='On-grid'):          
        productible_energy_DG = 0.
        consumed_energy_DG = 0.

    productible_energy = sum(np.sum(np.multiply(solution.production_set,Context.production.unit_prods.T),axis=0)/Context.time.time_resolution/Context.time.duration_years*Context.production.specs_num[:,PROD_LIFETIME])
    consumed_energy_production = sum(productible_energy/Context.production.specs_num[:,PROD_EROI])
    consumed_energy_storage = sum(np.nanmin(np.array([Context.storage.characteristics[STOR_CYCLE_LIFE,:],Context.storage.characteristics[STOR_LIFETIME,:]*Equivalent_cycles]),axis=0)*energy_storages/Context.storage.characteristics[STOR_ESOEI,:])

    EROI = (productible_energy+productible_energy_DG)/(consumed_energy_DG+consumed_energy_production+consumed_energy_storage)

    # =========================
    # ECONOMICS
    # =========================      
    
    CAPEX_storage_cost =  np.multiply(size_power,Context.storage.characteristics[STOR_POWER_COST,:]) + np.multiply(energy_storages,Context.storage.characteristics[STOR_ENERGY_COST,:]) + np.multiply(Context.storage.characteristics[STOR_INSTALLATION_COST,:],(size_power>np.zeros(n_store)))   
    economics_CAPEX_storage = sum(np.divide(CAPEX_storage_cost,Storage_lifetime))
    economics_OPEX_storage = sum(np.multiply(Context.storage.characteristics[STOR_OM_COST,:],size_power))

    if (Context.optimization.connexion=='Off-grid'):
        economics_exportation = 0
        economics_importation = 0
        economics_overrun = 0
        economics_contract_power = 0  
        DG_CAPEX_cost = DG_nominal_power*Context.genset.unit_cost/DG_lifetime_years
        DG_OPEX_cost = sum(DG_production)*Context.genset.maintenance_cost/Context.time.time_resolution/Context.time.duration_years
        annual_total_fuel_cost = annual_fuel_consumption*Context.genset.fuel_cost        
    elif (Context.optimization.connexion=='On-grid'): 
        economics_exportation = np.multiply(exportation,Context.grid.selling_price[solution.contract,:]).sum()/Context.time.time_resolution/Context.time.duration_years
        economics_importation = np.multiply(importation,Context.grid.prices[solution.contract,:]).sum()/Context.time.time_resolution/Context.time.duration_years
        economics_overrun = max(0,(max(importation)-Contract_power)*Context.grid.overrun[solution.contract])
        economics_contract_power = Context.grid.fixed_premium[solution.contract]*Contract_power
        DG_CAPEX_cost = 0
        DG_OPEX_cost = 0
        annual_total_fuel_cost = 0
        
    LCOE=(annual_cost_production+economics_CAPEX_storage+economics_OPEX_storage+DG_OPEX_cost+DG_CAPEX_cost+annual_total_fuel_cost+economics_importation+economics_contract_power+economics_overrun-economics_exportation)/(sum(Optimized_Load)/Context.time.time_resolution/Context.time.duration_years)   if sum(Optimized_Load )>0 else np.nan
    Annual_net_benefits = (-annual_cost_production-economics_CAPEX_storage-economics_OPEX_storage-DG_OPEX_cost-DG_CAPEX_cost-annual_total_fuel_cost-economics_importation-economics_contract_power-economics_overrun+economics_exportation)
    Value = Annual_net_benefits*Lifetime_installation
    Initial_investment = sum(CAPEX_storage_cost) +np.nanmin((0.,DG_CAPEX_cost*DG_lifetime_years))+ np.sum(Context.production.specs_num[:,PROD_CAPEX]*solution.production_set)
     
    # =========================
    # COST DECOMPOSITION
    # =========================        

    cost_production_per_kWh  = annual_cost_production/(sum(Optimized_Load)/Context.time.time_resolution/Context.time.duration_years)
    OPEX_storage_per_kWh = economics_OPEX_storage/(sum(Optimized_Load)/Context.time.time_resolution/Context.time.duration_years)
    CAPEX_storage_per_kWh = economics_CAPEX_storage/(sum(Optimized_Load)/Context.time.time_resolution/Context.time.duration_years)
    contract_power_per_kWh = economics_contract_power/(sum(Optimized_Load)/Context.time.time_resolution/Context.time.duration_years)
    overrun_penalty_per_kWh = economics_overrun/(sum(Optimized_Load)/Context.time.time_resolution/Context.time.duration_years)
    energy_importation_per_kWh = economics_importation/(sum(Optimized_Load)/Context.time.time_resolution/Context.time.duration_years)
    fixed_premium_per_kWh = economics_contract_power/(sum(Optimized_Load)/Context.time.time_resolution/Context.time.duration_years)
    DG_fuel_cost_per_kWh = annual_total_fuel_cost/(sum(Optimized_Load)/Context.time.time_resolution/Context.time.duration_years)
    DG_CAPEX_cost_per_kWh = DG_CAPEX_cost/(sum(Optimized_Load)/Context.time.time_resolution/Context.time.duration_years)
    DG_OPEX_cost_per_kWh = DG_OPEX_cost/(sum(Optimized_Load)/Context.time.time_resolution/Context.time.duration_years)
    energy_exportation_per_kWh = economics_exportation/(sum(Optimized_Load)/Context.time.time_resolution/Context.time.duration_years)

    # =========================
    # Demand-side management
    # =========================
    
    indexes_hour = [[int((i+j*Context.time.time_resolution*24)) for j in range(Context.time.n_days)] for i in range(int(Context.time.time_resolution*24))]
    indexes_days = [[int(i+j*Context.time.time_resolution*24) for i in range(int(Context.time.time_resolution*24))] for j in range(Context.time.n_days)]

    Daily_base_load = [np.mean(Context.loads.non_movable[indexes_hour[j]]) for j in range(int(Context.time.time_resolution*24))]
    Daily_movable_loads = [np.mean(Context.loads.D_movable[indexes_hour[j]]+Context.loads.Y_movable[indexes_hour[j]]) for j in range(int(Context.time.time_resolution*24))]
    Daily_final_loads = [np.mean(Optimized_Load[indexes_hour[j]]) for j in range(int(Context.time.time_resolution*24))]
    Daily_prod = [np.mean(production[indexes_hour[j]]) for j in range(int(Context.time.time_resolution*24))]

    # =========================
    # TIME BALANCING
    # =========================
    
    Yearly_base_load = [np.mean(Context.loads.non_movable[indexes_days[j]]+Context.loads.D_movable[indexes_days[j]]) for j in range(Context.time.n_days)]
    Yearly_movable_loads = [np.mean(Context.loads.Y_movable[indexes_days[j]]) for j in range(Context.time.n_days)]
    Yearly_final_loads = [np.mean(Optimized_Load[indexes_days[j]]) for j in range(Context.time.n_days)]
    Yearly_prod = [np.mean(production[indexes_days[j]]) for j in range(Context.time.n_days)]

 
    daily_time_balancing = pd.DataFrame({
        "Datetime": datetime[:int(Context.time.time_resolution * HOURS_PER_DAY)].strftime("%H:%M"),
        "Daily load (kW)": Daily_final_loads,
        "Daily prod (kW)": Daily_prod,
        "Daily importation (kW)": [np.mean(importation[idx]) for idx in indexes_hour],
        "Daily exportation (kW)": [np.mean(exportation[idx]) for idx in indexes_hour],
        "Daily storage (kW)": [np.mean(power_storage[idx]) for idx in indexes_hour],})
    
    yearly_time_balancing = pd.DataFrame({
        "Datetime": datetime[indexes_hour[0]].strftime('%Y-%m-%d'),
        "Yearly load (kW)": Yearly_final_loads,
        "Yearly prod (kW)": Yearly_prod,
        "Yearly importation (kW)": [np.mean(importation[indexes_days[j]]) for j in range(Context.time.n_days)],
        "Yearly exportation (kW)": [-np.mean(exportation[indexes_days[j]]) for j in range(Context.time.n_days)],
        "Yearly storage (kW)": [np.mean(power_storage[indexes_days[j]]) for j in range(Context.time.n_days)],})   

    # =========================
    # Demand-side management strategies
    # =========================

    DSM_daily_strategy = pd.DataFrame(data={'Datetime':datetime[0:int(Context.time.time_resolution*24)].strftime('%H:%M'),'Daily base load (kW)':Daily_base_load,'Daily movable load (kW)':Daily_movable_loads,'Daily final load (kW)':Daily_final_loads,'Daily prod (kW)':Daily_prod})
    DSM_yearly_strategy = pd.DataFrame(data={'Datetime':datetime[indexes_hour[0]].strftime('%Y-%m-%d'),'Yearly base load (kW)':Yearly_base_load,'Yearly movable load (kW)':Yearly_movable_loads,'Yearly final load (kW)':Yearly_final_loads,'Yearly prod (kW)':Yearly_prod})
    Load_strategy = pd.DataFrame(data={'Datetime':datetime,'Non controllable load (kW)':Context.loads.non_movable,'Daily movable load (kW)':Context.loads.D_movable,'Yearly movable load (kW)':Context.loads.Y_movable, 'Daily optimized load (kW)':D_DSM.flatten(),'Yearly optimized load (kW)':Y_DSM})
 
    # =========================
    # OVERVIEW
    # =========================   
    
    logicals_sells = np.where((trades<=0) & (signal<=0))[0]
    logicals_buys = np.where((trades>=0) & (signal>=0))[0]
    illogicals_buys = np.where((trades>0) & (signal<=0))[0]
    illogicals_sells = np.where((trades<0) & (signal>=0))[0]
    use_logical_sells = (sum(Optimized_Load[logicals_sells]),-sum(tuple(sum(np.where(storage_TS[i][logicals_sells]<0,storage_TS[i][logicals_sells],0) for i in range(n_store)))),-sum(trades[logicals_sells]))/Context.time.time_resolution/Context.time.duration_years-sum(tuple(sum(np.where(storage_TS[i][logicals_sells]>0,storage_TS[i][logicals_sells],0) for i in range(n_store))))
    use_logical_buys = (sum(production[logicals_buys]),sum(tuple(sum(np.where(storage_TS[i][logicals_buys]>0,storage_TS[i][logicals_buys],0) for i in range(n_store)))),sum(trades[logicals_buys]))/Context.time.time_resolution/Context.time.duration_years
    use_illogical_sells = (sum(production[illogicals_sells]),sum(tuple(sum(np.where(storage_TS[i][illogicals_sells]>0,storage_TS[i][illogicals_sells],0) for i in range(n_store))))-sum(trades[illogicals_sells]),-sum(trades[illogicals_sells]))/Context.time.time_resolution/Context.time.duration_years
    use_illogical_buys = (sum(Optimized_Load[illogicals_buys]),sum(production[illogicals_buys]-sum(Optimized_Load[illogicals_buys])))/Context.time.time_resolution/Context.time.duration_years
    
    when_prod_exceeds = pd.DataFrame(data=((sum(Optimized_Load[logicals_sells])+sum(Optimized_Load[illogicals_buys]))/Context.time.time_resolution/Context.time.duration_years,-sum(trades[logicals_sells])/Context.time.time_resolution,sum(trades[illogicals_buys])/Context.time.time_resolution) + tuple(-sum(storage_TS[i][(signal<0) & (storage_TS[i]<0)])/Context.time.time_resolution for i in range(n_store)) + tuple(sum(storage_TS[i][(signal<0) & (storage_TS[i]>0)])/Context.time.time_resolution for i in range(n_store)),index=['Load (kWh)','Grid export (kWh)','Grid import (kWh)']+['Storage '+Context.storage.technologies[i]+' charge (kWh)' for i in range(n_store)]+['Storage '+Context.storage.technologies[i]+' discharge (kWh)' for i in range(n_store)]).transpose()
    when_load_exceeds = pd.DataFrame(data=(sum(production[logicals_buys])/Context.time.time_resolution+sum(production[illogicals_sells])/Context.time.time_resolution,-sum(trades[(signal>0) & (trades<0) ])/Context.time.time_resolution,sum(trades[(signal>0) & (trades>0) ])/Context.time.time_resolution) + tuple(-sum(storage_TS[i][(signal>0) & (storage_TS[i]<0)])/Context.time.time_resolution for i in range(n_store)) + tuple(sum(storage_TS[i][(signal>0) & (storage_TS[i]>0)])/Context.time.time_resolution for i in range(n_store)),index=['Production (kWh)','Grid export (kWh)','Grid import (kWh)']+['Storage '+Context.storage.technologies[i]+' charge (kWh)' for i in range(n_store)]+['Storage '+Context.storage.technologies[i]+' discharge (kWh)' for i in range(n_store)]).transpose()
     
    useprod = pd.DataFrame(data=(use_logical_sells[1]+use_illogical_buys[1],use_logical_buys[0]+use_logical_sells[0]+use_illogical_sells[0]+use_illogical_buys[0],use_illogical_sells[2]+use_logical_sells[2] ,Annual_REN_production),index=['Storage (kWh)' ,'Load (kWh)','Exportation (kWh)','Annual REN production (kWh)']).transpose()
    Loadmeet = pd.DataFrame(data=(use_logical_buys[1]+use_illogical_sells[1] ,use_logical_buys[0]+use_logical_sells[0]+use_illogical_sells[0]+use_illogical_buys[0],use_logical_buys[2] ,Annual_load),index=['Storage (kWh)', 'Production (kWh)','Importation (kWh)','Annual load (kWh)']).transpose()
    Grid_use_export = pd.DataFrame(data=(useprod['Exportation (kWh)'], Annual_sum_exportation-useprod['Exportation (kWh)']),index=['Production (kWh)','Storage (kWh)']).transpose()
    Grid_use_import = pd.DataFrame(data=(Loadmeet['Importation (kWh)'], Annual_sum_importation-Loadmeet['Importation (kWh)']),index=['Load (kWh)','Storage (kWh)']).transpose()    

    # =========================
    # FITNESS & TECHNICAL
    # =========================   
    
    installation_lifetime = np.nanmin(np.array([DG_lifetime_years,min(Storage_lifetime),np.nanmin(np.where(solution.production_set>0,Context.production.specs_num[:,PROD_LIFETIME],np.nan))]))
    fitness = solution.fitness  
    
    return{'TimeSeries':{'Storage_TS (kW)':storage_TS,'Storage cumulative energy (kWh)':sum_diff_storages,'SOCs (%)':SOCs,'Losses (kW)': losses,'D_DSM (kW)':D_DSM,'Y_DSM (kW)':Y_DSM,'trades (kW)':trades,'Optimized load (kW)':Optimized_Load,'Non movable load (kW)':Context.loads.non_movable,'production (kW)':production,'Grid trading (kW)':Grid_trading,'Grid importation (kW)':Grid_importation,'Grid exportation (kW)':Grid_exportation,'DG production (kW)':DG_production,'Curtailment (kW)':curtailment},
           'Technical':{'fitness':fitness,'Self-sufficiency':obtained_self_sufficiency,'Self-consumption':self_consumption,'Autonomy':Autonomy,'REN fraction':REN_fraction,'REN Self-sufficiency':REN_self_sufficiency,'REN Autonomy':REN_autonomy,'Annual sum losses (kWh)':annual_total_losses,'Contract power (kW)':Contract_power,'Capacity factor':Capacity_factor,'Max power from grid (kW)':Max_power_from_grid,'Max power to grid (kW)':Max_power_to_grid,'Max curtailment (kW)':Max_curtailment,'Installation lifetime (yrs.)':installation_lifetime },
           'economics':{'LCOE (€/kWh)':LCOE,'Cost production (€/kWh)':cost_production_per_kWh,'OPEX storage (€/kWh)':OPEX_storage_per_kWh,'CAPEX storage (€/kWh)':CAPEX_storage_per_kWh, 'Contract power (€/kWh)':contract_power_per_kWh,'Overrun penalty (€/kWh)':overrun_penalty_per_kWh,'Energy importation (€/kWh)':energy_importation_per_kWh,'Fixed premium (€/kWh)':fixed_premium_per_kWh,'DG fuel cost (€/kWh)':DG_fuel_cost_per_kWh,'DG CAPEX cost (€/kWh)':DG_CAPEX_cost_per_kWh,'DG OPEX cost (€/kWh)':DG_OPEX_cost_per_kWh,'Energy exportation (€/kWh)':energy_exportation_per_kWh,'Annual net benefits (€/yrs.)':Annual_net_benefits,'Value (€)':Value,'Initial investment (€)':Initial_investment}, 
           'Storages':{'Capacity (kWh)':energy_storages,'Powers in (kW)':powers_in,'Powers out (kW)':powers_out,'Min SOCs (%)':minSOCs ,'Equivalent cycles':Equivalent_cycles,'Storage lifetime (yrs.)':Storage_lifetime}, 
           'Genset':{'nominal_power (kW)':DG_nominal_power, 'Max power from DG (kW)':Max_power_from_DG, 'DG lifetime (yrs.)':DG_lifetime_years,'DG min. production':solution.DG_min_production, 'DG min. runtime':solution.DG_min_runtime },
           'Genset power distribution':closest_levels,
           'EMS' : {'D_DSM min. levels':solution.D_DSM_minimum_levels,'Y_DSM min. levels':solution.Y_DSM_minimum_levels,'discharge order':solution.discharge_order,'strategy':solution.DG_strategy, 'overlaps':solution.overlaps,'energy repartition coefficient':solution.energy_use_coefficient },
           'Environment':{'Annual fossil fuel from grid (kWh)':annual_fossil_fuel_consumption_importation,'Annual fossil fuel consumption from DG (kWh)':annual_fuel_consumption_DG,'Annual fossil fuel consumption (kWh)':annual_fossil_fuel_consumption,'Annual CO2eq total (tCO2)':annual_CO2eq_total,'expected produced REN energy (kWh)':productible_energy,'expected produced energy from DG (kWh)':productible_energy_DG,'consumed energy for DG (kWh)':consumed_energy_DG ,'Consumed energy for production (kWh)':consumed_energy_production ,'Consumed energy for storage (kWh)':consumed_energy_storage,'EROI':EROI },
           'Flows':{'Annual load (kWh)':Annual_load,'Annual REN production' : Annual_REN_production ,'Sum importation (kWh)':Annual_sum_importation,'Sum exportation (kWh)':Annual_sum_exportation},
           'Flows storages':{'Annual stored energy (kWh)':stored_energy,'Annual reported energy (kWh)':reported_energy,'Annual losses (kWh)':annual_losses},
           'Extra_outputs':{'Uses':{'useprod':useprod,'Loadmeet':Loadmeet,'when_prod':when_prod_exceeds,'when_load':when_load_exceeds},'Logics':{'logicals_sells':logicals_sells,'logicals_buys':logicals_buys,'illogicals_sells':illogicals_sells,'illogicals_buys':illogicals_buys},'Grid usage':{'export':Grid_use_export,'import':Grid_use_import},'distribution_DOD': dist_DOD },
           'Balancing':{'daily time balancing':daily_time_balancing,'yearly time balancing':yearly_time_balancing},
           'Demand-side management':{'DSM daily strategy':DSM_daily_strategy,'DSM yearly strategy':DSM_yearly_strategy,'Load strategy':Load_strategy}
           }



def KPI_baseline(Context,solution,datetime):
    """
    Comprehensive cost and performance evaluation of the base scenario (current microgrid design without optimization).
    
    This function computes the annualized technical, economic, environmental,
    and operational metrics of a microgrid system, including:
    
        - Net load and production balance
        - Multi-storage operation and state-of-charge (SOC) evolution
        - Grid interaction (import/export)
        - Diesel generator (DG) operation, fuel consumption, and costs
        - Demand-Side Management (DSM) strategies for daily and yearly loads
        - Energy flows, losses, and curtailment
        - Economic indicators (LCOE, CAPEX, OPEX, net benefits)
        - Environmental indicators (CO2eq emissions, fossil fuel consumption)
        - Storage performance metrics (capacity, cycles, lifetime, SOC distribution)
        - Microgrid self-sufficiency, autonomy, and renewable fraction
        - Detailed time-series outputs for load, production, storage, and trades
    
    Args:
        Context (object): Simulation context containing loads, production,
            storage, grid, genset, time, and configuration parameters.
        datetime (pandas.DatetimeIndex | np.ndarray): Time index of simulation.
    
    Returns:
        dict: Nested dictionary containing comprehensive microgrid outputs structured as:
            - TimeSeries: Storage operation, SOCs, trades, grid flows, DG production
            - Technical: Self-sufficiency, autonomy, REN fraction, capacity factor, lifetime, etc.
            - Economics: LCOE, CAPEX/OPEX, contract power, import/export economics, DG costs
            - Storages: Capacity, powers in/out, min SOCs, equivalent cycles, lifetime
            - DG: Nominal power, maximum output, lifetime, min. production/runtime
            - DG power distribution: Power levels per timestep
            - Environment: Annual CO2eq, fossil fuel consumption, EROI
            - Flows: Annual load, renewable production, import/export sums
            - Flows storages: Stored, reported, and lost energy
            - Extra_outputs: Detailed uses, logics, grid usage, SOC distribution
            - Balancing: Daily and yearly time-series balancing
            - Demand-side management: Daily and yearly DSM strategies, optimized load
    
    Note:
        The function distinguishes between logical and illogical energy trades, calculates
        energy curtailment, storage losses, and accounts for efficiency of storage and DG.
        It is suitable for both on-grid and off-grid microgrid simulations.
    
    Important:
        This is the final evaluation function used to compute ERMESS outputs.
    """
    
    annual_cost_production = 0

    # =========================
    # CONSTANTS
    # =========================    
    TONS_CONVERSION_FACTOR = 1000000
    KILOS_CONVERSION_FACTOR = 1000
    HOURS_PER_DAY = 24
    
    # =========================
    # LOAD & PRODUCTION
    # =========================
    Load = Context.loads.non_movable+Context.loads.D_movable+Context.loads.Y_movable
    production = (Context.production.current_prod)/KILOS_CONVERSION_FACTOR    
    Annual_load = sum(Load)/Context.time.time_resolution/Context.time.duration_years
    Annual_REN_production = sum(production)/Context.time.time_resolution/Context.time.duration_years
    
    # =========================
    # CORE ENERGY BALANCE
    # =========================
    trades=Load-production    
    Optimized_Load = Load
    importation=np.where(trades>0,trades,0)
    exportation = np.where(trades<0,-trades,0)
    signal =  Optimized_Load - production  

    Annual_sum_importation=sum(importation)/Context.time.time_resolution/Context.time.duration_years
    Annual_sum_exportation=sum(exportation)/Context.time.time_resolution/Context.time.duration_years    

    # =========================
    # STORAGE INITIALIZATION
    # =========================
    n_store = Context.storage.n_store
    n_bits = Context.time.n_bits
    
    storage_TS = np.zeros((n_store, n_bits))
    losses = np.zeros((n_store, n_bits))

    sum_diff_storages = np.array([-np.cumsum(storage_TS[i,:]/Context.time.time_resolution+losses[i,:]/Context.time.time_resolution) for i in range(n_store)])

    reported_energy = np.sum(np.where(storage_TS>0,storage_TS,0),axis=1)/Context.time.time_resolution/Context.time.duration_years
    stored_energy = -np.sum(np.where(storage_TS<0,storage_TS,0),axis=1)/Context.time.time_resolution/Context.time.duration_years
 
    power_storage = np.sum(storage_TS,axis=0)
    Annual_sum_losses = np.sum(losses,axis=1)/Context.time.time_resolution/Context.time.duration_years    
    
    # =========================
    # GRID FLOWS
    # =========================

    Grid_importation = importation if (Context.optimization.connexion=='On-grid') else np.zeros(n_bits)
    Grid_exportation = exportation if (Context.optimization.connexion=='On-grid') else np.zeros(n_bits)
    Grid_trading = trades if (Context.optimization.connexion=='On-grid') else np.zeros(n_bits)
    
    DG_production = importation if (Context.optimization.connexion=='Off-grid') else np.zeros(n_bits)
    curtailment = exportation if (Context.optimization.connexion=='Off-grid') else np.zeros(n_bits)
    
    # =========================
    # SELF-SUFFICIENCY / AUTONOMY
    # =========================

    obtained_self_sufficiency = (1-sum(Grid_importation)/sum(Optimized_Load)) if sum(Optimized_Load )>0 else np.nan
    REN_self_sufficiency = (1-sum(importation)/sum(Optimized_Load)) if sum(Optimized_Load )>0 else np.nan
    self_consumption = (1-sum(exportation)/sum(production)) if sum(production )>0 else np.nan
    REN_fraction = Annual_REN_production/Annual_load if Annual_load>0 else np.nan
    
    Autonomy = 1-sum(Grid_importation>0)/n_bits
    REN_autonomy = 1-sum(importation>0)/n_bits
    Capacity_factor = sum(production)/(max(production)*n_bits) if max(production)>0 else np.nan
    
    obtained_constraint_level = obtained_self_sufficiency if Context.optimization.constraint_num==CONS_Self_sufficiency else self_consumption if (Context.optimization.constraint_num==CONS_Self_consumption) else REN_fraction if(Context.optimization.constraint_num==CONS_REN_fraction) else np.nan
    
    # =========================
    # GRID PARAMETERS
    # =========================
    
    Max_power_from_grid = np.max(Grid_importation) 
    Max_power_to_grid = np.max(Grid_exportation) 
    Max_curtailment = np.max(curtailment)  
    
    if (Context.optimization.connexion=='Off-grid'):       
        Contract_power=0
    elif (Context.optimization.connexion=='On-grid'):    
        Contract_power=max(0,max(trades))
  
    # =========================
    # GENSET
    # =========================  
    if (Context.optimization.connexion=='Off-grid'):
        DG_nominal_power = max(trades)
        DG_production=importation
        if (DG_nominal_power>0):
            closest_levels = np.array([int(10*(DG_production[i]/DG_nominal_power)-0.5) for i in range(n_bits)])
            annual_fuel_consumption_DG = sum(DG_production*Context.genset.fuel_consumption[closest_levels]/Context.time.time_resolution/Context.time.duration_years)
        else : 
            closest_levels = np.array([np.nan for i in range(n_bits)])
            annual_fuel_consumption_DG = 0
        DG_lifetime_years = Context.genset.lifetime/(sum(np.where(DG_production>0,1,0))/Context.time.time_resolution/Context.time.duration_years)
    elif (Context.optimization.connexion=='On-grid'):  
        DG_nominal_power = 0
        DG_production=np.repeat(0,n_bits)
        closest_levels = 0
        DG_lifetime_years = np.nan
        annual_fuel_consumption_DG = 0        

               
    Max_power_from_DG = np.max(DG_production) 
    DG_min_production = np.nan
    DG_min_runtime = np.nan

    
    # =========================
    # ENVIRONMENT
    # =========================
    if (Context.optimization.connexion=='Off-grid'):
        annual_CO2eq_DG = annual_fuel_consumption_DG*Context.genset.fuel_CO2eq_emissions/TONS_CONVERSION_FACTOR   
    elif (Context.optimization.connexion=='On-grid'):  
        annual_CO2eq_DG = 0.
            
    annual_CO2eq_prod = 0
    annual_CO2eq_importation = sum(Grid_importation)*Context.grid.C02eqemissions/TONS_CONVERSION_FACTOR/Context.time.time_resolution/Context.time.duration_years if (Context.optimization.connexion=='On-grid') else 0
    annual_CO2eq_storage = 0
    annual_fossil_fuel_consumption_importation =  Context.grid.fossil_fuel_ratio*sum(Grid_importation)/Context.time.time_resolution/Context.time.duration_years if (Context.optimization.connexion=='On-grid') else 0
    
    annual_fossil_fuel_consumption = annual_fossil_fuel_consumption_importation+annual_fuel_consumption_DG
    annual_CO2eq_total = annual_CO2eq_DG+annual_CO2eq_prod+annual_CO2eq_importation+annual_CO2eq_storage
    
    # =========================
    # STORAGE
    # =========================  
    size_power=np.zeros(n_store)
    energy_storages = np.zeros(n_store)
    powers_out = np.zeros(n_store)
    powers_in = np.zeros(n_store)
    Equivalent_cycles = np.zeros(n_store)
    Storage_lifetime = np.zeros(n_store)
    annual_total_losses = sum(np.sum(losses,axis=1))/Context.time.time_resolution/Context.time.duration_years
     
    # =========================
    # STORAGE SOC analysis
    # ========================= 
    minSOCs=np.repeat(np.nan,n_store)
    SOCs=np.repeat(np.nan,n_store*n_bits).reshape(n_store,n_bits)
    storage_NULL=tuple(energy_storages[i]==0 for i in range(n_store))
    
    State_SOCs=[np.searchsorted([(i+1)/100 for i in range(99)], SOCs[j,:]) for j in range(n_store)]
    distribution_Depth_of_discharge=np.array([[np.count_nonzero(State_SOCs[i]==j)/len(State_SOCs[i]) for j in range(100)] for i in range(n_store)])
    if sum(storage_NULL)>0:
        distribution_Depth_of_discharge[np.where(storage_NULL)[0][0]]=np.repeat(np.nan,100)
    
    dist_DOD = pd.DataFrame(data={'SOC Percentile':[i/100 for i in range(101)]})
    for i in range(n_store):
        dist_DOD = dist_DOD.join(pd.DataFrame(data={'SOC distribution ' + Context.storage.technologies[i]:distribution_Depth_of_discharge[i]}))

        
    # =========================
    # Energy return on investment
    # =========================        
    if (Context.optimization.connexion=='Off-grid'):
        productible_energy_DG = sum(DG_production)/Context.time.time_resolution/Context.time.duration_years*DG_lifetime_years
        consumed_energy_DG = productible_energy_DG/Context.genset.EROI
    elif (Context.optimization.connexion=='On-grid'):  
        productible_energy_DG = 0.
        consumed_energy_DG = 0.
        
    productible_energy = 0
    consumed_energy_production = 0
    consumed_energy_storage = sum(np.nanmin(np.array([Context.storage.characteristics[STOR_CYCLE_LIFE,:],Context.storage.characteristics[STOR_LIFETIME,:]*Equivalent_cycles]),axis=0)*energy_storages/Context.storage.characteristics[STOR_ESOEI,:])

    EROI = (productible_energy+productible_energy_DG)/(consumed_energy_DG+consumed_energy_production+consumed_energy_storage)
             
    # =========================
    # ECONOMICS
    # =========================          
    CAPEX_storage_cost = np.zeros(n_store)
    economics_CAPEX_storage = 0
    economics_OPEX_storage = 0
    
    if (Context.optimization.connexion=='Off-grid'):
        economics_exportation = 0
        economics_importation = 0
        economics_overrun = 0
        economics_contract_power = 0
        DG_CAPEX_cost = DG_nominal_power*Context.genset.unit_cost/DG_lifetime_years
        DG_OPEX_cost = sum(DG_production)*Context.genset.maintenance_cost/Context.time.time_resolution/Context.time.duration_years
        annual_total_fuel_cost = annual_fuel_consumption_DG*Context.genset.fuel_cost

    elif (Context.optimization.connexion=='On-grid'):  
        economics_exportation = np.multiply(exportation,Context.grid.selling_price[solution.contract,:]).sum()/Context.time.time_resolution/Context.time.duration_years
        economics_importation = np.multiply(importation,Context.grid.prices[solution.contract,:]).sum()/Context.time.time_resolution/Context.time.duration_years
        economics_overrun = max(0,(max(importation)-Contract_power)*Context.grid.overrun[solution.contract])
        economics_contract_power = Context.grid.fixed_premium[solution.contract]*Contract_power
        DG_CAPEX_cost = 0
        DG_OPEX_cost = 0   
        annual_total_fuel_cost = 0
        
    LCOE=(annual_cost_production+economics_CAPEX_storage+economics_OPEX_storage+DG_OPEX_cost+DG_CAPEX_cost+annual_total_fuel_cost+economics_importation+economics_contract_power+economics_overrun-economics_exportation)/(sum(Optimized_Load)/Context.time.time_resolution/Context.time.duration_years)   if sum(Optimized_Load )>0 else np.nan
    Annual_net_benefits = (-annual_cost_production-economics_CAPEX_storage-economics_OPEX_storage-DG_OPEX_cost-DG_CAPEX_cost-annual_total_fuel_cost-economics_importation-economics_contract_power-economics_overrun+economics_exportation)
    Initial_investment = sum(CAPEX_storage_cost)
    
    # =========================
    # COST DECOMPOSITION
    # =========================              
        
    cost_production_per_kWh = annual_cost_production/(sum(Optimized_Load)/Context.time.time_resolution/Context.time.duration_years)
    OPEX_storage_per_kWh = economics_OPEX_storage/(sum(Optimized_Load)/Context.time.time_resolution/Context.time.duration_years)
    CAPEX_storage_per_kWh = economics_CAPEX_storage/(sum(Optimized_Load)/Context.time.time_resolution/Context.time.duration_years)
    contract_power_per_kWh = economics_contract_power/(sum(Optimized_Load)/Context.time.time_resolution/Context.time.duration_years)
    overrun_penalty_per_kWh = economics_overrun/(sum(Optimized_Load)/Context.time.time_resolution/Context.time.duration_years)
    energy_importation_per_kWh = economics_importation/(sum(Optimized_Load)/Context.time.time_resolution/Context.time.duration_years)
    fixed_premium_per_kWh = economics_contract_power/(sum(Optimized_Load)/Context.time.time_resolution/Context.time.duration_years)
    DG_fuel_cost_per_kWh = annual_total_fuel_cost/(sum(Optimized_Load)/Context.time.time_resolution/Context.time.duration_years)
    DG_CAPEX_cost_per_kWh = DG_CAPEX_cost/(sum(Optimized_Load)/Context.time.time_resolution/Context.time.duration_years)
    DG_OPEX_cost_per_kWh = DG_OPEX_cost/(sum(Optimized_Load)/Context.time.time_resolution/Context.time.duration_years)
    energy_exportation_per_kWh = economics_exportation/(sum(Optimized_Load)/Context.time.time_resolution/Context.time.duration_years)

    # =========================
    # Demand-side management
    # =========================
    indexes_hour = [[int(i + j * Context.time.time_resolution * HOURS_PER_DAY) for j in range(Context.time.n_days)] for i in range(int(Context.time.time_resolution * HOURS_PER_DAY))]
    indexes_days = [[int(i+j*Context.time.time_resolution*24) for i in range(int(Context.time.time_resolution*24))] for j in range(Context.time.n_days)]

    Daily_base_load = [np.mean(Context.loads.non_movable[idx]) for idx in indexes_hour]
    Daily_movable_loads = [np.mean(Context.loads.D_movable[idx] + Context.loads.Y_movable[idx]) for idx in indexes_hour]
    Daily_final_loads = [np.mean(Optimized_Load[idx]) for idx in indexes_hour]
    Daily_prod = [np.mean(production[idx]) for idx in indexes_hour]
    
    # =========================
    # TIME BALANCING
    # =========================
    
    Yearly_base_load = [np.mean(Context.loads.non_movable[indexes_days[j]]+Context.loads.D_movable[indexes_days[j]]) for j in range(Context.time.n_days)]
    Yearly_movable_loads = [np.mean(Context.loads.Y_movable[indexes_days[j]]) for j in range(Context.time.n_days)]
    Yearly_final_loads = [np.mean(Optimized_Load[indexes_days[j]]) for j in range(Context.time.n_days)]
    Yearly_prod =  [np.mean(production[indexes_days[j]]) for j in range(Context.time.n_days)]
        
    daily_time_balancing = pd.DataFrame({
        "Datetime": datetime[:int(Context.time.time_resolution * HOURS_PER_DAY)].strftime("%H:%M"),
        "Daily load (kW)": Daily_final_loads,
        "Daily prod (kW)": Daily_prod,
        "Daily importation (kW)": [np.mean(importation[idx]) for idx in indexes_hour],
        "Daily exportation (kW)": [np.mean(exportation[idx]) for idx in indexes_hour],
        "Daily storage (kW)": [np.mean(power_storage[idx]) for idx in indexes_hour],})
    
    yearly_time_balancing = pd.DataFrame({
        "Datetime": datetime[indexes_hour[0]].strftime('%Y-%m-%d'),
        "Yearly load (kW)": Yearly_final_loads,
        "Yearly prod (kW)": Yearly_prod,
        "Yearly importation (kW)": [np.mean(importation[indexes_days[j]]) for j in range(Context.time.n_days)],
        "Yearly exportation (kW)": [-np.mean(exportation[indexes_days[j]]) for j in range(Context.time.n_days)],
        "Yearly storage (kW)": [np.mean(power_storage[indexes_days[j]]) for j in range(Context.time.n_days)],})
    
    # =========================
    # Demand-side management strategies
    # =========================
    
        
    D_DSM=np.zeros((int(n_bits/(HOURS_PER_DAY*Context.time.time_resolution)),HOURS_PER_DAY*int(Context.time.time_resolution)))
    Y_DSM=np.zeros(n_bits)

    DSM_daily_strategy = pd.DataFrame(data={'Datetime':datetime[0:int(Context.time.time_resolution*24)].strftime('%H:%M'),'Daily base load (kW)':Daily_base_load,'Daily movable load (kW)':Daily_movable_loads,'Daily final load (kW)':Daily_final_loads,'Daily prod (kW)':Daily_prod})
    DSM_yearly_strategy = pd.DataFrame(data={'Datetime':datetime[indexes_hour[0]].strftime('%Y-%m-%d'),'Yearly base load (kW)':Yearly_base_load,'Yearly movable load (kW)':Yearly_movable_loads,'Yearly final load (kW)':Yearly_final_loads,'Yearly prod (kW)':Yearly_prod})
    Load_strategy = pd.DataFrame(data={'Datetime':datetime,'Non controllable load (kW)':Context.loads.non_movable,'Daily movable load (kW)':Context.loads.D_movable,'Yearly movable load (kW)':Context.loads.Y_movable, 'Daily optimized load (kW)':D_DSM.flatten(),'Yearly optimized load (kW)':Y_DSM})
    
    # =========================
    # OVERVIEW
    # =========================   

    logicals_sells = np.where((trades<=0) & (signal<=0))[0]
    logicals_buys = np.where((trades>=0) & (signal>=0))[0]
    illogicals_buys = np.where((trades>0) & (signal<=0))[0]
    illogicals_sells = np.where((trades<0) & (signal>=0))[0]
    use_logical_sells = (sum(Optimized_Load[logicals_sells]),-sum(tuple(sum(np.where(storage_TS[i][logicals_sells]<0,storage_TS[i][logicals_sells],0) for i in range(n_store)))),-sum(trades[logicals_sells]))/Context.time.time_resolution/Context.time.duration_years-sum(tuple(sum(np.where(storage_TS[i][logicals_sells]>0,storage_TS[i][logicals_sells],0) for i in range(n_store))))
    use_logical_buys = (sum(production[logicals_buys]),sum(tuple(sum(np.where(storage_TS[i][logicals_buys]>0,storage_TS[i][logicals_buys],0) for i in range(n_store)))),sum(trades[logicals_buys]))/Context.time.time_resolution/Context.time.duration_years
    use_illogical_sells = (sum(production[illogicals_sells]),sum(tuple(sum(np.where(storage_TS[i][illogicals_sells]>0,storage_TS[i][illogicals_sells],0) for i in range(n_store))))-sum(trades[illogicals_sells]),-sum(trades[illogicals_sells]))/Context.time.time_resolution/Context.time.duration_years
    use_illogical_buys = (sum(Optimized_Load[illogicals_buys]),sum(production[illogicals_buys]-sum(Optimized_Load[illogicals_buys])))/Context.time.time_resolution/Context.time.duration_years
    
    when_prod_exceeds = pd.DataFrame(data=((sum(Optimized_Load[logicals_sells])+sum(Optimized_Load[illogicals_buys]))/Context.time.time_resolution/Context.time.duration_years,-sum(trades[logicals_sells])/Context.time.time_resolution,sum(trades[illogicals_buys])/Context.time.time_resolution) + tuple(-sum(storage_TS[i][(signal<0) & (storage_TS[i]<0)])/Context.time.time_resolution for i in range(n_store)) + tuple(sum(storage_TS[i][(signal<0) & (storage_TS[i]>0)])/Context.time.time_resolution for i in range(n_store)),index=['Load (kWh)','Exportation (kWh)','Importation (kWh)']+['Storage '+Context.storage.technologies[i]+' charge (kWh)' for i in range(n_store)]+['Storage '+Context.storage.technologies[i]+' discharge (kWh)' for i in range(n_store)]).transpose()
    when_load_exceeds = pd.DataFrame(data=(sum(production[logicals_buys])/Context.time.time_resolution+sum(production[illogicals_sells])/Context.time.time_resolution,-sum(trades[(signal>0) & (trades<0) ])/Context.time.time_resolution,sum(trades[(signal>0) & (trades>0) ])/Context.time.time_resolution) + tuple(-sum(storage_TS[i][(signal>0) & (storage_TS[i]<0)])/Context.time.time_resolution for i in range(n_store)) + tuple(sum(storage_TS[i][(signal>0) & (storage_TS[i]>0)])/Context.time.time_resolution for i in range(n_store)),index=['Production (kWh)','Exportation (kWh)','Importation (kWh)']+['Storage '+Context.storage.technologies[i]+' charge (kWh)' for i in range(n_store)]+['Storage '+Context.storage.technologies[i]+' discharge (kWh)' for i in range(n_store)]).transpose()
     
    useprod = pd.DataFrame(data=(use_logical_sells[1]+use_illogical_buys[1],use_logical_buys[0]+use_logical_sells[0]+use_illogical_sells[0]+use_illogical_buys[0],use_illogical_sells[2]+use_logical_sells[2] ,Annual_REN_production),index=['Storage (kWh)' ,'Load (kWh)','Exportation (kWh)','Annual REN production (kWh)']).transpose()
    Loadmeet = pd.DataFrame(data=(use_logical_buys[1]+use_illogical_sells[1] ,use_logical_buys[0]+use_logical_sells[0]+use_illogical_sells[0]+use_illogical_buys[0],use_logical_buys[2] ,Annual_load),index=['Storage (kWh)', 'Production (kWh)','Importation (kWh)','Annual load (kWh)']).transpose()    
    Grid_use_export = pd.DataFrame(data=(useprod['Exportation (kWh)'], Annual_sum_exportation-useprod['Exportation (kWh)']),index=['Production (kWh)','Storage (kWh)']).transpose()
    Grid_use_import = pd.DataFrame(data=(Loadmeet['Importation (kWh)'], Annual_sum_importation-Loadmeet['Importation (kWh)']),index=['Load (kWh)','Storage (kWh)']).transpose()
    
    # =========================
    # FITNESS & TECHNICAL
    # =========================   

    installation_lifetime = np.nan
    fitness_baseline = np.nan

    return{'TimeSeries':{'Storage_TS (kW)':storage_TS,'Storage cumulative energy (kWh)':sum_diff_storages,'SOCs (%)':SOCs,'Losses (kW)': losses,'D_DSM (kW)':D_DSM,'Y_DSM (kW)':Y_DSM,'trades (kW)':trades,'Optimized load (kW)':Optimized_Load,'Non movable load (kW)':Context.loads.non_movable,'production (kW)':production,'Grid trading (kW)':Grid_trading,'Grid importation (kW)':Grid_importation,'Grid exportation (kW)':Grid_exportation,'DG production (kW)':DG_production,'Curtailment (kW)':curtailment},
           'Technical':{'fitness':fitness_baseline,'Self-sufficiency':obtained_self_sufficiency,'Self-consumption':self_consumption,'Autonomy':Autonomy,'REN fraction':REN_fraction,'REN Self-sufficiency':REN_self_sufficiency,'REN Autonomy':REN_autonomy,'Annual sum losses (kWh)':annual_total_losses,'Contract power (kW)':Contract_power,'Capacity factor':Capacity_factor,'Max power from grid (kW)':Max_power_from_grid,'Max power to grid (kW)':Max_power_to_grid,'Max curtailment (kW)':Max_curtailment ,'Installation lifetime (yrs.)':installation_lifetime},
           'economics':{'LCOE (€/kWh)':LCOE,'Cost production (€/kWh)':cost_production_per_kWh ,'OPEX storage (€/kWh)':OPEX_storage_per_kWh,'CAPEX storage (€/kWh)':CAPEX_storage_per_kWh, 'Contract power (€/kWh)':contract_power_per_kWh,'Overrun penalty (€/kWh)':overrun_penalty_per_kWh,'Energy importation (€/kWh)':energy_importation_per_kWh,'Fixed premium (€/kWh)':fixed_premium_per_kWh,'DG fuel cost (€/kWh)':DG_fuel_cost_per_kWh,'DG CAPEX cost (€/kWh)':DG_CAPEX_cost_per_kWh,'DG OPEX cost (€/kWh)':DG_OPEX_cost_per_kWh,'Energy exportation (€/kWh)':energy_exportation_per_kWh,'Annual net benefits (€/yrs.)':Annual_net_benefits,'Initial investment (€)':Initial_investment }, 
           'Storages':{'Capacity (kWh)':energy_storages,'Powers in (kW)':powers_in,'Powers out (kW)':powers_out,'Min SOCs (%)':minSOCs ,'Equivalent cycles':Equivalent_cycles,'Storage lifetime (yrs.)':Storage_lifetime}, 
           'Genset':{'nominal_power (kW)':DG_nominal_power, 'Max power from DG (kW)':Max_power_from_DG, 'DG lifetime (yrs.)':DG_lifetime_years,'DG min. production':DG_min_production, 'DG min. runtime':DG_min_runtime},
           'Genset power distribution':closest_levels,
           'Environment':{'Annual fossil fuel from grid (kWh)':annual_fossil_fuel_consumption_importation,'Annual fossil fuel consumption from DG (kWh)':annual_fuel_consumption_DG,'Annual fossil fuel consumption (kWh)':annual_fossil_fuel_consumption,'Annual CO2eq total (tCO2)':annual_CO2eq_total,'expected produced REN energy (kWh)':productible_energy,'expected produced energy from DG (kWh)':productible_energy_DG,'consumed energy for DG (kWh)':consumed_energy_DG ,'Consumed energy for production (kWh)':consumed_energy_production ,'Consumed energy for storage (kWh)':consumed_energy_storage,'EROI':EROI } ,
           'Flows':{'Annual load (kWh)':Annual_load,'Annual REN production' : Annual_REN_production ,'Sum importation (kWh)':Annual_sum_importation,'Sum exportation (kWh)':Annual_sum_exportation},
           'Flows storages':{'Annual stored energy (kWh)':stored_energy,'Annual reported energy (kWh)':reported_energy,'Annual losses (kWh)':Annual_sum_losses},
           'Extra_outputs':{'Uses':{'useprod':useprod,'Loadmeet':Loadmeet,'when_prod':when_prod_exceeds,'when_load':when_load_exceeds},'Logics':{'logicals_sells':logicals_sells,'logicals_buys':logicals_buys,'illogicals_sells':illogicals_sells,'illogicals_buys':illogicals_buys},'Grid usage':{'export':Grid_use_export,'import':Grid_use_import},'distribution_DOD': dist_DOD },
           'Balancing':{'daily time balancing':daily_time_balancing,'yearly time balancing':yearly_time_balancing},
           'Demand-side management':{'DSM daily strategy':DSM_daily_strategy,'DSM yearly strategy':DSM_yearly_strategy,'Load strategy':Load_strategy}
}