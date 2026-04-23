# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 13:11:54 2024

@author: JoPHOBEA
"""
import pandas as pd
import numpy as np
from numba import jit
from numba.types import ListType
from numba.experimental import jitclass
from numba import float64, int64, types

from ERMESS_scripts.energy_model import ERMESS_EMS_models as Eems
from ERMESS_scripts.data.indices import *

pro_params = [
    ('total_D_Movable_load', float64[:]),
    ('total_Y_Movable_load', float64),
    ]

@jitclass(pro_params)
class ProParams:
    def __init__(self, total_D_Movable_load,total_Y_Movable_load):

        self.total_D_Movable_load = total_D_Movable_load
        self.total_Y_Movable_load = total_Y_Movable_load


global_params = [ 
    ('n_bits', int64),
    ('time_resolution', float64),
    ('duration_years', float64),
    ('constraint_num', int64),
    ('constraint_level', float64),
    ('cost_constraint', float64),
    ('Connexion', types.string),
    ('Non_movable_load', float64[:]),
]

@jitclass(global_params)
class GlobalParams:
    def __init__(self, n_bits,time_resolution,duration_years,constraint_num,constraint_level,cost_constraint,Connexion,Non_movable_load):

        self.n_bits = n_bits
        self.time_resolution = time_resolution
        self.duration_years = duration_years
        self.constraint_num = constraint_num
        self.constraint_level = constraint_level
        self.cost_constraint = cost_constraint
        self.Connexion = Connexion
        self.Non_movable_load = Non_movable_load

specs_grid = [
    ('prices', float64[:,:]),
    ('fixed_premium', float64[:]),
    ('Overrun', float64[:]),
    ('Selling_price', float64[:,:]),
    ('eqCO2emissions', float64),
    ('fossil_fuel_ratio', float64),
]

@jitclass(specs_grid)
class GridParams:
    def __init__(self, prices, fixed_premium, Overrun, Selling_price, eqCO2emissions, fossil_fuel_ratio):

        self.prices = prices
        self.fixed_premium = fixed_premium
        self.Overrun = Overrun
        self.Selling_price = Selling_price
        self.eqCO2emissions = eqCO2emissions
        self.fossil_fuel_ratio = fossil_fuel_ratio
        
@jitclass(specs_grid)
class GridParamsDummy:
    def __init__(self):
        self.prices = np.zeros((1, 1))
        self.fixed_premium = np.zeros(1)
        self.Overrun = np.zeros(1)
        self.Selling_price = np.zeros((1, 1))
        self.eqCO2emissions = np.float64(0)
        self.fossil_fuel_ratio = np.float64(0)
        
REN_specs_systems = [
    ('current_production', float64[:]),
    ('unit_productions', float64[:,:]),
    ('groups_production', ListType(int64[:]) ),
    ('specs_prod', float64[:,:]),
    ('capacities', int64[:]),
    ('specs_storage', float64[:,:]),
    ('n_store', int64),
]

@jitclass(REN_specs_systems)
class RENSystemsParams:
    def __init__(self, current_production, unit_productions,groups_production, specs_prod, capacities, specs_storage, n_store):

        self.current_production = current_production
        self.unit_productions = unit_productions
        self.groups_production = groups_production
        self.specs_prod = specs_prod
        self.capacities = capacities
        self.specs_storage = specs_storage
        self.n_store = n_store

spec_genset = [
    ('fuel_cost', float64),
    ('lifetime', float64),
    ('unit_cost', float64),
    ('maintenance_cost', float64),
    ('fuel_consumption', float64[:]),
    ('fuel_CO2eq_emissions', float64),
    ('EROI', float64),
]

@jitclass(spec_genset)
class GensetParams:
    def __init__(self, fuel_cost, lifetime, unit_cost,
                 maintenance_cost, fuel_consumption, fuel_CO2eq_emissions, EROI):

        self.fuel_cost = fuel_cost
        self.lifetime = lifetime
        self.unit_cost = unit_cost
        self.maintenance_cost = maintenance_cost
        self.fuel_consumption = fuel_consumption
        self.fuel_CO2eq_emissions = fuel_CO2eq_emissions
        self.EROI = EROI
        
@jitclass(spec_genset)
class GensetParamsDummy:
    def __init__(self):
        self.fuel_cost = np.float64(0)
        self.lifetime = np.float64(0)
        self.unit_cost = np.float64(0)
        self.maintenance_cost = np.float64(0)
        self.fuel_consumption = np.zeros(1)
        self.fuel_CO2eq_emissions = np.float64(0)
        self.EROI = np.float64(0)

def cost_baseline(Contexte,datetime):
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
        gene (structured object compatible with Numba): Microgrid operational strategy, design (production and storage), and PMS parameters.
        datetime (pandas.DatetimeIndex or numpy.ndarray): Time series for the simulation period.
        storage_characteristics (numpy.ndarray): Array of storage parameters including capacities, power ratings, efficiencies, and lifetime metrics.
        time_resolution (float): Time resolution.
        n_store (int): Number of storage units in the microgrid.
        duration_years (float): Duration of the simulation in years.
        Non_movable_load (numpy.ndarray): Time series of non-flexible (base) load.
        D_movable_load (numpy.ndarray): Time series of daily-flexible load.
        Y_movable_load (numpy.ndarray): Time series of yearly-flexible load.
        prod_C (numpy.ndarray): Current on-site renewable energy production time series (kW).
        prods_U, Bounds_prod, constraint_num, constraint_level, cost_constraint (various): Additional production constraints and parameters.
        n_bits (int): Number of timesteps in the simulation.
        Connexion (str): Grid connection mode ('On-grid' or 'Off-grid').
        DG_fuel_consumption (np.ndarray): Diesel generator fuel consumption per output level.
        DG_fuel_cost (float): Diesel fuel cost.
        DG_unit_cost (float): Capital cost per kW of diesel generator.
        DG_lifetime (float): Diesel generator lifetime in years.
        DG_maintenance_cost (float): Annual O&M cost of diesel generator.
        DG_EROI (float): Energy return on investment of diesel generator fuel.
        fuel_CO2eq_emissions (float): CO2eq emissions per kWh of diesel fuel.
        storage_techs (list of str): List of storage technology names.
        n_days (int): Number of days in the simulation period.
    
    Returns:
        dict: Dictionary containing comprehensive microgrid outputs structured as:
            - TimeSeries: Storage operation, SOCs, trades, grid flows, DG production
            - Technical: Self-sufficiency, autonomy, EnR fraction, capacity factor, lifetime, etc.
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

    from ERMESS_scripts.evolutionnary_core import ERMESS_functions as Ef
    
    TONS_CONVERSION_FACTOR = 1000000
    KILOS_CONVERSION_FACTOR = 1000
    
    Load = Contexte.loads.non_movable+Contexte.loads.D_movable+Contexte.loads.Y_movable
    production = (Contexte.production.current_prod)/KILOS_CONVERSION_FACTOR    
    Annual_REN_production = sum(production)/Contexte.time.time_resolution/Contexte.time.duration_years
    annual_cost_production = 0
    size_power=np.array([0 for i in range(Contexte.storage.n_store)])
    losses=np.repeat(0,Contexte.time.n_bits*Contexte.storage.n_store).reshape((Contexte.storage.n_store,Contexte.time.n_bits))
    Annual_sum_losses = np.sum(losses,axis=1)/Contexte.time.time_resolution/Contexte.time.duration_years      
    
    D_DSM=np.repeat(0,Contexte.time.n_bits).reshape((int(Contexte.time.n_bits/(24*Contexte.time.time_resolution)),24*int(Contexte.time.time_resolution)))
    Y_DSM=np.repeat(0,Contexte.time.n_bits)
    trades=Load-production        
    Optimized_Load = Load
    Annual_load = sum(Optimized_Load)/Contexte.time.time_resolution/Contexte.time.duration_years
    EnR_fraction = Annual_REN_production/Annual_load
    storage_TS=np.repeat(0,Contexte.time.n_bits*Contexte.storage.n_store).reshape((Contexte.storage.n_store,Contexte.time.n_bits))
    sum_diff_storages = np.array([-np.cumsum(storage_TS[i,:]/Contexte.time.time_resolution+losses[i,:]/Contexte.time.time_resolution) for i in range(Contexte.storage.n_store)])
    reported_energy = np.sum(np.where(storage_TS>0,storage_TS,0),axis=1)/Contexte.time.time_resolution/Contexte.time.duration_years
    stored_energy = -np.sum(np.where(storage_TS<0,storage_TS,0),axis=1)/Contexte.time.time_resolution/Contexte.time.duration_years
    power_storage = np.sum(storage_TS,axis=0)
   
    importation=np.where(trades>0,trades,0)
    exportation = np.where(trades<0,-trades,0)
    signal =  Optimized_Load - production      
    Annual_sum_importation=sum(importation)/Contexte.time.time_resolution/Contexte.time.duration_years
    Annual_sum_exportation=sum(exportation)/Contexte.time.time_resolution/Contexte.time.duration_years
        
    logicals_sells = np.where((trades<=0) & (signal<=0))[0]
    logicals_buys = np.where((trades>=0) & (signal>=0))[0]
    illogicals_buys = np.where((trades>0) & (signal<=0))[0]
    illogicals_sells = np.where((trades<0) & (signal>=0))[0]
    use_logical_sells = (sum(Optimized_Load[logicals_sells]),-sum(tuple(sum(np.where(storage_TS[i][logicals_sells]<0,storage_TS[i][logicals_sells],0) for i in range(Contexte.storage.n_store)))),-sum(trades[logicals_sells]))/Contexte.time.time_resolution/Contexte.time.duration_years-sum(tuple(sum(np.where(storage_TS[i][logicals_sells]>0,storage_TS[i][logicals_sells],0) for i in range(Contexte.storage.n_store))))
    use_logical_buys = (sum(production[logicals_buys]),sum(tuple(sum(np.where(storage_TS[i][logicals_buys]>0,storage_TS[i][logicals_buys],0) for i in range(Contexte.storage.n_store)))),sum(trades[logicals_buys]))/Contexte.time.time_resolution/Contexte.time.duration_years
    use_illogical_sells = (sum(production[illogicals_sells]),sum(tuple(sum(np.where(storage_TS[i][illogicals_sells]>0,storage_TS[i][illogicals_sells],0) for i in range(Contexte.storage.n_store))))-sum(trades[illogicals_sells]),-sum(trades[illogicals_sells]))/Contexte.time.time_resolution/Contexte.time.duration_years
    use_illogical_buys = (sum(Optimized_Load[illogicals_buys]),sum(production[illogicals_buys]-sum(Optimized_Load[illogicals_buys])))/Contexte.time.time_resolution/Contexte.time.duration_years
    
    when_prod_exceeds = pd.DataFrame(data=((sum(Optimized_Load[logicals_sells])+sum(Optimized_Load[illogicals_buys]))/Contexte.time.time_resolution/Contexte.time.duration_years,-sum(trades[logicals_sells])/Contexte.time.time_resolution,sum(trades[illogicals_buys])/Contexte.time.time_resolution) + tuple(-sum(storage_TS[i][(signal<0) & (storage_TS[i]<0)])/Contexte.time.time_resolution for i in range(Contexte.storage.n_store)) + tuple(sum(storage_TS[i][(signal<0) & (storage_TS[i]>0)])/Contexte.time.time_resolution for i in range(Contexte.storage.n_store)),index=['Load (kWh)','Exportation (kWh)','Importation (kWh)']+['Storage '+Contexte.storage.technologies[i]+' charge (kWh)' for i in range(Contexte.storage.n_store)]+['Storage '+Contexte.storage.technologies[i]+' discharge (kWh)' for i in range(Contexte.storage.n_store)]).transpose()
    when_load_exceeds = pd.DataFrame(data=(sum(production[logicals_buys])/Contexte.time.time_resolution+sum(production[illogicals_sells])/Contexte.time.time_resolution,-sum(trades[(signal>0) & (trades<0) ])/Contexte.time.time_resolution,sum(trades[(signal>0) & (trades>0) ])/Contexte.time.time_resolution) + tuple(-sum(storage_TS[i][(signal>0) & (storage_TS[i]<0)])/Contexte.time.time_resolution for i in range(Contexte.storage.n_store)) + tuple(sum(storage_TS[i][(signal>0) & (storage_TS[i]>0)])/Contexte.time.time_resolution for i in range(Contexte.storage.n_store)),index=['Production (kWh)','Exportation (kWh)','Importation (kWh)']+['Storage '+Contexte.storage.technologies[i]+' charge (kWh)' for i in range(Contexte.storage.n_store)]+['Storage '+Contexte.storage.technologies[i]+' discharge (kWh)' for i in range(Contexte.storage.n_store)]).transpose()
     
    useprod = pd.DataFrame(data=(use_logical_sells[1]+use_illogical_buys[1],use_logical_buys[0]+use_logical_sells[0]+use_illogical_sells[0]+use_illogical_buys[0],use_illogical_sells[2]+use_logical_sells[2] ,Annual_REN_production),index=['Storage (kWh)' ,'Load (kWh)','Exportation (kWh)','Annual REN production (kWh)']).transpose()
    Loadmeet = pd.DataFrame(data=(use_logical_buys[1]+use_illogical_sells[1] ,use_logical_buys[0]+use_logical_sells[0]+use_illogical_sells[0]+use_illogical_buys[0],use_logical_buys[2] ,Annual_load),index=['Storage (kWh)', 'Production (kWh)','Importation (kWh)','Annual load (kWh)']).transpose()    
    Grid_use_export = pd.DataFrame(data=(useprod['Exportation (kWh)'], Annual_sum_exportation-useprod['Exportation (kWh)']),index=['Production (kWh)','Storage (kWh)']).transpose()
    Grid_use_import = pd.DataFrame(data=(Loadmeet['Importation (kWh)'], Annual_sum_importation-Loadmeet['Importation (kWh)']),index=['Load (kWh)','Storage (kWh)']).transpose()

    energy_storages = np.repeat(0,Contexte.storage.n_store)
    powers_out = np.repeat(0,Contexte.storage.n_store)
    powers_in = np.repeat(0,Contexte.storage.n_store)
    size_power=np.repeat(0,Contexte.storage.n_store)
    CAPEX_storage_cost = np.repeat(0,Contexte.storage.n_store)
    Equivalent_cycles = np.repeat(0.,Contexte.storage.n_store)
    Lifetime = np.repeat(0,Contexte.storage.n_store)
    economics_CAPEX_storage = 0
    economics_OPEX_storage = 0
    
    minSOCs=np.repeat(np.nan,Contexte.storage.n_store)
    SOCs=np.repeat(np.nan,Contexte.storage.n_store*Contexte.time.n_bits).reshape(Contexte.storage.n_store,Contexte.time.n_bits)
    storage_NULL=tuple(energy_storages[i]==0 for i in range(Contexte.storage.n_store))
    
    State_SOCs=[np.searchsorted([(i+1)/100 for i in range(99)], SOCs[j,:]) for j in range(Contexte.storage.n_store)]
    distribution_Depth_of_discharge=np.array([[np.count_nonzero(State_SOCs[i]==j)/len(State_SOCs[i]) for j in range(100)] for i in range(Contexte.storage.n_store)])
    if sum(storage_NULL)>0:
        distribution_Depth_of_discharge[np.where(storage_NULL)[0][0]]=np.repeat(np.nan,100)
    
    dist_DOD = pd.DataFrame(data={'SOC Percentile':[i/100 for i in range(101)]})
    for i in range(Contexte.storage.n_store):
        dist_DOD = dist_DOD.join(pd.DataFrame(data={'SOC distribution ' + Contexte.storage.technologies[i]:distribution_Depth_of_discharge[i]}))


    Grid_trading = trades if (Contexte.config.connexion=='On-grid') else np.repeat(0,Contexte.time.n_bits)
    Grid_importation = importation if (Contexte.config.connexion=='On-grid') else np.repeat(0,Contexte.time.n_bits)
    Grid_exportation = exportation if (Contexte.config.connexion=='On-grid') else np.repeat(0,Contexte.time.n_bits)  
    DG_production = importation if (Contexte.config.connexion=='Off-grid') else np.repeat(0,Contexte.time.n_bits)
    curtailment = exportation if (Contexte.config.connexion=='Off-grid') else np.repeat(0,Contexte.time.n_bits)  
        
    annual_CO2eq_prod = 0
    annual_CO2eq_importation = sum(Grid_importation)*Contexte.grid.C02eqemissions/TONS_CONVERSION_FACTOR/Contexte.time.time_resolution/Contexte.time.duration_years if (Contexte.config.connexion=='On-grid') else 0
    annual_CO2eq_storage = 0

    annual_fossil_fuel_consumption_importation =  Contexte.grid.fossil_fuel_ratio*sum(Grid_importation)/Contexte.time.time_resolution/Contexte.time.duration_years if (Contexte.config.connexion=='On-grid') else 0
    
    productible_energy = 0
    consumed_energy_production = 0
    consumed_energy_storage = sum(np.nanmin(np.array([Contexte.storage.characteristics[STOR_CYCLE_LIFE,:],Contexte.storage.characteristics[STOR_LIFETIME,:]*Equivalent_cycles]),axis=0)*energy_storages/Contexte.storage.characteristics[STOR_ESOEI,:])

    if (Contexte.config.connexion=='Off-grid'):
        DG_nominal_power = max(trades)
        DG_production=importation
        if (DG_nominal_power>0):
            closest_levels = np.array([int(10*(DG_production[i]/DG_nominal_power)-0.5) for i in range(Contexte.time.n_bits)])
            annual_fuel_consumption = sum(DG_production*Contexte.genset.fuel_consumption[closest_levels]/Contexte.time.time_resolution/Contexte.time.duration_years)
        else : 
            closest_levels = np.array([np.nan for i in range(Contexte.time.n_bits)])
            annual_fuel_consumption = 0
        annual_total_fuel_cost = annual_fuel_consumption*Contexte.genset.fuel_cost
        DG_lifetime_years = Contexte.genset.lifetime/(sum(np.where(DG_production>0,1,0))/Contexte.time.time_resolution/Contexte.time.duration_years)
        DG_CAPEX_cost = DG_nominal_power*Contexte.genset.unit_cost/DG_lifetime_years
        DG_OPEX_cost = sum(DG_production)*Contexte.genset.maintenance_cost/Contexte.time.time_resolution/Contexte.time.duration_years
        Contract_power=0
        economics_exportation = 0
        economics_importation = 0
        economics_overrun = 0
        economics_contract_power = 0
        annual_CO2eq_DG = annual_fuel_consumption*Contexte.genset.fuel_CO2eq_emissions/TONS_CONVERSION_FACTOR   
        productible_energy_DG = sum(DG_production)/Contexte.time.time_resolution/Contexte.time.duration_years*DG_lifetime_years
        consumed_energy_DG = productible_energy_DG/Contexte.genset.EROI

    elif (Contexte.config.connexion=='On-grid'):  
        DG_nominal_power = 0
        DG_production=np.repeat(0,Contexte.time.n_bits)
        closest_levels = 0
        annual_fuel_consumption = 0
        annual_total_fuel_cost = 0
        DG_lifetime_years = np.nan
        DG_CAPEX_cost = 0
        DG_OPEX_cost = 0          
        Contract_power=max(0,max(trades))
        economics_exportation = np.multiply(exportation,Contexte.grid.selling_price[solution.contract,:]).sum()/Contexte.time.time_resolution/Contexte.time.duration_years
        economics_importation = np.multiply(importation,Contexte.grid.prices[solution.contract,:]).sum()/Contexte.time.time_resolution/Contexte.time.duration_years
        economics_overrun = max(0,(max(importation)-Contract_power)*Contexte.grid.overrun[solution.contract])
        economics_contract_power = Contexte.grid.fixed_premium[solution.contract]*Contract_power
        annual_CO2eq_DG = 0.
        productible_energy_DG = 0.
        consumed_energy_DG = 0.
        
    obtained_self_sufficiency = (1-sum(Grid_importation)/sum(Optimized_Load)) if sum(Optimized_Load )>0 else np.nan
    EnR_self_sufficiency = (1-sum(importation)/sum(Optimized_Load)) if sum(Optimized_Load )>0 else np.nan
    self_consumption = (1-sum(exportation)/sum(production)) if sum(production )>0 else np.nan
    EnR_fraction = sum(production)/Annual_load if Annual_load>0 else np.nan
        
    obtained_constraint_level = obtained_self_sufficiency if Contexte.optimization.constraint_num==CONS_Self_sufficiency else self_consumption if (Contexte.optimization.constraint_num==CONS_Self_consumption) else EnR_fraction if(Contexte.optimization.constraint_num==CONS_REN_fraction) else np.nan

    Autonomy = 1-sum(Grid_importation>0)/Contexte.time.n_bits
    EnR_autonomy = 1-sum(importation>0)/Contexte.time.n_bits
    Capacity_factor = sum(production)/(max(production)*Contexte.time.n_bits) if max(production)>0 else np.nan
    Max_power_from_grid = max(Grid_importation) 
    Max_power_from_DG = max(DG_production) 
    Max_power_to_grid = max(Grid_exportation) 
    Max_curtailment = max(curtailment)   
        
    #Demand-side management
    indexes_hour = [[int((i+j*Contexte.time.time_resolution*24)) for j in range(Contexte.time.n_days)] for i in range(int(Contexte.time.time_resolution*24))]
    Daily_base_load = [np.mean(Contexte.loads.non_movable[indexes_hour[j]]) for j in range(int(Contexte.time.time_resolution*24))]
    Daily_movable_loads = [np.mean(Contexte.loads.D_movable[indexes_hour[j]]+Contexte.loads.Y_movable[indexes_hour[j]]) for j in range(int(Contexte.time.time_resolution*24))]
    Daily_final_loads = [np.mean(Optimized_Load[indexes_hour[j]]) for j in range(int(Contexte.time.time_resolution*24))]
    Daily_prod = [np.mean(production[indexes_hour[j]]) for j in range(int(Contexte.time.time_resolution*24))]
    
    indexes_days = [[int(i+j*Contexte.time.time_resolution*24) for i in range(int(Contexte.time.time_resolution*24))] for j in range(Contexte.time.n_days)]
    Yearly_base_load = [np.mean(Contexte.loads.non_movable[indexes_days[j]]+Contexte.loads.D_movable[indexes_days[j]]) for j in range(Contexte.time.n_days)]
    Yearly_movable_loads = [np.mean(Contexte.loads.Y_movable[indexes_days[j]]) for j in range(Contexte.time.n_days)]
    Yearly_final_loads = [np.mean(Optimized_Load[indexes_days[j]]) for j in range(Contexte.time.n_days)]
    Yearly_prod = [np.mean(production[indexes_days[j]]) for j in range(Contexte.time.n_days)]
    
    DSM_daily_strategy = pd.DataFrame(data={'Datetime':datetime[0:int(Contexte.time.time_resolution*24)].strftime('%H:%M'),'Daily base load (kW)':Daily_base_load,'Daily movable load (kW)':Daily_movable_loads,'Daily final load (kW)':Daily_final_loads,'Daily prod (kW)':Daily_prod})
    DSM_yearly_strategy = pd.DataFrame(data={'Datetime':datetime[indexes_hour[0]].strftime('%Y-%m-%d'),'Yearly base load (kW)':Yearly_base_load,'Yearly movable load (kW)':Yearly_movable_loads,'Yearly final load (kW)':Yearly_final_loads,'Yearly prod (kW)':Yearly_prod})
    Load_strategy = pd.DataFrame(data={'Datetime':datetime,'Non controllable load (kW)':Contexte.loads.non_movable,'Daily movable load (kW)':Contexte.loads.D_movable,'Yearly movable load (kW)':Contexte.loads.Y_movable, 'Daily optimized load (kW)':D_DSM.flatten(),'Yearly optimized load (kW)':Y_DSM})
  
    daily_prod = [np.mean(production[indexes_hour[j]]) for j in range(int(Contexte.time.time_resolution*24))]
    yearly_prod = [np.mean(production[indexes_days[j]]) for j in range(Contexte.time.n_days)]
    daily_importation = [np.mean(importation[indexes_hour[j]]) for j in range(int(Contexte.time.time_resolution*24))]
    yearly_importation = [np.mean(importation[indexes_days[j]]) for j in range(Contexte.time.n_days)]
    daily_exportation = [-np.mean(exportation[indexes_hour[j]]) for j in range(int(Contexte.time.time_resolution*24))]
    yearly_exportation = [-np.mean(exportation[indexes_days[j]]) for j in range(Contexte.time.n_days)]
    daily_storage = [np.mean(power_storage[indexes_hour[j]]) for j in range(int(Contexte.time.time_resolution*24))]
    yearly_storage = [np.mean(power_storage[indexes_days[j]]) for j in range(Contexte.time.n_days)]   
    daily_time_balancing = pd.DataFrame(data={'Datetime':datetime[0:int(Contexte.time.time_resolution*24)].strftime('%H:%M'),'Daily load (kW)':Daily_final_loads,'Daily prod (kW)':daily_prod,'Daily importation (kW)':daily_importation,'Daily exportation (kW)':daily_exportation,'Daily storage (kW)':daily_storage})
    yearly_time_balancing = pd.DataFrame(data={'Datetime':datetime[indexes_hour[0]].strftime('%Y-%m-%d'),'Yearly load (kW)':Yearly_final_loads,'Yearly prod (kW)':yearly_prod,'Yearly importation (kW)':yearly_importation,'Yearly exportation (kW)':yearly_exportation,'Yearly storage (kW)':yearly_storage})  
    
    
    EROI = (productible_energy+productible_energy_DG)/(consumed_energy_DG+consumed_energy_production+consumed_energy_storage)
    
    annual_fossil_fuel_consumption = annual_fossil_fuel_consumption_importation+annual_fuel_consumption
    annual_CO2eq_total = annual_CO2eq_DG+annual_CO2eq_prod+annual_CO2eq_importation+annual_CO2eq_storage
    LCOE=(annual_cost_production+economics_CAPEX_storage+economics_OPEX_storage+DG_OPEX_cost+DG_CAPEX_cost+annual_total_fuel_cost+economics_importation+economics_contract_power+economics_overrun-economics_exportation)/(sum(Optimized_Load)/Contexte.time.time_resolution/Contexte.time.duration_years)   if sum(Optimized_Load )>0 else np.nan

    Annual_net_benefits = (-annual_cost_production-economics_CAPEX_storage-economics_OPEX_storage-DG_OPEX_cost-DG_CAPEX_cost-annual_total_fuel_cost-economics_importation-economics_contract_power-economics_overrun+economics_exportation)
    Initial_investment = sum(CAPEX_storage_cost) +0 +0
    fitness_baseline=np.nan

    return{'TimeSeries':{'Storage_TS (kW)':storage_TS,'Storage cumulative energy (kWh)':sum_diff_storages,'SOCs (%)':SOCs,'Losses (kW)': losses,'D_DSM (kW)':D_DSM,'Y_DSM (kW)':Y_DSM,'trades (kW)':trades,'Optimized load (kW)':Optimized_Load,'Non movable load (kW)':Contexte.loads.non_movable,'production (kW)':production,'Grid trading (kW)':Grid_trading,'Grid importation (kW)':Grid_importation,'Grid exportation (kW)':Grid_exportation,'DG production (kW)':DG_production,'Curtailment (kW)':curtailment},
           'Technical':{'fitness':fitness_baseline,'Self-sufficiency':obtained_self_sufficiency,'Self-consumption':self_consumption,'Autonomy':Autonomy,'EnR fraction':EnR_fraction,'EnR Self-sufficiency':EnR_self_sufficiency,'EnR Autonomy':EnR_autonomy,'Annual sum losses (kWh)':sum(np.sum(losses,axis=1))/Contexte.time.time_resolution/Contexte.time.duration_years,'Contract power (kW)':Contract_power,'Capacity factor':Capacity_factor,'Max power from grid (kW)':Max_power_from_grid,'Max power to grid (kW)':Max_power_to_grid,'Max curtailment (kW)':Max_curtailment ,'Installation lifetime (yrs.)':np.nan},
           'economics':{'LCOE (€/kWh)':LCOE,'Cost production (€/kWh)':annual_cost_production/(sum(Optimized_Load)/Contexte.time.time_resolution/Contexte.time.duration_years) ,'OPEX storage (€/kWh)':economics_OPEX_storage/(sum(Optimized_Load)/Contexte.time.time_resolution/Contexte.time.duration_years),'CAPEX storage (€/kWh)':economics_CAPEX_storage/(sum(Optimized_Load)/Contexte.time.time_resolution/Contexte.time.duration_years), 'Contract power (€/kWh)':economics_contract_power/(sum(Optimized_Load)/Contexte.time.time_resolution/Contexte.time.duration_years),'Overrun penalty (€/kWh)':economics_overrun/(sum(Optimized_Load)/Contexte.time.time_resolution/Contexte.time.duration_years),'Energy importation (€/kWh)':economics_importation/(sum(Optimized_Load)/Contexte.time.time_resolution/Contexte.time.duration_years),'Fixed premium (€/kWh)':economics_contract_power/(sum(Optimized_Load)/Contexte.time.time_resolution/Contexte.time.duration_years),'DG fuel cost (€/kWh)':annual_total_fuel_cost/(sum(Optimized_Load)/Contexte.time.time_resolution/Contexte.time.duration_years),'DG CAPEX cost (€/kWh)':DG_CAPEX_cost/(sum(Optimized_Load)/Contexte.time.time_resolution/Contexte.time.duration_years),'DG OPEX cost (€/kWh)':DG_OPEX_cost/(sum(Optimized_Load)/Contexte.time.time_resolution/Contexte.time.duration_years),'Energy exportation (€/kWh)':economics_exportation/(sum(Optimized_Load)/Contexte.time.time_resolution/Contexte.time.duration_years),'Annual net benefits (€/yrs.)':Annual_net_benefits,'Initial investment (€)':Initial_investment }, 
           'Storages':{'Capacity (kWh)':energy_storages,'Powers in (kW)':powers_in,'Powers out (kW)':powers_out,'Min SOCs (%)':minSOCs ,'Equivalent cycles':Equivalent_cycles,'Storage lifetime (yrs.)':Lifetime}, 
           'Genset':{'nominal_power (kW)':DG_nominal_power, 'Max power from DG (kW)':Max_power_from_DG, 'DG lifetime (yrs.)':DG_lifetime_years,'DG min. production':np.nan, 'DG min. runtime':np.nan},
           'Genset power distribution':closest_levels,
           'Environment':{'Annual fossil fuel from grid (kWh)':annual_fossil_fuel_consumption_importation,'Annual fossil fuel consumption from DG (kWh)':annual_fuel_consumption,'Annual fossil fuel consumption (kWh)':annual_fossil_fuel_consumption,'Annual CO2eq total (tCO2)':annual_CO2eq_total,'expected produced REN energy (kWh)':productible_energy,'expected produced energy from DG (kWh)':productible_energy_DG,'consumed energy for DG (kWh)':consumed_energy_DG ,'Consumed energy for production (kWh)':consumed_energy_production ,'Consumed energy for storage (kWh)':consumed_energy_storage,'EROI':EROI } ,
           'Flows':{'Annual load (kWh)':Annual_load,'Annual REN production' : Annual_REN_production ,'Sum importation (kWh)':Annual_sum_importation,'Sum exportation (kWh)':Annual_sum_exportation},
           'Flows storages':{'Annual stored energy (kWh)':stored_energy,'Annual reported energy (kWh)':reported_energy,'Annual losses (kWh)':Annual_sum_losses},
           'Extra_outputs':{'Uses':{'useprod':useprod,'Loadmeet':Loadmeet,'when_prod':when_prod_exceeds,'when_load':when_load_exceeds},'Logics':{'logicals_sells':logicals_sells,'logicals_buys':logicals_buys,'illogicals_sells':illogicals_sells,'illogicals_buys':illogicals_buys},'Grid usage':{'export':Grid_use_export,'import':Grid_use_import},'distribution_DOD': dist_DOD },
           'Balancing':{'daily time balancing':daily_time_balancing,'yearly time balancing':yearly_time_balancing},
           'Demand-side management':{'DSM daily strategy':DSM_daily_strategy,'DSM yearly strategy':DSM_yearly_strategy,'Load strategy':Load_strategy}
}

####  PRO COST FUNCTIONS ##################


def KPI_pro(solution,Contexte,datetime):
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
            - Technical: Technical KPIs such as self-sufficiency, autonomy, EnR fraction, capacity factor, and installation lifetime.
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
    
    TONS_CONVERSION_FACTOR = 1000000
    KILOS_CONVERSION_FACTOR = 1000

    production = ((Contexte.production.unit_prods.T*solution.production_set).sum(axis=1)+Contexte.production.current_prod)/KILOS_CONVERSION_FACTOR    
    Annual_REN_production = sum(production)/Contexte.time.time_resolution/Contexte.time.duration_years
    annual_cost_production = np.sum((Contexte.production.specs_num[:,PROD_CAPEX]/Contexte.production.specs_num[:,PROD_LIFETIME]+Contexte.production.specs_num[:,PROD_OPEX])*(solution.production_set))
    size_power=np.array([max(solution.storages[1:3,i]) for i in range(Contexte.storage.n_store)])
    
    pro_parameters, global_parameters, grid_parameters, RENSystems_parameters, Genset_parameters, extra_parameters = Ef.build_numba_params(Contexte)
    (storage_TS,trades,D_DSM,Y_DSM,SOCs_eff,losses,P_diff) = Eems.LFE_CCE(solution, global_parameters, pro_parameters, production ,RENSystems_parameters)
             
    sum_diff_storages = np.array([-np.cumsum(storage_TS[i,:]/Contexte.time.time_resolution+losses[i,:]/Contexte.time.time_resolution) for i in range(Contexte.storage.n_store)])             
    Annual_sum_losses = np.sum(losses,axis=1)/Contexte.time.time_resolution/Contexte.time.duration_years      
                                          
    Optimized_Load = Contexte.loads.non_movable+Y_DSM+D_DSM
    Annual_load = sum(Optimized_Load)/Contexte.time.time_resolution/Contexte.time.duration_years
    importation=np.where(trades>0,trades,0)
    exportation = np.where(trades<0,-trades,0)
    signal =  Optimized_Load - production
    Annual_sum_importation=sum(importation)/Contexte.time.time_resolution/Contexte.time.duration_years
    Annual_sum_exportation=sum(exportation)/Contexte.time.time_resolution/Contexte.time.duration_years
        
    logicals_sells = np.where((trades<=0) & (signal<=0))[0]
    logicals_buys = np.where((trades>=0) & (signal>=0))[0]
    illogicals_buys = np.where((trades>0) & (signal<=0))[0]
    illogicals_sells = np.where((trades<0) & (signal>=0))[0]
    use_logical_sells = (sum(Optimized_Load[logicals_sells]),-sum(tuple(sum(np.where(storage_TS[i][logicals_sells]<0,storage_TS[i][logicals_sells],0) for i in range(Contexte.storage.n_store)))),-sum(trades[logicals_sells]))/Contexte.time.time_resolution/Contexte.time.duration_years-sum(tuple(sum(np.where(storage_TS[i][logicals_sells]>0,storage_TS[i][logicals_sells],0) for i in range(Contexte.storage.n_store))))
    use_logical_buys = (sum(production[logicals_buys]),sum(tuple(sum(np.where(storage_TS[i][logicals_buys]>0,storage_TS[i][logicals_buys],0) for i in range(Contexte.storage.n_store)))),sum(trades[logicals_buys]))/Contexte.time.time_resolution/Contexte.time.duration_years
    use_illogical_sells = (sum(production[illogicals_sells]),sum(tuple(sum(np.where(storage_TS[i][illogicals_sells]>0,storage_TS[i][illogicals_sells],0) for i in range(Contexte.storage.n_store))))-sum(trades[illogicals_sells]),-sum(trades[illogicals_sells]))/Contexte.time.time_resolution/Contexte.time.duration_years
    use_illogical_buys = (sum(Optimized_Load[illogicals_buys]),sum(production[illogicals_buys]-sum(Optimized_Load[illogicals_buys])))/Contexte.time.time_resolution/Contexte.time.duration_years
    
    when_prod_exceeds = pd.DataFrame(data=((sum(Optimized_Load[logicals_sells])+sum(Optimized_Load[illogicals_buys]))/Contexte.time.time_resolution/Contexte.time.duration_years,-sum(trades[logicals_sells])/Contexte.time.time_resolution,sum(trades[illogicals_buys])/Contexte.time.time_resolution) + tuple(-sum(storage_TS[i][(signal<0) & (storage_TS[i]<0)])/Contexte.time.time_resolution for i in range(Contexte.storage.n_store)) + tuple(sum(storage_TS[i][(signal<0) & (storage_TS[i]>0)])/Contexte.time.time_resolution for i in range(Contexte.storage.n_store)),index=['Load (kWh)','Grid export (kWh)','Grid import (kWh)']+['Storage '+Contexte.storage.technologies[i]+' charge (kWh)' for i in range(Contexte.storage.n_store)]+['Storage '+Contexte.storage.technologies[i]+' discharge (kWh)' for i in range(Contexte.storage.n_store)]).transpose()
    when_load_exceeds = pd.DataFrame(data=(sum(production[logicals_buys])/Contexte.time.time_resolution+sum(production[illogicals_sells])/Contexte.time.time_resolution,-sum(trades[(signal>0) & (trades<0) ])/Contexte.time.time_resolution,sum(trades[(signal>0) & (trades>0) ])/Contexte.time.time_resolution) + tuple(-sum(storage_TS[i][(signal>0) & (storage_TS[i]<0)])/Contexte.time.time_resolution for i in range(Contexte.storage.n_store)) + tuple(sum(storage_TS[i][(signal>0) & (storage_TS[i]>0)])/Contexte.time.time_resolution for i in range(Contexte.storage.n_store)),index=['Production (kWh)','Grid export (kWh)','Grid import (kWh)']+['Storage '+Contexte.storage.technologies[i]+' charge (kWh)' for i in range(Contexte.storage.n_store)]+['Storage '+Contexte.storage.technologies[i]+' discharge (kWh)' for i in range(Contexte.storage.n_store)]).transpose()
     
    useprod = pd.DataFrame(data=(use_logical_sells[1]+use_illogical_buys[1],use_logical_buys[0]+use_logical_sells[0]+use_illogical_sells[0]+use_illogical_buys[0],use_illogical_sells[2]+use_logical_sells[2] ,Annual_REN_production),index=['Storage (kWh)' ,'Load (kWh)','Exportation (kWh)','Annual REN production (kWh)']).transpose()
    Loadmeet = pd.DataFrame(data=(use_logical_buys[1]+use_illogical_sells[1] ,use_logical_buys[0]+use_logical_sells[0]+use_illogical_sells[0]+use_illogical_buys[0],use_logical_buys[2] ,Annual_load),index=['Storage (kWh)', 'Production (kWh)','Importation (kWh)','Annual load (kWh)']).transpose()
    Grid_use_export = pd.DataFrame(data=(useprod['Exportation (kWh)'], Annual_sum_exportation-useprod['Exportation (kWh)']),index=['Production (kWh)','Storage (kWh)']).transpose()
    Grid_use_import = pd.DataFrame(data=(Loadmeet['Importation (kWh)'], Annual_sum_importation-Loadmeet['Importation (kWh)']),index=['Load (kWh)','Storage (kWh)']).transpose()    
    
    energy_storages = solution.storages[INDIV_PRO_VOLUME,:]/Contexte.storage.characteristics[STOR_DEPTH_OF_DISCHARGE,:]
    powers_out = solution.storages[INDIV_PRO_DISCHARGE_POWER,:]
    powers_in = solution.storages[INDIV_PRO_CHARGE_POWER,:]
    reported_energy = np.sum(np.where(storage_TS>0,storage_TS,0),axis=1)/Contexte.time.time_resolution/Contexte.time.duration_years
    stored_energy = -np.sum(np.where(storage_TS<0,storage_TS,0),axis=1)/Contexte.time.time_resolution/Contexte.time.duration_years
    
    power_storage = np.sum(storage_TS,axis=0)
    
    size_power=np.array([max(powers_in[i],powers_out[i]) for i in range(Contexte.storage.n_store)])
    CAPEX_storage_cost =  np.multiply(size_power,Contexte.storage.characteristics[STOR_POWER_COST,:]) + np.multiply(energy_storages,Contexte.storage.characteristics[STOR_ENERGY_COST,:]) + np.multiply(Contexte.storage.characteristics[STOR_INSTALLATION_COST,:],(size_power>np.repeat(0,Contexte.storage.n_store)))
   
    Equivalent_cycles =  np.array([np.sum(np.abs(storage_TS[i,:]))/(2*Contexte.time.time_resolution*max(energy_storages[i],np.float64(1e-15))*Contexte.time.duration_years) for i in range(Contexte.storage.n_store)]) 
    Lifetime = np.array([min(Contexte.storage.characteristics[STOR_LIFETIME,i],Contexte.storage.characteristics[STOR_CYCLE_LIFE,i]/max(1e-15,Equivalent_cycles[i])) for i in range(Contexte.storage.n_store)])
    economics_CAPEX_storage = sum(np.divide(CAPEX_storage_cost,Lifetime))
    economics_OPEX_storage = sum(np.multiply(Contexte.storage.characteristics[STOR_OM_COST,:],size_power))
    
    minSOCs=(1-Contexte.storage.characteristics[STOR_DEPTH_OF_DISCHARGE,:])/2 
    SOCs=(np.divide((sum_diff_storages.T-np.min(sum_diff_storages,axis=1)),energy_storages) + minSOCs).T
            
    storage_NULL=tuple(energy_storages[i]==0 for i in range(Contexte.storage.n_store))
    
    State_SOCs=[np.searchsorted([(i+1)/100 for i in range(99)], SOCs[j,:]) for j in range(Contexte.storage.n_store)]
    distribution_Depth_of_discharge=np.array([[np.count_nonzero(State_SOCs[i]==j)/len(State_SOCs[i]) for j in range(100)] for i in range(Contexte.storage.n_store)])
    if sum(storage_NULL)>0:
        distribution_Depth_of_discharge[np.where(storage_NULL)[0][0]]=np.repeat(np.nan,100)
    
    dist_DOD = pd.DataFrame(data={'SOC Percentile':[i/100 for i in range(101)]})
    for i in range(Contexte.storage.n_store):
        dist_DOD = dist_DOD.join(pd.DataFrame(data={'SOC distribution ' + Contexte.storage.technologies[i]:distribution_Depth_of_discharge[i]}))
    
    Grid_trading = trades if (Contexte.config.connexion=='On-grid') else np.repeat(0,Contexte.time.n_bits)
    Grid_importation = importation if (Contexte.config.connexion=='On-grid') else np.repeat(0,Contexte.time.n_bits)
    Grid_exportation = exportation if (Contexte.config.connexion=='On-grid') else np.repeat(0,Contexte.time.n_bits)  
    DG_production = importation if (Contexte.config.connexion=='Off-grid') else np.repeat(0,Contexte.time.n_bits)
    curtailment = exportation if (Contexte.config.connexion=='Off-grid') else np.repeat(0,Contexte.time.n_bits)  
        
    annual_CO2eq_prod = sum(sum(np.multiply(np.array([solution.production_set[i]*Contexte.production.unit_prods[i,:] for i in range(len(solution.production_set))]).T/KILOS_CONVERSION_FACTOR,np.array(Contexte.production.specs_num[:,PROD_EMISSIONS]))))/TONS_CONVERSION_FACTOR/Contexte.time.time_resolution/Contexte.time.duration_years
    annual_CO2eq_importation = sum(Grid_importation)*Contexte.grid.C02eqemissions/TONS_CONVERSION_FACTOR/Contexte.time.time_resolution/Contexte.time.duration_years if (Contexte.config.connexion=='On-grid') else 0
    annual_CO2eq_storage = np.inner(np.array([sum(np.where(storage_TS[i]>0,storage_TS[i],0)) for i in range(Contexte.storage.n_store)]),Contexte.storage.characteristics[STOR_EMISSIONS,:])/TONS_CONVERSION_FACTOR/Contexte.time.time_resolution/Contexte.time.duration_years

    annual_fossil_fuel_consumption_importation =  Contexte.grid.fossil_fuel_ratio*sum(Grid_importation)/Contexte.time.time_resolution/Contexte.time.duration_years if (Contexte.config.connexion=='On-grid') else 0
    
    productible_energy = sum(np.sum(np.multiply(solution.production_set,Contexte.production.unit_prods.T),axis=0)/Contexte.time.time_resolution/Contexte.time.duration_years*Contexte.production.specs_num[:,PROD_LIFETIME])
    consumed_energy_production = sum(productible_energy/Contexte.production.specs_num[:,PROD_EROI])
    consumed_energy_storage = sum(np.nanmin(np.array([Contexte.storage.characteristics[STOR_CYCLE_LIFE,:],Contexte.storage.characteristics[STOR_LIFETIME,:]*Equivalent_cycles]),axis=0)*energy_storages/Contexte.storage.characteristics[STOR_ESOEI,:])

     
    if (Contexte.config.connexion=='Off-grid'):
        DG_nominal_power = max(trades)
        DG_production=importation
        if (DG_nominal_power>0):
            closest_levels = np.array([int(10*(DG_production[i]/DG_nominal_power)-0.5) for i in range(Contexte.time.n_bits)])
            annual_fuel_consumption = sum(DG_production*Contexte.genset.fuel_consumption[closest_levels]/Contexte.time.time_resolution/Contexte.time.duration_years)
            DG_lifetime_years = Contexte.genset.lifetime/(sum(np.where(DG_production>0,1,0))/Contexte.time.time_resolution/Contexte.time.duration_years)
        else : 
            closest_levels = np.array([np.nan for i in range(Contexte.time.n_bits)])
            annual_fuel_consumption = 0
            DG_lifetime_years = np.nan
        annual_total_fuel_cost = annual_fuel_consumption*Contexte.genset.fuel_cost
        DG_CAPEX_cost = DG_nominal_power*Contexte.genset.unit_cost/DG_lifetime_years
        DG_OPEX_cost = sum(DG_production)*Contexte.genset.maintenance_cost/Contexte.time.time_resolution/Contexte.time.duration_years
        Contract_power=0
        economics_exportation = 0
        economics_importation = 0
        economics_overrun = 0
        economics_contract_power = 0        
        annual_CO2eq_DG = annual_fuel_consumption*Contexte.genset.fuel_CO2eq_emissions/TONS_CONVERSION_FACTOR  
        productible_energy_DG = sum(DG_production)/Contexte.time.time_resolution/Contexte.time.duration_years*DG_lifetime_years
        consumed_energy_DG = productible_energy_DG/Contexte.genset.EROI
        
    elif (Contexte.config.connexion=='On-grid'):          
        DG_nominal_power = 0
        DG_production=np.repeat(0,Contexte.time.n_bits)
        closest_levels = 0
        annual_fuel_consumption = 0
        annual_total_fuel_cost = 0
        DG_lifetime_years = np.nan
        DG_CAPEX_cost = 0
        DG_OPEX_cost = 0
        Contract_power=max(0,max(trades))
        economics_exportation = np.multiply(exportation,Contexte.grid.selling_price[solution.contract,:]).sum()/Contexte.time.time_resolution/Contexte.time.duration_years
        economics_importation = np.multiply(importation,Contexte.grid.prices[solution.contract,:]).sum()/Contexte.time.time_resolution/Contexte.time.duration_years
        economics_overrun = max(0,(max(importation)-Contract_power)*Contexte.grid.overrun[solution.contract])
        economics_contract_power = Contexte.grid.fixed_premium[solution.contract]*Contract_power
        annual_CO2eq_DG = 0.
        productible_energy_DG = 0.
        consumed_energy_DG = 0.
        
    obtained_self_sufficiency = (1-sum(Grid_importation)/sum(Optimized_Load)) if sum(Optimized_Load )>0 else np.nan
    EnR_self_sufficiency = (1-sum(importation)/sum(Optimized_Load)) if sum(Optimized_Load )>0 else np.nan
    self_consumption = (1-sum(exportation)/sum(production)) if sum(production )>0 else np.nan
    EnR_fraction = sum(production)/Annual_load if Annual_load>0 else np.nan
    
    obtained_constraint_level = obtained_self_sufficiency if global_parameters.constraint_num==CONS_Self_sufficiency else self_consumption if (global_parameters.constraint_num==CONS_Self_consumption) else EnR_fraction if(global_parameters.constraint_num==CONS_REN_fraction) else np.nan

    Autonomy = 1-sum(Grid_importation>0)/Contexte.time.n_bits
    EnR_autonomy = 1-sum(importation>0)/Contexte.time.n_bits
    Capacity_factor = sum(production)/(max(production)*Contexte.time.n_bits) if max(production)>0 else np.nan
    Max_power_from_grid = max(Grid_importation) 
    Max_power_from_DG = max(DG_production) 
    Max_power_to_grid = max(Grid_exportation) 
    Max_curtailment = max(curtailment)   
    
    #Demand-side management
    indexes_hour = [[int((i+j*Contexte.time.time_resolution*24)) for j in range(Contexte.time.n_days)] for i in range(int(Contexte.time.time_resolution*24))]
    Daily_base_load = [np.mean(Contexte.loads.non_movable[indexes_hour[j]]) for j in range(int(Contexte.time.time_resolution*24))]
    Daily_movable_loads = [np.mean(Contexte.loads.D_movable[indexes_hour[j]]+Contexte.loads.Y_movable[indexes_hour[j]]) for j in range(int(Contexte.time.time_resolution*24))]
    Daily_final_loads = [np.mean(Optimized_Load[indexes_hour[j]]) for j in range(int(Contexte.time.time_resolution*24))]
    Daily_prod = [np.mean(production[indexes_hour[j]]) for j in range(int(Contexte.time.time_resolution*24))]
    
    indexes_days = [[int(i+j*Contexte.time.time_resolution*24) for i in range(int(Contexte.time.time_resolution*24))] for j in range(Contexte.time.n_days)]
    Yearly_base_load = [np.mean(Contexte.loads.non_movable[indexes_days[j]]+Contexte.loads.D_movable[indexes_days[j]]) for j in range(Contexte.time.n_days)]
    Yearly_movable_loads = [np.mean(Contexte.loads.Y_movable[indexes_days[j]]) for j in range(Contexte.time.n_days)]
    Yearly_final_loads = [np.mean(Optimized_Load[indexes_days[j]]) for j in range(Contexte.time.n_days)]
    Yearly_prod = [np.mean(production[indexes_days[j]]) for j in range(Contexte.time.n_days)]
    
    DSM_daily_strategy = pd.DataFrame(data={'Datetime':datetime[0:int(Contexte.time.time_resolution*24)].strftime('%H:%M'),'Daily base load (kW)':Daily_base_load,'Daily movable load (kW)':Daily_movable_loads,'Daily final load (kW)':Daily_final_loads,'Daily prod (kW)':Daily_prod})
    DSM_yearly_strategy = pd.DataFrame(data={'Datetime':datetime[indexes_hour[0]].strftime('%Y-%m-%d'),'Yearly base load (kW)':Yearly_base_load,'Yearly movable load (kW)':Yearly_movable_loads,'Yearly final load (kW)':Yearly_final_loads,'Yearly prod (kW)':Yearly_prod})
    Load_strategy = pd.DataFrame(data={'Datetime':datetime,'Non controllable load (kW)':Contexte.loads.non_movable,'Daily movable load (kW)':Contexte.loads.D_movable,'Yearly movable load (kW)':Contexte.loads.Y_movable, 'Daily optimized load (kW)':D_DSM.flatten(),'Yearly optimized load (kW)':Y_DSM})
  
    daily_prod = [np.mean(production[indexes_hour[j]]) for j in range(int(Contexte.time.time_resolution*24))]
    yearly_prod = [np.mean(production[indexes_days[j]]) for j in range(Contexte.time.n_days)]
    daily_importation = [np.mean(importation[indexes_hour[j]]) for j in range(int(Contexte.time.time_resolution*24))]
    yearly_importation = [np.mean(importation[indexes_days[j]]) for j in range(Contexte.time.n_days)]
    daily_exportation = [-np.mean(exportation[indexes_hour[j]]) for j in range(int(Contexte.time.time_resolution*24))]
    yearly_exportation = [-np.mean(exportation[indexes_days[j]]) for j in range(Contexte.time.n_days)]
    daily_storage = [np.mean(power_storage[indexes_hour[j]]) for j in range(int(Contexte.time.time_resolution*24))]
    yearly_storage = [np.mean(power_storage[indexes_days[j]]) for j in range(Contexte.time.n_days)]   
    daily_time_balancing = pd.DataFrame(data={'Datetime':datetime[0:int(Contexte.time.time_resolution*24)].strftime('%H:%M'),'Daily load (kW)':Daily_final_loads,'Daily prod (kW)':daily_prod,'Daily importation (kW)':daily_importation,'Daily exportation (kW)':daily_exportation,'Daily storage (kW)':daily_storage})
    yearly_time_balancing = pd.DataFrame(data={'Datetime':datetime[indexes_hour[0]].strftime('%Y-%m-%d'),'Yearly load (kW)':Yearly_final_loads,'Yearly prod (kW)':yearly_prod,'Yearly importation (kW)':yearly_importation,'Yearly exportation (kW)':yearly_exportation,'Yearly storage (kW)':yearly_storage})  
    
    
    
    EROI = (productible_energy+productible_energy_DG)/(consumed_energy_DG+consumed_energy_production+consumed_energy_storage)
    
    annual_fossil_fuel_consumption = annual_fossil_fuel_consumption_importation+annual_fuel_consumption
    annual_CO2eq_total = annual_CO2eq_DG+annual_CO2eq_prod+annual_CO2eq_importation+annual_CO2eq_storage
    LCOE=(annual_cost_production+economics_CAPEX_storage+economics_OPEX_storage+DG_OPEX_cost+DG_CAPEX_cost+annual_total_fuel_cost+economics_importation+economics_contract_power+economics_overrun-economics_exportation)/(sum(Optimized_Load)/Contexte.time.time_resolution/Contexte.time.duration_years)   if sum(Optimized_Load )>0 else np.nan
    
    Lifetime_installation = np.nanmin(np.array([DG_lifetime_years,min(Lifetime),np.nanmin(np.where(solution.production_set>0,Contexte.production.specs_num[:,PROD_LIFETIME],np.nan))]))
    Annual_net_benefits = (-annual_cost_production-economics_CAPEX_storage-economics_OPEX_storage-DG_OPEX_cost-DG_CAPEX_cost-annual_total_fuel_cost-economics_importation-economics_contract_power-economics_overrun+economics_exportation)
    Value = Annual_net_benefits*Lifetime_installation
    Initial_investment = sum(CAPEX_storage_cost) +np.nanmin((0.,DG_CAPEX_cost*DG_lifetime_years))+ np.sum(Contexte.production.specs_num[:,PROD_CAPEX]*solution.production_set)
        
    return{'TimeSeries':{'Storage_TS (kW)':storage_TS,'Storage cumulative energy (kWh)':sum_diff_storages,'SOCs (%)':SOCs,'Losses (kW)': losses,'D_DSM (kW)':D_DSM,'Y_DSM (kW)':Y_DSM,'trades (kW)':trades,'Optimized load (kW)':Optimized_Load,'Non movable load (kW)':Contexte.loads.non_movable,'production (kW)':production,'Grid trading (kW)':Grid_trading,'Grid importation (kW)':Grid_importation,'Grid exportation (kW)':Grid_exportation,'DG production (kW)':DG_production,'Curtailment (kW)':curtailment},
           'Technical':{'fitness':solution.fitness,'Self-sufficiency':obtained_self_sufficiency,'Self-consumption':self_consumption,'Autonomy':Autonomy,'EnR fraction':EnR_fraction,'EnR Self-sufficiency':EnR_self_sufficiency,'EnR Autonomy':EnR_autonomy,'Annual sum losses (kWh)':sum(np.sum(losses,axis=1))/Contexte.time.time_resolution/Contexte.time.duration_years,'Contract power (kW)':Contract_power,'Capacity factor':Capacity_factor,'Max power from grid (kW)':Max_power_from_grid,'Max power to grid (kW)':Max_power_to_grid,'Max curtailment (kW)':Max_curtailment,'Installation lifetime (yrs.)':Lifetime_installation },
           'economics':{'LCOE (€/kWh)':LCOE,'Cost production (€/kWh)':annual_cost_production/(sum(Optimized_Load)/Contexte.time.time_resolution/Contexte.time.duration_years) ,'OPEX storage (€/kWh)':economics_OPEX_storage/(sum(Optimized_Load)/Contexte.time.time_resolution/Contexte.time.duration_years),'CAPEX storage (€/kWh)':economics_CAPEX_storage/(sum(Optimized_Load)/Contexte.time.time_resolution/Contexte.time.duration_years), 'Contract power (€/kWh)':economics_contract_power/(sum(Optimized_Load)/Contexte.time.time_resolution/Contexte.time.duration_years),'Overrun penalty (€/kWh)':economics_overrun/(sum(Optimized_Load)/Contexte.time.time_resolution/Contexte.time.duration_years),'Energy importation (€/kWh)':economics_importation/(sum(Optimized_Load)/Contexte.time.time_resolution/Contexte.time.duration_years),'Fixed premium (€/kWh)':economics_contract_power/(sum(Optimized_Load)/Contexte.time.time_resolution/Contexte.time.duration_years),'DG fuel cost (€/kWh)':annual_total_fuel_cost/(sum(Optimized_Load)/Contexte.time.time_resolution/Contexte.time.duration_years),'DG CAPEX cost (€/kWh)':DG_CAPEX_cost/(sum(Optimized_Load)/Contexte.time.time_resolution/Contexte.time.duration_years),'DG OPEX cost (€/kWh)':DG_OPEX_cost/(sum(Optimized_Load)/Contexte.time.time_resolution/Contexte.time.duration_years),'Energy exportation (€/kWh)':economics_exportation/(sum(Optimized_Load)/Contexte.time.time_resolution/Contexte.time.duration_years),'Annual net benefits (€/yrs.)':Annual_net_benefits,'Value (€)':Value,'Initial investment (€)':Initial_investment}, 
           'Storages':{'Capacity (kWh)':energy_storages,'Powers in (kW)':powers_in,'Powers out (kW)':powers_out,'Min SOCs (%)':minSOCs ,'Equivalent cycles':Equivalent_cycles,'Storage lifetime (yrs.)':Lifetime}, 
           'Genset':{'nominal_power (kW)':DG_nominal_power, 'Max power from DG (kW)':Max_power_from_DG, 'DG lifetime (yrs.)':DG_lifetime_years,'DG min. production':solution.DG_min_production, 'DG min. runtime':solution.DG_min_runtime },
           'Genset power distribution':closest_levels,
           'EMS' : {'D_DSM min. levels':solution.D_DSM_minimum_levels,'Y_DSM min. levels':solution.Y_DSM_minimum_levels,'discharge order':solution.discharge_order,'strategy':solution.DG_strategy, 'overlaps':solution.overlaps,'energy repartition coefficient':solution.energy_use_coefficient },
           'Environment':{'Annual fossil fuel from grid (kWh)':annual_fossil_fuel_consumption_importation,'Annual fossil fuel consumption from DG (kWh)':annual_fuel_consumption,'Annual fossil fuel consumption (kWh)':annual_fossil_fuel_consumption,'Annual CO2eq total (tCO2)':annual_CO2eq_total,'expected produced REN energy (kWh)':productible_energy,'expected produced energy from DG (kWh)':productible_energy_DG,'consumed energy for DG (kWh)':consumed_energy_DG ,'Consumed energy for production (kWh)':consumed_energy_production ,'Consumed energy for storage (kWh)':consumed_energy_storage,'EROI':EROI },
           'Flows':{'Annual load (kWh)':Annual_load,'Annual REN production' : Annual_REN_production ,'Sum importation (kWh)':Annual_sum_importation,'Sum exportation (kWh)':Annual_sum_exportation},
           'Flows storages':{'Annual stored energy (kWh)':stored_energy,'Annual reported energy (kWh)':reported_energy,'Annual losses (kWh)':Annual_sum_losses},
           'Extra_outputs':{'Uses':{'useprod':useprod,'Loadmeet':Loadmeet,'when_prod':when_prod_exceeds,'when_load':when_load_exceeds},'Logics':{'logicals_sells':logicals_sells,'logicals_buys':logicals_buys,'illogicals_sells':illogicals_sells,'illogicals_buys':illogicals_buys},'Grid usage':{'export':Grid_use_export,'import':Grid_use_import},'distribution_DOD': dist_DOD },
           'Balancing':{'daily time balancing':daily_time_balancing,'yearly time balancing':yearly_time_balancing},
           'Demand-side management':{'DSM daily strategy':DSM_daily_strategy,'DSM yearly strategy':DSM_yearly_strategy,'Load strategy':Load_strategy}
           }

@jit(nopython=True)
def get_constraint_level(global_parameters,importation,den_Optimized_load,trades,production):
    return(1-(sum(importation)/den_Optimized_load) if global_parameters.constraint_num==CONS_Self_sufficiency else (1+(sum(np.where(trades<0,trades,0))/sum(production)) if sum(production)>0 else 0) if (global_parameters.constraint_num==CONS_Self_consumption) else sum(production)/den_Optimized_load if(global_parameters.constraint_num==CONS_REN_fraction) else 0)
   
@jit(nopython=True)
def pro_annual_cost_production(gene,RENSystems_parameters):
    return(np.sum((RENSystems_parameters.specs_prod[:,PROD_CAPEX]/RENSystems_parameters.specs_prod[:,PROD_LIFETIME]+RENSystems_parameters.specs_prod[:,PROD_OPEX])*(gene.production_set)))

@jit(nopython=True)
def pro_cost_base_indicators(gene,RENSystems_parameters,global_parameters,pro_parameters):
    KILOS_CONVERSION_FACTOR = 1000
    production = ((RENSystems_parameters.unit_productions.T*gene.production_set).sum(axis=1)+RENSystems_parameters.current_production)/KILOS_CONVERSION_FACTOR   
    (storage_TS,trades,D_DSM,Y_DSM,SOCs_eff,losses,P_diff) = Eems.LFE_CCE(gene, global_parameters, pro_parameters, production ,RENSystems_parameters)
    Optimized_Load = global_parameters.Non_movable_load+Y_DSM+D_DSM
    return(production, Optimized_Load,trades,storage_TS,D_DSM,Y_DSM,SOCs_eff,losses,P_diff)

@jit(nopython=True)
def pro_update_storage_power(gene,RENSystems_parameters,storage_TS):
    for i in range(RENSystems_parameters.n_store):
        gene.storages[INDIV_PRO_CHARGE_POWER][i]=max(-(storage_TS[i]))   
        gene.storages[INDIV_PRO_DISCHARGE_POWER][i]=max(storage_TS[i])  
    return(gene.storages)

@jit(nopython=True)
def pro_size_storage_power(gene,RENSystems_parameters):
    powers_out = gene.storages[INDIV_PRO_DISCHARGE_POWER,:]
    powers_in = gene.storages[INDIV_PRO_CHARGE_POWER,:]
    size_power=np.array([max(powers_in[i],powers_out[i]) for i in range(RENSystems_parameters.n_store)])
    return(size_power)



@jit(nopython=True)
def pro_indicators_storage(gene,RENSystems_parameters,global_parameters,storage_TS,losses):
    energy_storages = gene.storages[INDIV_PRO_VOLUME,:]/RENSystems_parameters.specs_storage[STOR_DEPTH_OF_DISCHARGE,:]
    Equivalent_cycles =  np.array([np.sum(np.abs(storage_TS[i,:]))/(2*global_parameters.time_resolution*max(energy_storages[i],np.float64(1e-15))*global_parameters.duration_years) for i in range(RENSystems_parameters.n_store)]) 
    Lifetime = np.array([min(RENSystems_parameters.specs_storage[STOR_LIFETIME,i],RENSystems_parameters.specs_storage[STOR_CYCLE_LIFE,i]/max(1e-15,Equivalent_cycles[i])) for i in range(RENSystems_parameters.n_store)])
    return(energy_storages,Lifetime)


@jit(nopython=True)
def pro_indicators_Genset(gene,Genset_parameters,global_parameters,trades,importation):
    DG_nominal_power = max(trades)
    DG_production=importation
    if (DG_nominal_power>0):
        closest_levels = np.array([int(10*(DG_production[i]/DG_nominal_power)-0.5) for i in range(global_parameters.n_bits)])
        annual_fuel_consumption = sum(DG_production*Genset_parameters.fuel_consumption[closest_levels]/global_parameters.time_resolution/global_parameters.duration_years)
    else : 
        annual_fuel_consumption = 0
    return(DG_nominal_power,DG_production,annual_fuel_consumption)

@jit(nopython=True)
def pro_economics_Genset(annual_fuel_consumption,DG_production,DG_nominal_power,Genset_parameters,global_parameters):
    annual_total_fuel_cost = annual_fuel_consumption*Genset_parameters.fuel_cost   
    DG_CAPEX_cost = DG_nominal_power*Genset_parameters.unit_cost*sum(np.where(DG_production>0,1,0))/global_parameters.time_resolution/Genset_parameters.lifetime/global_parameters.duration_years
    DG_OPEX_cost = sum(DG_production)*Genset_parameters.maintenance_cost/global_parameters.time_resolution/global_parameters.duration_years
    return(annual_total_fuel_cost,DG_CAPEX_cost,DG_OPEX_cost)
    
@jit(nopython=True)
def pro_economics_Storage(gene,energy_storages,RENSystems_parameters,Lifetime,size_power):
    CAPEX_storage_cost =  np.multiply(size_power,RENSystems_parameters.specs_storage[STOR_POWER_COST,:]) + np.multiply(energy_storages,RENSystems_parameters.specs_storage[STOR_ENERGY_COST,:]) + np.multiply(RENSystems_parameters.specs_storage[STOR_INSTALLATION_COST,:],(size_power>np.repeat(0,RENSystems_parameters.n_store)))
    economics_CAPEX_storage = sum(np.divide(CAPEX_storage_cost,Lifetime))
    economics_OPEX_storage = sum(np.multiply(RENSystems_parameters.specs_storage[STOR_OM_COST,:],size_power))
    return(economics_CAPEX_storage,economics_OPEX_storage)    
  

@jit(nopython=True)
def pro_economics_Grid(gene,global_parameters,grid_parameters,importation,exportation):
    Contract_power=max(0,max(importation))
    economics_exportation = np.multiply(exportation,grid_parameters.Selling_price[gene.contract,:]).sum()/global_parameters.time_resolution/global_parameters.duration_years
    economics_importation = np.multiply(importation,grid_parameters.prices[gene.contract,:]).sum()/global_parameters.time_resolution/global_parameters.duration_years
    economics_overrun = max(0,(max(importation)-Contract_power)*grid_parameters.Overrun[gene.contract])
    economics_contract_power = grid_parameters.fixed_premium[gene.contract]*Contract_power
    return(economics_exportation,economics_importation,economics_overrun,economics_contract_power)



@jit(nopython=True)
def Compute_fitness(global_parameters,obtained_constraint_level,criterion_value):
    return((global_parameters.cost_constraint*(global_parameters.constraint_level-obtained_constraint_level) if obtained_constraint_level<global_parameters.constraint_level else 0) + criterion_value)



@jit(nopython=True)
def LCOE_pro(gene,RENSystems_parameters,global_parameters,Genset_parameters,grid_parameters,pro_parameters):
    """
    Computes the Levelized Cost of Energy (LCOE) fitness for a candidate microgrid configuration.
    
    Evaluates the economic performance of a candidate solution (`gene`) by combining
    production costs, storage CAPEX/OPEX, diesel generator costs, and grid interactions.
    Fitness penalizes solutions that violate specified system constraints.
    
    Args:
        gene (object): Candidate microgrid configuration, including production and storage sizing.
        storage_characteristics (np.ndarray): Storage parameters (capacity, efficiency, costs, lifetime).
        time_resolution (float): Simulation steps per hour.
        n_store (int): Number of storage units.
        duration_years (float): Simulation duration in years.
        specs_num (np.ndarray): Technical specs of production units (capacity, O&M, etc.).
        prices_num (np.ndarray): Grid electricity prices per timestep.
        fixed_premium (np.ndarray): Fixed contracted power costs.
        Overrun (np.ndarray): Overrun penalties for exceeding contracted grid power.
        Selling_price (np.ndarray): Price for exporting electricity to the grid.
        Non_movable_load (np.ndarray): Time series of inflexible loads.
        total_D_Movable_load (np.ndarray): Daily-shiftable load aggregated.
        total_Y_Movable_load (np.ndarray): Yearly-shiftable load aggregated.
        Main_grid_emissions (float): Grid CO2eq emissions (gCO2/kWh).
        prod_C (np.ndarray): Constant production values independent of decisions.
        prods_U (np.ndarray): Unit-specific renewable production profiles.
        Bounds_prod (np.ndarray): Operational bounds of production units.
        constraint_num (int): Constraint type (1=import, 2=export, 3=production ratio).
        constraint_level (float): Threshold value for the constraint.
        cost_constraint (float): Maximum allowed cost for penalization.
        n_bits (int): Total number of timesteps.
        Connexion (str): Grid connection type ('On-grid' or 'Off-grid').
        DG_fuel_consumption (np.ndarray): Diesel generator fuel consumption per output level.
        DG_fuel_cost (float): Diesel fuel cost.
        DG_unit_cost (float): Capital cost per kW of diesel generator.
        DG_lifetime (float): Diesel generator lifetime in years.
        DG_maintenance_cost (float): Annual O&M cost of diesel generator.
        DG_EROI (float): Energy return on investment of diesel generator fuel.
        fuel_CO2eq_emissions (float): CO2eq emissions per kWh of diesel fuel.
        Grid_fossil_fuel_ratio (float): Fraction of grid electricity from fossil fuel.
    
    Returns:
        object: The input `gene` object with updated fitness based on LCOE.
    
    Note:
        - This function updates `gene.storages` and `gene.fitness` in-place.
        - Uses `Eems.LFE_CCE` to simulate storage dispatch and DSM contributions.
        - Penalization is applied if `obtained_constraint_level < constraint_level`.
    
    Important:
        - When `Connexion='Off-grid'`, fitness accounts for diesel generator operation.
        - When `Connexion='On-grid'`, fitness accounts for import/export economics, contract premiums, and penalties.
    
    Warning:
        - Avoid divisions by zero: `sum(production)` and `sum(Optimized_Load)` are used in constraints and may produce NaN if zero.
        - Ensure `n_store` matches the dimensions of `storage_characteristics`.
        - `DG_production` computation assumes `DG_nominal_power > 0` to index fuel consumption.
    """
    annual_cost_production = pro_annual_cost_production(gene,RENSystems_parameters)
    (production, Optimized_Load,trades,storage_TS,D_DSM,Y_DSM,SOCs_eff,losses,P_diff) = pro_cost_base_indicators(gene,RENSystems_parameters,global_parameters,pro_parameters)

    gene.storages = pro_update_storage_power(gene,RENSystems_parameters,storage_TS)        
    
    importation=np.where(trades>0,trades,0)    
    den_Optimized_load = max(1e-15,sum(Optimized_Load))
    obtained_constraint_level = get_constraint_level(global_parameters,importation,den_Optimized_load,trades,production)
    size_power = pro_size_storage_power(gene,RENSystems_parameters) 
    energy_storages,Lifetime = pro_indicators_storage(gene,RENSystems_parameters,global_parameters,storage_TS,losses)
    (economics_CAPEX_storage,economics_OPEX_storage) = pro_economics_Storage(gene,energy_storages,RENSystems_parameters,Lifetime,size_power)
          
    if (global_parameters.Connexion==GRID_OFF):
        (DG_nominal_power,DG_production,annual_fuel_consumption) = pro_indicators_Genset(gene,Genset_parameters,global_parameters,trades,importation)
        (annual_total_fuel_cost,DG_CAPEX_cost,DG_OPEX_cost) = pro_economics_Genset(annual_fuel_consumption,DG_production,DG_nominal_power,Genset_parameters,global_parameters)
        LCOE=(annual_cost_production+economics_CAPEX_storage+economics_OPEX_storage+DG_OPEX_cost+DG_CAPEX_cost+annual_total_fuel_cost)/(den_Optimized_load/global_parameters.time_resolution/global_parameters.duration_years)
    
    elif (global_parameters.Connexion==GRID_ON):            
        exportation = np.where(trades<0,trades,0)
        (economics_exportation,economics_importation,economics_overrun,economics_contract_power) = pro_economics_Grid(gene,global_parameters,grid_parameters,importation,exportation)
        LCOE=(annual_cost_production+economics_CAPEX_storage+economics_OPEX_storage+economics_importation+economics_contract_power+economics_overrun+economics_exportation)/(den_Optimized_load/global_parameters.time_resolution/global_parameters.duration_years) 

    gene.fitness=Compute_fitness(global_parameters,obtained_constraint_level,LCOE) 

    return(gene)
                                                        
@jit(nopython=True)
def Self_consumption_pro(gene,RENSystems_parameters,global_parameters,Genset_parameters,grid_parameters,pro_parameters):
    """
    Evaluates the self-consumption fitness of a candidate microgrid configuration.
    
    The fitness is defined as the fraction of locally produced energy directly consumed
    by the loads, penalized if operational constraints are violated. Storage dispatch and
    demand-side management (DSM) are included in the calculation.
    
    See 'LCOE_pro' for full attributes description.
    
    Returns:
        object: Updated `gene` object with computed self-consumption fitness.
    
    Note:
        - Fitness penalizes candidate solutions that violate `constraint_level`.
        - Uses `Eems.LFE_CCE` to simulate storage dispatch and DSM.
        - Updates `gene.storages` in-place.
    
    Important:
        - The self-consumption metric is computed as 1 + exported energy fraction.
        - Penalization applies only if `obtained_constraint_level < constraint_level`.
    
    Warning:
        - Divisions by zero may occur if `sum(production) == 0`. Ensure production is non-zero.
        - The function assumes `n_store` matches the dimensions of `storage_characteristics`.
    """     
    (production, Optimized_Load,trades,storage_TS,D_DSM,Y_DSM,SOCs_eff,losses,P_diff) = pro_cost_base_indicators(gene,RENSystems_parameters,global_parameters,pro_parameters)
    gene.storages = pro_update_storage_power(gene,RENSystems_parameters,storage_TS)        
    importation=np.where(trades>0,trades,0)
    den_Optimized_load = max(1e-15,sum(Optimized_Load))
    
    obtained_constraint_level = get_constraint_level(global_parameters,importation,den_Optimized_load,trades,production)
    obtained_self_consumption = 1+(sum(np.where(trades<0,trades,0))/sum(production))
    gene.fitness=Compute_fitness(global_parameters,obtained_constraint_level, -obtained_self_consumption)  
    return(gene)

@jit(nopython=True)
def Self_sufficiency_pro(gene,RENSystems_parameters,global_parameters,Genset_parameters,grid_parameters,pro_parameters):
    """
    Evaluates the self-sufficiency fitness of a candidate microgrid configuration.
    
    Self-sufficiency is defined as the fraction of total load met by local generation,
    including storage dispatch and DSM, without reliance on grid import. Penalizes
    violations of operational constraints.
    
    See 'LCOE_pro' for full attributes description.
    
    Returns:
        object: Updated `gene` object with computed self-sufficiency fitness.
    
    Note:
        - Fitness penalizes candidate solutions that violate `constraint_level`.
        - Self-sufficiency is computed as the fraction of load met by local generation.
        - Uses `Eems.LFE_CCE` to simulate storage dispatch and DSM.
        - Updates `gene.storages` in-place.
    
    Important:
        - For Off-grid systems, grid import is assumed zero; for On-grid systems, import affects self-sufficiency.
        - Penalization applies only if `obtained_constraint_level < constraint_level`.
    
    Warning:
        - Divisions by zero may occur if `sum(Optimized_Load) == 0`. Ensure non-zero total load.
        - The function assumes `n_store` matches the dimensions of `storage_characteristics`.
    """
     
    (production, Optimized_Load,trades,storage_TS,D_DSM,Y_DSM,SOCs_eff,losses,P_diff) = pro_cost_base_indicators(gene,RENSystems_parameters,global_parameters,pro_parameters)
    gene.storages = pro_update_storage_power(gene,RENSystems_parameters,storage_TS)                                                           
    importation=np.where(trades>0,trades,0)
    Grid_importation = importation if (Connexion=='On-grid') else np.repeat(0.,n_bits)
    den_Optimized_load = max(1e-15,sum(Optimized_Load))
    
    obtained_constraint_level = get_constraint_level(global_parameters,importation,den_Optimized_load,trades,production)
    obtained_self_sufficiency = (1-sum(Grid_importation)/sum(Optimized_Load)) if sum(Optimized_Load )>0 else np.nan
    gene.fitness=Compute_fitness(global_parameters,obtained_constraint_level, -obtained_self_sufficiency)   
    return(gene)

@jit(nopython=True)
def Max_import_power_pro(gene,RENSystems_parameters,global_parameters,Genset_parameters,grid_parameters,pro_parameters):
    """
    Evaluates the Max import power fitness of a candidate microgrid configuration.
    
    Max import power evaluates the maximum
    positive power exchange (`trades > 0`) after energy management
    optimization and applies a constraint penalty if required.
    
    See 'LCOE_pro' for full attributes description.
    
    Returns:
        object: Updated `gene` object with computed max import power fitness.
    
    Note:
        - Fitness penalizes candidate solutions that violate `constraint_level`.
        - Uses `Eems.LFE_CCE` to simulate storage dispatch and DSM.
        - Updates `gene.storages` in-place.
    
    Important:
        - Penalization applies only if `obtained_constraint_level < constraint_level`.
    
    Warning:
        - Divisions by zero may occur if `sum(Optimized_Load) == 0`. Ensure non-zero total load.
    """

    (production, Optimized_Load,trades,storage_TS,D_DSM,Y_DSM,SOCs_eff,losses,P_diff) = pro_cost_base_indicators(gene,RENSystems_parameters,global_parameters,pro_parameters)
    gene.storages = pro_update_storage_power(gene,RENSystems_parameters,storage_TS)                                                              
    importation=np.where(trades>0,trades,0)
    den_Optimized_load = max(1e-15,sum(Optimized_Load))
    
    obtained_constraint_level = get_constraint_level(global_parameters,importation,den_Optimized_load,trades,production)
    Max_import = max(importation)
    gene.fitness=Compute_fitness(global_parameters,obtained_constraint_level,Max_import)  
    return(gene)

@jit(nopython=True)
def Losses_pro(gene,RENSystems_parameters,global_parameters,Genset_parameters,grid_parameters,pro_parameters):
    """
    Evaluates the Losses fitness of a candidate microgrid configuration.
    
    Losses fitness quantifies energy losses in the system after energy management
    optimization and applies a constraint penalty if required.
    
    See 'LCOE_pro' for full attributes description.
    
    Returns:
        object: Updated `gene` object with computed losses fitness.
    
    Note:
        - Fitness penalizes candidate solutions that violate `constraint_level`.
        - Uses `Eems.LFE_CCE` to simulate storage dispatch and DSM.
        - Updates `gene.storages` in-place.
    
    Important:
        - Penalization applies only if `obtained_constraint_level < constraint_level`.
    
    Warning:
        - Divisions by zero may occur if `sum(Optimized_Load) == 0`. Ensure non-zero total load.
    """
     
    (production, Optimized_Load,trades,storage_TS,D_DSM,Y_DSM,SOCs_eff,losses,P_diff) = pro_cost_base_indicators(gene,RENSystems_parameters,global_parameters,pro_parameters)
    gene.storages = pro_update_storage_power(gene,RENSystems_parameters,storage_TS)                                                               
    importation=np.where(trades>0,trades,0)  
    den_Optimized_load = max(1e-15,sum(Optimized_Load))
    
    obtained_constraint_level = get_constraint_level(global_parameters,importation,den_Optimized_load,trades,production)
    sum_losses = sum(losses)
    gene.fitness=Compute_fitness(global_parameters,obtained_constraint_level,sum_losses)  
    return(gene)

@jit(nopython=True)
def Annual_net_benefits_pro(gene,RENSystems_parameters,global_parameters,Genset_parameters,grid_parameters,pro_parameters):
    """
    Evaluates the Annual net benefits fitness of a candidate microgrid configuration.
    
    Annual net benefits fitness evaluates the total annual income after energy management
    optimization and applies a constraint penalty if required.
    
    See 'LCOE_pro' for full attributes description.
    
    Returns:
        object: Updated `gene` object with computed annual net benefits fitness.
    
    Note:
        - Fitness penalizes candidate solutions that violate `constraint_level`.
        - Uses `Eems.LFE_CCE` to simulate storage dispatch and DSM.
        - Updates `gene.storages` in-place.
    
    Important:
        - Penalization applies only if `obtained_constraint_level < constraint_level`.
    """
    annual_cost_production = pro_annual_cost_production(gene,RENSystems_parameters)
    (production, Optimized_Load,trades,storage_TS,D_DSM,Y_DSM,SOCs_eff,losses,P_diff) = pro_cost_base_indicators(gene,RENSystems_parameters,global_parameters,pro_parameters)
    gene.storages = pro_update_storage_power(gene,RENSystems_parameters,storage_TS)        
    importation=np.where(trades>0,trades,0)
    den_Optimized_load = max(1e-15,sum(Optimized_Load))
    
    obtained_constraint_level = get_constraint_level(global_parameters,importation,den_Optimized_load,trades,production)
    exportation = np.where(trades<0,-trades,0)
    energy_storages,Lifetime = pro_indicators_storage(gene,RENSystems_parameters,global_parameters,storage_TS,losses)
    size_power = pro_size_storage_power(gene,RENSystems_parameters)   
    (economics_CAPEX_storage,economics_OPEX_storage) = pro_economics_Storage(gene,energy_storages,RENSystems_parameters,Lifetime,size_power)

    
    if (global_parameters.Connexion==GRID_OFF):
        (DG_nominal_power,DG_production,annual_fuel_consumption) = pro_indicators_Genset(gene,Genset_parameters,global_parameters,trades,importation)
        (annual_total_fuel_cost,DG_CAPEX_cost,DG_OPEX_cost) = pro_economics_Genset(annual_fuel_consumption,DG_production,DG_nominal_power,Genset_parameters,global_parameters)
        Annual_net_benefits = (-annual_cost_production-economics_CAPEX_storage-economics_OPEX_storage-DG_OPEX_cost-DG_CAPEX_cost-annual_total_fuel_cost)

        
    elif (global_parameters.Connexion==GRID_ON):  
        (economics_exportation,economics_importation,economics_overrun,economics_contract_power) = pro_economics_Grid(gene,global_parameters,grid_parameters,importation,exportation)
        Annual_net_benefits = (-annual_cost_production-economics_CAPEX_storage-economics_OPEX_storage-economics_importation-economics_contract_power-economics_overrun+economics_exportation)
        
    gene.fitness = Compute_fitness(global_parameters,obtained_constraint_level, -Annual_net_benefits) 
    return(gene)

@jit(nopython=True)
def NPV_pro(gene,RENSystems_parameters,global_parameters,Genset_parameters,grid_parameters,pro_parameters):
    """
    Evaluates the Net Present Value (NPV) fitness of a candidate microgrid configuration.
    
    Net present value fitness evaluates the economic performance of a project over its full expected lifetime
    and applies a constraint penalty if required.
    
    See 'LCOE_pro' for full attributes description.
    
    Returns:
        object: Updated `gene` object with computed NPV fitness.
    
    Note:
        - Fitness penalizes candidate solutions that violate `constraint_level`.
        - Uses `Eems.LFE_CCE` to simulate storage dispatch and DSM.
        - Updates `gene.storages` in-place.
    
    Important:
        - Penalization applies only if `obtained_constraint_level < constraint_level`.
    
    Warning:
        - The actualisation factor is assumed to be 0.
    """
    annual_cost_production = pro_annual_cost_production(gene,RENSystems_parameters)
    (production, Optimized_Load,trades,storage_TS,D_DSM,Y_DSM,SOCs_eff,losses,P_diff) = pro_cost_base_indicators(gene,RENSystems_parameters,global_parameters,pro_parameters)
    
    gene.storages = pro_update_storage_power(gene,RENSystems_parameters,storage_TS)        
      
                                                        
    importation=np.where(trades>0,trades,0)
    den_Optimized_load = max(1e-15,sum(Optimized_Load))
    obtained_constraint_level = get_constraint_level(global_parameters,importation,den_Optimized_load,trades,production)
    exportation = np.where(trades<0,-trades,0)
    
    energy_storages,Lifetime = pro_indicators_storage(gene,RENSystems_parameters,global_parameters,storage_TS,losses)

    size_power = pro_size_storage_power(gene,RENSystems_parameters)
    (economics_CAPEX_storage,economics_OPEX_storage) = pro_economics_Storage(gene,energy_storages,RENSystems_parameters,Lifetime,size_power)
    
    if (global_parameters.Connexion==GRID_OFF):
        (DG_nominal_power,DG_production,annual_fuel_consumption) = pro_indicators_Genset(gene,Genset_parameters,global_parameters,trades,importation)
        (annual_total_fuel_cost,DG_CAPEX_cost,DG_OPEX_cost) = pro_economics_Genset(annual_fuel_consumption,DG_production,DG_nominal_power,Genset_parameters,global_parameters)
        Annual_net_benefits = (-annual_cost_production-economics_CAPEX_storage-economics_OPEX_storage-DG_OPEX_cost-DG_CAPEX_cost-annual_total_fuel_cost)
        DG_lifetime_years = Genset_parameters.lifetime/(sum(np.where(DG_production>0,1,0))/global_parameters.time_resolution/global_parameters.duration_years)

    elif (global_parameters.Connexion==GRID_ON):  
        (economics_exportation,economics_importation,economics_overrun,economics_contract_power) = pro_economics_Grid(gene,global_parameters,grid_parameters,importation,exportation)
        Annual_net_benefits = (-annual_cost_production-economics_CAPEX_storage-economics_OPEX_storage-economics_importation-economics_contract_power-economics_overrun+economics_exportation)
        DG_lifetime_years = np.nan

    Lifetime_installation = np.nanmin(DG_lifetime_years,min(Lifetime),np.nanmin(np.where(gene.production_set>0,RENSystems_parameters.specs_prod[:,PROD_LIFETIME],np.nan)))
    NPV = Annual_net_benefits*Lifetime_installation
    gene.fitness = Compute_fitness(global_parameters,obtained_constraint_level, -NPV) 
    return(gene)

@jit(nopython=True)
def Autonomy_pro(gene,RENSystems_parameters,global_parameters,Genset_parameters,grid_parameters,pro_parameters):
    """
    Evaluates the autonomy fitness of a candidate microgrid configuration.
    
    Autonomy represents the proportion of time during which the grid operates fully autonomously
    and applies a constraint penalty if required.
    
    See 'LCOE_pro' for full attributes description.
    
    Returns:
        object: Updated `gene` object with computed autonomy fitness.
    
    Note:
        - Fitness penalizes candidate solutions that violate `constraint_level`.
        - Uses `Eems.LFE_CCE` to simulate storage dispatch and DSM.
        - Updates `gene.storages` in-place.
    
    Important:
        - Penalization applies only if `obtained_constraint_level < constraint_level`.
    
    Warning:
        - Divisions by zero may occur if `sum(Optimized_Load) == 0`. Ensure non-zero total load.
    """ 
    (production, Optimized_Load,trades,storage_TS,D_DSM,Y_DSM,SOCs_eff,losses,P_diff) = pro_cost_base_indicators(gene,RENSystems_parameters,global_parameters,pro_parameters)
    
    gene.storages = pro_update_storage_power(gene,RENSystems_parameters,storage_TS)        
  
        
    importation=np.where(trades>0,trades,0)
    den_Optimized_load = max(1e-15,sum(Optimized_Load))
    obtained_constraint_level = get_constraint_level(global_parameters,importation,den_Optimized_load,trades,production)

    Grid_importation = importation if (global_parameters.Connexion==GRID_ON) else np.repeat(0.,global_parameters.n_bits)
    Autonomy = 1-sum(Grid_importation>0)/global_parameters.n_bits
    gene.fitness = Compute_fitness(global_parameters,obtained_constraint_level, -Autonomy) 

    return(gene)

@jit(nopython=True)
def eqCO2_emissions_pro(gene,RENSystems_parameters,global_parameters,Genset_parameters,grid_parameters,pro_parameters):
    """
    Evaluates the eqCO2 emissions fitness of a candidate microgrid configuration.
    
    eqCO2 emissions fitness represents the total annual greenhouse gas emissions
    and applies a constraint penalty if required.
    
    See 'LCOE_pro' for full attributes description.
    
    Returns:
        object: Updated `gene` object with computed eqCO2 emissions fitness.
    
    Note:
        - Fitness penalizes candidate solutions that violate `constraint_level`.
        - Uses `Eems.LFE_CCE` to simulate storage dispatch and DSM.
        - Updates `gene.storages` in-place.
    
    Important:
        - Penalization applies only if `obtained_constraint_level < constraint_level`.
    
    Warning:
        - Divisions by zero may occur if `sum(Optimized_Load) == 0`. Ensure non-zero total load.
    """  
    KILOS_CONVERSION_FACTOR = 1000
    TONS_CONVERSION_FACTOR = 1000000
    (production, Optimized_Load,trades,storage_TS,D_DSM,Y_DSM,SOCs_eff,losses,P_diff) = pro_cost_base_indicators(gene,RENSystems_parameters,global_parameters,pro_parameters)   
    gene.storages = pro_update_storage_power(gene,RENSystems_parameters,storage_TS)           
        
    importation=np.where(trades>0,trades,0)
    den_Optimized_load = max(1e-15,sum(Optimized_Load))
    obtained_constraint_level = get_constraint_level(global_parameters,importation,den_Optimized_load,trades,production)

    Grid_importation = importation if (global_parameters.Connexion==GRID_ON) else np.repeat(0.,global_parameters.n_bits)
    DG_production = importation if (global_parameters.Connexion==GRID_OFF) else np.repeat(0.,global_parameters.n_bits)
        
    annual_CO2eq_prod = sum(np.sum((gene.production_set*RENSystems_parameters.unit_productions.T)/KILOS_CONVERSION_FACTOR*RENSystems_parameters.specs_prod[:,PROD_EMISSIONS],axis=0))/TONS_CONVERSION_FACTOR/global_parameters.time_resolution/global_parameters.duration_years   
    annual_CO2eq_importation = sum(Grid_importation)*grid_parameters.eqCO2emissions/TONS_CONVERSION_FACTOR/global_parameters.time_resolution/global_parameters.duration_years
    annual_CO2eq_storage = np.inner(np.array([sum(np.where(storage_TS[i]>0,storage_TS[i],0)) for i in range(RENSystems_parameters.n_store)]),RENSystems_parameters.specs_storage[STOR_EMISSIONS,:])/TONS_CONVERSION_FACTOR/global_parameters.time_resolution/global_parameters.duration_years

    if (global_parameters.Connexion==GRID_OFF):
        (DG_nominal_power,DG_production,annual_fuel_consumption) = pro_indicators_Genset(gene,Genset_parameters,global_parameters,trades,importation)
        annual_CO2eq_DG = annual_fuel_consumption*Genset_parameters.fuel_CO2eq_emissions/TONS_CONVERSION_FACTOR      
    elif (global_parameters.Connexion==GRID_ON):          
        annual_CO2eq_DG = 0.

    annual_eqCO2emissions = annual_CO2eq_DG+annual_CO2eq_prod+annual_CO2eq_importation+annual_CO2eq_storage
    gene.fitness = Compute_fitness(global_parameters,obtained_constraint_level, annual_eqCO2emissions) 

    return(gene)

@jit(nopython=True)
def Fossil_fuel_consumption_pro(gene,RENSystems_parameters,global_parameters,Genset_parameters,grid_parameters,pro_parameters):
    """
    Evaluates the fossil fuel consumption fitness of a candidate microgrid configuration.
    
    Fossil fuel consumption fitness computes the total annual consumption of all types of fossil fuels
    and applies a constraint penalty if required.
    
    See 'LCOE_pro' for full attributes description.
    
    Returns:
        object: Updated `gene` object with computed fossil fuel consumption fitness.
    
    Note:
        - Fitness penalizes candidate solutions that violate `constraint_level`.
        - Uses `Eems.LFE_CCE` to simulate storage dispatch and DSM.
        - Updates `gene.storages` in-place.
    
    Important:
        - Penalization applies only if `obtained_constraint_level < constraint_level`.
    
    Warning:
        - Divisions by zero may occur if `sum(Optimized_Load) == 0`. Ensure non-zero total load.
    """  
    (production, Optimized_Load,trades,storage_TS,D_DSM,Y_DSM,SOCs_eff,losses,P_diff) = pro_cost_base_indicators(gene,RENSystems_parameters,global_parameters,pro_parameters)
    
    gene.storages = pro_update_storage_power(gene,RENSystems_parameters,storage_TS)        
  
        
    importation=np.where(trades>0,trades,0)
    den_Optimized_load = max(1e-15,sum(Optimized_Load))
    obtained_constraint_level = get_constraint_level(global_parameters,importation,den_Optimized_load,trades,production)

    if (global_parameters.Connexion==GRID_ON) : 
        Grid_importation = importation 
        annual_fossil_fuel_consumption_importation =  grid_parameters.fossil_fuel_ratio*sum(Grid_importation)/global_parameters.time_resolution/global_parameters.duration_years
    else : 
        annual_fossil_fuel_consumption_importation = 0
    
    if (global_parameters.Connexion==GRID_OFF):
        DG_nominal_power = max(trades)
        DG_production=importation
        if (DG_nominal_power>0):
            closest_levels = np.array([int(10*(DG_production[i]/DG_nominal_power)-0.5) for i in range(global_parameters.n_bits)])
            onsite_annual_fuel_consumption = sum(DG_production*Genset_parameters.fuel_CO2eq_emissions[closest_levels]/global_parameters.time_resolution/global_parameters.duration_years)
        else : 
            onsite_annual_fuel_consumption = 0    
    elif (global_parameters.Connexion==GRID_ON):          
        onsite_annual_fuel_consumption = 0
    
    annual_fossil_fuel_consumption = annual_fossil_fuel_consumption_importation+onsite_annual_fuel_consumption
    gene.fitness = Compute_fitness(global_parameters,obtained_constraint_level, annual_fossil_fuel_consumption) 

    return(gene)


@jit(nopython=True)
def EROI_pro(gene,RENSystems_parameters,global_parameters,Genset_parameters,grid_parameters,pro_parameters):
    """
    Evaluates the EROI (Energy Return on Investment) fitness of a candidate microgrid configuration.
    
    EROI fitness computes the total energy production over the total energy consumption
    over the system's lifetime and applies a constraint penalty if required.
    
    See 'LCOE_pro' for full attributes description.
    
    Returns:
        object: Updated `gene` object with computed EROI fitness.
    
    Note:
        - Fitness penalizes candidate solutions that violate `constraint_level`.
        - Uses `Eems.LFE_CCE` to simulate storage dispatch and DSM.
        - Updates `gene.storages` in-place.
    
    Important:
        - Penalization applies only if `obtained_constraint_level < constraint_level`.
    
    Warning:
        - Divisions by zero may occur if `sum(Optimized_Load) == 0`. Ensure non-zero total load.
    """ 
    (production, Optimized_Load,trades,storage_TS,D_DSM,Y_DSM,SOCs_eff,losses,P_diff) = pro_cost_base_indicators(gene,RENSystems_parameters,global_parameters,pro_parameters)
    gene.storages = pro_update_storage_power(gene,RENSystems_parameters,storage_TS)          
    importation=np.where(trades>0,trades,0)
    den_Optimized_load = max(1e-15,sum(Optimized_Load))
    obtained_constraint_level = get_constraint_level(global_parameters,importation,den_Optimized_load,trades,production)

    energy_storages,Lifetime = pro_indicators_storage(gene,RENSystems_parameters,global_parameters,storage_TS,losses)
    Equivalent_cycles =  np.array([np.sum(np.abs(storage_TS[i,:]))/(2*global_parameters.time_resolution*max(energy_storages[i],np.float64(1e-15))*global_parameters.duration_years) for i in range(RENSystems_parameters.n_store)]) 

    productible_energy = sum(np.sum(np.multiply(gene.production_set,RENSystems_parameters.unit_productions.T),axis=0)/global_parameters.time_resolution/global_parameters.duration_years*RENSystems_parameters.specs_prod[:,PROD_LIFETIME])
    consumed_energy_production = sum(productible_energy/RENSystems_parameters.specs_prod[:,PROD_EROI])
    consumed_energy_storage = sum(np.array([np.nanmin((RENSystems_parameters.specs_storage[STOR_CYCLE_LIFE,i],RENSystems_parameters.specs_storage[STOR_LIFETIME,i]*Equivalent_cycles[i]))*energy_storages[i]/RENSystems_parameters.specs_storage[STOR_ESOEI,i] for i in range(RENSystems_parameters.n_store)]))

    if (global_parameters.Connexion==GRID_OFF):
        DG_production=importation
        DG_lifetime_years = Genset_parameters.lifetime/(sum(np.where(DG_production>0,1,0))/global_parameters.time_resolution/global_parameters.duration_years)
        productible_energy_DG = sum(DG_production)/global_parameters.time_resolution/global_parameters.duration_years*DG_lifetime_years
        consumed_energy_DG = productible_energy_DG/Genset_parameters.EROI
        
    elif (global_parameters.Connexion==GRID_ON):          
        productible_energy_DG = 0.
        consumed_energy_DG = 0.
        
    EROI = (productible_energy+productible_energy_DG)/(consumed_energy_DG+consumed_energy_production+consumed_energy_storage)
    gene.fitness = Compute_fitness(global_parameters,obtained_constraint_level, -EROI) 

    return(gene)

####  RESEARCH COST FUNCTIONS ##################

def KPI_research(gene,datetime,storage_characteristics,time_resolution,n_store,duration_years,specs_num,prices_num,fixed_premium,Overrun,Selling_price,Non_movable_load,total_D_Movable_load,D_Movable_load, total_Y_Movable_load ,Y_Movable_load ,Grid_fossil_fuel_ratio,Main_grid_PoF_ratio,Main_grid_emissions,prod_C,prods_U,Bounds_prod,constraint_num,constraint_level,cost_constraint,n_bits,Connexion,DG_fuel_consumption,DG_fuel_cost,DG_unit_cost,DG_lifetime,DG_maintenance_cost,DG_EROI,fuel_CO2eq_emissions,storage_techs,n_days):
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
    KILOS_CONVERSION_FACTOR = 1000
    TONS_CONVERSION_FACTOR = 1000000
    production = ((prods_U.T*gene.production_set).sum(axis=1)+prod_C)/KILOS_CONVERSION_FACTOR    

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
        
    annual_CO2eq_prod = sum(sum(np.multiply(np.array([gene.production_set[i]*prods_U[i,:] for i in range(len(gene.production_set))]).T/KILOS_CONVERSION_FACTOR,np.array(specs_num[:,4]))))/TONS_CONVERSION_FACTOR/time_resolution/duration_years
    annual_CO2eq_importation = sum(Grid_importation)*Main_grid_emissions/TONS_CONVERSION_FACTOR/time_resolution/duration_years
    annual_CO2eq_storage = np.inner(np.array([sum(np.where(storage_TS[i]>0,storage_TS[i],0)) for i in range(n_store)]),storage_characteristics[6,:])/TONS_CONVERSION_FACTOR/time_resolution/duration_years

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
        annual_CO2eq_DG = annual_fuel_consumption*fuel_CO2eq_emissions  /TONS_CONVERSION_FACTOR  
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
def research_cost_base_indicators(gene,RENSystems_parameters,global_parameters):
    KILOS_CONVERSION_FACTOR = 1000
    production = ((RENSystems_parameters.unit_productions.T*gene.production_set).sum(axis=1)+RENSystems_parameters.current_production)/KILOS_CONVERSION_FACTOR    
    Optimized_Load = global_parameters.Non_movable_load+gene.Y_DSM+gene.D_DSM.flatten()   
    power_storage = np.sum(gene.storage_TS,axis=0)
    trades = Optimized_Load-production-power_storage
    return(production, Optimized_Load,trades)

@jit(nopython=True)
def compute_losses(gene,RENSystems_parameters):
    losses = np.transpose(np.divide(np.transpose(gene.storage_TS),RENSystems_parameters.specs_storage[STOR_ROUND_TRIP_EFF,:]))-gene.storage_TS 
    losses=np.where(losses>0,losses,0)
    return(losses)

@jit(nopython=True)
def size_storage_power(gene,RENSystems_parameters):
    powers_out = [np.max(gene.storage_TS[i,:]) for i in range(RENSystems_parameters.n_store)]
    powers_in = [-np.min(gene.storage_TS[i,:]) for i in range(RENSystems_parameters.n_store)]
    size_power=np.array([max(powers_in[i],powers_out[i]) for i in range(RENSystems_parameters.n_store)])
    return(size_power)

@jit(nopython=True)
def Indicators_storage(gene,RENSystems_parameters,global_parameters,losses):
    sum_diff_storages = [np.cumsum(gene.storage_TS[i,:]+losses[i,:])/global_parameters.time_resolution for i in range(RENSystems_parameters.n_store)]
    energy_storages = np.divide(np.array([np.max(sum_diff_storages[i]) - np.min(sum_diff_storages[i]) for i in range(RENSystems_parameters.n_store)]),RENSystems_parameters.specs_storage[STOR_DEPTH_OF_DISCHARGE,:])
    Equivalent_cycles =  np.array([np.sum(np.abs(gene.storage_TS[i,:]))/(2*global_parameters.time_resolution*max(energy_storages[i],np.float64(1e-15))*global_parameters.duration_years) for i in range(RENSystems_parameters.n_store)]) 
    Lifetime = np.array([min(RENSystems_parameters.specs_storage[STOR_LIFETIME,i],RENSystems_parameters.specs_storage[STOR_CYCLE_LIFE,i]/max(1e-15,Equivalent_cycles[i])) for i in range(RENSystems_parameters.n_store)])
    return(energy_storages,Lifetime)

@jit(nopython=True)
def Indicators_Genset(gene,Genset_parameters,global_parameters,trades,importation):
    DG_nominal_power = max(trades)
    DG_production=importation
    if (DG_nominal_power>0):
        closest_levels = np.array([int(10*(DG_production[i]/DG_nominal_power)-0.5) for i in range(global_parameters.n_bits)])
        annual_fuel_consumption = sum(DG_production*Genset_parameters.fuel_consumption[closest_levels]/global_parameters.time_resolution/global_parameters.duration_years)
    else : 
        annual_fuel_consumption = 0
    return(DG_nominal_power,DG_production,annual_fuel_consumption)

@jit(nopython=True)
def Economics_Genset(annual_fuel_consumption,DG_production,DG_nominal_power,Genset_parameters,global_parameters):
    if (sum(DG_production>0)):
        annual_total_fuel_cost = annual_fuel_consumption*Genset_parameters.fuel_cost
        DG_lifetime_years = Genset_parameters.lifetime/(sum(np.where(DG_production>0,1,0))/global_parameters.time_resolution/global_parameters.duration_years)
        DG_CAPEX_cost = DG_nominal_power*Genset_parameters.unit_cost/DG_lifetime_years
        DG_OPEX_cost = sum(DG_production)*Genset_parameters.maintenance_cost/global_parameters.time_resolution/global_parameters.duration_years
    else :
        (DG_lifetime_years,annual_total_fuel_cost,DG_CAPEX_cost,DG_OPEX_cost) = (np.nan,0,0,0)
    
    return(DG_lifetime_years,annual_total_fuel_cost,DG_CAPEX_cost,DG_OPEX_cost)

@jit(nopython=True)
def Economics_Grid(gene,global_parameters,grid_parameters,importation,exportation):
    Contract_power=max(0,max(importation))
    economics_exportation = np.multiply(exportation,grid_parameters.Selling_price[gene.contract,:]).sum()/global_parameters.time_resolution/global_parameters.duration_years
    economics_importation = np.multiply(importation,grid_parameters.prices[gene.contract,:]).sum()/global_parameters.time_resolution/global_parameters.duration_years
    economics_overrun = max(0,(max(importation)-Contract_power)*grid_parameters.Overrun[gene.contract])
    economics_contract_power = grid_parameters.fixed_premium[gene.contract]*Contract_power
    return(economics_exportation,economics_importation,economics_overrun,economics_contract_power)

@jit(nopython=True)
def Economics_Storage(gene,energy_storages,RENSystems_parameters,Lifetime,size_power):
    CAPEX_storage_cost =  np.multiply(size_power,RENSystems_parameters.specs_storage[STOR_POWER_COST,:]) + np.multiply(energy_storages,RENSystems_parameters.specs_storage[STOR_ENERGY_COST,:]) + np.multiply(RENSystems_parameters.specs_storage[STOR_INSTALLATION_COST,:],(size_power>np.repeat(0,RENSystems_parameters.n_store)))
    economics_CAPEX_storage = sum(np.divide(CAPEX_storage_cost,Lifetime))
    economics_OPEX_storage = sum(np.multiply(RENSystems_parameters.specs_storage[STOR_OM_COST,:],size_power))
    return(economics_CAPEX_storage,economics_OPEX_storage)

@jit(nopython=True)
def LCOE_research(gene,RENSystems_parameters,global_parameters,Genset_parameters,grid_parameters):
    """
    Computes the Levelized Cost of Energy (LCOE) and fitness for a candidate microgrid configuration.
    
    This function evaluates the economic performance of a candidate solution (`gene`) by combining:
    production costs, storage CAPEX/OPEX, diesel generator costs, and grid interactions. Fitness
    penalizes solutions that violate specified constraints. It returns the fitness value and
    the net power exchanges (trades). LCOE is computed as the ratio of total annualized costs
    to the annual energy supplied.
    
    Parameters:
        gene (object): Object containing decision variables (production_set, storage_TS, Y_DSM, D_DSM, contract).
        storage_characteristics (numpy.ndarray): Technical and economic parameters of storage systems.
        time_resolution (float): Number of simulation timesteps per hour.
        n_store (int): Number of storage units.
        duration_years (float): Duration of the simulation horizon in years.
        specs_num (numpy.ndarray): Technical/economic parameters of production units.
        prices_num (numpy.ndarray): Electricity import prices (€/kWh).
        fixed_premium (numpy.ndarray): Fixed premium for contracted grid power.
        Overrun (numpy.ndarray): Penalty for exceeding contracted grid power.
        Selling_price (numpy.ndarray): Electricity export prices (€/kWh).
        Non_movable_load (numpy.ndarray): Non-controllable load time series (kW).
        Main_grid_emissions (float): CO2eq intensity of grid electricity (g/kWh) (unused here).
        DG_fuel_consumption (numpy.ndarray): Diesel generator fuel consumption lookup table.
        DG_fuel_cost (float): Diesel fuel cost (€/unit).
        DG_lifetime (float): Diesel generator lifetime (hours).
        DG_unit_cost (float): Diesel generator CAPEX (€/kW).
        DG_maintenance_cost (float): Diesel generator OPEX (€/kWh).
        DG_EROI (float): Energy return on investment of the diesel generator.
        prod_C (float): Constant production term (kW).
        prods_U (numpy.ndarray): Unit production profiles.
        Bounds_prod (numpy.ndarray): Production bounds (unused here).
        constraint_num (int): Identifier for the applied constraint type.
        constraint_level (float): Target constraint threshold.
        cost_constraint (float): Penalty coefficient applied if constraint violated.
        Connexion (str): System connection mode ('On-grid' or 'Off-grid').
        fuel_CO2eq_emissions (float): Fuel emissions factor (unused here).
        Grid_fossil_fuel_ratio (float): Fossil fuel share of grid electricity (unused here).
    
    Returns:
        tuple:
            - fitness (float): Objective value used for optimization (penalized negative LCOE).
            - trade (numpy.ndarray): Net power exchange with grid or DG (kW).
    
    Notes:
        - Penalization is applied if `obtained_constraint_level < constraint_level`.
        - The function updates `gene.storages` in-place.
        - LCOE includes all relevant costs: production, storage, DG, grid interactions.
        - Ensure `n_store` matches the dimensions of `storage_characteristics`.
    """
    
    (production, Optimized_Load,trades)=research_cost_base_indicators(gene,RENSystems_parameters,global_parameters)
    losses = compute_losses(gene,RENSystems_parameters)
    size_power = size_storage_power(gene,RENSystems_parameters)
    
    annual_cost_production = np.sum((RENSystems_parameters.specs_prod[:,PROD_CAPEX]/RENSystems_parameters.specs_prod[:,PROD_LIFETIME]+RENSystems_parameters.specs_prod[:,PROD_OPEX])*(gene.production_set))
    
    (energy_storages,Lifetime) = Indicators_storage(gene,RENSystems_parameters,global_parameters,losses)
   
    importation=np.where(trades>0,trades,0)
    exportation = np.where(trades<0,trades,0)
    
    (economics_CAPEX_storage,economics_OPEX_storage)=Economics_Storage(gene,energy_storages,RENSystems_parameters,Lifetime,size_power)

    
    if (global_parameters.Connexion==GRID_OFF):
        (DG_nominal_power,DG_production,annual_fuel_consumption) = Indicators_Genset(gene,Genset_parameters,global_parameters,trades,importation)
        (DG_lifetime_years,annual_total_fuel_cost,DG_CAPEX_cost,DG_OPEX_cost) = Economics_Genset(annual_fuel_consumption,DG_production,DG_nominal_power,Genset_parameters,global_parameters)
        economics_exportation = 0
        economics_importation = 0
        economics_overrun = 0
        economics_contract_power = 0
        
        
    elif (global_parameters.Connexion==GRID_ON):          
        annual_total_fuel_cost = 0
        DG_CAPEX_cost = 0
        DG_OPEX_cost = 0
        (economics_exportation,economics_importation,economics_overrun,economics_contract_power) = Economics_Grid(gene,global_parameters,grid_parameters,importation,exportation)
        
    
    LCOE=(annual_cost_production+economics_CAPEX_storage+economics_OPEX_storage+DG_OPEX_cost+DG_CAPEX_cost+annual_total_fuel_cost+economics_importation+economics_contract_power+economics_overrun+economics_exportation)/(sum(Optimized_Load)/global_parameters.time_resolution/global_parameters.duration_years)   if sum(Optimized_Load )>0 else np.nan

    den_Optimized_load = max(1e-15,sum(Optimized_Load))   
    obtained_constraint_level = get_constraint_level(global_parameters,importation,den_Optimized_load,trades,production)
    fitness = Compute_fitness(global_parameters,obtained_constraint_level,LCOE)

    return(fitness,trades)

@jit(nopython=True)
def Annual_net_benefits_research(gene,RENSystems_parameters,global_parameters,Genset_parameters,grid_parameters):
    """
    Evaluates the Annual net benefits fitness of a candidate microgrid configuration.
    
    Annual net benefits evaluates the total annual income.
    Penalizes violations of operational constraints.
    
    See 'LCOE_research' for full attributes description.
    
    Returns:
        tuple: Tuple containing:
            - fitness (float): Objective value used for optimization (penalized negative Annual net benefits).
            - trades (numpy.ndarray): Net power exchange with grid or DG (kW).
    
    Notes:
        - Fitness penalizes candidate solutions that violate `constraint_level`.
        - Self-sufficiency is computed as the fraction of load met by local generation.
    
    Important:
        Penalization applies only if `obtained_constraint_level < constraint_level`.
    
    Warnings:
        Divisions by zero may occur if `sum(Optimized_Load) == 0`. Ensure non-zero total load.
    """
    (production, Optimized_Load,trades)=research_cost_base_indicators(gene,RENSystems_parameters,global_parameters)

    annual_cost_production = np.sum((RENSystems_parameters.specs_prod[:,PROD_CAPEX]/RENSystems_parameters.specs_prod[:,PROD_LIFETIME]+RENSystems_parameters.specs_prod[:,PROD_OPEX])*(gene.production_set))
    losses = compute_losses(gene,RENSystems_parameters)
    size_power = size_storage_power(gene,RENSystems_parameters)
  
    (energy_storages,Lifetime) = Indicators_storage(gene,RENSystems_parameters,global_parameters,losses)

    importation=np.where(trades>0,trades,0)
    exportation = np.where(trades<0,-trades,0)

    (economics_CAPEX_storage,economics_OPEX_storage)=Economics_Storage(gene,energy_storages,RENSystems_parameters,Lifetime,size_power)

    if (global_parameters.Connexion==GRID_OFF):
        (DG_nominal_power,DG_production,annual_fuel_consumption) = Indicators_Genset(gene,Genset_parameters,global_parameters,trades,importation)
        (DG_lifetime_years,annual_total_fuel_cost,DG_CAPEX_cost,DG_OPEX_cost) = Economics_Genset(annual_fuel_consumption,DG_production,DG_nominal_power,Genset_parameters,global_parameters)
        economics_exportation = 0
        economics_importation = 0
        economics_overrun = 0
        economics_contract_power = 0
        
    elif (global_parameters.Connexion==GRID_ON):          
        DG_nominal_power = 0
        annual_fuel_consumption = 0
        DG_CAPEX_cost = 0
        DG_OPEX_cost = 0
        (economics_exportation,economics_importation,economics_overrun,economics_contract_power) = Economics_Grid(gene,global_parameters,grid_parameters,importation,exportation)
        
    den_Optimized_load = max(1e-15,sum(Optimized_Load))   
    obtained_constraint_level = get_constraint_level(global_parameters,importation,den_Optimized_load,trades,production)

    Annual_net_benefits = (-annual_cost_production-economics_CAPEX_storage-economics_OPEX_storage-DG_OPEX_cost-DG_CAPEX_cost-annual_total_fuel_cost-economics_importation-economics_contract_power-economics_overrun+economics_exportation)
    fitness = Compute_fitness(global_parameters,obtained_constraint_level, -Annual_net_benefits)

    return(fitness,trades)

@jit(nopython=True)
def NPV_research(gene,RENSystems_parameters,global_parameters,Genset_parameters,grid_parameters):
    """
    Evaluates the NPV (net present value) fitness of a candidate microgrid configuration.
    
    Net present value evaluates the economic evaluation of a project over its full expected lifetime.
    Penalizes violations of operational constraints.
    
    See 'LCOE_research' for full attributes description.
    
    Returns:
        tuple: Tuple containing:
            - fitness (float): Objective value used for optimization (penalized negative NPV).
            - trades (numpy.ndarray): Net power exchange with grid or DG (kW).
    
    Notes:
        - Fitness penalizes candidate solutions that violate `constraint_level`.
        - Self-sufficiency is computed as the fraction of load met by local generation.
    
    Important:
        Penalization applies only if `obtained_constraint_level < constraint_level`.
    
    Warnings:
        Divisions by zero may occur if `sum(Optimized_Load) == 0`. Ensure non-zero total load.
    """
    (production, Optimized_Load,trades)=research_cost_base_indicators(gene,RENSystems_parameters,global_parameters)

    annual_cost_production = np.sum((RENSystems_parameters.specs_prod[:,PROD_CAPEX]/RENSystems_parameters.specs_prod[:,PROD_LIFETIME]+RENSystems_parameters.specs_prod[:,PROD_OPEX])*(gene.production_set))
    losses = compute_losses(gene,RENSystems_parameters)

    size_power = size_storage_power(gene,RENSystems_parameters) 
    
    (energy_storages,Lifetime) = Indicators_storage(gene,RENSystems_parameters,global_parameters,losses)
    importation=np.where(trades>0,trades,0)
    exportation = np.where(trades<0,-trades,0)

    (economics_CAPEX_storage,economics_OPEX_storage)=Economics_Storage(gene,energy_storages,RENSystems_parameters,Lifetime,size_power)

    if (global_parameters.Connexion==GRID_OFF):
        (DG_nominal_power,DG_production,annual_fuel_consumption) = Indicators_Genset(gene,Genset_parameters,global_parameters,trades,importation)
        (DG_lifetime_years,annual_total_fuel_cost,DG_CAPEX_cost,DG_OPEX_cost) = Economics_Genset(annual_fuel_consumption,DG_production,DG_nominal_power,Genset_parameters,global_parameters)
        economics_exportation = 0
        economics_importation = 0
        economics_overrun = 0
        economics_contract_power = 0
        
    elif (global_parameters.Connexion==GRID_ON):          
        DG_nominal_power = 0
        DG_CAPEX_cost = 0
        DG_OPEX_cost = 0
        (economics_exportation,economics_importation,economics_overrun,economics_contract_power) = Economics_Grid(gene,global_parameters,grid_parameters,importation,exportation)
    
    den_Optimized_load = max(1e-15,sum(Optimized_Load))   
    obtained_constraint_level = get_constraint_level(global_parameters,importation,den_Optimized_load,trades,production)

    Lifetime_installation = min(DG_lifetime_years,min(Lifetime),np.nanmin(np.where(gene.production_set>0,RENSystems_parameters.specs_prod[:,PROD_LIFETIME],np.nan)))
    Annual_net_benefits = (-annual_cost_production-economics_CAPEX_storage-economics_OPEX_storage-DG_OPEX_cost-DG_CAPEX_cost-annual_total_fuel_cost-economics_importation-economics_contract_power-economics_overrun+economics_exportation)
    NPV=Annual_net_benefits*Lifetime_installation
    fitness = Compute_fitness(global_parameters,obtained_constraint_level, -NPV)
    return(fitness,trades)

@jit(nopython=True)
def Max_import_power_research(gene,RENSystems_parameters,global_parameters,Genset_parameters,grid_parameters):
    """
    Evaluates the max import fitness of a candidate microgrid configuration.
    
    Max import power evaluates the maximum positive power exchange (`trades > 0`).
    Penalizes violations of operational constraints.
    
    See 'LCOE_research' for full attributes description.
    
    Returns:
        tuple: Tuple containing:
            - fitness (float): Objective value used for optimization (penalized negative Max import).
            - trades (numpy.ndarray): Net power exchange with grid or DG (kW).
    
    Notes:
        - Fitness penalizes candidate solutions that violate `constraint_level`.
        - Self-sufficiency is computed as the fraction of load met by local generation.
    
    Important:
        Penalization applies only if `obtained_constraint_level < constraint_level`.
    
    Warnings:
        Divisions by zero may occur if `sum(Optimized_Load) == 0`. Ensure non-zero total load.
    """
    (production, Optimized_Load,trades)=research_cost_base_indicators(gene,RENSystems_parameters,global_parameters)

    importation=np.where(trades>0,trades,0)

    Max_import = max(importation) 
    
    den_Optimized_load = max(1e-15,sum(Optimized_Load))   
    obtained_constraint_level = get_constraint_level(global_parameters,importation,den_Optimized_load,trades,production)
    fitness = Compute_fitness(global_parameters,obtained_constraint_level, Max_import)    
    return(fitness,trades)


@jit(nopython=True)
def Losses_research(gene,RENSystems_parameters,global_parameters,Genset_parameters,grid_parameters):
    """
    Evaluates the losses fitness of a candidate microgrid configuration.
    
    Losses represents the total annual emissions of greenhouse gases.
    Penalizes violations of operational constraints.
    
    See 'LCOE_research' for full attributes description.
    
    Returns:
        tuple: Tuple containing:
            - fitness (float): Objective value used for optimization (penalized negative Losses).
            - trades (numpy.ndarray): Net power exchange with grid or DG (kW).
    
    Notes:
        - Fitness penalizes candidate solutions that violate `constraint_level`.
        - Self-sufficiency is computed as the fraction of load met by local generation.
    
    Important:
        Penalization applies only if `obtained_constraint_level < constraint_level`.
    
    Warnings:
        Divisions by zero may occur if `sum(Optimized_Load) == 0`. Ensure non-zero total load.
    """
    (production, Optimized_Load,trades)=research_cost_base_indicators(gene,RENSystems_parameters,global_parameters)

    losses = compute_losses(gene,RENSystems_parameters)

                       
    Annual_sum_losses = np.sum(losses,axis=1)/global_parameters.time_resolution/global_parameters.duration_years 
    importation=np.where(trades>0,trades,0)

    den_Optimized_load = max(1e-15,sum(Optimized_Load))   
    obtained_constraint_level = get_constraint_level(global_parameters,importation,den_Optimized_load,trades,production)   
    fitness = Compute_fitness(global_parameters,obtained_constraint_level, Annual_sum_losses)    
    
    return(fitness,trades)

@jit(nopython=True)
def eqCO2_emissions_research(gene,RENSystems_parameters,global_parameters,Genset_parameters,grid_parameters):
    """
    Evaluates the eqCO2 emissions fitness of a candidate microgrid configuration.
    
    eqCO2 emissions represents the total annual emissions of greenhouse gases.
    Penalizes violations of operational constraints.
    
    See 'LCOE_research' for full attributes description.
    
    Returns:
        tuple: Tuple containing:
            - fitness (float): Objective value used for optimization (penalized negative eqCO2 emissions).
            - trades (numpy.ndarray): Net power exchange with grid or DG (kW).
    
    Notes:
        - Fitness penalizes candidate solutions that violate `constraint_level`.
        - Self-sufficiency is computed as the fraction of load met by local generation.
    
    Important:
        Penalization applies only if `obtained_constraint_level < constraint_level`.
    
    Warnings:
        Divisions by zero may occur if `sum(Optimized_Load) == 0`. Ensure non-zero total load.
    """    
    KILOS_CONVERSION_FACTOR = 1000
    TONS_CONVERSION_FACTOR = 1000000
    (production, Optimized_Load,trades)=research_cost_base_indicators(gene,RENSystems_parameters,global_parameters)

    

    importation=np.where(trades>0,trades,0)    
    Grid_importation = np.where(trades>0,trades,0) if (global_parameters.Connexion==GRID_ON) else np.repeat(0.,global_parameters.n_bits)
    
    annual_CO2eq_prod = sum(np.sum((gene.production_set*RENSystems_parameters.unit_productions.T)/KILOS_CONVERSION_FACTOR*RENSystems_parameters.specs_prod[:,PROD_EMISSIONS],axis=0))/TONS_CONVERSION_FACTOR/global_parameters.time_resolution/global_parameters.duration_years   
    annual_CO2eq_importation = sum(Grid_importation)*grid_parameters.eqCO2emissions/TONS_CONVERSION_FACTOR/global_parameters.time_resolution/global_parameters.duration_years
    annual_CO2eq_storage = sum(np.array([sum(np.where(gene.storage_TS[i]>0,gene.storage_TS[i],0)) for i in range(RENSystems_parameters.n_store)])[:]*RENSystems_parameters.specs_storage[STOR_EMISSIONS,:])/TONS_CONVERSION_FACTOR/global_parameters.time_resolution/global_parameters.duration_years
    if (global_parameters.Connexion==GRID_OFF):
        (DG_nominal_power,DG_production,annual_fuel_consumption) = Indicators_Genset(gene,Genset_parameters,global_parameters,trades,importation)
        annual_CO2eq_DG = annual_fuel_consumption*Genset_parameters.fuel_CO2eq_emissions/TONS_CONVERSION_FACTOR      
    elif (global_parameters.Connexion==GRID_ON):          
        annual_CO2eq_DG = 0.

    den_Optimized_load = max(1e-15,sum(Optimized_Load))   
    obtained_constraint_level = get_constraint_level(global_parameters,importation,den_Optimized_load,trades,production)  
    annual_CO2eq_total = annual_CO2eq_DG+annual_CO2eq_prod+annual_CO2eq_importation+annual_CO2eq_storage
    fitness = Compute_fitness(global_parameters,obtained_constraint_level, annual_CO2eq_total)    
    return(fitness,trades)

@jit(nopython=True)
def Autonomy_research(gene,RENSystems_parameters,global_parameters,Genset_parameters,grid_parameters):
    """
    Evaluates the autonomy fitness of a candidate microgrid configuration.
    
    Autonomy represents the proportion of time during which the grid operates fully autonomously.
    Penalizes violations of operational constraints.
    
    See 'LCOE_research' for full attributes description.
    
    Returns:
        tuple: Tuple containing:
            - fitness (float): Objective value used for optimization (penalized negative Autonomy).
            - trades (numpy.ndarray): Net power exchange with grid or DG (kW).
    
    Notes:
        - Fitness penalizes candidate solutions that violate `constraint_level`.
        - Self-sufficiency is computed as the fraction of load met by local generation.
    
    Important:
        Penalization applies only if `obtained_constraint_level < constraint_level`.
    
    Warnings:
        Divisions by zero may occur if `sum(Optimized_Load) == 0`. Ensure non-zero total load.
    """
    (production, Optimized_Load,trades)=research_cost_base_indicators(gene,RENSystems_parameters,global_parameters)
    
    importation = np.where(trades>0,trades,0)
    grid_importation = importation if (global_parameters.Connexion==GRID_ON) else np.repeat(0.,global_parameters.n_bits)

    den_Optimized_load = max(1e-15,sum(Optimized_Load))   
    obtained_constraint_level = get_constraint_level(global_parameters,importation,den_Optimized_load,trades,production)  
    Autonomy = 1-sum(grid_importation>0)/global_parameters.n_bits
    fitness = Compute_fitness(global_parameters,obtained_constraint_level, -Autonomy)    
    
    return(fitness,trades)


@jit(nopython=True)
def Fossil_fuel_consumption_research(gene,RENSystems_parameters,global_parameters,Genset_parameters,grid_parameters):
    """
    Evaluates the self-consumption fitness of a candidate microgrid configuration.
    
    Fossil fuel consumption is the total annual consumption of fossil fuels of all types. 
    Penalizes violations of operational constraints.
    
    See 'LCOE_research' for full attributes description.
    
    Returns:
        tuple: Tuple containing:
            - fitness (float): Objective value used for optimization (penalized negative Fossil fuel consumption).
            - trades (numpy.ndarray): Net power exchange with grid or DG (kW).
    
    Notes:
        - Fitness penalizes candidate solutions that violate `constraint_level`.
        - Self-sufficiency is computed as the fraction of load met by local generation.
    
    Important:
        Penalization applies only if `obtained_constraint_level < constraint_level`.
    
    Warnings:
        Divisions by zero may occur if `sum(Optimized_Load) == 0`. Ensure non-zero total load.
    """      
    (production, Optimized_Load,trades)=research_cost_base_indicators(gene,RENSystems_parameters,global_parameters)

    importation=np.where(trades>0,trades,0)
    Grid_importation = importation if (global_parameters.Connexion==GRID_ON) else np.repeat(0.,global_parameters.n_bits)    
    annual_fossil_fuel_consumption_importation =  grid_parameters.fossil_fuel_ratio*sum(Grid_importation)/global_parameters.time_resolution/global_parameters.duration_years

    if (global_parameters.Connexion==GRID_OFF):
        (DG_nominal_power,DG_production,annual_fuel_consumption) = Indicators_Genset(gene,Genset_parameters,global_parameters,trades,importation)
        
    elif (global_parameters.Connexion==GRID_ON):          
        annual_fuel_consumption = 0

    den_Optimized_load = max(1e-15,sum(Optimized_Load))   
    obtained_constraint_level =get_constraint_level(global_parameters,importation,den_Optimized_load,trades,production) 
    annual_fossil_fuel_consumption = annual_fossil_fuel_consumption_importation+annual_fuel_consumption
    fitness = Compute_fitness(global_parameters,obtained_constraint_level, annual_fossil_fuel_consumption)    

    return(fitness,trades)

@jit(nopython=True)
def EROI_research(gene,RENSystems_parameters,global_parameters,Genset_parameters,grid_parameters):
    """
    Evaluates the EROI fitness of a candidate microgrid configuration.
    
    EROI is the total energy production over the total energy consumption over the whole lifetime. 
    Penalizes violations of operational constraints.
    
    See 'LCOE_research' for full attributes description.
    
    Returns:
        tuple: Tuple containing:
            - fitness (float): Objective value used for optimization (penalized negative EROI).
            - trades (numpy.ndarray): Net power exchange with grid or DG (kW).
    
    Notes:
        - Fitness penalizes candidate solutions that violate `constraint_level`.
        - Self-sufficiency is computed as the fraction of load met by local generation.
    
    Important:
        Penalization applies only if `obtained_constraint_level < constraint_level`.
    
    Warnings:
        Divisions by zero may occur if `sum(Optimized_Load) == 0`. Ensure non-zero total load.
    """      
    (production, Optimized_Load,trades)=research_cost_base_indicators(gene,RENSystems_parameters,global_parameters)
    
    losses = compute_losses(gene,RENSystems_parameters)
    
    importation=np.where(trades>0,trades,0)
    
    (energy_storages,Lifetime) = Indicators_storage(gene,RENSystems_parameters,global_parameters,losses)
 
    Equivalent_cycles =  np.array([np.sum(np.abs(gene.storage_TS[i,:]))/(2*global_parameters.time_resolution*max(energy_storages[i],np.float64(1e-15))*global_parameters.duration_years) for i in range(RENSystems_parameters.n_store)]) 

    productible_energy = sum(np.sum(np.multiply(gene.production_set,RENSystems_parameters.current_production.T),axis=0)/global_parameters.time_resolution/global_parameters.duration_years*RENSystems_parameters.specs_prod[:,PROD_LIFETIME])
    consumed_energy_production = sum(productible_energy/RENSystems_parameters.specs_prod[:,PROD_EROI])
    consumed_energy_storage = sum(np.array([np.nanmin((RENSystems_parameters.spec_storage[STOR_CYCLE_LIFE,i],RENSystems_parameters.spec_storage[STOR_LIFETIME,i]*Equivalent_cycles[i]))*energy_storages[i]/RENSystems_parameters.spec_storage[STOR_ESOEI,i] for i in range(RENSystems_parameters.n_store)]))
    
    if (global_parameters.Connexion==GRID_OFF):
        DG_production=importation
        DG_lifetime_years = Genset_parameters.lifetime/(sum(np.where(DG_production>0,1,0))/global_parameters.time_resolution/global_parameters.duration_years)
        productible_energy_DG = sum(DG_production)/global_parameters.time_resolution/global_parameters.duration_years*DG_lifetime_years
        consumed_energy_DG = productible_energy_DG/Genset_parameters.EROI
        
    elif (global_parameters.Connexion==GRID_ON):          
        productible_energy_DG = 0.
        consumed_energy_DG = 0.

    den_Optimized_load = max(1e-15,sum(Optimized_Load))   
    obtained_constraint_level = get_constraint_level(global_parameters,importation,den_Optimized_load,trades,production)
    EROI = (productible_energy+productible_energy_DG)/(consumed_energy_DG+consumed_energy_production+consumed_energy_storage)
    fitness = Compute_fitness(global_parameters,obtained_constraint_level, -EROI)  
    return(fitness,trades)

@jit(nopython=True)
def Self_consumption_research(gene,RENSystems_parameters,global_parameters,Genset_parameters,grid_parameters):
    """
    Evaluates the self-consumption fitness of a candidate microgrid configuration.
    
    Self-consumption is defined as the fraction of locally produced energy directly consumed
    by the loads. Penalizes violations of operational constraints.
    
    See 'LCOE_research' for full attributes description.
    
    Returns:
        tuple: Tuple containing:
            - fitness (float): Objective value used for optimization (penalized negative Self-consumption).
            - trades (numpy.ndarray): Net power exchange with grid or DG (kW).
    
    Notes:
        - Fitness penalizes candidate solutions that violate `constraint_level`.
        - Self-sufficiency is computed as the fraction of load met by local generation.
    
    Important:
        Penalization applies only if `obtained_constraint_level < constraint_level`.
    
    Warnings:
        Divisions by zero may occur if `sum(Optimized_Load) == 0`. Ensure non-zero total load.
    """
    (production, Optimized_Load,trades)=research_cost_base_indicators(gene,RENSystems_parameters,global_parameters)


    exportation = np.where(trades<0,-trades,0)
    importation=np.where(trades>0,trades,0)

    den_Optimized_load = max(1e-15,sum(Optimized_Load))   
    obtained_constraint_level = get_constraint_level(global_parameters,importation,den_Optimized_load,trades,production)
    obtained_self_consumption = (1-sum(exportation)/sum(production))
    fitness = Compute_fitness(global_parameters,obtained_constraint_level, -obtained_self_consumption)  
    
    return(fitness,trades)



@jit(nopython=True)
def Self_sufficiency_research(gene,RENSystems_parameters,global_parameters,Genset_parameters,grid_parameters):
    """
    Evaluates the self-sufficiency fitness of a candidate microgrid configuration.
    
    Self-sufficiency is defined as the fraction of total load met by local generation,
    including storage dispatch and DSM, without reliance on grid import. Penalizes
    violations of operational constraints.
    
    See 'LCOE_research' for full attributes description.
    
    Returns:
        tuple: Tuple containing:
            - fitness (float): Objective value used for optimization (penalized negative Self-sufficiency).
            - trades (numpy.ndarray): Net power exchange with grid or DG (kW).
    
    Notes:
        - Fitness penalizes candidate solutions that violate `constraint_level`.
        - Self-sufficiency is computed as the fraction of load met by local generation.
    
    Important:
        - For Off-grid systems, grid import is assumed zero; for On-grid systems, import affects self-sufficiency.
        - Penalization applies only if `obtained_constraint_level < constraint_level`.
    
    Warnings:
        Divisions by zero may occur if `sum(Optimized_Load) == 0`. Ensure non-zero total load.
    """
    (production, Optimized_Load,trades)=research_cost_base_indicators(gene,RENSystems_parameters,global_parameters)

    importation=np.where(trades>0,trades,0)
    
    Grid_importation = importation if (global_parameters.Connexion==GRID_ON) else np.repeat(0.,global_parameters.n_bits)

    den_Optimized_load = max(1e-15,sum(Optimized_Load))   
    obtained_constraint_level = get_constraint_level(global_parameters,importation,den_Optimized_load,trades,production)
    obtained_self_sufficiency = (1-sum(Grid_importation)/sum(Optimized_Load)) if sum(Optimized_Load )>0 else np.nan
    fitness = Compute_fitness(global_parameters,obtained_constraint_level, -obtained_self_sufficiency)  
   
    return(fitness,trades)
       

