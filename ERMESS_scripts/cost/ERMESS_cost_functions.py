# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 13:11:54 2024

@author: JoPHOBEA
"""
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
class _ProParams:
    """Parameters related to movable loads.

    Attributes:
        total_D_Movable_load (np.ndarray): Daily movable load profile.
        total_Y_Movable_load (float): Yearly movable load total.
    """
    def __init__(self, total_D_Movable_load,total_Y_Movable_load):
        """Initialize movable load parameters.

        Args:
            total_D_Movable_load (np.ndarray): Daily movable load total.
            total_Y_Movable_load (float): Yearly movable load total.
        """
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
class _GlobalParams:
    """Global simulation parameters.

    Attributes:
        n_bits (int): Number of timesteps.
        time_resolution (float): Time resolution.
        duration_years (float): Simulation duration in years.
        constraint_num (int): ID of the constraint.
        constraint_level (float): Constraint threshold.
        cost_constraint (float): Cost factor of the constraint.
        Connexion (str): Grid connection type.
        Non_movable_load (np.ndarray): Fixed load profile.
    """
    def __init__(self, n_bits,time_resolution,duration_years,constraint_num,constraint_level,cost_constraint,Connexion,Non_movable_load):
        """Initialize global parameters.

        Args:
            n_bits (int): Number of timesteps.
            time_resolution (float): Time resolution.
            duration_years (float): Simulation duration in years.
            constraint_num (int): ID of the constraint.
            constraint_level (float): Constraint threshold.
            cost_constraint (float): Cost factor of the constraint.
            Connexion (str): Grid connection type.
            Non_movable_load (np.ndarray): Fixed load profile.
        """
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
class _GridParams:
    """Grid-related parameters.

    Attributes:
        prices (np.ndarray): Energy purchase prices.
        fixed_premium (np.ndarray): Fixed premium costs.
        Overrun (np.ndarray): Overrun penalties.
        Selling_price (np.ndarray): Energy selling prices.
        eqCO2emissions (float): CO2 equivalent emissions.
        fossil_fuel_ratio (float): Fossil fuel share.
    """
    def __init__(self, prices, fixed_premium, Overrun, Selling_price, eqCO2emissions, fossil_fuel_ratio):

        self.prices = prices
        self.fixed_premium = fixed_premium
        self.Overrun = Overrun
        self.Selling_price = Selling_price
        self.eqCO2emissions = eqCO2emissions
        self.fossil_fuel_ratio = fossil_fuel_ratio
        
@jitclass(specs_grid)
class _GridParamsDummy:
    """Dummy grid parameters (default zero values)."""
    def __init__(self):
        """Initialize dummy grid parameters."""
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
class _RENSystemsParams:
    """Renewable energy systems parameters.

    Attributes:
        current_production (np.ndarray): Current production timeseries.
        unit_productions (np.ndarray): Production per unit timeseries.
        groups_production (List[np.ndarray]): Exclusive producer indices.
        specs_prod (np.ndarray): Producer specifications.
        capacities (np.ndarray): Installation capacities.
        specs_storage (np.ndarray): Storage specifications.
        n_store (int): Maximum number of storage units.
    """
    def __init__(self, current_production, unit_productions,groups_production, specs_prod, capacities, specs_storage, n_store):
        """Initialize renewable systems parameters."""
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
class _GensetParams:
    """Generator (genset) parameters.

    Attributes:
        fuel_cost (float): Fuel cost.
        lifetime (float): Lifetime of the generator.
        unit_cost (float): Investment cost.
        maintenance_cost (float): Maintenance cost.
        fuel_consumption (np.ndarray): Fuel consumption.
        fuel_CO2eq_emissions (float): CO2 emissions factor.
        EROI (float): Energy Return On Investment.
    """
    def __init__(self, fuel_cost, lifetime, unit_cost,maintenance_cost, fuel_consumption, fuel_CO2eq_emissions, EROI):
        """Initialize generator parameters."""

        self.fuel_cost = fuel_cost
        self.lifetime = lifetime
        self.unit_cost = unit_cost
        self.maintenance_cost = maintenance_cost
        self.fuel_consumption = fuel_consumption
        self.fuel_CO2eq_emissions = fuel_CO2eq_emissions
        self.EROI = EROI
        
@jitclass(spec_genset)
class _GensetParamsDummy:
    """Dummy generator parameters (default zero values)."""
    def __init__(self):
        self.fuel_cost = np.float64(0)
        self.lifetime = np.float64(0)
        self.unit_cost = np.float64(0)
        self.maintenance_cost = np.float64(0)
        self.fuel_consumption = np.zeros(1)
        self.fuel_CO2eq_emissions = np.float64(0)
        self.EROI = np.float64(0)



####  PRO COST FUNCTIONS ##################




@jit(nopython=True)
def _get_constraint_level(global_parameters,importation,den_Optimized_load,trades,production):
    """
    Compute the constraint level based on the selected optimisation criterion.
    
    Notes:
        - Self-sufficiency: ratio of local supply vs imports.
        - Self-consumption: fraction of production locally consumed.
        - Renewable fraction: ratio of production over load.
        - Returns 0 if undefined (e.g., zero production).
    """
    return(1-(sum(importation)/den_Optimized_load) if global_parameters.constraint_num==CONS_Self_sufficiency else (1+(sum(np.where(trades<0,trades,0))/sum(production)) if sum(production)>0 else 0) if (global_parameters.constraint_num==CONS_Self_consumption) else sum(production)/den_Optimized_load if(global_parameters.constraint_num==CONS_REN_fraction) else 0)
   
@jit(nopython=True)
def _pro_annual_cost_production(gene,RENSystems_parameters):
    """
    Compute annualized production system cost.
    """
    return(np.sum((RENSystems_parameters.specs_prod[:,PROD_CAPEX]/RENSystems_parameters.specs_prod[:,PROD_LIFETIME]+RENSystems_parameters.specs_prod[:,PROD_OPEX])*(gene.production_set)))

@jit(nopython=True)
def _pro_cost_base_indicators(gene,RENSystems_parameters,global_parameters,pro_parameters):
    """
    Compute base indicators for production-oriented optimisation.
    
    Notes:
        - Uses EMS solver (`Eems.LFE_CCE`) for system balancing.
    """
    KILOS_CONVERSION_FACTOR = 1000
    production = ((RENSystems_parameters.unit_productions.T*gene.production_set).sum(axis=1)+RENSystems_parameters.current_production)/KILOS_CONVERSION_FACTOR   
    (storage_TS,trades,D_DSM,Y_DSM,SOCs_eff,losses,P_diff) = Eems.LFE_CCE(gene, global_parameters, pro_parameters, production ,RENSystems_parameters)
    Optimized_Load = global_parameters.Non_movable_load+Y_DSM+D_DSM
    return(production, Optimized_Load,trades,storage_TS,D_DSM,Y_DSM,SOCs_eff,losses,P_diff)

@jit(nopython=True)
def _pro_update_storage_power(gene,RENSystems_parameters,storage_TS):
    """
    Update storage power capacities based on operation.
    
    This function updates charging and discharging power values in the
    gene structure using observed storage time series.
    
    Notes:
        - Charging power is stored as a positive value.
        - Discharging power is the maximum positive output.
    """
    for i in range(RENSystems_parameters.n_store):
        gene.storages[INDIV_PRO_CHARGE_POWER][i]=max(-(storage_TS[i]))   
        gene.storages[INDIV_PRO_DISCHARGE_POWER][i]=max(storage_TS[i])  
    return(gene.storages)

@jit(nopython=True)
def _pro_size_storage_power(gene,RENSystems_parameters):
    """
    Compute required storage power capacity.
    
    This function determines the maximum required power for each storage unit
    based on charge and discharge profiles.
    
    Args:
        gene:
            Individual solution containing storage parameters.
    
        RENSystems_parameters:
            Object containing number of storage units.
    
    Returns:
        np.ndarray:
            Storage power capacity (kW) for each unit.
    """
    powers_out = gene.storages[INDIV_PRO_DISCHARGE_POWER,:]
    powers_in = gene.storages[INDIV_PRO_CHARGE_POWER,:]
    size_power=np.array([max(powers_in[i],powers_out[i]) for i in range(RENSystems_parameters.n_store)])
    return(size_power)



@jit(nopython=True)
def _pro_indicators_storage(gene,RENSystems_parameters,global_parameters,storage_TS,losses):
    """
    Compute storage energy capacity and lifetime indicators.
    
    Args:
        gene:
            Individual solution containing storage sizing.
    
        RENSystems_parameters:
            Object containing storage specifications.
    
        global_parameters:
            Object containing simulation parameters.
    
        storage_TS (np.ndarray):
            Storage power time series (kW).
    
        losses (np.ndarray):
            Storage losses.
    
    Returns:
        tuple:
            - energy_storages (np.ndarray): Storage capacities (kWh)
            - Lifetime (np.ndarray): Storage lifetimes (years)
    
    Notes:
        - Energy capacity is derived from volume and depth of discharge.
        - Lifetime is limited by cycle life or calendar life.
    """
    energy_storages = gene.storages[INDIV_PRO_VOLUME,:]/RENSystems_parameters.specs_storage[STOR_DEPTH_OF_DISCHARGE,:]
    Equivalent_cycles =  np.array([np.sum(np.abs(storage_TS[i,:]))/(2*global_parameters.time_resolution*max(energy_storages[i],np.float64(1e-15))*global_parameters.duration_years) for i in range(RENSystems_parameters.n_store)]) 
    Lifetime = np.array([min(RENSystems_parameters.specs_storage[STOR_LIFETIME,i],RENSystems_parameters.specs_storage[STOR_CYCLE_LIFE,i]/max(1e-15,Equivalent_cycles[i])) for i in range(RENSystems_parameters.n_store)])
    return(energy_storages,Lifetime)


@jit(nopython=True)
def _pro_indicators_Genset(gene,Genset_parameters,global_parameters,trades,importation):
    """
    Compute genset operational indicators.
    
    Args:
        gene:
            Individual solution.
    
        Genset_parameters:
            Object containing generator specifications.
    
        global_parameters:
            Object containing simulation parameters.
    
        trades (np.ndarray):
            Net power exchanges (kW).
    
        importation (np.ndarray):
            Genset production (kW).
    
    Returns:
        tuple:
            - DG_nominal_power (float): Required nominal power (kW)
            - DG_production (np.ndarray): Production time series (kW)
            - annual_fuel_consumption (float): Annual fuel consumption
    
    Notes:
        - Uses discretized load levels for fuel estimation.
    """
    DG_nominal_power = max(trades)
    DG_production=importation
    if (DG_nominal_power>0):
        closest_levels = np.array([int(10*(DG_production[i]/DG_nominal_power)-0.5) for i in range(global_parameters.n_bits)])
        annual_fuel_consumption = sum(DG_production*Genset_parameters.fuel_consumption[closest_levels]/global_parameters.time_resolution/global_parameters.duration_years)
    else : 
        annual_fuel_consumption = 0
    return(DG_nominal_power,DG_production,annual_fuel_consumption)

@jit(nopython=True)
def _pro_economics_Genset(annual_fuel_consumption,DG_production,DG_nominal_power,Genset_parameters,global_parameters):
    """
    Compute economic metrics of the genset.
    
    Args:
        annual_fuel_consumption (float):
            Annual fuel consumption.
    
        DG_production (np.ndarray):
            Genset production (kW).
    
        DG_nominal_power (float):
            Installed capacity (kW).
    
        Genset_parameters:
            Object containing cost parameters.
    
        global_parameters:
            Object containing simulation parameters.
    
    Returns:
        tuple:
            - annual_total_fuel_cost (float)
            - DG_CAPEX_cost (float)
            - DG_OPEX_cost (float)
    
    Notes:
        - CAPEX is prorated based on actual usage time.
    """
    annual_total_fuel_cost = annual_fuel_consumption*Genset_parameters.fuel_cost   
    DG_CAPEX_cost = DG_nominal_power*Genset_parameters.unit_cost*sum(np.where(DG_production>0,1,0))/global_parameters.time_resolution/Genset_parameters.lifetime/global_parameters.duration_years
    DG_OPEX_cost = sum(DG_production)*Genset_parameters.maintenance_cost/global_parameters.time_resolution/global_parameters.duration_years
    return(annual_total_fuel_cost,DG_CAPEX_cost,DG_OPEX_cost)
    
@jit(nopython=True)
def _pro_economics_Storage(gene,energy_storages,RENSystems_parameters,Lifetime,size_power):
    """
    Compute economic metrics of storage systems.
    
    Args:
        gene:
            Individual solution.
    
        energy_storages (np.ndarray):
            Storage capacities (kWh).
    
        RENSystems_parameters:
            Object containing storage cost parameters.
    
        Lifetime (np.ndarray):
            Storage lifetime (years).
    
        size_power (np.ndarray):
            Storage power capacities (kW).
    
    Returns:
        tuple:
            - economics_CAPEX_storage (float)
            - economics_OPEX_storage (float)
    
    Notes:
        - Includes power, energy, and installation costs.
    """
    CAPEX_storage_cost =  np.multiply(size_power,RENSystems_parameters.specs_storage[STOR_POWER_COST,:]) + np.multiply(energy_storages,RENSystems_parameters.specs_storage[STOR_ENERGY_COST,:]) + np.multiply(RENSystems_parameters.specs_storage[STOR_INSTALLATION_COST,:],(size_power>np.repeat(0,RENSystems_parameters.n_store)))
    economics_CAPEX_storage = sum(np.divide(CAPEX_storage_cost,Lifetime))
    economics_OPEX_storage = sum(np.multiply(RENSystems_parameters.specs_storage[STOR_OM_COST,:],size_power))
    return(economics_CAPEX_storage,economics_OPEX_storage)    
  

@jit(nopython=True)
def _pro_economics_Grid(gene,global_parameters,grid_parameters,importation,exportation):
    """
    Compute economic metrics of grid interaction.
    
    Args:
        gene:
            Individual solution containing contract choice.
    
        global_parameters:
            Object containing simulation parameters.
    
        grid_parameters:
            Object containing tariff structures.
    
        importation (np.ndarray):
            Imported energy (kW).
    
        exportation (np.ndarray):
            Exported energy (kW).
    
    Returns:
        tuple:
            - economics_exportation (float)
            - economics_importation (float)
            - economics_overrun (float)
            - economics_contract_power (float)
    
    Notes:
        - Contract power is based on maximum import.
    """
    Contract_power=max(0,max(importation))
    economics_exportation = np.multiply(exportation,grid_parameters.Selling_price[gene.contract,:]).sum()/global_parameters.time_resolution/global_parameters.duration_years
    economics_importation = np.multiply(importation,grid_parameters.prices[gene.contract,:]).sum()/global_parameters.time_resolution/global_parameters.duration_years
    economics_overrun = max(0,(max(importation)-Contract_power)*grid_parameters.Overrun[gene.contract])
    economics_contract_power = grid_parameters.fixed_premium[gene.contract]*Contract_power
    return(economics_exportation,economics_importation,economics_overrun,economics_contract_power)



@jit(nopython=True)
def _Compute_fitness(global_parameters,obtained_constraint_level,criterion_value):
    """
    Compute fitness value for optimisation.
    
    This function combines the objective function value with a penalty
    if the constraint level is not satisfied.
    
    Args:
        global_parameters:
            Object containing constraint settings and penalty coefficient.
    
        obtained_constraint_level (float):
            Computed constraint value.
    
        criterion_value (float):
            Objective function value (e.g., cost).
    
    Returns:
        float:
            Fitness value (penalized objective).
    
    Notes:
        - A penalty is applied if the constraint is not met.
    """
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
    annual_cost_production = _pro_annual_cost_production(gene,RENSystems_parameters)
    (production, Optimized_Load,trades,storage_TS,D_DSM,Y_DSM,SOCs_eff,losses,P_diff) = _pro_cost_base_indicators(gene,RENSystems_parameters,global_parameters,pro_parameters)

    gene.storages = _pro_update_storage_power(gene,RENSystems_parameters,storage_TS)        
    
    importation=np.where(trades>0,trades,0)    
    den_Optimized_load = max(1e-15,sum(Optimized_Load))
    obtained_constraint_level = _get_constraint_level(global_parameters,importation,den_Optimized_load,trades,production)
    size_power = _pro_size_storage_power(gene,RENSystems_parameters) 
    energy_storages,Lifetime = _pro_indicators_storage(gene,RENSystems_parameters,global_parameters,storage_TS,losses)
    (economics_CAPEX_storage,economics_OPEX_storage) = _pro_economics_Storage(gene,energy_storages,RENSystems_parameters,Lifetime,size_power)
          
    if (global_parameters.Connexion==GRID_OFF):
        (DG_nominal_power,DG_production,annual_fuel_consumption) = _pro_indicators_Genset(gene,Genset_parameters,global_parameters,trades,importation)
        (annual_total_fuel_cost,DG_CAPEX_cost,DG_OPEX_cost) = _pro_economics_Genset(annual_fuel_consumption,DG_production,DG_nominal_power,Genset_parameters,global_parameters)
        LCOE=(annual_cost_production+economics_CAPEX_storage+economics_OPEX_storage+DG_OPEX_cost+DG_CAPEX_cost+annual_total_fuel_cost)/(den_Optimized_load/global_parameters.time_resolution/global_parameters.duration_years)
    
    elif (global_parameters.Connexion==GRID_ON):            
        exportation = np.where(trades<0,trades,0)
        (economics_exportation,economics_importation,economics_overrun,economics_contract_power) = _pro_economics_Grid(gene,global_parameters,grid_parameters,importation,exportation)
        LCOE=(annual_cost_production+economics_CAPEX_storage+economics_OPEX_storage+economics_importation+economics_contract_power+economics_overrun+economics_exportation)/(den_Optimized_load/global_parameters.time_resolution/global_parameters.duration_years) 

    gene.fitness = _Compute_fitness(global_parameters,obtained_constraint_level,LCOE) 

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
    (production, Optimized_Load,trades,storage_TS,D_DSM,Y_DSM,SOCs_eff,losses,P_diff) = _pro_cost_base_indicators(gene,RENSystems_parameters,global_parameters,pro_parameters)
    gene.storages = _pro_update_storage_power(gene,RENSystems_parameters,storage_TS)        
    importation=np.where(trades>0,trades,0)
    den_Optimized_load = max(1e-15,sum(Optimized_Load))
    
    obtained_constraint_level = _get_constraint_level(global_parameters,importation,den_Optimized_load,trades,production)
    obtained_self_consumption = 1+(sum(np.where(trades<0,trades,0))/sum(production))
    gene.fitness = _Compute_fitness(global_parameters,obtained_constraint_level, -obtained_self_consumption)  
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
     
    (production, Optimized_Load,trades,storage_TS,D_DSM,Y_DSM,SOCs_eff,losses,P_diff) = _pro_cost_base_indicators(gene,RENSystems_parameters,global_parameters,pro_parameters)
    gene.storages = _pro_update_storage_power(gene,RENSystems_parameters,storage_TS)                                                           
    importation=np.where(trades>0,trades,0)
    Grid_importation = importation if (Connexion=='On-grid') else np.repeat(0.,n_bits)
    den_Optimized_load = max(1e-15,sum(Optimized_Load))
    
    obtained_constraint_level = _get_constraint_level(global_parameters,importation,den_Optimized_load,trades,production)
    obtained_self_sufficiency = (1-sum(Grid_importation)/sum(Optimized_Load)) if sum(Optimized_Load )>0 else np.nan
    gene.fitness = _Compute_fitness(global_parameters,obtained_constraint_level, -obtained_self_sufficiency)   
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

    (production, Optimized_Load,trades,storage_TS,D_DSM,Y_DSM,SOCs_eff,losses,P_diff) = _pro_cost_base_indicators(gene,RENSystems_parameters,global_parameters,pro_parameters)
    gene.storages = _pro_update_storage_power(gene,RENSystems_parameters,storage_TS)                                                              
    importation=np.where(trades>0,trades,0)
    den_Optimized_load = max(1e-15,sum(Optimized_Load))
    
    obtained_constraint_level = _get_constraint_level(global_parameters,importation,den_Optimized_load,trades,production)
    Max_import = max(importation)
    gene.fitness = _Compute_fitness(global_parameters,obtained_constraint_level,Max_import)  
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
     
    (production, Optimized_Load,trades,storage_TS,D_DSM,Y_DSM,SOCs_eff,losses,P_diff) = _pro_cost_base_indicators(gene,RENSystems_parameters,global_parameters,pro_parameters)
    gene.storages = _pro_update_storage_power(gene,RENSystems_parameters,storage_TS)                                                               
    importation=np.where(trades>0,trades,0)  
    den_Optimized_load = max(1e-15,sum(Optimized_Load))
    
    obtained_constraint_level = _get_constraint_level(global_parameters,importation,den_Optimized_load,trades,production)
    sum_losses = sum(losses)
    gene.fitness = _Compute_fitness(global_parameters,obtained_constraint_level,sum_losses)  
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
    annual_cost_production = _pro_annual_cost_production(gene,RENSystems_parameters)
    (production, Optimized_Load,trades,storage_TS,D_DSM,Y_DSM,SOCs_eff,losses,P_diff) = _pro_cost_base_indicators(gene,RENSystems_parameters,global_parameters,pro_parameters)
    gene.storages = _pro_update_storage_power(gene,RENSystems_parameters,storage_TS)        
    importation=np.where(trades>0,trades,0)
    den_Optimized_load = max(1e-15,sum(Optimized_Load))
    
    obtained_constraint_level = _get_constraint_level(global_parameters,importation,den_Optimized_load,trades,production)
    exportation = np.where(trades<0,-trades,0)
    energy_storages,Lifetime = _pro_indicators_storage(gene,RENSystems_parameters,global_parameters,storage_TS,losses)
    size_power = _pro_size_storage_power(gene,RENSystems_parameters)   
    (economics_CAPEX_storage,economics_OPEX_storage) = _pro_economics_Storage(gene,energy_storages,RENSystems_parameters,Lifetime,size_power)

    
    if (global_parameters.Connexion==GRID_OFF):
        (DG_nominal_power,DG_production,annual_fuel_consumption) = _pro_indicators_Genset(gene,Genset_parameters,global_parameters,trades,importation)
        (annual_total_fuel_cost,DG_CAPEX_cost,DG_OPEX_cost) = _pro_economics_Genset(annual_fuel_consumption,DG_production,DG_nominal_power,Genset_parameters,global_parameters)
        Annual_net_benefits = (-annual_cost_production-economics_CAPEX_storage-economics_OPEX_storage-DG_OPEX_cost-DG_CAPEX_cost-annual_total_fuel_cost)

        
    elif (global_parameters.Connexion==GRID_ON):  
        (economics_exportation,economics_importation,economics_overrun,economics_contract_power) = _pro_economics_Grid(gene,global_parameters,grid_parameters,importation,exportation)
        Annual_net_benefits = (-annual_cost_production-economics_CAPEX_storage-economics_OPEX_storage-economics_importation-economics_contract_power-economics_overrun+economics_exportation)
        
    gene.fitness = _Compute_fitness(global_parameters,obtained_constraint_level, -Annual_net_benefits) 
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
    annual_cost_production = _pro_annual_cost_production(gene,RENSystems_parameters)
    (production, Optimized_Load,trades,storage_TS,D_DSM,Y_DSM,SOCs_eff,losses,P_diff) = _pro_cost_base_indicators(gene,RENSystems_parameters,global_parameters,pro_parameters)
    
    gene.storages = _pro_update_storage_power(gene,RENSystems_parameters,storage_TS)        
      
                                                        
    importation=np.where(trades>0,trades,0)
    den_Optimized_load = max(1e-15,sum(Optimized_Load))
    obtained_constraint_level = _get_constraint_level(global_parameters,importation,den_Optimized_load,trades,production)
    exportation = np.where(trades<0,-trades,0)
    
    energy_storages,Lifetime = _pro_indicators_storage(gene,RENSystems_parameters,global_parameters,storage_TS,losses)

    size_power = _pro_size_storage_power(gene,RENSystems_parameters)
    (economics_CAPEX_storage,economics_OPEX_storage) = _pro_economics_Storage(gene,energy_storages,RENSystems_parameters,Lifetime,size_power)
    
    if (global_parameters.Connexion==GRID_OFF):
        (DG_nominal_power,DG_production,annual_fuel_consumption) = _pro_indicators_Genset(gene,Genset_parameters,global_parameters,trades,importation)
        (annual_total_fuel_cost,DG_CAPEX_cost,DG_OPEX_cost) = _pro_economics_Genset(annual_fuel_consumption,DG_production,DG_nominal_power,Genset_parameters,global_parameters)
        Annual_net_benefits = (-annual_cost_production-economics_CAPEX_storage-economics_OPEX_storage-DG_OPEX_cost-DG_CAPEX_cost-annual_total_fuel_cost)
        DG_lifetime_years = Genset_parameters.lifetime/(sum(np.where(DG_production>0,1,0))/global_parameters.time_resolution/global_parameters.duration_years)

    elif (global_parameters.Connexion==GRID_ON):  
        (economics_exportation,economics_importation,economics_overrun,economics_contract_power) = _pro_economics_Grid(gene,global_parameters,grid_parameters,importation,exportation)
        Annual_net_benefits = (-annual_cost_production-economics_CAPEX_storage-economics_OPEX_storage-economics_importation-economics_contract_power-economics_overrun+economics_exportation)
        DG_lifetime_years = np.nan

    Lifetime_installation = np.nanmin(DG_lifetime_years,min(Lifetime),np.nanmin(np.where(gene.production_set>0,RENSystems_parameters.specs_prod[:,PROD_LIFETIME],np.nan)))
    NPV = Annual_net_benefits*Lifetime_installation
    gene.fitness = _Compute_fitness(global_parameters,obtained_constraint_level, -NPV) 
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
    (production, Optimized_Load,trades,storage_TS,D_DSM,Y_DSM,SOCs_eff,losses,P_diff) = _pro_cost_base_indicators(gene,RENSystems_parameters,global_parameters,pro_parameters)
    
    gene.storages = _pro_update_storage_power(gene,RENSystems_parameters,storage_TS)        
  
        
    importation=np.where(trades>0,trades,0)
    den_Optimized_load = max(1e-15,sum(Optimized_Load))
    obtained_constraint_level = _get_constraint_level(global_parameters,importation,den_Optimized_load,trades,production)

    Grid_importation = importation if (global_parameters.Connexion==GRID_ON) else np.repeat(0.,global_parameters.n_bits)
    Autonomy = 1-sum(Grid_importation>0)/global_parameters.n_bits
    gene.fitness = _Compute_fitness(global_parameters,obtained_constraint_level, -Autonomy) 

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
    (production, Optimized_Load,trades,storage_TS,D_DSM,Y_DSM,SOCs_eff,losses,P_diff) = _pro_cost_base_indicators(gene,RENSystems_parameters,global_parameters,pro_parameters)   
    gene.storages = _pro_update_storage_power(gene,RENSystems_parameters,storage_TS)           
        
    importation=np.where(trades>0,trades,0)
    den_Optimized_load = max(1e-15,sum(Optimized_Load))
    obtained_constraint_level = _get_constraint_level(global_parameters,importation,den_Optimized_load,trades,production)

    Grid_importation = importation if (global_parameters.Connexion==GRID_ON) else np.repeat(0.,global_parameters.n_bits)
    DG_production = importation if (global_parameters.Connexion==GRID_OFF) else np.repeat(0.,global_parameters.n_bits)
        
    annual_CO2eq_prod = sum(np.sum((gene.production_set*RENSystems_parameters.unit_productions.T)/KILOS_CONVERSION_FACTOR*RENSystems_parameters.specs_prod[:,PROD_EMISSIONS],axis=0))/TONS_CONVERSION_FACTOR/global_parameters.time_resolution/global_parameters.duration_years   
    annual_CO2eq_importation = sum(Grid_importation)*grid_parameters.eqCO2emissions/TONS_CONVERSION_FACTOR/global_parameters.time_resolution/global_parameters.duration_years
    annual_CO2eq_storage = np.inner(np.array([sum(np.where(storage_TS[i]>0,storage_TS[i],0)) for i in range(RENSystems_parameters.n_store)]),RENSystems_parameters.specs_storage[STOR_EMISSIONS,:])/TONS_CONVERSION_FACTOR/global_parameters.time_resolution/global_parameters.duration_years

    if (global_parameters.Connexion==GRID_OFF):
        (DG_nominal_power,DG_production,annual_fuel_consumption) = _pro_indicators_Genset(gene,Genset_parameters,global_parameters,trades,importation)
        annual_CO2eq_DG = annual_fuel_consumption*Genset_parameters.fuel_CO2eq_emissions/TONS_CONVERSION_FACTOR      
    elif (global_parameters.Connexion==GRID_ON):          
        annual_CO2eq_DG = 0.

    annual_eqCO2emissions = annual_CO2eq_DG+annual_CO2eq_prod+annual_CO2eq_importation+annual_CO2eq_storage
    gene.fitness = _Compute_fitness(global_parameters,obtained_constraint_level, annual_eqCO2emissions) 

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
    (production, Optimized_Load,trades,storage_TS,D_DSM,Y_DSM,SOCs_eff,losses,P_diff) = _pro_cost_base_indicators(gene,RENSystems_parameters,global_parameters,pro_parameters)
    
    gene.storages = _pro_update_storage_power(gene,RENSystems_parameters,storage_TS)        
  
        
    importation=np.where(trades>0,trades,0)
    den_Optimized_load = max(1e-15,sum(Optimized_Load))
    obtained_constraint_level = _get_constraint_level(global_parameters,importation,den_Optimized_load,trades,production)

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
    gene.fitness = _Compute_fitness(global_parameters,obtained_constraint_level, annual_fossil_fuel_consumption) 

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
    (production, Optimized_Load,trades,storage_TS,D_DSM,Y_DSM,SOCs_eff,losses,P_diff) = _pro_cost_base_indicators(gene,RENSystems_parameters,global_parameters,pro_parameters)
    gene.storages = _pro_update_storage_power(gene,RENSystems_parameters,storage_TS)          
    importation=np.where(trades>0,trades,0)
    den_Optimized_load = max(1e-15,sum(Optimized_Load))
    obtained_constraint_level = _get_constraint_level(global_parameters,importation,den_Optimized_load,trades,production)

    energy_storages,Lifetime = _pro_indicators_storage(gene,RENSystems_parameters,global_parameters,storage_TS,losses)
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
    gene.fitness = _Compute_fitness(global_parameters,obtained_constraint_level, -EROI) 

    return(gene)

####  RESEARCH COST FUNCTIONS ##################

@jit(nopython=True)
def _research_cost_base_indicators(gene,RENSystems_parameters,global_parameters):
    """
    Compute core energy balance indicators.
    
    Notes:
        - Production is converted from W to kW using a factor of 1000.
        - Trades include the effect of storage power flows.
    """
    KILOS_CONVERSION_FACTOR = 1000
    production = ((RENSystems_parameters.unit_productions.T*gene.production_set).sum(axis=1)+RENSystems_parameters.current_production)/KILOS_CONVERSION_FACTOR    
    Optimized_Load = global_parameters.Non_movable_load+gene.Y_DSM+gene.D_DSM.flatten()   
    power_storage = np.sum(gene.storage_TS,axis=0)
    trades = Optimized_Load-production-power_storage
    return(production, Optimized_Load,trades)

@jit(nopython=True)
def _compute_losses(gene,RENSystems_parameters):
    """
    Compute storage energy losses based on round-trip efficiency.
    """
    losses = np.transpose(np.divide(np.transpose(gene.storage_TS),RENSystems_parameters.specs_storage[STOR_ROUND_TRIP_EFF,:]))-gene.storage_TS 
    losses=np.where(losses>0,losses,0)
    return(losses)

@jit(nopython=True)
def _size_storage_power(gene,RENSystems_parameters):
    """
    Determine required storage power capacity.

    Notes:
        - Power is defined as the maximum absolute value between charge
          and discharge over the time horizon.
    """
    powers_out = [np.max(gene.storage_TS[i,:]) for i in range(RENSystems_parameters.n_store)]
    powers_in = [-np.min(gene.storage_TS[i,:]) for i in range(RENSystems_parameters.n_store)]
    size_power=np.array([max(powers_in[i],powers_out[i]) for i in range(RENSystems_parameters.n_store)])
    return(size_power)

@jit(nopython=True)
def _Indicators_storage(gene,RENSystems_parameters,global_parameters,losses):
    """
    Compute storage energy capacity and lifetime indicators.
    
    Notes:
        - Equivalent cycles are computed from total energy throughput.
        - Lifetime is limited by either calendar life or cycle life.
    """
    sum_diff_storages = [np.cumsum(gene.storage_TS[i,:]+losses[i,:])/global_parameters.time_resolution for i in range(RENSystems_parameters.n_store)]
    energy_storages = np.divide(np.array([np.max(sum_diff_storages[i]) - np.min(sum_diff_storages[i]) for i in range(RENSystems_parameters.n_store)]),RENSystems_parameters.specs_storage[STOR_DEPTH_OF_DISCHARGE,:])
    Equivalent_cycles =  np.array([np.sum(np.abs(gene.storage_TS[i,:]))/(2*global_parameters.time_resolution*max(energy_storages[i],np.float64(1e-15))*global_parameters.duration_years) for i in range(RENSystems_parameters.n_store)]) 
    Lifetime = np.array([min(RENSystems_parameters.specs_storage[STOR_LIFETIME,i],RENSystems_parameters.specs_storage[STOR_CYCLE_LIFE,i]/max(1e-15,Equivalent_cycles[i])) for i in range(RENSystems_parameters.n_store)])
    return(energy_storages,Lifetime)

@jit(nopython=True)
def _Indicators_Genset(gene,Genset_parameters,global_parameters,trades,importation):
    """
    Compute diesel generator (genset) operational indicators.

    Notes:
        - Fuel consumption is estimated using discretized load levels.
        - If no genset is used, fuel consumption is zero.
    """
    DG_nominal_power = max(trades)
    DG_production=importation
    if (DG_nominal_power>0):
        closest_levels = np.array([int(10*(DG_production[i]/DG_nominal_power)-0.5) for i in range(global_parameters.n_bits)])
        annual_fuel_consumption = sum(DG_production*Genset_parameters.fuel_consumption[closest_levels]/global_parameters.time_resolution/global_parameters.duration_years)
    else : 
        annual_fuel_consumption = 0
    return(DG_nominal_power,DG_production,annual_fuel_consumption)

@jit(nopython=True)
def _Economics_Genset(annual_fuel_consumption,DG_production,DG_nominal_power,Genset_parameters,global_parameters):
    """
    Compute economic metrics of the diesel generator.
    
    This function evaluates annual fuel costs, capital expenditures (CAPEX),
    operational expenditures (OPEX), and effective lifetime.

    Notes:
        - If the genset is not used, costs are set to zero and lifetime to NaN.
    """
    if (sum(DG_production>0)):
        annual_total_fuel_cost = annual_fuel_consumption*Genset_parameters.fuel_cost
        DG_lifetime_years = Genset_parameters.lifetime/(sum(np.where(DG_production>0,1,0))/global_parameters.time_resolution/global_parameters.duration_years)
        DG_CAPEX_cost = DG_nominal_power*Genset_parameters.unit_cost/DG_lifetime_years
        DG_OPEX_cost = sum(DG_production)*Genset_parameters.maintenance_cost/global_parameters.time_resolution/global_parameters.duration_years
    else :
        (DG_lifetime_years,annual_total_fuel_cost,DG_CAPEX_cost,DG_OPEX_cost) = (np.nan,0,0,0)
    
    return(DG_lifetime_years,annual_total_fuel_cost,DG_CAPEX_cost,DG_OPEX_cost)

@jit(nopython=True)
def _Economics_Grid(gene,global_parameters,grid_parameters,importation,exportation):
    """
    Compute economic metrics related to grid interactions.
    
    This function evaluates costs and revenues associated with electricity
    import/export, contract power, and overrun penalties.

    Notes:
        - Contract power is defined as the maximum import power.
    """
    Contract_power=max(0,max(importation))
    economics_exportation = np.multiply(exportation,grid_parameters.Selling_price[gene.contract,:]).sum()/global_parameters.time_resolution/global_parameters.duration_years
    economics_importation = np.multiply(importation,grid_parameters.prices[gene.contract,:]).sum()/global_parameters.time_resolution/global_parameters.duration_years
    economics_overrun = max(0,(max(importation)-Contract_power)*grid_parameters.Overrun[gene.contract])
    economics_contract_power = grid_parameters.fixed_premium[gene.contract]*Contract_power
    return(economics_exportation,economics_importation,economics_overrun,economics_contract_power)

@jit(nopython=True)
def _Economics_Storage(gene,energy_storages,RENSystems_parameters,Lifetime,size_power):
    """
    Compute economic metrics of storage systems.
    
    This function evaluates capital and operational costs of storage
    based on power and energy sizing.

    Notes:
        - CAPEX includes power cost, energy cost, and installation cost.
        - OPEX is proportional to installed power.
    """
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
    
    (production, Optimized_Load,trades) = _research_cost_base_indicators(gene,RENSystems_parameters,global_parameters)
    losses = _compute_losses(gene,RENSystems_parameters)
    size_power = _size_storage_power(gene,RENSystems_parameters)
    
    annual_cost_production = np.sum((RENSystems_parameters.specs_prod[:,PROD_CAPEX]/RENSystems_parameters.specs_prod[:,PROD_LIFETIME]+RENSystems_parameters.specs_prod[:,PROD_OPEX])*(gene.production_set))
    
    (energy_storages,Lifetime) = _Indicators_storage(gene,RENSystems_parameters,global_parameters,losses)
   
    importation=np.where(trades>0,trades,0)
    exportation = np.where(trades<0,trades,0)
    
    (economics_CAPEX_storage,economics_OPEX_storage)=_Economics_Storage(gene,energy_storages,RENSystems_parameters,Lifetime,size_power)

    
    if (global_parameters.Connexion==GRID_OFF):
        (DG_nominal_power,DG_production,annual_fuel_consumption) = _Indicators_Genset(gene,Genset_parameters,global_parameters,trades,importation)
        (DG_lifetime_years,annual_total_fuel_cost,DG_CAPEX_cost,DG_OPEX_cost) = _Economics_Genset(annual_fuel_consumption,DG_production,DG_nominal_power,Genset_parameters,global_parameters)
        economics_exportation = 0
        economics_importation = 0
        economics_overrun = 0
        economics_contract_power = 0
        
        
    elif (global_parameters.Connexion==GRID_ON):          
        annual_total_fuel_cost = 0
        DG_CAPEX_cost = 0
        DG_OPEX_cost = 0
        (economics_exportation,economics_importation,economics_overrun,economics_contract_power) = _Economics_Grid(gene,global_parameters,grid_parameters,importation,exportation)
        
    
    LCOE=(annual_cost_production+economics_CAPEX_storage+economics_OPEX_storage+DG_OPEX_cost+DG_CAPEX_cost+annual_total_fuel_cost+economics_importation+economics_contract_power+economics_overrun+economics_exportation)/(sum(Optimized_Load)/global_parameters.time_resolution/global_parameters.duration_years)   if sum(Optimized_Load )>0 else np.nan

    den_Optimized_load = max(1e-15,sum(Optimized_Load))   
    obtained_constraint_level = _get_constraint_level(global_parameters,importation,den_Optimized_load,trades,production)
    fitness = _Compute_fitness(global_parameters,obtained_constraint_level,LCOE)

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
    (production, Optimized_Load,trades) = _research_cost_base_indicators(gene,RENSystems_parameters,global_parameters)

    annual_cost_production = np.sum((RENSystems_parameters.specs_prod[:,PROD_CAPEX]/RENSystems_parameters.specs_prod[:,PROD_LIFETIME]+RENSystems_parameters.specs_prod[:,PROD_OPEX])*(gene.production_set))
    losses = _compute_losses(gene,RENSystems_parameters)
    size_power = _size_storage_power(gene,RENSystems_parameters)
  
    (energy_storages,Lifetime) = _Indicators_storage(gene,RENSystems_parameters,global_parameters,losses)

    importation=np.where(trades>0,trades,0)
    exportation = np.where(trades<0,-trades,0)

    (economics_CAPEX_storage,economics_OPEX_storage)=_Economics_Storage(gene,energy_storages,RENSystems_parameters,Lifetime,size_power)

    if (global_parameters.Connexion==GRID_OFF):
        (DG_nominal_power,DG_production,annual_fuel_consumption) = _Indicators_Genset(gene,Genset_parameters,global_parameters,trades,importation)
        (DG_lifetime_years,annual_total_fuel_cost,DG_CAPEX_cost,DG_OPEX_cost) = _Economics_Genset(annual_fuel_consumption,DG_production,DG_nominal_power,Genset_parameters,global_parameters)
        economics_exportation = 0
        economics_importation = 0
        economics_overrun = 0
        economics_contract_power = 0
        
    elif (global_parameters.Connexion==GRID_ON):          
        DG_nominal_power = 0
        annual_fuel_consumption = 0
        DG_CAPEX_cost = 0
        DG_OPEX_cost = 0
        (economics_exportation,economics_importation,economics_overrun,economics_contract_power) = _Economics_Grid(gene,global_parameters,grid_parameters,importation,exportation)
        
    den_Optimized_load = max(1e-15,sum(Optimized_Load))   
    obtained_constraint_level = _get_constraint_level(global_parameters,importation,den_Optimized_load,trades,production)

    Annual_net_benefits = (-annual_cost_production-economics_CAPEX_storage-economics_OPEX_storage-DG_OPEX_cost-DG_CAPEX_cost-annual_total_fuel_cost-economics_importation-economics_contract_power-economics_overrun+economics_exportation)
    fitness = _Compute_fitness(global_parameters,obtained_constraint_level, -Annual_net_benefits)

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
    (production, Optimized_Load,trades) = _research_cost_base_indicators(gene,RENSystems_parameters,global_parameters)

    annual_cost_production = np.sum((RENSystems_parameters.specs_prod[:,PROD_CAPEX]/RENSystems_parameters.specs_prod[:,PROD_LIFETIME]+RENSystems_parameters.specs_prod[:,PROD_OPEX])*(gene.production_set))
    losses = _compute_losses(gene,RENSystems_parameters)

    size_power = _size_storage_power(gene,RENSystems_parameters) 
    
    (energy_storages,Lifetime) = _Indicators_storage(gene,RENSystems_parameters,global_parameters,losses)
    importation=np.where(trades>0,trades,0)
    exportation = np.where(trades<0,-trades,0)

    (economics_CAPEX_storage,economics_OPEX_storage)=_Economics_Storage(gene,energy_storages,RENSystems_parameters,Lifetime,size_power)

    if (global_parameters.Connexion==GRID_OFF):
        (DG_nominal_power,DG_production,annual_fuel_consumption) = _Indicators_Genset(gene,Genset_parameters,global_parameters,trades,importation)
        (DG_lifetime_years,annual_total_fuel_cost,DG_CAPEX_cost,DG_OPEX_cost) = _Economics_Genset(annual_fuel_consumption,DG_production,DG_nominal_power,Genset_parameters,global_parameters)
        economics_exportation = 0
        economics_importation = 0
        economics_overrun = 0
        economics_contract_power = 0
        
    elif (global_parameters.Connexion==GRID_ON):          
        DG_nominal_power = 0
        DG_CAPEX_cost = 0
        DG_OPEX_cost = 0
        (economics_exportation,economics_importation,economics_overrun,economics_contract_power) = _Economics_Grid(gene,global_parameters,grid_parameters,importation,exportation)
    
    den_Optimized_load = max(1e-15,sum(Optimized_Load))   
    obtained_constraint_level = _get_constraint_level(global_parameters,importation,den_Optimized_load,trades,production)

    Lifetime_installation = min(DG_lifetime_years,min(Lifetime),np.nanmin(np.where(gene.production_set>0,RENSystems_parameters.specs_prod[:,PROD_LIFETIME],np.nan)))
    Annual_net_benefits = (-annual_cost_production-economics_CAPEX_storage-economics_OPEX_storage-DG_OPEX_cost-DG_CAPEX_cost-annual_total_fuel_cost-economics_importation-economics_contract_power-economics_overrun+economics_exportation)
    NPV=Annual_net_benefits*Lifetime_installation
    fitness = _Compute_fitness(global_parameters,obtained_constraint_level, -NPV)
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
    (production, Optimized_Load,trades) = _research_cost_base_indicators(gene,RENSystems_parameters,global_parameters)

    importation=np.where(trades>0,trades,0)

    Max_import = max(importation) 
    
    den_Optimized_load = max(1e-15,sum(Optimized_Load))   
    obtained_constraint_level = _get_constraint_level(global_parameters,importation,den_Optimized_load,trades,production)
    fitness = _Compute_fitness(global_parameters,obtained_constraint_level, Max_import)    
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
    (production, Optimized_Load,trades) = _research_cost_base_indicators(gene,RENSystems_parameters,global_parameters)

    losses = _compute_losses(gene,RENSystems_parameters)

                       
    Annual_sum_losses = np.sum(losses,axis=1)/global_parameters.time_resolution/global_parameters.duration_years 
    importation=np.where(trades>0,trades,0)

    den_Optimized_load = max(1e-15,sum(Optimized_Load))   
    obtained_constraint_level = _get_constraint_level(global_parameters,importation,den_Optimized_load,trades,production)   
    fitness = _Compute_fitness(global_parameters,obtained_constraint_level, Annual_sum_losses)    
    
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
    (production, Optimized_Load,trades) = _research_cost_base_indicators(gene,RENSystems_parameters,global_parameters)

    

    importation=np.where(trades>0,trades,0)    
    Grid_importation = np.where(trades>0,trades,0) if (global_parameters.Connexion==GRID_ON) else np.repeat(0.,global_parameters.n_bits)
    
    annual_CO2eq_prod = sum(np.sum((gene.production_set*RENSystems_parameters.unit_productions.T)/KILOS_CONVERSION_FACTOR*RENSystems_parameters.specs_prod[:,PROD_EMISSIONS],axis=0))/TONS_CONVERSION_FACTOR/global_parameters.time_resolution/global_parameters.duration_years   
    annual_CO2eq_importation = sum(Grid_importation)*grid_parameters.eqCO2emissions/TONS_CONVERSION_FACTOR/global_parameters.time_resolution/global_parameters.duration_years
    annual_CO2eq_storage = sum(np.array([sum(np.where(gene.storage_TS[i]>0,gene.storage_TS[i],0)) for i in range(RENSystems_parameters.n_store)])[:]*RENSystems_parameters.specs_storage[STOR_EMISSIONS,:])/TONS_CONVERSION_FACTOR/global_parameters.time_resolution/global_parameters.duration_years
    if (global_parameters.Connexion==GRID_OFF):
        (DG_nominal_power,DG_production,annual_fuel_consumption) = _Indicators_Genset(gene,Genset_parameters,global_parameters,trades,importation)
        annual_CO2eq_DG = annual_fuel_consumption*Genset_parameters.fuel_CO2eq_emissions/TONS_CONVERSION_FACTOR      
    elif (global_parameters.Connexion==GRID_ON):          
        annual_CO2eq_DG = 0.

    den_Optimized_load = max(1e-15,sum(Optimized_Load))   
    obtained_constraint_level = _get_constraint_level(global_parameters,importation,den_Optimized_load,trades,production)  
    annual_CO2eq_total = annual_CO2eq_DG+annual_CO2eq_prod+annual_CO2eq_importation+annual_CO2eq_storage
    fitness = _Compute_fitness(global_parameters,obtained_constraint_level, annual_CO2eq_total)    
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
    (production, Optimized_Load,trades) = _research_cost_base_indicators(gene,RENSystems_parameters,global_parameters)
    
    importation = np.where(trades>0,trades,0)
    grid_importation = importation if (global_parameters.Connexion==GRID_ON) else np.repeat(0.,global_parameters.n_bits)

    den_Optimized_load = max(1e-15,sum(Optimized_Load))   
    obtained_constraint_level = _get_constraint_level(global_parameters,importation,den_Optimized_load,trades,production)  
    Autonomy = 1-sum(grid_importation>0)/global_parameters.n_bits
    fitness = _Compute_fitness(global_parameters,obtained_constraint_level, -Autonomy)    
    
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
    (production, Optimized_Load,trades) = _research_cost_base_indicators(gene,RENSystems_parameters,global_parameters)

    importation=np.where(trades>0,trades,0)
    Grid_importation = importation if (global_parameters.Connexion==GRID_ON) else np.repeat(0.,global_parameters.n_bits)    
    annual_fossil_fuel_consumption_importation =  grid_parameters.fossil_fuel_ratio*sum(Grid_importation)/global_parameters.time_resolution/global_parameters.duration_years

    if (global_parameters.Connexion==GRID_OFF):
        (DG_nominal_power,DG_production,annual_fuel_consumption) = _Indicators_Genset(gene,Genset_parameters,global_parameters,trades,importation)
        
    elif (global_parameters.Connexion==GRID_ON):          
        annual_fuel_consumption = 0

    den_Optimized_load = max(1e-15,sum(Optimized_Load))   
    obtained_constraint_level = _get_constraint_level(global_parameters,importation,den_Optimized_load,trades,production) 
    annual_fossil_fuel_consumption = annual_fossil_fuel_consumption_importation+annual_fuel_consumption
    fitness = _Compute_fitness(global_parameters,obtained_constraint_level, annual_fossil_fuel_consumption)    

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
    (production, Optimized_Load,trades) = _research_cost_base_indicators(gene,RENSystems_parameters,global_parameters)
    
    losses = _compute_losses(gene,RENSystems_parameters)
    
    importation=np.where(trades>0,trades,0)
    
    (energy_storages,Lifetime) = _Indicators_storage(gene,RENSystems_parameters,global_parameters,losses)
 
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
    obtained_constraint_level = _get_constraint_level(global_parameters,importation,den_Optimized_load,trades,production)
    EROI = (productible_energy+productible_energy_DG)/(consumed_energy_DG+consumed_energy_production+consumed_energy_storage)
    fitness = _Compute_fitness(global_parameters,obtained_constraint_level, -EROI)  
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
    (production, Optimized_Load,trades) = _research_cost_base_indicators(gene,RENSystems_parameters,global_parameters)


    exportation = np.where(trades<0,-trades,0)
    importation=np.where(trades>0,trades,0)

    den_Optimized_load = max(1e-15,sum(Optimized_Load))   
    obtained_constraint_level = _get_constraint_level(global_parameters,importation,den_Optimized_load,trades,production)
    obtained_self_consumption = (1-sum(exportation)/sum(production))
    fitness = _Compute_fitness(global_parameters,obtained_constraint_level, -obtained_self_consumption)  
    
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
    (production, Optimized_Load,trades) = _research_cost_base_indicators(gene,RENSystems_parameters,global_parameters)

    importation=np.where(trades>0,trades,0)
    
    Grid_importation = importation if (global_parameters.Connexion==GRID_ON) else np.repeat(0.,global_parameters.n_bits)

    den_Optimized_load = max(1e-15,sum(Optimized_Load))   
    obtained_constraint_level = _get_constraint_level(global_parameters,importation,den_Optimized_load,trades,production)
    obtained_self_sufficiency = (1-sum(Grid_importation)/sum(Optimized_Load)) if sum(Optimized_Load )>0 else np.nan
    fitness = _Compute_fitness(global_parameters,obtained_constraint_level, -obtained_self_sufficiency)  
   
    return(fitness,trades)
       

