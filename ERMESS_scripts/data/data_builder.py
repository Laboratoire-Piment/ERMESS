# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 11:53:44 2026

@author: JoPHOBEA
"""

import numpy as np

class OptimBlock:
    """
    Optimization problem definition.
    
    Defines the optimization objective, constraints, and global
    problem configuration.
    
    Attributes:
        constraint_num (int):
            Encoded type of constraint (e.g., self-sufficiency, self-consumption).
    
        constraint_level (float):
            Target level for the constraint.
    
        criterion_num (int):
            Encoded optimization objective.
    """
    __slots__ = (
        "constraint_num",
        "constraint_level",
        "criterion_num",
        "connexion",
    )

    def __init__(self, constraint_num, constraint_level, criterion_num, connexion):
        self.constraint_num = np.int64(constraint_num)
        self.constraint_level = np.float64(constraint_level)
        self.criterion_num = np.int64(criterion_num)
        self.connexion = connexion
        
class HyperparametersBlock:
    """
    Genetic algorithm hyperparameters.
    
    Stores configuration parameters controlling evolutionary behavior
    of the optimization algorithm.
    
    Attributes:
        r_cross (float):
            Crossover rate.
    
        n_iter (int):
            Number of iterations (generations).
            
        n_pop (int):
            Number of individuals in the population.
    
        operators (np.ndarray):
            Array of operator weights or probabilities.
                        
        cost_constraint (float):
            Penalty or cost associated with constraint violation.
            
        elitism_probability (float):
            Probability of the best individual to stay in the population.
    """
    __slots__ = (
        "r_cross",
        "n_iter",
        "n_pop",
        "operators_parameters",
        "cost_constraint",
        "elitism_probability"
    )

    def __init__(self, r_cross, n_iter,n_pop, operators_parameters,cost_constraint,elitism_probability):
        self.r_cross = np.float64(r_cross)
        self.n_iter = np.int64(n_iter)
        self.n_pop = np.int64(n_pop)
        self.operators_parameters = operators_parameters
        self.cost_constraint = cost_constraint
        self.elitism_probability = elitism_probability
        
class DispatchingBlock:
    """
    Global system configuration.
    
    Defines high-level system settings such as connectivity type
    and activated features.
    
    Attributes:
        connexion (str):
            Grid connection type ("On-grid" or "Off-grid").
    
        Defined_items (np.ndarray):
            List of non-controllable dispatching features.
    """
    __slots__ = (
        "defined_items",
        "energy_use_coefficient",
        "Y_DSM_minimum_levels",
        "D_DSM_minimum_levels",
        "DG_strategy",
        "DG_min_runtime",
        "DG_min_production",
        "discharge_order",
        "overlaps"
    )

    def __init__(self, defined_items,energy_use_coefficient,Y_DSM_minimum_levels,D_DSM_minimum_levels,DG_strategy,DG_min_runtime,DG_min_production,discharge_order,overlaps):
        self.defined_items = defined_items
        self.energy_use_coefficient = energy_use_coefficient if 'DSM' in defined_items else None
        self.Y_DSM_minimum_levels = Y_DSM_minimum_levels if 'DSM' in defined_items else None
        self.D_DSM_minimum_levels = D_DSM_minimum_levels if 'DSM' in defined_items else None
        self.DG_strategy = DG_strategy if 'Genset control' in defined_items else None
        self.DG_min_runtime = DG_min_runtime if 'Genset control' in defined_items else None
        self.DG_min_production = DG_min_production if 'Genset control' in defined_items else None
        self.discharge_order = discharge_order if 'Storages management' in defined_items else None
        self.overlaps = overlaps if 'Storages management' in defined_items else None 

class TimeParametersBlock:
    """
    Temporal configuration of the simulation environment.
    
    Defines the time resolution and global simulation duration used
    for demand, production, and optimization time series.
    
    Attributes:
        time_resolution (float):
            Number of time steps per hour.
    
        duration_years (float):
            Total simulation duration in years.
    
        n_steps (int):
            Total number of simulation time steps.
    """
    __slots__ = (
        "time_resolution",
        "duration_years",
        "n_bits",
        "n_days",
    )

    def __init__(self, time_resolution, duration_years, n_bits, n_days):
        self.time_resolution = np.float64(time_resolution)
        self.duration_years = duration_years
        self.n_bits = np.int64(n_bits)
        self.n_days = np.int64(n_days)
        
class ProductionBlock:
    """
    Production assets and generation profiles.
    
    Contains all information related to renewable and conventional
    production units, including technical characteristics and
    time-series production profiles.
    
    Attributes:
        specs_num (np.ndarray):
            Numerical characteristics of production units, incluing grouping structure, Total production volumes and capacity bounds.
    
        capacities (np.ndarray):
            Capacities of on-site installation of each production asset.
    
        groups (np.ndarray):
            Grouping structure of production assets.
    
        prod_C (np.ndarray):
            Production cost coefficients.
    
        prods_U (np.ndarray):
            Unit production time series for each asset.
    
        n_units (int):
            Number of production units.
    """
    __slots__ = (
        "specs_num",
        "Ids",
        "capacities",
        "groups",
        "current_prod",
        "unit_prods",
        "n_units",)

    def __init__(self, specs_num, Ids, capacities, groups, current_prod, unit_prods):
        self.specs_num = specs_num
        self.Ids = Ids
        self.capacities = capacities
        self.groups = groups
        self.current_prod = current_prod
        self.unit_prods = unit_prods
        self.n_units = np.int64(len(specs_num))
              
class LoadBlock:
    """
    Load characteristics and profiles.
    
    Aggregates all demand-related signals including non-controllable
    loads and DSM (Demand Side Management) components.
    
    Attributes:
        non_movable (np.ndarray):
            Non-controllable electrical load time series.
            
        Y_movable (np.array):
            Yearly dispatchable electrical load time series.
        
        D_movable (np.array):
            Daily dispatchable electrical load time series.
    
        total_Y (float):
            Total yearly movable load.
    
        total_D (np.ndarray):
            Aggregated daily movable load per day.
    
        D_DSM_indexes (np.ndarray):
            Indices of non-zero DSM day periods.
    """
    __slots__ = (
        "non_movable",
        "Y_movable",
        "D_movable",
        "D_DSM_indexes",
        "total_Y_movable",
        "total_D_movable",)

    def __init__(self, non_movable, Y_movable, D_movable, time_resolution):

        self.non_movable = non_movable
        self.D_movable = D_movable
        self.Y_movable = Y_movable
        self.total_Y_movable = np.sum(Y_movable)
        self.total_D_movable = np.array([np.sum(D_movable[np.arange(np.int32(i * time_resolution * 24),np.int32((i + 1) * time_resolution * 24))])
            for i in range( 0,np.int32(len(D_movable) / time_resolution / 24))], dtype=np.float64)
        self.D_DSM_indexes = np.where(self.total_D_movable != 0)[0]       

class StorageBlock:
    """
     Storage system description for the optimization environment.
    
     This block contains all data related to storage technologies,
     including their numerical characteristics and identifiers.
    
     Attributes:
         characteristics (np.ndarray):
             Numerical matrix describing storage parameters (costs, capacities, efficiencies, etc.). 
         n_store (int):
             Number of storage technologies in the system.
     """
    __slots__ = ("characteristics","n_store","technologies",)

    def __init__(self, characteristics,n_store,technologies):
        self.characteristics = characteristics
        self.n_store = n_store
        self.technologies = technologies

class GridBlock:
    """
     Electrical grid connection and pricing model.
    
     This block contains all information related to grid interaction,
     including electricity pricing, emissions factors, and contractual
     parameters. It is used in on-grid and hybrid system configurations.
    
     Attributes:
         n_contracts (int):
             number of available trade contracts
             
         prices (np.ndarray):
             Time-dependent electricity prices (buying from grid).
             
         C02eqemissions (float):
             Grid-specific CO₂ emissions factor (e.g., gCO₂/kWh).
    
         fossil_fuel_ratio (float):
             Share of fossil energy in the main grid mix.
    
         pof_ratio (float):
             Ratio between primary and final energy in the grid system. 
    
         fixed_premium (np.ndarray):
             Fixed tariff components or subsidies applied to grid pricing.
    
         overrun (np.ndarray):
             Penalty or cost associated with exceeding contracted power.
    
         selling_price (np.ndarray):
             Electricity selling price back to the grid.
     """
    __slots__ = ("C02eqemissions","fossil_fuel_ratio","pof_ratio","prices","fixed_premium","overrun","selling_price","n_contract",)

    def __init__(self, C02eqemissions, fossil_fuel_ratio, pof_ratio,prices, fixed_premium, overrun, selling_price):

        self.n_contracts = np.int64(len(overrun))
        self.prices = prices
        self.C02eqemissions = C02eqemissions
        self.fossil_fuel_ratio = fossil_fuel_ratio
        self.pof_ratio = pof_ratio
        self.fixed_premium = fixed_premium
        self.overrun = overrun
        self.selling_price = selling_price

class GensetBlock:
    """
     Diesel generator (DG) system characteristics.
    
     This block contains all technical and economic parameters related to
     diesel generator operation, including cost structure, efficiency,
     fuel consumption, and emissions.
    
     It is used in off-grid configuration where backup
     generation is required.
    
     Attributes:
         fuel_cost (float):
             Cost of fuel per unit of energy.
    
         lifetime (float):
             Operational lifetime of the generator (years or equivalent cycles).
    
         unit_cost (float):
             Investment cost.
    
         maintenance_cost (float):
             Annual maintenance cost.
    
         fuel_consumption (np.ndarray):
             Fuel consumption curve as a function of load or operating point.
    
         eroi (float):
             Energy Return on Investment of the generator.
    
         emissions (float):
             CO₂-equivalent emissions factor (e.g., gCO₂/kWh).
     """
    __slots__ = (
        "fuel_cost",
        "lifetime",
        "unit_cost",
        "maintenance_cost",
        "fuel_consumption",
        "eroi",
        "emissions",)

    def __init__(self, fuel_cost, lifetime, unit_cost,
                 maintenance_cost, fuel_consumption, eroi, C02eqemissions):

        self.fuel_cost = fuel_cost
        self.lifetime = lifetime
        self.unit_cost = unit_cost
        self.maintenance_cost = maintenance_cost
        self.fuel_consumption = fuel_consumption
        self.eroi = eroi
        self.C02eqemissions = C02eqemissions
        
class TrackingOpeBlock:
    """
    Operator tracking and diagnostics.
    
    Stores optional monitoring data for analyzing the behavior
    of genetic operators during optimization.
    
    Attributes:
        tracking_operators (integer):
            Defines the need for tracking (1:Yes, .
    """
    __slots__ = ("tracking_operators",)

    def __init__(self, tracking_operators):
        self.tracking_operators = tracking_operators


class Environment:
    """
     Core optimization environment used by the genetic algorithm.
    
     This class aggregates all structured data blocks required for the
     execution of the optimization process. It acts as the central
     container passed to the JIT-compiled genetic algorithm.
    
     The environment is intentionally designed to be:
     - Lightweight (no dynamic attributes)
     - Numba-friendly (static structure via __slots__)
     - Modular (separation into independent blocks)
     - Immutable in practice during optimization
    
     Attributes:
         optimization (OptimBlock):
             Definition of the optimization problem (objective, constraints).
    
         hyperparameters (HyperparametersBlock):
             Genetic algorithm configuration parameters (ERMESS RESEARCH).
             
         hyperparameters_pro (HyperparametersBlock):
             Genetic algorithm configuration parameters (ERMESS PRO).        
    
         config (ConfigurationBlock):
             Global system configuration (e.g., connectivity type).
    
         time (TimeParametersBlock):
             Temporal structure of the simulation.
    
         production (ProductionBlock):
             Production assets and generation profiles.
    
         loads (LoadBlock):
             Electrical demand profiles (including DSM components).
    
         storage (StorageBlock):
             Storage system characteristics.
    
         grid (GridBlock or None):
             Grid interaction data (None for off-grid systems).
    
         genset (GensetBlock or None):
             Diesel generator system data (None for on-grid systems).
    
         tracking (TrackingOpeBlock):
             Optional diagnostics and operator tracking data.
    
     Notes:
         This class is a pure data container with no business logic.
         It is designed to be passed directly into the optimization engine.
     """
    
    __slots__ = ("storage","time","production","loads","grid", "genset", "optimization","hyperparameters","hyperparameters_pro","config","postprocess_config","tracking",)

    def __init__(self, optimization,hyperparameters, hyperparameters_pro, config,time,production, loads,storage,grid, genset,postprocess_config,tracking):

        self.optimization = optimization
        self.hyperparameters = hyperparameters
        self.hyperparameters_pro = hyperparameters_pro
        self.config = config
        self.time = time
        self.production = production
        self.loads = loads
        self.storage = storage
        self.grid = grid
        self.genset = genset
        self.postprocess_config = postprocess_config
        self.tracking = tracking     


def build_environment(structured_data):
    
    """
    Build a Numba-compatible optimization environment from structured input data.
    
    This function transforms high-level structured input data into a
    fully-formed `Environment` object composed of modular blocks.
    It acts as the compilation layer between preprocessing and the
    optimization engine.
    
    The function performs the following steps:
        1. Extract and build storage system block.
        2. Construct temporal configuration.
        3. Build production assets and time-series.
        4. Build load profiles (including DSM aggregation).
        5. Define optimization problem parameters.
        6. Select hyperparameter set depending on optimization mode.
        7. Configure system connectivity (grid or genset).
        8. Attach optional tracking data.
        9. Assemble final Environment object.
    
    Parameters:
        structured_data (object):
            Fully parsed input structure containing all required fields:
            - storage
            - time
            - production
            - load
            - optimization
            - hyperparameters / hyperparameterspro
            - grid (optional)
            - genset (optional)
            - tracking
    
    Returns:
        Environment:
            Fully constructed environment ready to be passed into the
            genetic algorithm or JIT-compiled optimization engine.
    
    Raises:
        AttributeError:
            If required fields are missing in structured_data.
    
        ValueError:
            If optimization type is not recognized.
    
    Notes:
        This function is the single entry point for converting high-level
        inputs into a computationally optimized environment. It should be
        called once before launching the optimization process.
    """

    storage = StorageBlock(structured_data.storage.characteristics_num,structured_data.storage.n_store,structured_data.storage.techs,)

    time = TimeParametersBlock(structured_data.time.time_resolution,structured_data.time.duration_years,len(structured_data.time.datetime),structured_data.time.n_days)

    production = ProductionBlock(
        structured_data.production.characteristics_num,
        structured_data.production.ids,
        structured_data.production.capacities,
        structured_data.production.groups,
        structured_data.production.current_prod,
        structured_data.production.unit_prods,
    )

    loads = LoadBlock(
        structured_data.load.non_movable,
        structured_data.load.yearly_movable,
        structured_data.load.daily_movable,
        structured_data.time.time_resolution
    )

    optimization = OptimBlock(
        structured_data.optimization.constraint_num,
        structured_data.optimization.constraint_level,
        structured_data.optimization.criterion_num,
        structured_data.connexion,
    )

    if structured_data.optimization.type_optim=='research' : 
        hyperparameters = HyperparametersBlock(
        structured_data.hyperparameters.r_cross,
        structured_data.hyperparameters.n_iter,
        structured_data.hyperparameters.n_pop,
        structured_data.hyperparameters.operators_num,
        structured_data.hyperparameters.cost_constraint,
        structured_data.hyperparameters.elitism_probability,)
    
    hyperparameters_pro = HyperparametersBlock(
    structured_data.hyperparameterspro.r_cross,
    structured_data.hyperparameterspro.n_iter,
    structured_data.hyperparameterspro.n_pop,
    structured_data.hyperparameterspro.operators_num,
    structured_data.hyperparameterspro.cost_constraint,
    structured_data.hyperparameterspro.elitism_probability,)

    dispatching = DispatchingBlock(
        structured_data.dispatching.Defined_items,
        structured_data.dispatching.energy_use_coefficient,
        structured_data.dispatching.Y_DSM_minimum_levels,
        structured_data.dispatching.D_DSM_minimum_levels,
        structured_data.dispatching.DG_strategy,
        structured_data.dispatching.DG_min_runtime,
        structured_data.dispatching.DG_min_production,
        structured_data.dispatching.Discharge_order,
        structured_data.dispatching.Overlaps)

    tracking = TrackingOpeBlock(structured_data.tracking)

    grid = structured_data.grid if structured_data.grid is not None else None
    genset = structured_data.genset if structured_data.genset is not None else None
    postprocess_config = structured_data.postProcessConfig

    return Environment(
        optimization,
        hyperparameters,
        hyperparameters_pro,
        dispatching,
        time,
        production,
        loads,
        storage,        
        grid,
        genset,
        postprocess_config,
        tracking
    )