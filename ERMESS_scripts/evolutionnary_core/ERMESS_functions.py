# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 15:32:46 2026

@author: JoPHOBEA
"""
import numpy as np
from numba.experimental import jitclass
from numba import float64, int64
from enum import IntEnum


from .ERMESS_functions_research import jitting_pop_res 
from .ERMESS_functions_pro import jitting_pop_pro
from ERMESS_scripts.cost import ERMESS_cost_functions as Cfc
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
    ('Connexion', int64),
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
    ('specs_prod', float64[:,:]),
    ('capacities', int64[:]),
    ('specs_storage', float64[:,:]),
    ('n_store', int64),
]

@jitclass(REN_specs_systems)
class RENSystemsParams:
    def __init__(self, current_production, unit_productions, specs_prod, capacities, specs_storage, n_store):

        self.current_production = current_production
        self.unit_productions = unit_productions
        self.specs_prod = specs_prod
        self.capacities = capacities
        self.specs_storage = specs_storage
        self.n_store = n_store
        
mutation_extra_params = [
    ('groups_production', int64[:,:] ),
    ('groups_size', int64[:] ),
    ('unit_production_volumes', float64[:]),
    ('D_DSM_indexes', int64[:]),
    ('hyperparameters_operators', float64[:,:]),
    ('defined_items', int64[:]),
    ('n_contracts', int64),
]

@jitclass(mutation_extra_params)
class mutationParams:
    def __init__(self, groups_production,groups_size, unit_production_volumes,D_DSM_indexes, hyperparameters_operators, defined_items, n_contracts):

        self.groups_production = groups_production
        self.groups_size = groups_size
        self.unit_production_volumes = unit_production_volumes
        self.D_DSM_indexes = D_DSM_indexes
        self.hyperparameters_operators = hyperparameters_operators
        self.defined_items = defined_items
        self.n_contracts = n_contracts

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
        

def build_groups_matrix(groups_raw):
    max_len = max(len(g) for g in groups_raw)

    groups = np.zeros((len(groups_raw), max_len), dtype=np.int64)-1

    for i, g in enumerate(groups_raw):
        groups[i, :len(g)] = g

    return groups

def build_numba_params(Contexte,type_optim):
    
    pro_parameters = ProParams(Contexte.loads.total_D_movable,Contexte.loads.total_Y_movable)
    
    if Contexte.optimization.connexion == 'On-grid':
        connexion_num = GRID_ON
    elif Contexte.optimization.connexion == 'Off-grid':
        connexion_num = GRID_OFF
    else:
        raise ValueError("Unknown connexion type")
    
    if type_optim == 'pro':
        hyperparameters = Contexte.hyperparameters_pro
    elif type_optim == 'research':
        hyperparameters = Contexte.hyperparameters
    else:
        raise ValueError("Unknown optim type")
    
    global_parameters = GlobalParams(Contexte.time.n_bits,Contexte.time.time_resolution,Contexte.time.duration_years,Contexte.optimization.constraint_num,Contexte.optimization.constraint_level,hyperparameters.cost_constraint,connexion_num,Contexte.loads.non_movable)
    if not (Contexte.grid == None):
        grid_parameters = GridParams(Contexte.grid.prices, Contexte.grid.fixed_premium, Contexte.grid.overrun, Contexte.grid.selling_price, Contexte.grid.eqCO2emissions, Contexte.grid.fossil_fuel_ratio)      
    else :
        grid_parameters = GridParamsDummy()
    RENSystems_parameters = RENSystemsParams(Contexte.production.current_prod, Contexte.production.unit_prods, Contexte.production.specs_num, Contexte.production.capacities, Contexte.storage.characteristics, Contexte.storage.n_store)
    
    if not (Contexte.genset == None):
        Genset_parameters = GensetParams(Contexte.genset.fuel_cost, Contexte.genset.lifetime, Contexte.genset.unit_cost,Contexte.genset.maintenance_cost, Contexte.genset.fuel_consumption, Contexte.genset.fuel_CO2eq_emissions, Contexte.genset.EROI)
    else :
        Genset_parameters = GensetParamsDummy()
    
    groups = build_groups_matrix(Contexte.production.groups)
    groups_size = np.array([len(g) for g in Contexte.production.groups], dtype=np.int64)

    n_contracts = Contexte.grid.n_contracts if Contexte.grid!=None else 0
    mutation_parameters = mutationParams(groups, groups_size, Contexte.production.specs_num[:,ProdCharIdx.Volume], Contexte.loads.D_DSM_indexes,hyperparameters.operators_parameters, Contexte.config.defined_items, n_contracts )

    return pro_parameters, global_parameters, grid_parameters, RENSystems_parameters, Genset_parameters, mutation_parameters

def get_function_pro(index_criterion):
        PRO = {
                CRIT_LCOE: Cfc.LCOE_pro,
                CRIT_Annual_net_benefits: Cfc.Annual_net_benefits_pro,
                CRIT_NPV: Cfc.NPV_pro,
                CRIT_Self_sufficiency: Cfc.Self_sufficiency_pro,
                CRIT_Self_consumption: Cfc.Self_consumption_pro,
                CRIT_Autonomy: Cfc.Autonomy_pro,
                CRIT_EqCO2emissions: Cfc.eqCO2_emissions_pro,
                CRIT_Fossil_fuel_consumption: Cfc.Fossil_fuel_consumption_pro,
                CRIT_EROI: Cfc.EROI_pro,
                CRIT_Energy_losses: Cfc.Losses_pro,
                CRIT_Max_power_from_grid: Cfc.Max_import_power_pro
        }
        if index_criterion not in PRO:
            raise ValueError(f"Unknown criterion: {index_criterion}")
        return PRO[index_criterion]
        
def get_function_research(index_criterion):
    
        RESEARCH = {
                CRIT_LCOE: Cfc.LCOE_research,
                CRIT_Annual_net_benefits: Cfc.Annual_net_benefits_research,
                CRIT_NPV: Cfc.NPV_research,
                CRIT_Self_sufficiency: Cfc.Self_sufficiency_research,
                CRIT_Self_consumption: Cfc.Self_consumption_research,
                CRIT_Autonomy: Cfc.Autonomy_research,
                CRIT_EqCO2emissions: Cfc.eqCO2_emissions_research,
                CRIT_Fossil_fuel_consumption: Cfc.Fossil_fuel_consumption_research,
                CRIT_EROI: Cfc.EROI_research,
                CRIT_Energy_losses: Cfc.Losses_research,
                CRIT_Max_power_from_grid: Cfc.Max_import_power_research
        }
        if index_criterion not in RESEARCH:
            raise ValueError(f"Unknown criterion: {index_criterion}")
        return RESEARCH[index_criterion]


def find_cost_function_research(Context, global_parameters, grid_parameters, RENSystems_parameters, Genset_parameters):
    """
    Determines the right loss function to apply to the optimization problem (RESEARCH).
    
    Args:
        Contexte: Description of the constraints of the problem.
    
    Returns:
        int: ID of the appropriate cost function.
    """
    try:
        base_function = get_function_research(Context.optimization.criterion_num)
    except KeyError:
        raise ValueError("No proper optimisation criterion found!")
                
    def cost_function(ind):
        return base_function(ind,RENSystems_parameters,global_parameters,Genset_parameters,grid_parameters)

    return(cost_function )
    
def find_cost_function_pro(Context,pro_parameters, global_parameters, grid_parameters, RENSystems_parameters, Genset_parameters):
    """
    Determines the right loss function to apply to the optimization problem (PRO).
    
    Args:
        Contexte: Description of the constraints of the problem.
    
    Returns:
        int: ID of the appropriate cost function.
    """    
    try:
        base_function = get_function_pro(Context.optimization.criterion_num)
    except KeyError:
        raise ValueError("No proper optimisation criterion found!")
                
    def cost_function(ind):
        return base_function(ind,RENSystems_parameters,global_parameters,Genset_parameters,grid_parameters,pro_parameters)

    return(cost_function)


def fitness_list(inputs):
    """
    Updates the fitness values of a population.
    
    Args:
        inputs (population): Population of individuals.
    
    Returns:
        jitted_pop (List of individuals objects): Population with updated fitness values.
    """
    (population,Contexte)=tuple(inputs[i] for i in range(2))
    
    if (Contexte.type_optim == 'research' ):   
        jitted_pop = jitting_pop_res(population)
    elif (Contexte.type_optim == 'pro' ):
        jitted_pop = jitting_pop_pro(population)
                    
    fitness_function=find_cost_functions_research(Contexte)
    
    for j in range(len(jitted_pop)):
        jitted_pop[j]=(fitness_function(jitted_pop[j])[0])
       
    return(jitted_pop)  

# tournament selection
def selection_tournament(pop, k=3):
    """
    Select an individual from a sub-population using a random tournament.
    
    Args:
        pop (list): Population of individuals with a .fitness attribute.
        k (int): Tournament size (number of individuals randomly selected), default is 3.
    
    Returns:
        The winning individual from the tournament.
    """
 # first random selection
    selection_ix = np.random.randint(len(pop))
    for ix in np.random.randint(0, len(pop), k-1):
 # check if better (e.g. perform a tournament)
        if pop[ix].fitness < pop[selection_ix].fitness:
            selection_ix = ix
    return pop[selection_ix]
