# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 11:12:08 2025

@author: JoPHOBEA
"""

import numpy as np
from numba.experimental import jitclass
from numba import int64, float64, types

@jitclass([
           ('storage_characteristics', float64[:,:]),
           ('time_resolution', float64),
           ('n_store', int64),
           ('duration_years', float64),
           ('specs_num', float64[:,:]),
           ('groups', types.ListType((types.int64[:]))),
           ('n_bits', float64),
           ('prices_num', float64[:,:]),
           ('fixed_premium', float64[:]),
           ('Overrun', float64[:]),
           ('n_contract', int64),
           ('Selling_price', float64[:]),
           ('Non_movable_load', float64[:]),
           ('Y_movable_load', float64[:]),
           ('total_Y_Movable_load', float64),
           ('D_movable_load', float64[:]),
           ('total_D_Movable_load', float64[:]),
           ('D_DSM_indexes', int64[:]),
           ('Main_grid_emissions', float64),
           ('Grid_Fossil_fuel_ratio', float64),
           ('Main_grid_PoF_ratio',float64),
           ('prod_C', float64[:]),
           ('prods_U', float64[:,:]),
           ('Volums_prod', float64[:]),
           ('Bounds_prod', int64[:]),
           ('DG_fuel_cost', float64),
           ('DG_lifetime', float64),
           ('DG_unit_cost',float64 ), 
           ('DG_maintenance_cost', float64), 
           ('DG_fuel_consumption', float64[:]),
           ('DG_EROI', float64),
           ('fuel_CO2eq_emissions', float64),
           ('constraint_num', int64),
           ('constraint_level', float64),
           ('criterion_num', int64),
           ('cost_constraint', float64),
           ('r_cross', float64),
           ('n_iter', int64),
           ('hyperparameters_operators', float64[:,:]),           
           ('type_optim',types.string),
           ('Connexion',types.string), 
           ('Defined items', int64[:]), 
           ('tracking_ope',int64)
           ])
class Environnement(object):
    """
    Class defining the environment for the optimization problem.
    This class is targeted for Numba JIT compilation.
    
    Attributes:
        storage_characteristics (np.ndarray): Characteristics of the storage units, shape (n_store, n_features), dtype float64.
        time_resolution (float): Time resolution of the simulation in hours.
        n_store (int): Number of storage units.
        duration_years (float): Duration of the simulation in years.
        specs_num (np.ndarray): Specifications number, shape (n_specs, n_features), dtype float64.
        groups (list of np.ndarray): Groups of storage units.
        n_bits (float): Number of bits.
        prices_num (np.ndarray): Prices.
        fixed_premium (np.ndarray): Fixed premium.
        Overrun (np.ndarray): Overrun values.
        n_contract (int): Number of contracts.
        Selling_price (np.ndarray): Selling price.
        Non_movable_load (np.ndarray): Non-movable load profile.
        Y_movable_load (np.ndarray): Yearly movable load.
        total_Y_Movable_load (float): Total yearly movable load.
        D_movable_load (np.ndarray): Daily movable load.
        total_D_Movable_load (float): Total daily movable load.
        D_DSM_indexes (np.ndarray): Indexes for Demand Side Management.
        Main_grid_emissions (float): Emissions from the main grid.
        Grid_Fossil_fuel_ratio (float): Ratio of fossil fuel in the grid.
        Main_grid_PoF_ratio (float): Probability of Failure ratio of the main grid.
        prod_C (np.ndarray): Production capacity.
        prods_U (np.ndarray): Production units.
        Volums_prod (np.ndarray): Production volumes.
        Bounds_prod (np.ndarray): Production bounds.
        DG_fuel_cost (float): Diesel Generator fuel cost.
        DG_lifetime (float): Diesel Generator lifetime.
        DG_unit_cost (float): Diesel Generator unit cost.
        DG_maintenance_cost (float): Diesel Generator maintenance cost.
        DG_fuel_consumption (np.ndarray): Diesel Generator fuel consumption.
        DG_EROI (float): Diesel Generator Energy Return on Investment.
        fuel_CO2eq_emissions (float): CO2 equivalent emissions from fuel.
        constraint_num (int): Number of constraints.
        constraint_level (float): Level of constraints.
        criterion_num (int): Number of criteria.
        cost_constraint (float): Cost constraint.
        r_cross (float): Crossover rate.
        n_iter (int): Number of iterations.
        hyperparameters_operators (np.ndarray): Hyperparameters for genetic operators.
        type_optim (str): Type of optimization.
        Connexion (str): Connection type.
        Defined_items (np.ndarray): Defined items.
        tracking_ope (int): Tracking operation.
    """
    def __init__(self,storage_characteristics,time_resolution,n_store,duration_years,specs_num,groups,n_bits,prices_num,fixed_premium,Overrun,Selling_price,Non_movable_load,total_Y_Movable_load,total_D_Movable_load,D_DSM_indexes,Main_grid_emissions,Grid_Fossil_fuel_ratio,Main_grid_PoF_ratio,prod_C,prods_U,Volums_prod,Bounds_prod,DG_fuel_cost,DG_lifetime,DG_unit_cost,DG_maintenance_cost,DG_fuel_consumption,DG_EROI,fuel_CO2eq_emissions,constraint_num,constraint_level,criterion_num,cost_constraint,r_cross,n_iter,hyperparameters_operators,type_optim,Connexion,Defined_items,tracking_ope):
        self.storage_characteristics = storage_characteristics
        self.time_resolution = time_resolution
        self.n_store = n_store
        self.duration_years = duration_years
        self.specs_num = specs_num
        self.groups = groups
        self.n_bits = n_bits
        self.prices_num = prices_num
        self.fixed_premium = fixed_premium
        self.Overrun = Overrun
        self.n_contract = np.int64(len(self.Overrun))
        self.Selling_price = Selling_price
        self.Non_movable_load = Non_movable_load
        self.total_Y_Movable_load = total_Y_Movable_load
        self.total_D_Movable_load =total_D_Movable_load
        self.D_DSM_indexes = D_DSM_indexes
        self.Main_grid_emissions = Main_grid_emissions
        self.Grid_Fossil_fuel_ratio=Grid_Fossil_fuel_ratio
        self.Main_grid_PoF_ratio=Main_grid_PoF_ratio
        self.prod_C = prod_C
        self.prods_U = prods_U
        self.Volums_prod = Volums_prod
        self.Bounds_prod = Bounds_prod
        self.DG_fuel_cost=DG_fuel_cost
        self.DG_lifetime=DG_lifetime
        self.DG_unit_cost=DG_unit_cost
        self.DG_maintenance_cost=DG_maintenance_cost
        self.DG_fuel_consumption=DG_fuel_consumption
        self.DG_EROI=DG_EROI
        self.fuel_CO2eq_emissions=fuel_CO2eq_emissions
        self.constraint_num = constraint_num
        self.constraint_level = constraint_level
        self.criterion_num = criterion_num
        self.cost_constraint = cost_constraint
        self.n_iter = n_iter
        self.r_cross = r_cross
        self.hyperparameters_operators = hyperparameters_operators  
        self.type_optim = type_optim 
        self.Connexion = Connexion 
        self.Defined_items = Defined_items
        self.tracking_ope = tracking_ope          

 

def jitting_environment(non_jit_environnement):
    """
    Converts a Non_JIT_Environnement object to a JIT-compatible Environnement object.
    
    Args:
        non_jit_environnement (Non_JIT_Environnement): The non-JIT environment object to convert.
    
    Returns:
        Environnement: The JIT-compatible environment object.
    """
    jitted_env=Environnement(np.float64(non_jit_environnement.storage_characteristics),np.float64(non_jit_environnement.time_resolution),np.int64(non_jit_environnement.n_store),np.float64(non_jit_environnement.duration_years),np.float64(non_jit_environnement.specs_num),np.int64(non_jit_environnement.n_bits),np.float64(non_jit_environnement.prices_num),np.float64(non_jit_environnement.fixed_premium),np.float64(non_jit_environnement.Overrun),np.float64(non_jit_environnement.Selling_price),np.float64(non_jit_environnement.Non_movable_load),np.float64(non_jit_environnement.total_Y_Movable_load),np.float64(non_jit_environnement.total_D_Movable_load),np.int64(non_jit_environnement.D_DSM_indexes),np.float64(non_jit_environnement.Main_grid_emissions),np.float64(non_jit_environnement.prod_C),np.float64(non_jit_environnement.prods_U),np.float64(non_jit_environnement.Volums_prod),np.int64(non_jit_environnement.Bounds_prod),np.int64(non_jit_environnement.constraint_num),np.float64(non_jit_environnement.constraint_level),np.int64(non_jit_environnement.criterion_num),np.float64(non_jit_environnement.cost_constraint),np.float64(non_jit_environnement.r_cross),np.int64(non_jit_environnement.n_iter),np.float64(non_jit_environnement.hyperparameters_operators))
    return(jitted_env)

