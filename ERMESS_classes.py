# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 11:12:08 2025

@author: jlegalla
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

class Non_JIT_Environnement():
    def __init__(self,storage_characteristics,time_resolution,n_store,duration_years,specs_num,groups,n_bits,prices_num,fixed_premium,Overrun,Selling_price,Non_movable_load,Y_movable_load,D_movable_load,Main_grid_emissions,Grid_Fossil_fuel_ratio,Main_grid_PoF_ratio,prod_C,prods_U,Volums_prod,Bounds_prod,DG_fuel_cost,DG_lifetime,DG_unit_cost,DG_maintenance_cost,DG_fuel_consumption,DG_EROI,fuel_CO2eq_emissions,constraint_num,constraint_level,criterion_num,cost_constraint,r_cross,n_iter,hyperparameters_operators,type_optim,Connexion,Defined_items,tracking_ope):
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
        self.total_Y_Movable_load = np.sum(Y_movable_load)
        self.total_D_Movable_load =np.array([np.sum(D_movable_load[np.arange(np.int32(i*self.time_resolution*24),(np.int32((i+1)*self.time_resolution*24)))]) for i in range(0, np.int32(len(D_movable_load)/self.time_resolution/24))],dtype=np.float64)
        self.D_DSM_indexes = np.where(self.total_D_Movable_load != np.float64(0))[0]
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
    jitted_env=Environnement(np.float64(non_jit_environnement.storage_characteristics),np.float64(non_jit_environnement.time_resolution),np.int64(non_jit_environnement.n_store),np.float64(non_jit_environnement.duration_years),np.float64(non_jit_environnement.specs_num),np.int64(non_jit_environnement.n_bits),np.float64(non_jit_environnement.prices_num),np.float64(non_jit_environnement.fixed_premium),np.float64(non_jit_environnement.Overrun),np.float64(non_jit_environnement.Selling_price),np.float64(non_jit_environnement.Non_movable_load),np.float64(non_jit_environnement.total_Y_Movable_load),np.float64(non_jit_environnement.total_D_Movable_load),np.int64(non_jit_environnement.D_DSM_indexes),np.float64(non_jit_environnement.Main_grid_emissions),np.float64(non_jit_environnement.prod_C),np.float64(non_jit_environnement.prods_U),np.float64(non_jit_environnement.Volums_prod),np.int64(non_jit_environnement.Bounds_prod),np.int64(non_jit_environnement.constraint_num),np.float64(non_jit_environnement.constraint_level),np.int64(non_jit_environnement.criterion_num),np.float64(non_jit_environnement.cost_constraint),np.float64(non_jit_environnement.r_cross),np.int64(non_jit_environnement.n_iter),np.float64(non_jit_environnement.hyperparameters_operators))
    return(jitted_env)

@jitclass([
           ('production_set', int64[:]),
           ('storage_sum', float64[:]),
           ('storage_TS', float64[:,:]),
           ('contract', int64),
           ('Y_DSM', float64[:]),
           ('D_DSM', float64[:,:]),
           ('trades', float64[:]),
           ('fitness', float64),
           ])
class Individual_res(object):
    def __init__(self,production_set,storage_sum,storage_TS,contract,Y_DSM,D_DSM,fitness,trades):
        self.production_set = production_set
        self.storage_sum = storage_sum
        self.storage_TS = storage_TS
        self.contract = contract
        self.Y_DSM = Y_DSM
        self.D_DSM = D_DSM
        self.fitness = fitness
        self.trades = trades
        
    def copy(self):
        return Individual_res(self.production_set.copy(),self.storage_sum.copy(),self.storage_TS.copy(),self.contract,self.Y_DSM.copy(),self.D_DSM.copy(),self.fitness,self.trades.copy())

        
class Non_JIT_Individual_res():
    def __init__(self,production_set,storage_sum,storage_TS,contract,Y_DSM,D_DSM,fitness,trades):
        self.production_set = production_set
        self.storage_sum = storage_sum
        self.storage_TS = storage_TS
        self.contract = contract
        self.Y_DSM = Y_DSM
        self.D_DSM = D_DSM        
        self.fitness = fitness
        self.trades = trades
    
    def copy(self):
        return Non_JIT_Individual_res(self.production_set.copy(),self.storage_sum.copy(),self.storage_TS.copy(),self.contract,self.Y_DSM.copy(),self.D_DSM.copy(),self.fitness,self.trades.copy())

        
def jitting_pop_res(pop):
    jitted_pop=[]
    for ind in pop:
        jitted_pop.append(Individual_res(np.int64(ind.production_set),np.array(ind.storage_sum,dtype=np.float64),np.float64(ind.storage_TS),np.int64(ind.contract),np.float64(ind.Y_DSM),np.float64(ind.D_DSM),np.float64(ind.fitness),np.array(ind.trades,dtype=np.float64)))
    return(jitted_pop)

def unjitting_pop_res(jitted_pop):
    pop=[]
    for ind in jitted_pop:
        pop.append(Non_JIT_Individual_res(ind.production_set,ind.storage_sum,ind.storage_TS,ind.contract,ind.Y_DSM,ind.D_DSM,ind.fitness,ind.trades))
    return(pop)

@jitclass([
           ('production_set', int64[:]),
           ('contract', int64),
           ('PMS_strategy', types.string),
           ('PMS_discharge_order', int64[:]),
           ('energy_use_repartition_DSM', float64),
           ('PMS_taking_over', float64[:,:]),
           ('PMS_D_DSM_min_levels', float64[:]),
           ('PMS_Y_DSM_min_levels', float64[:]),
           ('PMS_DG_min_runtime', int64),
           ('PMS_DG_min_production', float64),
           ('storages', float64[:,:]),
           ('fitness', float64),
           ])
class Individual_pro(object):
    def __init__(self,production_set,contract,PMS_strategy,PMS_discharge_order,energy_use_repartition_DSM,PMS_taking_over,PMS_D_DSM_min_levels,PMS_Y_DSM_min_levels,PMS_DG_min_runtime,PMS_DG_min_production,storages,fitness):
        self.production_set = production_set
        self.contract = contract
        self.PMS_strategy = PMS_strategy
        self.PMS_discharge_order = PMS_discharge_order
        self.energy_use_repartition_DSM = energy_use_repartition_DSM
        self.PMS_taking_over = PMS_taking_over
        self.PMS_D_DSM_min_levels = PMS_D_DSM_min_levels
        self.PMS_Y_DSM_min_levels = PMS_Y_DSM_min_levels
        self.PMS_DG_min_runtime = PMS_DG_min_runtime        
        self.PMS_DG_min_production = PMS_DG_min_production       
        self.storages = storages
        self.fitness = fitness
    
    def copy(self):
        return Individual_pro(self.production_set.copy(),self.contract,self.PMS_strategy,self.PMS_discharge_order.copy(),self.energy_use_repartition_DSM,self.PMS_taking_over.copy(),self.PMS_D_DSM_min_levels.copy(),self.PMS_Y_DSM_min_levels.copy(),self.PMS_DG_min_runtime,self.PMS_DG_min_production,self.storages.copy(),self.fitness)

        
class Non_JIT_Individual_pro():
    def __init__(self,production_set,contract,PMS_strategy,PMS_discharge_order,energy_use_repartition_DSM,PMS_taking_over,PMS_D_DSM_min_levels,PMS_Y_DSM_min_levels,PMS_DG_min_runtime,PMS_DG_min_production,storages,fitness):
        self.production_set = production_set
        self.contract = contract
        self.PMS_strategy = PMS_strategy
        self.PMS_discharge_order = PMS_discharge_order
        self.energy_use_repartition_DSM = energy_use_repartition_DSM
        self.PMS_taking_over = PMS_taking_over
        self.PMS_D_DSM_min_levels = PMS_D_DSM_min_levels
        self.PMS_Y_DSM_min_levels = PMS_Y_DSM_min_levels
        self.PMS_DG_min_runtime = PMS_DG_min_runtime
        self.PMS_DG_min_production = PMS_DG_min_production 
        self.storages = storages
        self.fitness = fitness
        
    def copy(self):
        return Non_JIT_Individual_pro(self.production_set.copy(),self.contract,self.PMS_strategy,self.PMS_discharge_order.copy(),self.energy_use_repartition_DSM,self.PMS_taking_over.copy(),self.PMS_D_DSM_min_levels.copy(),self.PMS_Y_DSM_min_levels.copy(),self.PMS_DG_min_runtime,self.PMS_DG_min_production,self.storages.copy(),self.fitness)

        
def jitting_pop_pro(pop):
    jitted_pop=[]
    for ind in pop:
        jitted_pop.append(Individual_pro(np.int64(ind.production_set),np.int64(ind.contract),ind.PMS_strategy,np.array(ind.PMS_discharge_order,dtype=np.int64),np.float64(ind.energy_use_repartition_DSM),np.array(ind.PMS_taking_over,dtype=np.float64),np.array(ind.PMS_D_DSM_min_levels,dtype=np.float64),np.array(ind.PMS_Y_DSM_min_levels,dtype=np.float64),np.int64(ind.PMS_DG_min_runtime),np.float64(ind.PMS_DG_min_production),np.array(ind.storages,dtype=np.float64),np.float64(ind.fitness)))
    return(jitted_pop)

def unjitting_pop_pro(jitted_pop):
    pop=[]
    for ind in jitted_pop:
        pop.append(Non_JIT_Individual_pro(np.int64(ind.production_set),np.array(ind.contract,dtype=np.int64),(ind.PMS_strategy),np.array(ind.PMS_discharge_order,dtype=np.int64),np.float64(ind.energy_use_repartition_DSM),np.array(ind.PMS_taking_over,dtype=np.float64),np.array(ind.PMS_D_DSM_min_levels,dtype=np.float64),np.array(ind.PMS_Y_DSM_min_levels,dtype=np.float64),np.float64(ind.PMS_DG_min_runtime),np.float64(ind.PMS_DG_min_production),np.array(ind.storages,dtype=np.float64),np.float64(ind.fitness)))
    return(pop)