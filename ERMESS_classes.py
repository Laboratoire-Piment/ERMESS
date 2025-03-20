# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 11:12:08 2025

@author: jlegalla
"""

import numpy as np
from numba.experimental import jitclass
from numba import int64, float64

@jitclass([
           ('storage_characteristics', float64[:,:]),
           ('time_resolution', float64),
           ('n_store', int64),
           ('duration_years', float64),
           ('specs_num', float64[:,:]),
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
           ('prod_C', float64[:]),
           ('prods_U', float64[:,:]),
           ('Volums_prod', float64[:]),
           ('Bounds_prod', int64[:]),
           ('constraint_num', int64),
           ('constraint_level', float64),
           ('criterion_num', int64),
           ('cost_constraint', float64),
           ('r_cross', float64),
           ('n_iter', int64),
           ('hyperparameters_operators', float64[:,:]),
           ('tracking_ope',int64)
           ])
class Environnement(object):
#    def __init__(self,storage_characteristics,time_resolution,n_store,duration_years,specs_prod,n_bits,prices_num,fixed_premium,Overrun,Selling_price,Non_movable_load,Y_movable_load,D_movable_load,Main_grid_emissions,prod_C,prods_U,Volums_prod,Bounds_prod,Constraint, Constraint_level,criterion,cost_constraint,r_cross,n_iter,hyperparameters_operators):
    def __init__(self,storage_characteristics,time_resolution,n_store,duration_years,specs_num,n_bits,prices_num,fixed_premium,Overrun,Selling_price,Non_movable_load,total_Y_Movable_load,total_D_Movable_load,D_DSM_indexes,Main_grid_emissions,prod_C,prods_U,Volums_prod,Bounds_prod,constraint_num,constraint_level,criterion_num,cost_constraint,r_cross,n_iter,hyperparameters_operators,tracking_ope):
        self.storage_characteristics = storage_characteristics
        self.time_resolution = time_resolution
        self.n_store = n_store
        self.duration_years = duration_years
        self.specs_num = specs_num
        self.n_bits = n_bits
        self.prices_num = prices_num
        self.fixed_premium = fixed_premium
        self.Overrun = Overrun
        self.n_contract = np.int64(len(self.Overrun))
        self.Selling_price = Selling_price
        self.Non_movable_load = Non_movable_load
        self.total_Y_Movable_load = total_Y_Movable_load
#        self.total_D_Movable_load = np.add.reduceat(D_movable_load, range(0, len(D_movable_load), int(time_resolution*24)))
        self.total_D_Movable_load =total_D_Movable_load
#        self.D_DSM_indexes = np.int64(np.where(self.total_D_Movable_load != 0))
        self.D_DSM_indexes = D_DSM_indexes
        self.Main_grid_emissions = Main_grid_emissions
        self.prod_C = prod_C
        self.prods_U = prods_U
        self.Volums_prod = Volums_prod
        self.Bounds_prod = Bounds_prod
        self.constraint_num = constraint_num
        self.constraint_level = constraint_level
        self.criterion_num = criterion_num
        self.cost_constraint = cost_constraint
        self.n_iter = n_iter
        self.r_cross = r_cross
        self.hyperparameters_operators = hyperparameters_operators  
        self.tracking_ope = tracking_ope        

class Non_JIT_Environnement():
#    def __init__(self,storage_characteristics,time_resolution,n_store,duration_years,specs_prod,n_bits,prices_num,fixed_premium,Overrun,Selling_price,Non_movable_load,Y_movable_load,D_movable_load,Main_grid_emissions,prod_C,prods_U,Volums_prod,Bounds_prod,Constraint, Constraint_level,criterion,cost_constraint,r_cross,n_iter,hyperparameters_operators):
    def __init__(self,storage_characteristics,time_resolution,n_store,duration_years,specs_num,n_bits,prices_num,fixed_premium,Overrun,Selling_price,Non_movable_load,Y_movable_load,D_movable_load,Main_grid_emissions,prod_C,prods_U,Volums_prod,Bounds_prod,constraint_num,constraint_level,criterion_num,cost_constraint,r_cross,n_iter,hyperparameters_operators,tracking_ope):
        self.storage_characteristics = storage_characteristics
        self.time_resolution = time_resolution
        self.n_store = n_store
        self.duration_years = duration_years
        self.specs_num = specs_num
        self.n_bits = n_bits
        self.prices_num = prices_num
        self.fixed_premium = fixed_premium
        self.Overrun = Overrun
        self.n_contract = np.int64(len(self.Overrun))
        self.Selling_price = Selling_price
        self.Non_movable_load = Non_movable_load
        self.total_Y_Movable_load = np.sum(Y_movable_load)
#        self.total_D_Movable_load = np.add.reduceat(D_movable_load, range(0, len(D_movable_load), int(time_resolution*24)))
        self.total_D_Movable_load =np.array([np.sum(D_movable_load[np.arange(np.int32(i*self.time_resolution*24),(np.int32((i+1)*self.time_resolution*24)))]) for i in range(0, np.int32(len(D_movable_load)/self.time_resolution/24))],dtype=np.float64)
#        self.D_DSM_indexes = np.int64(np.where(self.total_D_Movable_load != 0))
        self.D_DSM_indexes = np.where(self.total_D_Movable_load != np.float64(0))[0]
        self.Main_grid_emissions = Main_grid_emissions
        self.prod_C = prod_C
        self.prods_U = prods_U
        self.Volums_prod = Volums_prod
        self.Bounds_prod = Bounds_prod
        self.constraint_num = constraint_num
        self.constraint_level = constraint_level
        self.criterion_num = criterion_num
        self.cost_constraint = cost_constraint
        self.n_iter = n_iter
        self.r_cross = r_cross
        self.hyperparameters_operators = hyperparameters_operators 
        self.tracking_ope = tracking_ope

#        if self.criterion_num=="LCOE":self.fun_fitness = cost_scenario_LCOE
#        elif self.criterion=="Self-consumption": self.fun_fitness =Self_consumption_cost
#        elif self.criterion=="Maximum power from grid":self.fun_fitness =Max_power_cost
#        elif self.criterion=="Energy losses":self.fun_fitness =Losses_cost
#        elif self.criterion=="Capacity factor":self.fun_fitness =Capacity_factor_cost
#        elif self.criterion=="Autonomy":self.fun_fitness =Autonomy_cost
#        elif self.criterion=="Saved CO2 emissions":self.fun_fitness =CO2_emissions_cost
#        elif self.criterion=="Saved fossil fuel consumption":self.fun_fitness =fossil_consumption_cost
        
#    def fitness(self,gene):
#        return(self.fun_fitness(gene,self.storage_characteristics,self.time_resolution,self.n_store,self.duration_years,self.specs_prod,self.grid_prices,self.fixed_premium,self.Overrun,self.Selling_price,self.Non_movable_load ,self.Main_grid_emissions,self.prod_C,self.prods_U,self.Bounds_prod,self.constraint,self.constraint_level,self.cost_constraint))

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
class Individual(object):
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
        return Individual(self.production_set.copy(),self.storage_sum.copy(),self.storage_TS.copy(),self.contract,self.Y_DSM.copy(),self.D_DSM.copy(),self.fitness,self.trades.copy())

        
class Non_JIT_Individual():
    def __init__(self,production_set,storage_sum,storage_TS,contract,Y_DSM,D_DSM,fitness,trades):
        self.production_set = production_set
        self.storage_sum = storage_sum
        self.storage_TS = storage_TS
        self.contract = contract
        self.Y_DSM = Y_DSM
        self.D_DSM = D_DSM        
        self.fitness = fitness
        self.trades = trades
        
def jitting_pop(pop):
    jitted_pop=[]
    for ind in pop:
        jitted_pop.append(Individual(np.int64(ind.production_set),np.array(ind.storage_sum,dtype=np.float64),np.float64(ind.storage_TS),np.array(ind.contract,dtype=np.int64),np.float64(ind.Y_DSM),np.float64(ind.D_DSM),np.float64(ind.fitness),np.array(ind.trades,dtype=np.float64)))
    return(jitted_pop)

def unjitting_pop(jitted_pop):
    pop=[]
    for ind in jitted_pop:
        pop.append(Non_JIT_Individual(ind.production_set,ind.storage_sum,ind.storage_TS,ind.contract,ind.Y_DSM,ind.D_DSM,ind.fitness,ind.trades))
    return(pop)

