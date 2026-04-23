# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 12:51:20 2026

@author: JoPHOBEA
"""

from ERMESS_scripts.utils import constraints as Cons
from ERMESS_scripts.evolutionnary_core import ERMESS_functions_pro as Efp
from ERMESS_scripts.evolutionnary_core import ERMESS_parallel_processing as ppGA
from ERMESS_scripts.data.indices import *


import copy
import numpy as np

def Time_downscaling (Transformed_variable,time_resolution,n_days):
    Daily_variable = np.array([sum(Transformed_variable[int(i*24*time_resolution):int((i+1)*24*time_resolution) ])/time_resolution for i in range(n_days)])   
    return(Daily_variable)

def Time_zooming (Original_variable,time_resolution,day):
    Zoomed_variable = np.array(Original_variable[(day*int(24*time_resolution)):((day+1)*int(24*time_resolution))])   
    return(Zoomed_variable)

def Context_time_resolution_downscaling(Context_initialisation_InDepth):
    
    Context_initialisation_InDepth_LR = copy.deepcopy(Context_initialisation_InDepth)
    LR_n_bits = Context_initialisation_InDepth.time.n_days
    Context_initialisation_InDepth_LR.time.n_bits = LR_n_bits
    Context_initialisation_InDepth_LR.time.time_resolution = np.float64(1/24)
    Context_initialisation_InDepth_LR.loads.non_movable = Time_downscaling(Context_initialisation_InDepth.loads.non_movable,Context_initialisation_InDepth.time.time_resolution,LR_n_bits)
    Context_initialisation_InDepth_LR.loads.Y_movable = Time_downscaling(Context_initialisation_InDepth.loads.Y_movable,Context_initialisation_InDepth.time.time_resolution,LR_n_bits)
    Context_initialisation_InDepth_LR.loads.D_movable = Time_downscaling(Context_initialisation_InDepth.loads.D_movable,Context_initialisation_InDepth.time.time_resolution,LR_n_bits)
    Context_initialisation_InDepth_LR.production.unit_prods = np.array([Time_downscaling(unit_prod_j, Context_initialisation_InDepth.time.time_resolution,LR_n_bits) for unit_prod_j in Context_initialisation_InDepth.production.unit_prods])
    Context_initialisation_InDepth_LR.production.specs_num[:,PROD_VOLUME] = np.sum(Context_initialisation_InDepth_LR.production.unit_prods,axis=1)
    Context_initialisation_InDepth_LR.production.current_prod = Time_downscaling(Context_initialisation_InDepth.production.current_prod,Context_initialisation_InDepth.time.time_resolution,LR_n_bits)
    Context_initialisation_InDepth_LR.production.current_prod = Time_downscaling(Context_initialisation_InDepth.production.current_prod,Context_initialisation_InDepth.time.time_resolution,LR_n_bits)
    if Context_initialisation_InDepth.grid is not None :
        Context_initialisation_InDepth_LR.grid.prices = np.array([Time_downscaling(Context_initialisation_InDepth.grid.prices,Context_initialisation_InDepth.time.time_resolution,LR_n_bits) for j in range(len(Context_initialisation_InDepth.grid.prices))] ) 
        Context_initialisation_InDepth_LR.grid.Selling_price = np.array([Time_downscaling(Context_initialisation_InDepth.grid.Selling_price,Context_initialisation_InDepth.time.time_resolution,LR_n_bits) for j in range(len(Context_initialisation_InDepth.grid.Selling_price))] ) 
    Context_initialisation_InDepth_LR.tracking.tracking_operators = False
    
    return(Context_initialisation_InDepth_LR)

def Context_time_resolution_zooming(Context_initialisation_InDepth,day):
    
    Context_initialisation_InDepth_HighRes = copy.deepcopy(Context_initialisation_InDepth)
    Context_initialisation_InDepth_HighRes.time.n_bits=np.int64(24*Context_initialisation_InDepth.time.time_resolution)
    Context_initialisation_InDepth_HighRes.time.time_resolution=np.float64(Context_initialisation_InDepth.time.time_resolution)
    Context_initialisation_InDepth_HighRes.time.duration_years = np.float64(1/365)
    Context_initialisation_InDepth_HighRes.loads.non_movable = Time_zooming(Context_initialisation_InDepth.loads.non_movable,Context_initialisation_InDepth.time.time_resolution,day)
    Context_initialisation_InDepth_HighRes.loads.Y_movable = Time_zooming(Context_initialisation_InDepth.loads.Y_movable,Context_initialisation_InDepth.time.time_resolution,day)
    Context_initialisation_InDepth_HighRes.loads.D_movable = Time_zooming(Context_initialisation_InDepth.loads.D_movable,Context_initialisation_InDepth.time.time_resolution,day)
    Context_initialisation_InDepth_HighRes.production.unit_prods = np.array([Time_zooming(Context_initialisation_InDepth.production.unit_prods[i],Context_initialisation_InDepth.time.time_resolution,day) for i in range(len(Context_initialisation_InDepth.production.unit_prods))])
    Context_initialisation_InDepth_HighRes.production.specs_num[:,PROD_VOLUME] = np.sum(Context_initialisation_InDepth_HighRes.production.unit_prods,axis=1)
    Context_initialisation_InDepth_HighRes.production.current_prod = Time_zooming(Context_initialisation_InDepth.production.current_prod,Context_initialisation_InDepth.time.time_resolution,day)
    if Context_initialisation_InDepth_HighRes.grid is not None :
        Context_initialisation_InDepth_HighRes.grid.prices = np.array([Time_zooming(Context_initialisation_InDepth.grid.prices[i],Context_initialisation_InDepth.time.time_resolution,day) for i in range(len(Context_initialisation_InDepth.grid.prices))])
        Context_initialisation_InDepth_HighRes.grid.Selling_price = np.array([Time_zooming(Context_initialisation_InDepth.grid.Selling_price[i],Context_initialisation_InDepth.time.time_resolution,day) for i in range(len(Context_initialisation_InDepth.grid.Selling_price))])   
    Context_initialisation_InDepth_HighRes.tracking.tracking_operators = False
    
    return(Context_initialisation_InDepth_HighRes)

def Initialisation_ERMESS_research(Contexte , structured_data):

    #Split the population in 2 equal groups : 0 : pro initialisation 1 : In-depth study of seasons and days
    population_sizes = (int(Contexte.hyperparameters.n_pop/2),int(Contexte.hyperparameters.n_pop/2)) if (Contexte.hyperparameters.n_pop%4==0) else (int((Contexte.hyperparameters.n_pop/2)-1),int((Contexte.hyperparameters.n_pop/2)+1)) 
    
    Context_initialisation_pro = copy.deepcopy(Contexte)
    Context_initialisation_pro.hyperparameters.n_pop = population_sizes[0]
    Context_initialisation_pro.tracking.tracking_operators = False
        
    ##============================================================================
    # 1. SOLVING THE PROBLEM WITH ERMESS PRO
    ##============================================================================
    pro_initial_populations =  [Efp.initial_population_pro(Context_initialisation_pro) for _ in range(structured_data.hyperparameters.n_core)]
    
    #Spreading the ERMESS PRO initial population 
    initialisation_pro_args = [(Context_initialisation_pro,pro_initial_populations[i]) for i in range(structured_data.hyperparameters.n_core)]
    local_pro_initial_solutions = ppGA.ere_evolutive_pro_PARALLEL(initialisation_pro_args)            
    pro_initial_solutions = [item for sublist in local_pro_initial_solutions for item in sublist]
    
    ##============================================================================
    # 1. SOLVING THE PROBLEM WITH THE INDEPTH PROCESS
    ##============================================================================
    #First : Solving at a low time resolution (daily granularity), using ERMESS RESEARCH
        
    pre_pop=[]
    MIN_INIT_CONSTRAINT_LEVEL = 0.6
    
    Context_initialisation_InDepth = copy.deepcopy(Contexte)
    Context_initialisation_InDepth.hyperparameters.operators_parameters = Cons.adaptation_hyperparameters_initialisation(Context_initialisation_InDepth.hyperparameters.operators_parameters)
    Context_initialisation_InDepth.hyperparameters.n_pop = population_sizes[1]

    Context_initialisation_InDepth_LowRes = Context_time_resolution_downscaling(Context_initialisation_InDepth)
    possible_constraint_levels=Cons.find_constraint_levels(Context_initialisation_InDepth_LowRes)
    constraint_levels_LowRes = np.random.uniform(min(MIN_INIT_CONSTRAINT_LEVEL,possible_constraint_levels),possible_constraint_levels,structured_data.hyperparameters.n_core)
    
    args_pop_init_LowRes = [[Context_initialisation_InDepth_LowRes,  population_sizes[1],constraint_levels_LowRes[i]] for i in range(structured_data.hyperparameters.n_core)]
    Initial_population_LowRes = ppGA.initial_population_research_PARALLEL(args_pop_init_LowRes)   
    List_Contexts_LowRes = [Context_initialisation_InDepth_LowRes for i in range(structured_data.hyperparameters.n_core)]
    args_init_LowRes = [(List_Contexts_LowRes[i],Initial_population_LowRes[i]) for i in range(structured_data.hyperparameters.n_core)]
        
    print('Beginning the preprocessing evolutive algorithm')
    local_populations_LowRes = ppGA.ere_evolutive_research_PARALLEL(args_init_LowRes)
    
    #Gathering and randomly re-spreading the population
    final_population_LowRes = [item for sublist in local_populations_LowRes for item in sublist]
    spread_final_populations_LowRes = sorted(final_population_LowRes, key=lambda x: np.random.rand())
    grouped_final_populations_LowRes = [spread_final_populations_LowRes[(i*population_sizes[0]):((i+1)*population_sizes[0])] for i in range(structured_data.hyperparameters.n_core)]
            
    print('End of low-resolution preprocessing')
    print('Beginning high-resolution preprocessing')
    
    #First : Solving at a high time resolution (original granularity) on random selected days, using ERMESS RESEARCH
    N_SELECTED_DAYS_PER_CORE = 5
    POWER_CONSTRAINT_LEVEL = 2
    FACTOR_CONSTRAINT_LEVEL = 1/2

    population_HighRes = []
    
    #Random selection of days
    days=np.random.choice(int(Context_initialisation_InDepth.time.n_days),N_SELECTED_DAYS_PER_CORE*structured_data.hyperparameters.n_core,replace=False)
        
    Contexts_initialisation_InDepth_HighRes = [copy.deepcopy(Context_initialisation_InDepth) for i in range(N_SELECTED_DAYS_PER_CORE*structured_data.hyperparameters.n_core)]
    args_pop_init_HighRes = []
    constraint_levels_HighRes = np.float64(np.random.uniform(FACTOR_CONSTRAINT_LEVEL*Context_initialisation_InDepth.optimization.constraint_level**POWER_CONSTRAINT_LEVEL,1,N_SELECTED_DAYS_PER_CORE*structured_data.hyperparameters.n_core))
    k=0
    for day in days :
        Contexts_initialisation_InDepth_HighRes[k] = Context_time_resolution_zooming(Context_initialisation_InDepth,day)     
        args_pop_init_HighRes.append([Contexts_initialisation_InDepth_HighRes[k],  population_sizes[1], constraint_levels_HighRes[k]])
        k+=1
    
    for j in range(N_SELECTED_DAYS_PER_CORE):
        Initial_population_HighRes = ppGA.initial_population_research_PARALLEL(args_pop_init_HighRes[(j*structured_data.hyperparameters.n_core):((j+1)*structured_data.hyperparameters.n_core)])   
        args_init_HighRes = [(Contexts_initialisation_InDepth_HighRes[(j*structured_data.hyperparameters.n_core)+i],Initial_population_HighRes[i]) for i in range(structured_data.hyperparameters.n_core)]
        local_populations_HighRes=ppGA.ere_evolutive_research_PARALLEL(args_init_HighRes)
        
        population_HighRes.append(local_populations_HighRes)
        
    print('End of high-resolution preprocessing')
                    
    Init_solutions = []
    for i in range(structured_data.hyperparameters.n_core) :
        Init_solutions.append(Efr.combining_solutions(grouped_final_populations_LowRes[i],[population_HighRes[j][i] for j in range(4)],[days[(j*n_core)+i] for j in range(4)],Context_initialisation_InDepth_LowRes.time.time_resolution,Contexts_initialisation_InDepth_HighRes[0].time.time_resolution,Context_initialisation_InDepth))
        
    Init_solutions2 = [item for sublist in Init_solutions for item in sublist]
            
    Init_solutions_final = Init_solutions2 + pro_initial_solutions
    shuffled_population = sorted(Init_solutions_final, key=lambda x: np.random.rand())
    args_fitness = [[shuffled_population[(i*n_pop):((i+1)*n_pop)],Contexte] for i in range(n_core)]
    fitnesses =ppGA.parallel_fitness(args_fitness)
    fitnesses = [item for sublist in fitnesses for item in sublist]
                
    for i in range(len(shuffled_population)) :
        shuffled_population[i].fitness=fitnesses[i]
          
    print('Score du meilleur individu : ',min(tuple(shuffled_population[i].fitness for i in range(len(shuffled_population)))))
    target_file = sys.argv[1]
    for j in range(n_nodes):
        with open(target_file+'_for_'+str(j)+'.dat', "wb") as f:
            pickle.dump(shuffled_population[int(j*n_pop*n_core/n_nodes):int((j+1)*n_pop*n_core/n_nodes)], f)