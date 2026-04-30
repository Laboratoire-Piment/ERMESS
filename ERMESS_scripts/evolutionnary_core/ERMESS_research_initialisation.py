# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 12:51:20 2026

@author: JoPHOBEA
"""

from ERMESS_scripts.utils import constraints as Cons
from ERMESS_scripts.evolutionnary_core import ERMESS_parallel_processing as ppGA
from ERMESS_scripts.evolutionnary_core import ERMESS_functions_research as Efr
from ERMESS_scripts.data.indices import *


import copy
import numpy as np
import pickle

        
def write_node_population(node_id,node_population):
    name = f"population_node_{node_id}.pkl"
    with open(name, "wb") as f:
        pickle.dump(node_population, f)

def _Time_downscaling (Transformed_variable,time_resolution,n_days):
    """
    Internal helper: aggregate time series to daily resolution.
    """
    Daily_variable = np.array([sum(Transformed_variable[int(i*24*time_resolution):int((i+1)*24*time_resolution) ])/time_resolution for i in range(n_days)])   
    return(Daily_variable)

def _Time_zooming (Original_variable,time_resolution,day):
    """
    Extract a single day from a high-resolution time series.
    """
    Zoomed_variable = np.array(Original_variable[(day*int(24*time_resolution)):((day+1)*int(24*time_resolution))])   
    return(Zoomed_variable)

def _build_LowRes_Context(Context_initialisation_InDepth):
    """
    Create a low-resolution (daily) version of a context object.
    
    All time-dependent data (loads, production, grid signals) are
    downscaled from high resolution to daily resolution.
    """
    
    Context_initialisation_InDepth_LR = copy.deepcopy(Context_initialisation_InDepth)
    LR_n_bits = Context_initialisation_InDepth.time.n_days
    Context_initialisation_InDepth_LR.time.n_bits = LR_n_bits
    Context_initialisation_InDepth_LR.time.time_resolution = np.float64(1/24)
    Context_initialisation_InDepth_LR.loads.non_movable = _Time_downscaling(Context_initialisation_InDepth.loads.non_movable,Context_initialisation_InDepth.time.time_resolution,LR_n_bits)
    Context_initialisation_InDepth_LR.loads.Y_movable = _Time_downscaling(Context_initialisation_InDepth.loads.Y_movable,Context_initialisation_InDepth.time.time_resolution,LR_n_bits)
    Context_initialisation_InDepth_LR.loads.D_movable = _Time_downscaling(Context_initialisation_InDepth.loads.D_movable,Context_initialisation_InDepth.time.time_resolution,LR_n_bits)
    Context_initialisation_InDepth_LR.production.unit_prods = np.array([_Time_downscaling(unit_prod_j, Context_initialisation_InDepth.time.time_resolution,LR_n_bits) for unit_prod_j in Context_initialisation_InDepth.production.unit_prods])
    Context_initialisation_InDepth_LR.production.specs_num[:,PROD_VOLUME] = np.sum(Context_initialisation_InDepth_LR.production.unit_prods,axis=1)
    Context_initialisation_InDepth_LR.production.current_prod = _Time_downscaling(Context_initialisation_InDepth.production.current_prod,Context_initialisation_InDepth.time.time_resolution,LR_n_bits)
    Context_initialisation_InDepth_LR.production.current_prod = _Time_downscaling(Context_initialisation_InDepth.production.current_prod,Context_initialisation_InDepth.time.time_resolution,LR_n_bits)
    if Context_initialisation_InDepth.grid is not None :
        Context_initialisation_InDepth_LR.grid.prices = np.array([_Time_downscaling(Context_initialisation_InDepth.grid.prices,Context_initialisation_InDepth.time.time_resolution,LR_n_bits) for j in range(len(Context_initialisation_InDepth.grid.prices))] ) 
        Context_initialisation_InDepth_LR.grid.Selling_price = np.array([_Time_downscaling(Context_initialisation_InDepth.grid.Selling_price,Context_initialisation_InDepth.time.time_resolution,LR_n_bits) for j in range(len(Context_initialisation_InDepth.grid.Selling_price))] ) 
    Context_initialisation_InDepth_LR.tracking.tracking_operators = False
    
    return(Context_initialisation_InDepth_LR)

def _build_HighRes_Context(Context_initialisation_InDepth,day):
    """
    Create a high-resolution context focused on a single day.
    
    Extracts time-dependent data (loads, production, grid signals)
    corresponding to a specific day while preserving original resolution.
    """
    
    Context_initialisation_InDepth_HighRes = copy.deepcopy(Context_initialisation_InDepth)
    Context_initialisation_InDepth_HighRes.time.n_bits=np.int64(24*Context_initialisation_InDepth.time.time_resolution)
    Context_initialisation_InDepth_HighRes.time.time_resolution=np.float64(Context_initialisation_InDepth.time.time_resolution)
    Context_initialisation_InDepth_HighRes.time.duration_years = np.float64(1/365)
    Context_initialisation_InDepth_HighRes.loads.non_movable = _Time_zooming(Context_initialisation_InDepth.loads.non_movable,Context_initialisation_InDepth.time.time_resolution,day)
    Context_initialisation_InDepth_HighRes.loads.Y_movable = _Time_zooming(Context_initialisation_InDepth.loads.Y_movable,Context_initialisation_InDepth.time.time_resolution,day)
    Context_initialisation_InDepth_HighRes.loads.D_movable = _Time_zooming(Context_initialisation_InDepth.loads.D_movable,Context_initialisation_InDepth.time.time_resolution,day)
    Context_initialisation_InDepth_HighRes.production.unit_prods = np.array([_Time_zooming(Context_initialisation_InDepth.production.unit_prods[i],Context_initialisation_InDepth.time.time_resolution,day) for i in range(len(Context_initialisation_InDepth.production.unit_prods))])
    Context_initialisation_InDepth_HighRes.production.specs_num[:,PROD_VOLUME] = np.sum(Context_initialisation_InDepth_HighRes.production.unit_prods,axis=1)
    Context_initialisation_InDepth_HighRes.production.current_prod = _Time_zooming(Context_initialisation_InDepth.production.current_prod,Context_initialisation_InDepth.time.time_resolution,day)
    if Context_initialisation_InDepth_HighRes.grid is not None :
        Context_initialisation_InDepth_HighRes.grid.prices = np.array([_Time_zooming(Context_initialisation_InDepth.grid.prices[i],Context_initialisation_InDepth.time.time_resolution,day) for i in range(len(Context_initialisation_InDepth.grid.prices))])
        Context_initialisation_InDepth_HighRes.grid.Selling_price = np.array([_Time_zooming(Context_initialisation_InDepth.grid.Selling_price[i],Context_initialisation_InDepth.time.time_resolution,day) for i in range(len(Context_initialisation_InDepth.grid.Selling_price))])   
    Context_initialisation_InDepth_HighRes.tracking.tracking_operators = False
    
    return(Context_initialisation_InDepth_HighRes)

def init_pro_population(Context, n_core, n_pop_pro):
    """
    Initialize a population using the PRO evolutionary strategy.
    
    This function generates initial populations using the PRO evolutionary solver.
    
    Args:
        Context: object
            Simulation context defining system parameters.
        n_core: int
            Number of parallel processes.
        n_pop_pro: int
            Number of individuals per PRO population.
    
    Returns:
        list
            Flattened list of all evolved PRO solutions.
    
    Notes:
        - Parallel execution is handled internally.
        - Output is a merged population from all cores.
    """
    Context_initialisation_pro = copy.deepcopy(Context)
    Context_initialisation_pro.hyperparameters.n_pop = n_pop_pro
    Context_initialisation_pro.tracking.tracking_operators = False
    
    #pro_initial_populations =  [Efp.initial_population_pro(Context_initialisation_pro) for _ in range(n_core)]
    
    #Spreading the ERMESS PRO initial population 
    #initialisation_pro_args = [(Context_initialisation_pro,pro_initial_populations[i]) for i in range(n_core)]
    List_Contexts = [(Context_initialisation_pro,Context) for _ in range(n_core)]
    local_pro_initial_solutions = ppGA.ERMESS_pro_PARALLEL(List_Contexts)          
    pro_initial_solutions = [item for sublist in local_pro_initial_solutions for item in sublist]
    
    return(pro_initial_solutions)

def init_low_res_population(Context_initialisation_Research, n_core, n_pop_research, MIN_INIT_CONSTRAINT_LEVEL):
    """
    Initialize a population using a low-resolution evolutionary process.
    
    This function generates and evolves populations on a downscaled
    (daily) version of the problem to reduce computational cost.
    
    Args:
        Context_initialisation_Research: object
            Full simulation context.
        n_core: int
            Number of parallel processes.
        n_pop_research: int
            Population size per core.
        MIN_INIT_CONSTRAINT_LEVEL: float
            Minimum constraint level for initialization.
    
    Returns:
        tuple:
            - grouped_populations (list of lists): populations split per core
            - context_low_res: object
                Low-resolution version of the context used.
    
    Notes:
        - Constraint levels are sampled randomly within valid range.
        - Population is shuffled before being split across cores.
    """

    Context_initialisation_Research_LowRes = _build_LowRes_Context(Context_initialisation_Research)
    possible_constraint_levels=Cons.find_constraint_levels(Context_initialisation_Research_LowRes)
    constraint_levels_LowRes = np.random.uniform(min(MIN_INIT_CONSTRAINT_LEVEL,possible_constraint_levels),possible_constraint_levels,n_core)
    
    args_pop_init_LowRes = [[Context_initialisation_Research_LowRes,  n_pop_research,constraint_levels_LowRes[i]] for i in range(n_core)]
    Initial_population_LowRes = ppGA.initial_population_research_PARALLEL(args_pop_init_LowRes)   
    List_Contexts_LowRes = [Context_initialisation_Research_LowRes for i in range(n_core)]
    args_init_LowRes = [(List_Contexts_LowRes[i],Initial_population_LowRes[i]) for i in range(n_core)]
        
    print('Beginning the preprocessing evolutive algorithm')
    local_populations_LowRes = ppGA.ere_evolutive_research_PARALLEL(args_init_LowRes)
    
    #Gathering and randomly re-spreading the population
    final_population_LowRes = [item for sublist in local_populations_LowRes for item in sublist]
    spread_final_populations_LowRes = sorted(final_population_LowRes, key=lambda x: np.random.rand())
    grouped_final_populations_LowRes = [spread_final_populations_LowRes[(i*n_pop_research):((i+1)*n_pop_research)] for i in range(n_core)]
    return(grouped_final_populations_LowRes,Context_initialisation_Research_LowRes)
    
def init_high_res_population(Context_initialisation_Research, n_core, n_pop_research):
    """
    Initialize a population using high-resolution local daily simulations.
    
    This function selects random days and performs evolutionary
    optimization on full-resolution data for each selected day.
    
    Args:
        Context_initialisation_Research: object
            Simulation context.
        n_core: int
            Number of parallel processes.
        n_pop_research: int
            Population size per core.
    
    Returns:
        tuple:
            - population_high_res (list): evolved populations per selected day
            - days (np.ndarray): selected day indices
            - context_example: object
                Example high-resolution context used for initialization.
    
    Notes:
        - Multiple days are sampled without replacement.
        - Constraint levels are randomized per sample.
    """
    N_SELECTED_DAYS_PER_CORE = 5
    POWER_CONSTRAINT_LEVEL = 2
    FACTOR_CONSTRAINT_LEVEL = 1/2

    population_HighRes = []
    
    #Random selection of days
    days=np.random.choice(int(Context_initialisation_Research.time.n_days),N_SELECTED_DAYS_PER_CORE*n_core,replace=False)
    
    List_Context_initialisation_Research_HighRes = []
    args_pop_init_HighRes = []
    constraint_levels_HighRes = np.float64(np.random.uniform(FACTOR_CONSTRAINT_LEVEL*Context_initialisation_Research.optimization.constraint_level**POWER_CONSTRAINT_LEVEL,1,N_SELECTED_DAYS_PER_CORE*n_core))
    k=0
    for day in days :
        List_Context_initialisation_Research_HighRes.append(_build_HighRes_Context(Context_initialisation_Research,day))     
        args_pop_init_HighRes.append([List_Context_initialisation_Research_HighRes[k],  n_pop_research, constraint_levels_HighRes[k]])
        k+=1
    
    for j in range(N_SELECTED_DAYS_PER_CORE):
        Initial_population_HighRes = ppGA.initial_population_research_PARALLEL(args_pop_init_HighRes[(j*n_core):((j+1)*n_core)])   
        args_init_HighRes = [(List_Context_initialisation_Research_HighRes[(j*n_core)+i],Initial_population_HighRes[i]) for i in range(n_core)]
        local_populations_HighRes=ppGA.ere_evolutive_research_PARALLEL(args_init_HighRes)
        
        population_HighRes.append(local_populations_HighRes)
    return(population_HighRes,days,List_Context_initialisation_Research_HighRes[0])
 
    
def combine_populations(pro_initial_solutions, grouped_final_populations_LowRes, population_HighRes, n_core, days, Context_initialisation_Research, Context_initialisation_Research_LowRes, Context_initialisation_Research_HighRes):
    """
    Combine PRO, low-resolution, and high-resolution populations.

    This function merges solutions from different initialization strategies
    and reconstructs full individuals using multi-resolution information.

    Args:
        pro_initial_solutions: list
            Population generated by PRO strategy.
        grouped_final_populations_LowRes: list of list
            Low-resolution evolved populations per core.
        population_HighRes: list
            High-resolution evolved populations.
        n_core: int
            Number of parallel processes.
        days: array-like
            Selected high-resolution days used for reconstruction.
        Context_initialisation_Research: object
            Full context.
        Context_initialisation_Research_LowRes: object
            Low-resolution context.
        Context_initialisation_Research_HighRes: object
            High-resolution context.

    Returns:
        list
            Final shuffled population.

    Notes:
        - Solutions are combined using multi-resolution reconstruction.
        - Final population is randomly shuffled.
    """

    N_DAYS_HIGH_RES = len(population_HighRes)
    Research_initial_solutions_local = []
    for i in range(n_core) :
        Research_initial_solutions_local.append(Efr.combining_solutions(grouped_final_populations_LowRes[i],[population_HighRes[j][i] for j in range(N_DAYS_HIGH_RES)],[days[(j*n_core)+i] for j in range(N_DAYS_HIGH_RES)],Context_initialisation_Research_LowRes.time.time_resolution,Context_initialisation_Research_HighRes.time.time_resolution,Context_initialisation_Research))
        
    Research_initial_solutions = [item for sublist in Research_initial_solutions_local for item in sublist]
            
    Init_solutions_final = Research_initial_solutions + pro_initial_solutions
    shuffled_population = sorted(Init_solutions_final, key=lambda x: np.random.rand())
    return(shuffled_population)

def Initialize_ERMESS_research(Context , structured_data,node_id):
    """
    Initialize and save a population for the ERMESS genetic algorithm.
    
    This function combines:
    - A "PRO" initialization strategy
    - A multi-resolution "In-Depth" initialization (low + high resolution)
    
    Args:
    Context : object
        Full optimization context.
    structured_data : object
        Contains hyperparameters and execution settings.
    
    Returns:
        None
    """

    n_core = structured_data.hyperparameters.n_core
    n_pop = Context.hyperparameters.n_pop
    n_bins = structured_data.hyperparameters.n_nodes
    
    #Split the population in 2  groups : 0 : pro initialisation 1 : In-depth study of seasons and days
    
    n_pop_pro = n_pop//2  if (Context.hyperparameters.n_pop%4==0) else (int((Context.hyperparameters.n_pop/2)-1))
    n_pop_research =  n_pop - n_pop_pro
    Context.hyperparameters_pro.n_pop = n_pop_pro
        
    ##============================================================================
    # 1. SOLVING THE PROBLEM WITH ERMESS PRO
    ##============================================================================

    pro_initial_solutions = init_pro_population(Context, n_core, n_pop_pro)
    
    ##============================================================================
    # 1. SOLVING THE PROBLEM WITH ERMESS RESEARCH ON SIMPLIFIED DATA AND INTERPOLATION
    ##============================================================================
    #First : Solving at a low time resolution (daily granularity), using ERMESS RESEARCH
        
    MIN_INIT_CONSTRAINT_LEVEL = 0.6
    
    Context_initialisation_Research = copy.deepcopy(Context)
    Context_initialisation_Research.hyperparameters.operators_parameters = Cons.adaptation_hyperparameters_initialisation(Context_initialisation_Research.hyperparameters.operators_parameters)
    Context_initialisation_Research.hyperparameters.n_pop = n_pop_research

    #1. Solving at a Low time resolution (daily granularity), using ERMESS RESEARCH
    (grouped_final_populations_LowRes,Context_initialisation_Research_LowRes) = init_low_res_population(Context_initialisation_Research, n_core, n_pop_research, MIN_INIT_CONSTRAINT_LEVEL)
    print('End of low-resolution preprocessing')
    
    #2. Solving at a high time resolution (original granularity) on random selected days, using ERMESS RESEARCH
    (population_HighRes,days,Context_initialisation_Research_HighRes) = init_high_res_population(Context_initialisation_Research, n_core, n_pop_research)
    print('End of high-resolution preprocessing')
        
    #3. Merging the solutions
    shuffled_population = combine_populations(pro_initial_solutions, grouped_final_populations_LowRes, population_HighRes, n_core, days, Context_initialisation_Research, Context_initialisation_Research_LowRes, Context_initialisation_Research_HighRes)            
    spread_shuffled_population = [shuffled_population[(i*n_pop):((i+1)*n_pop)] for i in range(n_core)]
    
    write_node_population(node_id,spread_shuffled_population) 

   
