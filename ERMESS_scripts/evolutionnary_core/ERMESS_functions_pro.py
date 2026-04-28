# -*- coding: utf-8 -*-
"""
Created on Tue Jun 10 11:33:54 2025

@author: JoPHOBEA
"""

import numpy as np
from numba import jit
from numba.experimental import jitclass
from numba import int64, float64, types
from dataclasses import dataclass

from ERMESS_scripts.evolutionnary_core import ERMESS_evolutionnary_operators as Eop
from ERMESS_scripts.data.indices import *

@dataclass
class PIDConfig:
    Kp: float = 2.0
    Ki: float = 0.05
    Kd: float = 1.0
    u_min: float = 0.0
    u_max: float = 2.0
    anti_windup: float = 10.0
    beta: float = 0.3

    def validate(self):
        assert self.Kp >= 0, "Kp must be >= 0"
        assert self.Ki >= 0, "Ki must be >= 0"
        assert self.Kd >= 0, "Kd must be >= 0"
        assert self.u_min < self.u_max, "u_min must be < u_max"
        assert 0 <= self.beta <= 1, "beta must be in [0,1]"

@jitclass([
           ('production_set', int64[:]),
           ('contract', int64),
           ('DG_strategy', types.string),
           ('discharge_order', int64[:]),
           ('energy_use_coefficient', float64),
           ('overlaps', float64[:,:]),
           ('D_DSM_minimum_levels', float64[:]),
           ('Y_DSM_minimum_levels', float64[:]),
           ('DG_min_runtime', int64),
           ('DG_min_production', float64),
           ('storages', float64[:,:]),
           ('fitness', float64),
           ])

class Individual_pro(object):
    """
    Class representing an individual result in the optimization process (Production side).
    Targeted for Numba JIT compilation.
    
    Attributes:
        production_set (numpy.ndarray): Set of production units.
        contract (int): Contract ID.
        PMS_strategy (str): Power Management System strategy.
        PMS_discharge_order (numpy.ndarray): Order of discharge for PMS.
        energy_use_repartition_DSM (float): Energy use repartition for DSM.
        PMS_taking_over (numpy.ndarray): PMS taking over values.
        PMS_D_DSM_min_levels (numpy.ndarray): Minimum levels for Daily DSM in PMS.
        PMS_Y_DSM_min_levels (numpy.ndarray): Minimum levels for Yearly DSM in PMS.
        PMS_DG_min_runtime (int): Minimum runtime for Diesel Generator in PMS.
        PMS_DG_min_production (float): Minimum production for Diesel Generator in PMS.
        storages (numpy.ndarray): Storage units.
        fitness (float): Fitness score of the individual.
    """
    def __init__(self,production_set,contract,DG_strategy,discharge_order,energy_use_coefficient,overlaps,D_DSM_minimum_levels,Y_DSM_minimum_levels,DG_min_runtime,DG_min_production,storages,fitness):
        self.production_set = production_set
        self.contract = contract
        self.DG_strategy = DG_strategy
        self.discharge_order = discharge_order
        self.energy_use_coefficient = energy_use_coefficient
        self.overlaps = overlaps
        self.D_DSM_minimum_levels = D_DSM_minimum_levels
        self.Y_DSM_minimum_levels = Y_DSM_minimum_levels
        self.DG_min_runtime = DG_min_runtime        
        self.DG_min_production = DG_min_production       
        self.storages = storages
        self.fitness = fitness
    
    def copy(self):
        """
        Creates a copy of an object of class Individual_pro (JIT type)
        
        Returns:
            A JIT Individual PRO object
        """
        return Individual_pro(self.production_set.copy(),self.contract,self.DG_strategy,self.discharge_order.copy(),self.energy_use_coefficient,self.overlaps.copy(),self.D_DSM_minimum_levels.copy(),self.Y_DSM_minimum_levels.copy(),self.DG_min_runtime,self.DG_min_production,self.storages.copy(),self.fitness)

        
class Non_JIT_Individual_pro():
    """
    Class representing an individual result in the optimization process (Production side).
    Non targeted for Numba JIT compilation.
    
    See 'Individual_pro' for full attribute specification.
    """
    def __init__(self,production_set,contract,DG_strategy,discharge_order,energy_use_coefficient,overlaps,D_DSM_minimum_levels,Y_DSM_minimum_levels,DG_min_runtime,DG_min_production,storages,fitness):
        self.production_set = production_set
        self.contract = contract
        self.DG_strategy = DG_strategy
        self.discharge_order = discharge_order
        self.energy_use_coefficient = energy_use_coefficient
        self.overlaps = overlaps
        self.D_DSM_minimum_levels = D_DSM_minimum_levels
        self.Y_DSM_minimum_levels = Y_DSM_minimum_levels
        self.DG_min_runtime = DG_min_runtime
        self.DG_min_production = DG_min_production 
        self.storages = storages
        self.fitness = fitness
        
    def copy(self):
        """        
        Creates a copy of an object of class Individual_pro (non-JIT type)
        :returns: A non-JIT Individual PRO object
        :rtype: object of class Individual PRO
        """
        return Non_JIT_Individual_pro(self.production_set.copy(),self.contract,self.DG_strategy,self.discharge_order.copy(),self.energy_use_coefficient,self.overlaps.copy(),self.D_DSM_minimum_levels.copy(),self.Y_DSM_minimum_levels.copy(),self.DG_min_runtime,self.DG_min_production,self.storages.copy(),self.fitness)

class RandomFactors:
    def __init__(self, arr):
        self.contract = arr[0]
        self.production_main = arr[1]
        self.production_swap = arr[2]
        self.production_transfer = arr[3]
        self.strategy = arr[4]
        self.storage_order = arr[5]
        self.dsm_energy = arr[6]
        self.overlap = arr[7]
        self.dsm_levels = arr[8]
        self.ydsm_levels = arr[9]
        self.dg_runtime = arr[10]
        self.dg_production = arr[11]
        self.storage_capacity = arr[12]
        self.storage_inpower = arr[13]
        self.storage_outpower = arr[14]
        self.init_soc = arr[15]

       
def jitting_pop_pro(pop):
    """
    Converts a list of Individual_pro objects back to a list of Non_JIT_Individual_pro objects.
    
    Args:
        jitted_pop (list of Individual_pro): The population of JIT-compatible individuals.
    
    Returns:
        The population of non-JIT individuals.
    """
    jitted_pop=[]
    for ind in pop:
        jitted_pop.append(Individual_pro(np.int64(ind.production_set),np.int64(ind.contract),ind.DG_strategy,np.array(ind.discharge_order,dtype=np.int64),np.float64(ind.energy_use_coefficient),np.array(ind.overlaps,dtype=np.float64),np.array(ind.D_DSM_minimum_levels,dtype=np.float64),np.array(ind.Y_DSM_minimum_levels,dtype=np.float64),np.int64(ind.DG_min_runtime),np.float64(ind.DG_min_production),np.array(ind.storages,dtype=np.float64),np.float64(ind.fitness)))
    return(jitted_pop)

def unjitting_pop_pro(jitted_pop):
    """
    Converts a list of Individual_pro objects back to a list of Non_JIT_Individual_pro objects.

    :param  jitted_pop : The population of JIT-compatible individuals (production side).
    :type list of Individual_pro
        

    :returns: The population of non-JIT individuals.
    :rtype: list of Non_JIT_Individual_pro
        
    """
    pop=[]
    for ind in jitted_pop:
        pop.append(Non_JIT_Individual_pro(np.int64(ind.production_set),np.array(ind.contract,dtype=np.int64),(ind.DG_strategy),np.array(ind.discharge_order,dtype=np.int64),np.float64(ind.energy_use_coefficient),np.array(ind.overlaps,dtype=np.float64),np.array(ind.D_DSM_minimum_levels,dtype=np.float64),np.array(ind.Y_DSM_minimum_levels,dtype=np.float64),np.float64(ind.DG_min_runtime),np.float64(ind.DG_min_production),np.array(ind.storages,dtype=np.float64),np.float64(ind.fitness)))
    return(pop)




def find_cost_function_pro(Contexte,pro_parameters, global_parameters, grid_parameters, RENSystems_parameters, Genset_parameters):
    """
    Determines the right loss function to apply to the optimization problem (PRO).
    
    Args:
        Contexte: Description of the constraints of the problem.
    
    Returns:
        int: ID of the appropriate cost function.
    """    
    try:
        criterion = CriterionEnum(Contexte.optimization.criterion_num)
        base_function = criterion.get_function_pro()
    except KeyError:
        raise ValueError("No proper optimisation criterion found!")
                
    def cost_function(ind):
        return base_function(ind,RENSystems_parameters,global_parameters,Genset_parameters,grid_parameters,pro_parameters)

    return(cost_function)
                            
def initial_population_pro(Context):
#,
                           #n_bits, n_pop,n_store,time_resolution,Bounds_prod,groups,Non_movable_load,prod_C,prods_U,storage_characteristics,constraint_num,Constraint_level,n_contracts,Dispatching,specs_num,type_optim): 
    """
    Generate the initial population for the pro mode of ERMESS.
    
    This function creates a population of individuals representing candidate
    microgrid configurations. Each individual includes:
        
        - Production unit sizing (group-constrained),
        - Energy contract selection,
        - Power Management System (PMS) strategy parameters,
        - Demand Side Management (DSM) parameters,
        - Diesel generator constraints (if applicable),
        - Storage sizing (capacity, charge/discharge powers, initial SOC).
        
    The initialization combines structured seeding (first individuals scaled
    deterministically) and stochastic sampling to ensure both feasibility
    and diversity in the search space.
    
    Args:
        n_bits (int): Length of the timeseries.
        n_pop (int): Population size.
        n_store (int): Number of storage technologies.
        time_resolution (float): Time-resolution.
        Bounds_prod (array-like): Upper bounds for each production unit (integer capacities).
        groups (list of array-like): Mutually exclusive production groups. Only one unit per group
            can be active.
        Non_movable_load (ndarray): Time series of non-flexible electrical load.
        prod_C (ndarray): Current on-site production timeseries (kW).
        prods_U (ndarray): Unit production timeseries (kW per unit).
        storage_characteristics (any): Technical characteristics of storage technologies (not directly
            used here but required for compatibility).
        constraint_num (any): Constraint identifiers (not directly applied here).
        Constraint_level (any): Constraint levels (not directly applied here).
        n_contracts (int): Number of available energy contracts.
        Dispatching (tuple): User-Predefined dispatching parameters. If specific keywords are
            present, corresponding parameters are fixed instead of randomized.
        specs_num (any): Specification parameters (not directly used here).
        type_optim ({'pro', 'research'}): If 'pro', returns JIT-compatible Individual_pro objects.
            Otherwise, returns Non_JIT_Individual_pro objects.
    
    Returns:
        List of initialized individuals (instances of Individual_pro or Non_JIT_Individual_pro).
    
    Note:
        Storage sizing is randomly distributed across units using normalized
        random proportions.
    
    Note:
        The first min(n_pop, 20) individuals are deterministically scaled
        to improve initial coverage of the search space.
    """  
    n_pop = Context.hyperparameters_pro.n_pop
    Initial_prod_index = np.random.rand(n_pop,Context.production.n_units)
    Initial_prod = np.array([[np.random.randint(0,Bound,1)[0] for Bound in Context.production.capacities] for j in range(n_pop)])
    Initial_prod[0:min(n_pop,20)]=[((i+11)*Context.production.capacities/30).astype(int) for i in range(min(n_pop,20))]
    if Context.grid == None : 
        Initial_contracts = np.repeat(-1,n_pop)
    else:
        Initial_contracts = np.random.randint(0, Context.grid.n_contracts, n_pop)
        
    if 'Genset control' not in Context.config.defined_items :
        Random_DG_strategies = np.random.rand(n_pop)
        Random_DG_min_runtime = np.random.randint(1,10,n_pop)
        Random_DG_min_production = np.random.rand(n_pop)
    
    if 'Storages management' not in Context.config.defined_items :               
        Random_overlaps = np.random.rand(n_pop,2,9)
        Random_discharge_order = np.array([np.random.permutation(Context.storage.n_store) for i in range(n_pop)],dtype=np.int64)
        
    if 'DSM' not in Context.config.defined_items :
        Random_energy_use_repartition = np.random.rand(n_pop)
        Random_D_DSM_min_levels = np.random.rand(n_pop,23)
        Random_Y_DSM_min_levels = np.random.rand(n_pop,11)  
        
    
    Random_storages_init_SOCs = np.random.rand(Context.storage.n_store,n_pop)
    Initial_population = list()
    
    for j in range(n_pop):
        ones_prod=[Context.production.groups[i][np.argmax(Initial_prod_index[j][Context.production.groups[i]])] for i in range(len(Context.production.groups))]
        Initial_prod[j][np.array([i not in ones_prod for i in range(Context.production.n_units)])]=0
        prod = np.dot(Initial_prod[j],Context.production.unit_prods)/1000+Context.production.current_prod/1000
        storage_total_capacity=np.random.uniform(0,max(abs(np.cumsum(prod-Context.loads.non_movable))))
        storage_total_discharge_power=np.random.uniform(0,max(Context.loads.non_movable))
        storage_total_charge_power=np.random.uniform(0,max(prod))
        distributions = np.random.rand(Context.storage.n_store,3)
        distributions = distributions/np.sum(distributions,axis=0)
        storages_discharge_powers = storage_total_discharge_power*distributions[:,0]
        storages_charge_powers = storage_total_charge_power*distributions[:,1]
        storages_volumes = storage_total_capacity*distributions[:,2]
        storages_SOCs_Init = Random_storages_init_SOCs[:,j]
        storages_param = np.concatenate((storages_volumes,storages_charge_powers,storages_discharge_powers,storages_SOCs_Init)).reshape(4,Context.storage.n_store)
        
        if ('Discharge order' in Context.config.defined_items):
            discharge_order = np.array(Context.config.discharge_order,dtype=np.int64)
            overlaps = Context.config.overlaps/100
            energy_use_coefficient=Context.config.energy_use_coefficient
        else : 
            discharge_order = Random_discharge_order[j]
            overlaps = np.sort(Random_overlaps[j,:])
            energy_use_coefficient=Random_energy_use_repartition[j]
            
        if 'DSM' in Context.config.defined_items :
            D_DSM_minimum_levels = Context.config.D_DSM_minimum_levels/100
            Y_DSM_minimum_levels = Context.config.Y_DSM_minimum_levels/100
        else : 
            D_DSM_minimum_levels = np.sort(Random_D_DSM_min_levels[j,:])
            Y_DSM_minimum_levels = np.sort(Random_Y_DSM_min_levels[j,:])
            
        if ('Genset control' in Context.config.defined_items):
            DG_strategy = Context.config.DG_strategy
            DG_min_runtime = Context.config.DG_min_runtime
            DG_min_production = Context.config.DG_min_production
        else : 
            DG_strategy = 'LF' if Random_DG_strategies[j]<0.5 else 'CC'
            DG_min_runtime = Random_DG_min_runtime[j]
            DG_min_production = max(Context.loads.non_movable)*0.2*Random_DG_min_production[j]

       # if (type_optim=='pro'): 
        Init_pop_j=Individual_pro(production_set=np.array(Initial_prod[j],dtype=np.int64),contract=Initial_contracts[j],DG_strategy=DG_strategy,discharge_order=discharge_order,energy_use_coefficient=energy_use_coefficient,overlaps=overlaps,D_DSM_minimum_levels=D_DSM_minimum_levels,Y_DSM_minimum_levels=Y_DSM_minimum_levels,DG_min_runtime=DG_min_runtime,DG_min_production=DG_min_production,storages=storages_param,fitness=np.nan)
      #  else :
      #      Init_pop_j=Non_JIT_Individual_pro(production_set=Initial_prod[j],contract=Initial_contracts[j],PMS_strategy=PMS_strategy,PMS_discharge_order=PMS_Discharge_order,energy_use_repartition_DSM=energy_use_repartition_DSM,PMS_taking_over=PMS_taking_over,PMS_D_DSM_min_levels=PMS_D_DSM_min_levels,PMS_Y_DSM_min_levels=PMS_Y_DSM_min_levels,PMS_DG_min_runtime=PMS_DG_min_runtime,PMS_DG_min_production=PMS_DG_min_production,storages=storages_param,fitness=np.nan)
        Initial_population.append(Init_pop_j)
       
    return(Initial_population)   

@jit(nopython=True)
def compute_diversity_pro(fitnesses):
    """
    JIT-compatible computation of population diversity based on fitness dispersion. 
    
    Diversity is defined as the standard deviation of fitness values
    across the population.
    
    Args:
        fitnesses (ndarray): Array of fitness values for the current population.
    
    Returns:
        Standard deviation of fitness values.
    """

    diversity = np.std(fitnesses)

    return(diversity)

@jit(nopython=True)
def PID_correction (stagnation,diversity_threshold,diversity,integrale_PID,prev_error,Kp,Ki,Kd,u_min,u_max,anti_windup):
    """
    Adaptive mutation control. This function adjusts the mutation factor according to population
    diversity and stagnation level using a proportional–integral–derivative
    (PID) controller.
    
    Args:
        stagnation (int): Number of consecutive generations without improvement.
        diversity_threshold (float): Target diversity level.
        diversity (float): Current diversity measure.
        integrale_PID (float): Accumulated integral term from previous generation.
        prev_error (float): Forecast of the expected error (from previous generation).
        Kp (float): Proportional gain.
        Ki (float): Integral gain.
        Kd (float): Derivative gain.
        u_min (float): Minimum allowable control signal.
        u_max (float): Maximum allowable control signal.
        anti_windup (float): Saturation limit for integral term to prevent windup.
    
    Returns:
        Tuple containing :
            -Updated integral term (float).
            -Multiplicative mutation adjustment factor (1 + boost) (float).
            -Updated error value (float).
    
    Note:
        A stagnation factor scales the correction intensity.
        Anti-windup prevents excessive integral accumulation.
        The mutation factor dynamically balances exploration and exploitation.
    """
    
    stagnation_factor = min(1.0,stagnation/10)
    error = (diversity_threshold - diversity)/diversity_threshold
    integrale_PID=integrale_PID+error
    #On ajoute un windup de 10 pour empêcher le terme intégral d'exploser
    if integrale_PID > anti_windup:
        integrale_PID = anti_windup
    elif integrale_PID < -anti_windup:
        integrale_PID = -anti_windup
    
    derivative = error - prev_error
    boost = (Kp*error + Ki*integrale_PID + Kd*derivative)*stagnation_factor
    if boost > u_max:
        boost = u_max
    elif boost < u_min:
        boost = u_min
    mutation_factor = 1+boost
    prev_error=error
    return(integrale_PID,mutation_factor,prev_error)

def NON_JIT_mutation_contraintes_pro(c, random_factors, choices,global_parameters, RENSystems_parameters ,grid_parameters ,extra_parameters ):
        """
        Apply mutation operators to an individual (non-JIT).
        
        This function probabilistically applies a set of mutation operators
        affecting individual's genotype'
        
        Mutation probabilities are defined in the operator hyperparameter matrix.
        
        Args:
            c (Individual_pro): Individual to mutate.
            random_factors (ndarray): Random values used to decide operator activation.
            choices (list): Additional stochastic parameters for storage operators.
            n_bits (int): Length of timeseries.
            Bounds_prod (ndarray): Production bounds.
            groups (list): Production exclusivity groups.
            Non_movable_load (ndarray): Non-flexible load timeseries.
            constraint_num (any): Constraint identifiers.
            constraint_level (any): Constraint severity levels.
            prods_U (ndarray): Unit production capacities.
            prod_C (ndarray): Fixed production component.
            n_store (int): Number of storage units.
            n_contracts (int): Number of energy contracts.
            time_resolution (float): Time resolution.
            storage_characteristics (any): Storage technology parameters.
            Volums_prod (any): Production volumes (context-specific).
            D_DSM_indexes (any): DSM indexing structure.
            hyperparameters_operators_num_pro (ndarray): Matrix of operator activation probabilities and intensities.
            Defined_items (list): Parameters fixed by the user and therefore excluded from mutation.
        
        Returns:
            Tuple containing :
                - Mutated individual (Individual pro).
                - indicator of the application of operators (Binary vector).
        
        Note:
            Each operator is independently triggered based on predefined probabilities.
        """      
        usage_ope = np.repeat(0, 17)
        
        #MUTATION DU CONTRAT
        if ((random_factors[RF_CONTRACT]<extra_parameters.hyperparameters_operators[PRO_OPER_PROBABILITY,PRO_CONTRACT]) and global_parameters.Connexion==GRID_ON) :  
              c=Eop.switch_contract_operator(c,extra_parameters.n_contracts)
              usage_ope[switch_contract]=1
                   
        #Mutation de la production           
        if (random_factors[RF_PRODUCTION_MAIN]<extra_parameters.hyperparameters_operators[PRO_OPER_PROBABILITY,PRO_PRODUCTION]) :            
             c=Eop.Mutate_production_capacity_operator_pro(c,RENSystems_parameters.capacities,extra_parameters.groups_production,extra_parameters.groups_size,extra_parameters.hyperparameters_operators)         
             usage_ope[Mutate_production_capacity_operator]=1

        if (random_factors[RF_PRODUCTION_SWAP]<(extra_parameters.hyperparameters_operators[PRO_OPER_PROBABILITY,PRO_PRODUCTION]/10) and max(extra_parameters.groups_size)>1) :            
             c=Eop.Switch_intragroup_productor_operator(c,RENSystems_parameters.capacities,extra_parameters.groups_production,extra_parameters.groups_size)         
             usage_ope[Switch_intragroup_productor]=1             
             
        if (random_factors[RF_PRODUCTION_TRANSFER]<extra_parameters.hyperparameters_operators[PRO_OPER_PROBABILITY,PRO_PRODUCTION]/2) :
                  c=Eop.Transfer_production_capacity_operator(c,RENSystems_parameters.capacities)
                  usage_ope[Transfer_production_capacity_operator]=1
                  
        if ((random_factors[RF_STRATEGY]<extra_parameters.hyperparameters_operators[PRO_OPER_PROBABILITY,PRO_STRATEGY]) and (DEFINED_GENSET not in extra_parameters.defined_items) ):         
                c=Eop.Switch_dispatching_strategy_operator(c)
                usage_ope[Switch_dispatching_strategy]=1
                
        if ((random_factors[RF_STORAGE_ORDER]<extra_parameters.hyperparameters_operators[PRO_OPER_PROBABILITY,PRO_DISCHARGE_ORDER]) and (DEFINED_STORAGE not in extra_parameters.defined_items)) :         
                c=Eop.Switch_storages_order_operator(c,random_factors[RF_STORAGE_EFFECT_START:(RF_STORAGE_EFFECT_START+RENSystems_parameters.n_store)])
                usage_ope[Switch_storages_order]=1
            
        if ((random_factors[RF_ENERGY_COEFF]<extra_parameters.hyperparameters_operators[PRO_OPER_PROBABILITY,PRO_ENERGY_COEFF]) and (DEFINED_STORAGE not in extra_parameters.defined_items)) :         
                c=Eop.Mutate_DSM_storage_distribution_operator(c,extra_parameters.hyperparameters_operators)
                usage_ope[Mutate_DSM_storage_distribution]=1
                
        if ((random_factors[RF_OVERLAP]<extra_parameters.hyperparameters_operators[PRO_OPER_PROBABILITY,PRO_OVERLAP]) and (DEFINED_STORAGE not in extra_parameters.defined_items)) :         
                c=Eop.Mutate_EMS_Overlap_operator(c,random_factors[RF_OVERLAP_EFFECT_START : RF_OVERLAP_EFFECT_START+2],extra_parameters.hyperparameters_operators)
                usage_ope[Mutate_EMS_Overlap]=1

        if ((random_factors[RF_DDSM]<extra_parameters.hyperparameters_operators[PRO_OPER_PROBABILITY,PRO_DSM_LEVELS]) and (DEFINED_DSM not in extra_parameters.defined_items)) :         
                c=Eop.Mutate_DDSM_levels_operator(c,extra_parameters.hyperparameters_operators)
                usage_ope[Mutate_DDSM_levels]=1
                
        if ((random_factors[RF_YDSM]<extra_parameters.hyperparameters_operators[PRO_OPER_PROBABILITY,PRO_DSM_LEVELS]) and (DEFINED_DSM not in extra_parameters.defined_items)) :         
                c=Eop.Mutate_YDSM_levels_operator(c,extra_parameters.hyperparameters_operators)
                usage_ope[Mutate_YDSM_levels]=1
        
        if ((random_factors[RF_DG_RUNTIME]<extra_parameters.hyperparameters_operators[PRO_OPER_PROBABILITY,PRO_DG_CONTROL]) and (DEFINED_GENSET not in extra_parameters.defined_items)) :         
                c=Eop.Mutate_DG_min_runtime_operator(c)
                usage_ope[Mutate_DG_min_runtime]=1
                
        if ((random_factors[RF_DG_PRODUCTION]<extra_parameters.hyperparameters_operators[PRO_OPER_PROBABILITY,PRO_DG_CONTROL]) and (DEFINED_GENSET not in extra_parameters.defined_items)) :         
                c=Eop.Mutate_DG_min_production_operator(c,extra_parameters.hyperparameters_operators)
                usage_ope[Mutate_DG_min_production]=1
                
        if (random_factors[RF_STORAGE_CAPACITY]<extra_parameters.hyperparameters_operators[PRO_OPER_PROBABILITY,PRO_STORAGE_CAPACITIES]) :         
                c=Eop.Mutate_storages_capacity_operator(c,extra_parameters.hyperparameters_operators,choices[0])
                usage_ope[Mutate_storages_capacity]=1
                
        if (random_factors[RF_STORAGE_INPOWER]<extra_parameters.hyperparameters_operators[PRO_OPER_PROBABILITY,PRO_STORAGE_POWERS]) :         
                c=Eop.Mutate_storages_inpower_operator(c,extra_parameters.hyperparameters_operators,choices[1])
                usage_ope[Mutate_storages_inpower]=1
                
        if (random_factors[RF_STORAGE_OUTPOWER]<extra_parameters.hyperparameters_operators[PRO_OPER_PROBABILITY,PRO_STORAGE_POWERS]) :         
                c=Eop.Mutate_storages_outpower_operator(c,extra_parameters.hyperparameters_operators,choices[2])
                usage_ope[Mutate_storages_outpower]=1
                
        if (random_factors[RF_INIT_SOC]<extra_parameters.hyperparameters_operators[PRO_OPER_PROBABILITY,PRO_INITIAL_SOC]) :         
                c=Eop.Mutate_initSOC_operator(c,random_factors[RF_INIT_SOC_EFFECT],choices[3])
                usage_ope[Mutate_initSOC_operator]=1
                
        return(c,usage_ope)
    
@jit(nopython=True)
def crossover_reduit_pro(p1, p2, r_cross,RENSystems_parameters , extra_parameters):    
    """
   Performs crossover between two pro individuals using a hybrid evolutionary operator.

   This operator combines:
       - Group-based crossover for production units (respecting exclusivity),
       - Binary swap for discrete strategy variables,
       - Convex combination for continuous parameters,
       - Partial crossover for storage-related configurations.

   Args:
       p1 (Individual_pro): First parent individual.
       p2 (Individual_pro): Second parent individual.
       r_cross (float): Probability of applying crossover.
       RENSystems_parameters (object): System-level parameters (must include `n_store`).
       extra_parameters (object): Additional parameters including:
           - groups_size (ndarray): Sizes of production groups.
           - groups_production (ndarray): Indices of production units per group.

   Returns:
       Tuple[Individual_pro, Individual_pro, int64]:
           - First offspring.
           - Second offspring.
           - Indicator (1 if crossover occurred, 0 otherwise).

   Notes:
       - Production crossover preserves group exclusivity constraints.
       - Continuous variables are recombined using convex combinations.
       - If crossover is not applied, offspring are clones of parents.
       - This function is optimized for Numba `nopython` mode.
   """
 
    # children are copies of parents by default

    c1 = Individual_pro(p1.production_set.copy(),p1.contract,p1.DG_strategy,p1.discharge_order.copy(),p1.energy_use_coefficient,p1.overlaps.copy(),p1.D_DSM_minimum_levels.copy(),p1.Y_DSM_minimum_levels.copy(),p1.DG_min_runtime,p1.DG_min_production,p1.storages.copy(),p1.fitness)
    c2 = Individual_pro(p2.production_set.copy(),p2.contract,p2.DG_strategy,p2.discharge_order.copy(),p2.energy_use_coefficient,p2.overlaps.copy(),p2.D_DSM_minimum_levels.copy(),p2.Y_DSM_minimum_levels.copy(),p2.DG_min_runtime,p2.DG_min_production,p2.storages.copy(),p2.fitness)


    cross_rand = np.random.rand()
 # check for recombination
    if cross_rand < r_cross:
    # select random weights

        n_weights_base = 15
        weights = np.random.random(n_weights_base+len(extra_parameters.groups_size))
 # perform crossover        
        mask_prod = weights[n_weights_base:len(weights)]<0.5
        cross_indexes = np.zeros(len(c1.production_set), dtype=np.bool_)

        for k in range(len(mask_prod)):
            if mask_prod[k]:
                int_idx=extra_parameters.groups_production[k, : extra_parameters.groups_size[k]]
                cross_indexes[int_idx] =int_idx

        for i in range(len(RENSystems_parameters.capacities)):
            if cross_indexes[i]:
                c1.production_set[i] = p2.production_set[i]  #Loops often more efficient than np.where (Numba)
                c2.production_set[i] = p1.production_set[i]

        c1.contract=p1.contract if weights[1]<0.5 else p2.contract
        c2.contract=p1.contract if weights[2]<0.5 else p2.contract
        
        c1.DG_strategy = p1.DG_strategy if weights[3]<0.5 else p2.DG_strategy
        c2.DG_strategy = p2.DG_strategy if weights[4]<0.5 else p1.DG_strategy
        
#        if (RENSystems_parameters.n_store>1) :   # Determining the discharge order
#            break_point = np.random.randint(RENSystems_parameters.n_store-1)+1
            
#            used = np.zeros(RENSystems_parameters.n_store, dtype=np.bool_)

#            for i in range(break_point):
#                 v = p1.discharge_order[i]
#                 c1.discharge_order[i] = v
#                 used[v] = True

#            j = 0
#            for i in range(RENSystems_parameters.n_store):
#                if c1.discharge_order[i] != -1:
#                    continue
#                while used[p2[j]]:
#                    j += 1
#                v = p2[j]
#                c1.discharge_order[i] = v
#                used[v] = True
#                j += 1

  #          c1.discharge_order = np.unique(np.concatenate((p1.discharge_order[0:break_point],p2.discharge_order)))
  #          c2.discharge_order = np.unique(np.concatenate((p2.discharge_order[0:break_point],p1.discharge_order)))
            
        c1.energy_use_coefficient = weights[5]*p1.energy_use_coefficient+(1-weights[5])*p2.energy_use_coefficient
        c2.energy_use_coefficient = weights[5]*p2.energy_use_coefficient+(1-weights[5])*p1.energy_use_coefficient
        
        c1.overlaps = weights[6]*p1.overlaps+(1-weights[6])*p2.overlaps
        c2.overlaps = weights[6]*p2.overlaps+(1-weights[6])*p1.overlaps
            
        c1.D_DSM_minimum_levels = weights[7]*p1.D_DSM_minimum_levels+(1-weights[7])*p2.D_DSM_minimum_levels
        c2.D_DSM_minimum_levels = weights[7]*p2.D_DSM_minimum_levels+(1-weights[7])*p1.D_DSM_minimum_levels
        
        c1.Y_DSM_minimum_levels = weights[8]*p1.Y_DSM_minimum_levels+(1-weights[8])*p2.Y_DSM_minimum_levels
        c2.Y_DSM_minimum_levels = weights[8]*p2.Y_DSM_minimum_levels+(1-weights[8])*p1.Y_DSM_minimum_levels
        
        c1.DG_min_runtime = np.int64(weights[9]*p1.DG_min_runtime + (1-weights[9])*p2.DG_min_runtime + 0.5) #Substitution for round (Numba constraint)
        c2.DG_min_runtime = np.int64(weights[9]*p1.DG_min_runtime + (1-weights[9])*p2.DG_min_runtime + 0.5) #Substitution for round (Numba constraint)
        
        c1.DG_min_production = weights[10]*p1.DG_min_production+(1-weights[10])*p2.DG_min_production
        c2.DG_min_production = weights[10]*p2.DG_min_production+(1-weights[10])*p1.DG_min_production
        
        c1.storages[INDIV_PRO_VOLUME,:] = weights[11]*p1.storages[0,:]+(1-weights[11])*p2.storages[0,:]
        c2.storages[INDIV_PRO_VOLUME,:] = weights[11]*p2.storages[0,:]+(1-weights[11])*p1.storages[0,:]
        
        c1.storages[INDIV_PRO_CHARGE_POWER,:] = weights[12]*p1.storages[1,:]+(1-weights[12])*p2.storages[1,:]
        c2.storages[INDIV_PRO_CHARGE_POWER,:] = weights[12]*p2.storages[1,:]+(1-weights[12])*p1.storages[1,:]
        
        c1.storages[INDIV_PRO_DISCHARGE_POWER,:] = weights[13]*p1.storages[2,:]+(1-weights[13])*p2.storages[2,:]
        c2.storages[INDIV_PRO_DISCHARGE_POWER,:] = weights[13]*p2.storages[2,:]+(1-weights[13])*p1.storages[2,:]
        
        c1.storages[INDIV_PRO_SOC_INIT,:] = weights[14]*p1.storages[3,:]+(1-weights[14])*p2.storages[3,:]
        c2.storages[INDIV_PRO_SOC_INIT,:] = weights[14]*p2.storages[3,:]+(1-weights[14])*p1.storages[3,:]
    
    return ( c1,c2,np.int64(cross_rand<r_cross))


