# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 19:23:34 2023

@author: JoPHOBEA
"""

import numpy as np
from numba import jit
from numba.experimental import jitclass
from numba import int64, float64, types

from ERMESS_scripts.evolutionnary_core import ERMESS_evolutionnary_operators as Eop
from ERMESS_scripts.energy_model import ERMESS_EMS_models as Eems
from ERMESS_scripts.data.indices import *


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
    """
    Class representing an individual in the evolutionary process (RESEARCH mode).
    Targeted for Numba JIT compilation.
    
    Attributes:
        production_set (np.ndarray of int64): Set of production units.
        storage_sum (np.ndarray of float64): Absolute sum of storage actions.
        storage_TS (np.ndarray of float64): Storage time series.
        contract (int): Contract ID.
        Y_DSM (np.ndarray of float64): Yearly Demand Side Management values.
        D_DSM (np.ndarray of float64): Daily Demand Side Management values.
        fitness (float64): Fitness score of the individual.
        trades (np.ndarray of float64): Trades time series.
    """
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
        """
        Create a copy of an Individual_res object (JIT type).
        
        Returns:
            Individual: A JIT-compatible Individual object in RESEARCH mode.
        """
        return Individual_res(self.production_set.copy(),self.storage_sum.copy(),self.storage_TS.copy(),self.contract,self.Y_DSM.copy(),self.D_DSM.copy(),self.fitness,self.trades.copy())

        
class Non_JIT_Individual_res():
    """
    Represents an individual result in the optimization process (Non-JIT version).
    
    This class mirrors `Individual_res`.
    
    Attributes:
        See `Individual_res` for full attribute descriptions.
    """
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
        """
        Create a copy of a non-JIT Individual_res object.
        
        Returns:
            Individual: A non-JIT Individual RESEARCH object.
        """
        return Non_JIT_Individual_res(self.production_set.copy(),self.storage_sum.copy(),self.storage_TS.copy(),self.contract,self.Y_DSM.copy(),self.D_DSM.copy(),self.fitness,self.trades.copy())

        
def jitting_pop_res(pop):
    """
    Convert a list of Non-JIT Individual_res objects to JIT-compatible Individual_res objects.
    
    Args:
        pop (list of Non_JIT_Individual_res): Population of non-JIT individuals.
    
    Returns:
        list of Individual_res: Population of JIT-compatible individuals.
    """
    jitted_pop=[]
    for ind in pop:
        jitted_pop.append(Individual_res(np.int64(ind.production_set),np.array(ind.storage_sum,dtype=np.float64),np.float64(ind.storage_TS),np.int64(ind.contract),np.float64(ind.Y_DSM),np.float64(ind.D_DSM),np.float64(ind.fitness),np.array(ind.trades,dtype=np.float64)))
    return(jitted_pop)

def unjitting_pop_res(jitted_pop):
    """
    Convert a list of JIT-compatible Individual_res objects back to Non-JIT Individual_res objects.
    
    Args:
        jitted_pop (list of Individual_res): Population of JIT-compatible individuals.
    
    Returns:
        list of Non_JIT_Individual_res: Population of non-JIT individuals.
    """
    pop=[]
    for ind in jitted_pop:
        pop.append(Non_JIT_Individual_res(ind.production_set,ind.storage_sum,ind.storage_TS,ind.contract,ind.Y_DSM,ind.D_DSM,ind.fitness,ind.trades))
    return(pop)
        
def pro_to_research(pop_pro,Contexte):
    """
    Transform a PRO population into a RESEARCH population.
    
    Computes time series, DSM values, and other relevant metrics for each individual.
    
    Args:
        pop_pro (list of ECl.Individual_pro): List of professional individuals.
        Contexte (object): Problem context containing required attributes:
            - prods_U, prod_C
            - Non_movable_load, total_D_Movable_load, total_Y_Movable_load
            - n_bits, n_store, time_resolution, Connexion
            - storage_characteristics
    
    Returns:
        list of ECl.Non_JIT_Individual_res: List of research individuals.
    """
    pop_res = []
    for ind_pro in pop_pro:
        production = ((Contexte.prods_U.T*ind_pro.production_set).sum(axis=1)+Contexte.prod_C)/1000    
        (storage_TS,trades,D_DSM,Y_DSM,SOCs_eff,losses,P_diff) = Eems.LFE_CCE.py_func(ind_pro, Contexte.Non_movable_load, Contexte.total_D_Movable_load, Contexte.total_Y_Movable_load, production , Contexte.n_bits,Contexte.n_store,Contexte.time_resolution, Contexte.Connexion, Contexte.storage_characteristics)
        D_DSM = D_DSM.reshape((int(Contexte.n_bits/(Contexte.time_resolution*24)),int(Contexte.time_resolution*24)))
        storage_sum=np.array([-np.sum(np.where(storage_TS[i]<0,storage_TS[i],0)) for i in range(Contexte.n_store)])
        ind_res = Non_JIT_Individual_res(ind_pro.production_set, storage_sum, storage_TS, ind_pro.contract, Y_DSM, D_DSM, np.nan, trades)
        pop_res.append(ind_res)
    return (pop_res)

def find_cost_function_research(Contexte, global_parameters, grid_parameters, RENSystems_parameters, Genset_parameters):
    """
    Determines the right loss function to apply to the optimization problem (RESEARCH).
    
    Args:
        Contexte: Description of the constraints of the problem.
    
    Returns:
        int: ID of the appropriate cost function.
    """
    try:
        criterion = CriterionEnum(Contexte.optimization.criterion_num)
        base_function = criterion.get_function_research()
    except KeyError:
        raise ValueError("No proper optimisation criterion found!")
                
    def cost_function(ind):
        return base_function(ind,RENSystems_parameters,global_parameters,Genset_parameters,grid_parameters)

    return(cost_function )

@jit(nopython=True)
def _initialize_child(parent, global_parameters):
    return Individual_res(parent.production_set,parent.storage_sum,parent.storage_TS,parent.contract,parent.Y_DSM,parent.D_DSM, np.float64(np.nan),np.full(global_parameters.n_bits, np.nan) )

@jit(nopython=True)
def _crossover_production(c1, c2, p1_mut, p2_mut,weights,RENSystems_parameters,extra_parameters):

    n_groups = len(extra_parameters.groups_size)
    group_mask = weights[5:] < 0.5
    crossover_mask = np.full(len(RENSystems_parameters.capacities),-1,dtype=np.int64)
    for group in range(n_groups):
     if group_mask[group]:
         indices = extra_parameters.groups_production[group, :extra_parameters.groups_size[group]]
         crossover_mask[indices] = indices
         
    c1.production_set = np.where(crossover_mask >= 0,p2_mut.production_set,p1_mut.production_set)
    c2.production_set = np.where(crossover_mask >= 0,p1_mut.production_set,p2_mut.production_set)

@jit(nopython=True)    
def _crossover_storage(c1, c2, p1_mut, p2_mut, alpha):

    c1.storage_TS = alpha * p1_mut.storage_TS + (1 - alpha) * p2_mut.storage_TS
    c2.storage_TS = alpha * p2_mut.storage_TS + (1 - alpha) * p1_mut.storage_TS
    
@jit(nopython=True)
def _crossover_contract(c1, c2, p1_mut, p2_mut, w1, w2):

    c1.contract = p1_mut.contract if w1 < 0.5 else p2_mut.contract
    c2.contract = p1_mut.contract if w2 < 0.5 else p2_mut.contract
    
@jit(nopython=True)    
def _crossover_Y_DSM(c1, c2, p1_mut, p2_mut, alpha):

    c1.Y_DSM = alpha * p1_mut.Y_DSM + (1 - alpha) * p2_mut.Y_DSM
    c2.Y_DSM = alpha * p2_mut.Y_DSM + (1 - alpha) * p1_mut.Y_DSM

@jit(nopython=True)
def _crossover_D_DSM(c1, c2, p1_mut, p2_mut):

    row_weights = np.random.random(len(c1.D_DSM))

    c1.D_DSM = ((p1_mut.D_DSM.T * row_weights + p2_mut.D_DSM.T * (1 - row_weights)).T)
    c2.D_DSM = ((p2_mut.D_DSM.T * row_weights + p1_mut.D_DSM.T * (1 - row_weights)).T)


@jit(nopython=True)
def crossover_reduit(p1_mut, p2_mut, r_cross,global_parameters,RENSystems_parameters,extra_parameters):
    """
    Perform a reduced crossover between two research individuals.
    
    Args:
        p1 (ECl.Individual_res): Parent 1.
        p2 (ECl.Individual_res): Parent 2.
        r_cross (float): Crossover probability.
        n_bits (int): Length of the time series.
        groups (list): Production unit groups for structured crossover.
        n_store (int): Number of storage units.
        storage_characteristics (array): Storage technical characteristics.
    
    Returns:
        tuple: A tuple containing:
            - c1 (ECl.Individual_res): Child 1.
            - c2 (ECl.Individual_res): Child 2.
            - crossover_flag (int): 1 if crossover was performed, 0 otherwise.
    """
 
    # children are copies of parents by default
    c1 = _initialize_child(p1_mut, global_parameters) 
    c2 = _initialize_child(p2_mut, global_parameters) 

 # check for recombination
    if np.random.rand() > r_cross:
        return(c1,c2,False)
    # select random weights

    n_groups = len(extra_parameters.groups_size)
    weights = np.random.random(5+n_groups)
 
    _crossover_production(c1, c2,p1_mut, p2_mut, weights,RENSystems_parameters,extra_parameters)
    _crossover_storage(c1, c2, p1_mut, p2_mut, weights[0])
    _crossover_contract(c1, c2, p1_mut, p2_mut, weights[1], weights[2])
    _crossover_Y_DSM(c1, c2, p1_mut, p2_mut, weights[3])
    _crossover_D_DSM(c1, c2, p1_mut, p2_mut)
        
    return (c1, c2,True)
    
def NON_JIT_mutation_contraintes_research(c , random_factors, choices, activate_Y_DSM, activate_D_DSM, global_parameters, RENSystems_parameters, grid_parameters, extra_parameters):
        """
        Apply a series of semi-random mutations to a research individual.
        
        Each mutation is conditioned on a random factor and a threshold defined
        in `hyperparameters_operators_num`. Mutations may include contracts,
        production, time series, storage, long-term and inter-daily consistency,
        smoothing, and DSM modifications.
        
        Args:
            c (ECl.Individual_res): Individual to mutate.
            random_factors (array): Vector of random numbers used to activate mutations.
            choices (list): Indices and information for targeted mutations.
            n_bits (int): Number of time steps.
            Bounds_prod (array): Production bounds.
            groups (list): Production unit groups.
            Non_movable_load (array): Fixed load.
            constraint_num (int): Constraint number.
            constraint_level (float): Constraint level.
            prods_U (array): Unit production conversion matrix.
            prod_C (array): Base production.
            n_store (int): Number of storage units.
            n_contracts (int): Number of contracts.
            time_resolution (float): Time step resolution.
            storage_characteristics (array): Storage technical characteristics.
            Volums_prod (array): Production volumes.
            D_DSM_indexes (array): Daily DSM indexes.
            activate_Y_DSM (bool): Whether to activate yearly DSM mutation.
            hyperparameters_operators_num (array): Matrix of operator hyperparameters.
        
        Returns:
            tuple: A tuple containing:
                - c (ECl.Individual_res): Individual after mutations.
                - usage_ope (array): Binary vector indicating which mutations were applied (1 = applied, 0 = not applied).
        """
        usage_ope = np.repeat(0, 29)
        
        MIN_LIMIT_SMOOTHING = 0.3
        MIN_LIMIT_INTERDAILY = 0.1
        LIMIT_RESOLUTION_DAILY_PATTERN = 0.1
        #MUTATION DU CONTRAT     
        if ((random_factors[RES_RF_CONTRACT]<extra_parameters.hyperparameters_operators[OPER_PROBABILITY,RESEARCH_CONTRACT]) and global_parameters.Connexion==GRID_ON) :  
              c=Eop.switch_contract_operator(c,extra_parameters.n_contracts)
              usage_ope[0]=1
              
        ## Semi-driven operator
        ## DIMINUTION DE LA PUISSANCE DU CONTRAT
        if ((random_factors[RES_RF_POWER_CONTRACT]<extra_parameters.hyperparameters_operators[OPER_PROBABILITY,RESEARCH_CONTRACT]) and np.any(c.trades>0)) :
            c=Eop.reduce_power_trading_operator(c,choices[0],extra_parameters.hyperparameters_operators)
            usage_ope[1]=1
              
        if (random_factors[RES_RF_PRODUCTION_MAIN]<extra_parameters.hyperparameters_operators[OPER_PROBABILITY,RESEARCH_PRODUCTION]) :            
             c=Eop.Mutate_production_capacity_operator(c,choices[1],RENSystems_parameters.capacities,extra_parameters.groups_production,extra_parameters.groups_size,RENSystems_parameters.unit_productions,global_parameters.n_bits,extra_parameters.hyperparameters_operators)    
             usage_ope[2]=1             
             
        if (random_factors[RES_RF_PRODUCTION_TRANSFER]<extra_parameters.hyperparameters_operators[OPER_PROBABILITY,RESEARCH_PRODUCTION]/2) :
             c=Eop.Transfer_production_capacity_operator(c,RENSystems_parameters.capacities)
             usage_ope[3]=1
                  
        if (random_factors[RES_RF_PRODUCTION_SWAP]<extra_parameters.hyperparameters_operators[OPER_PROBABILITY,RESEARCH_PRODUCTION]/10) :
                  c=Eop.Switch_intragroup_productor_operator(c,RENSystems_parameters.capacities,extra_parameters.groups_production,extra_parameters.groups_size)
                  usage_ope[4]=1
        
        #MUTATIONS DES SERIES TEMPORELLES
        #RANDOM OPERATORS
        if ((random_factors[RES_RF_MUTATION_STORAGE]<extra_parameters.hyperparameters_operators[OPER_PROBABILITY,RESEARCH_STORAGE_PATTERNS])) :
                  c=Eop.Mutate_storage_points_operator(c,choices[2],global_parameters.n_bits,extra_parameters.hyperparameters_operators)
                  usage_ope[5]=1
                   
        # Mutation aléatoire de la série temporelle sur des séquences voisines en respectant les sens
        if ((random_factors[RES_RF_STORAGE_WINDOWS_NOISE]<extra_parameters.hyperparameters_operators[OPER_PROBABILITY,RESEARCH_STORAGE_PATTERNS])) :
                  c=Eop.Mutate_storage_windows_noise_operator(c,choices[3],global_parameters.n_bits,extra_parameters.hyperparameters_operators)
                  usage_ope[6]=1

        if ((random_factors[RES_RF_STORAGE_PATTERNS]<extra_parameters.hyperparameters_operators[OPER_PROBABILITY,RESEARCH_STORAGE_PATTERNS]) and (global_parameters.time_resolution>LIMIT_RESOLUTION_DAILY_PATTERN)) :                  
                  c = Eop.Mutate_storage_dailypattern_operator(c,choices[4],global_parameters.time_resolution,global_parameters.n_bits,extra_parameters.hyperparameters_operators)
                  usage_ope[7]=1
                              
        ##MODIFICATION DE L'UTILISATION GLOBALE DES STOCKAGES
        if ((random_factors[RES_RF_STORAGE_GLOBAL]<extra_parameters.hyperparameters_operators[OPER_PROBABILITY,RESEARCH_STORAGE_GLOBAL])) :
              c=Eop.Mutate_storage_global_operator(c,choices[5],RENSystems_parameters.n_store,extra_parameters.hyperparameters_operators)
              usage_ope[8]=1             
              
        ##MODIFICATION DE L'UTILISATION DES STOCKAGES SUR DES SOUS-ENSEMBLES
        if (random_factors[RES_RF_STORAGE_WINDOWS_SCALE]<(extra_parameters.hyperparameters_operators[OPER_PROBABILITY,RESEARCH_STORAGE_GLOBAL])):
               c=Eop.Mutate_storage_windows_scaling_operator(c,choices[6],global_parameters.n_bits,extra_parameters.hyperparameters_operators)
               usage_ope[9]=1              
            
               #On introduit un transfert éventuel entre stockages
        if ((RENSystems_parameters.n_store>1) and (random_factors[RES_RF_STORAGE_MIX_POINTS]<extra_parameters.hyperparameters_operators[OPER_PROBABILITY,RESEARCH_STORAGE_MIX]) ) :
               c=Eop.Reallocate_storage_mix_points_operator(c,RENSystems_parameters.n_store,random_factors[RES_RF_STORAGE_MIX_POINTS_EFFECT],global_parameters.n_bits,extra_parameters.hyperparameters_operators)
               usage_ope[10]=1
                                                  
                #On introduit un transfert éventuel entre stockages_v2  
        if ((RENSystems_parameters.n_store>1) and (random_factors[RES_RF_STORAGE_TRANSFER]<extra_parameters.hyperparameters_operators[OPER_PROBABILITY,RESEARCH_STORAGE_MIX])) :
                 c=Eop.Transfer_storage_flows_operator(c,RENSystems_parameters.n_store,RENSystems_parameters.specs_storage,global_parameters.n_bits,extra_parameters.hyperparameters_operators)
                 usage_ope[11]=1        
        
        ##SEMI-ORIENTED OPERATORS
        ## DIMINUTION DU VOLUME D'UN STOCKAGE ALEATOIRE  
        if ((random_factors[RES_RF_STORAGE_VOLUME]<extra_parameters.hyperparameters_operators[OPER_PROBABILITY,RESEARCH_STORAGE_VOLUME]) and (sum(c.storage_TS[choices[6]])!=0)) :
            c=Eop.Reduce_storage_volume_operator(c,choices[7],RENSystems_parameters.specs_storage,global_parameters.n_bits,extra_parameters.hyperparameters_operators)
            usage_ope[12]=1                

        ## DIMINUTION DE LA PUISSANCE D'UN STOCKAGE ALEATOIRE
        if (random_factors[RES_RF_STORAGE_POWER]<(extra_parameters.hyperparameters_operators[OPER_PROBABILITY,RESEARCH_STORAGE_POWER])) :
            c=Eop.Reduce_storage_power_operator(c,choices[8],global_parameters.n_bits,extra_parameters.hyperparameters_operators)        
            usage_ope[13]=1

        ##ANNULATION DES MOUVEMENTS OPPOSES DE 2 STOCKAGES ALEATOIRES
        if (RENSystems_parameters.n_store>1) and (random_factors[RES_RF_STORAGE_OPPOSITE]<extra_parameters.hyperparameters_operators[OPER_PROBABILITY,RESEARCH_INTER_STORAGES]):
              c=Eop.Merge_opposite_storage_flows_operator(c,RENSystems_parameters.n_store,extra_parameters.hyperparameters_operators)
              usage_ope[14]=1
         
    ### ANNULATION DES DECHARGES/EXPORT ou CHARGES/IMPORTS 
        if (random_factors[RES_RF_STORAGE_TRADE_CONSISTENCY]<extra_parameters.hyperparameters_operators[OPER_PROBABILITY,RESEARCH_STORAGE_TRADES_CONSISTENCY]):
             c=Eop.Force_storage_trade_consistency_operator(c,choices[9],global_parameters.n_bits,extra_parameters.hyperparameters_operators)
             usage_ope[15]=1   

        ## Long-term consistency
        if ((random_factors[RES_RF_STORAGE_INTERDAILY_SMOOTHING]<extra_parameters.hyperparameters_operators[OPER_PROBABILITY,RESEARCH_INTERDAILY_CONSISTENCY]) and (global_parameters.duration_years>MIN_LIMIT_SMOOTHING)):
            c=Eop.Smooth_interdaily_storage_timeseries_operator(c,choices[10],random_factors[RES_RF_STORAGE_INTERDAILY_SMOOTHING_EFFECT],global_parameters.time_resolution,global_parameters.n_bits,extra_parameters.hyperparameters_operators)
            usage_ope[16]=1        

            ###Interdaily consistency
        if ((random_factors[RES_RF_STORAGE_COPY]<extra_parameters.hyperparameters_operators[OPER_PROBABILITY,RESEARCH_INTERDAILY_CONSISTENCY]) and (global_parameters.duration_years>MIN_LIMIT_INTERDAILY)):
            c=Eop.Copy_interdaily_patterns_operator(c,choices[11],random_factors[RES_RF_STORAGE_COPY_EFFECT],global_parameters.n_bits,global_parameters.time_resolution)
            usage_ope[17]=1
        
        ##APPLATISSEMENT DES COURBES
        if (random_factors[RES_RF_STORAGE_SMOOTHING]<extra_parameters.hyperparameters_operators[OPER_PROBABILITY,RESEARCH_CURVE_SMOOTHING]):
               c=Eop.Smooth_storage_noise_operator(c,choices[12],random_factors[RES_RF_STORAGE_SMOOTHING_EFFECT],global_parameters.n_bits,extra_parameters.hyperparameters_operators)
               usage_ope[18]=1
               
        #Specification des rôles des stockages
        if  (RENSystems_parameters.n_store>1 and random_factors[RES_RF_STORAGE_DISTRIBUTION]<extra_parameters.hyperparameters_operators[OPER_PROBABILITY,RESEARCH_STORAGE_MIX]):
            c=Eop.Distribute_storage_roles_operator(c,random_factors[RES_RF_STORAGE_DISTRIBUTION_EFFECT],RENSystems_parameters.n_store,RENSystems_parameters.specs_storage,global_parameters.n_bits,extra_parameters.hyperparameters_operators)
            usage_ope[19]=1
   
            ## OPERATEUR DE CONTRAINTE
        if (random_factors[RES_RF_STORAGE_CONSTRAINT]<extra_parameters.hyperparameters_operators[OPER_PROBABILITY,RESEARCH_CONSTRAINT_FORCING]) :
            c=Eop.Force_constraint_operator(c,global_parameters.constraint_num,choices[13],extra_parameters.hyperparameters_operators)
            usage_ope[20]=1
                
        #Mutation du DSM 
        if (activate_Y_DSM and (random_factors[RES_RF_YDSM_NOISE]<extra_parameters.hyperparameters_operators[OPER_PROBABILITY,RESEARCH_DSM_NOISE])) :
              c=Eop.Mutate_YDSM_points_operator(c,global_parameters.n_bits,extra_parameters.hyperparameters_operators)
              usage_ope[21]=1
        if (activate_Y_DSM and (random_factors[RES_RF_YDSM_WINDOWS]<extra_parameters.hyperparameters_operators[OPER_PROBABILITY,RESEARCH_CURVE_SMOOTHING])) :
              c=Eop.Smooth_YDSM_windows_operator(c,random_factors[RES_RF_YDSM_WINDOWS_EFFECT],global_parameters.n_bits,extra_parameters.hyperparameters_operators)
              usage_ope[22]=1
        if (activate_Y_DSM and (random_factors[RES_RF_YDSM_INTERDAILY_PATTERNS]<extra_parameters.hyperparameters_operators[OPER_PROBABILITY,RESEARCH_INTERDAILY_CONSISTENCY]) and (global_parameters.duration_years>MIN_LIMIT_INTERDAILY)):
              c=Eop.Copy_YDSM_interdaily_patterns_operator(c,random_factors[RES_RF_YDSM_INTERDAILY_PATTERNS_EFFECT],global_parameters.n_bits,global_parameters.time_resolution)
              usage_ope[23]=1   
        if (activate_Y_DSM and (random_factors[RES_RF_YDSM_TRADES_CONSISTENCY]<extra_parameters.hyperparameters_operators[OPER_PROBABILITY,RESEARCH_DSM_TRADES_CONSISTENCY])):
              c=Eop.Force_YDSM_trades_consistency_operator(c,choices[14],random_factors[RES_RF_YDSM_TRADES_CONSISTENCY_EFFECT],global_parameters.n_bits,extra_parameters.hyperparameters_operators)
              usage_ope[24]=1   
        if (activate_D_DSM and (random_factors[RES_RF_DDSM_NOISE]<extra_parameters.hyperparameters_operators[OPER_PROBABILITY,RESEARCH_DSM_NOISE])) :
                c=Eop.Mutate_DDSM_points_operator(c,extra_parameters.D_DSM_indexes,extra_parameters.hyperparameters_operators)
                usage_ope[25]=1 
        if (activate_D_DSM and (random_factors[RES_RF_DDSM_WINDOWS]<extra_parameters.hyperparameters_operators[OPER_PROBABILITY,RESEARCH_CURVE_SMOOTHING])) :
              c=Eop.Smooth_DDSM_points_operator(c,extra_parameters.D_DSM_indexes,random_factors[RES_RF_DDSM_WINDOWS_EFFECT],extra_parameters.hyperparameters_operators)
              usage_ope[26]=1
        if (activate_D_DSM and (random_factors[RES_RF_DDSM_INTERDAILY_PATTERNS]<extra_parameters.hyperparameters_operators[OPER_PROBABILITY,RESEARCH_INTERDAILY_CONSISTENCY]) and (global_parameters.duration_years>MIN_LIMIT_INTERDAILY)):
              c=Eop.Copy_DDSM_interdaily_patterns_operator(c,extra_parameters.D_DSM_indexes,random_factors[RES_RF_DDSM_INTERDAILY_PATTERNS_EFFECT],global_parameters.time_resolution)
              usage_ope[27]=1       
        if (activate_D_DSM and (random_factors[RES_RF_DDSM_TRADES_CONSISTENCY]<extra_parameters.hyperparameters_operators[OPER_PROBABILITY,RESEARCH_DSM_TRADES_CONSISTENCY])) :
              c=Eop.Force_DDSM_trades_consistency_operator(c,extra_parameters.D_DSM_indexes,global_parameters.time_resolution,choices[15],random_factors[RES_RF_DDSM_TRADES_CONSISTENCY_EFFECT],extra_parameters.hyperparameters_operators)
              usage_ope[28]=1
         
        return(c,usage_ope)

def bouclages_old(c,n_store,storage_characteristics,total_Y_movable_load,total_D_Movable_load,D_DSM_indexes ):

####Correction des TS pour boucler le stockage
    #On met tout à zéro s'il n'y a que des valeurs positives ou négatives.
    c.storage_TS = np.multiply(1-np.all(c.storage_TS<=0,axis=1),c.storage_TS.T).T
    c.storage_TS = np.multiply(1-np.all(c.storage_TS>=0,axis=1),c.storage_TS.T).T
    c.storage_sum=-np.sum(np.where(c.storage_TS<0,c.storage_TS,0),axis=1)
    
    coeffs = np.multiply(c.storage_sum,np.divide(storage_characteristics[4,:],1*(c.storage_sum==0)+np.sum(np.where(c.storage_TS>0,c.storage_TS,0),axis=1))) if np.sum(c.storage_TS>0)>0 else 1
    c.storage_TS=np.where(c.storage_TS>0,np.multiply(coeffs,c.storage_TS.T).T,c.storage_TS)
    
    if (total_Y_movable_load!=0 ) :        
        c.Y_DSM = c.Y_DSM*total_Y_movable_load/sum(c.Y_DSM)
             
    c.D_DSM[D_DSM_indexes[0],:] = np.multiply(c.D_DSM[D_DSM_indexes[0],:].T,total_D_Movable_load[D_DSM_indexes[0]]/np.sum(c.D_DSM[D_DSM_indexes[0],:],axis=1)).T
        
    return (c)

def non_JIT_bouclages(c,n_store,storage_characteristics,total_Y_movable_load,total_D_Movable_load,D_DSM_indexes ):

####Correction des TS pour boucler le stockage
    #On met tout à zéro s'il n'y a que des valeurs positives ou négatives.
    c.storage_TS = np.multiply(1-np.all(c.storage_TS<=0,axis=1),c.storage_TS.T).T
    c.storage_TS = np.multiply(1-np.all(c.storage_TS>=0,axis=1),c.storage_TS.T).T
    c.storage_sum=-np.sum(np.where(c.storage_TS<0,c.storage_TS,0),axis=1)
    absolutes = np.sum(abs(c.storage_TS),axis=1)
    volumes = absolutes-c.storage_sum*(1+np.array(storage_characteristics[4,:]))
    volumes = np.where(volumes>0,volumes/np.array(storage_characteristics[4,:]),volumes)
    ind=(volumes==0)
    
    if (sum(ind)!=n_store):
    
        ran = np.delete(np.arange(n_store),ind)
        subset =[np.where(np.sign(c.storage_TS[store])*np.sign(volumes[store])==-1)[0] for store in ran]

        lengths = np.array([len(x) for x in subset])

        subset2=np.array([x[range(min(lengths))] for x in subset])

        len_subset = np.random.randint(1,1+min(lengths))

        choices = np.random.choice(min(lengths),len_subset,replace=False)
        subset_reduced = np.array(subset2[:,choices])
        indexes = np.repeat(ran,len_subset),subset_reduced.flatten()

        coeffs = np.divide(abs(volumes[~ ind]),abs(np.sum(c.storage_TS[indexes].reshape(len(ran),len_subset),axis=1)))
     
        c.storage_TS[indexes]=c.storage_TS[indexes]*(np.repeat(1+coeffs,len_subset))
        c.storage_sum=-np.sum(np.where(c.storage_TS<0,c.storage_TS,0),axis=1)
    
    if (total_Y_movable_load!=0 ) :        
        c.Y_DSM = c.Y_DSM*total_Y_movable_load/sum(c.Y_DSM)
             
    c.D_DSM[D_DSM_indexes[0],:] = np.multiply(c.D_DSM[D_DSM_indexes[0],:].T,total_D_Movable_load[D_DSM_indexes[0]]/np.sum(c.D_DSM[D_DSM_indexes[0],:],axis=1)).T
        
    return (c)

@jit(nopython=True)
def enforce_energy_consistency(c,RENSystems_parameters,pro_parameters,extra_parameters,activate_Y_DSM ):
    """
    Adjust storage time series and DSM loads for physical consistency and long-term sustainability (JIT-compatible).
    
    Optimizes the storage state and scales DSM loads efficiently.
    
    Args:
        c (object): Individual solution containing attributes `storage_TS`, `storage_sum`, `Y_DSM`, and `D_DSM`.
        n_store (int): Number of storage units in the microgrid.
        storage_characteristics (np.ndarray): Array of storage characteristics (including charge/discharge efficiency at index 4).
        total_Y_movable_load (float): Total flexible load in the yearly (Y) DSM.
        total_D_Movable_load (np.ndarray): Total flexible load in the daily (D) DSM.
        D_DSM_indexes (np.ndarray): Indexes indicating which rows in `D_DSM` are flexible and should be scaled.
        activate_Y_DSM (bool): Flag to activate adjustment of `Y_DSM` loads.
    
    Returns:
        object: Updated individual solution with corrected `storage_TS` and scaled DSM loads.
    """
    
    n_store = RENSystems_parameters.n_store
    storage_timeseries = c.storage_TS
    storage_efficiency = RENSystems_parameters.specs_storage[STOR_ROUND_TRIP_EFF, :]
    D_DSM_indices = extra_parameters.D_DSM_indexes

    #Remove non-bdirectional storage profiles
    for i in range(n_store) : 
        if not (np.any(storage_timeseries[i]<0) and np.any(storage_timeseries[i]>0)) :
            storage_timeseries[i]=0
            
    charge_energy = np.array([-np.sum(np.where(storage_timeseries[i] < 0, storage_timeseries[i], 0.0)) for i in range(n_store)])
    discharge_energy = np.array([np.sum(np.where(storage_timeseries[i] > 0, storage_timeseries[i], 0.0)) for i in range(n_store)])
    
    c.storage_sum = charge_energy
    
    expected_discharge = charge_energy * storage_efficiency
    discharge_excess = discharge_energy - expected_discharge
    # Normalize imbalance depending on direction
    discharge_excess = np.where(discharge_excess > 0,discharge_excess / storage_efficiency,discharge_excess * storage_efficiency)

    is_balanced=(discharge_excess==0)
    
    if (sum(is_balanced)!=n_store):
    
        active_stores = np.delete(np.arange(n_store), np.where(is_balanced)[0])      
        directions = np.sign(discharge_excess)
        
        candidate_indices = [np.where(np.sign(storage_timeseries[s]) * directions[s] == -1)[0]for s in active_stores]
        candidate_lengths = np.array([len(idx) for idx in candidate_indices])
        n_selected = np.random.randint(1, 1 + np.min(candidate_lengths))
        selected_positions = np.random.choice(np.min(candidate_lengths),n_selected,replace=False)
        reduced_indices = [candidate_indices[i][selected_positions] for i in range(len(active_stores))]
        

        correction_factors = np.array([abs(discharge_excess[i])/(storage_efficiency[i] if directions[i]==-1 else 1)/abs(np.sum(storage_timeseries[i][reduced_indices[i]])) for i in range(len(active_stores))])

        for i in range(len(active_stores)):
            storage_timeseries[i][reduced_indices[i]] *= (1.0 + correction_factors[i])
            c.storage_sum[i] = -np.sum(np.where(storage_timeseries[i] < 0, storage_timeseries[i], 0.0))
    
    #Normalize Y_DSM
    if (activate_Y_DSM):    
        total_Y = np.sum(c.Y_DSM)
        if total_Y > 0:
            c.Y_DSM *= pro_parameters.total_Y_Movable_load / total_Y
       
    #Normalize D_DSM
    row_sums = np.sum(c.D_DSM[D_DSM_indices, :], axis=1)

    scaling_factors = (pro_parameters.total_D_Movable_load[D_DSM_indices] / row_sums)

    c.D_DSM[D_DSM_indices,:] = (c.D_DSM[D_DSM_indices, :].T * scaling_factors).T
    
    return (c)

def check_bouclage(c,storage_characteristics):
    """
    Evaluate the storage loop consistency for a given individual.
    
    Computes the sum of adjusted storage time series considering efficiency losses.
    
    Args:
        c (object): Individual solution containing the `storage_TS` attribute.
        storage_characteristics (np.ndarray): Array of storage characteristics, including charge/discharge efficiency at index 4.
    
    Returns:
        float: Sum of storage loops across all storage units. A value of zero indicates perfect balance.
    """
    bouclages=[]
    for i in range(len(c.storage_TS)):
        bouclages.append(sum(np.where(c.storage_TS[i]>0,c.storage_TS[i],0))+storage_characteristics[4,i]*sum(np.where(c.storage_TS[i]<0,c.storage_TS[i],0)))
    return(sum(bouclages))

def initial_population_research(inputs):
    """
    Generate an initial population of solutions with storage losses and constraints.
    
    Each individual is randomly initialized with production, storage, DSM, and contract variables.
    
    Args:
        inputs (tuple): Tuple containing the following elements:
            - n_bits (int): Number of time steps per day.
            - n_pop (int): Population size.
            - n_store (int): Number of storage units.
            - time_resolution (int): Time resolution of the simulation.
            - Bounds_prod (list): Production bounds per generator.
            - groups (list): Groups of generators for crossover operations.
            - sum_load (float): Total load in the microgrid.
            - Y_movable_load (float): Total hourly flexible load.
            - D_movable_load (np.ndarray): Total daily flexible load.
            - storage_characteristics (np.ndarray): Array describing storage characteristics (efficiencies, capacities).
            - constraint_num (int): Number of constraints.
            - constraint_level (float): Constraint intensity for initial distribution.
            - n_contracts (int): Number of contract options.
    
    Returns:
        list: List of individual solution objects (`Non_JIT_Individual_res`) initialized for the population.
    """
    (Context,  population_size, constraint_level)=tuple(inputs[i] for i in range(3))
    
    PROB_UNUSED_STORAGE = 0.1
    MEAN_POWER_STORAGE_FACTOR = 2
    SD_POWER_STORAGE_FACTOR = 1
    
    n_store = Context.storage.n_store
    n_bits = Context.time.n_bits
    
    sum_load = sum(Context.loads.non_movable + Context.loads.D_movable + Context.loads.Y_movable)
    
    #Estimation of the volume of energy that could be stored in each storage
    mean_stored_volume = sum_load/n_store*(constraint_level/1.5)**1.5
    sd_stored_volume = sum_load/n_store*(constraint_level/2)**2
    
    #Random computation of the volume stored in each storage
    stored_volumes = np.float64(np.multiply(np.random.choice([0,1],(population_size,n_store),p=[PROB_UNUSED_STORAGE,1-PROB_UNUSED_STORAGE]),abs(np.random.normal(mean_stored_volume,sd_stored_volume,(population_size,n_store)))))
        
    # Random production set
    Initial_prod_index = np.random.rand(population_size,len(Context.production.capacities))     
    Initial_prod = np.array([[np.random.randint(0,capacity,1)[0] for capacity in Context.production.capacities] for j in range(population_size)],dtype=np.int64)
    
    # For the 20 first individuals, the production set is deterministic
    Initial_prod[0:min(population_size,20)]=[((i+11)*Context.production.capacities/30).astype(int) for i in range(min(population_size,20))]
    
    # Random contracts
    if Context.grid is not None : 
        Initial_contracts = np.random.randint(0,Context.grid.n_contracts,population_size)
    else:
        Initial_contracts = np.repeat(-1,population_size)
    
    # Random Demand-side managements
    Initial_Y_DSM = [np.random.rand(n_bits) for i in range(population_size)]
    Initial_Y_DSM = [Initial_Y_DSM[i]/sum(Initial_Y_DSM[i])*sum(Context.loads.Y_movable) for i in range(population_size)]
    Initial_D_DSM = [[np.random.rand(int(Context.time.time_resolution*24)) for j in range(int(n_bits/Context.time.time_resolution/24))] for i in range(population_size)]
    Initial_D_DSM = [np.array([Initial_D_DSM[i][j]/sum(Initial_D_DSM[i][j])*np.sum(Context.loads.D_movable[j*int(Context.time.time_resolution*24):((j+1)*int(Context.time.time_resolution*24))]) for j in range(int(n_bits/Context.time.time_resolution/24))]) for i in range(population_size)]
    Initial_population = list()
    
    for j in range(population_size):
        #Keeping only the productor which has the highest Initial_prod_index, for each production group
        ones_prod=[Context.production.groups[i][np.argmax(Initial_prod_index[j][Context.production.groups[i]])] for i in range(len(Context.production.groups))]
        Initial_prod[j][np.array([i not in ones_prod for i in range(len(Context.production.capacities))])]=0      
        
        #Storage timeseries computation
        Initial_storage_power=np.zeros((n_store,n_bits))
        for i in range(n_store):
            #Random setting of charge and discharge steps. Assume that the respective length are proportionnal to the round-trip efficiency
            where_charge=np.random.choice([-1,1],n_bits-1,p=[1/(1+Context.storage.characteristics[STOR_ROUND_TRIP_EFF,:][i]),Context.storage.characteristics[STOR_ROUND_TRIP_EFF,:][i]/(1+Context.storage.characteristics[STOR_ROUND_TRIP_EFF,:][i])])
            #Finishing the timeserie with a charge to ensure at least one step of charging
            where_charge=np.append(where_charge,np.array((-where_charge[np.random.choice(n_bits-1)])))
            
            #Random setting of the power of the steps of the charge timeseries
            Initial_storage_power[i,:] = where_charge*abs(np.random.normal(MEAN_POWER_STORAGE_FACTOR,SD_POWER_STORAGE_FACTOR,n_bits))
            
            #Normalization with the global volumes stored in the storage
            Initial_storage_power[i,:][Initial_storage_power[i,:]>0] = np.round(Initial_storage_power[i,:][Initial_storage_power[i,:]>0]/sum(Initial_storage_power[i,:][Initial_storage_power[i,:]>0])*stored_volumes[j,i]*Context.storage.characteristics[STOR_ROUND_TRIP_EFF,:][i],7)
            Initial_storage_power[i,:][Initial_storage_power[i,:]<0] = -Initial_storage_power[i,:][Initial_storage_power[i,:]<0]/sum(Initial_storage_power[i,:][Initial_storage_power[i,:]<0])*stored_volumes[j,i]
        Init_pop_j = Individual_res(production_set=Initial_prod[j],storage_sum=stored_volumes[j],storage_TS=Initial_storage_power,contract=Initial_contracts[j],Y_DSM=Initial_Y_DSM[j],D_DSM=Initial_D_DSM[j],fitness=np.nan,trades=np.full([n_bits], np.float64(np.nan)))
        Initial_population.append(Init_pop_j)
        nonjit_initial_population = unjitting_pop_res(Initial_population)
       
    return(nonjit_initial_population)   

def _interpolate_signal(high_res_seq, days_sorted, n_days_total, signal_getter):
        """
        Interpolate a signal between sparse high-resolution reference days.
        
        Supports both 1D (e.g. Y_DSM) and 2D (e.g. storage_TS) arrays.
        
        Args:
            high_res_seq (list): Ordered list of high-resolution individuals
            days_sorted (np.ndarray): Sorted days corresponding to high_res_seq
            n_days_total (int): Total number of days in the final signal
            signal_getter (callable): Function extracting the signal from an individual
        
        Returns:
            np.ndarray: Interpolated full-length signal
        """

        segments = []

        # Before first known day
        first_signal = signal_getter(high_res_seq[0])
        segments.append(np.tile(first_signal, int(days_sorted[0])))


        # Interpolation between known days
        for j in range(len(high_res_seq) - 1):
            start_signal = signal_getter(high_res_seq[j])
            end_signal = signal_getter(high_res_seq[j + 1])
    
            n_steps = int(1 + days_sorted[j + 1] - days_sorted[j])
    
            interpolated = np.linspace(start_signal, end_signal, n_steps, axis=0)[:-1]
    
            segments.append(np.concatenate(interpolated, axis=-1))

        # After last known day
        last_signal = signal_getter(high_res_seq[-1])
        segments.append(np.tile(last_signal, int(n_days_total - days_sorted[-1])))

        return np.concatenate(segments, axis=-1)


def combining_solutions (population_LowRes,populations_HighRes,days,Low_time_resolution,High_time_resolution,Context):
    """
    Combine low-resolution and high-resolution optimized solutions into hybrid solutions.

     The function:
     - Merges production decisions across resolutions
     - Interpolates high-resolution trajectories over missing days
     - Injects low-resolution signals into high-resolution time series
     - Ensures consistency across storage and DSM variables

     Args:
         population_low_res (list): Individuals optimized at low temporal resolution
         populations_high_res (list of lists): Individuals optimized at high resolution (per selected day)
         days (array-like): Days associated with high-resolution solutions
         low_time_resolution (float): Time resolution of low-res solutions
         high_time_resolution (float): Time resolution of high-res solutions
         context (Environment): Simulation context

     Returns:
         list: Hybrid population
     """
    
    HOURS_PER_DAY = 24
    n_high_res_days = len(days)
    n_prod = len(Context.production.capacities)
    n_store = Context.storage.n_store
    n_population = len(population_LowRes)
    n_days_total = Context.time.n_days
    
    days_sorted_idx = np.argsort(days)
    days_sorted = np.sort(days)

    HighRes_solutions_sorted = [populations_HighRes[i] for i in days_sorted_idx] 
    
    contract_choices = np.random.choice(n_high_res_days,n_population)
    
    group_selection = np.random.choice(n_high_res_days,(n_population, len(Context.production.groups)),replace=True,p=(0.5, np.repeat(0.5/n_high_res_days,n_high_res_days)))

    #weights_prod = np.random.rand(n_pop_LowRes,n_prod,4)
    combined_population =[]
    
    for i in range(n_population):
    
        HighRes_solution_seq = [HighRes_solutions_sorted[j][i] for j in range(n_high_res_days)]
        LowRes_solution = population_LowRes[i]
            
        #Production combination
        combined_production_set=np.zeros(n_prod)
        for g,group_indices in enumerate(Context.production.groups) :
            
            candidates = [LowRes_solution.production.production_set,*[h.production.production_set for h in HighRes_solution_seq]]
            selected_source = group_selection[i][g]
            combined_production_set[group_indices] = candidates[selected_source][group_indices]
            
#            combined_production_set[Context.production.groups[k]]=[LowRes_solution.production.production_set,HighRes_solution_seq[0].production.production_set,HighRes_solution_seq[1].production.production_set,HighRes_solution_seq[2].production.production_set,HighRes_solution_seq[3].production.production_set][indexes_solution_prod[i][k]]      

        #Storage combination
        storage_base = _interpolate_signal(HighRes_solution_seq,days_sorted,n_days_total,lambda h: h.storage_TS )


        # Inject low-resolution signal
        low_res_expanded = np.repeat(LowRes_solution.storage_TS / (High_time_resolution * HOURS_PER_DAY),int(High_time_resolution / Low_time_resolution)).reshape(n_store, -1)

        combined_storage_TS = storage_base + low_res_expanded
    
        #Contract combination
        if contract_choices[i] < n_high_res_days:
            combined_contract = HighRes_solution_seq[contract_choices[i]].contract
        else:
            combined_contract = LowRes_solution.contract

        # Y_DSM combination 
   
        Y_DSM_base = _interpolate_signal(HighRes_solution_seq,days_sorted, n_days_total,lambda h: h.Y_DSM)
        Y_DSM_LowRes_expanded = np.repeat(LowRes_solution.Y_DSM / (High_time_resolution * HOURS_PER_DAY),int(High_time_resolution / Low_time_resolution))
        combined_Y_DSM = Y_DSM_base + Y_DSM_LowRes_expanded         
 
        # D_DSM combination 
   
        D_DSM_base = _interpolate_signal(HighRes_solution_seq,days_sorted, n_days_total,lambda h: h.D_DSM)
        D_DSM_LowRes_expanded = np.repeat(LowRes_solution.D_DSM / (High_time_resolution * HOURS_PER_DAY),int(High_time_resolution / Low_time_resolution)).reshape(n_days_total,int(HOURS_PER_DAY * High_time_resolution))
        combined_D_DSM = D_DSM_base + D_DSM_LowRes_expanded         
            
        # Storage correction
        combined_storage_sum = -np.sum(np.where(combined_storage_TS < 0, combined_storage_TS, 0),axis=1)

        positive_flow = np.sum( np.where(combined_storage_TS > 0, combined_storage_TS, 0),axis=1)
        
            
        if (np.any(positive_flow >0)) :
            scaling = combined_storage_sum * Context.storage.characteristics[STOR_ROUND_TRIP_EFF, :] / (positive_flow + (positive_flow == 0))
            combined_storage_TS = np.where(combined_storage_TS > 0,(combined_storage_TS.T * scaling).T,combined_storage_TS )

    
        combined_population.append(Non_JIT_Individual_res(production_set=combined_production_set,storage_sum=combined_storage_sum,storage_TS=combined_storage_TS,contract=combined_contract,Y_DSM=combined_Y_DSM,D_DSM=combined_D_DSM,fitness=np.float64(np.nan),trades=np.full(np.int64(n_days_total*High_time_resolution*HOURS_PER_DAY),np.nan,dtype=np.float64)* np.float64(np.nan)))

    return(combined_population)


