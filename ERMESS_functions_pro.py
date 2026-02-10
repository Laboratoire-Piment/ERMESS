# -*- coding: utf-8 -*-
"""
Created on Tue Jun 10 11:33:54 2025

@author: jlegalla
"""

import numpy as np
import ERMESS_classes as ECl
import ERMESS_operators as Eop
from numba import jit
                            
def initial_population_pro(n_bits, n_pop,n_store,time_resolution,Bounds_prod,groups,Non_movable_load,prod_C,prods_U,storage_characteristics,constraint_num,Constraint_level,n_contracts,Dispatching,specs_num,type_optim): 
    Initial_prod_index = np.random.rand(n_pop,len(Bounds_prod))
    Initial_prod = np.array([[np.random.randint(0,Bound,1)[0] for Bound in Bounds_prod] for j in range(n_pop)])
    Initial_prod[0:min(n_pop,20)]=[((i+11)*Bounds_prod/30).astype(int) for i in range(min(n_pop,20))]
    Initial_contracts = np.random.randint(0,n_contracts,n_pop)
    Random_PMS_strategies = np.random.rand(n_pop)
    Random_PMS_taking_overs = np.random.rand(n_pop,2,9)
    Random_energy_use_repartition_DSM = np.random.rand(n_pop)
    Random_PMS_D_DSM_min_levels = np.random.rand(n_pop,23)
    Random_PMS_Y_DSM_min_levels = np.random.rand(n_pop,11)
    Random_PMS_DG_min_runtime = np.random.randint(1,10,n_pop)
    Random_PMS_DG_min_production = np.random.rand(n_pop)
    Random_storages_init_SOCs = np.random.rand(n_store,n_pop)
    Initial_population = list()
    
    for j in range(n_pop):
        ones_prod=[groups[i][np.argmax(Initial_prod_index[j][groups[i]])] for i in range(len(groups))]
        Initial_prod[j][np.array([i not in ones_prod for i in range(len(Bounds_prod))])]=0
        prod = np.dot(Initial_prod[j],prods_U)/1000+prod_C/1000
        storage_total_capacity=np.random.uniform(0,max(abs(np.cumsum(prod-Non_movable_load))))
        storage_total_discharge_power=np.random.uniform(0,max(Non_movable_load))
        storage_total_charge_power=np.random.uniform(0,max(prod))
        distributions = np.random.rand(n_store,3)
        distributions = distributions/np.sum(distributions,axis=0)
        storages_discharge_powers = storage_total_discharge_power*distributions[:,0]
        storages_charge_powers = storage_total_charge_power*distributions[:,1]
        storages_volumes = storage_total_capacity*distributions[:,2]
        storages_SOCs_Init = Random_storages_init_SOCs[:,j]
        storages_param = np.concatenate((storages_volumes,storages_charge_powers,storages_discharge_powers,storages_SOCs_Init)).reshape(4,n_store)
        
        if ('Discharge order' in Dispatching[0]):
            PMS_Discharge_order = Dispatching[1]
            PMS_taking_over = Dispatching[2]/100
            energy_use_repartition_DSM=Dispatching[8]
        else : 
            PMS_Discharge_order = np.random.permutation(n_store)
            PMS_taking_over = np.sort(Random_PMS_taking_overs[j,:])
            energy_use_repartition_DSM=Random_energy_use_repartition_DSM[j]
            
        if ('D_DSM_levels' in Dispatching[0]):
            PMS_D_DSM_min_levels = Dispatching[3]/100
        else : 
            PMS_D_DSM_min_levels = np.sort(Random_PMS_D_DSM_min_levels[j,:])
        
        if ('Y_DSM_levels' in Dispatching[0]):
            PMS_Y_DSM_min_levels = Dispatching[4]/100
        else : 
            PMS_Y_DSM_min_levels = np.sort(Random_PMS_Y_DSM_min_levels[j,:])
            
        if ('Diesel generator' in Dispatching[0]):
            PMS_strategy = Dispatching[5]
            PMS_DG_min_runtime = int(Dispatching[6]/(60/time_resolution))
            PMS_DG_min_production = Dispatching[7]
        else : 
            PMS_strategy = 'LF' if Random_PMS_strategies[j]<0.5 else 'CC'
            PMS_DG_min_runtime = Random_PMS_DG_min_runtime[j]
            PMS_DG_min_production = max(Non_movable_load)*0.2*Random_PMS_DG_min_production[j]

        if (type_optim=='pro'): 
            Init_pop_j=ECl.Individual_pro(production_set=np.array(Initial_prod[j],dtype=np.int64),contract=Initial_contracts[j],PMS_strategy=PMS_strategy,PMS_discharge_order=PMS_Discharge_order,energy_use_repartition_DSM=energy_use_repartition_DSM,PMS_taking_over=PMS_taking_over,PMS_D_DSM_min_levels=PMS_D_DSM_min_levels,PMS_Y_DSM_min_levels=PMS_Y_DSM_min_levels,PMS_DG_min_runtime=PMS_DG_min_runtime,PMS_DG_min_production=PMS_DG_min_production,storages=storages_param,fitness=np.nan)
        else :
            Init_pop_j=ECl.Non_JIT_Individual_pro(production_set=Initial_prod[j],contract=Initial_contracts[j],PMS_strategy=PMS_strategy,PMS_discharge_order=PMS_Discharge_order,energy_use_repartition_DSM=energy_use_repartition_DSM,PMS_taking_over=PMS_taking_over,PMS_D_DSM_min_levels=PMS_D_DSM_min_levels,PMS_Y_DSM_min_levels=PMS_Y_DSM_min_levels,PMS_DG_min_runtime=PMS_DG_min_runtime,PMS_DG_min_production=PMS_DG_min_production,storages=storages_param,fitness=np.nan)
        Initial_population.append(Init_pop_j)
       
    return(Initial_population)   

@jit(nopython=True)
def compute_diversity_pro(fitnesses):

    diversity = np.std(fitnesses)

    return(diversity)

@jit(nopython=True)
def PID_correction (stagnation,diversity_threshold,diversity,integrale_PID,prev_error,Kp,Ki,Kd,u_min,u_max,anti_windup):
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

def NON_JIT_mutation_contraintes_pro(c, random_factors, choices,n_bits, Bounds_prod, groups, Non_movable_load, constraint_num, constraint_level, prods_U,prod_C, n_store, n_contracts, time_resolution, storage_characteristics,Volums_prod,D_DSM_indexes,hyperparameters_operators_num_pro,Defined_items):
       
        usage_ope = np.repeat(0, 17)
        
        #MUTATION DU CONTRAT
        if ((random_factors[0]<hyperparameters_operators_num_pro[0,0])) :  
              c=Eop.contract_operator(c,n_contracts)
              usage_ope[0]=1
                   
        #Mutation de la production           
        if (random_factors[1]<hyperparameters_operators_num_pro[0,1]) :            
             c=Eop.production_operator_pro(c,Bounds_prod,groups,hyperparameters_operators_num_pro)         
             usage_ope[1]=1

        if (random_factors[2]<(hyperparameters_operators_num_pro[0,1]/10)) :            
             c=Eop.production_ingroup_switch_operator(c,Bounds_prod,groups)         
             usage_ope[2]=1             
             
        if (random_factors[3]<hyperparameters_operators_num_pro[0,1]/2) :
                  c=Eop.production_switch_operator(c,Bounds_prod,groups)
                  usage_ope[3]=1
                  
        if ((random_factors[4]<hyperparameters_operators_num_pro[0,2]) & ('Diesel generator' not in Defined_items) ):         
                c=Eop.strategy_operator(c)
                usage_ope[4]=1
                
        if ((random_factors[5]<hyperparameters_operators_num_pro[0,3]) & ('Discharge order' not in Defined_items)) :         
                c=Eop.discharge_order_operator(c,random_factors[25:(25+n_store)])
                usage_ope[5]=1
            
        if ((random_factors[6]<hyperparameters_operators_num_pro[0,4]) & ('Discharge order' not in Defined_items)) :         
                c=Eop.energy_use_repartition_DSM_operator(c,hyperparameters_operators_num_pro)
                usage_ope[6]=1
                
        if ((random_factors[7]<hyperparameters_operators_num_pro[0,5]) & ('Discharge order' not in Defined_items)) :         
                c=Eop.PMS_taking_over_operator(c,random_factors[20:22],hyperparameters_operators_num_pro)
                usage_ope[7]=1

        if ((random_factors[8]<hyperparameters_operators_num_pro[0,6]) & ('D_DSM_levels' not in Defined_items)) :         
                c=Eop.PMS_D_DSM_min_levels_operator(c,hyperparameters_operators_num_pro)
                usage_ope[8]=1
                
        if ((random_factors[9]<hyperparameters_operators_num_pro[0,6]) & ('Y_DSM_levels' not in Defined_items)) :         
                c=Eop.PMS_Y_DSM_min_levels_operator(c,hyperparameters_operators_num_pro)
                usage_ope[9]=1
        
        if ((random_factors[10]<hyperparameters_operators_num_pro[0,7]) & ('Diesel generator' not in Defined_items)) :         
                c=Eop.PMS_DG_min_runtime_operator(c)
                usage_ope[10]=1
                
        if ((random_factors[11]<hyperparameters_operators_num_pro[0,8]) & ('Diesel generator' not in Defined_items)) :         
                c=Eop.PMS_DG_min_production_operator(c,hyperparameters_operators_num_pro)
                usage_ope[11]=1
                
        if (random_factors[12]<hyperparameters_operators_num_pro[0,9]) :         
                c=Eop.storages_capacity_operator(c,hyperparameters_operators_num_pro,choices[0])
                usage_ope[12]=1
                
        if (random_factors[13]<hyperparameters_operators_num_pro[0,10]) :         
                c=Eop.storages_inpower_operator(c,hyperparameters_operators_num_pro,choices[1])
                usage_ope[13]=1
                
        if (random_factors[14]<hyperparameters_operators_num_pro[0,10]) :         
                c=Eop.storages_outpower_operator(c,hyperparameters_operators_num_pro,choices[2])
                usage_ope[14]=1
                
        if (random_factors[15]<hyperparameters_operators_num_pro[0,11]) :         
                c=Eop.SOC_init_operator(c,random_factors[22],choices[3])
                usage_ope[15]=1
                
#        if (random_factors[15]<hyperparameters_operators_num_pro[0,9]) :         
#                c=Eop.storages_repartition(c,hyperparameters_operators_num_pro,choices[3],random_factors[23])
#                usage_ope[15]=1
                
        return(c,usage_ope)
    
@jit(nopython=True)
def crossover_reduit_pro(p1, p2, r_cross,n_bits,groups, n_store,storage_characteristics):
 
    # children are copies of parents by default
    c1 = ECl.Individual_pro(p1.production_set.copy(),p1.contract,p1.PMS_strategy,p1.PMS_discharge_order.copy(),p1.energy_use_repartition_DSM,p1.PMS_taking_over.copy(),p1.PMS_D_DSM_min_levels.copy(),p1.PMS_Y_DSM_min_levels.copy(),p1.PMS_DG_min_runtime,p1.PMS_DG_min_production,p1.storages.copy(),p1.fitness)
    c2 = ECl.Individual_pro(p2.production_set.copy(),p2.contract,p2.PMS_strategy,p2.PMS_discharge_order.copy(),p2.energy_use_repartition_DSM,p2.PMS_taking_over.copy(),p2.PMS_D_DSM_min_levels.copy(),p2.PMS_Y_DSM_min_levels.copy(),p2.PMS_DG_min_runtime,p2.PMS_DG_min_production,p2.storages.copy(),p2.fitness)

    cross_rand = np.random.rand()
 # check for recombination
    if cross_rand < r_cross:
    # select random weights

        weights = np.random.random(15+len(groups))
 # perform crossover        
        mask_prod = weights[15:len(weights)]<0.5
        cross_indexes = np.full(len(c1.production_set),-1,dtype=np.int64)
        for k in range(len(mask_prod)):
            if mask_prod[k]:
                int_idx=groups[k]
                cross_indexes[int_idx] =int_idx

        c1.production_set=np.where(cross_indexes>=0,p2.production_set,p1.production_set)
        c2.production_set=np.where(cross_indexes>=0,p1.production_set,p2.production_set)
        
        c1.contract=c1.contract if weights[1]<0.5 else c2.contract
        c2.contract=c1.contract if weights[2]<0.5 else c2.contract
        
        c1.PMS_strategy = c1.PMS_strategy if weights[3]<0.5 else c2.PMS_strategy
        c2.PMS_strategy = c2.PMS_strategy if weights[4]<0.5 else c1.PMS_strategy
        
        if (n_store>1) :
            break_point = np.random.randint(n_store-1)+1
            c1.PMS_discharge_order = np.unique(np.concatenate((p1.PMS_discharge_order[0:break_point],p2.PMS_discharge_order)))
            c2.PMS_discharge_order = np.unique(np.concatenate((p2.PMS_discharge_order[0:break_point],p1.PMS_discharge_order)))
            
        c1.energy_use_repartition_DSM = weights[5]*p1.energy_use_repartition_DSM+(1-weights[5])*p2.energy_use_repartition_DSM
        c2.energy_use_repartition_DSM = weights[5]*p2.energy_use_repartition_DSM+(1-weights[5])*p1.energy_use_repartition_DSM
        
        c1.PMS_taking_over = weights[6]*p1.PMS_taking_over+(1-weights[6])*p2.PMS_taking_over
        c2.PMS_taking_over = weights[6]*p2.PMS_taking_over+(1-weights[6])*p1.PMS_taking_over
            
        c1.PMS_D_DSM_min_levels = weights[7]*p1.PMS_D_DSM_min_levels+(1-weights[7])*p2.PMS_D_DSM_min_levels
        c2.PMS_D_DSM_min_levels = weights[7]*p2.PMS_D_DSM_min_levels+(1-weights[7])*p1.PMS_D_DSM_min_levels
        
        c1.PMS_Y_DSM_min_levels = weights[8]*p1.PMS_Y_DSM_min_levels+(1-weights[8])*p2.PMS_Y_DSM_min_levels
        c2.PMS_Y_DSM_min_levels = weights[8]*p2.PMS_Y_DSM_min_levels+(1-weights[8])*p1.PMS_Y_DSM_min_levels
        
        c1.PMS_DG_min_runtime = np.int64(round(weights[9]*p1.PMS_DG_min_runtime+(1-weights[9])*p2.PMS_DG_min_runtime))
        c2.PMS_DG_min_runtime = np.int64(round(weights[9]*p2.PMS_DG_min_runtime+(1-weights[9])*p1.PMS_DG_min_runtime))
        
        c1.PMS_DG_min_production = weights[10]*p1.PMS_DG_min_production+(1-weights[10])*p2.PMS_DG_min_production
        c2.PMS_DG_min_production = weights[10]*p2.PMS_DG_min_production+(1-weights[10])*p1.PMS_DG_min_production
        
        c1.storages[0,:] = weights[11]*p1.storages[0,:]+(1-weights[11])*p2.storages[0,:]
        c2.storages[0,:] = weights[11]*p2.storages[0,:]+(1-weights[11])*p1.storages[0,:]
        
        c1.storages[1,:] = weights[12]*p1.storages[1,:]+(1-weights[12])*p2.storages[1,:]
        c2.storages[1,:] = weights[12]*p2.storages[1,:]+(1-weights[12])*p1.storages[1,:]
        
        c1.storages[2,:] = weights[13]*p1.storages[2,:]+(1-weights[13])*p2.storages[2,:]
        c2.storages[2,:] = weights[13]*p2.storages[2,:]+(1-weights[13])*p1.storages[2,:]
        
        c1.storages[3,:] = weights[14]*p1.storages[3,:]+(1-weights[14])*p2.storages[3,:]
        c2.storages[3,:] = weights[14]*p2.storages[3,:]+(1-weights[14])*p1.storages[3,:]
    
    return ( c1,c2,np.int64(cross_rand<r_cross))


