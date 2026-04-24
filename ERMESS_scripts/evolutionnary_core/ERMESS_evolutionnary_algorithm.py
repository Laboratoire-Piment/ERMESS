# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 19:25:38 2023

@author: JoPHOBEA
"""

import numpy as np

from ERMESS_scripts.evolutionnary_core import ERMESS_functions as Ef
from ERMESS_scripts.evolutionnary_core import ERMESS_functions_research as Efr
from ERMESS_scripts.evolutionnary_core import ERMESS_functions_pro as Efp

from ERMESS_scripts.cost import ERMESS_cost_functions as Cfc

# genetic algorithm
def evolutionnary_algorithm_research(inputs):
   """
   Runs the genetic algorithm for microgrid optimization with constraints.
    
   This function performs selection, crossover, mutation, and elitism for a population of solutions.
   Storage and DSM (demand-side management) adjustments are applied at each generation. Fitness 
   evaluation is done using the cost functions defined in the context.
    
   Args:
        inputs (tuple): Tuple containing:
            - Contexte (object): Simulation context including microgrid parameters, constraints, and hyperparameters.
            - unjitted_pop (list): List of unjitted individual solutions (Individual_res instances).
    
   Returns:
        final_pop: List of optimized individuals (Individual_res) after n_iter generations
        operators_perf (optional): Array recording performance of operators per generation. Returned only if Contexte.tracking_ope == 1.
   """
   (Context,nonjit_pop)=(inputs[i] for i in range(2))   
   
   pop = Efr.jitting_pop_res(nonjit_pop)
   n_iter = Context.hyperparameters.n_iter
   r_cross = Context.hyperparameters.r_cross
   n_pop = Context.hyperparameters.n_pop

   pro_parameters, global_parameters, grid_parameters, RENSystems_parameters, Genset_parameters, extra_parameters = Ef.build_numba_params(Context,'research')
   fitness_function_GA=Ef.find_cost_function_research(Context, global_parameters, grid_parameters, RENSystems_parameters, Genset_parameters)
   activate_Y_DSM = Context.loads.total_Y_movable>0
   activate_D_DSM = (Context.time.time_resolution>1/24 and (len(extra_parameters.D_DSM_indexes)>0))
 
   for i in range(n_pop):
       (pop[i].fitness,pop[i].trades)=fitness_function_GA(pop[i])
       
   best = Efr.Individual_res.copy(pop[np.argmin([pop[i].fitness for i in range(n_pop)])])
          
   operators_perf = []            
   random_elitism_prob = np.random.random(n_iter)
   random_elitism_choice = np.random.randint(0,n_pop,n_iter)   
   random_factors_set = np.random.rand(n_iter,n_pop,55)
   choices_set = np.random.choice(Context.storage.n_store,(n_iter,n_pop,15),p=best.storage_sum/np.nansum(best.storage_sum) if np.nansum(best.storage_sum)>0 else np.repeat(1,Context.storage.n_store)/Context.storage.n_store)
   
 # enumerate generations
   for gen in range(n_iter):

     # select parents
     selected = [Ef.selection_tournament(pop) for _ in range(n_pop)]
 # create the next generation
     children = list()
     for i in range(0, n_pop, 2):
 # get selected parents in pairs
             p1, p2 = selected[i], selected[i+1]

 # crossover and mutation
             random_factors,choices = random_factors_set[gen,i,:],choices_set[gen,i,:]
             (p1_mut,p1_ope) = Efr.NON_JIT_mutation_contraintes_research(p1 , random_factors, choices, activate_Y_DSM, activate_D_DSM, global_parameters, RENSystems_parameters, grid_parameters, extra_parameters)
             random_factors,choices = random_factors_set[gen,i+1,:],choices_set[gen,i+1,:]
             (p2_mut,p2_ope) = Efr.NON_JIT_mutation_contraintes_research(p2 , random_factors, choices, activate_Y_DSM, activate_D_DSM, global_parameters, RENSystems_parameters, grid_parameters, extra_parameters)

             [c1,c2,ind] = Efr.crossover_reduit(p1_mut, p2_mut, r_cross,global_parameters,RENSystems_parameters,extra_parameters)

  # Bouclage
             c1=Efr.enforce_energy_consistency (c1,RENSystems_parameters,pro_parameters,extra_parameters,activate_Y_DSM )                 
             (c1.fitness,c1.trades)=fitness_function_GA(c1)

                    
             if c1.fitness < p1.fitness : 
                 operators_perf.append(np.hstack((gen,p1_ope,ind)))
             if c1.fitness < best.fitness:
                     best = Efr.Individual_res.copy(c1)

                 # store for next generation
             children.append(c1.copy())
             c2=Efr.enforce_energy_consistency (c2,RENSystems_parameters,pro_parameters,extra_parameters,activate_Y_DSM)
             (c2.fitness,c2.trades)=fitness_function_GA(c2) 
 
             if c2.fitness < p2.fitness : 
                 operators_perf.append(np.hstack((gen,p2_ope,ind)))
             if c2.fitness < best.fitness:
                     best = Efr.Individual_res.copy(c2)

             children.append(c2.copy())

 # replace population
     pop = children
      
     if (random_elitism_prob[gen]<Context.hyperparameters.elitism_probability):
         pop[random_elitism_choice[gen]]=Efr.Individual_res.copy(best)
    #On introduit de l'élitisme
   del pop[np.random.randint(len(pop))]
   pop.append(best)

   final_pop = Efr.unjitting_pop_res(pop)

   operators_perf=np.array(operators_perf)
   if Context.tracking.tracking_operators == 1 :
       return [final_pop,operators_perf]
   else :
       return (final_pop)

def evolutionnary_algorithm_pro(inputs):
   """
   Runs the PRO mode of the genetic algorithm for microgrid optimization.
    
   This function integrates PID-controlled mutation factors to maintain population diversity, 
   applies selection, crossover, mutation, and elitism, and evaluates fitness using cost functions.
    
   Args:
        inputs (tuple): Tuple containing:
            - Contexte (object): Simulation context including microgrid parameters, constraints, storage characteristics, and operator hyperparameters.
            - pop (list): List of initial individual solutions (Individual_pro instances).
            - type_pop (str): Type of population; 'NON_JIT' indicates unjitted individuals requiring preprocessing.
    
   Returns:
        final_pop (list): List of optimized individuals (Individual_pro) after n_iter generations.
        operators_perf (np.ndarray, optional): Array recording performance of genetic operators per generation. Returned only if Contexte.tracking_ope == 1.
   
   Warning:
       PID parameters are tuned for stability.
       Modifying them may lead to divergence of the algorithm.  
   """
   (Context,pop) = inputs
   
   # Initialization of the constants of the algorithm
   
   pid = Efp.PIDConfig()
   pid.validate()

   Kp,Ki,Kd,u_min,u_max,anti_windup,beta = pid.Kp,pid.Ki,pid.Kd,pid.u_min,pid.u_max,pid.anti_windup,pid.beta
   
   n_iter = Context.hyperparameters_pro.n_iter
   r_cross = Context.hyperparameters_pro.r_cross
   n_pop=Context.hyperparameters_pro.n_pop
   
   PID_init_steps = 10
   n_random_factors = 43
   n_random_choices = 13

   pro_parameters, global_parameters, grid_parameters, RENSystems_parameters, Genset_parameters, extra_parameters = Ef.build_numba_params(Context,'pro')
   fitness_function_GA=Ef.find_cost_function_pro(Context,pro_parameters, global_parameters, grid_parameters, RENSystems_parameters, Genset_parameters)  
   
   for i in range(n_pop):
       pop[i]=fitness_function_GA(pop[i])
   best = Efp.Individual_pro.copy(pop[np.argmin([pop[i].fitness for i in range(n_pop)])])
   
   operators_perf = []     
       
   random_elitism_prob = np.random.random(n_iter)
   random_elitism_choice = np.random.randint(0,n_pop,n_iter)
   
   random_factors_set = np.random.rand(n_iter,n_pop,n_random_factors)
   choices_set = np.random.choice(Context.storage.n_store,(n_iter,n_pop,n_random_choices))   

   # for diversity :
   diversities_10 = np.zeros(10)
   integrale_PID = 0 
   prev_error = 0
   fitnesses = np.array([ind.fitness for ind in pop])
   stagnation=0.0

 # enumerate generations
   for gen in range(n_iter):
           
                  
 #if needed, calculation of the diversity   
     diversity = Efp.compute_diversity_pro(fitnesses)
     stagnation = stagnation+1
     
     if gen<PID_init_steps :
         diversities_10[gen] = diversity*beta
         mutation_factor=1
         print(gen,best.fitness,round(diversity,0),round(mutation_factor,2))
     else : 
         if (gen==10):
             diversity_threshold = np.median(diversities_10)
         integrale_PID,mutation_factor,prev_error = Efp.PID_correction(stagnation,diversity_threshold,diversity,integrale_PID,prev_error,Kp,Ki,Kd,u_min,u_max,anti_windup)
         print(gen,best.fitness,round(diversity,0),'/',round(diversity_threshold,0),round(mutation_factor,2))
         
     hyperparameters_matrix = Context.hyperparameters_pro.operators_parameters.copy()     
     hyperparameters_matrix = hyperparameters_matrix*mutation_factor
     
     # select parents
     selected = [Ef.selection_tournament(pop) for _ in range(n_pop)]
 # create the next generation
     children = list()
     for i in range(0, n_pop, 2):
 # get selected parents in pairs
             p1, p2 = selected[i], selected[i+1]

 # crossover and mutation
             random_factors,choices = random_factors_set[gen,i,:],choices_set[gen,i,:]
             (p1_mut,p1_ope) = Efp.NON_JIT_mutation_contraintes_pro(p1 , random_factors , choices ,global_parameters, RENSystems_parameters ,grid_parameters ,extra_parameters )
             random_factors,choices = random_factors_set[gen,i+1,:],choices_set[gen,i+1,:]
             (p2_mut,p2_ope) = Efp.NON_JIT_mutation_contraintes_pro(p2 , random_factors , choices ,global_parameters, RENSystems_parameters ,grid_parameters ,extra_parameters)

             (c1,c2,ind) = Efp.crossover_reduit_pro(p1_mut, p2_mut, r_cross,RENSystems_parameters , extra_parameters)

             c1=fitness_function_GA(c1)
             
             if c1.fitness < p1.fitness : 
                 operators_perf.append(np.hstack((gen,p1_ope,ind)))
             if c1.fitness < best.fitness:
                     best = Efp.Individual_pro.copy(c1)    
                     stagnation=0
             children.append(c1.copy()) # store for next generation
             c2=fitness_function_GA(c2) 
             if c2.fitness < p2.fitness : 
                 operators_perf.append(np.hstack((gen,p2_ope,ind)))
             if c2.fitness < best.fitness:
                     best = Efp.Individual_pro.copy(c2)
                     stagnation=0
             fitnesses[i],fitnesses[i+1]=c1.fitness,c2.fitness
             children.append(c2.copy())  # store for next generation
 # replace population
     pop = children
           
     if (random_elitism_prob[gen]<Context.hyperparameters_pro.elitism_probability):
         pop[random_elitism_choice[gen]]=Efp.Individual_pro.copy(best)
    #On introduit de l'élitisme
   del pop[np.random.randint(len(pop))]
   pop.append(best)
   final_pop = Efp.unjitting_pop_pro(pop)
#   final_pop=pop
#   if (type_pop=='NON_JIT'):
#       final_pop = Efr.pro_to_research(final_pop, Contexte)

   operators_perf=np.array(operators_perf)
   if Context.tracking.tracking_operators :
       return [final_pop,operators_perf]
   else :
       return (final_pop)
