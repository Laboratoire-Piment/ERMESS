# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 19:25:38 2023

@author: jlegalla
"""
import ERMESS_functions as fGA
import ERMESS_cost_functions as Cfc
import ERMESS_classes as ECl
import ERMESS_functions_pro as Efp
import numpy as np

# genetic algorithm
def genetic_algorithm_avec_pertes_contraintes_parallel2(inputs):
   (Contexte,unjitted_pop)=(inputs[i] for i in range(2))   
   
   n_iter = Contexte.n_iter
   r_cross = Contexte.r_cross
   # keep track of best solution
   n_pop=len(unjitted_pop)
   pop=ECl.jitting_pop_res(unjitted_pop)
   scores = np.repeat(np.nan,n_pop)

   fitness_function_GA=Cfc.find_cost_functions(Contexte)
   activate_Y_DSM = Contexte.total_Y_Movable_load>0
 
   for i in range(n_pop):
       (pop[i].fitness,pop[i].trades)=fitness_function_GA(pop[i])
       scores[i]=pop[i].fitness
       
   best = ECl.Individual_res.copy(pop[np.argmin(scores)])
   
   operators_perf = []            
   random_elitism_prob = np.random.random(n_iter)
   random_elitism_choice = np.random.randint(0,n_pop,n_iter)   
   random_factors_set = np.random.rand(n_iter,n_pop,55)
   choices_set = np.random.choice(Contexte.n_store,(n_iter,n_pop,15),p=best.storage_sum/np.nansum(best.storage_sum) if np.nansum(best.storage_sum)>0 else np.repeat(1,Contexte.n_store)/Contexte.n_store)
   
 # enumerate generations
   for gen in range(n_iter):

     # select parents
     selected = [fGA.selection2(pop) for _ in range(n_pop)]
 # create the next generation
     children = list()
     for i in range(0, n_pop, 2):
 # get selected parents in pairs
             p1, p2 = selected[i], selected[i+1]

 # crossover and mutation
             random_factors,choices = random_factors_set[gen,i,:],choices_set[gen,i,:]
             (p1_mut,p1_ope) = fGA.NON_JIT_mutation_contraintes_3(p1 , random_factors , choices ,Contexte.n_bits, Contexte.Bounds_prod,Contexte.groups,Contexte.Non_movable_load,Contexte.constraint_num, Contexte.constraint_level , Contexte.prods_U, Contexte.prod_C, Contexte.n_store,Contexte.n_contract,Contexte.time_resolution,Contexte.storage_characteristics,Contexte.Volums_prod,Contexte.D_DSM_indexes,activate_Y_DSM,Contexte.hyperparameters_operators)
             random_factors,choices = random_factors_set[gen,i+1,:],choices_set[gen,i+1,:]
             (p2_mut,p2_ope) = fGA.NON_JIT_mutation_contraintes_3(p2 , random_factors , choices ,Contexte.n_bits, Contexte.Bounds_prod,Contexte.groups,Contexte.Non_movable_load,Contexte.constraint_num, Contexte.constraint_level , Contexte.prods_U, Contexte.prod_C, Contexte.n_store,Contexte.n_contract,Contexte.time_resolution,Contexte.storage_characteristics,Contexte.Volums_prod,Contexte.D_DSM_indexes,activate_Y_DSM,Contexte.hyperparameters_operators)

             [c1,c2,ind] = fGA.crossover_reduit(p1_mut, p2_mut, r_cross,Contexte.n_bits, Contexte.groups,Contexte.n_store,Contexte.storage_characteristics)

  # Bouclage
             c1=fGA.bouclages (c1,Contexte.n_store,Contexte.storage_characteristics,Contexte.total_Y_Movable_load,Contexte.total_D_Movable_load,Contexte.D_DSM_indexes,activate_Y_DSM)                 
             (c1.fitness,c1.trades)=fitness_function_GA(c1)

                    
             if c1.fitness < p1.fitness : 
                 operators_perf.append(np.hstack((gen,p1_ope,ind)))
             if c1.fitness < best.fitness:
                     best = ECl.Individual_res.copy(c1)

                 # store for next generation
             children.append(c1.copy())
             c2=fGA.bouclages (c2,Contexte.n_store,Contexte.storage_characteristics,Contexte.total_Y_Movable_load,Contexte.total_D_Movable_load,Contexte.D_DSM_indexes,activate_Y_DSM)
             (c2.fitness,c2.trades)=fitness_function_GA(c2) 
 
             if c2.fitness < p2.fitness : 
                 operators_perf.append(np.hstack((gen,p2_ope,ind)))
             if c2.fitness < best.fitness:
                     best = ECl.Individual_res.copy(c2)

             children.append(c2.copy())

 # replace population
     pop = children
      
     if (random_elitism_prob[gen]<0.6):
         pop[random_elitism_choice[gen]]=ECl.Individual_res.copy(best)
    #On introduit de l'élitisme
   del pop[np.random.randint(len(pop))]
   pop.append(best)

   final_pop = ECl.unjitting_pop_res(pop)

   operators_perf=np.array(operators_perf)
   if Contexte.tracking_ope == 1 :
       return [final_pop,operators_perf]
   else :
       return (final_pop)


# genetic algorithm
def genetic_algorithm_pro(inputs):
   (Contexte,pop,type_pop)=(inputs[i] for i in range(3))   
   
   Kp,Ki,Kd,u_min,u_max,anti_windup,beta = 2.0,0.05,1.0,0.0,2.0,10.0,0.3
   
   n_iter = Contexte.n_iter
   r_cross = Contexte.r_cross
   # keep track of best solution
   n_pop=len(pop)

   fitness_function_GA=Cfc.find_cost_functions(Contexte)  

   if (type_pop=='NON_JIT'):
       pop = ECl.jitting_pop_pro(pop)
   
   for i in range(n_pop):
       pop[i]=fitness_function_GA(pop[i])
   best = ECl.Individual_pro.copy(pop[np.argmin([pop[i].fitness for i in range(n_pop)])])
   
   operators_perf = []     
       
   random_elitism_prob = np.random.random(n_iter)
   random_elitism_choice = np.random.randint(0,n_pop,n_iter)
   
   random_factors_set = np.random.rand(n_iter,n_pop,43)
   choices_set = np.random.choice(Contexte.n_store,(n_iter,n_pop,13))   

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
             
     if gen<10 :
         diversities_10[gen] = diversity*beta
         mutation_factor=1
         print(gen,best.fitness,round(diversity,0),round(mutation_factor,2))
     else : 
         if (gen==10):
             diversity_threshold = np.median(diversities_10)
         integrale_PID,mutation_factor,prev_error = Efp.PID_correction(stagnation,diversity_threshold,diversity,integrale_PID,prev_error,Kp,Ki,Kd,u_min,u_max,anti_windup)
         print(gen,best.fitness,round(diversity,0),'/',round(diversity_threshold,0),round(mutation_factor,2))
         
     hyperparameters_matrix = Contexte.hyperparameters_operators.copy()
       

     
     hyperparameters_matrix = hyperparameters_matrix*mutation_factor
     
     # select parents
     selected = [fGA.selection2(pop) for _ in range(n_pop)]
 # create the next generation
     children = list()
     for i in range(0, n_pop, 2):
 # get selected parents in pairs
             p1, p2 = selected[i], selected[i+1]

 # crossover and mutation
             random_factors,choices = random_factors_set[gen,i,:],choices_set[gen,i,:]
             (p1_mut,p1_ope) = Efp.NON_JIT_mutation_contraintes_pro(p1 , random_factors , choices ,Contexte.n_bits, Contexte.Bounds_prod,Contexte.groups,Contexte.Non_movable_load,Contexte.constraint_num, Contexte.constraint_level , Contexte.prods_U, Contexte.prod_C, Contexte.n_store,Contexte.n_contract,Contexte.time_resolution,Contexte.storage_characteristics,Contexte.Volums_prod,Contexte.D_DSM_indexes,Contexte.hyperparameters_operators,Contexte.Defined_items)
             random_factors,choices = random_factors_set[gen,i+1,:],choices_set[gen,i+1,:]
             (p2_mut,p2_ope) = Efp.NON_JIT_mutation_contraintes_pro(p2 , random_factors , choices ,Contexte.n_bits, Contexte.Bounds_prod,Contexte.groups,Contexte.Non_movable_load,Contexte.constraint_num, Contexte.constraint_level , Contexte.prods_U, Contexte.prod_C, Contexte.n_store,Contexte.n_contract,Contexte.time_resolution,Contexte.storage_characteristics,Contexte.Volums_prod,Contexte.D_DSM_indexes,Contexte.hyperparameters_operators,Contexte.Defined_items)

             (c1,c2,ind) = Efp.crossover_reduit_pro(p1_mut, p2_mut, r_cross,Contexte.n_bits, Contexte.groups,Contexte.n_store,Contexte.storage_characteristics)

             c1=fitness_function_GA(c1)
             
             if c1.fitness < p1.fitness : 
                 operators_perf.append(np.hstack((gen,p1_ope,ind)))
             if c1.fitness < best.fitness:
                     best = ECl.Individual_pro.copy(c1)    
                     stagnation=0
             children.append(c1.copy()) # store for next generation
             c2=fitness_function_GA(c2) 
             if c2.fitness < p2.fitness : 
                 operators_perf.append(np.hstack((gen,p2_ope,ind)))
             if c2.fitness < best.fitness:
                     best = ECl.Individual_pro.copy(c2)
                     stagnation=0
             fitnesses[i],fitnesses[i+1]=c1.fitness,c2.fitness
             children.append(c2.copy())  # store for next generation
 # replace population
     pop = children
           
     if (random_elitism_prob[gen]<0.6):
         pop[random_elitism_choice[gen]]=ECl.Individual_pro.copy(best)
    #On introduit de l'élitisme
   del pop[np.random.randint(len(pop))]
   pop.append(best)
   final_pop = ECl.unjitting_pop_pro(pop)
   if (type_pop=='NON_JIT'):
       final_pop = fGA.pro_to_research(final_pop, Contexte)

   operators_perf=np.array(operators_perf)
   if Contexte.tracking_ope == 1 :
       return [final_pop,operators_perf]
   else :
       return (final_pop)