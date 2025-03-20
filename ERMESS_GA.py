# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 19:25:38 2023

@author: jlegalla
"""
import ERMESS_functions as fGA
import ERMESS_cost_functions as Cfc
import ERMESS_classes as ECl
import numpy as np
import time
import copy
from numba import jit


# genetic algorithm
def genetic_algorithm_avec_pertes_contraintes_parallel(inputs):
   (objective, n_bits, n_iter, n_pop, r_cross, storage_characteristics,pop_init,time_resolution,duration_years,specs_prod,grid_prices,fixed_premium,Overrun,Selling_price,process,Non_movable_load,Main_grid_emissions,prod_C,prods_U,Bounds_prod,self_sufficiency)=(inputs[i] for i in range(21))
   # keep track of best solution
   pop=pop_init.copy()
   n_store = len(storage_characteristics['Technology'])
   n_contracts = len(grid_prices)
   best, best_eval = 0, 10e22
 # enumerate generations
   for gen in range(n_iter):
   #  print ('iteration ',gen,' score ' ,best_eval)
    # print(100*(gen+1)/n_iter," %")
 # evaluate all candidates in the population
  #   print(gen)
     scores = [objective(c,storage_characteristics,time_resolution,n_store,duration_years,specs_prod,grid_prices,fixed_premium,Overrun,Selling_price,Non_movable_load,Main_grid_emissions,prod_C,prods_U,Bounds_prod,self_sufficiency) for c in pop]
     
     for i in range(n_pop):
         if scores[i] < best_eval:
             best, best_eval = pop[i], scores[i]
  #           print(">%d, new best f(%s) = %.3f : ", scores[i],flush=True)
 # select parents
     selected = [fGA.selection(pop, scores) for _ in range(n_pop)]
 # create the next generation
     children = list()
     for i in range(0, n_pop, 2):
 # get selected parents in pairs
             p1, p2 = selected[i], selected[i+1]
 # crossover and mutation
             for c in fGA.crossover_avec_pertes_contraintes_3(p1, p2, r_cross,n_bits, n_store,storage_characteristics):
 # mutation
                 c=fGA.mutation_contraintes_2(c , Bounds_prod,Non_movable_load,self_sufficiency, prods_U, prod_C, n_store,n_contracts)
                 # store for next generation
                 children.append(c)
 # replace population
     pop = children
   if (process=='main GA') :
       pop.append(best)
   return pop


# genetic algorithm
def genetic_algorithm_avec_pertes_contraintes_parallel2(inputs):
   (Contexte,pop_init)=(inputs[i] for i in range(2))   
   
   n_iter = Contexte.n_iter
   r_cross = Contexte.r_cross
   # keep track of best solution
   n_pop=len(pop_init)
   scores = np.repeat(np.nan,n_pop)
   
   pop = ECl.jitting_pop(pop_init)
   

   fitness_function=Cfc.find_cost_functions(Contexte.criterion_num)[1]
   fitness_function_GA=lambda ind: fitness_function(ind,Contexte.storage_characteristics,Contexte.time_resolution,Contexte.n_store,Contexte.duration_years,Contexte.specs_num,Contexte.prices_num,Contexte.fixed_premium,Contexte.Overrun,Contexte.Selling_price,Contexte.Non_movable_load,Contexte.Main_grid_emissions,Contexte.prod_C,Contexte.prods_U,Contexte.Bounds_prod,Contexte.constraint_num,Contexte.constraint_level,Contexte.cost_constraint)
         
   for i in range(n_pop):
       (pop[i].fitness,pop[i].trades)=fitness_function_GA(pop[i])
       scores[i]=pop[i].fitness
       
   best = ECl.Individual.copy(pop[np.argmin(scores)])
   
   operators_perf = []     
       
   random_elitism_prob = np.random.random(n_iter)
   random_elitism_choice = np.random.randint(0,n_pop,n_iter)
   
   random_factors_set = np.random.rand(n_iter,n_pop,34)
   choices_set = np.random.choice(Contexte.n_store,(n_iter,n_pop,12),p=best.storage_sum/np.nansum(best.storage_sum) if np.nansum(best.storage_sum)>0 else np.repeat(1,Contexte.n_store)/Contexte.n_store)
   
 # enumerate generations
   for gen in range(n_iter):
###     print(best.storage_sum[0]/(best.storage_sum[0]+best.storage_sum[1]))

###     print ('iteration ',gen,' score ' ,best.fitness,' ',fitness_function_GA(best)[0], ' ',fGA.check_bouclage(best, Contexte.storage_characteristics) )

    # print(100*(gen+1)/n_iter," %")
 # evaluate all candidates in the population

  #   scores = [Contexte.fitness(c) for c in pop]
     
  ##   for i in range(n_pop):
  ##       if pop[i].fitness < best_eval:
  ##           best, best_eval = pop[i], pop[i].fitness
  #           print(">%d, new best f(%s) = %.3f : ", scores[i],flush=True)

     # select parents
     selected = [fGA.selection2(pop) for _ in range(n_pop)]
 # create the next generation
     children = list()
     for i in range(0, n_pop, 2):
 # get selected parents in pairs
             #t3=time.time()
             p1, p2 = selected[i], selected[i+1]
 # crossover and mutation
             ##t0 = time.time()
             random_factors,choices = random_factors_set[gen,i,:],choices_set[gen,i,:]
             (p1_mut,p1_ope) = fGA.NON_JIT_mutation_contraintes_3(p1 , random_factors , choices ,Contexte.n_bits, Contexte.Bounds_prod,Contexte.Non_movable_load,Contexte.constraint_num, Contexte.constraint_level , Contexte.prods_U, Contexte.prod_C, Contexte.n_store,Contexte.n_contract,Contexte.time_resolution,Contexte.storage_characteristics,Contexte.Volums_prod,Contexte.D_DSM_indexes,Contexte.hyperparameters_operators)

             random_factors,choices = random_factors_set[gen,i+1,:],choices_set[gen,i+1,:]
             (p2_mut,p2_ope) = fGA.NON_JIT_mutation_contraintes_3(p2 , random_factors , choices ,Contexte.n_bits, Contexte.Bounds_prod,Contexte.Non_movable_load,Contexte.constraint_num, Contexte.constraint_level , Contexte.prods_U, Contexte.prod_C, Contexte.n_store,Contexte.n_contract,Contexte.time_resolution,Contexte.storage_characteristics,Contexte.Volums_prod,Contexte.D_DSM_indexes,Contexte.hyperparameters_operators)
             [c1,c2,ind] = fGA.crossover_reduit(p1_mut, p2_mut, r_cross,Contexte.n_bits, Contexte.n_store,Contexte.storage_characteristics)
 ###            print ('crossover ',i ,best.fitness,' ',fitness_function_GA(best)[0]  )
    # mutation

                #t2=time.time()
                 #print('crossover' ,t2-t0)
         ###        c=fGA.mutation_contraintes_3(c , Contexte.Bounds_prod,Contexte.Non_movable_load,Contexte.self_sufficiency, Contexte.prods_U, Contexte.prod_C, n_store,n_contracts,Contexte.storage_characteristics,Contexte.Volums_prod,Contexte.D_DSM_indexes)
                 #t3=time.time()
                 #print('mutation' ,t3-t2)
             c1=fGA.bouclages (c1,Contexte.n_store,Contexte.storage_characteristics,Contexte.total_Y_Movable_load,Contexte.total_D_Movable_load,Contexte.D_DSM_indexes)                 
             #t4=time.time()
                 #print('bouclage' ,t4-t3)
             (c1.fitness,c1.trades)=fitness_function_GA(c1)
 ###            print ('bouclage ',i ,best.fitness,' ',fitness_function_GA(best)[0]  )
 ###            print ('bouclage 1 ',best.fitness,' ',fitness_function_GA(best)[0] )  

 ###                print(c.fitness)                     
             if c1.fitness < p1.fitness : 
                 operators_perf.append(np.hstack((gen,p1_ope,ind)))
             if c1.fitness < best.fitness:
                     best = ECl.Individual.copy(c1)
 ###                    print ('new best ' ,best.fitness,' ',fitness_function_GA(best)[0]  )
 ###                    print('new best : ',best.fitness)
                 #t5=time.time()
                 #print(t4-t3,t3-t2)
                 # store for next generation
             children.append(c1.copy())
             c2=fGA.bouclages (c2,Contexte.n_store,Contexte.storage_characteristics,Contexte.total_Y_Movable_load,Contexte.total_D_Movable_load,Contexte.D_DSM_indexes)
             (c2.fitness,c2.trades)=fitness_function_GA(c2) 
 ###            print ('bouclage ',i+1 ,best.fitness,' ',fitness_function_GA(best)[0]  )
 ###            print ('bouclage 2 ',best.fitness,' ',fitness_function_GA(best)[0] )  
             if c2.fitness < p2.fitness : 
                 operators_perf.append(np.hstack((gen,p2_ope,ind)))
             if c2.fitness < best.fitness:
                     best = ECl.Individual.copy(c2)
 ###                    print ('new best ' ,best.fitness,' ',fitness_function_GA(best)[0]  )

             children.append(c2.copy())

             ##t4=time.time()
             ##print('fin iteration : ',t4-t0)
 # replace population
     pop = children
      
     if (random_elitism_prob[gen]<0.6):
         pop[random_elitism_choice[gen]]=ECl.Individual.copy(best)
 ###        print ('added best ' ,best.fitness,' ',fitness_function_GA(best)[0]  )
    #On introduit de l'élitisme
###   print('avant')
###   for i in pop :
###       print (Contexte.fitness(i)[0])
   del pop[np.random.randint(len(pop))]
   pop.append(best)
###   print(gen,best_eval,Contexte.fitness(best)[0],min([Contexte.fitness(pop[i])[0] for i in range(24)]))
###   print('apres')
###   for i in pop :
###       print (Contexte.fitness(i)[0])
   final_pop = ECl.unjitting_pop(pop)

   operators_perf=np.array(operators_perf)
   if Contexte.tracking_ope == 1 :
       return [final_pop,operators_perf]
   else :
       return (final_pop)
