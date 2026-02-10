# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 16:21:50 2023

@author: jlegalla
"""

import ERMESS_GA as GA
import ERMESS_functions as fGA
import ERMESS_functions_pro as Efp
import ERMESS_cost_functions as Cfc
from multiprocessing import Pool

def create_initial_population(args_pop_init):
            pool = Pool()
            pop_init = pool.map(fGA.initial_population_avec_pertes_contraintes_3,args_pop_init)
            pool.close()
            pool.join()
            return(pop_init)       
        
def create_initial_population_pro(pro_args_pop_init):
            pool = Pool()
            pro_pop_init = pool.map(Efp.initial_population_pro_parallel,pro_args_pop_init)
            pool.close()
            pool.join()
            return(pro_pop_init)       
        
def changing_scale(args_ch):
            pool = Pool()
            scaled_pop=pool.map(fGA.solution_changing_scale, args_ch)
            pool.close()
            pool.join()
            return(scaled_pop)
        
def extending_scale(args_ex):
            pool = Pool()
            scaled_pop=pool.map(fGA.solution_extending_scale, args_ex)
            pool.close()
            pool.join()
            return(scaled_pop)
        
def ere_evolutive(args,ere,n_ere):
            pool = Pool()
            local_populations=pool.map(GA.genetic_algorithm_avec_pertes_contraintes_parallel2, args)
            pool.close()
            pool.join()
            return(local_populations)

def ere_evolutive_pro(args_pro):
            pool = Pool()
            local_populations_pro=pool.map(GA.genetic_algorithm_pro, args_pro)
            pool.close()
            pool.join()
            return(local_populations_pro)
        
def parallel_fitness(args_fitness):
            pool = Pool()
            local_populations=pool.map(Cfc.fitness_list, args_fitness)
            pool.close()
            pool.join()
            return(local_populations)
