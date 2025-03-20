# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 15:10:39 2024

@author: jlegalla
"""

import os
import sys
import numpy as np
import pandas as pd
import ERMESS_classes as ECl
import ERMESS_functions_2 as fGA2
import ERMESS_functions as fGA
import ERMESS_GA as GA
import ERMESS_cost_functions as Cfc
import warnings
import ERMESS_parallel_processing as ppGA
from multiprocessing import freeze_support,set_start_method
import pickle
import time
from numba import jit

from tkinter import filedialog


warnings.simplefilter(action='ignore', category=UserWarning)


if __name__ == '__main__':
    
    freeze_support()
    t1=time.time()
    ##### A utiliser seulement sur le calculateur UNIX
    set_start_method('spawn')

    file_name = 'inputs_GEMS_frontal.xlsx' # shows dialog box and return the path

    xl_file = pd.ExcelFile(file_name)
    
    Data = {sheet_name:xl_file.parse(sheet_name) for sheet_name in xl_file.sheet_names}
    
    (datetime,Grid_Fossil_fuel_ratio,Main_grid_emissions,Main_grid_ratio,specs_names,specs_Id,specs_num,prod_C,prods_U,Volums_prod,Y_movable_load,D_movable_load,Non_movable_load,time_resolution,constraint_num, Constraint_level,criterion_num,storage_techs,storage_characteristics_names,storage_characteristics,Bounds_prod,n_UP,duration_years,prices_hour_type,prices_num,fixed_premium,Overrun,Selling_price,Contract_Id,hyperparameters_main,hyperparameters_operators_names,hyperparameters_operators_num,Cost_sequence)=fGA2.read_data(Data)
    hyperparameters_main_evol = hyperparameters_main['Evolution']
    
    n_contracts = np.int64(len(Selling_price))
    n_store = np.int64(len(storage_techs))
    
    #Définition des variables de l'algorithme génétique
    n_steps=prod_C.size
    
    n_core,n_nodes,r_cross,n_pop,n_iter,nb_ere = hyperparameters_main_evol
    
    n_bits = n_steps
    n_days=n_bits/time_resolution/24
    
    
    #Avec pré-traitement
    ##print('Début de l\'algorithme évolutif principal')
    
    num_node = sys.argv[1]
    cost_phase = int(sys.argv[2])
    cost_constraint=Cost_sequence[cost_phase]
    
    Contexte=ECl.Non_JIT_Environnement(storage_characteristics, time_resolution, n_store, duration_years, specs_num, n_bits, prices_num, fixed_premium, Overrun, Selling_price, Non_movable_load,Y_movable_load,D_movable_load, Main_grid_emissions, prod_C, prods_U, Volums_prod,Bounds_prod, constraint_num, Constraint_level, criterion_num, cost_constraint,r_cross,n_iter,hyperparameters_operators_num,tracking_ope=1)    
    
    Initial_populations = []
    
    t2=time.time()
    print('fin pre-processing ',t2-t1)
    
    for file_number in range(n_nodes):
        with open('pop_'+str(file_number)+'_for_'+str(num_node)+'.dat', 'rb') as input_file:
                    added_pop =pickle.load(input_file)
                    Initial_populations.append(added_pop)
    
    Initial_populations=[x for xs in Initial_populations for x in xs]
    Initial_populations = sorted(Initial_populations, key=lambda x: np.random.rand())
    Initial_populations = [[Initial_populations[i] for i in range(j*n_pop, (j+1)*n_pop)] for j in range(n_core)] 
    args1_2 = [(Contexte,Initial_populations[i]) for i in range(n_core)]  
    
    t3=time.time()    
    operators=[]
    print('populations récupérées ',t3-t2)
    for ere in range(nb_ere):
        print('Ere ',ere+1,'/',nb_ere,' ',time.time())
        print('Evolution indépendante des populations sur chacune des îles sur ',n_iter,' générations')
        t4=time.time()
        local_populations_1=ppGA.ere_evolutive(args1_2, ere,nb_ere)
    
        world_population1 = [item for sublist in local_populations_1 for item in sublist[0]]
        operators.append([item for sublist in local_populations_1 for item in sublist[1]])
        #world_scores1 = [item for sublist in scores1 for item in sublist]
        #world_population1=[world_population1[i] for i in np.argsort(scores1)[0:(n_pop*n_core)]]

        world_mix = [world_population1[i] for i in np.random.choice(range(n_pop*n_core),n_pop*n_core,replace=False)]
        init_population_2 = [world_mix[(island*n_pop):((island+1)*n_pop)] for island in range(n_core)]
        
        #file_name_out = 'output_GEMS_'+str(ere)+'.xlsx'
        #fGA2.post_traitement(world_population1[np.argmin(tuple(world_population1[i].fitness for i in range(len(world_population1))))],Cfc.cost_scenario_LCOE_detail, Cfc.cost_base, prod_C, prods_U, Non_movable_load,D_movable_load,Y_movable_load, storage_characteristics, time_resolution, Main_grid_emissions,Grid_Fossil_fuel_ratio,Main_grid_ratio,specs,duration_years,grid_prices,fixed_premium,Overrun,Selling_price,Contract_Id,Bounds_prod,self_sufficiency,Data,n_days,file_name_out)
        #print('Mélange aléatoire des populations résultantes')
        print('Score du meilleur individu actuel : ',min(tuple(world_population1[i].fitness for i in range(len(world_population1)))))
        args1_2 = [(Contexte,init_population_2[i],n_iter,r_cross) for i in range(n_core)]
        t5=time.time()
        print('Ere ',t5-t4)
        
    print('Fin de l\'algorithme évolutif ',time.time())
    world_population3 = [item for sublist in local_populations_1 for item in sublist[0]]
    scores3 = tuple(world_population3[i].fitness for i in range(len(world_population3)))
    shuffled_population = sorted(world_population3, key=lambda x: np.random.rand())
        
    for file_number in range(n_nodes):
        os.remove("pop_"+str(num_node)+'_for_'+str(file_number)+".dat")
        with open("pop_"+str(num_node)+'_for_'+str(file_number)+".dat", "wb") as f:
            #for indiv in shuffled_population[int(j*n_pop*n_core/10):int((j+1)*n_pop*n_core/10)]:
                pickle.dump(shuffled_population[int(file_number*n_pop*n_core/n_nodes):int((file_number+1)*n_pop*n_core/n_nodes)], f)    
    
    try:
        os.remove("operators_"+str(num_node)+".dat")
    except OSError:
        pass
    
    with open("operators_"+str(num_node)+".dat", "wb") as f:
        pickle.dump(operators, f)

    t6=time.time()
    print('fin ',t6-t5)    
    print('temps total : ',t6-t1)      
    #GA_solution = world_population3[np.argmin(scores3)]

    #file_name_out='output_GEMS_end.xlsx'
    #fGA2.post_traitement(GA_solution,Cfc.cost_scenario_LCOE_detail, Cfc.cost_base, prod_C, prods_U, Non_movable_load,D_movable_load,Y_movable_load, storage_characteristics, time_resolution, Main_grid_emissions,Grid_Fossil_fuel_ratio,Main_grid_ratio,specs,duration_years,grid_prices,fixed_premium,Overrun,Selling_price,Contract_Id,Bounds_prod,self_sufficiency,Data,n_days,file_name_out)
