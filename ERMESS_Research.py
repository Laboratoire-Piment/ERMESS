# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 15:10:39 2024

@author: JoPHOBEA

Runs an era in the ERMESS RESEARCH algorithm

Modules used:
    - data.data_classes
    - data.read_excel
    - evolutionnary_core
    
important :
    -Works only on a parallel cluster
    
"""

import os
import sys
import numpy as np
import pandas as pd
import warnings
import pickle
import time
from multiprocessing import freeze_support,set_start_method

from tkinter import filedialog

from ERMESS_scripts.data import data_classes as Dcl
from ERMESS_scripts.data import read_excel as Eex
from ERMESS_scripts.data import data_parsers as Edp
from ERMESS_scripts.data import data_builder as Dbl

from ERMESS_scripts.evolutionnary_core import ERMESS_research_initialisation as Eri
from ERMESS_scripts.evolutionnary_core import ERMESS_parallel_processing as ppGA

warnings.simplefilter(action='ignore', category=UserWarning)

def ERMESS_research(input_file_path = None,initialisation = False) :
    
    freeze_support()
   
    ##============================================================================
    # 1. INPUT DATA LOADING
    ##============================================================================
    
    if input_file_path is None:
        input_file_path = filedialog.askopenfilename(filetypes =[('Excel Files', '*.xlsx')], initialdir="#Specify the file path")
    
    data = Eex.load_excel(input_file_path)
    Edp._data_validation(data)
    structured_data = Edp._parse_ERMESSInputs(data)

    ##============================================================================
    # 2. ENVIRONMENT INITIALIZATION
    ##============================================================================
              
    Context=Dbl.build_environment(structured_data)
        
    if (initialisation) : 
        Initial_population = Eri.Initialisation_ERMESS_research(Context, structured_data)
    
    ##### A utiliser seulement sur le calculateur UNIX
    set_start_method('spawn')
    
    #Avec pré-traitement
    ##print('Début de l\'algorithme évolutif principal')
    
    num_node = sys.argv[1]
    cost_phase = int(sys.argv[2])
    
    Contexte=Dcl.Non_JIT_Environnement(storage_characteristics, time_resolution, n_store, duration_years, specs_num, groups, n_bits, prices_num, fixed_premium, Overrun, Selling_price, Non_movable_load,Y_movable_load,D_movable_load, Main_grid_emissions, Grid_Fossil_fuel_ratio,Main_grid_PoF_ratio, prod_C, prods_U, Volums_prod,Bounds_prod,DG_fuel_cost,DG_lifetime,DG_unit_cost,DG_maintenance_cost,DG_fuel_consumption,DG_EROI,fuel_CO2eq_emissions, constraint_num, Constraint_level, criterion_num, cost_constraint,r_cross,n_iter,hyperparameters_operators_num,type_optim,Connexion,Defined_items,tracking_ope=1)    

    Initial_populations = []
    
    t2=time.time()
    print('fin pre-processing ',t2-t1)
    

    for file_number in range(n_nodes):
        with open('pop_'+str(file_number)+'_for_'+str(num_node)+'.dat', 'rb') as input_file:
            added_pop =pickle.load(input_file)
            Initial_populations.append(added_pop)
    
    Initial_populations=[x for xs in Initial_populations for x in xs]
        
    #Initial_populations = ECl.jitting_pop_res(Initial_populations)
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

if __name__ == '__main__':
    ERMESS_research(sys.argv[1])

