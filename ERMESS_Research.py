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
import random

from tkinter import filedialog

from ERMESS_scripts.data import data_classes as Dcl
from ERMESS_scripts.data import read_excel as Eex
from ERMESS_scripts.data import data_parsers as Edp
from ERMESS_scripts.data import data_validation as Edv
from ERMESS_scripts.data import data_builder as Dbl

from ERMESS_scripts.evolutionnary_core import ERMESS_research_initialisation as Eri
from ERMESS_scripts.evolutionnary_core import ERMESS_parallel_processing as ppGA

warnings.simplefilter(action='ignore', category=UserWarning)

def ERMESS_research(node_id , input_file_path = None,initialisation = False) :
    
    freeze_support()
   
    ##============================================================================
    # 1. INPUT DATA LOADING
    ##============================================================================
    
    if input_file_path is None:
        input_file_path = filedialog.askopenfilename(filetypes =[('Excel Files', '*.xlsx')], initialdir="#Specify the file path")
    
    data = Eex.load_excel(input_file_path)
    Edv._data_validation(data)
    structured_data = Edp._parse_ERMESSInputs(data)

    ##============================================================================
    # 2. ENVIRONMENT INITIALIZATION
    ##============================================================================
              
    Context=Dbl.build_environment(structured_data)
        
    if (initialisation) : 
        Eri.Initialize_ERMESS_research(Context , structured_data, node_id)
        
    run_ERMESS_research(Context, structured_data.hyperparameters.nb_ere, structured_data.hyperparameters.n_core, node_id, structured_data.hyperparameters.n_nodes)
        
        
#    run_ERMESS_research(Context, Initial_populations, nb_ere, n_iter, n_pop, structured_data.hyperparameters.n_core, node_id, n_nodes)
    
    ##### A utiliser seulement sur le calculateur UNIX
#    set_start_method('spawn')
    
    #Avec pré-traitement
    ##print('Début de l\'algorithme évolutif principal')
    
#    for file_number in range(n_nodes):
#        with open('pop_'+str(file_number)+'_for_'+str(num_node)+'.dat', 'rb') as input_file:
#            added_pop =pickle.load(input_file)
#            Initial_populations.append(added_pop)
    
        
    #Initial_populations = ECl.jitting_pop_res(Initial_populations)
#    Initial_populations = sorted(Initial_populations, key=lambda x: np.random.rand())
    
#    t3=time.time()    
#    operators=[]
#    print('populations récupérées ',t3-t2)
#    for ere in range(nb_ere):
#        print('Ere ',ere+1,'/',nb_ere,' ',time.time())
#        print('Evolution indépendante des populations sur chacune des îles sur ',n_iter,' générations')
#        t4=time.time()
#        local_populations_1=ppGA.ere_evolutive(args1_2, ere,nb_ere)
    
#        world_population1 = [item for sublist in local_populations_1 for item in sublist[0]]
#        operators.append([item for sublist in local_populations_1 for item in sublist[1]])

#        world_mix = [world_population1[i] for i in np.random.choice(range(n_pop*n_core),n_pop*n_core,replace=False)]
#        init_population_2 = [world_mix[(island*n_pop):((island+1)*n_pop)] for island in range(n_core)]
        
#        print('Score du meilleur individu actuel : ',min(tuple(world_population1[i].fitness for i in range(len(world_population1)))))
#        args1_2 = [(Contexte,init_population_2[i],n_iter,r_cross) for i in range(n_core)]
#        t5=time.time()
#        print('Ere ',t5-t4)

def load_migrants(files):
    pool = []
    for f in files:
        with open(f, "rb") as infile:
            pool.extend(pickle.load(infile))
    return pool

def write_migrants(migrants, node_id, ere):
    tmp_name = f"migrants_node_{node_id}_ere_{ere}.pkl.tmp"
    final_name = f"migrants_node_{node_id}_ere_{ere}.pkl"

    with open(tmp_name, "wb") as f:
        pickle.dump(migrants, f)

    os.rename(tmp_name, final_name) 
        
def collect_migrants(migrants,len_incomers):
    selected_migrants = random.sample(migrants, len_incomers)
    return(selected_migrants)

def select_migrants_internodes(len_pop, MIGRATION_TOP_RATE, MIGRATION_RANDOM_RATE,n_core):
    N_TOP = int(MIGRATION_TOP_RATE*len_pop)
    N_RAND = int(MIGRATION_RANDOM_RATE*len_pop)
    indices = np.random.choice(len_pop - N_TOP, N_RAND, replace = False)+N_TOP
    #First part of rand : migrants (copied), second part : replaced by incoming migrants
    return np.array([*np.arange(N_TOP),*indices])

def select_replaced_internodes(len_pop, MIGRATION_TOP_RATE, len_incomers,n_core):

    len_incomers_local = len_incomers//n_core
    N_TOP = int(MIGRATION_TOP_RATE*len_pop)
    N_RAND = int(len_incomers_local - N_TOP)

    if (N_RAND + N_TOP) > (len_pop - N_TOP):
        raise ValueError("Migration size too large")
    replaced_indices = np.random.choice(len_pop - N_TOP, N_RAND+N_TOP, replace = False)+N_TOP
    return(replaced_indices)

def replace_population_internodes (n_core,len_pop,local_populations,incomers,killed_indices,MIGRATION_TOP_RATE,MIGRATION_RANDOM_RATE):     
    
    if (n_core*len(killed_indices) != len(incomers)):
        print(n_core,killed_indices)
        print(len(incomers))
        raise ValueError("Mismatch migration sizes")
    for j in range(n_core*len(killed_indices)) :
        core = j//len(killed_indices)
        index = j%len(killed_indices)
        local_populations[core][index] = incomers[j]
   #     if (type(incomers[j])==list):
   #         print('detected list incomer',j)
   #         safe_assign(local_populations, i, killed_indices[j], incomers_chunk[j])
    
    return(local_populations)

def select_migrants_intranode(len_pop, MIGRATION_TOP_RATE, MIGRATION_RANDOM_RATE):
    N_TOP = int(MIGRATION_TOP_RATE*len_pop)
    N_RAND = int(MIGRATION_RANDOM_RATE*len_pop)
    rand = np.random.choice(len_pop - N_TOP, N_RAND+N_TOP, replace = False)+N_TOP
    #First part of rand : random migrants (moved and replaced), second part : replaced by copies of top
    return np.array([*np.arange(N_TOP),*rand[0:N_RAND]]),rand

def migration_process(local_populations, len_pop, MIGRATION_TOP_RATE, MIGRATION_RANDOM_RATE, n_core):
        migrants_indices = select_migrants_intranode(len_pop, MIGRATION_TOP_RATE, MIGRATION_RANDOM_RATE)
        migrants_pool = [local_populations[i][index] for index in migrants_indices[0]  for i in range(n_core)]
        #migrants_pool = [item for sublist in migrants for item in sublist[0] ]
        target_core = np.random.permutation(np.repeat(np.arange(n_core), len(migrants_indices[0])))
        for i in range(n_core) : 
            mask = np.where(target_core == i)[0]
            for j in range(len(migrants_indices[1])):
                idx = migrants_indices[1][j]
                local_populations[i][idx] = migrants_pool[mask[j]]
        return(local_populations)
    
def wait_for_all(ere, n_nodes, timeout=1200, sleep_time=5):
    """
    Wait for all migrant files to be present.
    """
    start_time = time.time()
    prefix = f"migrants_node_"
    suffix = f"_ere_{ere}.pkl"
    directory = "." 

    while True:
        files = [f for f in os.listdir(directory) if f.startswith(prefix) and f.endswith(suffix)]
        stable = False
        while not stable:
            sizes1 = [os.path.getsize(f) for f in files if os.path.exists(f)]
            time.sleep(0.5)
            sizes2 = [os.path.getsize(f) for f in files if os.path.exists(f)]
            stable = sizes1 == sizes2 and len(sizes1) == n_nodes

        if len(files) >= n_nodes:
            return files

        if time.time() - start_time > timeout:
            print(f"[WARNING] Timeout atteint à l'ère {ere}. {len(files)}/{n_nodes} fichiers trouvés.")
            return files

        time.sleep(sleep_time)

import traceback

def inspect(obj, name="obj"):
    """
    Détecte et trace toute apparition de list au lieu de Individual_res.
    """
    for o in obj:
        if isinstance(o, list):
            print("\n💥 LIST DETECTED:", name)
            print("   type:", type(o))
            print("   len:", len(o))
            print("   stack trace:")
            traceback.print_stack(limit=5)
            return "LIST"

    return "OK"

def safe_assign(local_populations, i, j, value):
    if isinstance(value, list):
        print("\n💥 INVALID ASSIGNMENT DETECTED at", i, j)
        inspect(value, "ASSIGNED_VALUE")
        raise ValueError("Attempt to inject list instead of Individual_res")

    local_populations[i][j] = value
    
class StructureTracer:
    def check(self, obj, path="root"):
        if hasattr(obj, "production_set"):
            return

        if isinstance(obj, list):
            for i, x in enumerate(obj):
                self.check(x, f"{path}[{i}]")
            return

        print("\n💥 STRUCTURE CORRUPTION DETECTED")
        print("Path:", path)
        traceback.print_stack(limit=15)

        raise TypeError(f"Invalid structure at {path}: expected Individual_res")

    

def run_ERMESS_research(Context, nb_ere, n_core, node_id, n_nodes):

    MIGRATION_TOP_RATE = 0.05    
    MIGRATION_RANDOM_RATE = 0.05    
    INTERVAL_EXCHANGE = 4
    len_pop = Context.hyperparameters.n_pop
    Initial_populations = []
    population_file = f"population_node_{node_id}.pkl"
    with open(population_file, "rb") as infile:
        Initial_populations = pickle.load(infile)  

    print('debut ', type(Initial_populations),len(Initial_populations),type(Initial_populations[0]),len(Initial_populations[0]))
    
    for ere in range(nb_ere):
        TRACER = StructureTracer()
        print(ere)

        # -----------------------
        # 1. Local optimization
        # -----------------------

        args_evolutionnary_algorithm = [(Context,Initial_populations[i]) for i in range(n_core)]  
        local_populations = ppGA.ere_evolutive_research_PARALLEL(args_evolutionnary_algorithm)
        print('after ere ', type(local_populations),len(local_populations),type(local_populations[0]),len(local_populations[0]))

        

        # -----------------------
        # 2. intra-node mix
        # -----------------------
        if ere % INTERVAL_EXCHANGE != 0:
            local_populations = migration_process(local_populations, len_pop, MIGRATION_TOP_RATE, MIGRATION_RANDOM_RATE, n_core)

        # -----------------------
        # 3. inter-nodes mix
        # -----------------------
        if ere % INTERVAL_EXCHANGE == 0:

            migrant_internodes = select_migrants_internodes(len_pop, MIGRATION_TOP_RATE, MIGRATION_RANDOM_RATE,n_core)
            local_migrants = [[local_populations[i][j] for j in migrant_internodes] for i in range(n_core)]
            for i in range(n_core) :
                inspect(local_migrants[i], "local_migrants")
            migrants = [ item for sublist in local_migrants for item in sublist ]
            for mig in migrants : 
                if type(mig)==list : 
                    print('MIG NOK',ere)
            
            write_migrants(migrants, node_id, ere)

            files = wait_for_all(ere, n_nodes)
            potential_incomers = load_migrants(files)
            print('len POT ', len(potential_incomers))
            for pot in potential_incomers : 
                if type(pot)==list : 
                    print('POT INC NOK',ere,len(pot))

            len_incomers = int((MIGRATION_TOP_RATE+MIGRATION_RANDOM_RATE)*len_pop*n_core)
            incomers = collect_migrants(potential_incomers,len_incomers)
            TRACER.check(incomers, "incomers")

            killed_indices = select_replaced_internodes(len_pop, MIGRATION_TOP_RATE, len_incomers,n_core)           
            for i, x in enumerate(incomers):
                TRACER.check(x, f"incomers[{i}]")
            local_populations = replace_population_internodes (n_core,len_pop,local_populations,incomers,killed_indices,MIGRATION_TOP_RATE,MIGRATION_RANDOM_RATE)
            TRACER.check(local_populations, "local_populations")

        
        Initial_populations = local_populations
    
    node_population = [ item for sublist in local_populations for item in sublist ]
    best_score = min(ind.fitness for ind in node_population)
    print('best score : ',best_score)
    Eri.write_node_population(node_id,node_population)
        
#    print('Fin de l\'algorithme évolutif ',time.time())
#    world_population3 = [item for sublist in local_populations_1 for item in sublist[0]]
#    scores3 = tuple(world_population3[i].fitness for i in range(len(world_population3)))
#    shuffled_population = sorted(world_population3, key=lambda x: np.random.rand())
        
#    for file_number in range(n_nodes):
#        os.remove("pop_"+str(num_node)+'_for_'+str(file_number)+".dat")
#        with open("pop_"+str(num_node)+'_for_'+str(file_number)+".dat", "wb") as f:
#                pickle.dump(shuffled_population[int(file_number*n_pop*n_core/n_nodes):int((file_number+1)*n_pop*n_core/n_nodes)], f)    
    
#    try:
#        os.remove("operators_"+str(num_node)+".dat")
#    except OSError:
#        pass
    
#    with open("operators_"+str(num_node)+".dat", "wb") as f:
#        pickle.dump(operators, f)

#    t6=time.time()
#    print('fin ',t6-t5)    
#    print('temps total : ',t6-t1)      


if __name__ == '__main__':
    ERMESS_research(node_id = sys.argv[1] , input_file_path = sys.argv[2],initialisation = sys.argv[3])

