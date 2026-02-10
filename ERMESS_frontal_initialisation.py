# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 16:16:58 2024

@author: jlegalla
"""

import sys
import numpy as np
import pandas as pd
import ERMESS_classes as ECl
import ERMESS_functions_2 as fGA2
import ERMESS_functions as fGA
import ERMESS_functions_pro as Efp
import ERMESS_GA as GA
import ERMESS_cost_functions as Cfc
import warnings
import ERMESS_parallel_processing as ppGA
from multiprocessing import freeze_support,set_start_method
import pickle
import time

from tkinter import filedialog


warnings.simplefilter(action='ignore', category=UserWarning)

if __name__ == '__main__':
    
    freeze_support()
    ##### A utiliser seulement sur le calculateur UNIX
##    set_start_method('spawn')

    start=time.time()
   # file_name = 'inputs_GEMS_frontal.xlsx' # shows dialog box and return the path
    file_name = filedialog.askopenfilename(filetypes =[('Excel Files', '*.xlsx')], initialdir="#Specify the file path")
    #file_name = '..\campus\ERMESS_pro\On_grid\SS70_DSM_EV+5%\inputs_ERMESS_pro_On_grid_SS70_noselling_fulloptim.xlsx'

    xl_file = pd.ExcelFile(file_name)
    Data = {sheet_name:xl_file.parse(sheet_name) for sheet_name in xl_file.sheet_names}
    xl_file.close()
    (datetime_data,Grid_Fossil_fuel_ratio,Main_grid_emissions,Main_grid_PoF_ratio,Connexion,specs_names,specs_Id,specs_num,prod_C,prods_U,Volums_prod,Y_movable_load,D_movable_load,Non_movable_load,time_resolution,constraint_num, Constraint_level,criterion_num,storage_techs,storage_characteristics_names,storage_characteristics,Bounds_prod,n_UP,duration_years,prices_hour_type,prices_num,fixed_premium,Overrun,Selling_price,Contract_Id,hyperparameters_main,hyperparameters_operators_names,hyperparameters_operators_num,hyperparameters_main_pro,hyperparameters_operators_names_pro,hyperparameters_operators_num_pro,Cost_sequence,type_optim,Dispatching,DG_fuel_cost,DG_lifetime,DG_unit_cost,DG_maintenance_cost,DG_fuel_consumption,DG_EROI,fuel_CO2eq_emissions,groups)=fGA2.read_data(Data)

    hyperparameters_operators_init = fGA2.adaptation_hyperparameters_initialisation(hyperparameters_operators_num,hyperparameters_operators_names)
   # hyperparameters_operators_init = hyperparameters_operators
    
    n_contracts = np.int64(len(Selling_price))
    n_store = np.int64(len(storage_techs))
            
    #Définition des variables de l'algorithme génétique
    n_steps=prod_C.size
    n_core,n_nodes,r_cross_init,n_pop,n_iter_pre = hyperparameters_main['Initialisation'] 
    
    cost_constraint=Cost_sequence[4]
    n_bits = n_steps 
    Defined_items=Dispatching[0]
      
    Contexte=ECl.Non_JIT_Environnement(storage_characteristics, time_resolution, n_store, duration_years, specs_num, groups, n_bits, prices_num, fixed_premium, Overrun, Selling_price, Non_movable_load,Y_movable_load,D_movable_load, Main_grid_emissions, Grid_Fossil_fuel_ratio,Main_grid_PoF_ratio, prod_C, prods_U, Volums_prod,Bounds_prod,DG_fuel_cost,DG_lifetime,DG_unit_cost,DG_maintenance_cost,DG_fuel_consumption,DG_EROI,fuel_CO2eq_emissions, constraint_num, Constraint_level, criterion_num, cost_constraint,r_cross_init,n_iter_pre,hyperparameters_operators_init,type_optim,Connexion,Defined_items,tracking_ope=1)
        
    n_days=n_bits/time_resolution/24
   
    if (Contexte.type_optim=='pro'):
        print('Optimisation PRO')
        r_cross,n_pop_pro,n_iter_pro = hyperparameters_main_pro
        Init_solutions2 = Efp.initial_population_pro(n_bits, n_pop_pro,n_store,time_resolution,Bounds_prod,groups,Non_movable_load,prod_C,prods_U,storage_characteristics,constraint_num,Constraint_level,n_contracts,Dispatching,Contexte.specs_num,Contexte.type_optim)           
        cost_constraint = 100000
        Contexte=ECl.Non_JIT_Environnement(storage_characteristics, time_resolution, n_store, duration_years, specs_num, groups, n_bits, prices_num, fixed_premium, Overrun, Selling_price, Non_movable_load,Y_movable_load,D_movable_load, Main_grid_emissions, Grid_Fossil_fuel_ratio,Main_grid_PoF_ratio, prod_C, prods_U, Volums_prod,Bounds_prod,DG_fuel_cost,DG_lifetime,DG_unit_cost,DG_maintenance_cost,DG_fuel_consumption,DG_EROI, fuel_CO2eq_emissions,constraint_num, Constraint_level, criterion_num, cost_constraint,r_cross,n_iter_pro,hyperparameters_operators_num_pro,type_optim,Connexion,Defined_items,tracking_ope=1)    
        inputs = (Contexte,Init_solutions2,'JIT') 
        (Final_population,perf_ope) = GA.genetic_algorithm_pro(inputs) 
        scores=tuple(ind.fitness for ind in Final_population)
        best=Final_population[np.nanargmin(scores)]
        file_name_out='output_ERMESS_pro.xlsx'
        fGA2.post_traitement(best,datetime_data,Cfc.KPI_pro, Cfc.cost_base,D_movable_load,Y_movable_load, storage_techs,specs_Id,Contract_Id,n_days,file_name_out,Contexte)

    
    elif (Contexte.type_optim=='research'):
        print('Optimisation RESEARCH')
        #Répartition de l'importance relative des 2 processus d'initialisation. 0: pro, 2 : discrétisation.
        nb_populations = (int(n_pop/2),int(n_pop/2)) if (n_pop%4==0) else (int((n_pop/2)-1),int((n_pop/2)+1)) 
        
        #première étape : on résoud le problème avec l'algorithme "pro"
        
        unlist_pro_initial_populations =  Efp.initial_population_pro(n_bits, n_core*nb_populations[0],n_store,time_resolution,Bounds_prod,groups,Non_movable_load,prod_C,prods_U,storage_characteristics,constraint_num,Constraint_level,n_contracts,Dispatching,Contexte.specs_num,Contexte.type_optim)
        pro_initial_populations = [unlist_pro_initial_populations[(i*nb_populations[0]):((i+1)*nb_populations[0])] for i in range(n_core)]
        cost_constraint = 100000
        r_cross_pro,n_iter_pro = hyperparameters_main['Evolution'][2],hyperparameters_main['Evolution'][4]
        Contexte_pro=ECl.Non_JIT_Environnement(storage_characteristics, time_resolution, n_store, duration_years, specs_num, groups, n_bits, prices_num, fixed_premium, Overrun, Selling_price, Non_movable_load,Y_movable_load,D_movable_load, Main_grid_emissions, Grid_Fossil_fuel_ratio,Main_grid_PoF_ratio, prod_C, prods_U, Volums_prod,Bounds_prod,DG_fuel_cost,DG_lifetime,DG_unit_cost,DG_maintenance_cost,DG_fuel_consumption,DG_EROI, fuel_CO2eq_emissions,constraint_num, Constraint_level, criterion_num, cost_constraint,r_cross_pro,n_iter_pro,hyperparameters_operators_num_pro,'pro',Connexion,Defined_items,tracking_ope=0)    
        pro_args = [(Contexte_pro,pro_initial_populations[i],'NON_JIT') for i in range(n_core)]
        local_pro_initial_solutions= ppGA.ere_evolutive_pro(pro_args)
        
        pro_initial_solutions = [item for sublist in local_pro_initial_solutions for item in sublist]

        #Deuxième étape : on résoud le problème à une maille journalière !!!
    
        daily_non_movable_load = np.array([sum(Non_movable_load[int(i*24*time_resolution):int((i+1)*24*time_resolution) ])/time_resolution for i in range(len(Non_movable_load)//int(time_resolution*24))])    
        DY_movable_load = np.array([sum(Y_movable_load[int(i*24*time_resolution):int((i+1)*24*time_resolution) ])/time_resolution for i in range(len(Y_movable_load)//int(time_resolution*24))])
        DD_movable_load = np.array([sum(D_movable_load[int(i*24*time_resolution):int((i+1)*24*time_resolution) ])/time_resolution for i in range(len(D_movable_load)//int(time_resolution*24))])
        daily_prods_U = np.array([tuple([sum(prods_U_j[int(i*24*time_resolution):int((i+1)*24*time_resolution) ])/time_resolution for i in range(len(prods_U_j)//int(time_resolution*24))]) for prods_U_j in prods_U]  )
        daily_volumes = np.sum(daily_prods_U,axis=1)
        daily_prod_C = np.array([sum(prod_C[int(i*24*time_resolution):int((i+1)*24*time_resolution) ])/time_resolution for i in range(len(prod_C)//int(time_resolution*24))])  
        daily_grid_prices = np.array([[np.sum(prices_num[j][int(i*24*time_resolution):int((i+1)*24*time_resolution)])/time_resolution/24 for i in range(len(Non_movable_load)//int(time_resolution*24))] for j in range(len(prices_num)) ] ) 
        daily_selling_price = np.array([[np.sum(Selling_price[j][int(i*24*time_resolution):int((i+1)*24*time_resolution)])/time_resolution/24 for i in range(len(Non_movable_load)//int(time_resolution*24))] for j in range(len(Selling_price)) ] ) 

        pre_pop=[]
        D_n_bits=np.int64(len(DY_movable_load))
        D_time_resolution=np.float64(1/24)
        
        possible_constraint_levels=fGA2.find_constraint_levels(constraint_num,Constraint_level,daily_non_movable_load,DY_movable_load,storage_characteristics,daily_prod_C,daily_prods_U,Bounds_prod)
        daily_constraint_levels = np.random.uniform(min(0.6,possible_constraint_levels[2]),possible_constraint_levels[2],n_core)
    
        D_sum_load=sum(DY_movable_load)+sum(daily_non_movable_load)
        D_args_pop_init = [[D_n_bits,  nb_populations[1],n_store,D_time_resolution,Bounds_prod,groups,D_sum_load,DY_movable_load,DD_movable_load,storage_characteristics,constraint_num,daily_constraint_levels[i],n_contracts] for i in range(n_core)]
        D_Initial_populations = ppGA.create_initial_population(D_args_pop_init)   
        D_contextes = [ECl.Non_JIT_Environnement(storage_characteristics, D_time_resolution, n_store, duration_years, specs_num,groups,D_n_bits, daily_grid_prices, fixed_premium, Overrun, daily_selling_price, daily_non_movable_load,DY_movable_load,DD_movable_load, Main_grid_emissions, Grid_Fossil_fuel_ratio, Main_grid_PoF_ratio,daily_prod_C, daily_prods_U,daily_volumes, Bounds_prod, DG_fuel_cost,DG_lifetime,DG_unit_cost,DG_maintenance_cost,DG_fuel_consumption,DG_EROI,fuel_CO2eq_emissions, constraint_num,daily_constraint_levels[i], criterion_num,cost_constraint,r_cross_init,n_iter_pre,hyperparameters_operators_init,type_optim,Connexion,Defined_items,tracking_ope=0) for i in range(n_core)]

        D_args1_2 = [(D_contextes[i],D_Initial_populations[i]) for i in range(n_core)]
    
        t0 = time.time()
        print('Début de l\'algorithme évolutif de pré-traitement',t0-start)
             
        D_local_populations_1=ppGA.ere_evolutive(D_args1_2,0,1)
            
                            
        D_final_populations = [item for sublist in D_local_populations_1 for item in sublist]
        D_final_populations_random = sorted(D_final_populations, key=lambda x: np.random.rand())
        
        print('fin du travail à la maille journalière')
        print('Etude sur des journées type')

        #Troisième étape : Etape détaillée sur certaines journées
        H_world_population4=[]
        days=np.random.choice(int(n_days),5*n_core,replace=False)
    
        H_contextes=[]
        H_args_pop_init = []
        hourly_constraint_levels = np.float64(np.random.uniform(Contexte.constraint_level**2/2,1,n_core))
    
        k=0
        for day in days :
            hourly_non_movable_load = np.array(Non_movable_load[(day*int(24*time_resolution)):((day+1)*int(24*time_resolution))])
            HY_movable_load = np.array(Y_movable_load[(day*int(24*time_resolution)):((day+1)*int(24*time_resolution))])
            HD_movable_load = np.array(D_movable_load[(day*int(24*time_resolution)):((day+1)*int(24*time_resolution))])
            hourly_prods_U = np.array([prods_U[i][(day*int(24*time_resolution)):((day+1)*int(24*time_resolution))] for i in range(len(prods_U))] )
            hourly_volumes = np.sum(hourly_prods_U,axis=1)
            hourly_prod_C = np.array(prod_C[(day*int(24*time_resolution)):((day+1)*int(24*time_resolution))])
            hourly_grid_prices = np.array([prices_num[j][(day*int(24*time_resolution)):((day+1)*int(24*time_resolution))] for j in range(len(prices_num)) ]  )
            hourly_selling_price = np.array([Selling_price[j][(day*int(24*time_resolution)):((day+1)*int(24*time_resolution))] for j in range(len(Selling_price)) ]  )


            pre_pop=[]
            H_n_bits=np.int64(len(hourly_non_movable_load))
            H_time_resolution=np.float64(time_resolution)

            H_sum_load=np.sum(HD_movable_load)+sum(hourly_non_movable_load)
            H_duration_years=np.float64(1/365)
        
    
            H_contextes.append(ECl.Non_JIT_Environnement(storage_characteristics, H_time_resolution, n_store, H_duration_years, specs_num,groups,H_n_bits, hourly_grid_prices, fixed_premium, Overrun, hourly_selling_price, hourly_non_movable_load,HY_movable_load,HD_movable_load, Main_grid_emissions, Grid_Fossil_fuel_ratio,Main_grid_PoF_ratio, hourly_prod_C, hourly_prods_U,hourly_volumes, Bounds_prod,DG_fuel_cost,DG_lifetime,DG_unit_cost,DG_maintenance_cost,DG_fuel_consumption,DG_EROI,fuel_CO2eq_emissions, constraint_num,hourly_constraint_levels[k%n_core], criterion_num,cost_constraint,r_cross_init,np.int64(n_iter_pre/2),hyperparameters_operators_init,type_optim,Connexion,Defined_items,tracking_ope=0))
            H_args_pop_init.append([H_n_bits,  nb_populations[1],n_store,H_time_resolution,Bounds_prod,groups,H_sum_load,HY_movable_load,HD_movable_load,storage_characteristics,constraint_num,hourly_constraint_levels[k%n_core],n_contracts])
            k+=1

        for j in range(5):
            H_Initial_populations = ppGA.create_initial_population(H_args_pop_init[(j*n_core):((j+1)*n_core)])   
            H_args1_2 = [(H_contextes[(j*n_core)+i],H_Initial_populations[i]) for i in range(n_core)]
            H_local_populations_1=ppGA.ere_evolutive(H_args1_2, 0,1)
                
            H_world_population4.append(H_local_populations_1)
    
        print('fin du travail sur des journées type')
                
        Init_solutions = []
        for i in range(n_core) :
            Init_solutions.append( fGA.combining_HD_solutions2 (D_final_populations_random[(i* nb_populations[1]) : ((i+1)* nb_populations[1])],[H_world_population4[j][i] for j in range(4)],[days[(j*n_core)+i] for j in range(4)],D_time_resolution,H_time_resolution,n_days,Contexte))
    
        Init_solutions2 = [item for sublist in Init_solutions for item in sublist]
        
        Init_solutions_final = Init_solutions2 + pro_initial_solutions
        shuffled_population = sorted(Init_solutions_final, key=lambda x: np.random.rand())
        args_fitness = [[shuffled_population[(i*n_pop):((i+1)*n_pop)],Contexte] for i in range(n_core)]
        fitnesses =ppGA.parallel_fitness(args_fitness)
        fitnesses = [item for sublist in fitnesses for item in sublist]
            
        for i in range(len(shuffled_population)) :
            shuffled_population[i].fitness=fitnesses[i]
      
        print('Score du meilleur individu : ',min(tuple(shuffled_population[i].fitness for i in range(len(shuffled_population)))))
        target_file = sys.argv[1]
        for j in range(n_nodes):
            with open(target_file+'_for_'+str(j)+'.dat', "wb") as f:
                pickle.dump(shuffled_population[int(j*n_pop*n_core/n_nodes):int((j+1)*n_pop*n_core/n_nodes)], f)
                
                
    else :
        print('Type d\'optimisation incorrect')
        