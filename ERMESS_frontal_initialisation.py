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
import ERMESS_GA as GA
import ERMESS_cost_functions as Cfc
import warnings
import ERMESS_parallel_processing as ppGA
from multiprocessing import freeze_support,set_start_method
import pickle
from numba import jit
import time

from tkinter import filedialog


warnings.simplefilter(action='ignore', category=UserWarning)



if __name__ == '__main__':
    
    freeze_support()
    ##### A utiliser seulement sur le calculateur UNIX
    set_start_method('spawn')

    start=time.time()
    file_name = 'inputs_GEMS_frontal.xlsx' # shows dialog box and return the path

    xl_file = pd.ExcelFile(file_name)
    
    Data = {sheet_name:xl_file.parse(sheet_name) for sheet_name in xl_file.sheet_names}
    
    #(datetime,specs,prod_C,prods_U,Volums_prod,Y_movable_load,D_movable_load,Non_movable_load,time_resolution,Constraint,Constraint_level,criterion,storage_characteristics,Bounds_prod,n_UP,costs_production,duration_years,grid_prices,fixed_premium,Overrun,Selling_price,Contract_Id,hyperparameters_main,hyperparameters_operators)=fGA2.read_data(Data)
    (datetime,Grid_Fossil_fuel_ratio,Main_grid_emissions,Main_grid_ratio,specs_names,specs_Id,specs_num,prod_C,prods_U,Volums_prod,Y_movable_load,D_movable_load,Non_movable_load,time_resolution,constraint_num, Constraint_level,criterion_num,storage_techs,storage_characteristics_names,storage_characteristics,Bounds_prod,n_UP,duration_years,prices_hour_type,prices_num,fixed_premium,Overrun,Selling_price,Contract_Id,hyperparameters_main,hyperparameters_operators_names,hyperparameters_operators_num,Cost_sequence)=fGA2.read_data(Data)

    hyperparameters_operators_init = fGA2.adaptation_hyperparameters_initialisation(hyperparameters_operators_num,hyperparameters_operators_names)
   # hyperparameters_operators_init = hyperparameters_operators
    
    n_contracts = np.int64(len(Selling_price))
    n_store = np.int64(len(storage_techs))
            
    #Définition des variables de l'algorithme génétique
    n_steps=prod_C.size
    n_core,n_nodes,r_cross_init,n_pop,n_iter_pre = hyperparameters_main['Initialisation']
    
    cost_constraint=Cost_sequence[4]
    n_bits = n_steps 
      
    Contexte=ECl.Non_JIT_Environnement(storage_characteristics, time_resolution, n_store, duration_years, specs_num, n_bits, prices_num, fixed_premium, Overrun, Selling_price, Non_movable_load,Y_movable_load,D_movable_load, Main_grid_emissions, prod_C, prods_U, Volums_prod,Bounds_prod, constraint_num, Constraint_level, criterion_num, cost_constraint,r_cross_init,n_iter_pre,hyperparameters_operators_init,tracking_ope=1)
    
        
    n_days=n_bits/time_resolution/24

    #première étape : on résoud le problème à une maille journalière !!!
    
    daily_non_movable_load = np.array([sum(Non_movable_load[int(i*24*time_resolution):int((i+1)*24*time_resolution) ])/time_resolution for i in range(len(Non_movable_load)//int(time_resolution*24))])    
    DY_movable_load = np.array([sum(Y_movable_load[int(i*24*time_resolution):int((i+1)*24*time_resolution) ])/time_resolution for i in range(len(Y_movable_load)//int(time_resolution*24))])
    DD_movable_load = np.array([sum(D_movable_load[int(i*24*time_resolution):int((i+1)*24*time_resolution) ])/time_resolution for i in range(len(D_movable_load)//int(time_resolution*24))])
    daily_prods_U = np.array([tuple([sum(prods_U_j[int(i*24*time_resolution):int((i+1)*24*time_resolution) ])/time_resolution for i in range(len(prods_U_j)//int(time_resolution*24))]) for prods_U_j in prods_U]  )
    daily_volumes = np.sum(daily_prods_U,axis=1)
    daily_prod_C = np.array([sum(prod_C[int(i*24*time_resolution):int((i+1)*24*time_resolution) ])/time_resolution for i in range(len(prod_C)//int(time_resolution*24))])  
    daily_grid_prices = np.array([[np.sum(prices_num[j][int(i*24*time_resolution):int((i+1)*24*time_resolution)])/time_resolution/24 for i in range(len(Non_movable_load)//int(time_resolution*24))] for j in range(len(prices_num)) ] ) 

    pre_pop=[]
    D_n_bits=np.int64(len(DY_movable_load))
    D_time_resolution=np.float64(1/24)
        
    possible_constraint_levels=fGA2.find_constraint_levels(constraint_num,Constraint_level,daily_non_movable_load,DY_movable_load,storage_characteristics,daily_prod_C,daily_prods_U,Bounds_prod)
    #daily_constraint_levels = np.linspace(start=possible_constraint_levels[1]+(possible_constraint_levels[2]-possible_constraint_levels[1])/3 if possible_constraint_levels[1]<possible_constraint_levels[2] else possible_constraint_levels[2], stop=possible_constraint_levels[2], num=n_core+2)[1:(n_core+1)]
    daily_constraint_levels = np.random.uniform(min(0.6,possible_constraint_levels[2]),possible_constraint_levels[2],n_core)
    
    D_sum_load=sum(DY_movable_load)+sum(daily_non_movable_load)
    D_args_pop_init = [[D_n_bits, n_pop,n_store,D_time_resolution,Bounds_prod,D_sum_load,DY_movable_load,DD_movable_load,storage_characteristics,constraint_num,daily_constraint_levels[i],n_contracts] for i in range(n_core)]
    D_Initial_populations = ppGA.create_initial_population(D_args_pop_init)   
    D_contextes = [ECl.Non_JIT_Environnement(storage_characteristics, D_time_resolution, n_store, duration_years, specs_num,D_n_bits, daily_grid_prices, fixed_premium, Overrun, Selling_price, daily_non_movable_load,DY_movable_load,DD_movable_load, Main_grid_emissions, daily_prod_C, daily_prods_U,daily_volumes, Bounds_prod, constraint_num,daily_constraint_levels[i], criterion_num,cost_constraint,r_cross_init,n_iter_pre,hyperparameters_operators_init,tracking_ope=0) for i in range(n_core)]

    D_args1_2 = [(D_contextes[i],D_Initial_populations[i]) for i in range(n_core)]
    
    t0 = time.time()
    print('Début de l\'algorithme évolutif de pré-traitement',t0-start)
    print('Etape 1/2')
        
        
    D_local_populations_1=ppGA.ere_evolutive(D_args1_2,0,1)
    
                        
#    D_final_populations = D_local_populations_1
    
    D_final_populations = [item for sublist in D_local_populations_1 for item in sublist]  
        

##    D_args_ch=[(D_local_populations_1[i],D_time_resolution,time_resolution,D_movable_load,n_bits,n_store) for i in range(n_core)]
##    print('D_args_ch created')
##    D_final_populations=ppGA.changing_scale(D_args_ch)
##    t1 = time.time()
##    print('D_final_populations created',t1-t0)
##    print('Score du meilleur individu actuel horaire avant : ',min(tuple(Contexte.fitness(D_final_populations[0][i])[0] for i in range(len(D_final_populations[0])))))
##    print('post D storage : ',D_final_populations[0][0].fitness,' ', D_final_populations[0][0].storage_sum[0],' ', D_final_populations[0][0].storage_sum[1])

##    print(sum(D_final_populations[0][9].storage_TS[0][D_final_populations[0][9].storage_TS[0]>0]),sum(D_final_populations[0][9].storage_TS[0][D_final_populations[0][9].storage_TS[0]<0])*0.93,sum(D_final_populations[0][9].storage_TS[1][D_final_populations[0][9].storage_TS[1]>0]),sum(D_final_populations[0][9].storage_TS[1][D_final_populations[0][9].storage_TS[1]<0])*0.93)
    #Optimisation du remplissage des solutions à échelle journalière 
    
##    for ere in range(2*nb_ere_pre) :
        
##        D_args_post_optim = [(Contexte,D_final_populations[i],n_iter_pre,r_cross) for i in range(n_core)]

##        D_local_populations_1=ppGA.ere_evolutive(D_args_post_optim, ere,nb_ere_pre)
##        D_world_population1 = [item for sublist in D_local_populations_1 for item in sublist]

##        D_world_mix = [D_world_population1[i] for i in np.random.choice(range(n_pop_pre*n_core),n_pop_pre*n_core,replace=False)]
    
##        D_final_populations =  [D_world_mix[(i*n_pop_pre):((i+1)*n_pop_pre)] for i in range(n_core)]



    
    #D_world_population3 = [item for sublist in D_world_population1 for item in sublist]
##    D_world_population3 = [item for sublist in D_final_populations for item in sublist]
    
##    t2=time.time()
##    print('D_world_population3 created',t2-t1)
##    print('post D storage : ',D_world_population3[0].fitness,' ', D_world_population3[0].storage_sum[0],' ', D_world_population3[0].storage_sum[1])



##    print('Score du meilleur individu actuel horaire apres : ',min(tuple(D_world_population3[i].fitness for i in range(len(D_world_population3)))))



    #Etape détaillée sur certaines journées
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


        pre_pop=[]
        H_n_bits=np.int64(len(hourly_non_movable_load))
        H_time_resolution=np.float64(time_resolution)
        
        ##H_possible_constraint_levels=fGA2.find_constraint_levels(Constraint,Constraint_level,hourly_non_movable_load,HD_movable_load,storage_characteristics,hourly_prod_C,hourly_prods_U,Bounds_prod)
        
        ##hourly_constraint_levels = np.linspace(start=H_possible_constraint_levels[1]+(H_possible_constraint_levels[2]-H_possible_constraint_levels[1])/2, stop=H_possible_constraint_levels[2], num=n_core+2)[1:(n_core+1)]        
        ##hourly_constraint_levels = hourly_constraint_levels[np.random.randint(0,len(hourly_constraint_levels))]        
        ##hourly_constraint_levels = np.random.uniform(0,H_possible_constraint_levels[2],1)

        H_sum_load=np.sum(HD_movable_load)+sum(hourly_non_movable_load)
        H_duration_years=np.float64(1/365)
        
    
        H_contextes.append(ECl.Non_JIT_Environnement(storage_characteristics, H_time_resolution, n_store, H_duration_years, specs_num,H_n_bits, hourly_grid_prices, fixed_premium, Overrun, Selling_price, hourly_non_movable_load,HY_movable_load,HD_movable_load, Main_grid_emissions, hourly_prod_C, hourly_prods_U,hourly_volumes, Bounds_prod, constraint_num,hourly_constraint_levels[k%n_core], criterion_num,cost_constraint,r_cross_init,np.int64(n_iter_pre/2),hyperparameters_operators_init,tracking_ope=0))
        H_args_pop_init.append([H_n_bits, n_pop,n_store,H_time_resolution,Bounds_prod,H_sum_load,HY_movable_load,HD_movable_load,storage_characteristics,constraint_num,hourly_constraint_levels[k%n_core],n_contracts])
        k+=1

    for j in range(5):
        H_Initial_populations = ppGA.create_initial_population(H_args_pop_init[(j*n_core):((j+1)*n_core)])   
        H_args1_2 = [(H_contextes[(j*n_core)+i],H_Initial_populations[i]) for i in range(n_core)]
            
        H_local_populations_1=ppGA.ere_evolutive(H_args1_2, 0,1)
    

 #       H_world_population3 = [item for sublist in H_local_populations_1 for item in sublist]
        H_world_population4.append(H_local_populations_1)
    
 ######   H_world_population4 = [item for sublist in H_world_population4 for item in sublist]   
    
      
##    t3=time.time()
##    print('Fin de l\'algorithme évolutif de pré-traitement',t3-t2)
    
    
##    H_mixed_population = [H_world_population4[i] for i in np.random.choice(range(n_pop_pre*n_core),n_pop_pre*n_core,replace=False)]
##    args_ex = [(H_mixed_population[int((i)*len(H_world_population4)/n_core):int((i+1)*len(H_mixed_population)/n_core)],D_movable_load,Y_movable_load,time_resolution,H_duration_years,duration_years,n_bits,n_store) for i in range(n_core)]
##    H_final_populations=ppGA.extending_scale(args_ex)
##    print('pre H storage : ',H_final_populations[0][0].fitness,' ', H_final_populations[0][0].storage_sum[0],' ', H_final_populations[0][0].storage_sum[1])

    
##    t4=time.time()
##    print('Fin de l\'algorithme évolutif de pré-traitement',t4-t3)
    print('H_final_populations created')
        
##    print('Score du meilleur individu actuel horaire avant : ',min(tuple(Contexte.fitness(H_final_populations[0][i])[0] for i in range(len(H_final_populations[0])))))
    
    
    
#    couples_D = np.random.choice(len(D_final_populations),len(D_final_populations),replace=False)
#    couples_H = np.random.choice(len(H_world_population4),len(H_world_population4),replace=False)
    
    D_final_populations_random = sorted(D_final_populations, key=lambda x: np.random.rand())
    Init_solutions = []

    for i in range(n_core) :
        Init_solutions.append( fGA.combining_HD_solutions2 (D_final_populations_random[(i*n_pop) : ((i+1)*n_pop)],[H_world_population4[j][i] for j in range(4)],[days[(j*n_core)+i] for j in range(4)],D_time_resolution,H_time_resolution,n_days,Contexte))
    
    Init_solutions2 = [item for sublist in Init_solutions for item in sublist]
    
   
    args_fitness = [[Init_solutions2[(i*n_pop):((i+1)*n_pop)],Contexte] for i in range(n_core)]

    fitnesses =ppGA.parallel_fitness(args_fitness)
    fitnesses = [item for sublist in fitnesses for item in sublist]
    
    for i in range(len(Init_solutions2)) :
        Init_solutions2[i].fitness=fitnesses[i]
      
    
##    for ere in range(1*nb_ere_pre) :
##        t00=time.time()
##        H_args_post_optim = [(Contexte,H_final_populations[i],n_iter_pre,r_cross) for i in range(n_core)]
        #Optimisation du remplissage des solutions à échelle journalière 

##        H_local_populations_1=ppGA.ere_evolutive(H_args_post_optim, ere,nb_ere_pre)
##        t01=time.time()
##        print('fin evolution ere ',ere, ' ', t01-t00)
##        H_world_population_postoptim = [item for sublist in H_local_populations_1 for item in sublist]

 ##       H_world_mix = [H_world_population_postoptim[i] for i in np.random.choice(range(n_pop_pre*n_core),n_pop_pre*n_core,replace=False)]
    
##        H_final_populations =  [H_world_mix[(i*n_pop_pre):((i+1)*n_pop_pre)] for i in range(n_core)]
##        print(ere,' storage : ',H_final_populations[0][0].fitness,' ', H_final_populations[0][0].storage_sum[0],' ', H_final_populations[0][0].storage_sum[1])
##        t02=time.time()

    
##    H_final_populations = [item for sublist in H_final_populations for item in sublist]


    
    ##H_final_populations = [item for sublist in H_world_population1 for item in sublist]
##    t5=time.time()
##    print('fin post-optim H',t5-t4)
##    print('Score du meilleur individu actuel horaire apres : ',min(tuple(H_final_populations[i].fitness for i in range(len(H_final_populations)))))

    
##    sum_load=sum(Non_movable_load)+sum(D_movable_load)+sum(Y_movable_load)
##    args_pop_init = [[n_bits, max(1,int((int(n_core*n_pop)-len(H_final_populations)-len(D_world_population3))/n_core)),n_store,time_resolution,Bounds_prod,sum_load,Y_movable_load,D_movable_load,storage_characteristics,Constraint,Constraint_level,n_contracts] for i in range(n_core)]
##    Extra_Initial_populations = ppGA.create_initial_population(args_pop_init) 
##    Extra_Initial_populations = [item for sublist in Extra_Initial_populations for item in sublist]
##    t6=time.time()
##    print('Extra_Initial_populations created',t6-t5)
    
##    Ordered_populations = D_world_population3 + H_final_populations +Extra_Initial_populations
##    print('Ordered_populations created')
    #On passe au processus principal sur un pas de temps détaillé

##    shuffled_population = sorted(Ordered_populations, key=lambda x: np.random.rand())
##    t7=time.time()
##    print('shuffled_population created',t7-t6)
    print('Score du meilleur individu : ',min(tuple(Init_solutions2[i].fitness for i in range(len(Init_solutions2)))))

    target_file = sys.argv[1]
    #for j in range(n_nodes):
    for j in range(n_nodes):
        with open(target_file+'_for_'+str(j)+'.dat', "wb") as f:
           # for indiv in shuffled_population[int(j*n_pop*n_core/10):int((j+1)*n_pop*n_core/10)]:
                pickle.dump(Init_solutions2[int(j*n_pop*n_core/n_nodes):int((j+1)*n_pop*n_core/n_nodes)], f)