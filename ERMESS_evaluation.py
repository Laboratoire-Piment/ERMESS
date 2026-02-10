# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 12:00:55 2024

@author: jlegalla
"""
import sys
import pickle
import ERMESS_functions_2 as fGA2
import pandas as pd
import ERMESS_cost_functions as Cfc
import numpy as np
import ERMESS_classes as ECl


number_node=sys.argv[1]
cost_phase=int(sys.argv[2])
n_nodes = 12

final_pop=[]
for i in range(n_nodes):
    with open('pop_'+number_node+'_for_'+str(i)+'.dat', 'rb') as input_file:
            pop =pickle.load(input_file)
            final_pop.append(pop)
            
Final_population=[x for xs in final_pop for x in xs]

file_name = 'inputs_GEMS_frontal.xlsx' # shows dialog box and return the path

xl_file = pd.ExcelFile(file_name)
    
Data = {sheet_name:xl_file.parse(sheet_name) for sheet_name in xl_file.sheet_names}
(datetime_data,Grid_Fossil_fuel_ratio,Main_grid_emissions,Main_grid_PoF_ratio,Connexion,specs_names,specs_Id,specs_num,prod_C,prods_U,Volums_prod,Y_movable_load,D_movable_load,Non_movable_load,time_resolution,constraint_num, Constraint_level,criterion_num,storage_techs,storage_characteristics_names,storage_characteristics,Bounds_prod,n_UP,duration_years,prices_hour_type,prices_num,fixed_premium,Overrun,Selling_price,Contract_Id,hyperparameters_main,hyperparameters_operators_names,hyperparameters_operators_num,hyperparameters_main_pro,hyperparameters_operators_names_pro,hyperparameters_operators_num_pro,Cost_sequence,type_optim,Dispatching,DG_fuel_cost,DG_lifetime,DG_unit_cost,DG_maintenance_cost,DG_fuel_consumption,DG_EROI,fuel_CO2eq_emissions,groups)=fGA2.read_data(Data)
(Grid_Fossil_fuel_ratio,Main_grid_emissions,Main_grid_ratio) = (Data['Environment']['Main grid fossil fuel ratio'][0],Data['Environment']['Main grid emissions (gCO2/kWh)'][0],Data['Environment']['Main grid ratio primary over final energy'][0])
n_contracts = len(Selling_price)

cost_constraint=Cost_sequence[cost_phase]

n_store = np.int64(len(storage_techs))
    
    #Définition des variables de l'algorithme génétique
n_bits=prod_C.size
hyperparameters_main_evol = hyperparameters_main['Evolution']    
n_core,n_nodes,r_cross,n_pop,n_iter,nb_ere = hyperparameters_main_evol
Defined_items=Dispatching[0]   
Contexte=ECl.Non_JIT_Environnement(storage_characteristics, time_resolution, n_store, duration_years, specs_num, groups, n_bits, prices_num, fixed_premium, Overrun, Selling_price, Non_movable_load,Y_movable_load,D_movable_load, Main_grid_emissions, Grid_Fossil_fuel_ratio,Main_grid_PoF_ratio, prod_C, prods_U, Volums_prod,Bounds_prod,DG_fuel_cost,DG_lifetime,DG_unit_cost,DG_maintenance_cost,DG_fuel_consumption,DG_EROI,fuel_CO2eq_emissions, constraint_num, Constraint_level, criterion_num, cost_constraint,r_cross,n_iter,hyperparameters_operators_num,type_optim,Connexion,Defined_items,tracking_ope=1)    
fitness_functions = Cfc.find_cost_functions(Contexte)
 
n_store = np.int64(len(storage_techs))
    
#Définition des variables de l'algorithme génétique
n_steps=prod_C.size

n_bits = n_steps
n_days=n_bits/time_resolution/24
            
scores=tuple(ind.fitness for ind in Final_population)
best=Final_population[np.nanargmin(scores)]

with open("pop.dat", "wb") as f:
    pickle.dump(best, f)    


file_name_out='output_GEMS_end.xlsx'
fGA2.post_traitement(best,datetime_data,Cfc.KPI_research, Cfc.cost_base,D_movable_load,Y_movable_load, storage_techs,specs_Id,Contract_Id,n_days,file_name_out,Contexte)
