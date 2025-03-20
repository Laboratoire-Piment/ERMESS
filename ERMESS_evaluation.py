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
(datetime,Grid_Fossil_fuel_ratio,Main_grid_emissions,Main_grid_ratio,specs_names,specs_Id,specs_num,prod_C,prods_U,Volums_prod,Y_movable_load,D_movable_load,Non_movable_load,time_resolution,constraint_num, Constraint_level,criterion_num,storage_techs,storage_characteristics_names,storage_characteristics,Bounds_prod,n_UP,duration_years,prices_hour_type,prices_num,fixed_premium,Overrun,Selling_price,Contract_Id,hyperparameters_main,hyperparameters_operators_names,hyperparameters_operators_num,Cost_sequence)=fGA2.read_data(Data)
(Grid_Fossil_fuel_ratio,Main_grid_emissions,Main_grid_ratio) = (Data['Environment']['Main grid fossil fuel ratio'][0],Data['Environment']['Main grid emissions (gCO2/kWh)'][0],Data['Environment']['Main grid ratio primary over final energy'][0])
n_contracts = len(Selling_price)

cost_constraint=Cost_sequence[cost_phase]


fitness_functions = Cfc.find_cost_functions(criterion_num)
 
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
fGA2.post_traitement(best,datetime,Cfc.cost_scenario_LCOE_detail, Cfc.cost_base, prod_C, prods_U, Non_movable_load,D_movable_load,Y_movable_load, storage_characteristics, storage_techs, time_resolution, Main_grid_emissions,Grid_Fossil_fuel_ratio,Main_grid_ratio,specs_num,specs_Id,duration_years,prices_num,fixed_premium,Overrun,Selling_price,Contract_Id,Bounds_prod,constraint_num,Constraint_level,Data,n_days,file_name_out,cost_constraint)

