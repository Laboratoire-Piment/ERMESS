# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 16:16:58 2024

@author: joPHOBEA

This script either initialises the ERMESS RESEARCH algorithm to creates initial population OR runs the ERMESS PRO algorithm

It :
1. Loads input data from an Excel configuration file
2. Builds the optimization environment
3. Runs the optimization model

The algorithm relies on a hierarchical initialization strategy
and parallel evolutionary computation.

Modules used:
    - data.data_classes
    - data.read_excel
    - utils.constraints
    - evolutionnary_core
    - cost.ERMESS_cost_functions
    - reporting.write_excel

"""

import sys
import numpy as np
from multiprocessing import freeze_support
import warnings
from tkinter import filedialog
import logging

from ERMESS_scripts.data import data_reader as Edr
from ERMESS_scripts.data import data_parsers as Edp
from ERMESS_scripts.data import data_builder as Dbl
from ERMESS_scripts.data import data_validation as Edv

from ERMESS_scripts.evolutionnary_core import ERMESS_functions_pro as Efp
from ERMESS_scripts.evolutionnary_core import ERMESS_evolutionnary_algorithm as EA
from ERMESS_scripts.reporting import write_excel as Wex

warnings.simplefilter(action='ignore', category=UserWarning)


def ERMESS_pro(input_file_path=None) :
    
    freeze_support()  
    ##============================================================================
    # 1. INPUT DATA LOADING
    ##============================================================================
    
    if input_file_path is None:
        input_file_path = filedialog.askopenfilename(filetypes =[('Excel Files', '*.xlsx')], initialdir="#Specify the file path")

    data = Edr.load_excel(input_file_path)
    Edv._data_validation(data)
    structured_data = Edp._parse_ERMESSInputs(data)

    ##============================================================================
    # 2. ENVIRONMENT INITIALIZATION
    ##============================================================================
          
    Context=Dbl.build_environment(structured_data)
            
    ##============================================================================
    # 3. PRO OPTIMIZATION (SIMULATING EMS DISPATCHING)
    ##============================================================================  
   
    if (structured_data.optimization.type_optim=='pro'):
        logging.info("Starting optimization ERMESS pro")
        Initial_population_pro = Efp.initial_population_pro(Context)           
 #       Context=Dcl.Non_JIT_Environnement(storage_characteristics, time_resolution, n_store, duration_years, specs_num, groups, n_bits, prices_num, fixed_premium, Overrun, Selling_price, Non_movable_load,Y_movable_load,D_movable_load, Main_grid_emissions, Grid_Fossil_fuel_ratio,Main_grid_PoF_ratio, prod_C, prods_U, Volums_prod,Bounds_prod,DG_fuel_cost,DG_lifetime,DG_unit_cost,DG_maintenance_cost,DG_fuel_consumption,DG_EROI, fuel_CO2eq_emissions,constraint_num, Constraint_level, criterion_num, cost_constraint,r_cross,n_iter_pro,hyperparameters_operators_num_pro,type_optim,Connexion,Defined_items,tracking_ope=1)    
        args_pro = (Context,Initial_population_pro)
        (Final_population,perf_ope) = EA.evolutionnary_algorithm_pro(args_pro) 
        scores=tuple(ind.fitness for ind in Final_population)
        best=Final_population[np.nanargmin(scores)]
        Wex.post_traitement(solution=best, Context=Context, datetime=structured_data.time.datetime)    
        
if __name__ == '__main__':
    ERMESS_pro(sys.argv[1])
        