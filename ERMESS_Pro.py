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

from ERMESS_scripts.reporting import post_processing as Epp

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
   
    if (Context.optimization.type_optim=='pro'):
        logging.info("Starting optimization ERMESS pro")
        Initial_population_pro = Efp.initial_population_pro(Context)           
        args_pro = (Context,Initial_population_pro)
        (Final_population,perf_ope) = EA.evolutionnary_algorithm_pro(args_pro) 
        scores=tuple(ind.fitness for ind in Final_population)
        best=Final_population[np.nanargmin(scores)]
        Epp.post_traitement(solution=best, Context=Context, datetime=structured_data.time.datetime)    
        
if __name__ == '__main__':
    ERMESS_pro(sys.argv[1])
        