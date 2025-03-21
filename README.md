<a href='https://twinsolar.eu/'><p align="center"><img src="https://twinsolar.eu/wp-content/uploads/2023/03/logo_twinsolar_seul.png" width="200"></p></a>
<p align="center"><img src="https://twinsolar.eu/wp-content/uploads/2023/03/EN_FundedbytheEU_RGB_POS.png" width="200"></p>

# <b>ERMESS</b> (EvolutionnaRy Microgrid Energy Systems Sizing)

<a href='https://twinsolar.eu/'>https://twinsolar.eu/</a>

This repository contains the source of ERMESS, which is Python code to design and size microgrids.

Main contributor: Josselin Le Gal La Salle


## Scripts
#ERMESS_GA.py : contains the core evolutionnary algorithm used for evolution
#ERMESS_classes.py : defines the classes used in ERMESS
#ERMESS_cost_functions.py : proposes some objective functions for common microgrids performance indicators
#ERMESS_evaluation.py : post-processing of the final solution
#ERMESS_evaluation_operators.py : functions for developers - functions for the assessment of the influence of algorithm hyperparameters
#ERMESS_frontal_evolution.py : Retrieve populations of solutions and run an era of evolution
#ERMESS_frontal_initialisation.py : Run pre-optimization and creates an initial population
#ERMESS_functions.py : functions used in the GA algorithm
#ERMESS_functions_2.py : functions used in pre-processing and post-processing scripts
#ERMESS_parallel_processing.py : manage the parallelisation of the algorithm

##  files

#input : 
Excel file named : "inputs_GEMS_frontal.xlsx". 

Data needed in this file : 
Constraint, Constraint level, Optimisation criterion, installable production units characteristics (Capital unit cost, operational unit cost, Lifetime, Capacity, eqCO2 emissions, EROI), installable storage technologies
Timeseries : Current production (if applicable), Critic load, Daily movable load, Yearly movable load, production unit
If applicable : Main grid emissions, Main grid fossil fuel ratio, Main grid ratio primary over final energy, available trading contracts with detailed prices


#output : 
Excel file named : "output_GEMS_end.xlsx". 

## Related articles

https://www.techniques-ingenieur.fr/base-documentaire/innovation-th10/innovations-en-energie-et-environnement-42503210/concevoir-et-dimensionner-des-microreseaux-autonomes-l-exemple-de-twinsolar-in199/
DOI : https://doi.org/10.51257/a-v1-in199


