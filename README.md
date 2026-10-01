<a href='https://twinsolar.eu/'><p align="center"><img src="https://twinsolar.eu/wp-content/uploads/2023/03/logo_twinsolar_seul.png" width="200"></p></a>
<p align="center"><img src="https://twinsolar.eu/wp-content/uploads/2023/03/EN_FundedbytheEU_RGB_POS.png" width="200"></p>

# <b>ERMESS</b> (EvolutionnaRy Microgrid Energy Systems Sizing)

<a href='https://twinsolar.eu/'>https://twinsolar.eu/</a>

This repository contains the source of ERMESS, a Python optimization tool derived from evolutionnary algorithms, with the goal of finding the optimal design of microgrid energy systems.

Main contributor: Josselin Le Gal La Salle

## Documentation

Complete documentation available here :

[![Documentation](https://readthedocs.org/projects/ermess/badge/?version=latest)](https://ermess.readthedocs.io/en/latest/)

## DeepWiki Documentation

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/Laboratoire-Piment/ERMESS)

## DOI

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21947535.svg)](https://doi.org/10.5281/zenodo.21947535)

##How to use ERMESS ?

No need to install anything !

###Windows
1. Go to "Releases"
2. Download ERMESS_Pro.exe
3. Open the file and click on the ERMESS.exe file
4. Follow the instructions

## Scripts

ERMESS_Pro.py : runs the PRO optimization mode (Basic Usage/Examples/Experiments).

ERMESS_Research.py : runs the RESEARCH optimization mode (Large-scale Experiments).

## Package

cost : Cost model and objective functions

energy_management_model : dispatching simulation package

energy_production_model : package for computation of REN production systems

evolutionnary_core : optimization model based on genetic algorithm

load_model : package for computation of loads

reporting : functions used for output, reporting and display

##  files

###INPUTS : 

Excel file named "inputs_ERMESS_Pro.xlsx". 

Data needed in this file : 

Constraint, Constraint level, Optimisation criterion, installable production units characteristics (Capital unit cost, operational unit cost, Lifetime, Capacity, eqCO2 emissions, EROI), installable storage technologies

Timeseries : Current production (if applicable), Critic load, Daily movable load, Yearly movable load, production unit

If applicable : Main grid emissions, Main grid fossil fuel ratio, Main grid ratio primary over final energy, available trading contracts with detailed prices

Please follow the example file given in this folder


###OUTPUT : 

Excel file named "output_ERMESS_end.xlsx". 

## Related articles

https://www.techniques-ingenieur.fr/base-documentaire/innovation-th10/innovations-en-energie-et-environnement-42503210/concevoir-et-dimensionner-des-microreseaux-autonomes-l-exemple-de-twinsolar-in199/

DOI : https://doi.org/10.51257/a-v1-in199


![poster ERMESS-1](https://github.com/user-attachments/assets/aa768874-d14e-48b4-abfc-4864b5e56f90)


