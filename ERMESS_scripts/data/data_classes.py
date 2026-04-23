# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 14:37:18 2026

@author: JoPHOBEA
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Callable


@dataclass
class PostProcessConfig:
    evaluation_function: Callable
    evaluation_base: Callable
    file_name: str

    export_type: str
    export_charts: bool = True
    include_baseline: bool = True

@dataclass
class SiteData:
    latitude: float
    longitude: float
    altitude: float
    timezone: str
    
@dataclass
class TimeData:
    n_bits: int
    n_days: int
    datetime: pd.Series
    time_resolution: float
    duration_years: float
    
@dataclass
class ProductionData:
    ids: np.ndarray
    characteristics_num: np.ndarray
    capacities: np.ndarray
    groups: np.ndarray
    current_prod: np.ndarray
    unit_prods: np.ndarray
    numbers: int
    
@dataclass
class StorageData:
    n_store: int
    techs: np.ndarray
    characteristics_num: np.ndarray
    


@dataclass
class LoadData:
    non_movable: np.ndarray
    daily_movable: np.ndarray
    yearly_movable: np.ndarray
    
@dataclass
class GridData:
    n_contracts: int
    fossil_fuel_ratio: float
    energy_ratio: float
    CO2eq_emissions: float
    prices_hour_type: np.ndarray
    prices: np.ndarray
    fixed_premium: np.ndarray
    Overrun: np.ndarray
    Selling_price: np.ndarray
    Contract_Ids: np.ndarray
    
@dataclass
class GensetData:
    fuel_cost: float
    lifetime: float
    unit_cost: float
    maintenance_cost: float
    fuel_consumption: np.ndarray
    fuel_CO2eq_emissions: float
    EROI: float
    
@dataclass
class OptimizationData:
    constraint_num: int
    constraint_level: float
    criterion_num: int
    type_optim: int

@dataclass
class HyperparametersData:
    n_iter: int 
    n_iter_init: int
    elitism_probability_init: float
    nb_ere: int
    n_pop: int
    r_cross: float
    r_cross_init: float
    n_nodes: int
    n_core: int
    cost_constraint: float
    elitism_probability: float
    operators_num: np.ndarray
    
@dataclass
class HyperparametersProData:
    r_cross: float
    n_pop: int
    n_iter: int
    cost_constraint: float
    elitism_probability: float
    operators_num: np.ndarray
    
@dataclass
class DispatchingData:
    Defined_items : np.ndarray
    Discharge_order : Optional[np.ndarray]
    Overlaps : Optional[np.ndarray]
    energy_use_coefficient : Optional[float ]
    D_DSM_minimum_levels : Optional[np.ndarray]
    Y_DSM_minimum_levels : Optional[np.ndarray]
    DG_strategy : Optional[float] 
    DG_min_runtime : Optional[int]
    DG_min_production : Optional[float ]
    
@dataclass
class ERMESSInputs:
    time: TimeData
    production: ProductionData
    storage: StorageData
    load: LoadData
    grid: Optional[GridData]
    genset: Optional[GensetData]
    optimization: OptimizationData
    hyperparameters: Optional[HyperparametersData]
    hyperparameterspro: HyperparametersProData
    dispatching: DispatchingData
    connexion: str
    postProcessConfig: PostProcessConfig
    tracking: bool

class Non_JIT_Environnement():
    """
    Class defining the environment for the optimization problem (Non-JIT version).
    This class mirrors the structure of `Environnement` but without Numba types for easier initialization.

    See `Environnement` for full attribute descriptions.
    """
    def __init__(self,storage_characteristics,time_resolution,n_store,duration_years,specs_num,groups,n_bits,prices_num,fixed_premium,Overrun,Selling_price,Non_movable_load,Y_movable_load,D_movable_load,Main_grid_emissions,Grid_Fossil_fuel_ratio,Main_grid_PoF_ratio,prod_C,prods_U,Volums_prod,Bounds_prod,DG_fuel_cost,DG_lifetime,DG_unit_cost,DG_maintenance_cost,DG_fuel_consumption,DG_EROI,fuel_CO2eq_emissions,constraint_num,constraint_level,criterion_num,cost_constraint,r_cross,n_iter,hyperparameters_operators,type_optim,Connexion,Defined_items,tracking_ope):
        self.storage_characteristics = storage_characteristics
        self.time_resolution = time_resolution
        self.n_store = n_store
        self.duration_years = duration_years
        self.specs_num = specs_num
        self.groups = groups
        self.n_bits = n_bits
        self.prices_num = prices_num
        self.fixed_premium = fixed_premium
        self.Overrun = Overrun
        self.n_contract = np.int64(len(self.Overrun))
        self.Selling_price = Selling_price
        self.Non_movable_load = Non_movable_load
        self.total_Y_Movable_load = np.sum(Y_movable_load)
        self.total_D_Movable_load =np.array([np.sum(D_movable_load[np.arange(np.int32(i*self.time_resolution*24),(np.int32((i+1)*self.time_resolution*24)))]) for i in range(0, np.int32(len(D_movable_load)/self.time_resolution/24))],dtype=np.float64)
        self.D_DSM_indexes = np.where(self.total_D_Movable_load != np.float64(0))[0]
        self.Main_grid_emissions = Main_grid_emissions
        self.Grid_Fossil_fuel_ratio=Grid_Fossil_fuel_ratio
        self.Main_grid_PoF_ratio=Main_grid_PoF_ratio
        self.prod_C = prod_C
        self.prods_U = prods_U
        self.Volums_prod = Volums_prod
        self.Bounds_prod = Bounds_prod
        self.DG_fuel_cost=DG_fuel_cost
        self.DG_lifetime=DG_lifetime
        self.DG_unit_cost=DG_unit_cost
        self.DG_maintenance_cost=DG_maintenance_cost
        self.DG_fuel_consumption=DG_fuel_consumption
        self.DG_EROI=DG_EROI
        self.fuel_CO2eq_emissions=fuel_CO2eq_emissions
        self.constraint_num = constraint_num
        self.constraint_level = constraint_level
        self.criterion_num = criterion_num
        self.cost_constraint = cost_constraint
        self.n_iter = n_iter
        self.r_cross = r_cross
        self.hyperparameters_operators = hyperparameters_operators 
        self.type_optim = type_optim    
        self.Connexion = Connexion    
        self.Defined_items = Defined_items
        self.tracking_ope = tracking_ope       

