# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 12:30:54 2026



Numba-safe indices for ERMESS project.

This module provides integer constants derived from IntEnum definitions
to ensure compatibility with Numba (@jit).

IMPORTANT:
- Only use THESE constants inside Numba functions.
- Do NOT use Enum or module attribute access inside @njit.

@author: JoPHOBEA
"""

# =========================================================
# ENUM DEFINITIONS (Python-side only)
# =========================================================

class DefinedItemsIdx:
    Storage = 0
    DSM = 1
    Genset = 2

class ProdCharIdx:
    Capex = 0
    Opex = 1
    Lifetime = 2
    Emissions = 3
    EROI = 4
    Volume = 5
    
class StorCharIdx:
    Energy_cost = 0
    PCS_Cost = 1
    BOP_Cost = 2
    OM_Cost = 3
    Round_trip_efficiency = 4
    Depth_of_discharge = 5
    Emissions = 6
    Lifetime = 7
    Cycle_life = 8
    Installation_cost = 9
    ESOEI = 10
    Power_Cost = 11
    
class OperatorsOpeIdX:
    Contract = 0
    Production = 1
    Storage_volume = 2
    Storage_global = 3
    Storage_power = 4
    Storage_trades_consistency = 5
    Storage_patterns = 6
    Inter_storages = 7
    Storage_mix = 8
    Curve_smoothing = 9
    Constraint_forcing = 10
    Interdaily_consistency = 11
    DSM_trades_consistency = 12
    DSM_noise = 13
    
class OperatorsCharIdx:
    Probability = 0
    Inv_length = 1
    Min_average = 2
    Max_average = 3
    Min_deviation = 4
    Max_deviation = 5
    Inv_magnitude = 6
    
class OperatorsProOpeIdx:
    Contract = 0
    Production = 1
    Strategy = 2
    Discharge_order = 3
    Energy_use = 4
    Overlap = 5
    DSM_levels = 6
    DG_control = 7
    Storage_capacities = 8
    Storage_powers = 9
    Initial_SOC = 10

class OperatorsProCharIdx:
    Probability = 0
    Min_deviation = 1   
    
class OverlapsCharIdx:
    Intern = 0
    Extern = 1   
    
class Indiv_Pro_StoreCharIdx:
    Volume = 0
    Charge_power = 1
    Discharge_power = 2
    SOC_init = 3
    
class CriterionIdx:
    LCOE = 1
    Annual_net_benefits = 2
    NPV = 3
    Self_sufficiency = 4
    Self_consumption = 5
    Autonomy = 6
    EqCO2emissions = 7
    Fossil_fuel_consumption = 8
    EROI = 9
    Energy_losses = 10
    Max_power_from_grid = 11
    
class ConstraintIdx:
    Self_sufficiency = 0
    Self_consumption = 1
    REN_fraction = 2
    
# =========================================================
# CONNEXION MODE
# =========================================================
GRID_ON = 0
GRID_OFF = 1    

# =========================================================
# DEFINED ITEMS
# =========================================================

DEFINED_STORAGE = int(DefinedItemsIdx.Storage)
DEFINED_DSM = int(DefinedItemsIdx.DSM)
DEFINED_GENSET = int(DefinedItemsIdx.Genset)
    
# =========================================================
# PRODUCTION CHARACTERISTICS
# =========================================================

PROD_CAPEX = int(ProdCharIdx.Capex)
PROD_OPEX = int(ProdCharIdx.Opex)
PROD_LIFETIME = int(ProdCharIdx.Lifetime)
PROD_EMISSIONS = int(ProdCharIdx.Emissions)
PROD_EROI = int(ProdCharIdx.EROI)
PROD_VOLUME = int(ProdCharIdx.Volume)


# =========================================================
# STORAGE CHARACTERISTICS
# =========================================================

STOR_ENERGY_COST = int(StorCharIdx.Energy_cost)
STOR_PCS_COST = int(StorCharIdx.PCS_Cost)
STOR_BOP_COST = int(StorCharIdx.BOP_Cost)
STOR_OM_COST = int(StorCharIdx.OM_Cost)
STOR_ROUND_TRIP_EFF = int(StorCharIdx.Round_trip_efficiency)
STOR_DEPTH_OF_DISCHARGE = int(StorCharIdx.Depth_of_discharge)
STOR_EMISSIONS = int(StorCharIdx.Emissions)
STOR_LIFETIME = int(StorCharIdx.Lifetime)
STOR_CYCLE_LIFE = int(StorCharIdx.Cycle_life)
STOR_INSTALLATION_COST = int(StorCharIdx.Installation_cost)
STOR_ESOEI = int(StorCharIdx.ESOEI)
STOR_POWER_COST = int(StorCharIdx.Power_Cost)


# =========================================================
# RESEARCH OPERATORS
# =========================================================

RESEARCH_CONTRACT = int(OperatorsOpeIdX.Contract)
RESEARCH_PRODUCTION = int(OperatorsOpeIdX.Production)
RESEARCH_STORAGE_VOLUME = int(OperatorsOpeIdX.Storage_volume)
RESEARCH_STORAGE_GLOBAL = int(OperatorsOpeIdX.Storage_global)
RESEARCH_STORAGE_POWER = int(OperatorsOpeIdX.Storage_power)
RESEARCH_STORAGE_TRADES_CONSISTENCY = int(OperatorsOpeIdX.Storage_trades_consistency)
RESEARCH_STORAGE_PATTERNS = int(OperatorsOpeIdX.Storage_patterns)
RESEARCH_INTER_STORAGES = int(OperatorsOpeIdX.Inter_storages)
RESEARCH_STORAGE_MIX = int(OperatorsOpeIdX.Storage_mix)
RESEARCH_CURVE_SMOOTHING = int(OperatorsOpeIdX.Curve_smoothing)
RESEARCH_CONSTRAINT_FORCING = int(OperatorsOpeIdX.Constraint_forcing)
RESEARCH_INTERDAILY_CONSISTENCY = int(OperatorsOpeIdX.Interdaily_consistency)
RESEARCH_DSM_TRADES_CONSISTENCY = int(OperatorsOpeIdX.DSM_trades_consistency)
RESEARCH_DSM_NOISE = int(OperatorsOpeIdX.DSM_noise)


# =========================================================
# RESEARCH OPERATORS CHARACTERISTICS
# =========================================================

OPER_PROBABILITY = int(OperatorsCharIdx.Probability)
OPER_INV_LENGTH = int(OperatorsCharIdx.Inv_length)
OPER_MIN_AVERAGE = int(OperatorsCharIdx.Min_average)
OPER_MAX_AVERAGE = int(OperatorsCharIdx.Max_average)
OPER_MIN_DEVIATION = int(OperatorsCharIdx.Min_deviation)
OPER_MAX_DEVIATION = int(OperatorsCharIdx.Max_deviation)
OPER_INV_MAGNITUDE = int(OperatorsCharIdx.Inv_magnitude)


# =========================================================
# PRO OPERATORS
# =========================================================

PRO_CONTRACT = int(OperatorsProOpeIdx.Contract)
PRO_PRODUCTION = int(OperatorsProOpeIdx.Production)
PRO_STRATEGY = int(OperatorsProOpeIdx.Strategy)
PRO_DISCHARGE_ORDER = int(OperatorsProOpeIdx.Discharge_order)
PRO_ENERGY_COEFF = int(OperatorsProOpeIdx.Energy_use)
PRO_OVERLAP = int(OperatorsProOpeIdx.Overlap)
PRO_DSM_LEVELS = int(OperatorsProOpeIdx.DSM_levels)
PRO_DG_CONTROL = int(OperatorsProOpeIdx.DG_control)
PRO_STORAGE_CAPACITIES = int(OperatorsProOpeIdx.Storage_capacities)
PRO_STORAGE_POWERS = int(OperatorsProOpeIdx.Storage_powers)
PRO_INITIAL_SOC = int(OperatorsProOpeIdx.Initial_SOC)


# =========================================================
# PRO OPERATOR CHARACTERISTICS
# =========================================================

PRO_OPER_PROBABILITY = int(OperatorsProCharIdx.Probability)
PRO_OPER_DEVIATION = int(OperatorsProCharIdx.Min_deviation)


# =========================================================
# OVERLAPS
# =========================================================

OVERLAP_INTERN = int(OverlapsCharIdx.Intern)
OVERLAP_EXTERN = int(OverlapsCharIdx.Extern)


# =========================================================
# INDIVIDUAL STORAGE STRUCTURE (PRO)
# =========================================================

INDIV_PRO_VOLUME = int(Indiv_Pro_StoreCharIdx.Volume)
INDIV_PRO_CHARGE_POWER = int(Indiv_Pro_StoreCharIdx.Charge_power)
INDIV_PRO_DISCHARGE_POWER = int(Indiv_Pro_StoreCharIdx.Discharge_power)
INDIV_PRO_SOC_INIT = int(Indiv_Pro_StoreCharIdx.SOC_init)

# =========================================================
# CRITERION
# =========================================================

CRIT_LCOE = 1
CRIT_Annual_net_benefits = 2
CRIT_NPV = 3
CRIT_Self_sufficiency = 4
CRIT_Self_consumption = 5
CRIT_Autonomy = 6
CRIT_EqCO2emissions = 7
CRIT_Fossil_fuel_consumption = 8
CRIT_EROI = 9
CRIT_Energy_losses = 10
CRIT_Max_power_from_grid = 11
    
# =========================================================
# CONSTRAINT
# =========================================================
    

CONS_Self_sufficiency = 0
CONS_Self_consumption = 1
CONS_REN_fraction = 2
    

# =========================================================
# RANDOM FACTORS PRO
# =========================================================

RF_CONTRACT = 0
RF_PRODUCTION_MAIN = 1
RF_PRODUCTION_SWAP = 2
RF_PRODUCTION_TRANSFER = 3
RF_STRATEGY = 4
RF_STORAGE_ORDER = 5
RF_ENERGY_COEFF = 6
RF_OVERLAP = 7
RF_DDSM = 8
RF_YDSM = 9
RF_DG_RUNTIME = 10
RF_DG_PRODUCTION = 11
RF_STORAGE_CAPACITY = 12
RF_STORAGE_INPOWER = 13
RF_STORAGE_OUTPOWER = 14
RF_INIT_SOC = 15

RF_INIT_SOC_EFFECT = 19
RF_OVERLAP_EFFECT_START = 20
RF_STORAGE_EFFECT_START = 25

# =========================================================
# RANDOM FACTORS RESEARCH
# =========================================================

RES_RF_CONTRACT = 0
RES_RF_POWER_CONTRACT = 1
RES_RF_PRODUCTION_MAIN = 2
RES_RF_PRODUCTION_TRANSFER = 3
RES_RF_PRODUCTION_SWAP = 4
RES_RF_MUTATION_STORAGE = 5
RES_RF_STORAGE_WINDOWS_NOISE = 6
RES_RF_STORAGE_PATTERNS = 7
RES_RF_STORAGE_GLOBAL = 8
RES_RF_STORAGE_WINDOWS_SCALE = 9
RES_RF_STORAGE_MIX_POINTS = 10
RES_RF_STORAGE_TRANSFER = 11
RES_RF_STORAGE_VOLUME = 12
RES_RF_STORAGE_POWER = 13
RES_RF_STORAGE_OPPOSITE = 14
RES_RF_STORAGE_TRADE_CONSISTENCY = 15
RES_RF_STORAGE_INTERDAILY_SMOOTHING = 16
RES_RF_STORAGE_COPY = 17
RES_RF_STORAGE_SMOOTHING = 18
RES_RF_STORAGE_DISTRIBUTION = 19 
RES_RF_STORAGE_CONSTRAINT = 20
RES_RF_YDSM_NOISE = 21
RES_RF_YDSM_WINDOWS = 22
RES_RF_YDSM_INTERDAILY_PATTERNS = 23
RES_RF_YDSM_TRADES_CONSISTENCY = 24
RES_RF_DDSM_NOISE = 25
RES_RF_DDSM_WINDOWS = 26
RES_RF_DDSM_INTERDAILY_PATTERNS = 27
RES_RF_DDSM_TRADES_CONSISTENCY = 28

RES_RF_STORAGE_MIX_POINTS_EFFECT= 29
RES_RF_STORAGE_INTERDAILY_SMOOTHING_EFFECT = 30
RES_RF_STORAGE_COPY_EFFECT = 31
RES_RF_STORAGE_SMOOTHING_EFFECT = 32
RES_RF_STORAGE_DISTRIBUTION_EFFECT = 33
RES_RF_YDSM_WINDOWS_EFFECT = 34
RES_RF_YDSM_INTERDAILY_PATTERNS_EFFECT = 35
RES_RF_YDSM_TRADES_CONSISTENCY_EFFECT = (36,37,38)
RES_RF_DDSM_WINDOWS_EFFECT = 39
RES_RF_DDSM_INTERDAILY_PATTERNS_EFFECT = 40
RES_RF_DDSM_TRADES_CONSISTENCY_EFFECT = (41,42,43)

# =========================================================
# PRO TRACKER OPERATORS
# =========================================================

switch_contract = 0
Mutate_production_capacity_operator = 1
Switch_intragroup_productor = 2             
Transfer_production_capacity_operator = 3
Switch_dispatching_strategy = 4
Switch_storages_order = 5
Mutate_DSM_storage_distribution = 6
Mutate_EMS_Overlap = 7
Mutate_DDSM_levels = 8
Mutate_YDSM_levels = 9
Mutate_DG_min_runtime = 10
Mutate_DG_min_production = 11
Mutate_storages_capacity = 12
Mutate_storages_inpower = 13
Mutate_storages_outpower = 14
Mutate_initSOC_operator = 15

# =========================================================
# SAFETY CHECKS
# =========================================================

def _check_indices():
    """Ensure Enum values are consistent (fail fast if modified)."""

    # Production
    assert PROD_CAPEX == 0
    assert PROD_OPEX == 1
    assert PROD_LIFETIME == 2
    assert PROD_EMISSIONS == 3
    assert PROD_EROI == 4
    assert PROD_VOLUME == 5

    # Storage
    assert STOR_ENERGY_COST == 0
    assert STOR_PCS_COST == 1
    assert STOR_BOP_COST == 2
    assert STOR_OM_COST == 3
    assert STOR_ROUND_TRIP_EFF == 4
    assert STOR_DEPTH_OF_DISCHARGE == 5
    assert STOR_EMISSIONS == 6
    assert STOR_LIFETIME == 7
    assert STOR_CYCLE_LIFE == 8
    assert STOR_INSTALLATION_COST == 9
    assert STOR_ESOEI == 10
    assert STOR_POWER_COST == 11

    # Operators
    assert RESEARCH_CONTRACT == 0
    assert RESEARCH_PRODUCTION == 1

    # Pro operators
    assert PRO_CONTRACT == 0
    assert PRO_PRODUCTION == 1
    assert PRO_STRATEGY == 2
    assert PRO_DISCHARGE_ORDER == 3

    # Overlap
    assert OVERLAP_INTERN == 0
    assert OVERLAP_EXTERN == 1

    # Individual
    assert INDIV_PRO_VOLUME == 0
    assert INDIV_PRO_CHARGE_POWER == 1
    assert INDIV_PRO_DISCHARGE_POWER == 2
    assert INDIV_PRO_SOC_INIT == 3
    



# Run once at import time
_check_indices()
    