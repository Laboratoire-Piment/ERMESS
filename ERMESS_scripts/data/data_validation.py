# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 13:00:07 2026

@author: JoPHOBEA
"""
import numpy as np

from ERMESS_scripts.data.indices import *

def _validate_hyperparameters_matrix(H, used_indices):
    """
    Validate critical entries of hyperparameters matrix.

    Args:
        H (np.ndarray): hyperparameters matrix
        used_indices (list[tuple[int,int]]): required (i,j) indices

    Raises:
        ValueError if invalid values are found
    """
    mask = np.zeros(H.shape, dtype=bool)

    for i, j in used_indices:
        mask[i, j] = True

    critical_values = H[mask]

    # --- NaN / Inf ---
    if np.isnan(critical_values).any():
        raise ValueError("NaN detected in critical hyperparameters")

    if np.isinf(critical_values).any():
        raise ValueError("Inf detected in critical hyperparameters")

    # --- Zeros dangereux (divisions) ---
    if np.any(critical_values == 0):
        raise ValueError("Zero detected in critical hyperparameters (division risk)")

    # --- Valeurs négatives problématiques ---
    if np.any(critical_values < 0):
        raise ValueError("Negative values detected in critical hyperparameters (invalid for scales)")


def _data_validation(data):
   """
    Validate raw input data before parsing into ERMESSInputs.

    This function checks the integrity, structure, and consistency of the raw
    input data (typically loaded from Excel or other sources). It ensures that
    all required sheets, columns, and values are present and valid before
    running any preprocessing or optimization.

    The validation includes:
    - Presence of required sheets
    - Presence of required columns in each sheet
    - Basic type and value checks
    - Consistency of time series
    - Logical checks (e.g., valid configuration choices)

    Args:
        data (dict[str, pandas.DataFrame]): Dictionary of input data tables,
            typically loaded from an Excel file.

    Raises:
        TypeError: If the input is not a dictionary of DataFrames.
        ValueError: If required data is missing or inconsistent.

    Returns:
        None

    """

    # =========================
    # 1. GLOBAL STRUCTURE
    # =========================
   if not isinstance(data, dict):
       raise TypeError("data must be a dictionary of pandas DataFrames")

   for key, value in data.items():
       if not isinstance(value, pd.DataFrame):
           raise TypeError(f"'{key}' is not a pandas DataFrame")

    # =========================
    # 2. REQUIRED SHEETS
    # =========================
   required_sheets = ["Environment","TimeSeries","PV_production_specs","WT_production_specs","Storages","Hyperparameters"]

   for sheet in required_sheets:
        if sheet not in data:
            raise ValueError(f"Missing required sheet: '{sheet}'")

    # =========================
    # 3. ENVIRONMENT CHECKS
    # =========================
   env = data["Environment"]

   required_env_cols = ["Constraint","Constraint level","Optimisation criterion","type","Latitude (°)","Longitude (°)","Altitude (m)","time resolution (steps/h)","Production","Meteo","Connexion","Tracking"]

   for col in required_env_cols:
        if col not in env:
            raise ValueError(f"Missing column '{col}' in Environment sheet")

    # values
   if env["time resolution (steps/h)"][0] <= 0:
        raise ValueError("time resolution must be > 0")
   
   optim_type = env["type"][0]
   if optim_type not in ["research", "pro"]:
         raise ValueError("type must be 'pro' or 'research'")

   production_mode = env["Production"][0]
   if production_mode not in ["automatic", "manual"]:
        raise ValueError("Production must be 'automatic' or 'manual'")

   if env["Meteo"][0] not in ["automatic", "manual"]:
        raise ValueError("Meteo must be 'automatic' or 'manual'")

   if env["Connexion"][0] not in ["On-grid", "Off-grid"]:
        raise ValueError("Connexion must be 'On-grid' or 'Off-grid'")

    # =========================
    # 4. TIMESERIES CHECKS
    # =========================
   ts = data["TimeSeries"]

   required_ts_cols = [ "Datetime","Non-controllable load (kW)","Current_production (W)","Daily movable load (kW)","Yearly movable load (kW)"]

   for col in required_ts_cols:
        if col not in ts:
            raise ValueError(f"Missing column '{col}' in TimeSeries")

    # datetime parsing
   try:
        dt = pd.to_datetime(ts["Datetime"], format="%d/%m/%Y %H:%M")
   except Exception:
        raise ValueError("Invalid datetime format in TimeSeries")

   if dt.isnull().any():
        raise ValueError("NaN values in TimeSeries datetime")

   if not dt.is_monotonic_increasing:
        raise ValueError("Datetime must be sorted")

    # =========================
    # 5. LOAD VALUES
    # =========================
   for col in [
        "Non-controllable load (kW)",
        "Yearly movable load (kW)",
        "Daily movable load (kW)"
   ]:
        if col in ts:
            if np.any(ts[col] < 0):
                raise ValueError(f"Negative values in {col}")

    # =========================
    # 6. PRODUCTION CONFIG
    # =========================
   production_mode = env["Production"][0]

   if production_mode == "manual":
        if "Unit_productions" not in data:
            raise ValueError("Missing 'Unit_productions' sheet for manual mode")

    # =========================
    # 7. GRID / GENSET
    # =========================
   connexion = env["Connexion"][0]

   if connexion == "On-grid":
        if "Grid_prices" not in data:
            raise ValueError("Missing Grid_prices sheet for Grid connexion")
            
        required_connexion_cols = ['Contract_Id','Fixed premium (€/kW)','Peak (c€/kWh)','Summer full hours (c€/kWh)','Summer off-peak hours (c€/kWh)','Winter full hours (c€/kWh)','Winter off-peak hours (c€/kWh)','Power overrun (€/kW)','CSPE (c€/kWh)','CTA','TVA load','TVA CTA','Octroi de mer','TGCA','Winter months','Summer months','Workday peak hours','Workday full hours','Workday off-peak hours','Weekend peak hours','Weekend full hours','Weekend off-peak hours','Selling peak hours','Selling base price (c€/kWh)','Selling peak price (c€/kWh)']
        connexion_sheet = data['Grid_prices']
        
   if connexion == "Off-grid":
        if "Diesel generator" not in data:
            raise ValueError("Missing Diesel generator sheet")
            
        required_connexion_cols = ['Parameter','Value','DG fuel consumption']
        connexion_sheet = data['Diesel generator']
            
   for col in required_connexion_cols:
         if col not in connexion_sheet:
             raise ValueError(f"Missing column '{col}' in Connexion sheet")

    # =========================
    # 8. PV / WT SPECS
    # =========================

   pv = data["PV_production_specs"]
    
   if production_mode=='manual':
        required_cols = ["Id","Capital unit cost (€)","Operational unit cost (€/yrs)", "Lifetime (years)","Capacity","eqCO2 Emissions (gCO2/kWh)","EROI","Surface group"] 
   else :     
        required_cols = ["Id","Capital unit cost (€)","Operational unit cost (€/yrs)", "Lifetime (years)","Capacity","eqCO2 Emissions (gCO2/kWh)","EROI","Module type","Mounting","Module","Inverter","Modules per string","Strings","Surface type","Surface group"]

   for col in required_cols:
            if col not in pv:
                raise ValueError(f"Missing column '{col}' in PV specs")

   if np.any(pv["Capacity"] <= 0):
            raise ValueError("PV capacity must be > 0")

   wt = data["WT_production_specs"]

   if production_mode=='manual':
             required_cols = ["Id","Capital unit cost (€)","Operational unit cost (€/yrs)", "Lifetime (years)","Capacity","eqCO2 Emissions (gCO2/kWh)","EROI","Surface group"] 
   else :     
             required_cols = ["Id","Capital unit cost (€)","Operational unit cost (€/yrs)", "Lifetime (years)","Capacity","eqCO2 Emissions (gCO2/kWh)","EROI","Model","Hub height (m)","Surface group"]

   for col in required_cols:
                 if col not in wt:
                     raise ValueError(f"Missing column '{col}' in WT specs")

   if np.any(wt["Capacity"] <= 0):
                 raise ValueError("WT capacity must be > 0")

    # =========================
    # 9. UNIT PRODUCTIONS
    # =========================
   if ("Unit_productions" in data) and (production_mode=='manual'):

        up = data["Unit_productions"]

        if "Datetime" not in up:
            raise ValueError("Missing Datetime in Unit_productions")

        for prod in pv['Id'].values:
            if prod not in up.keys() :
                raise ValueError(f"Unit production '{prod}' not provided")
        for prod in wt['Id'].values:
            if prod not in up.keys() :
                raise ValueError(f"Unit production '{prod}' not provided")
                
    # =========================
    # 9. HYPERPARAMETERS
    # =========================
   hyperparameters_operators_num = np.float64(data['Hyperparameters'][['Contract','Production','Storage volume','Storage_global','Storage_power','Storage_trades_consistency','Storage_patterns','Inter_storages','Storage_mix','Curve_smoothing','Constraint_forcing','Interdaily_consistency','DSM_trades_consistency','DSM_noise']])

   _validate_hyperparameters_matrix(hyperparameters_operators_num, used_indices)

    # =========================
    # 10. BASIC NUMERICAL CHECKS
    # =========================
   for sheet_name, df in data.items():

        numeric_df = df.select_dtypes(include=[np.number])

        if np.any(np.isinf(numeric_df)):
            raise ValueError(f"Infinite values detected in sheet '{sheet_name}'")

    #---------------------------
    #HYPERPARAMETERS
    #---------------------------
used_indices = [
        # OPER_INV_LENGTH
        (OPER_INV_LENGTH, RESEARCH_PRODUCTION),
        (OPER_INV_LENGTH, RESEARCH_STORAGE_PATTERNS),
        (OPER_INV_LENGTH, RESEARCH_STORAGE_GLOBAL),
        (OPER_INV_LENGTH, RESEARCH_STORAGE_MIX),
        (OPER_INV_LENGTH, RESEARCH_STORAGE_VOLUME),
        (OPER_INV_LENGTH, RESEARCH_STORAGE_POWER),
        (OPER_INV_LENGTH, RESEARCH_INTER_STORAGES),
        (OPER_INV_LENGTH, RESEARCH_CONSTRAINT_FORCING),
        (OPER_INV_LENGTH, RESEARCH_CURVE_SMOOTHING),
        (OPER_INV_LENGTH, RESEARCH_STORAGE_TRADES_CONSISTENCY),
        (OPER_INV_LENGTH, RESEARCH_DSM_NOISE),
        (OPER_INV_LENGTH, RESEARCH_DSM_TRADES_CONSISTENCY),

        # OPER_INV_MAGNITUDE
        (OPER_INV_MAGNITUDE, RESEARCH_STORAGE_GLOBAL),
        (OPER_INV_MAGNITUDE, RESEARCH_STORAGE_VOLUME),
        (OPER_INV_MAGNITUDE, RESEARCH_STORAGE_POWER),
        (OPER_INV_MAGNITUDE, RESEARCH_STORAGE_MIX),
        (OPER_INV_MAGNITUDE, RESEARCH_INTERDAILY_CONSISTENCY),
        (OPER_INV_MAGNITUDE, RESEARCH_CURVE_SMOOTHING),

        # OPER_MAX_AVERAGE
        (OPER_MAX_AVERAGE, RESEARCH_CONTRACT),
        (OPER_MAX_AVERAGE, RESEARCH_PRODUCTION),
        (OPER_MAX_AVERAGE, RESEARCH_STORAGE_MIX),
        (OPER_MAX_AVERAGE, RESEARCH_STORAGE_VOLUME),
        (OPER_MAX_AVERAGE, RESEARCH_CONSTRAINT_FORCING),

        # OPER_MIN_AVERAGE
        (OPER_MIN_AVERAGE, RESEARCH_CONSTRAINT_FORCING),

        # OPER_MAX_DEVIATION
        (OPER_MAX_DEVIATION, RESEARCH_PRODUCTION),
        (OPER_MAX_DEVIATION, RESEARCH_STORAGE_PATTERNS),
        (OPER_MAX_DEVIATION, RESEARCH_STORAGE_MIX),
        (OPER_MAX_DEVIATION, RESEARCH_STORAGE_VOLUME),
        (OPER_MAX_DEVIATION, RESEARCH_DSM_NOISE),
        (OPER_MAX_DEVIATION, RESEARCH_CURVE_SMOOTHING),
    ]

