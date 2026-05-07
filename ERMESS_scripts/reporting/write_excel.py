# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 13:50:18 2026

@author: JoPHOBEA
"""

import pandas as pd
import numpy as np
import openpyxl

from dataclasses import dataclass

from ERMESS_scripts.data.indices import *
from ERMESS_scripts.reporting import graphs_excel as EgE
from ERMESS_scripts.reporting import charts_config as Ecc

@dataclass
class _PostProcessingResults:
    timeseries: pd.DataFrame
    technical: pd.DataFrame
    economic: pd.DataFrame
    environmental: pd.DataFrame
    flows: pd.DataFrame
    storages: pd.DataFrame
    production: pd.DataFrame
    global_dispatching: pd.DataFrame
    SOC_distribution: pd.DataFrame
    genset: pd.DataFrame
    demand_side_management: list
    time_balancing: list
    EMS: list
    
def _as_text(value):
    """
    Convert a value to a string, returning an empty string for None.
    
    This utility ensures that None values are represented as an empty string
    instead of the literal "None", which is useful for Excel or text output.
    
    Args:
        value (Any): Any value that might be written in a cell.
    
    Returns:
        str: String representation of the value, or empty string if None.
    """
    if value is None:
        return ""
    return str(value)

def _set_column_width(ws,offset=0):
    """
    Automatically adjust the width of all columns in an OpenPyXL worksheet
    based on the maximum length of the content in each column.
    
    Args:
        ws (openpyxl.worksheet.worksheet.Worksheet): Worksheet to format.
        offset (int): Extra width to add to each column.
    """
    dim_holder = openpyxl.worksheet.dimensions.DimensionHolder(worksheet=ws)
    j=0

    for col in ws.columns:
        j+=1
        length = max(len(_as_text(cell.value)) for cell in col)
        dim_holder[openpyxl.utils.get_column_letter(j)] = openpyxl.worksheet.dimensions.ColumnDimension(ws, min=j, max=j, width=length+offset)

    ws.column_dimensions = dim_holder
        
def _get_nested(data, keys):
    if isinstance(keys, str):
        return data[keys]

    for k in keys:
        data = data[k]
    return data


def _build_comparison_block(outputs_solution, output_baseline, keys):
    """
    Generic builder for Optimization vs Baseline comparison blocks.

    keys : str OR list/tuple
    """

    data_opt = _get_nested(outputs_solution, keys)
    data_base = _get_nested(output_baseline, keys)

    if isinstance(data_opt, pd.DataFrame):
        df = pd.concat([data_opt, data_base])

    else:
        df_opt = pd.DataFrame(data_opt, index=["Optimization"])
        df_base = pd.DataFrame(data_base, index=["Baseline"])
        df = pd.concat([df_opt, df_base])

    df.index = ['Optimization', 'Baseline']

    return df
  
def compute_economic_comparisons(outputs_solution, output_baseline):
    NPV = outputs_solution['economics']['Value (€)']-output_baseline['economics']['Annual net benefits (€/yrs.)']*outputs_solution['Technical']['Installation lifetime (yrs.)']
    Payback = outputs_solution['economics']['Initial investment (€)']/(outputs_solution['economics']['Annual net benefits (€/yrs.)']-output_baseline['economics']['Annual net benefits (€/yrs.)'])
    if Payback<0 : 
        Payback = np.nan
    return(NPV, Payback)

def output_build_production(solution, Context):
    return pd.DataFrame({
        'ID': Context.specs_Id,
        'Number of units': solution.production_set,
        'Coverage ratio': solution.production_set / Context.production.capacities,
        'Initial investment (€)': np.multiply(Context.production.specs_num[:, PROD_CAPEX], solution.production_set)
    })    

def _write_sheet(writer, df, name, **kwargs):
    df.to_excel(writer, sheet_name=name, **kwargs)
    
def _write_multiple_tables(writer, dfs, sheet_name, positions,index):
    """
    dfs : list of DataFrames
    positions : list of (row, col)
    """
    for df, (row, col) in zip(dfs, positions):
        df.to_excel(writer, sheet_name=sheet_name,startrow=row, startcol=col, index=index)
    
def export_to_excel(results, Context):

    with pd.ExcelWriter(Context.postprocess_config.file_name, engine='openpyxl') as writer:

        _write_sheet(writer, results.flows, "Flows")
        _write_multiple_tables(writer,results.global_dispatching,"Global dispatching",positions=[(0, 0), (3, 0),(6, 0),(9, 0),(12, 0), (15, 0),],index=True)
        _write_sheet(writer, results.economic, "Financial outputs")
        _write_sheet(writer, results.technical, "Technical")
        _write_sheet(writer, results.storages, "Storages")
        _write_sheet(writer, results.environmental, "Environment outputs")
        _write_sheet(writer, results.SOC_distribution, "SOC distributions", index=False)
        _write_sheet(writer, results.timeseries, "TimeSeries", index=False)
        _write_sheet(writer, results.production, "Production", index=False)
        _write_sheet(writer, results.genset, "Genset", index=False)
        _write_multiple_tables(writer,results.demand_side_management,"Demand side management",positions=[(0, 0), (0, 7),(0, 13),], index=False)
        _write_multiple_tables(writer, results.time_balancing, "Balancing",positions=[(0, 0), (0, 7),(0, 14),(0, 21),], index=False)
        
        _write_multiple_tables(writer, results.EMS, "EMS",positions=[(0, 0), (0, 1),(0, 3),(0, 6),(0, 9),(0, 11),], index=False)
    
        wb = writer.book
        for ws in wb.worksheets:
            _set_column_width(ws)
    
def _build_timeseries(outputs_solution, solution, Context, datetime):
    """
    Build a standardized time series DataFrame from optimization outputs.

    Parameters
    ----------
    outputs_solution : dict
        Output dictionary from evaluation_function.
    Context : object
        Simulation context (contains time, prices, etc.).
    datetime_data : array-like
        Timestamps.
    Returns
    -------
    pd.DataFrame
        Clean time series DataFrame ready for analysis/export.
    """

    TS = outputs_solution["TimeSeries"]

    # --- Datetime ---
    datetime_index = pd.to_datetime(datetime, unit="s").tz_localize(None)

    # --- Core signals ---
    load = TS["Optimized load (kW)"]
    production = TS["production (kW)"]
    grid = TS["Grid trading (kW)"]
    dg = TS["DG production (kW)"]
    curtailment = TS["Curtailment (kW)"]

    # --- Storages ---
    power_storages = pd.DataFrame(TS["Storage_TS (kW)"],index=[f"{tech} power (kW)" for tech in Context.storage.technologies],).T
    losses = pd.DataFrame(TS["Losses (kW)"],index=[f"{tech} losses (kW)" for tech in Context.storage.technologies],).T
    socs = pd.DataFrame(TS["SOCs (%)"],index=[f"{tech} SOC (%)" for tech in Context.storage.technologies],).T
    power_storage_total = power_storages.sum(axis=1)

    # --- Grid price ---
    grid_price = Context.grid.prices[solution.contract] if Context.config.connexion=="On-grid" else None

    # --- Imbalance ---
    imbalance = (production+ power_storage_total+ grid- load- curtailment+ dg)

    # --- Merging dataframe ---
    df = pd.concat([pd.DataFrame({"Datetime": datetime_index,"Load (kW)": load,"Power production (kW)": production,}),power_storages,losses,
                    pd.DataFrame({"Grid power (kW)": grid,"Grid price (€/kWh)": grid_price,"Diesel production (kW)": dg,
                    "Curtailment (kW)": curtailment,"Imbalance (kW)": imbalance,}),socs,],axis=1)

    return df

def _build_flows(outputs_solution, output_baseline, Context):
    """
    Build a standardized flows DataFrame comparing optimization vs baseline.

    Parameters
    ----------
    outputs_solution : dict
    output_baseline : dict
    Context : object

    Returns
    -------
    pd.DataFrame
    """

    # --- Base flows ---
    flows_opt = pd.DataFrame(outputs_solution["Flows"], index=["Optimization"])
    flows_base = pd.DataFrame(output_baseline["Flows"], index=["Baseline"])

    df = pd.concat([flows_opt, flows_base])

    # --- Storage flows helper ---
    def build_storage_metric(metric_name, label):
        opt_values = outputs_solution["Flows storages"][metric_name]
        base_values = output_baseline["Flows storages"][metric_name]

        data_opt = {f"{tech} {label}": opt_values[i] for i, tech in enumerate(Context.storage.technologies)}
        data_base = {f"{tech} {label}": base_values[i] for i, tech in enumerate(Context.storage.technologies)}

        return pd.concat([pd.DataFrame(data_opt, index=["Optimization"]),pd.DataFrame(data_base, index=["Baseline"]),])

    # --- Storage metrics ---
    metrics = [
        ("Annual stored energy (kWh)", "annual stored energy (kWh)"),
        ("Annual reported energy (kWh)", "annual reported energy (kWh)"),
        ("Annual losses (kWh)", "annual losses (kWh)"),
    ]

    storage_blocks = [build_storage_metric(metric, label)for metric, label in metrics]

    # --- Merge everything ---
    df = pd.concat([df] + storage_blocks, axis=1)

    return df

def _build_technical(outputs_solution, output_baseline):
    """
    Build technical comparison DataFrame.
    """
    return _build_comparison_block(outputs_solution, output_baseline, "Technical")

def _build_environmental(outputs_solution, output_baseline):
    """
    Build environmental comparison DataFrame.
    """
    return _build_comparison_block(outputs_solution, output_baseline, "Environment")

def _build_genset(outputs_solution, output_baseline):
    """
    Build genset comparison DataFrame.
    """
    return _build_comparison_block(outputs_solution, output_baseline, "Genset")

def _build_global_dispatching(outputs_solution, output_baseline):
    """
    Build balancing comparison DataFrame.
    """
    output_useprod   = _build_comparison_block(outputs_solution, output_baseline, ['Extra_outputs','Uses','useprod'])
    output_loadmeet  = _build_comparison_block(outputs_solution, output_baseline, ['Extra_outputs','Uses','Loadmeet'])
    output_whenprod  = _build_comparison_block(outputs_solution, output_baseline, ['Extra_outputs','Uses','when_prod'])
    output_whenload  = _build_comparison_block(outputs_solution, output_baseline, ['Extra_outputs','Uses','when_load'])
    output_gridexport = _build_comparison_block(outputs_solution, output_baseline, ['Extra_outputs','Grid usage','export'])
    output_gridimport = _build_comparison_block(outputs_solution, output_baseline, ['Extra_outputs','Grid usage','import'])
    
    return output_useprod,output_loadmeet,output_whenprod,output_whenload,output_gridexport,output_gridimport

def _build_demand_side_management(outputs_solution, output_baseline):
    """
    Build demand_side_management comparison DataFrame.
    """
    Load_strategy = pd.DataFrame(outputs_solution['Demand-side management']['Load strategy'])
    # Removing timezones of Load Strategy to enable excel export
    if pd.api.types.is_datetime64tz_dtype(Load_strategy['Datetime']):
        Load_strategy['Datetime'] = Load_strategy['Datetime'].dt.tz_convert(None)
    DSM_daily_strategy = outputs_solution['Demand-side management']['DSM daily strategy']
    DSM_yearly_strategy = outputs_solution['Demand-side management']['DSM yearly strategy']
   
    return Load_strategy,DSM_daily_strategy,DSM_yearly_strategy

def _build_economic(outputs_solution, output_baseline, Context):
    """
    Build economic comparison DataFrame with NPV and Payback.
    """

    econ_opt = outputs_solution["economics"].copy()
    econ_base = output_baseline["economics"].copy()
    
    # --- External computation ---
    NPV,Payback = compute_economic_comparisons(outputs_solution, output_baseline) if Context.postprocess_config.include_baseline else (np.nan,np.nan)

    # --- Assign ---
    econ_opt["NPV (€)"] = NPV
    econ_opt["Payback (yrs.)"] = Payback

    econ_base["NPV (€)"] = 0
    econ_base["Payback (yrs.)"] = np.nan

    return _build_comparison_block({"economics": econ_opt},{"economics": econ_base},"economics",)

def _build_production(solution, Context):
    """
    Build production assets DataFrame.
    """

    production_set = solution.production_set

    df = pd.DataFrame({"ID": Context.production.Ids,"Number of units": production_set,"Coverage ratio": production_set / Context.production.capacities,
        "Initial investment (€)": np.multiply(Context.production.specs_num[:,PROD_CAPEX],production_set),})
    return df

def _build_SOC_distribution(outputs_solution, output_baseline):
    """
    Build distribution of depth of discharge storage DataFrame.
    """
    return pd.DataFrame(outputs_solution['Extra_outputs']['distribution_DOD'],index=range(100))

def _build_time_balancing(outputs_solution, output_baseline):
    """
    Build time balancing DataFrames.
    """
    
    daily_optim_time_balancing = outputs_solution['Balancing']['daily time balancing']
    daily_baseline_time_balancing = output_baseline['Balancing']['daily time balancing']
    yearly_optim_time_balancing = outputs_solution['Balancing']['yearly time balancing']
    yearly_baseline_time_balancing = output_baseline['Balancing']['yearly time balancing']

    return [daily_optim_time_balancing,daily_baseline_time_balancing,yearly_optim_time_balancing,yearly_baseline_time_balancing]

def _build_EMS(outputs_solution,Context):
    """
    Build EMS data.
    """
    strategy = pd.DataFrame(data={'Strategy':outputs_solution['EMS']['strategy']},index=[0])
    repartition_coefficient = pd.DataFrame(data={'energy repartition coefficient':outputs_solution['EMS']['energy repartition coefficient']},index=[0])
    D_DSM_minimum_levels = pd.DataFrame(data={'Hour': np.arange(24)+1, 'D_DSM level (%)':100*np.concatenate((outputs_solution['EMS']['D_DSM min. levels'],np.array([1.])))})
    Y_DSM_minimum_levels = pd.DataFrame(data={'Month': np.arange(12)+1,'Y_DSM level (%)':100*np.concatenate((outputs_solution['EMS']['Y_DSM min. levels'],np.array([1.])))})
    discharge_order = pd.DataFrame({'Order':np.arange(Context.storage.n_store)+1,'Storage':Context.storage.technologies[outputs_solution['EMS']['discharge order']]})
    overlaps = pd.concat((pd.DataFrame({'effective SOC (%)':10*(np.arange(9,0,-1))}),pd.DataFrame(100*outputs_solution['EMS']['overlaps']).T),axis=1)
    overlaps.columns = ['effective SOC (%)',"intern overlaps","extern overlaps"]
   
    return [strategy,repartition_coefficient,D_DSM_minimum_levels,Y_DSM_minimum_levels,discharge_order,overlaps]


def build_results(solution, Context, datetime):
    """Build all post-processing result tables for a solution.

    This function evaluates the solution, computes comparison metrics
    against the baseline scenario, and generates all post-processing
    DataFrames used for exports and visualization.
    
    Args:
        solution: Optimized solution object.
        Context: Global ERMESS context object.
        datetime (np.ndarray | pandas.DatetimeIndex): Simulation timestamps.
    
    Returns:
        _PostProcessingResults: Container gathering all generated
        post-processing tables and indicators.
    """

    evaluation_function = Context.postprocess_config.evaluation_function
    evaluation_baseline = Context.postprocess_config.evaluation_base

    output_baseline = evaluation_baseline(Context,datetime)
    outputs_solution = evaluation_function(solution,Context,datetime)
    NPV,Payback = compute_economic_comparisons(outputs_solution, output_baseline)
    outputs_solution['economics']["NPV (€)"] = NPV
    outputs_solution['economics']["Payback (yrs.)"] = Payback

    # --- DataFrames ---
    timeseries = _build_timeseries(outputs_solution, solution, Context, datetime)
    flows = _build_flows(outputs_solution, output_baseline, Context)
    technical = _build_technical(outputs_solution, output_baseline)
    economic = _build_economic(outputs_solution, output_baseline, Context)
    environmental = _build_environmental(outputs_solution, output_baseline)
    storages = pd.DataFrame(outputs_solution['Storages'], index=Context.storage.technologies)
    production = _build_production(solution, Context)
    global_dispatching = _build_global_dispatching(outputs_solution, output_baseline)
    SOC_distribution = _build_SOC_distribution(outputs_solution, output_baseline)
    genset = _build_genset(outputs_solution, output_baseline)
    demand_side_management = _build_demand_side_management(outputs_solution, output_baseline)
    time_balancing = _build_time_balancing(outputs_solution, output_baseline)  
    EMS = _build_EMS(outputs_solution,Context) 
    
    return _PostProcessingResults(timeseries=timeseries,technical=technical,economic=economic,environmental=environmental,flows=flows,
        storages=storages,production=production,global_dispatching=global_dispatching,SOC_distribution=SOC_distribution,genset=genset,
        demand_side_management=demand_side_management,time_balancing=time_balancing,EMS=EMS)

def post_traitement(solution, Context, datetime):
    """Run the complete post-processing workflow.

    This function generates post-processing results and exports them
    according to the configuration defined in the context.
    
    Args:
        solution: Optimized solution object.
        Context: Global ERMESS context object.
        datetime (np.ndarray | pandas.DatetimeIndex): Simulation timestamps.
    """

    results = build_results(solution, Context, datetime)

    if Context.postprocess_config.export_type=='Excel':
        export_to_excel(results, Context)

    if Context.postprocess_config.export_charts:
        EgE.add_excel_charts(Context=Context,charts_config=charts_config)


