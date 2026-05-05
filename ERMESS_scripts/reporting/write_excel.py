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

def output_build_production(solution, Contexte):
    return pd.DataFrame({
        'ID': Contexte.specs_Id,
        'Number of units': solution.production_set,
        'Coverage ratio': solution.production_set / Contexte.production.capacities,
        'Initial investment (€)': np.multiply(Contexte.production.specs_num[:, PROD_CAPEX], solution.production_set)
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
    
def export_to_excel(results, Contexte):

    with pd.ExcelWriter(Contexte.postprocess_config.file_name, engine='openpyxl') as writer:

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
    
def _build_timeseries(outputs_solution, solution, Contexte, datetime):
    """
    Build a standardized time series DataFrame from optimization outputs.

    Parameters
    ----------
    outputs_solution : dict
        Output dictionary from evaluation_function.
    Contexte : object
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
    power_storages = pd.DataFrame(TS["Storage_TS (kW)"],index=[f"{tech} power (kW)" for tech in Contexte.storage.technologies],).T
    losses = pd.DataFrame(TS["Losses (kW)"],index=[f"{tech} losses (kW)" for tech in Contexte.storage.technologies],).T
    socs = pd.DataFrame(TS["SOCs (%)"],index=[f"{tech} SOC (%)" for tech in Contexte.storage.technologies],).T
    power_storage_total = power_storages.sum(axis=1)

    # --- Grid price ---
    grid_price = Contexte.grid.prices[solution.contract] if Contexte.config.connexion=="On-grid" else None

    # --- Imbalance ---
    imbalance = (production+ power_storage_total+ grid- load- curtailment+ dg)

    # --- Merging dataframe ---
    df = pd.concat([pd.DataFrame({"Datetime": datetime_index,"Load (kW)": load,"Power production (kW)": production,}),power_storages,losses,
                    pd.DataFrame({"Grid power (kW)": grid,"Grid price (€/kWh)": grid_price,"Diesel production (kW)": dg,
                    "Curtailment (kW)": curtailment,"Imbalance (kW)": imbalance,}),socs,],axis=1)

    return df

def _build_flows(outputs_solution, output_baseline, Contexte):
    """
    Build a standardized flows DataFrame comparing optimization vs baseline.

    Parameters
    ----------
    outputs_solution : dict
    output_baseline : dict
    Contexte : object

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

        data_opt = {f"{tech} {label}": opt_values[i] for i, tech in enumerate(Contexte.storage.technologies)}
        data_base = {f"{tech} {label}": base_values[i] for i, tech in enumerate(Contexte.storage.technologies)}

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

def _build_economic(outputs_solution, output_baseline, Contexte):
    """
    Build economic comparison DataFrame with NPV and Payback.
    """

    econ_opt = outputs_solution["economics"].copy()
    econ_base = output_baseline["economics"].copy()
    
    # --- External computation ---
    NPV,Payback = compute_economic_comparisons(outputs_solution, output_baseline) if Contexte.postprocess_config.include_baseline else (np.nan,np.nan)

    # --- Assign ---
    econ_opt["NPV (€)"] = NPV
    econ_opt["Payback (yrs.)"] = Payback

    econ_base["NPV (€)"] = 0
    econ_base["Payback (yrs.)"] = np.nan

    return _build_comparison_block({"economics": econ_opt},{"economics": econ_base},"economics",)

def _build_production(solution, Contexte):
    """
    Build production assets DataFrame.
    """

    production_set = solution.production_set

    df = pd.DataFrame({"ID": Contexte.production.Ids,"Number of units": production_set,"Coverage ratio": production_set / Contexte.production.capacities,
        "Initial investment (€)": np.multiply(Contexte.production.specs_num[:,PROD_CAPEX],production_set),})
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

def _build_EMS(outputs_solution,Contexte):
    """
    Build EMS data.
    """
    strategy = pd.DataFrame(data={'Strategy':outputs_solution['EMS']['strategy']},index=[0])
    repartition_coefficient = pd.DataFrame(data={'energy repartition coefficient':outputs_solution['EMS']['energy repartition coefficient']},index=[0])
    D_DSM_minimum_levels = pd.DataFrame(data={'Hour': np.arange(24)+1, 'D_DSM level (%)':100*np.concatenate((outputs_solution['EMS']['D_DSM min. levels'],np.array([1.])))})
    Y_DSM_minimum_levels = pd.DataFrame(data={'Month': np.arange(12)+1,'Y_DSM level (%)':100*np.concatenate((outputs_solution['EMS']['Y_DSM min. levels'],np.array([1.])))})
    discharge_order = pd.DataFrame({'Order':np.arange(Contexte.storage.n_store)+1,'Storage':Contexte.storage.technologies[outputs_solution['EMS']['discharge order']]})
    overlaps = pd.concat((pd.DataFrame({'effective SOC (%)':10*(np.arange(9,0,-1))}),pd.DataFrame(100*outputs_solution['EMS']['overlaps']).T),axis=1)
    overlaps.columns = ['effective SOC (%)',"intern overlaps","extern overlaps"]
   
    return [strategy,repartition_coefficient,D_DSM_minimum_levels,Y_DSM_minimum_levels,discharge_order,overlaps]


def build_results(solution, Contexte, datetime):

    evaluation_function = Contexte.postprocess_config.evaluation_function
    evaluation_baseline = Contexte.postprocess_config.evaluation_base

    output_baseline = evaluation_baseline(Contexte,datetime)
    outputs_solution = evaluation_function(solution,Contexte,datetime)
    NPV,Payback = compute_economic_comparisons(outputs_solution, output_baseline)
    outputs_solution['economics']["NPV (€)"] = NPV
    outputs_solution['economics']["Payback (yrs.)"] = Payback

    # --- DataFrames ---
    timeseries = _build_timeseries(outputs_solution, solution, Contexte, datetime)
    flows = _build_flows(outputs_solution, output_baseline, Contexte)
    technical = _build_technical(outputs_solution, output_baseline)
    economic = _build_economic(outputs_solution, output_baseline, Contexte)
    environmental = _build_environmental(outputs_solution, output_baseline)
    storages = pd.DataFrame(outputs_solution['Storages'], index=Contexte.storage.technologies)
    production = _build_production(solution, Contexte)
    global_dispatching = _build_global_dispatching(outputs_solution, output_baseline)
    SOC_distribution = _build_SOC_distribution(outputs_solution, output_baseline)
    genset = _build_genset(outputs_solution, output_baseline)
    demand_side_management = _build_demand_side_management(outputs_solution, output_baseline)
    time_balancing = _build_time_balancing(outputs_solution, output_baseline)  
    EMS = _build_EMS(outputs_solution,Contexte) 
    
    return _PostProcessingResults(timeseries=timeseries,technical=technical,economic=economic,environmental=environmental,flows=flows,
        storages=storages,production=production,global_dispatching=global_dispatching,SOC_distribution=SOC_distribution,genset=genset,
        demand_side_management=demand_side_management,time_balancing=time_balancing,EMS=EMS)

def post_traitement(solution, Contexte, datetime):

    results = build_results(solution, Contexte, datetime)

    if Contexte.postprocess_config.export_type=='Excel':
        export_to_excel(results, Contexte)

    if Contexte.postprocess_config.export_charts:
        EgE.add_excel_charts(Contexte=Contexte,charts_config=charts_config)


#def post_traitement(solution,datetime_data,evaluation_function,cost_base,D_movable_load,Y_movable_load,storage_techs,specs_Id,Contract_Id,n_days,file_name_out,Contexte):
    """
    Post-process an energy system optimization solution and export detailed outputs to Excel.
    
    This function evaluates an optimization solution against baseline costs,
    aggregates technical, economic, and environmental metrics, prepares
    time series data for production, storage, grid, and load balancing,
    and exports structured results and charts to an Excel workbook.
    
    Args:
        solution (object): Optimization solution object.
        datetime_data (array-like): Simulation datetime series (in seconds or timestamps).
        evaluation_function (callable): Function to evaluate the solution, returning detailed metrics.
        cost_base (callable): Function to compute baseline cost for comparison.
        D_movable_load (array-like): Daily movable load time series (kW).
        Y_movable_load (array-like): Yearly movable load time series (kW).
        storage_techs (list): List of storage technology names.
        specs_Id (array-like): Identifiers of production units.
        Contract_Id (array-like): Identifiers of contracts.
        n_days (int): Number of simulation days.
        file_name_out (str): Path to output Excel file.
        Contexte (object): Context object containing system parameters, prices,
            storage characteristics, time resolution, diesel generator specs,
            and optimization type.
    
    Returns:
        None: Saves results and charts to Excel.
    
    Note:
        Produces multiple sheets in the Excel output, including:
            - Flows, Balancing, Financial outputs, Technical, Environment outputs,
              Storages, SOC distributions, TimeSeries, Production, EMS, DG, DSM,
              and Time_balancing.
              
        Adds charts for production use.
        Computes additional economic indicators such as NPV and Payback period.
        Handles both "pro" (advanced) and standard optimization types.
        Uses `openpyxl` for Excel writing and chart generation.
    """               
#    production_set=solution.production_set 
#    outputs_solution = evaluation_function(solution,datetime_data,Contexte.storage_characteristics,Contexte.time_resolution,Contexte.n_store,Contexte.duration_years,Contexte.specs_num,Contexte.prices_num,Contexte.fixed_premium,Contexte.Overrun,Contexte.Selling_price,Contexte.Non_movable_load,Contexte.total_D_Movable_load,D_movable_load, Contexte.total_Y_Movable_load,Y_movable_load,Contexte.Grid_Fossil_fuel_ratio,Contexte.Main_grid_PoF_ratio, Contexte.Main_grid_emissions,Contexte.prod_C,Contexte.prods_U,Contexte.Bounds_prod ,Contexte.constraint_num,Contexte.constraint_level,Contexte.cost_constraint,Contexte.n_bits,Contexte.Connexion,Contexte.DG_fuel_consumption,Contexte.DG_fuel_cost,Contexte.DG_unit_cost,Contexte.DG_lifetime,Contexte.DG_maintenance_cost,Contexte.DG_EROI,Contexte.fuel_CO2eq_emissions,storage_techs,n_days)       
#    datetime_excel = pd.to_datetime(datetime_data,unit='s').dt.tz_localize(None)

#    load = outputs_solution['TimeSeries']['Optimized load (kW)']
#    output_baseline = cost_base(solution,datetime_data,Contexte.storage_characteristics,Contexte.time_resolution,Contexte.n_store,Contexte.duration_years,Contexte.specs_num,Contexte.prices_num[0],Contexte.fixed_premium[0],Contexte.Overrun[0],Contexte.Selling_price[0],Contexte.Non_movable_load,D_movable_load, Y_movable_load,Contexte.Grid_Fossil_fuel_ratio,Contexte.Main_grid_PoF_ratio,Contexte.Main_grid_emissions,Contexte.prod_C,Contexte.prods_U,Contexte.Bounds_prod ,Contexte.constraint_num,Contexte.constraint_level,Contexte.cost_constraint,Contexte.n_bits,Contexte.Connexion,Contexte.DG_fuel_consumption,Contexte.DG_fuel_cost,Contexte.DG_unit_cost,Contexte.DG_lifetime,Contexte.DG_maintenance_cost,Contexte.DG_EROI,Contexte.fuel_CO2eq_emissions,storage_techs,n_days)

    
#    losses = pd.DataFrame(outputs_solution['TimeSeries']['Losses (kW)'] ,index=[storage_techs[i]+' losses' for i in range(Contexte.n_store)],columns=None).T   
#    power_storages = pd.DataFrame(outputs_solution['TimeSeries']['Storage_TS (kW)'],index=[storage_techs[i]+' power (kW)' for i in range(Contexte.n_store)],columns=None).transpose()
#    power_storage = power_storages.sum(axis=1)
      
#    SOCs=pd.DataFrame(outputs_solution['TimeSeries']['SOCs (%)'],index=[storage_techs[i]+' SOC' for i in range(Contexte.n_store)],columns=None).transpose()   
#    production = outputs_solution['TimeSeries']['production (kW)']
#    Grid_trading = outputs_solution['TimeSeries']['Grid trading (kW)']

    #Technical
    
    #Environmental
    
#    DG_production_solution = outputs_solution['TimeSeries']['DG production (kW)']
        
#    curtailment_solution = outputs_solution['TimeSeries']['Curtailment (kW)']

#    outputs_TS = pd.DataFrame(data={'Datetime':datetime_excel,'Load (kW)':load,'Power production (kW)':production},index=None)
#    outputs_TS = outputs_TS.join(power_storages)
#    outputs_TS = outputs_TS.join(losses)
#    outputs_TS = outputs_TS.join(pd.DataFrame(data={'Grid power (kW)':Grid_trading,'Grid price (€/kWh)':Contexte.prices_num[solution.contract],'Diesel production (kW)':DG_production_solution,'Curtailment (kW)':curtailment_solution,'Imbalance (kW)':production + power_storage + Grid_trading - load - curtailment_solution + DG_production_solution}))    
#    outputs_TS = outputs_TS.join(SOCs)
    

#    Technical_outputs=pd.concat((pd.DataFrame(outputs_solution['Technical'],index=['Optimization']),pd.DataFrame(output_baseline['Technical'],index=['Baseline'])))
#    Environmental_outputs=pd.concat((pd.DataFrame(outputs_solution['Environment'],index=['Optimization']),pd.DataFrame(output_baseline['Environment'],index=['Baseline'])))
#    NPV = outputs_solution['economics']['Value (€)']-output_baseline['economics']['Annual net benefits (€/yrs.)']*outputs_solution['Technical']['Installation lifetime (yrs.)']
#    Payback = outputs_solution['economics']['Initial investment (€)']/(outputs_solution['economics']['Annual net benefits (€/yrs.)']-output_baseline['economics']['Annual net benefits (€/yrs.)'])
#    if Payback<0 : 
#        Payback = np.nan
        
#    outputs_solution['economics']["NPV (€)"] = NPV
#    outputs_solution['economics']["Payback (yrs.)"] = Payback

#    output_baseline['economics']["NPV (€)"] = 0
#    output_baseline['economics']["Payback (yrs.)"] = np.nan
#    Economic_outputs=pd.concat((pd.DataFrame(outputs_solution['economics'],index=['Optimization']),pd.DataFrame(output_baseline['economics'],index=['Baseline'])))
    
    
#    Output_production = pd.DataFrame(data={'ID':specs_Id,'Number of units':production_set,'Coverage ratio':production_set/Contexte.specs_num[:,3],'Initial investment (€)':np.multiply(Contexte.specs_num[:,0],production_set)})
#    dist_DOD = outputs_solution['Extra_outputs']['distribution_DOD']
#    output_storages = pd.DataFrame(outputs_solution['Storages'],index=storage_techs)
    
#    output_useprod = pd.concat((outputs_solution['Extra_outputs']['Uses']['useprod'],output_baseline['Extra_outputs']['Uses']['useprod']))
#    output_loadmeet = pd.concat((outputs_solution['Extra_outputs']['Uses']['Loadmeet'],output_baseline['Extra_outputs']['Uses']['Loadmeet']))
#    output_whenprod = pd.concat((outputs_solution['Extra_outputs']['Uses']['when_prod'],output_baseline['Extra_outputs']['Uses']['when_prod']))
#    output_whenload = pd.concat((outputs_solution['Extra_outputs']['Uses']['when_load'],output_baseline['Extra_outputs']['Uses']['when_load']))
#    output_gridexport = pd.concat((outputs_solution['Extra_outputs']['Grid usage']['export'],output_baseline['Extra_outputs']['Grid usage']['export']))
#    output_gridimport = pd.concat((outputs_solution['Extra_outputs']['Grid usage']['import'],output_baseline['Extra_outputs']['Grid usage']['import']))

#    output_useprod.index,output_loadmeet.index,output_whenprod.index,output_whenload.index,output_gridexport.index,output_gridimport.index = [['Optimization','Baseline'] for i in range(6)]

    
#    Flows = pd.concat((pd.DataFrame(outputs_solution['Flows'],index=['Optimization']),pd.DataFrame(output_baseline['Flows'],index=['Baseline'])))
#    for i in range(len(storage_techs)):
#        Flows = Flows.join(pd.concat((pd.DataFrame(data={storage_techs[i] + ' annual stored energy (kWh)' : outputs_solution['Flows storages']['Annual stored energy (kWh)'][i]},index=['Optimization']),pd.DataFrame(data={storage_techs[i] + ' annual stored energy (kWh)' : output_baseline['Flows storages']['Annual stored energy (kWh)'][i]},index=['Baseline'] ))))
#        Flows = Flows.join(pd.concat((pd.DataFrame(data={storage_techs[i] + ' annual reported energy (kWh)' : outputs_solution['Flows storages']['Annual reported energy (kWh)'][i]},index=['Optimization']),pd.DataFrame(data={storage_techs[i] + ' annual reported energy (kWh)' : output_baseline['Flows storages']['Annual reported energy (kWh)'][i]},index=['Baseline'] ))))
#        Flows = Flows.join(pd.concat((pd.DataFrame(data={storage_techs[i] + ' annual losses (kWh)' : outputs_solution['Flows storages']['Annual losses (kWh)'][i]},index=['Optimization']),pd.DataFrame(data={storage_techs[i] + ' annual losses (kWh)' : output_baseline['Flows storages']['Annual losses (kWh)'][i]},index=['Baseline'] ))))
    
#    if (Contexte.type_optim=='pro'):
#        PMS_D_DSM_min_levels = pd.DataFrame({'Hour': np.arange(24)+1, 'min. level (%)': 100*np.concatenate((outputs_solution['PMS']['D_DSM min. levels'],np.array([1.])))})
#        PMS_Y_DSM_min_levels = pd.DataFrame({'Month': np.arange(12)+1, 'min. level (%)': 100*np.concatenate((outputs_solution['PMS']['Y_DSM min. levels'],np.array([1.])))})
#        discharge_order = pd.DataFrame({'Order':np.arange(Contexte.n_store)+1,'Storage':storage_techs[outputs_solution['PMS']['discharge order']]})
#        taking_over = pd.DataFrame({'effective SOC (%)':10*(np.arange(9,0,-1)),'Taking over level (%)':100*outputs_solution['PMS']['taking over'][0],'DG taking over level (%)' if Contexte.Connexion=='Off-grid' else 'Grid taking over level (%)' :100*outputs_solution['PMS']['taking over'][1]})
        
#    DG=pd.concat((pd.DataFrame(outputs_solution['DG'],index=['Optimization']),pd.DataFrame(output_baseline['DG'],index=['Baseline'])))
#    outputs_solution['Demand-side management']['Load strategy']['Datetime']=datetime_excel
    
#    with pd.ExcelWriter(file_name_out,engine='openpyxl') as writer:
#        Flows.to_excel(writer,sheet_name='Flows')
#        output_useprod.to_excel(writer,sheet_name='Balancing')
#        output_loadmeet.to_excel(writer,sheet_name='Balancing',startrow=3)
#        output_whenprod.to_excel(writer,sheet_name='Balancing',startrow=6)
#        output_whenload.to_excel(writer,sheet_name='Balancing',startrow=9)
#        output_gridexport.to_excel(writer,sheet_name='Balancing',startrow=12)
#        output_gridimport.to_excel(writer,sheet_name='Balancing',startrow=15)
#        Economic_outputs.to_excel(writer,sheet_name='Financial outputs')
#        Technical_outputs.to_excel(writer,sheet_name='Technical')        
#        pd.DataFrame(output_storages).to_excel(writer,sheet_name='Storages')
#        Environmental_outputs.to_excel(writer,sheet_name='Environment outputs')
#        dist_DOD.to_excel(writer,sheet_name='SOC distributions',index=None)
#        outputs_TS.to_excel(writer,sheet_name='TimeSeries',index=None)
#        Output_production.to_excel(writer,sheet_name='Production',index=None)
#        if (Contexte.type_optim=='pro'):
#            pd.DataFrame(data={'Strategy':outputs_solution['PMS']['strategy']},index=[0]).to_excel(writer,sheet_name='EMS',index=None)
#            pd.DataFrame(data={'DSM coefficient':outputs_solution['PMS']['surplus repartition coefficient']},index=[0]).to_excel(writer,sheet_name='EMS',index=None,startrow=3)
#            PMS_D_DSM_min_levels.to_excel(writer,sheet_name='EMS',startcol = 2,index=None)
#            PMS_Y_DSM_min_levels.to_excel(writer,sheet_name='EMS',startcol = 5,index=None)
#            discharge_order.to_excel(writer,sheet_name='EMS',startcol = 8,index=None)
#            taking_over.to_excel(writer,sheet_name='EMS',startcol = 11,index=None)
#        DG.to_excel(writer,sheet_name='DG')
#        outputs_solution['Demand-side management']['Load strategy'].to_excel(writer,sheet_name='DSM',index=None)
#        outputs_solution['Demand-side management']['DSM daily strategy'].to_excel(writer,sheet_name='DSM',index=None,startcol=7)
#        outputs_solution['Demand-side management']['DSM yearly strategy'].to_excel(writer,sheet_name='DSM',index=None,startcol=13)
#        outputs_solution['Balancing']['daily time balancing'].to_excel(writer,sheet_name='Time_balancing',index=None)
#        output_baseline['Balancing']['daily time balancing'].to_excel(writer,sheet_name='Time_balancing',index=None,startcol=7)
#        outputs_solution['Balancing']['yearly time balancing'].to_excel(writer,sheet_name='Time_balancing',index=None,startcol=14)
#        output_baseline['Balancing']['yearly time balancing'].to_excel(writer,sheet_name='Time_balancing',index=None,startcol=21)
#        
    #Output Charts
#    wb = openpyxl.load_workbook(file_name_out)
    
#    set_column_width(wb['Flows'])
    
#    ws = wb['Balancing']
#    c01 = openpyxl.chart.BarChart()
#    c01.title = "Use of production"
#    c01.grouping='stacked'
#    c01.overlap=100
#    c01.add_data(openpyxl.chart.Reference(ws,min_col=2, min_row=1, max_col=4, max_row=3), titles_from_data=True)
#    c01.set_categories(openpyxl.chart.Reference(ws,min_col=1, min_row=2, max_col=1, max_row=3))
#    c01.x_axis.title = 'Scenario'
#    c01.y_axis.title = 'Energy (kWh)'
#    c01.layout = openpyxl.chart.layout.Layout(manualLayout=openpyxl.chart.layout.ManualLayout(x=0.025, y=0.005,w=0.6, h=0.75))
#    c01.layout.layoutTarget = "inner" 
#    c01.y_axis.scaling.min = 0
#    c01.y_axis.scaling.max = 1.3*max(sum(production),sum(load))/Contexte.time_resolution
#    c01.x_axis.delete = False
#    c01.y_axis.delete = False
#    c01.height = 7.5
#    c01.width = 20
#    c01.legend.overlay = False
#    c01.x_axis.tickLblPos = "low"
#    c01.x_axis.tickLblPos = "low"
#    ws.add_chart(c01, "A1")   
def old_post_processing(): 
    c02 = openpyxl.chart.BarChart()
    c02.title = "Load response"
    c02.grouping='stacked'
    c02.overlap=100
    c02.add_data(openpyxl.chart.Reference(ws,min_col=2, min_row=4, max_col=4, max_row=6), titles_from_data=True)
    c02.set_categories(openpyxl.chart.Reference(ws,min_col=1, min_row=5, max_col=1, max_row=6))
    c02.x_axis.title = 'Scenario'
    c02.y_axis.title = 'Energy (kWh)'
    c02.layout = openpyxl.chart.layout.Layout(manualLayout=openpyxl.chart.layout.ManualLayout(x=0.025, y=0.005,w=0.6, h=0.75))
    c02.layout.layoutTarget = "inner" 
    c02.y_axis.scaling.min = 0
    c02.y_axis.scaling.max = 1.3*max(sum(production),sum(load))/Contexte.time_resolution
    c02.x_axis.delete = False
    c02.y_axis.delete = False
    c02.height = 7.5
    c02.width = 20
    c02.legend.overlay = False
    c02.x_axis.tickLblPos = "low"
    c02.x_axis.tickLblPos = "low"
    ws.add_chart(c02, "F1")   
    
    c03 = openpyxl.chart.BarChart()
    c03.title = "When production exceeds load"
    c03.grouping='stacked'
    c03.overlap=100
    c03.add_data(openpyxl.chart.Reference(ws,min_col=2, min_row=7, max_col=4+2*Contexte.n_store, max_row=9), titles_from_data=True)
    c03.set_categories(openpyxl.chart.Reference(ws,min_col=1, min_row=8, max_col=1, max_row=9))
    c03.x_axis.title = 'Scenario'
    c03.y_axis.title = 'Energy (kWh)'
    c03.layout = openpyxl.chart.layout.Layout(manualLayout=openpyxl.chart.layout.ManualLayout(x=0.025, y=0.005,w=0.6, h=0.75))
    c03.layout.layoutTarget = "inner" 
    c03.y_axis.scaling.min = 0
    c03.x_axis.delete = False
    c03.y_axis.delete = False
    c03.height = 7.5
    c03.width = 20
    c03.legend.overlay = False
    c03.x_axis.tickLblPos = "low"
    c03.x_axis.tickLblPos = "low"
    ws.add_chart(c03, "A31") 
    
    c04 = openpyxl.chart.BarChart()
    c04.title = "When load exceeds production"
    c04.grouping='stacked'
    c04.overlap=100
    c04.add_data(openpyxl.chart.Reference(ws,min_col=2, min_row=10, max_col=4+2*Contexte.n_store, max_row=12), titles_from_data=True)
    c04.set_categories(openpyxl.chart.Reference(ws,min_col=1, min_row=11, max_col=1, max_row=12))
    c04.x_axis.title = 'Scenario'
    c04.y_axis.title = 'Energy (kWh)'
    c04.layout = openpyxl.chart.layout.Layout(manualLayout=openpyxl.chart.layout.ManualLayout(x=0.025, y=0.005,w=0.6, h=0.75))
    c04.layout.layoutTarget = "inner" 
    c04.y_axis.scaling.min = 0
    c04.x_axis.delete = False
    c04.y_axis.delete = False
    c04.height = 7.5
    c04.width = 20
    c04.legend.overlay = False
    c04.x_axis.tickLblPos = "low"
    c04.x_axis.tickLblPos = "low"
    ws.add_chart(c04, "F31") 
    
    c05 = openpyxl.chart.BarChart()
    c05.title = "Grid usage (exportation)"
    c05.grouping='stacked'
    c05.overlap=100
    c05.add_data(openpyxl.chart.Reference(ws,min_col=2, min_row=13, max_col=3, max_row=15), titles_from_data=True)
    c05.set_categories(openpyxl.chart.Reference(ws,min_col=1, min_row=14, max_col=1, max_row=15))
    c05.x_axis.title = 'Scenario'
    c05.y_axis.title = 'Energy (kWh)'
    c05.layout = openpyxl.chart.layout.Layout(manualLayout=openpyxl.chart.layout.ManualLayout(x=0.025, y=0.005,w=0.6, h=0.75))
    c05.layout.layoutTarget = "inner" 
    c05.y_axis.scaling.min = 0
    c05.x_axis.delete = False
    c05.y_axis.delete = False
    c05.height = 7.5
    c05.width = 20
    c05.legend.overlay = False
    c05.x_axis.tickLblPos = "low"
    c05.x_axis.tickLblPos = "low"
    ws.add_chart(c05, "A16") 
    
    c06 = openpyxl.chart.BarChart()
    c06.title = "Grid usage (importation)"
    c06.grouping='stacked'
    c06.overlap=100
    c06.add_data(openpyxl.chart.Reference(ws,min_col=2, min_row=16, max_col=3, max_row=18), titles_from_data=True)
    c06.set_categories(openpyxl.chart.Reference(ws,min_col=1, min_row=17, max_col=1, max_row=18))
    c06.x_axis.title = 'Scenario'
    c06.y_axis.title = 'Energy (kWh)'
    c06.layout = openpyxl.chart.layout.Layout(manualLayout=openpyxl.chart.layout.ManualLayout(x=0.025, y=0.005,w=0.6, h=0.75))
    c06.layout.layoutTarget = "inner" 
    c06.y_axis.scaling.min = 0
    c06.x_axis.delete = False
    c06.y_axis.delete = False
    c06.height = 7.5
    c06.width = 20
    c06.legend.overlay = False
    c06.x_axis.tickLblPos = "low"
    c06.x_axis.tickLblPos = "low"
    ws.add_chart(c06, "F16") 
    
    set_column_width(ws)
    
    ws = wb['Financial outputs']
    c2 = openpyxl.chart.BarChart()
    c2.title = "LCOE decomposition"
    c2.grouping='stacked'
    c2.overlap=100
    c2.add_data(openpyxl.chart.Reference(ws,min_col=3, min_row=1, max_col=12, max_row=3), titles_from_data=True)
    c2.set_categories(openpyxl.chart.Reference(ws,min_col=1, min_row=2, max_col=1, max_row=3))
    c2.x_axis.title = 'Scenario'
    c2.y_axis.title = 'Cost (€/kWh)'
    c2.layout = openpyxl.chart.layout.Layout(manualLayout=openpyxl.chart.layout.ManualLayout(x=0.025, y=0.005,w=0.85, h=0.7))
    c2.layout.layoutTarget = "inner"  
    c2.x_axis.delete = False
    c2.y_axis.delete = False
    c2.legend.position = 'b'  
    c2.height = 15
    c2.width = 15
    c2.legend.overlay = False
    c2.x_axis.tickLblPos = "low"
    c2.x_axis.tickLblPos = "low"
    ws.add_chart(c2, "B6")  
    
    c2_2 = openpyxl.chart.BarChart()
    c2_2.title = "Exportation income"
    c2_2.grouping='stacked'
    c2_2.overlap=100
    c2_2.add_data(openpyxl.chart.Reference(ws,min_col=13, min_row=1, max_col=13, max_row=3), titles_from_data=True)
    c2_2.set_categories(openpyxl.chart.Reference(ws,min_col=1, min_row=2, max_col=1, max_row=3))
    c2_2.x_axis.title = 'Scenario'
    c2_2.y_axis.title = 'Income (€/kWh)'
    c2_2.layout = openpyxl.chart.layout.Layout(manualLayout=openpyxl.chart.layout.ManualLayout(x=0.025, y=0.005,w=0.85, h=0.7))
    c2_2.layout.layoutTarget = "inner"  
    c2_2.x_axis.delete = False
    c2_2.y_axis.delete = False
    c2_2.legend.position = 'b'
    c2_2.height = 15
    c2_2.width = 15
    c2_2.legend.overlay = False
    c2_2.x_axis.tickLblPos = "low"
    c2_2.x_axis.tickLblPos = "low"
    ws.add_chart(c2_2, "F6")   
    
    set_column_width(ws)
    
    set_column_width(wb['Technical'])
    set_column_width(wb['Environment outputs'])  
    set_column_width(wb['Storages'])
    
    ws = wb['SOC distributions']
    c1 = openpyxl.chart.LineChart()
    c1.title = "SOC percentile distributions"
    c1.add_data(openpyxl.chart.Reference(ws,min_col=2, min_row=1, max_col=1+Contexte.n_store, max_row=101), titles_from_data=True)
    c1.set_categories(openpyxl.chart.Reference(ws,min_col=1, min_row=2, max_col=1, max_row=102))
    c1.layout = openpyxl.chart.layout.Layout(manualLayout=openpyxl.chart.layout.ManualLayout(x=0.025, y=0.005,w=0.85, h=0.7))
    c1.layout.layoutTarget = "inner"    
    c1.height,c1.width = (10,18)
    c1.x_axis.scaling.min,c1.y_axis.scaling.min = (0,0)
    c1.x_axis.delete = False
    c1.y_axis.delete = False
    c1.legend.position = 'b'
    c1.x_axis.title = 'Percentile'
    c1.y_axis.title = 'Density'
    ws.add_chart(c1, "F2") 
    
    set_column_width(ws)
    #fmt = '%Y-%m-%d %H:%M:%S'
   
    ws = wb['TimeSeries']
    c3 = openpyxl.chart.ScatterChart()
    c3.title = "Storages SOCs"
    xvalues = openpyxl.chart.Reference(ws,min_col=1, min_row=1, max_col=1, max_row=Contexte.n_bits+1)
    for i in range(Contexte.n_store):
        values = openpyxl.chart.Reference(ws,min_col=9+Contexte.n_store*2+i, min_row=1, max_col=9+Contexte.n_store*2+i, max_row=Contexte.n_bits+1)
        series = openpyxl.chart.Series(values, xvalues, title_from_data=True)
        series.graphicalProperties.line.width = 12000
        c3.series.append(series)
    c3.y_axis.title = 'SOC'
    c3.x_axis.title = 'Date'
    c3.layout = openpyxl.chart.layout.Layout(manualLayout=openpyxl.chart.layout.ManualLayout(x=0.0015, y=0.005,w=0.97, h=0.7))
    c3.layout.layoutTarget = "inner"    
    c3.x_axis.delete = True
    c3.y_axis.delete = False
    c3.legend.position = 'l'
    (c3.height,c3.width) = (9,200)
    c3.legend.overlay = False
    c3.x_axis.majorUnit = Contexte.time_resolution*24*7
    c3.y_axis.scaling.min,c3.y_axis.scaling.max = (0.0,1.0)    
    c3.y_axis.majorUnit = 0.2
    c3.x_axis.scaling.min,c3.x_axis.scaling.max =(0,Contexte.n_bits)
    ws.add_chart(c3, "O3") 
    
    set_column_width(ws)
    
    c4 = openpyxl.chart.ScatterChart()
    c4.title = "TimeSeries"
    xvalues = openpyxl.chart.Reference(ws,min_col=1, min_row=1, max_col=1, max_row=Contexte.n_bits+1)
    values1 = openpyxl.chart.Reference(ws,min_col=2, min_row=1, max_col=2, max_row=Contexte.n_bits+1)
    series1 = openpyxl.chart.Series(values1, xvalues, title_from_data=True)
    series1.graphicalProperties.line.width = 12000
    values2 = openpyxl.chart.Reference(ws,min_col=3, min_row=1, max_col=3, max_row=Contexte.n_bits+1)
    series2 = openpyxl.chart.Series(values2, xvalues, title_from_data=True)
    series2.graphicalProperties.line.width = 12000
    values3 = openpyxl.chart.Reference(ws,min_col=6+2*Contexte.n_store, min_row=1, max_col=6+2*Contexte.n_store, max_row=Contexte.n_bits+1)
    series3 = openpyxl.chart.Series(values3, xvalues, title_from_data=True)
    series3.graphicalProperties.line.width = 12000
    c4.series.append(series1)
    c4.series.append(series2)
    c4.series.append(series3)    
    c4.x_axis.title = 'Date'
    c4.y_axis.title = 'Power (kW)'
    c4.layout = openpyxl.chart.layout.Layout(manualLayout=openpyxl.chart.layout.ManualLayout(x=0.0015, y=0.005,w=0.97, h=0.7))
    c4.layout.layoutTarget = "inner"    
    c4.x_axis.delete = True
    c4.y_axis.delete = False
    c4.legend.position = 'l'
    (c4.height,c4.width) = (9,200)
    c4.legend.overlay = False
    c4.x_axis.majorUnit = Contexte.time_resolution*24*7
    c4.x_axis.scaling.min,c4.x_axis.scaling.max =(0,Contexte.n_bits)
    ws.add_chart(c4, "O23")  
    
    c5 = openpyxl.chart.ScatterChart()
    c5.title = "Storage timeSeries"
    xvalues = openpyxl.chart.Reference(ws,min_col=1, min_row=1, max_col=1, max_row=Contexte.n_bits+1)
    for i in range(Contexte.n_store) :
        values = openpyxl.chart.Reference(ws,min_col=4+i, min_row=1, max_col=4+i, max_row=Contexte.n_bits+1)
        series = openpyxl.chart.Series(values, xvalues, title_from_data=True)
        series.graphicalProperties.line.width = 12000
        c5.series.append(series) 
    c5.x_axis.title = 'Date'
    c5.y_axis.title = 'Power (kW)'
    c5.layout = openpyxl.chart.layout.Layout(manualLayout=openpyxl.chart.layout.ManualLayout(x=0.0015, y=0.005,w=0.97, h=0.7))
    c5.layout.layoutTarget = "inner"    
    c5.x_axis.delete = True
    c5.y_axis.delete = False
    c5.legend.position = 'l'
    (c5.height,c5.width) = (9,200)
    c5.legend.overlay = False
    c5.x_axis.majorUnit = Contexte.time_resolution*24*7
    c5.x_axis.scaling.min,c5.x_axis.scaling.max =(0,Contexte.n_bits)
    ws.add_chart(c5, "O43")  
    
    set_column_width(ws)
    
    ws = wb['Production']
    cProd = openpyxl.chart.BarChart()
    cProd.title = "Production set ratios"
    cProd.overlap=30
    cProd.add_data(openpyxl.chart.Reference(ws,min_col=3, min_row=1, max_col=3, max_row=len(production_set)+1), titles_from_data=True)
    cProd.set_categories(openpyxl.chart.Reference(ws,min_col=1, min_row=2, max_col=1, max_row=len(production_set)+1))
    cProd.x_axis.title = 'Production ID'
    cProd.y_axis.title = 'Coverage ratio'
    cProd.layout = openpyxl.chart.layout.Layout(manualLayout=openpyxl.chart.layout.ManualLayout(x=0.01, y=0.005,w=0.85, h=0.7))
    cProd.layout.layoutTarget = "inner"  
    cProd.x_axis.delete = False
    cProd.y_axis.delete = False
    cProd.legend = None
    cProd.height = 8
    cProd.width = 26
    #cProd.legend.overlay = False
    cProd.x_axis.tickLblPos = "low"
    cProd.x_axis.tickLblPos = "low"
    cProd.series[0].graphicalProperties.solidFill = "ff9900" 
    cProd.series[0].graphicalProperties.line.solidFill = "00000" 
    cProd.y_axis.scaling.min,cProd.y_axis.scaling.max =(0,1)
    ws.add_chart(cProd, "F2")   
    
    set_column_width(ws)
    
    ws = wb['DSM']
    c6 = openpyxl.chart.LineChart()
    c6.title = "Daily Demand-side management"
    c6.add_data(openpyxl.chart.Reference(ws,min_col=9, min_row=1, max_col=12, max_row=int(Contexte.time_resolution*24)+1), titles_from_data=True)
    c6.set_categories(openpyxl.chart.Reference(ws,min_col=8, min_row=2, max_col=8, max_row=int(Contexte.time_resolution*24)+1))
    c6.x_axis.title = 'Datetime'
    c6.y_axis.title = 'Load (kW)'
    c6.layout = openpyxl.chart.layout.Layout(manualLayout=openpyxl.chart.layout.ManualLayout(x=0.025, y=0.005,w=0.85, h=0.7))
    c6.layout.layoutTarget = "inner"    
    c6.height,c6.width = (10,18)
    c6.x_axis.delete = False
    c6.y_axis.delete = False
    c6.legend.position = 'b'
    c6.legend.overlay = False
    c6.tickLblPos = "low"
    c6.x_axis.tickLblPos = "low"
    ws.add_chart(c6, "D10") 
    
    c7 = openpyxl.chart.LineChart()
    c7.title = "Yearly Demand-side management"
    c7.add_data(openpyxl.chart.Reference(ws,min_col=15, min_row=1, max_col=18, max_row=int(n_days)+1), titles_from_data=True)
    c7.set_categories(openpyxl.chart.Reference(ws,min_col=14, min_row=2, max_col=14, max_row=int(n_days)+1))
    c7.x_axis.title = 'Datetime'
    c7.y_axis.title = 'Load (kW)'
    c7.layout = openpyxl.chart.layout.Layout(manualLayout=openpyxl.chart.layout.ManualLayout(x=0.025, y=0.005,w=0.85, h=0.7))
    c7.layout.layoutTarget = "inner"    
    c7.height,c7.width = (10,18)
    c7.x_axis.delete = False
    c7.y_axis.delete = False
    c7.legend.position = 'b'
    c7.legend.overlay = False
    c7.tickLblPos = "low"
    c7.x_axis.tickLblPos = "low"
    ws.add_chart(c7, "N10") 
    
    set_column_width(ws)
    
    ws = wb['Time_balancing']
    c8 = openpyxl.chart.LineChart()
    c8.title = "Daily balancing (optimization)"
    c8.add_data(openpyxl.chart.Reference(ws,min_col=2, min_row=1, max_col=6, max_row=int(Contexte.time_resolution*24)+1), titles_from_data=True)
    c8.set_categories(openpyxl.chart.Reference(ws,min_col=1, min_row=2, max_col=1, max_row=int(Contexte.time_resolution*24)))
    c8.x_axis.title = 'Datetime'
    c8.y_axis.title = 'Power (kW)'
    c8.layout = openpyxl.chart.layout.Layout(manualLayout=openpyxl.chart.layout.ManualLayout(x=0.025, y=0.005,w=0.85, h=0.7))
    c8.layout.layoutTarget = "inner"    
    c8.height,c8.width = (10,18)
    c8.x_axis.delete = False
    c8.y_axis.delete = False
    c8.legend.position = 'b'
    c8.legend.overlay = False
    c8.tickLblPos = "low"
    c8.x_axis.tickLblPos = "low"
    ws.add_chart(c8, "A10") 
    
    c9 = openpyxl.chart.LineChart()
    c9.title = "Daily balancing (baseline)"
    c9.add_data(openpyxl.chart.Reference(ws,min_col=9, min_row=1, max_col=13, max_row=int(Contexte.time_resolution*24)+1), titles_from_data=True)
    c9.set_categories(openpyxl.chart.Reference(ws,min_col=8, min_row=2, max_col=8, max_row=int(Contexte.time_resolution*24)))
    c9.x_axis.title = 'Datetime'
    c9.y_axis.title = 'Power (kW)'
    c9.layout = openpyxl.chart.layout.Layout(manualLayout=openpyxl.chart.layout.ManualLayout(x=0.025, y=0.005,w=0.85, h=0.7))
    c9.layout.layoutTarget = "inner"    
    c9.height,c9.width = (10,18)
    c9.x_axis.delete = False
    c9.y_axis.delete = False
    c9.legend.position = 'b'
    c9.legend.overlay = False
    c9.tickLblPos = "low"
    c9.x_axis.tickLblPos = "low"
    ws.add_chart(c9, "H10") 
    
    c10 = openpyxl.chart.LineChart()
    c10.title = "Yearly balancing (optimization)"
    c10.add_data(openpyxl.chart.Reference(ws,min_col=16, min_row=1, max_col=20, max_row=int(n_days)+1), titles_from_data=True)
    c10.set_categories(openpyxl.chart.Reference(ws,min_col=15, min_row=2, max_col=15, max_row=int(n_days)))
    c10.x_axis.title = 'Datetime'
    c10.y_axis.title = 'Power (kW)'
    c10.layout = openpyxl.chart.layout.Layout(manualLayout=openpyxl.chart.layout.ManualLayout(x=0.025, y=0.005,w=0.85, h=0.7))
    c10.layout.layoutTarget = "inner"    
    c10.height,c10.width = (10,18)
    c10.x_axis.delete = False
    c10.y_axis.delete = False
    c10.legend.position = 'b'
    c10.legend.overlay = False
    c10.tickLblPos = "low"
    c10.x_axis.tickLblPos = "low"
    ws.add_chart(c10, "O10") 
    
    c11 = openpyxl.chart.LineChart()
    c11.title = "Yearly balancing (baseline)"
    c11.add_data(openpyxl.chart.Reference(ws,min_col=23, min_row=1, max_col=27, max_row=int(n_days)+1), titles_from_data=True)
    c11.set_categories(openpyxl.chart.Reference(ws,min_col=22, min_row=2, max_col=22, max_row=int(n_days)))
    c11.x_axis.title = 'Datetime'
    c11.y_axis.title = 'Power (kW)'
    c11.layout = openpyxl.chart.layout.Layout(manualLayout=openpyxl.chart.layout.ManualLayout(x=0.025, y=0.005,w=0.85, h=0.7))
    c11.layout.layoutTarget = "inner"    
    c11.height,c11.width = (10,18)
    c11.x_axis.delete = False
    c11.y_axis.delete = False
    c11.legend.position = 'b'
    c11.legend.overlay = False
    c11.tickLblPos = "low"
    c11.x_axis.tickLblPos = "low"
    ws.add_chart(c11, "V10") 
    
    set_column_width(ws)
    
    set_column_width(wb['DG'],offset=2)
    
    if  (Contexte.type_optim=='pro'):
        ws = wb['EMS']
    
        c12_1 = openpyxl.chart.LineChart()
        c12_1.title = "Taking over"
        data = openpyxl.chart.Reference(ws, min_col=13, min_row=1, max_col=14, max_row=10)
        c12_1.add_data(data, titles_from_data=True)
        c12_1.set_categories(openpyxl.chart.Reference(ws,min_col=12, min_row=2, max_col=12, max_row=10))
        for i in range(2):
            c12_1.series[i].marker.symbol = "square"
            c12_1.series[i].smooth = False
        c12_1.x_axis.title = 'effective SOC (%)'
        c12_1.y_axis.title = 'Secondary storage involvment (%)'
        c12_1.layout = openpyxl.chart.layout.Layout(manualLayout=openpyxl.chart.layout.ManualLayout(x=0.025, y=0.005,w=0.85, h=0.7))
        c12_1.layout.layoutTarget = "inner"
        c12_1.x_axis.delete = False
        c12_1.y_axis.delete = False
        c12_1.legend.position = 'b'
        c12_1.legend.overlay = False
        c12_1.tickLblPos = "low"
        c12_1.x_axis.tickLblPos = "low"
        c12_1.x_axis.scaling.min,c12_1.y_axis.scaling.min,c12_1.y_axis.scaling.max = (0,0,100)
        ws.add_chart(c12_1, "G20") 
    
        c12_2 = openpyxl.chart.ScatterChart()
        c12_2.title = "D_DSM minimum levels"    
        xvalues=openpyxl.chart.Reference(ws,min_col=3, min_row=2, max_col=3, max_row=25)
        values = openpyxl.chart.Reference(ws, min_col=4, min_row=2, max_col=4, max_row=25)
        series = openpyxl.chart.Series(values, xvalues, title='Minimum level (%)')
        series.marker=openpyxl.chart.marker.Marker('square')
        series.smooth = False
        c12_2.varyColors = False
        c12_2.series.append(series)  
        c12_2.x_axis.title = 'Hour'
        c12_2.y_axis.title = 'minimum level (%)'
        c12_2.layout = openpyxl.chart.layout.Layout(manualLayout=openpyxl.chart.layout.ManualLayout(x=0.025, y=0.005,w=0.85, h=0.7))
        c12_2.layout.layoutTarget = "inner"
        c12_2.x_axis.delete = False
        c12_2.y_axis.delete = False
        c12_2.legend.position = 'b'
        c12_2.legend.overlay = False
        c12_2.tickLblPos = "low"
        c12_2.x_axis.tickLblPos = "low"
        c12_2.x_axis.scaling.min,c12_2.x_axis.scaling.max,c12_2.y_axis.scaling.min,c12_2.y_axis.scaling.max = (1,24,0,100)

        ws.add_chart(c12_2, "P3") 
    
    
        c12_3 = openpyxl.chart.ScatterChart()
        c12_3.title = "Y_DSM minimum levels"
        xvalues=openpyxl.chart.Reference(ws,min_col=6, min_row=2, max_col=6, max_row=13)
        values = openpyxl.chart.Reference(ws, min_col=7, min_row=2, max_col=7, max_row=13)
        series = openpyxl.chart.Series(values, xvalues, title='Minimum level (%)')
        series.marker=openpyxl.chart.marker.Marker('square')
        series.smooth = False
        c12_3.varyColors = False
        c12_3.series.append(series)  
        c12_3.x_axis.title = 'Month'
        c12_3.y_axis.title = 'minimum level (%)'
        c12_3.layout = openpyxl.chart.layout.Layout(manualLayout=openpyxl.chart.layout.ManualLayout(x=0.025, y=0.005,w=0.85, h=0.7))
        c12_3.layout.layoutTarget = "inner"
        c12_3.x_axis.delete = False
        c12_3.y_axis.delete = False
        c12_3.legend.position = 'b'
        c12_3.legend.overlay = False
        c12_3.tickLblPos = "low"
        c12_3.x_axis.tickLblPos = "low"
        c12_3.x_axis.scaling.min,c12_3.x_axis.scaling.max,c12_3.y_axis.scaling.min,c12_3.y_axis.scaling.max = (1,12,0,100)
        ws.add_chart(c12_3, "P20") 
    
        set_column_width(wb['EMS'],offset=2)

    
    
    wb.save(file_name_out)