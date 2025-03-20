# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 16:18:47 2023

@author: jlegalla
"""

import pandas as pd
import numpy as np
import openpyxl              
import copy  

#def find_constraint_levels(constraint_num,Constraint_level,D_Non_movable_load,D_Movable_load,storage_characteristics,prod_C,prods_U,Bounds_prod):
#    min_production=prod_C
#    max_production = prod_C/1000+np.inner(prods_U.T,Bounds_prod)/1000 
#    D_load=np.repeat(0,len(D_Non_movable_load))
    
#    margin=sum(D_Movable_load)
#    day=0
#    while (day<(len(D_Non_movable_load))):
#        D_load[day]=min(max_production[day],margin+D_Non_movable_load[day])  
#        margin=margin-(D_load[day]-D_Non_movable_load[day])
#        print(margin)
#        day=day+1
        

#    D_load=D_load+margin/len(D_Non_movable_load)
    
#    D_Instantaneous_energy=np.minimum(max_production,D_Non_movable_load)
    

#    D_exces=max_production-D_load
#    D_exces[D_exces<0]=0
#    D_default = D_load-max_production
#    D_default[D_default<0]=0
#    D_reportable_energy=max(storage_characteristics[4,:])*D_exces
#    D_max_self_sufficiency = min(1,sum(D_Instantaneous_energy+D_reportable_energy)/sum(D_load))
    ## The max. self-consumption is always 1 because we can always lose artificially energy usong storage
    
#    D_max_self_consumption = 1
#    D_max_constraint_level = D_max_self_sufficiency if constraint_num ==1 else D_max_self_consumption
    #if (self_sufficiency)>D_max_self_sufficiency :
        #print("Self_sufficiency impossible to reach")
#    return(tuple((constraint_num ,Constraint_level,D_max_constraint_level)))

def find_constraint_levels(constraint_num,Constraint_level,D_Non_movable_load,D_Movable_load,storage_characteristics,prod_C,prods_U,Bounds_prod):
    min_production=prod_C
    max_production = prod_C/1000+np.inner(prods_U.T,Bounds_prod)/1000 
    total_load=sum(D_Non_movable_load)+sum(D_Movable_load)
        
    extra_prod = np.where(max_production>D_Non_movable_load,max_production-D_Non_movable_load,0)

    sum_extra_movable_load = max(0,sum(D_Movable_load)-sum(extra_prod))
    
    sum_extra_prod = sum(extra_prod)-sum(D_Movable_load)
    
    sum_extra_non_movable_load = sum(np.where(D_Non_movable_load-max_production>0,D_Non_movable_load-max_production,0))
    
    D_Instantaneous_energy=np.minimum(max_production,D_Non_movable_load)
    

    D_reportable_energy=max(storage_characteristics[4,:])*sum_extra_prod
    D_max_self_sufficiency = min(1,(sum(D_Instantaneous_energy)+D_reportable_energy)/total_load)
    ## The max. self-consumption is always 1 because we can always lose artificially energy usong storage
    
    D_max_self_consumption = 1
    D_max_constraint_level = D_max_self_sufficiency if constraint_num ==1 else D_max_self_consumption
    #if (self_sufficiency)>D_max_self_sufficiency :
        #print("Self_sufficiency impossible to reach")
    return(tuple((constraint_num ,Constraint_level,D_max_constraint_level)))

def compute_grid_prices(datetime,grid_price):
    grid_price=grid_price.replace(np.nan, '', regex=True)
    prices = [pd.DataFrame(data={'Datetime':datetime,'Hour type':'NONE','Price':None}) for i in range(len(grid_price))]
    prices_hour_type = []
    prices_num = []


    fixed_premium = []
    Overrun = []
    Selling_price = []
    
    for i in range(len(grid_price)):
        prices[i].loc[(prices[i]['Datetime'].apply(lambda x: x.dayofweek).isin([5,6])) & (prices[i]['Datetime'].apply(lambda x: str(x.hour)).isin(grid_price.iloc[i]['Weekend peak hours'].split(' '))),'Hour type']='Peak'
        prices[i].loc[(prices[i]['Datetime'].apply(lambda x: x.dayofweek).isin([5,6])) & (prices[i]['Datetime'].apply(lambda x: str(x.hour)).isin(grid_price.iloc[i]['Weekend full hours'].split(' '))),'Hour type']='WE full'
        prices[i].loc[(prices[i]['Datetime'].apply(lambda x: x.dayofweek).isin([5,6])) & (prices[i]['Datetime'].apply(lambda x: str(x.hour)).isin(grid_price.iloc[i]['Weekend off-peak hours'].split(' '))),'Hour type']='WE off'
        prices[i].loc[(prices[i]['Datetime'].apply(lambda x: x.dayofweek).isin([0,1,2,3,4])) & (prices[i]['Datetime'].apply(lambda x: str(x.hour)).isin(grid_price.iloc[i]['Workday peak hours'].split(' '))),'Hour type']='Peak'
        prices[i].loc[(prices[i]['Datetime'].apply(lambda x: x.dayofweek).isin([0,1,2,3,4])) & (prices[i]['Datetime'].apply(lambda x: str(x.hour)).isin(grid_price.iloc[i]['Workday full hours'].split(' '))),'Hour type']='W full'
        prices[i].loc[(prices[i]['Datetime'].apply(lambda x: x.dayofweek).isin([0,1,2,3,4])) & (prices[i]['Datetime'].apply(lambda x: str(x.hour)).isin(grid_price.iloc[i]['Workday off-peak hours'].split(' '))),'Hour type']='W off'
    
        prices[i].loc[prices[i]['Hour type']=='Peak' ,'Price']=grid_price.iloc[i]['Peak (c€/kWh)']
        prices[i].loc[(prices[i]['Hour type']=='WE full') & (prices[i]['Datetime'].apply(lambda x: str(x.month)).isin(grid_price.iloc[i]['Summer months'].split(' '))) ,'Price']=grid_price.iloc[i]['Summer full hours (c€/kWh)']
        prices[i].loc[(prices[i]['Hour type']=='W full') & (prices[i]['Datetime'].apply(lambda x: str(x.month)).isin(grid_price.iloc[i]['Summer months'].split(' '))) ,'Price']=grid_price.iloc[i]['Summer full hours (c€/kWh)']
        prices[i].loc[(prices[i]['Hour type']=='WE off') & (prices[i]['Datetime'].apply(lambda x: str(x.month)).isin(grid_price.iloc[i]['Summer months'].split(' '))) ,'Price']=grid_price.iloc[i]['Summer off-peak hours (c€/kWh)']
        prices[i].loc[(prices[i]['Hour type']=='W off') & (prices[i]['Datetime'].apply(lambda x: str(x.month)).isin(grid_price.iloc[i]['Summer months'].split(' '))) ,'Price']=grid_price.iloc[i]['Summer off-peak hours (c€/kWh)']
        prices[i].loc[(prices[i]['Hour type']=='WE full') & (prices[i]['Datetime'].apply(lambda x: str(x.month)).isin(grid_price.iloc[i]['Winter months'].split(' '))) ,'Price']=grid_price.iloc[i]['Winter full hours (c€/kWh)']
        prices[i].loc[(prices[i]['Hour type']=='W full') & (prices[i]['Datetime'].apply(lambda x: str(x.month)).isin(grid_price.iloc[i]['Winter months'].split(' '))) ,'Price']=grid_price.iloc[i]['Winter full hours (c€/kWh)']
        prices[i].loc[(prices[i]['Hour type']=='WE off') & (prices[i]['Datetime'].apply(lambda x: str(x.month)).isin(grid_price.iloc[i]['Winter months'].split(' '))) ,'Price']=grid_price.iloc[i]['Winter off-peak hours (c€/kWh)']
        prices[i].loc[(prices[i]['Hour type']=='W off') & (prices[i]['Datetime'].apply(lambda x: str(x.month)).isin(grid_price.iloc[i]['Winter months'].split(' '))) ,'Price']=grid_price.iloc[i]['Winter off-peak hours (c€/kWh)']

        prices[i]['Price']=prices[i]['Price']*(1+grid_price.iloc[i]['TVA load'])
        prices[i]['Price']+=grid_price.iloc[i]['CSPE (c€/kWh)']
        prices[i]['Price']=prices[i]['Price']/100
        prices_hour_type.append(prices[i]['Hour type'])
        prices_num.append(prices[i]['Price'])

        fixed_premium.append( grid_price.iloc[i]['Fixed premium (€/kW)'])
        Overrun.append( grid_price.iloc[i]['Power overrun (€/kW)'])
        Selling_price.append( grid_price.iloc[i]['Selling price (€/kWh)'])
        
        #Converting into Numpy
    fixed_premium=np.array(fixed_premium,dtype=np.float64)
    Overrun=np.array(Overrun,dtype=np.float64)
    Selling_price=np.array(Selling_price,dtype=np.float64)
    prices_num=np.float64(np.vstack(prices_num))
    prices_hour_type=np.vstack(prices_hour_type).astype('U')

    return([prices_hour_type,prices_num,fixed_premium,Overrun,Selling_price])

def read_data(Data) :
    specs_num=Data['Production_specs'].drop('Id',axis=1).to_numpy(dtype=np.float64)
    specs_Id=Data['Production_specs']['Id'].to_numpy(dtype='U')
    specs_names=Data['Production_specs'].columns[1:].to_numpy(dtype='U')
    prod_C=(Data['TimeSeries']['Current_production (W)'].to_numpy(dtype=np.float64))
    prod_C[prod_C<0]=0
    prods_U = Data['Unit_productions'].drop('Datetime',axis=1)
    prods_U=np.array(prods_U,dtype=np.float64).T
    Volums_prod = np.sum(prods_U,axis=1)
    
    Non_movable_load = np.array(Data['TimeSeries']['Non-controllable load (kW)'],dtype=np.float64)
    Y_movable_load = np.array(Data['TimeSeries']['Yearly movable load (kW)'],dtype=np.float64)
    D_movable_load = np.array(Data['TimeSeries']['Daily movable load (kW)'],dtype=np.float64)
    
    grid_price = Data['Grid_prices']
    Contract_Id = grid_price['Contract_Id']
    datetime=Data['TimeSeries']['Datetime']
    datetime=pd.to_datetime(datetime).round('1s')

    time_resolution = (datetime.iloc[-1].round(freq='min')-datetime.iloc[1].round(freq='min'))/(len(datetime)-2)
    time_resolution = np.float64(3600/time_resolution.seconds)
    Constraint = Data['Environment']['Constraint'][0]
        
    if Constraint== 'Self-sufficiency':
            constraint_num = np.int64(1) 
    elif Constraint== 'Self-consumption':
            constraint_num = np.int64(2)
    else : 
            constraint_num = np.int64(0)
                
    Constraint_level = np.float64(Data['Environment']['Constraint level'][0])
    criterion = Data['Environment']['Optimisation criterion'][0]
    
    if criterion== 'LCOE':
            criterion_num = np.int64(1 )
    elif criterion== 'Self-consumption':
            criterion_num = np.int64(2)
    elif criterion== 'Maximum power from grid':
            criterion_num = np.int64(3)
    elif criterion== 'Energy losses':
            criterion_num = np.int64(4)
    elif criterion== 'Capacity factor':
            criterion_num = np.int64(5)
    elif criterion== 'Autonomy':
            criterion_num = np.int64(6)
    elif criterion== 'Saved CO2 Emissions':
            criterion_num = np.int64(7)
    elif criterion== 'Saved fossil fuel consumption':
            criterion_num = np.int64(8)
    elif criterion== 'EROI':
            criterion_num = np.int64(9)
    else : 
            criterion_num = np.int64(0)
    
    
    storages = Data['Storages']
    Cost_power = storages['PCS cost (€/kW)']+storages['BOP cost (€/kW)']
    storage_characteristics = storages.T[1:].to_numpy(dtype=np.float64)
    storage_techs = storages['Technology'].to_numpy(dtype='U')
    storage_characteristics = np.vstack((storage_characteristics,Cost_power),dtype=np.float64)
    storage_characteristics_names = np.hstack((storages.columns[1:].to_numpy(),'Cost power'))

    Bounds_prod = np.int64(specs_num[:,np.where(specs_names=='Capacity')[0][0]])
    n_UP=len(Bounds_prod)
    duration_years = np.float64(len(prod_C)/time_resolution/8760)
    
    (Grid_Fossil_fuel_ratio,Main_grid_emissions,Main_grid_ratio) = (np.float64(Data['Environment']['Main grid fossil fuel ratio'][0]),np.float64(Data['Environment']['Main grid emissions (gCO2/kWh)'][0]),np.float64(Data['Environment']['Main grid ratio primary over final energy'][0]))
   
    
    r_cross_init=np.float64(Data['Hyperparameters']['Initialisation values'][0])
    n_iter_init=np.int64(Data['Hyperparameters']['Initialisation values'][1])
    r_cross=np.float64(Data['Hyperparameters']['Evolution values'][0])
    n_iter,nb_ere,n_pop,n_nodes,n_core =np.int64(Data['Hyperparameters']['Evolution values'][1:6])
    Cost_sequence = np.float64(Data['Hyperparameters']['Constraint costs'])
    operator_contract = np.float64(Data['Hyperparameters']['Contract'])
    operator_production = np.float64(Data['Hyperparameters']['Production'])
    operator_storage_volumes = np.float64(Data['Hyperparameters']['Storage volume'])
    operator_storage_use = np.float64(Data['Hyperparameters']['Storage use'])
    operator_storage_power = np.float64(Data['Hyperparameters']['Storage power'])
    operator_scheduling_consistency = np.float64(Data['Hyperparameters']['Scheduling consistency'])
    operator_storage_timeserie = np.float64(Data['Hyperparameters']['Storage timeserie'])
    operator_storage_opposite = np.float64(Data['Hyperparameters']['Storage opposite moves'])
    operator_storage_transfers = np.float64(Data['Hyperparameters']['Storage transfers'])
    operator_storage_specification = np.float64(Data['Hyperparameters']['Storage specification'])
    operator_curve_smoothing = np.float64(Data['Hyperparameters']['Curve smoothing'])
    operator_constraint = np.float64(Data['Hyperparameters']['Constraint'])
    operator_long_term_consistency = np.float64(Data['Hyperparameters']['Long-term consistency'])
    operator_YDSM = np.float64(Data['Hyperparameters']['Yearly demand-side management'])
    operator_DDSM = np.float64(Data['Hyperparameters']['Daily demand-side management'])
    hyperparameters_main = {'Initialisation':(n_core,n_nodes,r_cross_init,n_pop,n_iter_init),'Evolution':(n_core,n_nodes,r_cross,n_pop,n_iter,nb_ere)}
    #hyperparameters_operators = {'Contract':operator_contract,'Production':operator_production,'Storage volume':operator_storage_volumes,'Storage power':operator_storage_power,'Storage use':operator_storage_use,'Storage timeserie':operator_storage_timeserie,'Scheduling consistency':operator_scheduling_consistency,'Storage opposite':operator_storage_opposite,'Storage transfer':operator_storage_transfers,'Long-term consistency':operator_long_term_consistency ,'Curve smoothing':operator_curve_smoothing,'Storage specification':operator_storage_specification,'Constraint':operator_constraint,'Y_DSM':operator_YDSM,'D_DSM':operator_DDSM}
    hyperparameters_operators_names = np.array(('Contract','Production','Storage volume','Storage use','Storage power','Scheduling consistency','Storage timeserie','Storage opposite moves','Storage transfers','Storage specification','Curve smoothing','Constraint','Long-term consistency','Yearly demand-side management','Daily demand-side management'))
    hyperparameters_operators_num = np.array((operator_contract,operator_production,operator_storage_volumes,operator_storage_use,operator_storage_power,operator_scheduling_consistency,operator_storage_timeserie,operator_storage_opposite,operator_storage_transfers,operator_storage_specification,operator_curve_smoothing,operator_constraint,operator_long_term_consistency,operator_YDSM,operator_DDSM)).T

    [prices_hour_type,prices_num,fixed_premium,Overrun,Selling_price] = compute_grid_prices(datetime,grid_price)
    return (datetime,Grid_Fossil_fuel_ratio,Main_grid_emissions,Main_grid_ratio,specs_names,specs_Id,specs_num,prod_C,prods_U,Volums_prod,Y_movable_load,D_movable_load,Non_movable_load,time_resolution,constraint_num, Constraint_level ,criterion_num,storage_techs,storage_characteristics_names,storage_characteristics,Bounds_prod,n_UP,duration_years,prices_hour_type,prices_num,fixed_premium,Overrun,Selling_price,Contract_Id,hyperparameters_main,hyperparameters_operators_names,hyperparameters_operators_num,Cost_sequence)    

def NONJIT_adaptation_hyperparameters_initialisation(hyperparameters_operators):
    adaptated_hyperparameters_operators=copy.deepcopy(hyperparameters_operators)
    adaptated_hyperparameters_operators['Contract'][0]=1*adaptated_hyperparameters_operators['Contract'][0]
    adaptated_hyperparameters_operators['Production'][0]=1*adaptated_hyperparameters_operators['Production'][0]
    adaptated_hyperparameters_operators['Storage volume'][0]=0*adaptated_hyperparameters_operators['Storage volume'][0]
    adaptated_hyperparameters_operators['Storage power'][0]=0*adaptated_hyperparameters_operators['Storage power'][0]
    adaptated_hyperparameters_operators['Storage use'][0]=1*adaptated_hyperparameters_operators['Storage use'][0]
    adaptated_hyperparameters_operators['Storage timeserie'][0]=1*adaptated_hyperparameters_operators['Storage timeserie'][0]
    adaptated_hyperparameters_operators['Storage opposite'][0]=1*adaptated_hyperparameters_operators['Storage opposite'][0]
    adaptated_hyperparameters_operators['Storage transfer'][0]=1*adaptated_hyperparameters_operators['Storage transfer'][0]
    adaptated_hyperparameters_operators['Long-term consistency'][0]=0*adaptated_hyperparameters_operators['Long-term consistency'][0]
    adaptated_hyperparameters_operators['Scheduling consistency'][0]=0.2*adaptated_hyperparameters_operators['Scheduling consistency'][0]
    adaptated_hyperparameters_operators['Storage specification'][0]=0*adaptated_hyperparameters_operators['Storage specification'][0]
    adaptated_hyperparameters_operators['Curve smoothing'][0]=0.7*adaptated_hyperparameters_operators['Curve smoothing'][0]
    adaptated_hyperparameters_operators['Y_DSM'][0]=adaptated_hyperparameters_operators['Y_DSM'][0]
    adaptated_hyperparameters_operators['D_DSM'][0]=adaptated_hyperparameters_operators['D_DSM'][0]

    return (adaptated_hyperparameters_operators)

def adaptation_hyperparameters_initialisation(hyperparameters_operators_num,hyperparameters_operators_names):
    adaptated_hyperparameters_operators=copy.deepcopy(hyperparameters_operators_num)
    adaptated_hyperparameters_operators[0,hyperparameters_operators_names=='Contract']=1*hyperparameters_operators_num[:,hyperparameters_operators_names=='Contract'][0]
    adaptated_hyperparameters_operators[0,hyperparameters_operators_names=='Production']=1*hyperparameters_operators_num[:,hyperparameters_operators_names=='Production'][0]
    adaptated_hyperparameters_operators[0,hyperparameters_operators_names=='Storage volume']=1*hyperparameters_operators_num[:,hyperparameters_operators_names=='Storage volume'][0][0]
    adaptated_hyperparameters_operators[0,hyperparameters_operators_names=='Storage power']=0.2*hyperparameters_operators_num[:,hyperparameters_operators_names=='Storage power'][0]
    #adaptated_hyperparameters_operators[1,hyperparameters_operators_names=='Storage power']=40
    adaptated_hyperparameters_operators[0,hyperparameters_operators_names=='Storage use']=1*hyperparameters_operators_num[:,hyperparameters_operators_names=='Storage use'][0]
    adaptated_hyperparameters_operators[0,hyperparameters_operators_names=='Storage timeserie']=1*hyperparameters_operators_num[:,hyperparameters_operators_names=='Storage timeserie'][0]
    adaptated_hyperparameters_operators[0,hyperparameters_operators_names=='Storage opposite']=1*hyperparameters_operators_num[:,hyperparameters_operators_names=='Storage opposite'][0]
    adaptated_hyperparameters_operators[0,hyperparameters_operators_names=='Storage transfers']=1*hyperparameters_operators_num[:,hyperparameters_operators_names=='Storage transfers'][0]
    adaptated_hyperparameters_operators[1,hyperparameters_operators_names=='Storage transfers']=5
    adaptated_hyperparameters_operators[0,hyperparameters_operators_names=='Long-term consistency']=0*hyperparameters_operators_num[:,hyperparameters_operators_names=='Long-term consistency'][0]
    adaptated_hyperparameters_operators[0,hyperparameters_operators_names=='Scheduling consistency']=0.5*hyperparameters_operators_num[:,hyperparameters_operators_names=='Scheduling consistency'][0]
    adaptated_hyperparameters_operators[0,hyperparameters_operators_names=='Storage specification']=1*hyperparameters_operators_num[:,hyperparameters_operators_names=='Storage specification'][0]
    adaptated_hyperparameters_operators[1,hyperparameters_operators_names=='Storage specification']=1
    adaptated_hyperparameters_operators[0,hyperparameters_operators_names=='Constraint']=1*hyperparameters_operators_num[:,hyperparameters_operators_names=='Constraint'][0]
    adaptated_hyperparameters_operators[1,hyperparameters_operators_names=='Constraint']=20
    adaptated_hyperparameters_operators[0,hyperparameters_operators_names=='Curve smoothing']=0.7*hyperparameters_operators_num[:,hyperparameters_operators_names=='Curve smoothing'][0]
    adaptated_hyperparameters_operators[0,hyperparameters_operators_names=='Y_DSM']=hyperparameters_operators_num[:,hyperparameters_operators_names=='Y_DSM'][0]
    adaptated_hyperparameters_operators[0,hyperparameters_operators_names=='D_DSM']=hyperparameters_operators_num[:,hyperparameters_operators_names=='D_DSM'][0]

    return (adaptated_hyperparameters_operators)
    

def as_text(value):
    if value is None:
        return ""
    return str(value)

def set_column_width(ws):
    dim_holder = openpyxl.worksheet.dimensions.DimensionHolder(worksheet=ws)
    j=0

    for col in ws.columns:
        j+=1
        length = max(len(as_text(cell.value)) for cell in col)
        dim_holder[openpyxl.utils.get_column_letter(j)] = openpyxl.worksheet.dimensions.ColumnDimension(ws, min=j, max=j, width=length)

    ws.column_dimensions = dim_holder

def post_traitement(solution,datetime,cost_function,cost_base,prod_C, prods_U,Non_movable_load,D_movable_load,Y_movable_load,characteristics,storage_techs,time_resolution,Main_grid_emissions,Grid_Fossil_fuel_ratio,Main_grid_ratio,specs_num,specs_Id,duration_years,prices_num,fixed_premium,Overrun,Selling_price,Contract_Id,Bounds_prod,constraint_num,Constraint_level,Data,n_days,file_name_out,cost_constraint):
    
    n_store = len(storage_techs)
    
    production_set=solution.production_set
    prod = np.sum(np.array([production_set[i]*prods_U[i,:] for i in range(len(production_set))])/1000,axis=0)+prod_C/1000
    
    TS_prod = pd.DataFrame(data={'Id':specs_Id,'Capacity':np.int64(specs_num[:,3]),'Installed':production_set},index=None)
    
    load=Non_movable_load+solution.Y_DSM+solution.D_DSM.flatten()
    load_unchanged = Non_movable_load+Y_movable_load+D_movable_load
    
    signal=load-prod
    cost = cost_function(solution,characteristics,time_resolution,n_store,duration_years,specs_num,prices_num,fixed_premium,Overrun,Selling_price,Non_movable_load,Main_grid_emissions,prod_C,prods_U,Bounds_prod,constraint_num,Constraint_level,cost_constraint)   
    costs_without = [cost_base(prod_C,time_resolution,duration_years,prices_num[i],fixed_premium[i],Overrun[i],Selling_price[i],Non_movable_load,D_movable_load,Y_movable_load) for i in range(len(Contract_Id))]
    cost_without=costs_without[np.argmin(costs_without[i]['LCOE (€/kWh)'] for i in range(len(costs_without)))]
    Contract_name=Contract_Id[solution.contract]
    Contract_name_without=Contract_Id[np.argmin(costs_without[i]['LCOE (€/kWh)'] for i in range(len(costs_without)))]
    Contract_power = cost['Contract power (kW)']
    Contract_power_without = cost_without['Contract power (kW)']
    
    losses = pd.DataFrame([solution.storage_TS[i]/characteristics[4,:][i]-solution.storage_TS[i] for i in range(n_store)],index=[storage_techs[i]+' losses' for i in range(n_store)],columns=None).transpose()
    losses[losses<0]=0
    sum_losses=sum(np.sum(losses,axis=0))/time_resolution
    power_storages = pd.DataFrame([solution.storage_TS[i] for i in range(n_store)],index=[storage_techs[i]+' power' for i in range(n_store)],columns=None).transpose()
    power_storage = power_storages.sum(axis=1)
    
    sum_diff_storages = [-np.cumsum(solution.storage_TS[i]/time_resolution+losses.iloc[:,i]/time_resolution) for i in range(n_store)]
    energy_storages = [(max(sum_diff_storages[i]) - min(sum_diff_storages[i]))/characteristics[5,:][i] for i in range(n_store)]
    storage_NULL=tuple(energy_storages[i]==0 for i in range(n_store))
    powers_out = [max(solution.storage_TS[i]) for i in range(n_store)]
    powers_in = [-min(solution.storage_TS[i]) for i in range(n_store)]
    Techs = pd.DataFrame(data={'Technology':storage_techs,'Energy sizing (kWh)':energy_storages,'power in (kW)':powers_in,'power out (kW)':powers_out})
    
    minSOCs=[(1-characteristics[5,:][i])/2 for i in range(n_store)]
    SOCs=pd.DataFrame([(sum_diff_storages[i]-min(sum_diff_storages[i]))/energy_storages[i] + minSOCs[i] for i in range(n_store)],index=[storage_techs[i]+' SOC' for i in range(n_store)],columns=None).transpose()
    Grid_trading = load-prod-np.sum(solution.storage_TS,axis=0)
     
    obtained_self_sufficiency = (1-sum(Grid_trading[Grid_trading>0])/sum(load))
    self_consumption = (1+sum(Grid_trading[Grid_trading<0])/sum(prod))
    Autonomy = 1-sum(Grid_trading>0)/len(solution.storage_TS[0])
    Capacity_factor = sum(prod)/(max(prod)*len(solution.storage_TS[0]))
    Max_power_from_grid = max(Grid_trading) 
    
    #Technical
    Annual_cycle_number_storages=np.divide([sum(abs(power_storages.iloc[:,i]))/time_resolution for i in range(n_store)],energy_storages)/duration_years
    State_SOCs=[np.searchsorted([(i+1)/100 for i in range(99)], SOCs.iloc[:,j]) for j in range(n_store)]
    distribution_Depth_of_discharge=[[np.count_nonzero(State_SOCs[i]==j)/len(State_SOCs[i]) for j in range(100)] for i in range(n_store)]
    if sum(storage_NULL)>0:
        distribution_Depth_of_discharge[np.where(storage_NULL)[0][0]]=np.repeat(np.nan,100)
    size_power=max(powers_in,powers_out)
    
    EROI_production = sum(solution.production_set/sum(solution.production_set)*specs_num[:,5])
    
    #Economics  
    Lifetime = tuple(min(characteristics[7,:][i],np.divide(characteristics[8,:],Annual_cycle_number_storages)[i]) for i in range(n_store))
    importation = load-prod-power_storage
    importation [importation<0]=0
    sum_importation=sum(importation)/time_resolution
    
    exportation = -(load-prod-power_storage)
    exportation [exportation<0]=0
    sum_exportation=sum(exportation)/time_resolution
       
    saved_fossil_fuel_consumption = Grid_Fossil_fuel_ratio*sum(prod)/time_resolution  #Attention§ Ici on néglige les pertes dues au transport du réseau vers le campus !!
    
    annual_production_emissions = sum_importation*Main_grid_emissions/1000000/duration_years+sum(sum(np.multiply(specs_num[:,4],np.multiply(production_set,prods_U.T/1000))))/time_resolution/duration_years/1000000
    base_C02_emissions = Main_grid_emissions*sum(load-prod_C)/1000000/duration_years/time_resolution
    saved_CO2_emissions = (base_C02_emissions-annual_production_emissions)*duration_years
    ratio_primary_final_energy = obtained_self_sufficiency*sum(prod)/((sum(prod)-sum_losses))+(1-obtained_self_sufficiency)*Main_grid_ratio

    outputs_TS = pd.DataFrame(data={'Datetime':datetime,'Load (kW)':load,'Power production (kW)':prod},index=None)
    outputs_TS = outputs_TS.join(power_storages)
    outputs_TS = outputs_TS.join(losses)
    outputs_TS = outputs_TS.join(pd.DataFrame(data={'Grid power (kW)':Grid_trading,'Grid price (€/kWh)':prices_num[solution.contract],'Imbalance (kW)':prod + power_storage + Grid_trading - load}))
    outputs_TS = outputs_TS.join(SOCs)
    
    Technical_outputs = {'LCOE (€/kWh)':cost['LCOE (€/kWh)'],'Self-sufficiency':obtained_self_sufficiency,'Self-consumption':self_consumption,'Autonomy':Autonomy,"Capacity factor":Capacity_factor,"Max. power from grid (kW)":Max_power_from_grid,'saved_fossil_fuel_consumption (kWh)':saved_fossil_fuel_consumption,'saved_CO2_emissions (tCO2)':saved_CO2_emissions,'Ratio primary over final energy':ratio_primary_final_energy,'Losses (kWh)':sum_losses}
    Technical_outputs= pd.DataFrame(data=Technical_outputs,index = ['Technicals'])
    
    list_index = storage_techs.copy()
    
    
    dist_DOD = pd.DataFrame(data={'SOC Percentile':[i/100 for i in range(101)]})
    for i in range(n_store):
        dist_DOD = dist_DOD.join(pd.DataFrame(data={'SOC distribution ' + list_index[i]:distribution_Depth_of_discharge[i]}))
        
    cost_final = pd.DataFrame(data = {'Scenario':'Optimisation','LCOE (€/kWh)':cost['LCOE (€/kWh)'],'Contract':Contract_name,'Contract power (kW)':Contract_power,'Energy importation (€/kWh)': cost['Energy importation (€/kWh)'],'Fixed premium (€/kWh)':cost['Fixed premium (€/kWh)'],'Overrun penalty (€/kWh)': cost['Overrun penalty (€/kWh)'],'Energy exportation (€/kWh)': cost['Energy exportation (€/kWh)']},index=['Optimization scenario'])
    
    cost_final = cost_final.join(pd.DataFrame(data={'CAPEX production (€/kWh)':cost['CAPEX production (€/kWh)']},index=['Optimization scenario']))
    cost_final = cost_final.join(pd.DataFrame(data={'OPEX production (€/kWh)':cost['OPEX production (€/kWh)']},index=['Optimization scenario']))

    cost_final_without = pd.DataFrame(data = {'Scenario':'Base','LCOE (€/kWh)':cost_without['LCOE (€/kWh)'],'Contract':Contract_name_without,'Contract power (kW)':Contract_power_without,'Energy importation (€/kWh)': cost_without['Energy importation (€/kWh)'],'Fixed premium (€/kWh)':cost_without['Fixed premium (€/kWh)'],'Overrun penalty (€/kWh)': cost_without['Overrun penalty (€/kWh)'],'Energy exportation (€/kWh)': cost['Energy exportation (€/kWh)']},index=['Base scenario'])
    
    for i in range(n_store):
        cost_final = cost_final.join(pd.DataFrame(data={'CAPEX '+list_index[i]+' (€/kWh)':cost['CAPEX storage (€/kWh)'][i]},index=['Optimization scenario']))
        cost_final = cost_final.join(pd.DataFrame(data={'OPEX '+list_index[i]+' (€/kWh)':cost['OPEX storage (€/kWh)'][i]},index=['Optimization scenario']))
        cost_final_without = cost_final_without.join(pd.DataFrame(data={'CAPEX '+list_index[i]+' (€/kWh)':0.0},index=['Optimization scenario']))
        cost_final_without = cost_final_without.join(pd.DataFrame(data={'OPEX '+list_index[i]+' (€/kWh)':0.0},index=['Optimization scenario']))
        
    cost_final = cost_final.join(pd.DataFrame(data={'Initial investment (€)':cost['Initial investment (€)']},index=['Optimization scenario']))
    cost_final_without = cost_final_without.join(pd.DataFrame(data={'Initial investment (€)':0},index=['Base scenario']))    

    cost_final_merged = pd.concat([cost_final,cost_final_without])
    cost_final_merged = cost_final_merged.fillna(0)
    
        
    Output_production = pd.DataFrame(data={'ID':Data['Production_specs']['Id'],'Number of units':production_set,'Coverage ratio':production_set/Data['Production_specs']['Capacity'],'Initial investment (€)':np.multiply(Data['Production_specs']['Capital unit cost (€)'],production_set)})
    
    Overview_outputs = pd.DataFrame(data={'Total load (kWh)':sum(load)/time_resolution,'Total production (kWh)':sum(prod)/time_resolution,'Total importation (kWh)':sum_importation,'Total exportation (kWh)':sum_exportation},index=['Flows'])
    for i in range(n_store):
        Overview_outputs = Overview_outputs.join(pd.DataFrame(data={'Stored energy ('+list_index[i]+') (kWh)':-sum(power_storages.iloc[:,i][power_storages.iloc[:,i]<0])/time_resolution},index=['Flows']))
    for i in range(n_store):
        Overview_outputs = Overview_outputs.join(pd.DataFrame(data={'Reported energy ('+list_index[i]+') (kWh)':sum(power_storages.iloc[:,i][power_storages.iloc[:,i]>0])/time_resolution},index=['Flows']))
    for i in range(n_store):
        Overview_outputs = Overview_outputs.join(pd.DataFrame(data={'Loss energy ('+list_index[i]+') (kWh)':np.sum(losses,axis=0).iloc[i]/time_resolution},index=['Flows']))
    
    logicals_prod = np.where((Grid_trading<=0) & (signal<=0))[0]
    logicals_loads = np.where((Grid_trading>=0) & (signal>=0))[0]
    illogicals_prod = np.where((Grid_trading>0) & (signal<=0))[0]
    illogicals_loads = np.where((Grid_trading<0) & (signal>=0))[0]
    
    #useprod = pd.DataFrame(data=tuple(-sum(solution[2][i][logicals_prod])/time_resolution for i in range(n_store))+(sum(load[logicals_prod])/time_resolution+sum(load[illogicals_prod])/time_resolution+sum(prod[logicals_loads])/time_resolution+sum(prod[illogicals_loads])/time_resolution,-sum(Grid_trading[logicals_prod])/time_resolution,sum(prod)/time_resolution),index=['Storage '+list_index[i]+' (kWh)' for i in range(n_store)]+['Simultaneous load (kWh)','Grid export (kWh)','Total prod (kWh)']).transpose()
    #Loadmeet = pd.DataFrame(data=tuple(sum(solution[2][i][logicals_loads])/time_resolution  for i in range(n_store))+(sum(prod[logicals_loads])/time_resolution+sum(prod[illogicals_loads])/time_resolution+sum(load[logicals_prod])/time_resolution+sum(load[illogicals_prod])/time_resolution,sum(Grid_trading[logicals_loads])/time_resolution,sum(load)/time_resolution),index=['Storage '+list_index[i]+' (kWh)' for i in range(n_store)]+['Simultaneous production (kWh)','Grid import (kWh)','Total load (kWh)']).transpose()
    
    useprod = pd.DataFrame(data=(sum(tuple(-sum(solution.storage_TS[i][logicals_prod])/time_resolution for i in range(n_store)))+sum(prod[illogicals_prod])/time_resolution-sum(load[illogicals_prod])/time_resolution,sum(load[logicals_prod])/time_resolution+sum(load[illogicals_prod])/time_resolution+sum(prod[logicals_loads])/time_resolution+sum(prod[illogicals_loads])/time_resolution,-sum(Grid_trading[logicals_prod])/time_resolution,sum(prod)/time_resolution),index=['Storage (kWh)' ,'Simultaneous load (kWh)','Grid export (kWh)','Total prod (kWh)']).transpose()
    Loadmeet = pd.DataFrame(data=(sum(tuple(sum(solution.storage_TS[i][logicals_loads])/time_resolution  for i in range(n_store)))+sum(load[illogicals_loads])/time_resolution-sum(prod[illogicals_loads])/time_resolution,sum(prod[logicals_loads])/time_resolution+sum(prod[illogicals_loads])/time_resolution+sum(load[logicals_prod])/time_resolution+sum(load[illogicals_prod])/time_resolution,sum(Grid_trading[logicals_loads])/time_resolution,sum(load)/time_resolution),index=['Storage (kWh)', 'Simultaneous production (kWh)','Grid import (kWh)','Total load (kWh)']).transpose()

    when_prod = pd.DataFrame(data=(sum(load[logicals_prod])/time_resolution+sum(load[illogicals_prod])/time_resolution,-sum(Grid_trading[(signal<0) & (Grid_trading<0) ])/time_resolution,sum(Grid_trading[(signal<0) & (Grid_trading>0) ])/time_resolution) + tuple(-sum(solution.storage_TS[i][(signal<0) & (solution.storage_TS[i]<0)])/time_resolution for i in range(n_store)) + tuple(sum(solution.storage_TS[i][(signal<0) & (solution.storage_TS[i]>0)])/time_resolution for i in range(n_store)),index=['Load (kWh)','Grid export (kWh)','Grid import (kWh)']+['Storage '+list_index[i]+' charge (kWh)' for i in range(n_store)]+['Storage '+list_index[i]+' discharge (kWh)' for i in range(n_store)]).transpose()
    when_load = pd.DataFrame(data=(sum(prod[logicals_loads])/time_resolution+sum(prod[illogicals_loads])/time_resolution,-sum(Grid_trading[(signal>0) & (Grid_trading<0) ])/time_resolution,sum(Grid_trading[(signal>0) & (Grid_trading>0) ])/time_resolution) + tuple(-sum(solution.storage_TS[i][(signal>0) & (solution.storage_TS[i]<0)])/time_resolution for i in range(n_store)) + tuple(sum(solution.storage_TS[i][(signal>0) & (solution.storage_TS[i]>0)])/time_resolution for i in range(n_store)),index=['Production (kWh)','Grid export (kWh)','Grid import (kWh)']+['Storage '+list_index[i]+' charge (kWh)' for i in range(n_store)]+['Storage '+list_index[i]+' discharge (kWh)' for i in range(n_store)]).transpose()
    Grid_use_export = pd.DataFrame(data=(useprod['Grid export (kWh)'], sum_exportation-useprod['Grid export (kWh)']),index=['Production export (kWh)','Storage discharge(kWh)']).transpose()
    Grid_use_import = pd.DataFrame(data=(Loadmeet['Grid import (kWh)'], sum_importation-Loadmeet['Grid import (kWh)']),index=['To meet load (kWh)','Storage charge(kWh)']).transpose()
  
    #Demand-side management
    indexes_hour = [[int((i+j*time_resolution*24)) for j in range(int(n_days))] for i in range(int(time_resolution*24))]
    Daily_base_load = [np.mean(Non_movable_load[indexes_hour[j]]) for j in range(int(time_resolution*24))]
    Daily_movable_loads = [np.mean(D_movable_load[indexes_hour[j]]+Y_movable_load[indexes_hour[j]]) for j in range(int(time_resolution*24))]
    Daily_final_loads = [np.mean(load[indexes_hour[j]]) for j in range(int(time_resolution*24))]
    Daily_prod = [np.mean(prod[indexes_hour[j]]) for j in range(int(time_resolution*24))]
    
    indexes_days = [[int(i+j*time_resolution*24) for i in range(int(time_resolution*24))] for j in range(int(n_days))]
    Yearly_base_load = [np.mean(Non_movable_load[indexes_days[j]]+D_movable_load[indexes_days[j]]) for j in range(int(n_days))]
    Yearly_movable_loads = [np.mean(Y_movable_load[indexes_days[j]]) for j in range(int(n_days))]
    Yearly_final_loads = [np.mean(load[indexes_days[j]]) for j in range(int(n_days))]
    Yearly_prod = [np.mean(prod[indexes_days[j]]) for j in range(int(n_days))]
    
    DSM_daily_strategy = pd.DataFrame(data={'Datetime':datetime[0:int(time_resolution*24)].apply(lambda x: x.strftime('%H:%M')),'Daily base load (kW)':Daily_base_load,'Daily movable load (kW)':Daily_movable_loads,'Daily final load (kW)':Daily_final_loads,'Daily prod (kW)':Daily_prod})
    DSM_yearly_strategy = pd.DataFrame(data={'Datetime':datetime[indexes_hour[0]].apply(lambda x: x.strftime('%Y-%m-%d')),'Yearly base load (kW)':Yearly_base_load,'Yearly movable load (kW)':Yearly_movable_loads,'Yearly final load (kW)':Yearly_final_loads,'Yearly prod (kW)':Yearly_prod})
    Loads = pd.DataFrame(data={'Datetime':datetime,'Non controllable load (kW)':Non_movable_load,'Daily movable load (kW)':D_movable_load,'Yearly movable load (kW)':Y_movable_load, 'Daily optimized load (kW)':solution.D_DSM.flatten(),'Yearly optimized load (kW)':solution.Y_DSM})
  
    daily_prod = [np.mean(prod[indexes_hour[j]]) for j in range(int(time_resolution*24))]
    yearly_prod = [np.mean(prod[indexes_days[j]]) for j in range(int(n_days))]
    daily_importation = [np.mean(importation[indexes_hour[j]]) for j in range(int(time_resolution*24))]
    yearly_importation = [np.mean(importation[indexes_days[j]]) for j in range(int(n_days))]
    daily_exportation = [-np.mean(exportation[indexes_hour[j]]) for j in range(int(time_resolution*24))]
    yearly_exportation = [-np.mean(exportation[indexes_days[j]]) for j in range(int(n_days))]
    daily_storage = [np.mean(power_storage[indexes_hour[j]]) for j in range(int(time_resolution*24))]
    yearly_storage = [np.mean(power_storage[indexes_days[j]]) for j in range(int(n_days))]
    
    daily_time_balancing = pd.DataFrame(data={'Datetime':datetime[0:int(time_resolution*24)].apply(lambda x: x.strftime('%H:%M')),'Daily load (kW)':Daily_final_loads,'Daily prod (kW)':daily_prod,'Daily importation (kW)':daily_importation,'Daily exportation (kW)':daily_exportation,'Daily storage (kW)':daily_storage})
    yearly_time_balancing = pd.DataFrame(data={'Datetime':datetime[indexes_hour[0]].apply(lambda x: x.strftime('%Y-%m-%d')),'Yearly load (kW)':Yearly_final_loads,'Yearly prod (kW)':yearly_prod,'Yearly importation (kW)':yearly_importation,'Yearly exportation (kW)':yearly_exportation,'Yearly storage (kW)':yearly_storage})

    with pd.ExcelWriter(file_name_out,engine='openpyxl') as writer:
        Overview_outputs.to_excel(writer,sheet_name='Main overview',index=None)
        useprod.to_excel(writer,sheet_name='Balancing',index=None)
        Loadmeet.to_excel(writer,sheet_name='Balancing',index=None,startrow=4)
        when_prod.to_excel(writer,sheet_name='Balancing',index=None,startrow=7)
        when_load.to_excel(writer,sheet_name='Balancing',index=None,startrow=10)
        Grid_use_export.to_excel(writer,sheet_name='Balancing',index=None,startrow=13)
        Grid_use_import.to_excel(writer,sheet_name='Balancing',index=None,startrow=16)
        cost_final_merged.to_excel(writer,sheet_name='Financial outputs',index=None)
        Technical_outputs.to_excel(writer,sheet_name='Technical',index=None)        
        Techs.to_excel(writer,sheet_name='Storages',index=None)
        dist_DOD.to_excel(writer,sheet_name='SOC distributions',index=None)
        outputs_TS.to_excel(writer,sheet_name='TimeSeries',index=None)
        Output_production.to_excel(writer,sheet_name='Production',index=None)
        Loads.to_excel(writer,sheet_name='DSM',index=None)
        DSM_daily_strategy.to_excel(writer,sheet_name='DSM',index=None,startcol=7)
        DSM_yearly_strategy.to_excel(writer,sheet_name='DSM',index=None,startcol=13)
        daily_time_balancing.to_excel(writer,sheet_name='Time_balancing',index=None)
        yearly_time_balancing.to_excel(writer,sheet_name='Time_balancing',index=None,startcol=7)
          
    #Output Charts
    wb = openpyxl.load_workbook(file_name_out)
    
    set_column_width(wb['Main overview'])
    
    ws = wb['Balancing']
    c01 = openpyxl.chart.BarChart()
    c01.title = "Use of production"
    c01.grouping='stacked'
    c01.overlap=100
    c01.add_data(openpyxl.chart.Reference(ws,min_col=1, min_row=1, max_col=3, max_row=2), titles_from_data=True)
    c01.set_categories(openpyxl.chart.Reference(ws,min_col=8, min_row=20, max_col=8, max_row=20))
    #c01.x_axis.title = 'Scenario'
    c01.y_axis.title = 'Energy (kWh)'
    c01.y_axis.scaling.min = 0
    c01.y_axis.scaling.max = 1.3*max(sum(prod),sum(load))/time_resolution
    ws.add_chart(c01, "A1")   
    
    c02 = openpyxl.chart.BarChart()
    c02.title = "Load response"
    c02.grouping='stacked'
    c02.overlap=100
    c02.add_data(openpyxl.chart.Reference(ws,min_col=1, min_row=5, max_col=3, max_row=6), titles_from_data=True)
    c02.set_categories(openpyxl.chart.Reference(ws,min_col=8, min_row=20, max_col=8, max_row=20))
    #c01.x_axis.title = 'Scenario'
    c02.y_axis.title = 'Energy (kWh)'
    c02.y_axis.scaling.min = 0
    c02.y_axis.scaling.max = 1.3*max(sum(prod),sum(load))/time_resolution
    ws.add_chart(c02, "E1")   
    
    c03 = openpyxl.chart.BarChart()
    c03.title = "When production exceeds load"
    c03.grouping='stacked'
    c03.overlap=100
    c03.add_data(openpyxl.chart.Reference(ws,min_col=2, min_row=8, max_col=5+n_store, max_row=9), titles_from_data=True)
    c03.set_categories(openpyxl.chart.Reference(ws,min_col=8, min_row=20, max_col=8, max_row=20))
    #c01.x_axis.title = 'Scenario'
    c03.y_axis.title = 'Energy (kWh)'
    c03.y_axis.scaling.min = 0
    ws.add_chart(c03, "A31") 
    
    c04 = openpyxl.chart.BarChart()
    c04.title = "When load exceeds production"
    c04.grouping='stacked'
    c04.overlap=100
    c04.add_data(openpyxl.chart.Reference(ws,min_col=2, min_row=11, max_col=5+n_store, max_row=12), titles_from_data=True)
    c04.set_categories(openpyxl.chart.Reference(ws,min_col=8, min_row=20, max_col=8, max_row=20))
    #c01.x_axis.title = 'Scenario'
    c04.y_axis.title = 'Energy (kWh)'
    c04.y_axis.scaling.min = 0
    ws.add_chart(c04, "E31") 
    
    c05 = openpyxl.chart.BarChart()
    c05.title = "Grid usage (exportation)"
    c05.grouping='stacked'
    c05.overlap=100
    c05.add_data(openpyxl.chart.Reference(ws,min_col=1, min_row=14, max_col=2, max_row=15), titles_from_data=True)
    c05.set_categories(openpyxl.chart.Reference(ws,min_col=8, min_row=20, max_col=8, max_row=20))
    c05.y_axis.title = 'Energy (kWh)'
    c05.y_axis.scaling.min = 0
    ws.add_chart(c05, "A16") 
    
    c06 = openpyxl.chart.BarChart()
    c06.title = "Grid usage (importation)"
    c06.grouping='stacked'
    c06.overlap=100
    c06.add_data(openpyxl.chart.Reference(ws,min_col=1, min_row=17, max_col=2, max_row=18), titles_from_data=True)
    c06.set_categories(openpyxl.chart.Reference(ws,min_col=8, min_row=20, max_col=8, max_row=20))
    c06.y_axis.title = 'Energy (kWh)'
    c06.y_axis.scaling.min = 0
    ws.add_chart(c06, "E16") 
    
    set_column_width(ws)
    
    ws = wb['Financial outputs']
    c2 = openpyxl.chart.BarChart()
    c2.title = "LCOE decomposition"
    c2.grouping='stacked'
    c2.overlap=100
    c2.add_data(openpyxl.chart.Reference(ws,min_col=5, min_row=1, max_col=10+2*n_store, max_row=3), titles_from_data=True)
    c2.set_categories(openpyxl.chart.Reference(ws,min_col=1, min_row=2, max_col=1, max_row=3))
    c2.x_axis.title = 'Scenario'
    c2.y_axis.title = 'Cost (€/kWh)'
    ws.add_chart(c2, "B6")  
    
    c2_2 = openpyxl.chart.BarChart()
    c2_2.title = "Exportation income"
    c2_2.grouping='stacked'
    c2_2.overlap=100
    c2_2.add_data(openpyxl.chart.Reference(ws,min_col=8, min_row=1, max_col=8, max_row=3), titles_from_data=True)
    c2_2.set_categories(openpyxl.chart.Reference(ws,min_col=1, min_row=2, max_col=1, max_row=3))
    c2_2.x_axis.title = 'Scenario'
    c2_2.y_axis.title = 'Income (€/kWh)'
    ws.add_chart(c2_2, "F6")   
    
    set_column_width(ws)
    
    set_column_width(wb['Technical'])
    
    set_column_width(wb['Storages'])
    
    ws = wb['SOC distributions']
    c1 = openpyxl.chart.LineChart()
    c1.title = "SOC percentile distributions"
    c1.add_data(openpyxl.chart.Reference(ws,min_col=2, min_row=1, max_col=1+n_store, max_row=101), titles_from_data=True)
    c1.set_categories(openpyxl.chart.Reference(ws,min_col=1, min_row=2, max_col=1, max_row=102))
    c1.x_axis.title = 'Percentile'
    c1.y_axis.title = 'Density'
    ws.add_chart(c1, "F2") 
    
    set_column_width(ws)
   
    ws = wb['TimeSeries']
    c3 = openpyxl.chart.LineChart()
    c3.title = "Storages SOCs"
    c3.add_data(openpyxl.chart.Reference(ws,min_col=7+n_store*2, min_row=1, max_col=6+n_store*3, max_row=len(signal)+1), titles_from_data=True)
    c3.set_categories(openpyxl.chart.Reference(ws,min_col=1, min_row=2, max_col=1, max_row=len(signal)))
    c3.x_axis.title = 'Date'
    c3.y_axis.title = 'SOC'
    c3.y_axis.scaling.max = 1
    ws.add_chart(c3, "O3") 
    
    set_column_width(ws)
    
    c4 = openpyxl.chart.LineChart()
    c4.title = "TimeSeries"
    c4.add_data(openpyxl.chart.Reference(ws,min_col=2, min_row=1, max_col=3, max_row=len(signal)+1), titles_from_data=True)
    c4.add_data(openpyxl.chart.Reference(ws,min_col=4+2*n_store, min_row=1, max_col=4+2*n_store, max_row=len(signal)+1), titles_from_data=True)
    c4.set_categories(openpyxl.chart.Reference(ws,min_col=1, min_row=2, max_col=1, max_row=len(signal)))
    c4.x_axis.title = 'Date'
    c4.y_axis.title = 'Power (kW)'
    ws.add_chart(c4, "O20")  
    
    c5 = openpyxl.chart.LineChart()
    c5.title = "Storage timeSeries"
    c5.add_data(openpyxl.chart.Reference(ws,min_col=4, min_row=1, max_col=3+n_store, max_row=len(signal)+1), titles_from_data=True)
    c5.set_categories(openpyxl.chart.Reference(ws,min_col=1, min_row=2, max_col=1, max_row=len(signal)))
    c5.x_axis.title = 'Date'
    c5.y_axis.title = 'Power (kW)'
    ws.add_chart(c5, "O37")  
    
    set_column_width(ws)
    
    set_column_width(wb['Production'])
    
    ws = wb['DSM']
    c6 = openpyxl.chart.LineChart()
    c6.title = "Daily Demand-side management"
    c6.add_data(openpyxl.chart.Reference(ws,min_col=9, min_row=1, max_col=12, max_row=int(time_resolution*24)+1), titles_from_data=True)
    c6.set_categories(openpyxl.chart.Reference(ws,min_col=8, min_row=2, max_col=8, max_row=int(time_resolution*24)))
    c6.x_axis.title = 'Datetime'
    c6.y_axis.title = 'Load (kW)'
    ws.add_chart(c6, "H10") 
    
    c7 = openpyxl.chart.LineChart()
    c7.title = "Yearly Demand-side management"
    c7.add_data(openpyxl.chart.Reference(ws,min_col=15, min_row=1, max_col=18, max_row=int(n_days)+1), titles_from_data=True)
    c7.set_categories(openpyxl.chart.Reference(ws,min_col=14, min_row=2, max_col=14, max_row=int(n_days)))
    c7.x_axis.title = 'Datetime'
    c7.y_axis.title = 'Load (kW)'
    ws.add_chart(c7, "N10") 
    
    set_column_width(ws)
    
    ws = wb['Time_balancing']
    c8 = openpyxl.chart.LineChart()
    c8.title = "Daily balancing"
    c8.add_data(openpyxl.chart.Reference(ws,min_col=2, min_row=1, max_col=6, max_row=int(time_resolution*24)), titles_from_data=True)
    c8.set_categories(openpyxl.chart.Reference(ws,min_col=1, min_row=2, max_col=1, max_row=int(time_resolution*24)))
    c8.x_axis.title = 'Datetime'
    c8.y_axis.title = 'Power (kW)'
    ws.add_chart(c8, "C10") 
    
    c9 = openpyxl.chart.LineChart()
    c9.title = "Yearly balancing"
    c9.add_data(openpyxl.chart.Reference(ws,min_col=9, min_row=1, max_col=13, max_row=int(n_days)+1), titles_from_data=True)
    c9.set_categories(openpyxl.chart.Reference(ws,min_col=8, min_row=2, max_col=8, max_row=int(n_days)))
    c9.x_axis.title = 'Datetime'
    c9.y_axis.title = 'Power (kW)'
    ws.add_chart(c9, "J10") 
    
    set_column_width(ws)
    
    wb.save(file_name_out)
  
    