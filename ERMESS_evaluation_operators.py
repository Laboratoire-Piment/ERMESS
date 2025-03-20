# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 12:00:55 2024

@author: jlegalla
"""
import sys
import pickle
import numpy as np
import pandas as pd
import ERMESS_functions_2 as fGA2
import openpyxl


number_node=sys.argv[1]
operators_perf=[]

with open('operators_'+number_node+'.dat', 'rb') as input_file:
            operators =pickle.load(input_file)
            operators_perf.append(operators)
            
operators_perf=np.concatenate(operators_perf[0], axis=0 ) 

#file_name = 'inputs_GEMS_frontal.xlsx' # shows dialog box and return the path

#xl_file = pd.ExcelFile(file_name)
    
#Data = {sheet_name:xl_file.parse(sheet_name) for sheet_name in xl_file.sheet_names}
#(datetime,specs,prod_C,prods_U,Volums_prod,Y_movable_load,D_movable_load,Non_movable_load,time_resolution,Constraint,Constraint_level,criterion,storage_characteristics,Bounds_prod,n_UP,costs_production,duration_years,grid_prices,fixed_premium,Overrun,Selling_price,Contract_Id,hyperparameters_main,hyperparameters_operators)=fGA2.read_data(Data)
#(Grid_Fossil_fuel_ratio,Main_grid_emissions,Main_grid_ratio) = (Data['Environment']['Main grid fossil fuel ratio'][0],Data['Environment']['Main grid emissions (gCO2/kWh)'][0],Data['Environment']['Main grid ratio primary over final energy'][0])

gens = np.unique(operators_perf[:,0])
perfs=np.repeat(np.nan,(operators_perf.shape[1]-1)*len(gens) ).reshape(len(gens),(operators_perf.shape[1]-1))
for gen in gens:
    sub_table = operators_perf[operators_perf[:,0]==gen,]
    perfs[gen,:] = [np.mean(sub_table[:,i]) for i in range(1,sub_table.shape[1])]  
    
names_operators=("Contract","Contract_power","Production","Storage_timeserie_1","Storage_timeserie_2","Storage_use_global","Storage_use_local","Storage_transfer_1","Storage_transfer_2","Storage_volume","Storage_power","Storage_opposite","Scheduling_consistency","Long-term consistency","Curve_smoothing","Storage_specification","Y_DSM","D_DSM","Crossover")

Contract = pd.DataFrame(data={'Generation':gens,'Contract efficiency':perfs[:,0]})
Contract_power = pd.DataFrame(data={'Generation':gens,'Contract_power efficiency':perfs[:,1]})
Production = pd.DataFrame(data={'Generation':gens,'Production efficiency':perfs[:,2]})
Storage_timeserie_1 = pd.DataFrame(data={'Generation':gens,'Storage_timeserie_1 efficiency':perfs[:,3]})
Storage_timeserie_2 = pd.DataFrame(data={'Generation':gens,'Storage_timeserie_2 efficiency':perfs[:,4]})
Storage_use_global = pd.DataFrame(data={'Generation':gens,'Storage_use_global efficiency':perfs[:,5]})
Storage_use_local = pd.DataFrame(data={'Generation':gens,'Storage_use_local efficiency':perfs[:,6]})
Storage_transfer_1 = pd.DataFrame(data={'Generation':gens,'Storage_transfer_1 efficiency':perfs[:,7]})
Storage_transfer_2 = pd.DataFrame(data={'Generation':gens,'Storage_transfer_2 efficiency':perfs[:,8]})
Storage_volume = pd.DataFrame(data={'Generation':gens,'Storage_volume efficiency':perfs[:,9]})
Storage_power = pd.DataFrame(data={'Generation':gens,'Storage_power efficiency':perfs[:,10]})
Storage_opposite = pd.DataFrame(data={'Generation':gens,'Storage_opposite efficiency':perfs[:,11]})
Scheduling_consistency = pd.DataFrame(data={'Generation':gens,'Scheduling_consistency efficiency':perfs[:,12]})
Long_term_consistency = pd.DataFrame(data={'Generation':gens,'Long_term_consistency efficiency':perfs[:,13]})
Curve_smoothing = pd.DataFrame(data={'Generation':gens,'Curve_smoothing efficiency':perfs[:,14]})
Storage_specification = pd.DataFrame(data={'Generation':gens,'Storage_specification efficiency':perfs[:,15]})
Y_DSM = pd.DataFrame(data={'Generation':gens,'Y_DSM efficiency':perfs[:,16]})
D_DSM = pd.DataFrame(data={'Generation':gens,'D_DSM efficiency':perfs[:,17]})   
Crossover = pd.DataFrame(data={'Generation':gens,'Crossover efficiency':perfs[:,18]})
Overview = pd.DataFrame(data={'Operators':names_operators,'efficiency':np.mean(operators_perf,axis=0)[1:]})     

file_name_out='output_operators.xlsx'

with pd.ExcelWriter(file_name_out,engine='openpyxl') as writer:
    Contract.to_excel(writer,sheet_name='Operators overview',index=None)
    Contract_power.to_excel(writer,sheet_name='Operators overview',index=None,startcol=4)
    Production.to_excel(writer,sheet_name='Operators overview',index=None,startcol=8)
    Storage_timeserie_1.to_excel(writer,sheet_name='Operators overview',index=None,startcol=12)
    Storage_timeserie_2.to_excel(writer,sheet_name='Operators overview',index=None,startcol=16)
    Storage_use_global.to_excel(writer,sheet_name='Operators overview',index=None,startcol=20)
    Storage_use_local.to_excel(writer,sheet_name='Operators overview',index=None,startcol=24)
    Storage_transfer_1.to_excel(writer,sheet_name='Operators overview',index=None,startcol=28)
    Storage_transfer_2.to_excel(writer,sheet_name='Operators overview',index=None,startcol=32)
    Storage_volume.to_excel(writer,sheet_name='Operators overview',index=None,startcol=36)
    Storage_power.to_excel(writer,sheet_name='Operators overview',index=None,startcol=40)
    Storage_opposite.to_excel(writer,sheet_name='Operators overview',index=None,startcol=44)
    Scheduling_consistency.to_excel(writer,sheet_name='Operators overview',index=None,startcol=48)
    Long_term_consistency.to_excel(writer,sheet_name='Operators overview',index=None,startcol=52)
    Curve_smoothing.to_excel(writer,sheet_name='Operators overview',index=None,startcol=56)
    Storage_specification.to_excel(writer,sheet_name='Operators overview',index=None,startcol=60)
    Y_DSM.to_excel(writer,sheet_name='Operators overview',index=None,startcol=64)
    D_DSM.to_excel(writer,sheet_name='Operators overview',index=None,startcol=68)
    Crossover.to_excel(writer,sheet_name='Operators overview',index=None,startcol=72)
    Overview.to_excel(writer,sheet_name='Operators overview',index=None,startcol=76)

    #Output Charts
wb = openpyxl.load_workbook(file_name_out)
    
fGA2.set_column_width(wb['Operators overview'])
    
ws = wb['Operators overview']

c4 = openpyxl.chart.LineChart()
c4.title = "Contract operator efficiency"
c4.add_data(openpyxl.chart.Reference(ws,min_col=2, min_row=2, max_col=2, max_row=len(gens)+1))
c4.set_categories(openpyxl.chart.Reference(ws,min_col=1, min_row=2, max_col=1, max_row=len(gens)+1))
c4.x_axis.title = 'Generation'
c4.y_axis.title = 'Efficiency'
ws.add_chart(c4, "A1")  

c5 = openpyxl.chart.LineChart()
c5.title = "Contract power operator efficiency"
c5.add_data(openpyxl.chart.Reference(ws,min_col=6, min_row=2, max_col=6, max_row=len(gens)+1))
c5.set_categories(openpyxl.chart.Reference(ws,min_col=5, min_row=2, max_col=5, max_row=len(gens)+1))
c5.x_axis.title = 'Generation'
c5.y_axis.title = 'Efficiency'
ws.add_chart(c5, "E1")  

c6 = openpyxl.chart.LineChart()
c6.title = "Production operator efficiency"
c6.add_data(openpyxl.chart.Reference(ws,min_col=10, min_row=2, max_col=10, max_row=len(gens)+1))
c6.set_categories(openpyxl.chart.Reference(ws,min_col=9, min_row=2, max_col=9, max_row=len(gens)+1))
c6.x_axis.title = 'Generation'
c6.y_axis.title = 'Efficiency'
ws.add_chart(c6, "I1")  

c7 = openpyxl.chart.LineChart()
c7.title = "Storage_timeserie_1 operator efficiency"
c7.add_data(openpyxl.chart.Reference(ws,min_col=14, min_row=2, max_col=14, max_row=len(gens)+1))
c7.set_categories(openpyxl.chart.Reference(ws,min_col=13, min_row=2, max_col=13, max_row=len(gens)+1))
c7.x_axis.title = 'Generation'
c7.y_axis.title = 'Efficiency'
ws.add_chart(c7, "M1")  

c8 = openpyxl.chart.LineChart()
c8.title = "Storage_timeserie_2 operator efficiency"
c8.add_data(openpyxl.chart.Reference(ws,min_col=18, min_row=2, max_col=18, max_row=len(gens)+1))
c8.set_categories(openpyxl.chart.Reference(ws,min_col=17, min_row=2, max_col=17, max_row=len(gens)+1))
c8.x_axis.title = 'Generation'
c8.y_axis.title = 'Efficiency'
ws.add_chart(c8, "Q1")  

c9 = openpyxl.chart.LineChart()
c9.title = "Storage_use_global operator efficiency"
c9.add_data(openpyxl.chart.Reference(ws,min_col=22, min_row=2, max_col=22, max_row=len(gens)+1))
c9.set_categories(openpyxl.chart.Reference(ws,min_col=21, min_row=2, max_col=21, max_row=len(gens)+1))
c9.x_axis.title = 'Generation'
c9.y_axis.title = 'Efficiency'
ws.add_chart(c9, "U1")  

c10 = openpyxl.chart.LineChart()
c10.title = "Storage_use_local operator efficiency"
c10.add_data(openpyxl.chart.Reference(ws,min_col=26, min_row=2, max_col=26, max_row=len(gens)+1))
c10.set_categories(openpyxl.chart.Reference(ws,min_col=25, min_row=2, max_col=25, max_row=len(gens)+1))
c10.x_axis.title = 'Generation'
c10.y_axis.title = 'Efficiency'
ws.add_chart(c10, "Y1")  

c11 = openpyxl.chart.LineChart()
c11.title = "Storage_transfer_1 operator efficiency"
c11.add_data(openpyxl.chart.Reference(ws,min_col=30, min_row=2, max_col=30, max_row=len(gens)+1))
c11.set_categories(openpyxl.chart.Reference(ws,min_col=29, min_row=2, max_col=29, max_row=len(gens)+1))
c11.x_axis.title = 'Generation'
c11.y_axis.title = 'Efficiency'
ws.add_chart(c11, "AC1")  

c12 = openpyxl.chart.LineChart()
c12.title = "Storage_transfer_2 operator efficiency"
c12.add_data(openpyxl.chart.Reference(ws,min_col=34, min_row=2, max_col=34, max_row=len(gens)+1))
c12.set_categories(openpyxl.chart.Reference(ws,min_col=33, min_row=2, max_col=33, max_row=len(gens)+1))
c12.x_axis.title = 'Generation'
c12.y_axis.title = 'Efficiency'
ws.add_chart(c12, "AG1")  

c13 = openpyxl.chart.LineChart()
c13.title = "Storage_volume operator efficiency"
c13.add_data(openpyxl.chart.Reference(ws,min_col=38, min_row=2, max_col=38, max_row=len(gens)+1))
c13.set_categories(openpyxl.chart.Reference(ws,min_col=37, min_row=2, max_col=37, max_row=len(gens)+1))
c13.x_axis.title = 'Generation'
c13.y_axis.title = 'Efficiency'
ws.add_chart(c13, "AK1")  

c1 = openpyxl.chart.LineChart()
c1.title = "Storage_power operator efficiency"
c1.add_data(openpyxl.chart.Reference(ws,min_col=42, min_row=2, max_col=42, max_row=len(gens)+1))
c1.set_categories(openpyxl.chart.Reference(ws,min_col=41, min_row=2, max_col=41, max_row=len(gens)+1))
c1.x_axis.title = 'Generation'
c1.y_axis.title = 'Efficiency'
ws.add_chart(c1, "AO1")  

c2 = openpyxl.chart.LineChart()
c2.title = "Storage_opposite operator efficiency"
c2.add_data(openpyxl.chart.Reference(ws,min_col=46, min_row=2, max_col=46, max_row=len(gens)+1))
c2.set_categories(openpyxl.chart.Reference(ws,min_col=45, min_row=2, max_col=45, max_row=len(gens)+1))
c2.x_axis.title = 'Generation'
c2.y_axis.title = 'Efficiency'
ws.add_chart(c2, "AS1")  

c3 = openpyxl.chart.LineChart()
c3.title = "Scheduling_consistency operator efficiency"
c3.add_data(openpyxl.chart.Reference(ws,min_col=50, min_row=2, max_col=50, max_row=len(gens)+1))
c3.set_categories(openpyxl.chart.Reference(ws,min_col=49, min_row=2, max_col=49, max_row=len(gens)+1))
c3.x_axis.title = 'Generation'
c3.y_axis.title = 'Efficiency'
ws.add_chart(c3, "AW1")  

c14 = openpyxl.chart.LineChart()
c14.title = "Long_term_consistency operator efficiency"
c14.add_data(openpyxl.chart.Reference(ws,min_col=54, min_row=2, max_col=54, max_row=len(gens)+1))
c14.set_categories(openpyxl.chart.Reference(ws,min_col=53, min_row=2, max_col=53, max_row=len(gens)+1))
c14.x_axis.title = 'Generation'
c14.y_axis.title = 'Efficiency'
ws.add_chart(c14, "BA1")  

c15 = openpyxl.chart.LineChart()
c15.title = "Curve_smoothing operator efficiency"
c15.add_data(openpyxl.chart.Reference(ws,min_col=58, min_row=2, max_col=58, max_row=len(gens)+1))
c15.set_categories(openpyxl.chart.Reference(ws,min_col=57, min_row=2, max_col=57, max_row=len(gens)+1))
c15.x_axis.title = 'Generation'
c15.y_axis.title = 'Efficiency'
ws.add_chart(c15, "BE1")  

c16 = openpyxl.chart.LineChart()
c16.title = "Storage_specification operator efficiency"
c16.add_data(openpyxl.chart.Reference(ws,min_col=62, min_row=2, max_col=62, max_row=len(gens)+1))
c16.set_categories(openpyxl.chart.Reference(ws,min_col=61, min_row=2, max_col=61, max_row=len(gens)+1))
c16.x_axis.title = 'Generation'
c16.y_axis.title = 'Efficiency'
ws.add_chart(c16, "BI1")  

c17 = openpyxl.chart.LineChart()
c17.title = "Y_DSM operator efficiency"
c17.add_data(openpyxl.chart.Reference(ws,min_col=66, min_row=2, max_col=66, max_row=len(gens)+1))
c17.set_categories(openpyxl.chart.Reference(ws,min_col=65, min_row=2, max_col=65, max_row=len(gens)+1))
c17.x_axis.title = 'Generation'
c17.y_axis.title = 'Efficiency'
ws.add_chart(c17, "BM1")  

c18 = openpyxl.chart.LineChart()
c18.title = "D_DSM operator efficiency"
c18.add_data(openpyxl.chart.Reference(ws,min_col=70, min_row=2, max_col=70, max_row=len(gens)+1))
c18.set_categories(openpyxl.chart.Reference(ws,min_col=69, min_row=2, max_col=69, max_row=len(gens)+1))
c18.x_axis.title = 'Generation'
c18.y_axis.title = 'Efficiency'
ws.add_chart(c18, "BQ1")  

c19 = openpyxl.chart.LineChart()
c19.title = "Crossover operator efficiency"
c19.add_data(openpyxl.chart.Reference(ws,min_col=74, min_row=2, max_col=74, max_row=len(gens)+1))
c19.set_categories(openpyxl.chart.Reference(ws,min_col=73, min_row=2, max_col=73, max_row=len(gens)+1))
c19.x_axis.title = 'Generation'
c19.y_axis.title = 'Efficiency'
ws.add_chart(c19, "BU1")  

c2_2 = openpyxl.chart.BarChart()
c2_2.title = "Operators overview"
c2_2.overlap=100
c2_2.add_data(openpyxl.chart.Reference(ws,min_col=78, min_row=2, max_col=78, max_row=19))
c2_2.set_categories(openpyxl.chart.Reference(ws,min_col=77, min_row=2, max_col=77, max_row=19))
c2_2.x_axis.title = 'Operator'
c2_2.y_axis.title = 'Efficiency'
ws.add_chart(c2_2, "A10")   
   
 
fGA2.set_column_width(ws)
    
wb.save(file_name_out)
