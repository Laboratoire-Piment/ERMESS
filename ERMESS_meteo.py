# -*- coding: utf-8 -*-
"""
Created on Tue Sep  2 10:51:13 2025

@author: jlegalla
"""

import cdsapi
import os
import xarray as xr
import pandas as pd
import zipfile
import numpy as np
import datetime
import pytz


def import_meteo(latitude,longitude,altitude,date_from,date_to,timezone_str):

    output_dir = "meteo_data/era5"
    os.makedirs(output_dir, exist_ok=True)
    strdates = (date_from-datetime.timedelta(1)).strftime("%Y-%m-%d"),(date_to+datetime.timedelta(1)).strftime("%Y-%m-%d")

    folder_name = f"ERA5_{latitude}_{longitude}_{strdates[0]}_to_{strdates[1]}"
    filename = "ERA5.zip"
    
    os.makedirs(os.path.join(output_dir, folder_name), exist_ok=True)

    
    output_path = os.path.join(output_dir,folder_name, filename)    

    if os.path.exists(output_path):
        os.remove(output_path)    

    # Crée un client CDS
    c = cdsapi.Client(url='https://cds.climate.copernicus.eu/api',
    key='0201efea-2d1b-45c6-b53d-74b4572667b5')

# Lancement de la requête
    c.retrieve(
 #   'reanalysis-era5-single-levels-timeseries',
    'reanalysis-era5-land-timeseries',
    {
        'product_type': 'reanalysis',
        'format': 'csv',
        'variable': [
            '2m_temperature',
            'surface_pressure',
            '10m_u_component_of_wind',
            '10m_v_component_of_wind',
##            'surface_roughness',
            'surface_solar_radiation_downwards',  # GHI approximatif
        ],
        "location": {"longitude": str(longitude), "latitude": str(latitude)},
        "date": [strdates[0]+"/"+strdates[1]]
                 },
    
    output_path
)

    name_files = zipfile.ZipFile(output_path).namelist()
    with zipfile.ZipFile(output_path, 'r') as zip_ref:
        zip_ref.extractall(os.path.join(output_dir, folder_name))
    
    meteo = pd.DataFrame()
    for file in name_files:
        ds = xr.open_dataset(os.path.join(output_dir, folder_name,file))
        if 'u10' in list(ds.keys()):
            meteo=meteo.assign(wind_speed=pd.Series(ds["u10"].values**2+ds["v10"].values**2)**0.5)
            meteo=meteo.assign(wind_direction=pd.Series(np.mod(180+np.rad2deg(np.arctan2(ds["u10"].values,ds["v10"].values)),360)))        
        elif 't2m' in list(ds.keys()):
            meteo=meteo.assign(temp_air=pd.Series(ds["t2m"].values - 273.15))# Conversion en °C
        elif "sp" in list(ds.keys()):
            meteo=meteo.assign(pressure=pd.Series(ds["sp"].values / 100))# Pa → hPa
        elif "ssrd" in list(ds.keys()):
            meteo=meteo.assign(GHI=pd.Series(ds["ssrd"].values / 3600))# Wh/m² → W/m²
    meteo=meteo.set_index(pd.to_datetime(ds["valid_time"].values))


    output_dir = "meteo_data/CAMS"
    os.makedirs(output_dir, exist_ok=True)

#    filename = f"CAMS_{latitude}_{longitude}_{strdates[0]}_to_{strdates[1]}.nc"
#    output_path = os.path.join(output_dir, filename)    

#    c = cdsapi.Client(
#        url='https://ads.atmosphere.copernicus.eu/api',
#        key='0201efea-2d1b-45c6-b53d-74b4572667b5'
#        )

#    if os.path.exists(output_path):
#        os.remove(output_path)

#    c.retrieve(
#    'cams-solar-radiation-timeseries',
#    {
#    "sky_type": "observed_cloud",
#    "location": {"longitude": str(longitude), "latitude": str(latitude)},
#    "altitude": [str(altitude)],
#    "date": [strdates[0]+"/"+strdates[1]],
#    "time_step": "1hour",
#    "time_reference": "universal_time",
#    "data_format": "netcdf"
#    },
#    output_path
#    )    
    
#    ds = xr.open_dataset(output_path)
#    dss=ds.squeeze()
#    PV_meteo = pd.DataFrame({
#    "ghi": dss["GHI"].values,
#    "dni": dss["BNI"].values ,
#    "dhi": dss["DHI"].values ,            
#}, index=pd.to_datetime(dss["time"].values))
    
    
    
#    df_meteo=pd.merge(meteo,PV_meteo,left_index=True,right_index=True)
    df_meteo=meteo    
    
    
    df_meteo.index = df_meteo.index.tz_localize('utc').tz_convert(timezone_str)
    
    df_meteo = df_meteo.loc[date_from:date_to]
    df_meteo = df_meteo.round(2)
    
    Wind_heights = np.array([10.,2.,0.])
#    PV_meteo = df_meteo.loc[:, ['ghi','dni','dhi','temp_air','wind_speed']]
    PV_meteo = pd.DataFrame(data=df_meteo.loc[:, ['GHI','temp_air','wind_speed']].values,index=df_meteo.index,columns=np.array(['ghi','temp_air','wind_speed']))
    Wind_meteo = pd.DataFrame(data=df_meteo.loc[:, ['wind_speed','temp_air','pressure']].values,index=df_meteo.index,columns=[np.array(['wind_speed','temperature','pressure']),Wind_heights])
        
    return(PV_meteo,Wind_meteo)

