# -*- coding: utf-8 -*-
"""
Created on Tue Sep  2 10:51:13 2025

@author: JoPHOBEA
"""

import cdsapi
import os
import xarray as xr
import pandas as pd
import zipfile
import numpy as np
import datetime


def import_meteo(latitude,longitude,altitude,date_from,date_to,timezone_str):
    
    """
    Download and preprocess ERA5-Land meteorological data when needed by ERMESS.

    This function retrieves hourly reanalysis data from the Copernicus Climate
    Data Store (ERA5-Land timeseries product) for a given geographical location
    and time range. The downloaded variables are processed and converted into
    two structured pandas DataFrames suitable for photovoltaic (PV) and wind
    energy production modelling within the ERMESS optimisation workflow.

    Retrieved variables:
        - 2 m air temperature (converted from K to °C)
        - Surface pressure (converted from Pa to hPa)
        - 10 m wind components (converted to wind speed and direction)
        - Surface solar radiation downwards (converted from Wh/m² to W/m²)

    The function:
        1. Downloads ERA5-Land data in CSV format.
        2. Extracts meteorological variables from archived files.
        3. Computes derived quantities (wind speed, wind direction).
        4. Converts units to energy-system compatible standards.
        5. Applies timezone conversion.
        6. Returns formatted datasets for PV and wind modelling.

    :param latitude: Geographic latitude of the site (decimal degrees).
    :type latitude: float

    :param longitude: Geographic longitude of the site (decimal degrees).
    :type longitude: float

    :param altitude: Site altitude above sea level (meters).
                     Currently not directly used but kept for consistency
                     with ERMESS site parameterisation.
    :type altitude: float

    :param date_from: Start datetime of the requested period.
    :type date_from: datetime.datetime or pandas.Timestamp

    :param date_to: End datetime of the requested period.
    :type date_to: datetime.datetime or pandas.Timestamp

    :param timezone_str: Target timezone string (e.g. ``"Europe/Paris"``)
                         for localisation of output data.
    :type timezone_str: str

    :returns: Tuple containing:

              1. **PV_meteo** – DataFrame formatted for photovoltaic models with columns:
                     - ``ghi`` (W/m²)
                     - ``temp_air`` (°C)
                     - ``wind_speed`` (m/s)

              2. **Wind_meteo** – Multi-index DataFrame formatted for windpowerlib models with:
                     - wind speed (m/s)
                     - temperature (°C)
                     - pressure (hPa)

    :rtype: tuple(pandas.DataFrame, pandas.DataFrame)

    :raises Exception: If the CDS API request fails.
    
    :raises FileNotFoundError: If downloaded data cannot be extracted.
    
    :raises KeyError: If expected ERA5 variables are missing.

    :note:
        The function automatically expands the requested period by ±1 day
        before retrieval to ensure complete boundary coverage, then trims
        the final dataset to the exact user-defined time range.

    :warning:
        A valid Copernicus Climate Data Store API key is required.
        Hard-coded credentials should be avoided in production environments.

    :warning:
        Surface solar radiation (ssrd) is converted to hourly average
        irradiance assuming ERA5 cumulative values (division by 3600).
        Users should verify consistency with the selected dataset version.

    :important:
        The returned datasets are directly compatible with the wind and PV
        production models used in the ERMESS evolutionary optimisation loop.
    """

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
    'reanalysis-era5-land-timeseries',
    {
        'product_type': 'reanalysis',
        'format': 'csv',
        'variable': [
            '2m_temperature',
            'surface_pressure',
            '10m_u_component_of_wind',
            '10m_v_component_of_wind',
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

    df_meteo=meteo    
    
    
    df_meteo.index = df_meteo.index.tz_localize('utc').tz_convert(timezone_str)
    
    df_meteo = df_meteo.loc[date_from:date_to]
    df_meteo = df_meteo.round(2)
    
    Wind_heights = np.array([10.,2.,0.])
    PV_meteo = pd.DataFrame(data=df_meteo.loc[:, ['GHI','temp_air','wind_speed']].values,index=df_meteo.index,columns=np.array(['ghi','temp_air','wind_speed']))
    Wind_meteo = pd.DataFrame(data=df_meteo.loc[:, ['wind_speed','temp_air','pressure']].values,index=df_meteo.index,columns=[np.array(['wind_speed','temperature','pressure']),Wind_heights])
        
    return(PV_meteo,Wind_meteo)

