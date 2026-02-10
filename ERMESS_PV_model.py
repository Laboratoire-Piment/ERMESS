# -*- coding:utf-8 -*-
'''
:Created: 2025-06-19 16:47:59
:Project: virtual PMS for microgrids
:Version: 1.0
:Author: Mathieu Lafitte
:Description: Tool to simulate the behavior of a PV system with a single array and a single inverter
- input : PV system parameters (panels, inverter, location) + timeseries : solar irradiance, temperature, wind speed
- output : AC power of the inverter (and DC PV output if required)
'''
#---------------------
#%%
import numpy as np
import pandas as pd
from pvlib import temperature, location, pvsystem, modelchain, iotools
from pvlib.irradiance import erbs,get_extra_radiation

LossesParam_default = {# Losses parameters for the PV array
                    "soiling": 2, # Soiling losses (default 2%) [%]
                    "shading": 3, # Shading losses (default 3%) [%]
                    "snow": 0, # Snow losses (default 0%) [%]
                    "mismatch": 2, # Mismatch losses (default 2%) [%]
                    "wiring": 2, # Wiring losses (default 2%) [%]
                    "connections": 0.5, # Wiring and connections losses (default 0.5%) [%]
                    "lid": 1.5, # Light Induced Degradation (LID) losses (default 1.5%) [%]
                    "nameplate_rating": 1, # Nameplate rating losses (default 1%) [%]
                    "age": 0, # Age of the PV array (in years, default 0)
                    "availability": 3 # Availability losses (default 3%) [%]
                    }

# Dimensionnement automatique d'une centrale PV (modules par string, strings en parallèle)

def dimensionnement_pv(module, onduleur, temperature_min):
    # Calcul du Voc en condition froide
    delta_temp = 25 - temperature_min
    voc_cold = module['voc'] * (1 + abs(module['temp_coeff_voc']) * delta_temp)
    
    # Nombre max de modules en série (tension à froid < tension max onduleur)
    n_series_max = int(onduleur['v_dc_max'] // voc_cold)
    
    # Vérification de la tension MPPT
    def est_dans_mppt(n):
        v_mp_string = n * module['vmp']
        return onduleur['v_mppt_min'] <= v_mp_string <= onduleur['v_mppt_max']
    
    # Trouver le plus grand nombre de modules en série valide
    n_series = max([n for n in range(1, n_series_max + 1) if est_dans_mppt(n)], default=0)

    # Calcul de la puissance par string
    p_string = n_series * module['p_mp']
    
    # Courant de court-circuit d'un string
    i_string = module['isc']
    
    # Nombre max de strings (courant total < courant max onduleur)
    n_parallel_max_current = int(onduleur['i_dc_max'] // i_string)
    
    # Nombre de strings pour atteindre une puissance DC cible (facultatif)
    dc_target = onduleur['p_ac_nom'] * onduleur.get('dc_ac_ratio', 1.2)
    n_parallel_power = int(dc_target // p_string)
    
    # Nombre final de strings (le plus petit des deux)
    n_parallel = min(n_parallel_max_current, n_parallel_power)
    
    # Puissance totale DC
    p_total_dc = n_parallel * p_string
    
    return {
        'modules_en_serie': n_series,
        'strings_en_parallele': n_parallel,
        'puissance_totale_dc_kW': round(p_total_dc / 1000, 2),
        'tension_string_froide_V': round(n_series * voc_cold, 2),
        'tension_mppt_string_V': round(n_series * module['vmp'], 2),
        'courant_total_dc_A': round(n_parallel * i_string, 2)
    }

def pvmodel(siteDict: dict, weather, TempParam, 
            Module, Inverter, ModMount: dict,
            arrayParam: dict, LossesParam: dict=LossesParam_default,
            csv: bool = True, plot: bool=False, write_PDC: bool=True, write_POA: bool=True)->tuple[pd.DataFrame, modelchain.ModelChainResult]:
    """Simulates the behavior of a PV system with a single array of modules and a single inverter.
    
    Args:
        siteDict (dict): location of the panels {longitude, latitude, tz, altitude}
        weather (tuple | str): use a tuple if you want to take data from PVGIS https://joint-research-centre.ec.europa.eu/photovoltaic-geographical-information-system-pvgis_en
                -> (startyear, endyear), see https://pvlib-python.readthedocs.io/en/latest/reference/generated/pvlib.iotools.get_pvgis_tmy.html#pvlib.iotools.get_pvgis_tmy
            use a string if you want to use a local file (see read_weatherData for the format)
        TempParam (dict): temperature model's parameters, see https://pvlib-python.readthedocs.io/en/latest/reference/pv_modeling/temperature.html#pv-temperature-models
        Module (str | dict): module's characteristics (all modules of the array are the same)
            use a string (module's 'Name') if you find the reference of your PV module here : https://github.com/NREL/SAM/blob/develop/deploy/libraries/CEC%20Modules.csv
            otherwise you can manually specify the parameters of your modules in a dictionary
        Inverter (str | dict): inverter's characteristics
            use a string (inverter's 'Name') if you find the reference of your inverter here : https://github.com/NREL/SAM/blob/develop/deploy/libraries/CEC%20Inverters.csv
            otherwise you can manually specify the parameters of your inverter in a dictionary
        ModMount (dict): characteristics of the array's mount (Fixed or Single Axis Tracker availables)
        arrayParam (dict): characteristics of the array {surface_type, module_type, modules_per_string, strings}
            more info at https://pvlib-python.readthedocs.io/en/latest/reference/generated/pvlib.pvsystem.Array.html#pvlib-pvsystem-array
        LossesParam (dict, optional): losses of the system {soiling, shading, snow, mismatch, wiring, connections, lid, nameplate_rating, age, availability} Defaults to LossesParam_default.
        csv (bool, optional): True to save the output power dataframe using csv format. Defaults to True.
        plot (bool, optional): True to plot the power timeseries in the end of the simulation. Defaults to False.
        write_PDC (bool, optional): add the output DC power of the array to the results DataFrame. Defaults to True.
        write_POA (bool, optional): add the Plane Of Array Irradiance to the results DataFrame. Defaults to True.
    
    Returns:
        tuple[pd.DataFrame, modelchain.ModelChainResult]: 
            dfpower : AC inverter's output in W + DC panels power in W (optional) + POA irradiance received by the entire array in W (optional)
            model.results : every intermediary results calculated during the simulation. more info at https://pvlib-python.readthedocs.io/en/latest/reference/generated/pvlib.modelchain.ModelChainResult.html#pvlib.modelchain.ModelChainResult
    """
    assert(ModMount["type"].lower() in ["sat", "fixed"])
    # Define the location object knowing the coordinates of the site + get the position of the sun over time
    # ---------------------------------------------------
    site = location.Location(
                             longitude=siteDict["longitude"],
                             latitude=siteDict["latitude"],
                             tz=siteDict["tz"],
                             altitude=siteDict["altitude"],
                             )
    # Weather data : get from PVGIS https://joint-research-centre.ec.europa.eu/photovoltaic-geographical-information-system-pvgis_en
    if type(weather) == tuple: # get from online database
        weatherData = iotools.get_pvgis_tmy(
                                        latitude=site.latitude,
                                        longitude=site.longitude,
                                        startyear=weather[0],
                                        endyear=weather[1],
                                        coerce_year=None
                                        )[0]
    elif type(weather) == str: # read from local file
        weatherData = read_weatherData(weather)
    elif type(weather) ==pd.DataFrame :
        weatherData=weather
        
    # Define the parameters of the PV array and the inverter
    # ---------------------------------------------------
    # NB : You can either enter every parameter manually or pick a predefined inverter from the SAM database https://github.com/NREL/SAM/tree/develop/deploy/libraries
    if type(Module) == str: # database config
        ModDB = pvsystem.retrieve_sam('CECMod')
        ModuleParam = ModDB[Module]
    else: # manual config
        ModuleParam = Module

    if type(Inverter) == str: # database config
        InvDB = pvsystem.retrieve_sam('CECInverter')
        InverterParam = InvDB[Inverter]
    else: # manual config
        InverterParam = Inverter

    if ModMount["type"] == "SAT":
        Mount = pvsystem.SingleAxisTrackerMount(
            axis_tilt=ModMount["axis_tilt"], # Axis tilt angle (degrees, 0° = horizontal)
            axis_azimuth=ModMount["axis_azimuth"], # Axis azimuth angle (degrees, 180° = South)
            max_angle=ModMount["max_angle"], # Maximum angle of the tracker (degrees)
            backtrack=ModMount["backtrack"],
            gcr=ModMount["gcr"] # Ground coverage ratio (GCR) for the tracker
        )
    elif ModMount["type"] == "Fixed":
        Mount = pvsystem.FixedMount(
            surface_tilt=ModMount["surface_tilt"], # Fixed mount tilt angle (degrees)
            surface_azimuth=ModMount["surface_azimuth"], # Fixed mount azimuth angle (degrees)
            racking_model=ModMount["racking_model"] # Racking model (options: 'open_rack', 'close_mount', 'insulated_back', 'freestanding', 'insulated', and 'semi_integrated')
        )

    array = pvsystem.Array(
                           mount=Mount, # choose between SATmount or FixMount
                           surface_type=arrayParam["surface_type"], # options: 'grass', 'concrete', 'asphalt', 'snow', 'water', 'sand', 'urban'
                           module_type=arrayParam["module_type"],  # Module type (options: 'glass_glass', 'glass_polymer')
                           module_parameters=ModuleParam,
                           temperature_model_parameters=TempParam,
                           modules_per_string=arrayParam["modules_per_string"],  # Number of modules in series per string (int, default 1)
                           strings=arrayParam["strings"] # Number of strings in parallel (int, default 1)
                           )
    
    system = pvsystem.PVSystem(
                               arrays=array, # PV array object
                               surface_tilt=ModMount["surface_tilt"], # roof inclination (angle between the roof and the horizontal plane)
                               surface_azimuth=ModMount["surface_azimuth"], # roof orientation (Azimuth angle of the sun in degrees East of North)
                               module_parameters=ModuleParam,
                               inverter_parameters=InverterParam,
                               temperature_model_parameters=TempParam,
                               losses_parameters=LossesParam
                               )
    
    
    if ('DNI' in (weatherData.columns) or 'dni' in (weatherData.columns)) :
        model = modelchain.ModelChain(
                                  system,
                                  site,
                                  temperature_model='sapm',
                                  aoi_model='physical',
                                  losses_model='pvwatts'
                                #   ,dc_model='cec'
                                  )
    else : 
        
        zenith = site.get_solarposition(weatherData.index)['zenith']
        mask_day = zenith < 90
        dni_extra = get_extra_radiation(weatherData.index.dayofyear)
        out = erbs(weatherData.loc[:,'ghi'], zenith, dni_extra)
        dni = pd.DataFrame(data=out['dni'],index=weatherData.index,columns=['dni'])
        dhi = pd.DataFrame(data=out['dhi'],index=weatherData.index,columns=['dhi'])
        weatherData = pd.concat([weatherData, pd.DataFrame(dni,columns=['dni']), pd.DataFrame(dhi,columns=['dhi'])], axis=1)
        weatherData.loc[~mask_day, ['dni', 'dhi']] = 0
        model = modelchain.ModelChain(
                                  system,
                                  site,
                                  temperature_model='sapm',
                                  aoi_model='physical',
                                  losses_model='pvwatts',
                                  )

    model.prepare_inputs(weather=weatherData)
    model.run_model(weather=weatherData)
    # Check results and prepare output DataFrame
    # ---------------------------------------------------
    P_AC_inv = model.results.ac
    P_DC_mod = model.results.dc.p_mp

    assert((P_AC_inv.index == P_DC_mod.index).any())  # dates are in the same order
    assert((P_AC_inv <= P_DC_mod).all())  # AC power cannot exceed DC power

    dfPowerOut = pd.DataFrame({'P_AC_inv': P_AC_inv})
    if write_PDC:
        dfPowerOut['P_DC_mod'] = P_DC_mod
    if write_POA:
        total_surface = ModuleParam["A_c"] * arrayParam["modules_per_string"] * arrayParam["strings"]
        dfPowerOut["poa_global"] = model.results.total_irrad.poa_global * total_surface

    # Set index name (if not already set)
    dfPowerOut.index.name = model.results.ac.index.name

    # Define units
    units = {
        'P_AC_inv': 'W',
        'P_DC_mod': 'W',
        'poa_global': 'W'
    }
    units_full = {col: units.get(col, '') for col in dfPowerOut.columns}

    # Use the timestamp format string as the first index row
    #units_row = pd.DataFrame([units_full], index=['%Y-%m-%d %H:%M:%S+00:00'])
    #units_row.index.name = dfPowerOut.index.name  # Preserve index name

    # Combine units row and data
    #dfpower = pd.concat([units_row, dfPowerOut])
    dfpower=dfPowerOut
    power = dfpower['P_AC_inv'].to_numpy()

    if csv:
        dfpower.to_csv('outputs_PV_model.csv')

    return power, model.results

if __name__ == "__main__":
    # Define the location object knowing the coordinates of the site + get the position of the sun over time
    # ---------------------------------------------------
    siteDict = {
        "longitude":55.483,           # change value
        "latitude":-21.340,           # change value
        "tz":'UTC',                   # change value
        "altitude":0,                 # change value
    }

    weather_local = "meteo_PV.xlsx"
    weather_web = (2021,2022)

    ModuleRef = "Ablytek_5MN6C175_A0"
    InverterRef = "ABB__PVI_3_6_OUTD_S_US__240V_"
    TempParam = temperature.TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']  # SAPM temperature model parameters for open rack glass-glass modules
    
    LossesParam = {# Losses parameters for the PV array
        "soiling": 2, # Soiling losses (default 2%) [%]
        "shading": 3, # Shading losses (default 3%) [%]
        "snow": 0, # Snow losses (default 0%) [%]
        "mismatch": 2, # Mismatch losses (default 2%) [%]
        "wiring": 2, # Wiring losses (default 2%) [%]
        "connections": 0.5, # Wiring and connections losses (default 0.5%) [%]
        "lid": 1.5, # Light Induced Degradation (LID) losses (default 1.5%) [%]
        "nameplate_rating": 1, # Nameplate rating losses (default 1%) [%]
        "age": 0, # Age of the PV array (in years, default 0)
        "availability": 3 # Availability losses (default 3%) [%]
    }
    
    ModuleParam_manual = {# PV system parameters if you can't find your module in the SAM database
        "R_sh_ref":287.102203,
        "R_s":0.316688,
        "Adjust":16.057121,
        "alpha_sc":0.002146,
        "I_L_ref":5.175703,
        "I_o_ref":1.149158e-09,
        "a_ref":1.981696,
        }

    InverterParam_manual = {# Inverter parameters if you can't find your inverter in the SAM database,
        "Pso":22.814655,
        "Paco":3600.0,
        "Pdco":3739.761475,
        "Vdco":340.0,
        "C0":-3.048336e-06,
        "C1":-1.8e-05,
        "C2":0.000483,
        "C3":0.001023,
        "Pnt":0.2,
        # "Vac":240,
        # "Vdcmax":480.0,
        # "Idcmax":10.999298,
        # "Mppt_low":100.0,
        # "Mppt_high":480.0,
    }

    TempParam_manual = { # SAPM coeffs
        'a': -3.47,
        'b': -0.0594,
        'deltaT': 3
    }

    SATModMount = {
        "type":"SAT",
        "axis_tilt":0, # Axis tilt angle (degrees, 0° = horizontal)
        "axis_azimuth":180, # Axis azimuth angle (degrees, 180° = South)
        "surface_tilt":9, # Fixed mount tilt angle (degrees)
        "surface_azimuth":14, # Fixed mount azimuth angle (degrees)
        "max_angle":60, # Maximum angle of the tracker (degrees)
        "backtrack":True,
        "gcr":0.35 # Ground coverage ratio (GCR) for the tracker
    }

    FixModMount = {
        "type":"Fixed",
        "surface_tilt":9, # Fixed mount tilt angle (degrees)
        "surface_azimuth":14, # Fixed mount azimuth angle (degrees)
        "racking_model":'fixed' # Racking model (options: 'open_rack', 'close_mount', 'insulated_back', 'freestanding', 'insulated', and 'semi_integrated')
    }

    arrayParam = {
        "surface_type":'grass', # options: 'grass', 'concrete', 'asphalt', 'snow', 'water', 'sand', 'urban'
        "module_type":'glass_glass',  # Module type (options: 'glass_glass', 'glass_polymer')
        "modules_per_string":12,  # Number of modules in series per string (int, default 1)
        "strings":2  # Number of strings in parallel (int, default 1)
    }

    SATmount = pvsystem.SingleAxisTrackerMount(
                                               axis_tilt=0, # Axis tilt angle (degrees, 0° = horizontal)
                                               axis_azimuth=180, # Axis azimuth angle (degrees, 180° = South)
                                               max_angle=60, # Maximum angle of the tracker (degrees)
                                               backtrack=True,
                                               gcr=0.35 # Ground coverage ratio (GCR) for the tracker
                                               )

    FixMount = pvsystem.FixedMount(
                                   surface_tilt=9, # Fixed mount tilt angle (degrees)
                                   surface_azimuth=14, # Fixed mount azimuth angle (degrees)
                                   racking_model='fixed' # Racking model (options: 'open_rack', 'close_mount', 'insulated_back', 'freestanding', 'insulated', and 'semi_integrated')
                                   )
    
    write_PDC = True # True to get the DC power of the PV array connected to the inverter
    write_POA = True # True to get the Plane Of Array Irrandiance in Wm-2
    csv = True
    plot = True

    # LOSSES ???
    # ArrayLosses = system.pvwatts_losses()
    # LOSS_FACTOR = (100 - ArrayLosses)/100

    dfpower, ModelRes = pvmodel(
        siteDict=siteDict,
        weather=weather_local,
        TempParam=TempParam,
        Module=ModuleRef,
        ModMount=FixModMount,
        Inverter=InverterRef,
        arrayParam=arrayParam,
        LossesParam=LossesParam,
        csv=csv,
        plot=plot
    )
    
    # tests (can be deleted)
    dfpower.drop(dfpower.index[0], inplace=True)
    time = np.array([i for i in range (len(dfpower))])
    e_irrad = np.trapz(dfpower["poa_global"], time)
    e_out_dc = np.trapz(dfpower["P_DC_mod"], time)
    print("nb_mod =", arrayParam["modules_per_string"])
    print("nb_strings =",arrayParam["strings"])
    print("e_irrad =", np.round(e_irrad/1000),"kWh")
    print("e_out_dc =", np.round(e_out_dc/1000),"kWh")
# %%
