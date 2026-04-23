# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 21:17:59 2026

@author: JoPHOBEA
"""

charts_config = [

        # =========================
        # GLOBAL DISPATCHING
        # =========================
        {
            "sheet": "Global dispatching",
            "type": "bar",
            "title": "Use of production",
            "data": (2, 1, 4, 3),
            "categories": (1, 2, 1, 3),
            "position": "A1",
            "x_title": "Scenario",
            "y_title": "Energy (kWh)",
            "ymax": "auto_energy",
            "height":8,
            "stacked":True
        },
        {
            "sheet": "Global dispatching",
            "type": "bar",
            "title": "Load response",
            "data": (2, 4, 4, 6),
            "categories": (1, 5, 1, 6),
            "position": "G1",
            "x_title": "Scenario",
            "y_title": "Energy (kWh)",
            "ymax": "auto_energy",
            "height":8,
            "stacked":True
        },
        {
            "sheet": "Global dispatching",
            "type": "bar",
            "title": "When production exceeds load",
            "data": lambda C: (2, 7, 4 + 2 * C.storage.n_store, 9),
            "categories": (1, 8, 1, 9),
            "position": "A32",
            "x_title": "Scenario",
            "y_title": "Energy (kWh)",
            "height":8,
            "stacked":True
        },
        {
            "sheet": "Global dispatching",
            "type": "bar",
            "title": "When load exceeds production",
            "data": lambda C: (2, 10, 4 + 2 * C.storage.n_store, 12),
            "categories": (1, 11, 1, 12),
            "position": "G32",
            "x_title": "Scenario",
            "y_title": "Energy (kWh)",
            "height":8,
            "stacked":True
        },
        {
            "sheet": "Global dispatching",
            "type": "bar",
            "title": "Grid usage (exportation)",
            "data": (2, 13, 3, 15),
            "categories": (1, 14, 1, 15),
            "position": "A16",
            "x_title": "Scenario",
            "y_title": "Energy (kWh)",
            "height":8,
            "stacked":True
        },
        {
            "sheet": "Global dispatching",
            "type": "bar",
            "title": "Grid usage (importation)",
            "data": (2, 16, 3, 18),
            "categories": (1, 17, 1, 18),
            "position": "G16",
            "x_title": "Scenario",
            "y_title": "Energy (kWh)",
            "height":8,
            "stacked":True
        },

        # =========================
        # FINANCIAL
        # =========================
        {
            "sheet": "Financial outputs",
            "type": "bar",
            "title": "LCOE decomposition",
            "data": (3, 1, 12, 3),
            "categories": (1, 2, 1, 3),
            "position": "B6",
            "y_title": "Cost (€/kWh)",
            "legend_position": "b",
            "height": 15,
            "width": 15,
            "stacked":True
        },
        {
            "sheet": "Financial outputs",
            "type": "bar",
            "title": "Exportation income",
            "data": (13, 1, 13, 3),
            "categories": (1, 2, 1, 3),
            "position": "F6",
            "y_title": "Income (€/kWh)",
            "legend_position": "b",
            "height": 15,
            "width": 15,
            "stacked":True
        },

        # =========================
        # SOC DISTRIBUTIONS
        # =========================
        {
            "sheet": "SOC distributions",
            "type": "series of lines",
            "title": "SOC percentile distributions",
            "xvalues": (1, 1, 1, 101),
            "series_col_start": lambda C: range( 2 , 2+C.storage.n_store ),
            "series_len": 100,
            "position": "F2",
            "x_title": "Percentile",
            "y_title": "Density",
            "height": 10,
            "width": 18,
            "legend": False,
            "y_min":0,
            "y_max":100
        },

        # =========================
        # TIMESERIES
        # =========================
        {
            "sheet": "TimeSeries",
            "type": "series of lines",
            "title": "Storages SOCs",
            "position": "O3",
            "series_loop": True,
            "xvalues": (1, 1, 1, "n_bits+1"),
            "series_col_start": lambda C: range(9 + 2*C.storage.n_store, 9 + 3 * C.storage.n_store),
            "series_len": lambda C: C.time.n_bits,
            "ymin": 0,
            "ymax": 1,
            "hide_x_axis":True,
            "thin lines":True,
            "width":120,
            "legend_position": "l"
        },

        {
            "sheet": "TimeSeries",
            "type": "series of lines",
            "title": "TimeSeries",
            "position": "O23",
            "custom_series": [2,3,"6+2*n_store"],
            "series_len": lambda C: C.time.n_bits,
            "hide_x_axis":True,
            "thin lines":True,
            "width":120,
            "legend_position": "l"
        },

        {
            "sheet": "TimeSeries",
            "type": "series of lines",
            "title": "Storage timeSeries",
            "position": "O43",
            "series_loop": True,
            "series_col_start": 4,
            "series_len": lambda C: C.time.n_bits,
            "hide_x_axis":True,
            "thin lines":True,
            "width":120,
            "legend_position": "l"
        },

        # =========================
        # PRODUCTION
        # =========================
        {
            "sheet": "Production",
            "type": "bar",
            "title": "Production set ratios",
            "data": lambda C: (3, 1, 3, None), 
            "categories": lambda C: (1, 2, 1, None),
            "position": "F2",
            "ymin": 0,
            "ymax": 1,
            "x_title": "Production ID",
            "y_title": "Coverage ratio",
            "legend": False
        },

        # =========================
        # DSM
        # =========================
        {
            "sheet": "Demand side management",
            "type": "line",
            "title": "Daily Demand-side management",
            "data": lambda C: (9, 1, 12, int(C.time.time_resolution * 24 + 1)),
            "categories": lambda C: (8, 2, 8, int(C.time.time_resolution * 24 + 1)),
            "position": "D10",
            "y_title": "Load (kW)",
            "xtick_rotation": True,
            "legend_position": "b",
        },
        {
            "sheet": "Demand side management",
            "type": "line",
            "title": "Yearly Demand-side management",
            "data": lambda C: (15, 1, 18, int(C.time.n_days + 1)),
            "categories": lambda C: (14, 2, 14, int(C.time.n_days + 1)),
            "position": "N10",
            "y_title": "Load (kW)",
            "legend_position": "b",
        },

        # =========================
        # TIME BALANCING
        # =========================
        {
            "sheet": "Balancing",
            "type": "line",
            "title": "Daily balancing (optimization)",
            "data": lambda C: (2, 1, 6, int(C.time.time_resolution * 24 + 1)),
            "categories": lambda C: (1, 2, 1, int(C.time.time_resolution * 24)),
            "position": "A10",
            "xtick_rotation": True,
            "legend_position": "b",
        },
        {
            "sheet": "Balancing",
            "type": "line",
            "title": "Daily balancing (baseline)",
            "data": lambda C: (9, 1, 13, int(C.time.time_resolution * 24 + 1)),
            "categories": lambda C: (8, 2, 8, int(C.time.time_resolution * 24 + 1)),
            "position": "H10",
            "xtick_rotation": True,
            "legend_position": "b",
        },
        {
            "sheet": "Balancing",
            "type": "line",
            "title": "Yearly balancing (optimization)",
            "data": lambda C: (16, 1, 20, int(C.time.n_days + 1)),
            "categories": lambda C: (15, 2, 15, int(C.time.n_days + 1)),
            "position": "O10",
            "legend_position": "b",
        },
        {
            "sheet": "Balancing",
            "type": "line",
            "title": "Yearly balancing (baseline)",
            "data": lambda C: (23, 1, 27, int(C.time.n_days + 1)),
            "categories": lambda C: (22, 2, 22, int(C.time.n_days)),
            "position": "V10",
            "legend_position": "b",
        },

        # =========================
        # EMS
        # =========================
        {
            "sheet": "EMS",
            "type": "series of lines",
            "title": "Overlaps",
            "x_title": "effective SOC (%)",
            "y_title": "Secondary source involvment (%)",
            "xvalues": (12, 2, 12, 10),
            "series_col_start": lambda C: range(13,15),
            "series_len": 9,
            "position": "O13",
            "no smooth": True,
            "width":14,
            "xmax":100,
            "ymax":100,
            "legend_position": "b",
        },
        {
            "sheet": "EMS",
            "type": "series of lines",
            "title": "Daily demand-side management",
            "x_title": "Hour",
            "y_title": "Minimum level (%)",
            "xvalues": (4, 2, 4, 25),
            "series_col_start": lambda C: range(5,6),
            "series_len": 24,
            "position": "A24",
            "no smooth": True,
            "width":14,
            "legend": False,
            "xmax":24,
            "ymax":100,
        },
        {
            "sheet": "EMS",
            "type": "series of lines",
            "title": "Yearly demand-side management",
            "x_title": "Month",
            "y_title": "Minimum level (%)",
            "series_col_start": lambda C: range(8,9),
            "xvalues": (7, 2, 7, 13),
            "series_len": 12,
            "position": "G13",
            "no smooth": True,
            "width":14,
            "legend": False,
            "xmax":12,
            "ymax":100,
        }
    ]
