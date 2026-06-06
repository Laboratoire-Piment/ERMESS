# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 09:49:53 2026

@author: JoPHOBEA
"""

NORMALIZED_DAILY_ENERGY = 1.0
PROFILES = {
    "tertiary" : {
        
        "weekend_factor": 0.20,
        "vacation_factor": 0.50,
        "holiday_factor": 0.10,
        "temp_sensitivity": 1.10,
        "ghi_sensitivity": 0.30,
        "wind_sensitivity": 0.35,
        "autocorrelation_memory": 6.00,
        "power_per_user": 0.70,
        "simultaneity_factor": 0.65,

        "hourly_profile": {
            0: 0.20, 1: 0.20, 2: 0.20, 3: 0.20,
            4: 0.20, 5: 0.20, 6: 0.40, 7: 0.40,
            8: 1.00, 9: 1.00, 10: 1.00, 11: 1.00,
            12: 1.00, 13: 1.00,
            14: 1.00, 15: 1.00, 16: 1.00, 17: 1.00,
            18: 0.40, 19: 0.40,
            20: 0.40, 21: 0.40, 22: 0.20, 23: 0.20
        }
        },
    
    "residential" : {
        
        "weekend_factor": 1.10,
        "vacation_factor": 0.20,
        "holiday_factor": 1.05,
        "temp_sensitivity": 1.50,
        "ghi_sensitivity": 0.15,
        "wind_sensitivity": 0.50,
        "autocorrelation_memory": 3.00,
        "power_per_user": 1.00,
        "simultaneity_factor": 0.40,

        "hourly_profile": {
            0: 0.30, 1: 0.30, 2: 0.30, 3: 0.30,
            4: 0.30, 5: 0.30, 6: 0.30, 7: 1.00,
            8: 1.00, 9: 0.30, 10: 0.30, 11: 0.30,
            12: 0.30, 13: 0.30,
            14: 0.30, 15: 0.30, 16: 0.30, 17: 0.30,
            18: 1.00, 19: 1.00,
            20: 1.00, 21: 1.00, 22: 1.00, 23: 0.30
        }
        },

    "teaching": {

        "weekend_factor": 0.05,
        "vacation_factor": 0.20,
        "holiday_factor": 0.05,
        "temp_sensitivity": 0.80,
        "ghi_sensitivity": 0.32,
        "wind_sensitivity": 0.30,
        "autocorrelation_memory": 5.00,
        "power_per_user": 0.75,
        "simultaneity_factor": 0.75,

        "hourly_profile": {
            0: 0.10, 1: 0.10, 2: 0.10, 3: 0.10,
            4: 0.10, 5: 0.10, 6: 0.10, 7: 0.20,
            8: 1.00, 9: 1.00, 10: 1.00, 11: 1.00,
            12: 0.70, 13: 0.70,
            14: 1.00, 15: 1.00, 16: 1.00, 17: 1.00,
            18: 0.30, 19: 0.30,
            20: 0.10, 21: 0.10, 22: 0.10, 23: 0.10
        }
    },

    "office": {

        "weekend_factor": 0.20,
        "vacation_factor": 0.50,
        "holiday_factor": 0.10,
        "temp_sensitivity": 0.90,
        "ghi_sensitivity": 0.35,
        "wind_sensitivity": 0.30,
        "autocorrelation_memory": 7.00,
        "power_per_user": 0.60,
        "simultaneity_factor": 0.65,

        "hourly_profile": {
            0: 0.15, 1: 0.15, 2: 0.15, 3: 0.15,
            4: 0.15, 5: 0.15,
            6: 0.40, 7: 0.40,
            8: 1.00, 9: 1.00, 10: 1.00, 11: 1.00,
            12: 1.00, 13: 1.00,
            14: 1.00, 15: 1.00, 16: 1.00, 17: 1.00,
            18: 0.40, 19: 0.40, 20: 0.40, 21: 0.40,
            22: 0.15, 23: 0.15
        }
    },

    "lab": {

        "weekend_factor": 0.80,
        "vacation_factor": 0.85,
        "holiday_factor": 0.80,
        "temp_sensitivity": 1.60,
        "ghi_sensitivity": 0.10,
        "wind_sensitivity": 0.60,
        "autocorrelation_memory": 12.00,
        "power_per_user": 0.80,
        "simultaneity_factor": 0.85,

        "hourly_profile": {
            0: 0.60, 1: 0.60, 2: 0.60, 3: 0.60,
            4: 0.60, 5: 0.60,
            6: 0.70, 7: 0.70,
            8: 1.00, 9: 1.00, 10: 1.00, 11: 1.00,
            12: 1.00, 13: 1.00,
            14: 1.00, 15: 1.00, 16: 1.00, 17: 1.00,
            18: 0.80, 19: 0.80, 20: 0.80, 21: 0.80,
            22: 0.60, 23: 0.60
        }
    },

    "dormitory": {

        "weekend_factor": 1.10,
        "vacation_factor": 0.60,
        "holiday_factor": 1.05,
        "temp_sensitivity": 1.25,
        "ghi_sensitivity": 0.17,
        "wind_sensitivity": 0.45,
        "autocorrelation_memory": 4.00,
        "power_per_user": 0.45,
        "simultaneity_factor": 0.55,

        "hourly_profile": {
            0: 0.60, 1: 0.60, 2: 0.60, 3: 0.60,
            4: 0.60, 5: 0.60,
            6: 0.90, 7: 0.90,
            8: 0.70, 9: 0.50, 10: 0.50, 11: 0.50,
            12: 0.50, 13: 0.50, 14: 0.50, 15: 0.50,
            16: 0.50, 17: 0.70,
            18: 1.00, 19: 1.00, 20: 1.00, 21: 1.00,
            22: 0.80, 23: 0.60
        }
    }
}