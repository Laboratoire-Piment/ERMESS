# -*- coding: utf-8 -*-
"""
Created on Thu May  7 15:47:28 2026

@author: JoPHOBEA
"""

import pandas as pd

def load_excel(path):
    """Load ERMESS input Excel file."""
    xl = pd.ExcelFile(path)
    data = {sheet: xl.parse(sheet) for sheet in xl.sheet_names}
    xl.close()
    return data
