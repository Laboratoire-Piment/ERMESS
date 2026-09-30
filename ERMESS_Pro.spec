# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['ERMESS_Pro.py'],
    pathex=[],
    binaries=[],
    datas=[('C:\\Users\\jlegalla\\AppData\\Local\\miniconda3\\envs\\env_ERMESS_Numba\\Lib\\site-packages\\pvlib\\data\\sam-library-cec-modules-2019-03-05.csv', 'pvlib/data'), ('C:\\Users\\jlegalla\\AppData\\Local\\miniconda3\\envs\\env_ERMESS_Numba\\Lib\\site-packages\\pvlib\\data\\sam-library-cec-inverters-2019-03-05.csv', 'pvlib\\data'), ('C:\\Users\\jlegalla\\AppData\\Local\\miniconda3\\envs\\env_ERMESS_Numba\\Lib\\site-packages\\windpowerlib\\oedb\\power_coefficient_curves.csv', 'windpowerlib/data'), ('C:\\Users\\jlegalla\\AppData\\Local\\miniconda3\\envs\\env_ERMESS_Numba\\Lib\\site-packages\\windpowerlib\\oedb\\power_curves.csv', 'windpowerlib/data')],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='ERMESS_Pro',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
