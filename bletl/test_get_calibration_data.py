import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import urllib.request
import configparser

import bletl
import pathlib

url = 'http://updates.m2p-labs.com/CalibrationLot.ini'
content = urllib.request.urlopen(url).read().decode()

config = configparser.ConfigParser(strict=False)
config.read_string(content)

lot = '1820-hc-Temp30'

calibration_parameters = {
    'cal_0': float(config[lot]['k0']),
    'cal_100': float(config[lot]['k100']),
    'phi_min': float(config[lot]['irmin']),
    'phi_max': float(config[lot]['irmax']),
    'pH_0': float(config[lot]['ph0']),
    'dpH': float(config[lot]['dph']),
}

blfile = pathlib.Path('bletl', 'data', 'BL1', 'NT_1200rpm_30C_DO-GFP75-pH-BS10_12min_20171221_121339.csv')
bldata = bletl.parse(blfile)

bldata.calibrate(calibration_parameters)
print(bldata.calibrated_data)
