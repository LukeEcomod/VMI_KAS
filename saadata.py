# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 12:08:58 2022

@author: 03081268
"""
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset, num2date, date2num
import xarray as xr
import pandas as pd
ffile = r'Data/hiladata/north_south_finland_1980_2021_VMI.nc'


# met = Dataset(ffile, 'r')

# datevar = []

# datevar=num2date(met['time'][:],
#                         units=met['time'].units,
#                         calendar=met['time'].calendar, 
#                         only_use_cftime_datetimes=False)

forc = xr.open_dataset(ffile)
met = Dataset(ffile, 'r')

#%%
from spafhy.spafhy_io import read_HydeDaily

hfile = r'Data/HydeDaily2000-2010.txt'

dat, forc = read_HydeDaily(hfile)
#%%
dat = xr.open_dataset(ffile)

# time axis
tvec = dat['time'].values
tvec = pd.DatetimeIndex(tvec)

T = dat['temperature_avg'][:,:,:] # degC
h2o = 1e-1 * dat['water_vapor_pressure'][:,:,:]# kPa

# aggregate to annual data
# https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases

v = ['global_radiation',
     'precipitation',
     'water_vapor_pressure',
     'temperature_avg',
     'temperature_min',
     'temperature_max']


esa = 0.6112*np.exp((17.67*T) / (T + 273.16 - 29.66))  # kPa
vpd = esa - h2o # kPa
vpd[vpd < 0] = 0.0
rh = 100.0* h2o / esa
rh[rh < 0] = 0.0
rh[rh > 100] = 100.0

dat.assign(rh=((dat.coords['lat'], dat.coords['lon'], dat.coords['time']), rh))
dat['esa'] = esa
dat['VPD'] = vpd

adat = dat.resample(time='1A').sum()