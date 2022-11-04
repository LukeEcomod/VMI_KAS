# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 13:46:03 2022

@author: 03081268
"""
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.colors as mplcolors
import matplotlib.cm as mplcm

from netCDF4 import Dataset#, date2num, num2date

# spafhy-files
from spafhy import spafhy_vmikas as spafhy
#from spafhy.spafhy_io import read_FMI_weather, read_HydeDaily
from spafhy.spafhy_io import read_forcing_nc
from spafhy.parameters_vmikas import spafhy_inputs


eps = np.finfo(float).eps

""" paths defined in parameters"""

# read parameters
LAIbins = 5
lat = 62.0 # 65.0 62.0
lon = 26.0

lat0 =  7485750.  # 7485750.  6902250.
lon0 = 3500000.
datafolder = r'data/'
#forcingfile = r'data/SVE_saa.csv'
outfile = r'results/Northern_finland_1980_2021_new.nc'
outfile_pickle = r'results/Northern_finland_1980_2021_new.pk'

pgen, pcpy, pbu, sitedata = spafhy_inputs(LAIbins, lat, lon)

soildata = sitedata['soildata']


""" read forcing data and catchment runoff file """
# FORC = read_FMI_weather(site_id,
#                         pgen['start_date'],
#                         pgen['end_date'],
#                         sourcefile=pgen['forcing_file'])

# dat, FORC = read_HydeDaily(r'Data/HydeDaily2000-2010.txt',
#                            pgen['start_date'],
#                            pgen['end_date'])

# FORC['Prec'] = FORC['Prec'] / pgen['dt']  # mms-1
# FORC['U'] = 2.0 # use constant wind speed ms-1

FORC = read_forcing_nc(r'Data/hiladata/north_south_finland_1980_2021_VMI.nc',
                       pgen['start_date'], pgen['end_date'],
                       lat0=lat0 ,lon0=lon0, dt=pgen['dt']) # 7485750. 6902250
Nsteps = len(FORC)

#%%
# initialize spafhy

spa = spafhy.initialize(pgen, pcpy, pbu, soildata.copy(), sitedata.copy(), cpy_outputs=False, 
                        bu_outputs=False, flatten=True)
# create netCDF output file

dlat, dlon = np.shape(spa.GisData['cmask'])

ncf, ncf_file = spafhy.initialize_netCDF(ID=spa.id, fname=outfile, lat0=spa.GisData['lat0'], 
                                         lon0=spa.GisData['lon0'], dlat=dlat, dlon=dlon, dtime=None)

# run spafhy spinup using 1st year of data
#Nspin = np.where(FORC.index == pgen['spinup_end'])[0][0]
Nspin = 365
for k in range(0, Nspin):
    forc= FORC[['doy', 'Rg', 'Par', 'T', 'Prec', 'VPD', 'CO2','U']].iloc[k]
    spa.run_timestep(forc, ncf=False)

Nsteps = len(FORC)
spa.step_nr = 0
# run spafhy for Nsteps
for k in range(0, Nsteps):
    print(k/Nsteps*100)
    forc= FORC[['doy', 'Rg', 'Par', 'T', 'Prec', 'VPD', 'CO2','U']].iloc[k]
    
    spa.run_timestep(forc, ncf=ncf)

# append inputdata to ncf
ncf['LAI_conif'][:,:] = spa.GisData['LAI_conif']
ncf['LAI_decid'][:,] = spa.GisData['LAI_decid']
#ncf['hc'][:,:] = spa.GisData['hc']
#ncf['ba'][:,:] = spa.GisData['ba']
ncf['sitetype'][:,:] = spa.GisData['sitetype']
    

#del spa, ncf, ncf_file, gisdata

# make dataframe for combined model and forcing

res = pd.DataFrame(data=None, index=FORC.index[:], 
                   columns=['doy', 'Rg', 'Par', 'T', 'Tmin', 'Tmax', 'Prec', 'h2o',
                            'VPD', 'CO2','U', 'dds', 'SWE',
                            'Wliq_herb_rich', 'Psi_herb_rich', 'ET_herb_rich', 'Transpi_herb_rich', 'fW_herb_rich', 'fQ_herb_rich',
                            'Wliq_mesic', 'Psi_mesic', 'ET_mesic', 'Transpi_mesic', 'fW_mesic', 'fQ_mesic',
                            'Wliq_sub_xeric', 'Psi_sub-xeric', 'ET_sub_xeric', 'Transpi_sub_xeric', 'fW_sub_xeric', 'fQ_sub_xeric',
                            'Wliq_xeric', 'Psi_xeric', 'ET_xeric', 'Transpi_xeric', 'fW_xeric', 'fQ_xeric'
                            ])

for k in ['doy', 'Rg', 'Par', 'T', 'Tmin', 'Tmax', 'Prec', 'h2o', 'VPD', 'CO2','U', 'dds']:
    res[k] = FORC[k][:].values
    
n = 2 # LAI = 4.0
sitetypes = ['herb_rich', 'mesic', 'sub_xeric', 'xeric']
for k in range(4):
    res['Wliq_' + sitetypes[k]] = ncf['Wliq'][:,k,n]
    res['Psi_' + sitetypes[k]] = ncf['Psi'][:,k,n]
    res['ET_' + sitetypes[k]] = ncf['ET'][:,k,n]
    res['Transpi_' + sitetypes[k]] = ncf['Transpi'][:,k,n]
    res['fW_' + sitetypes[k]] = ncf['fW'][:,k,n]
    res['fQ_' + sitetypes[k]] = ncf['fQ'][:,k,n]
    res['Rew_' + sitetypes[k]] = ncf['Rew'][:,k,n]

res['SWE'] = ncf['SWE'][:,k,n]

# dump to pickle

with open(outfile_pickle, "wb") as f: # "wb" because we want to write in binary mode
    pickle.dump(res, f)

# make dataframe of effPrec & ET0
a = np.array([FORC['Prec']*spa.dt, ncf['effPrec'][:,2,0], ncf['ET0'][:,2,0],\
              res['Wliq_herb_rich'], res['Wliq_mesic'], res['Wliq_sub_xeric'], res['Wliq_xeric'],\
              res['Rew_herb_rich'], res['Rew_mesic'], res['Rew_sub_xeric'], res['Rew_xeric']]
              )
a = a.T
out = pd.DataFrame(data=a, index=FORC.index, columns=['Prec_mmd', 'effPrec_mmd', 'ETa_mmd',
                                                     'Wliq_herb_rich', 'Wliq_mesic', 'Wliq_sub_xeric', 'Wliq_xeric',
                                                     'Rew_herb_rich', 'Rew_mesic', 'Rew_sub_xeric', 'Rew_xeric']
                   )
out.to_csv(r'results/NF_1980_2021.csv', sep=';')
# close output nc file
ncf.close()
