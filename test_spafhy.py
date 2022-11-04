# -*- coding: utf-8 -*-
"""
Created on Tue Oct 02 14:47:14 2018

@author: slauniai
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

from netCDF4 import Dataset#, date2num, num2date
from spafhy import spafhy

from spafhy.spafhy_io import read_FMI_weather

from spafhy.spafhy_parameters_tram import parameters, soil_properties
from spafhy.spafhy_io import create_catchment

eps = np.finfo(float).eps

""" paths defined in parameters"""
siteid = 1

# read parameters, list sub-catchment files for site
(pgen, _, _, _) = parameters()
datafolder = pgen['gis_folder']
resultfolder = pgen['ncf_file']

site = 'site' + str(siteid) + '_'
 
dirs = os.listdir(datafolder)
fpaths = [s for s in dirs if site in s]

""" -- run for sub-catchments """
for f in fpaths[0:1]:
    forcingfile = os.path.join(datafolder, 'weather', 'site' + str(siteid) +'.csv')
    print(forcingfile)
    infile = os.path.join(datafolder, f)
    print(infile)
    outfile = os.path.join(resultfolder, f + '.nc')
    print(outfile)
    
    print('*** running site ' + str(siteid) + ' subcatchment ' + str(f) + ' ***')
    
    gisdata = create_catchment(infile, plotgrids=True, plotdistr=True)

    """ set up SpaFHy for the site """
    
    # load parameter dictionaries
    (_, pcpy, pbu, ptop) = parameters()
    psoil = soil_properties()
    
    """ read forcing data and catchment runoff file """
    FORC = read_FMI_weather(str(siteid),
                            pgen['start_date'],
                            pgen['end_date'],
                            sourcefile=forcingfile)
    
    FORC['Prec'] = FORC['Prec'] / pgen['dt']  # mms-1
    FORC['U'] = 2.0 # use constant wind speed ms-1
    Nsteps = len(FORC)
    
    # initialize spafhy
    spa = spafhy.initialize(pgen, pcpy, pbu, ptop, psoil, gisdata, cpy_outputs=False, 
                            bu_outputs=False, top_outputs=False, flatten=True)

    # create netCDF output file

    dlat, dlon = np.shape(spa.GisData['cmask'])

    ncf, ncf_file = spafhy.initialize_netCDF(ID=spa.id, fname=outfile, lat0=spa.GisData['lat0'], 
                                             lon0=spa.GisData['lon0'], dlat=dlat, dlon=dlon, dtime=None)
    
    # run spafhy for Nsteps
    for k in range(0, Nsteps):
        forc= FORC[['doy', 'Rg', 'Par', 'T', 'Prec', 'VPD', 'CO2','U']].iloc[k]
        
        spa.run_timestep(forc, ncf=ncf)

    # close output file
    ncf.close()
    
    # plot soil water content
    plt.figure()
    # function spa._to_grid(x) converts x from 1D array back to 2d
    plt.imshow(spa._to_grid(spa.bu.Wliq)); plt.colorbar(); plt.title('wliq, step ' +str(spa.step_nr))
    plt.savefig('Figure1.png')
    
    #del spa, ncf, ncf_file, gisdata
