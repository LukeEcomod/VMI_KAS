# -*- coding: utf-8 -*-
"""
NetCDF-tools

Created on Tue Mar 15 09:57:25 2022

@author: Samuli Launiainen
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from netCDF4 import Dataset

eps = np.finfo(float).eps

def combine_spafhy_nc(site, ncfiles, outfile, showfigs=False):
    """
    Combines SpaFHy netCDF outputs from adjacent catchments to single
    netCDF-file. Assumes same time-dimension and variables in all infiles
    Args:
        site - str sitename
        ncfiles - list of ncfiles to be combined
        outfile - output filename
    Returns:
        none
    """    
    from spafhy.spafhy import initialize_netCDF
    
    # folder = r'c:\tempdata\csc\results'
    # ncfiles = os.listdir(folder)
    # ncfiles = [s for s in ncfiles if 'site' in s]
    
    #os.chdir(folder)
    
    #outfile = os.path.join(folder, 'combined', 'site' + str(siteid) + '_combined.nc')
    
    N = len(ncfiles)
    
    """ initialize combined ncfile """
    # combine lat and lon
    df = Dataset(ncfiles[0], 'r')
    lat = np.append([], df['lat'])
    lon = np.append([], df['lon'])
    dtime = df.dimensions['dtime'].size
    df.close()
    
    for k in range(1, len(ncfiles)):
        df = Dataset(ncfiles[k], 'r')
        lat = np.append(lat, df['lat'])
        lon = np.append(lon, df['lon'])
        df.close()
    
    lon = np.unique(lon.data)
    lat = np.unique(lat.data)
    lat = - np.sort(-lat)
    
    # create output nc
    cf, nf = initialize_netCDF(site, outfile, lat, lon, len(lat), len(lon), dtime=dtime)
    
    groups = list(cf.groups)
    
    # now loop through ncfiles and variables in groups 'cpy' and 'bu'
    for k in range(len(ncfiles)):
        df = Dataset(ncfiles[k], 'r')
        ix_lat = np.where(np.logical_and(cf['lat'] <= np.max(df['lat']), 
                                         cf['lat'] >= np.min(df['lat'])))[0] 
    
        ix_lon = np.where(np.logical_and(cf['lon'] <= np.max(df['lon']),
                                         cf['lon'] >= np.min(df['lon'])))[0]
        
        # loop groups and variables
        for g in groups:
            for v in list(df[g].variables):
                if len(np.shape(df[g][v])) > 1: # spatial arrays
                    # plt.figure(k)
                    # plt.imshow(df['bu']['Wliq'][180,:,:])
        
                    a = np.array(cf[g][v][:,ix_lat, ix_lon])
                    b = np.array(df[g][v][:,:,:])
        
                    ix = np.where(~np.isnan(b))
                    a[ix] = b[ix]
                    cf[g][v][:,ix_lat, ix_lon] = a
                    del a, b, ix
                else: # topmodel scalar outputs are computed as averages
                    cf[g][v][:] = df[g][v][:] * 1. / N

    return cf
