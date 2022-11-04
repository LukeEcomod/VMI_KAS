
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 10:34:37 2016

@author: slauniai

"""
import numpy as np
import pandas as pd
import os
import configparser
import matplotlib.pyplot as plt
from netCDF4 import Dataset, date2num

#import seaborn as sns
# from scipy import interpolate

eps = np.finfo(float).eps  # machine epsilon

#""" ****** set here spathy data path *** """
# data_path = os.path.join('c:/', 'Datat/SpathyData/SVECatchments')
# spathy_path = os.path.join('c:/', 'Repositories/spathy')
# os.chdir(spathy_path)

def clear_console():
    """
    clears Spyder console window - does not affect namespace
    """
    import os
    clear = lambda: os.system('cls')
    clear()
    return None

""" ******* netcdf output file ****** """

def initialize_netCDF(ID, fname, lat0, lon0, dlat, dlon, dtime=None):
    """
    SpatHy netCDF4 format output file initialization
    IN:
        ID -catchment id as str
        fname - filename
        lat0, lon0 - latitude and longitude
        dlat - nr grid cells in lat
        dlon - nr grid cells in lon
        dtime - nr timesteps, dtime=None --> unlimited
    OUT:
        ncf - netCDF file handle. Initializes data
        ff - netCDF filename incl. path
    LAST EDIT 05.10.2018 / Samuli
    """

    from netCDF4 import Dataset #, date2num, num2date
    from datetime import datetime

    print('**** creating SpaFHy netCDF4 file: ' + fname + ' ****')
    
    # create dataset & dimensions
    directory = os.path.dirname(fname)
    print(directory)
    if not os.path.exists(directory):
        os.makedirs(directory)
    ncf = Dataset(fname, 'w')
    ncf.description = 'SpatHy results. Catchment : ' + str(ID)
    ncf.history = 'created ' + datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    ncf.source = 'SpaFHy v.1.0'

    ncf.createDimension('dtime', dtime)
    ncf.createDimension('dlon', dlon)
    ncf.createDimension('dlat', dlat)


    # call as createVariable(varname,type,(dimensions))
    time = ncf.createVariable('time', 'f8', ('dtime',))
    time.units = "days since 0001-01-01 00:00:00.0"
    time.calendar = 'standard'

    lat = ncf.createVariable('lat', 'f4', ('dlat',))
    lat.units = 'ETRS-TM35FIN'
    lon = ncf.createVariable('lon', 'f4', ('dlon',))
    lon.units = 'ETRS-TM35FIN'

    lon[:] = lon0
    lat[:] = lat0
    
    # CanopyGrid outputs
    W = ncf.createVariable('/cpy/W', 'f4', ('dtime', 'dlat', 'dlon',))
    W.units = 'canopy storage [mm]'
    SWE = ncf.createVariable('/cpy/SWE', 'f4', ('dtime', 'dlat', 'dlon',))
    SWE.units = 'snow water equiv. [mm]'
    Trfall = ncf.createVariable('/cpy/Trfall', 'f4', ('dtime', 'dlat', 'dlon',))
    Trfall.units = 'throughfall [mm]'
    Inter = ncf.createVariable('/cpy/Inter', 'f4', ('dtime', 'dlat', 'dlon',))
    Inter.units = 'interception [mm]'
    Potinf = ncf.createVariable('/cpy/Potinf', 'f4', ('dtime', 'dlat', 'dlon',))
    Potinf.units = 'pot. infiltration [mm]'
    ET = ncf.createVariable('/cpy/ET', 'f4', ('dtime', 'dlat', 'dlon',))
    ET.units = 'dry-canopy et. [mm]'
    Transpi = ncf.createVariable('/cpy/Transpi', 'f4', ('dtime', 'dlat', 'dlon',))
    Transpi.units = 'transpiration [mm]'
    Efloor = ncf.createVariable('/cpy/Efloor', 'f4', ('dtime', 'dlat', 'dlon',))
    Efloor.units = 'forest floor evap. [mm]'
    Evap = ncf.createVariable('/cpy/Evap', 'f4', ('dtime', 'dlat', 'dlon',))
    Evap.units = 'interception evap. [mm]'
    Mbe = ncf.createVariable('/cpy/Mbe', 'f4', ('dtime', 'dlat', 'dlon',))
    Mbe.units = 'mass-balance error [mm]'

    # BucketGrid outputs
    Wliq = ncf.createVariable('/bu/Wliq', 'f4', ('dtime', 'dlat', 'dlon',))
    Wliq.units = 'root zone vol. water cont. [m3m-3]'
    Wliq_top = ncf.createVariable('/bu/Wliq_top', 'f4', ('dtime', 'dlat', 'dlon',))
    Wliq_top.units = 'org. layer vol. water cont. [m3m-3]'
    PondSto = ncf.createVariable('/bu/PondSto', 'f4', ('dtime', 'dlat', 'dlon',))
    PondSto.units = 'pond storage [mm]'
    Infil = ncf.createVariable('/bu/Infil', 'f4', ('dtime', 'dlat', 'dlon',))
    Infil.units = 'infiltration [mm]'
    Drain = ncf.createVariable('/bu/Drain', 'f4', ('dtime', 'dlat', 'dlon',))
    Drain.units = 'drainage [mm]'
    Mbe = ncf.createVariable('/bu/Mbe', 'f4', ('dtime', 'dlat', 'dlon',))
    Mbe.units = 'mass-balance error [mm]'

    # topmodel outputs
    Qt = ncf.createVariable('/top/Qt', 'f4', ('dtime',))
    Qt.units = 'streamflow[m]'
    Qb = ncf.createVariable('/top/Qb', 'f4', ('dtime',))
    Qb.units = 'baseflow [m]'
    Qr = ncf.createVariable('/top/Qr', 'f4', ('dtime',))
    Qr.units = 'returnflow [m]'
    Qs = ncf.createVariable('/top/Qs', 'f4', ('dtime',))
    Qs.units = 'surface runoff [m]'
    R = ncf.createVariable('/top/R', 'f4', ('dtime',))
    R.units = 'average recharge [m]'
    S = ncf.createVariable('/top/S', 'f4', ('dtime',))
    S.units = 'average sat. deficit [m]'
    fsat = ncf.createVariable('/top/fsat', 'f4', ('dtime',))
    fsat.units = 'saturated area fraction [-]'
    #This addition is for the saturation map
    Sloc = ncf.createVariable('/top/Sloc', 'f4', ('dtime','dlat','dlon',))
    Sloc.units = 'local sat. deficit [m]'
    
    # gisdata
    soilclass = ncf.createVariable('/gis/soilclass', 'f4', ('dlat', 'dlon',))
    soilclass.units = 'soil type code [int]'
    sitetype = ncf.createVariable('/gis/sitetype', 'f4', ('dlat', 'dlon',))
    sitetype.units = 'sitetype code [int]'
    twi = ncf.createVariable('/gis/twi', 'f4', ('dlat', 'dlon',))
    twi.units = 'twi'
    LAI_conif = ncf.createVariable('/gis/LAI_conif', 'f4', ('dlat', 'dlon',))
    LAI_conif.units = 'LAI_conif [m2m-2]'
    LAI_decid = ncf.createVariable('/gis/LAI_decid', 'f4', ('dlat', 'dlon',))
    LAI_decid.units = 'LAI_decid [m2m-2]'
    stream = ncf.createVariable('/gis/stream', 'f4', ('dlat', 'dlon',))
    stream.units = 'stream mask'
    dem = ncf.createVariable('/gis/dem', 'f4', ('dlat', 'dlon',))
    dem.units = 'dem'
    slope = ncf.createVariable('/gis/slope', 'f4', ('dlat', 'dlon',))
    slope.units = 'slope'
    flowacc = ncf.createVariable('/gis/flowacc', 'f4', ('dlat', 'dlon',))
    flowacc.units = 'flowacc'
    cmask = ncf.createVariable('/gis/cmask', 'f4', ('dlat', 'dlon',))
    cmask.units = 'cmask'
    
    print('**** netCDF4 file created ****')
    return ncf, fname

def combine_spafhy_nc(site, ncfiles, outfile):
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
    print('*** combining nc files: ' + site)
    
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
    
    # now loop through ncfiles and variables in groups
    m = 0
    for k in range(len(ncfiles)):
        df = Dataset(ncfiles[k], 'r')
        ix_lat = np.where(np.logical_and(cf['lat'] <= np.max(df['lat']), 
                                         cf['lat'] >= np.min(df['lat'])))[0] 
    
        ix_lon = np.where(np.logical_and(cf['lon'] <= np.max(df['lon']),
                                         cf['lon'] >= np.min(df['lon'])))[0]
        
        # loop groups and variables
        for g in groups:
            for v in list(df[g].variables):
                if len(np.shape(df[g][v])) == 3: # dtime, dlat, dlon spatial arrays
                    # plt.figure(k)
                    # plt.imshow(df['bu']['Wliq'][180,:,:])
        
                    a = np.array(cf[g][v][:,ix_lat, ix_lon])
                    b = np.array(df[g][v][:,:,:])
        
                    ix = np.where(~np.isnan(b))
                    a[ix] = b[ix]
                    cf[g][v][:,ix_lat, ix_lon] = a
                    del a, b, ix
                elif len(np.shape(df[g][v])) == 2: # dlat, dlon spatial arrays
                    # plt.figure(k)
                    # plt.imshow(df['bu']['Wliq'][180,:,:])
        
                    a = np.array(cf[g][v][ix_lat, ix_lon])
                    b = np.array(df[g][v][:,:])
        
                    ix = np.where(~np.isnan(b))
                    a[ix] = b[ix]
                    cf[g][v][ix_lat, ix_lon] = a
                    del a, b, ix
                else: # topmodel scalar outputs are computed as averages
                    if m == 0:
                        cf[g][v][:] = df[g][v][:] * 1. / N
                    else:
                        cf[g][v][:] += df[g][v][:] * 1. / N
        m += 1
        df.close()
        print(m)
    
    return cf

"""
***** Get gis data to create catchment ******
"""
def create_catchment(fpath, plotgrids=False, plotdistr=False):
    """
    reads gis-data grids from selected catchments and returns numpy 2d-arrays
    IN:
        fpath - filepath+name
        plotgrids - True plots grids
        plotdistr - True plots distributions
    OUT:
        GisData - dictionary with 2d numpy arrays and some vectors/scalars.

        keys [units]:'dem'[m],'slope'[deg],'soil'[coding 1-4], 'cf'[-],'flowacc'[m2], 'twi'[log m??],
        'vol'[m3/ha],'ba'[m2/ha], 'age'[yrs], 'hc'[m], 'bmroot'[1000kg/ha],'LAI_pine'[m2/m2 one-sided],'LAI_spruce','LAI_decid',
        'info','lat0'[latitude, euref_fin],'lon0'[longitude, euref_fin],loc[outlet coords,euref_fin],'cellsize'[cellwidth,m],
        'peatm','stream','cmask','rockm'[masks, 1=True]      
        
    """
       
    # specific leaf area (m2/kg) for converting leaf mass to leaf area        
    # SLA = {'pine': 5.54, 'spruce': 5.65, 'decid': 18.46}  # m2/kg, Kellomäki et al. 2001 Atm. Env.
    SLA = {'pine': 6.8, 'spruce': 4.7, 'decid': 14.0}  # Härkönen et al. 2015 BER 20, 181-195
    
    # values to be set for 'open peatlands' and 'not forest land'
    nofor = {'vol': 0.1, 'ba': 0.01, 'height': 0.1, 'cf': 0.01, 'age': 0.0, 
             'LAIpine': 0.01, 'LAIspruce': 0.01, 'LAIdecid': 0.01, 'bmroot': 0.01}
    opeatl = {'vol': 0.01, 'ba': 0.01, 'height': 0.1, 'cf': 0.1, 'age': 0.0,
              'LAIpine': 0.01, 'LAIspruce': 0.01, 'LAIdecid': 0.1, 'bmroot': 0.01}

    # dem, set values outside boundaries to NaN
    dem, info, pos, cellsize, nodata = read_AsciiGrid(os.path.join(fpath, 'dem.asc'))
    # latitude, longitude arrays    
    nrows, ncols = np.shape(dem)
    lon0 = np.arange(pos[0], pos[0] + cellsize*ncols, cellsize)
    lat0 = np.arange(pos[1], pos[1] + cellsize*nrows, cellsize)
    lat0 = np.flipud(lat0)  # why this is needed to get coordinates correct when plotting?
    # catchment mask cmask ==1, np.NaN outside
    cmask = dem.copy()
    cmask[np.isfinite(cmask)] = 1.0
    # flowacc, D-infinity, nr of draining cells
    flowacc, _, _, _, _ = read_AsciiGrid(os.path.join(fpath, 'flow_acc.asc'))
    flowacc = flowacc*cellsize**2  # in m2
    # slope, degrees
    slope, _, _, _, _ = read_AsciiGrid(os.path.join(fpath, 'slope.asc'))
    # twi
    twi, _, _, _, _ = read_AsciiGrid(os.path.join(fpath, 'twi.asc'))
    """
    Create soiltype grid and masks for waterbodies, streams, peatlands and rocks
    """
    # Maastotietokanta water bodies: 1=waterbody
    stream, _, _, _, _ = read_AsciiGrid(os.path.join(fpath, 'stream.asc'))
    stream[np.isfinite(stream)] = 1.0
    
    # maastotietokanta peatlandmask
    peatm, _, _, _, _ = read_AsciiGrid(os.path.join(fpath, 'peat.asc'))
    peatm[np.isfinite(peatm)] = 1.0
    #print(np.shape(peatm))
    
    # maastotietokanta kalliomaski
    rockm, _, _, _, _ = read_AsciiGrid(os.path.join(fpath, 'kallio.asc'))
    rockm[np.isfinite(rockm)] = 1.0
    """
    gtk soilmap: read and re-classify into 4 texture classes
    #GTK-pintamaalaji grouped to 4 classes (Samuli Launiainen, Jan 7, 2017)
    #Codes based on maalaji 1:20 000 AND ADD HERE ALSO 1:200 000
    """
    #CoarseTextured = [195213, 195314, 19531421, 195313, 195310]
    #MediumTextured = [195315, 19531521, 195215, 195214, 195601, 195411, 195112,
    #                  195311, 195113, 195111, 195210, 195110, 195312]
    FineTextured = [19531521, 195412, 19541221, 195511, 195413, 195410,
                    19541321, 195618]
    Peats = [195512, 195513, 195514, 19551822, 19551891, 19551892]
    CoarseTextured = [195213, 195314, 19531421, 195313, 195310, 195111, 195311, 
                      195312, 195113, 195112, 195110] 
    #195111 kalliomaa 195311 lohkareita 195312 kiviä 195113 rapakallio 195112 rakka 195110 kalliopaljastuma
    MediumTextured = [195315, 19531521, 195215, 195214, 195601, 195411, 195210] # tasta siirretty joitakin maalajikoodeja luokkaan CoarseTextured, jatkossa voisi luoda oman kallioluokan...
    Water = [195603]

    gtk_s, _, _, _, _ = read_AsciiGrid(os.path.join(fpath, 'soilmap.asc')) 

    r, c = np.shape(gtk_s)
    soil = np.ravel(gtk_s)

    #del gtk_s

    soil[np.in1d(soil, CoarseTextured)] = 1.0  # ; soil[f]=1; del f
    soil[np.in1d(soil, MediumTextured)] = 2.0
    soil[np.in1d(soil, FineTextured)] = 3.0
    soil[np.in1d(soil, Peats)] = 4.0
    soil[np.in1d(soil, Water)] = -1.0

    # reshape back to original grid
    soil = soil.reshape(r, c)
    del r, c
    
    # update waterbody mask
    ix = np.where(soil == -1.0)
    stream[ix] = 1.0
    
    #Warn, remove this
    cmask[soil <= 0] = np.NaN

    soil = soil * cmask
  
    """ stand data (MNFI)"""
    # stand volume [m3ha-1]
    vol, _, _, _, _ = read_AsciiGrid(os.path.join(fpath, 'vol.asc'), setnans=False)
    vol = vol*cmask
    
    # indexes for cells not recognized in mNFI
    ix_n = np.where((vol >= 32727) | (vol == -9999) )  # no satellite cover or not forest land: assign arbitrary values 
    ix_p = np.where((vol >= 32727) & (soil == 4.0))  # open peatlands: assign arbitrary values
    ix_w = np.where((vol >= 32727) & (stream == 1))  # waterbodies: leave out
        
    cmask_cc=cmask.copy() # uusi cmask get_clear_cutsien kasittelyyn
    #cmask_cc[ix_ww] = np.NaN  
    cmask[ix_w] = np.NaN  # NOTE: leaves waterbodies out of catchment mask
    
    vol[ix_n] = nofor['vol']
    vol[ix_p] = opeatl['vol']
    vol[ix_w] = np.NaN

    #pine volume [m3 ha-1]
    p_vol, _, _, _, _ = read_AsciiGrid(os.path.join(fpath, 'vol_pine.asc'))
    p_vol = p_vol*cmask
    #spruce volume [m3 ha-1]
    s_vol, _, _, _, _ = read_AsciiGrid(os.path.join(fpath, 'vol_spruce.asc'))
    s_vol = s_vol*cmask
    #birch volume [m3 ha-1]
    b_vol, _, _, _, _ = read_AsciiGrid(os.path.join(fpath, 'vol_birch.asc'))
    b_vol = b_vol*cmask
    

    # basal area [m2 ha-1]
    ba, _, _, _, _ = read_AsciiGrid(os.path.join(fpath, 'ba.asc') )
    ba[ix_n] = nofor['ba']
    ba[ix_p] = opeatl['ba']
    ba[ix_w] = np.NaN

    # tree height [m]
    height, _, _, _, _ = read_AsciiGrid(os.path.join(fpath, 'hc.asc'))
    height = 0.1*height  # m
    height[ix_n] = nofor['height']
    height[ix_p] = opeatl['height']
    height[ix_w] = np.NaN

    # canopy closure [-]    
    #cf, _, _, _, _ = read_AsciiGrid(os.path.join(fpath, 'latvuspeitto.asc'))
    #cf = 1e-2*cf
    cf = 0.1939 * ba / (0.1939 * ba + 1.69) + 0.01 # lisatty

    cf[ix_n] = nofor['cf']
    cf[ix_p] = opeatl['cf']
    cf[ix_w] = np.NaN


    # stand age [yrs]
    age, _, _, _, _ = read_AsciiGrid(os.path.join(fpath, 'age.asc'))
    age[ix_n] = nofor['age']
    age[ix_p] = opeatl['age']
    age[ix_w] = np.NaN

    # leaf biomasses and one-sided LAI
    bmleaf_pine, _, _, _, _ = read_AsciiGrid(os.path.join(fpath, 'bmleaf_pine.asc'))
    bmleaf_spruce, _, _, _, _ = read_AsciiGrid(os.path.join(fpath, 'bmleaf_spruce.asc'))
    bmleaf_decid, _, _, _, _ = read_AsciiGrid(os.path.join(fpath, 'bmleaf_decid.asc'))
    bmleaf_pine[np.where(bmleaf_pine >= 32767)]=np.NaN
    bmleaf_spruce[np.where(bmleaf_spruce >= 32767)]=np.NaN
    bmleaf_decid[np.where(bmleaf_decid >= 32767)]=np.NaN

    LAI_pine = 1e-3 * bmleaf_pine * SLA['pine']  # 1e-3 converts 10kg/ha to kg/m2
    LAI_pine[ix_n] = nofor['LAIpine']
    LAI_pine[ix_p] = opeatl['LAIpine']
    LAI_pine[ix_w] = np.NaN

    LAI_spruce = 1e-3*bmleaf_spruce * SLA['spruce']
    LAI_spruce[ix_n] = nofor['LAIspruce']
    LAI_spruce[ix_p] = opeatl['LAIspruce']
    LAI_spruce[ix_w] = np.NaN

    LAI_decid = 1e-3*bmleaf_decid * SLA['decid']
    LAI_decid[ix_n] = nofor['LAIdecid']
    LAI_decid[ix_p] = opeatl['LAIdecid']
    LAI_decid[ix_w] = np.NaN

    bmroot_pine, _, _, _, _ = read_AsciiGrid(os.path.join(fpath, 'bmroot_pine.asc'))
    bmroot_spruce, _, _, _, _ = read_AsciiGrid(os.path.join(fpath, 'bmroot_spruce.asc'))
    bmroot_decid, _, _, _, _ = read_AsciiGrid(os.path.join(fpath, 'bmroot_decid.asc'))  
    bmroot = 1e-2*(bmroot_pine + bmroot_spruce + bmroot_decid)  # 1000 kg/ha
    bmroot[ix_n] = nofor['bmroot']
    bmroot[ix_p] = opeatl['bmroot']
    bmroot[ix_w] = np.NaN
    
    # site types
    maintype, _, _, _, _ = read_AsciiGrid(os.path.join(fpath, 'maintype.asc'))
    maintype = maintype*cmask
    sitetype, _, _, _, _ = read_AsciiGrid(os.path.join(fpath, 'sitetype.asc'))
    sitetype = sitetype*cmask
    
    sitetype_id = np.ones(np.shape(sitetype))*cmask
    
    #herb-rich
    ixhr = np.where((maintype == 1) & (sitetype <= 2)) # OMaT, OMT
    sitetype_id[ixhr] = 1
    #mesic
    ixm = np.where((maintype == 1) & (sitetype == 3)) # MT
    sitetype_id[ixm] = 2
    #sub-xeric
    ixsx = np.where((maintype == 1) & (sitetype >= 4)) # VT
    sitetype_id[ixsx] = 3
    # xeric
    ixx = np.where((maintype == 1) &  (sitetype >= 5)) # CT, CIT, lakimetsät
    sitetype_id[ixx] = 4   
    
    # all peatlands assigned now to same class
    ixp = np.where((maintype >= 2) & (maintype <= 4))
    sitetype_id[ixp] = 0
    
    #mvmi_kasvupaikka = Site fertility class 2019 (1-10): 
	#1=lehto tai vastaava suo (tvs) herb-rich forest /herb-rich hw-spruce swamps, pine mires, fens,  
	#2=lehtomainen kangas tvs,herb-rich heat f. /V.myrtillus / tall sedge spruce swamps, tall sedge pine fens, tall sedge fens, 
	#3=tuore kangas tvs, mesic heath f. / Carex clobularis / V.vitis-idaea swamps, Carex globularis pine swamps, low sedge (oligotrophic) fens,
	#4=kuivahko kangas tvs, sub-xeric heath f./Low sedge, dwarf-shrub & cottongrass pine bogs, ombo-oligotrophic bogs,
	#5= kuiva kangas tvs , xeric heath f. / S.fuscum pine bogs, ombotrophic and S.fuscum low sedge bogs.
	#6=karukkokangas tvs, barren heath f. / 
	#7=kalliomaa, hietikko tai vesijÃ¤ttÃ¶maa, rock,cliff or sand f. 
	#8=lakimetsÃ¤, 
	#9=tunturikoivikko, 
	#10=avotunturi
    #mvmi_paatyyppi = Site main class 2019 (1-4); 1=mineral soil, 2= spruce mire, 3= pine bog, 4=open peatland
    #32766 is a missing value: the pixel belongs to forestry land but without satellite image cover
    #32767 is a null value: the pixel does not belong to forestry land or is outside of the country
    
    # catchment outlet location and catchment mean elevation
    (iy, ix) = np.where(flowacc == np.nanmax(flowacc))
    loc = {'lat': lat0[iy], 'lon': lon0[ix], 'elev': np.nanmean(dem)}
    # dict of all rasters
    GisData = {'cmask': cmask, 'cmask_cc': cmask_cc, 'dem': dem, 'flowacc': flowacc, 
               'slope': slope, 'twi': twi, 'gtk_soilcode': gtk_s, 'soilclass': soil, 
               #'smc':smc, 'sfc': sfc, 'soil':soil,
               'peatm': peatm, 'stream': stream, 'rockm': rockm, 'LAI_pine': LAI_pine, 
               'LAI_spruce': LAI_spruce, 'LAI_conif': LAI_pine + LAI_spruce, 
               'LAI_decid': LAI_decid, 'bmroot': bmroot, 'ba': ba, 'hc': height,
               'vol': vol, 'p_vol':p_vol,'s_vol':s_vol,'b_vol':b_vol,'cf': cf, 
               'age': age, 'sitetype': sitetype_id, 'maintype': maintype,
               'cellsize': cellsize, 'info': info, 'lat0': lat0, 'lon0': lon0, 'loc': loc,
              }   

    if plotgrids is True:
        # %matplotlib qt
        # xx, yy = np.meshgrid(lon0, lat0)
        
        # mask for plotting
        mask = cmask.copy()*0.0
        
        ccmask = cmask.copy()
        ccmask[ix_n] = 0.0
        
        plt.close('all')

        plt.figure()

        plt.subplot(221)
        plt.imshow(dem); plt.colorbar(); plt.title('DEM (m)')
        plt.plot(ix, iy,'rs')
        plt.subplot(222)
        plt.imshow(twi); plt.colorbar(); plt.title('TWI')
        plt.subplot(223)
        plt.imshow(slope); plt.colorbar(); plt.title('slope(deg)')
        plt.subplot(224)
        plt.imshow(np.log(flowacc)); plt.colorbar(); plt.title('log flowacc (m2)')

        plt.figure(figsize=(6, 14))

        plt.subplot(221)
        plt.imshow(soil); plt.colorbar(); plt.title('soiltype')
        
        plt.subplot(222)
        plt.imshow(sitetype_id); plt.colorbar(); plt.title('sitetype')
        
        mask = cmask.copy()*0.0
        mask[soil==4] = 1
        #mask[np.isfinite(rockm)] = 2
        mask[np.isfinite(stream)] = 3

        plt.subplot(223)
        plt.imshow(mask); plt.colorbar(); plt.title('masks')
        plt.subplot(224)
        LAIt = (LAI_pine+ LAI_spruce + LAI_decid) * ccmask
        plt.imshow(LAIt); plt.colorbar(); plt.title('LAI (m2/m2)')
        plt.subplot(224)
        plt.imshow(cf * ccmask); plt.colorbar(); plt.title('cf (-)')

        
        plt.figure()
        plt.subplot(321)
        plt.imshow(vol * ccmask); plt.colorbar(); plt.title('vol (m3/ha)')
        plt.subplot(323)
        plt.imshow(height * ccmask); plt.colorbar(); plt.title('hc (m)')
        #plt.subplot(223)
        #plt.imshow(ba); plt.colorbar(); plt.title('ba (m2/ha)')
        plt.subplot(325)
        plt.imshow(age * ccmask); plt.colorbar(); plt.title('age (yr)')
        plt.subplot(322)
        plt.imshow(p_vol * ccmask); plt.colorbar(); plt.title('pine volume (m3/ha)')
        plt.subplot(324)
        plt.imshow(s_vol * ccmask); plt.colorbar(); plt.title('spruce vol. (m3/ha)')
        plt.subplot(326)
        plt.imshow(ba); plt.colorbar(); plt.title('ba (m2/ha)')
        #plt.imshow(1e-3*bmleaf_decid * ccmask); plt.colorbar(); plt.title('decid. vol (m3/ha)')

    if plotdistr is True:
        twi0 = twi[np.isfinite(twi)]
        vol = vol[np.isfinite(vol)]    
        lai = LAIt[np.isfinite(LAIt)]
        soil0 = soil[np.isfinite(soil)]
        
        plt.figure()
        plt.subplot(221)
        plt.hist(twi0, bins=100, color='b', alpha=0.5, density=True, stacked=True)
        plt.ylabel('f');plt.ylabel('twi')

        s = np.unique(soil0)
        colcode = 'rgcym'
        for k in range(0,len(s)):
            # print k
            a = twi[np.where(soil==s[k])]
            a = a[np.isfinite(a)]
            plt.hist(a, bins=50, alpha=0.5, color=colcode[k], density=True, stacked=True, label='soil ' +str(s[k]))
        plt.legend()
        plt.show()

        plt.subplot(222)
        plt.hist(vol, bins=100, color='k', density=True, stacked=True); plt.ylabel('f'); plt.ylabel('vol')
        plt.subplot(223)
        plt.hist(lai, bins=100, color='g', density=True, stacked=True); plt.ylabel('f'); plt.ylabel('lai')
        plt.subplot(224)
        plt.hist(soil0, bins=5, color='r', density=True, stacked=True); plt.ylabel('f');plt.ylabel('soiltype')

    return GisData


def preprocess_soildata(pbu, psoil, soiltype, cmask, spatial=True):
    """
    creates input dictionary for initializing BucketGrid
    Args:
        bbu - bucket parameters dict
        psoil - soiltype dict
        soiltype - soiltype code classified into 5 groups
        cmask - catchment mask
    """
    # create dict for initializing soil bucket.
    # copy pbu into sdata and make each value np.array(np.shape(cmask))
    data = pbu.copy()
    data.update((x, y*cmask) for x, y in data.items())

    if spatial:
        for key in psoil.keys():
            c = psoil[key]['soil_id']
            ix = np.where(soiltype == c)
            data['poros'][ix] = psoil[key]['poros']
            data['fc'][ix] = psoil[key]['fc']
            data['wp'][ix] = psoil[key]['wp']
            data['ksat'][ix] = psoil[key]['ksat']
            data['beta'][ix] = psoil[key]['beta']
            del ix

        data['soilcode'] = soiltype
    return data
 
def sitetype_to_soilproperties(pbu, psoil, sitetype_id, cmask, spatial=True):
    """
    creates input dictionary for initializing BucketGrid
    Args:
        bbu - bucket parameters dict
        psoil - soiltype dict
        sitetype_id - soiltype code classified into 5 groups
        cmask - catchment mask
    """
    # create dict for initializing soil bucket.
    # copy pbu into sdata and make each value np.array(np.shape(cmask))
    data = pbu.copy()
    data.update((x, y*cmask) for x, y in data.items())

    if spatial:
        for key in psoil.keys():
            c = psoil[key]['soil_id']
            ix = np.where(sitetype_id == c)
            data['poros'][ix] = psoil[key]['poros']
            data['fc'][ix] = psoil[key]['fc']
            data['wp'][ix] = psoil[key]['wp']
            data['ksat'][ix] = psoil[key]['ksat']
            data['beta'][ix] = psoil[key]['beta']
            del ix

        data['soilcode'] = sitetype_id
    return data   

""" ************ Reading and writing Ascii -grids ********* """   
 
def read_AsciiGrid(fname, setnans=True):
    
    """ reads AsciiGrid format in fixed format as below:
    
        ncols         750
        nrows         375
        xllcorner     350000
        yllcorner     6696000
        cellsize      16
        NODATA_value  -9999
        -9999 -9999 -9999 -9999 -9999
        -9999 4.694741 5.537514 4.551162
        -9999 4.759177 5.588773 4.767114
    IN:
        fname - filename (incl. path)
    OUT:
        data - 2D numpy array
        info - 6 first lines as list of strings
        (xloc,yloc) - lower left corner coordinates (tuple)
        cellsize - cellsize (in meters?)
        nodata - value of nodata in 'data'
    Samuli Launiainen Luke 7.9.2016
    """
    import numpy as np
    fid = open(fname, 'r')
    info = fid.readlines()[0:6]
    fid.close()

    # print info
    # conversion to float is needed for non-integers read from file...
    xloc = float(info[2].split(' ')[-1])
    yloc = float(info[3].split(' ')[-1])
    cellsize = float(info[4].split(' ')[-1])
    nodata = float(info[5].split(' ')[-1])

    # read rest to 2D numpy array
    data = np.loadtxt(fname, skiprows=6)

    if setnans is True:
        data[data == nodata] = np.NaN
        nodata = np.NaN
    return data, info, (xloc, yloc), cellsize, nodata


def write_AsciiGrid(fname, data, info, fmt='%.18e'):
    """ writes AsciiGrid format txt file
    IN:
        fname - filename
        data - data (numpy array)
        info - info-rows (list, 6rows)
        fmt - output formulation coding
        
    Samuli Launiainen Luke 7.9.2016
    """
    import numpy as np

    # replace nans with nodatavalue according to info
    #nodata = int(info[-1].split(' ')[-1])
    nodata = float(info[5].split(' ')[-1])

    #data[np.isnan(data)] = nodata
    # write info
    fid = open(fname, 'w')
    fid.writelines(info)
    fid.close()

    # write data
    fid = open(fname, 'a')
    np.savetxt(fid, data, fmt=fmt, delimiter=' ')
    fid.close()

""" ********* Flatten 2d array with nans to dense 1d array ********** """


def matrix_to_array(x, nodata=None):
    """ returns 1d array and their indices in original 2d array"""

    s = np.shape(x)
    if nodata is None:  # Nan
        ix = np.where(np.isfinite(x))
    else:
        ix = np.where(x != nodata)
    y = x[ix].copy()
    return y, ix, s


def array_to_matrix(y, ix, s, nodata=None):
    """returns 1d array reshaped into 2d array x of shape s"""
    if nodata is None:
        x = np.ones(s)*np.NaN
    else:
        x = np.ones(s)*nodata
    x[ix] = y

    return x


def inputs_netCDF(ID, fname, data):
    """
    Store gridded data required by SpaFHy into netCDF 
    IN:
        ID -catchment id as str
        fname - filename
        data - dict with keys:
            cmask - catchment mask; integers within np.Nan outside
            LAI_conif [m2m-2]
            LAI_decid [m2m-2]
            hc, canopy closure [m]
            fc, canopy closure fraction [-]
            soil, soil type integer code 1-5
            flowacc - flow accumulation [units]
            slope - local surface slope [units]
            
            cellsize - gridcell size
            lon0 - x-grid
            lat0 - y-grid
    OUT:
        ncf - netCDF file handle. Initializes data
        ff - netCDF filename incl. path
    LAST EDIT 05.10.2018 / Samuli
    """

    from netCDF4 import Dataset #, date2num, num2date
    from datetime import datetime

    print('**** creating SpaFHy input netCDF4 file: ' + fname + ' ****')
    
    # create dataset & dimensions
    ncf = Dataset(fname, 'w')
    ncf.description = 'SpatialData from : ' + str(ID)
    ncf.history = 'created ' + datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    ncf.source = 'SpaFHy v.1.0 inputs'
    
    dlat, dlon = np.shape(data['cmask'])

    ncf.createDimension('dlon', int(dlon))
    ncf.createDimension('dlat', int(dlat))
    ncf.createDimension('scalar', 1)

    # create variables    
    # call as createVariable(varname,type,(dimensions))
    cellsize = ncf.createVariable('cellsize', 'f4', ('scalar',))
    cellsize.units = 'm'
    lat = ncf.createVariable('lat', 'f4', ('dlat',))
    lat.units = 'ETRS-TM35FIN'
    lon = ncf.createVariable('lon', 'f4', ('dlon',))
    lon.units = 'ETRS-TM35FIN'

    cellsize[0] = data['cellsize']
    lon[:] = data['lon0']
    lat[:] = data['lat0']
    
    # required inputs
    cmask = ncf.createVariable('cmask', 'i4', ('dlat','dlon',))
    cmask.units = 'integer inside catchment, Nan outside'
    LAI_conif = ncf.createVariable('LAI_conif', 'f4', ('dlat','dlon',))
    LAI_conif.units = 'conifer LAI (m2m-2)'
    LAI_decid = ncf.createVariable('LAI_decid', 'f4', ('dlat','dlon',))
    LAI_decid.units = 'deciduous annual max LAI (m2m-2)'    
    hc = ncf.createVariable('hc', 'f4', ('dlat','dlon',))
    hc.units = 'canopy height m'    
    cf = ncf.createVariable('cf', 'f4', ('dlat','dlon',))
    cf.units = 'canopy closure (-)' 
    
    soilclass = ncf.createVariable('soilclass', 'i4', ('dlat','dlon',))
    soilclass.units = 'soil class (1 - 5)'
    
    flowacc = ncf.createVariable('flowacc', 'f4', ('dlat','dlon',))
    flowacc.units = 'flow accumualtion area m2'
    slope = ncf.createVariable('slope', 'f4', ('dlat','dlon',))
    slope.units = 'local slope (deg)' 
    
    for k in ['LAI_conif', 'LAI_decid', 'hc', 'cf', 'soilclass', 'flowacc', 'slope']:
        ncf[k][:,:] = data[k]
    
    print('**** done  ****')


""" ********* Get Forcing data  ****** """

def read_forcing_nc(ffile, start_date, end_date, lat0=6902200., lon0=0.0, CO2=400.0, U=2.0, dt=86400.):
    """
    Reads weather data from NetCDF-files and returns dictionary
    Args:
        ffile - nc file
        start_date - str
        end_date -str
    Returns:
        forc - dataframe (dict)
    """
    #from netCDF4 import Dataset, date2num, num2date
    import xarray as xr
    import pandas as pd
    met = xr.open_dataset(ffile)
    
    met = met.sel(time=slice(start_date, end_date))
    met = met.sel(lat=lat0, method='nearest')
    met = met.sel(lon=lon0, method='nearest')
    
    N = len(met.time)
    # forc = {'date': np.ones(N) * np.NaN, 'doy': np.ones(N) * np.NaN, 
    #         'Rg': np.ones(N) * np.NaN, 'Par':np.ones(N) * np.NaN,
    #         'h2o': np.ones(N) * np.NaN, 'VPD': np.ones(N) * np.NaN,'RH': np.ones(N) * np.NaN, 
    #         'Prec': np.ones(N) * np.NaN, 'CO2': np.ones(N) * np.NaN, 'U': np.ones(N) * np.NaN,
    #         'T': np.ones(N) * np.NaN, 'Tmin': np.ones(N) * np.NaN, 'Tmax': np.ones(N) * np.NaN,
    #         'dds': np.ones(N) * np.NaN
    #         }
    
    forc = pd.DataFrame(data=None, index=pd.DatetimeIndex(met.time.data), 
                        columns=['doy', 'Rg', 'Par', 'h2o', 'VPD', 'RH', 'Prec',
                                 'T', 'Tmin', 'Tmax', 'dds'])
    
    tvec = met.time
    #forc['date'] = pd.DatetimeIndex(tvec.data)
    forc['doy'] = tvec.dt.dayofyear.values
    forc['Rg'] = met['global_radiation'].values
    forc['Par'] = met['par'].values
    forc['h2o'] = met['h2o'].values
    forc['VPD'] = met['vpd'].values
    forc['RH'] = met['rh'].values
    forc['Prec'] = met['precipitation'].values / dt
    forc['CO2'] = CO2
    forc['U'] = U
    forc['T'] = met['temperature_avg'].values
    forc['Tmin'] = met['temperature_min'].values
    forc['Tmax'] = met['temperature_max'].values
    forc['dds'] = met['temperature_yearly_sum'].values   
    
    return forc

def read_FMI_weather(ID, start_date, end_date, sourcefile, CO2=380.0):
    """
    reads FMI interpolated daily weather data from file
    IN:
        ID - sve catchment ID. set ID=0 if all data wanted
        start_date - 'yyyy-mm-dd'
        end_date - 'yyyy-mm-dd'
        sourcefile - optional
        CO2 - atm. CO2 concentration (float), optional
    OUT:
        fmi - pd.dataframe with datetimeindex
            fmi columns:['ID','Kunta','aika','lon','lat','T','Tmax','Tmin',
                         'Prec','Rg','h2o','dds','Prec_a','Par',
                         'RH','esa','VPD','doy']
            units: T, Tmin, Tmax, dds[degC], VPD, h2o,esa[kPa],
            Prec, Prec_a[mm], Rg,Par[Wm-2],lon,lat[deg]
    """
    
    # OmaTunniste;OmaItä;OmaPohjoinen;Kunta;siteid;vuosi;kk;paiva;longitude;latitude;t_mean;t_max;t_min;
    # rainfall;radiation;hpa;lamposumma_v;rainfall_v;lamposumma;lamposumma_cum
    # -site number
    # -date (yyyy mm dd)
    # -latitude (in KKJ coordinates, metres)
    # -longitude (in KKJ coordinates, metres)
    # -T_mean (degrees celcius)
    # -T_max (degrees celcius)
    # -T_min (degrees celcius)
    # -rainfall (mm)
    # -global radiation (per day in kJ/m2)
    # -H2O partial pressure (hPa)

    sourcefile = os.path.join(sourcefile)

    #ID = int(ID)

    # import forcing data
    fmi = pd.read_csv(sourcefile, sep=';', header='infer', 
                      usecols=['OmaTunniste', 'Kunta', 'aika', 'longitude',
                      'latitude', 't_mean', 't_max', 't_min', 'rainfall',
                      'radiation', 'hpa', 'lamposumma_v', 'rainfall_v'],
                      parse_dates=['aika'],encoding="ISO-8859-1")
    
    time = pd.to_datetime(fmi['aika'], format='%Y%m%d')

    fmi.index = time
    fmi = fmi.rename(columns={'OmaTunniste': 'ID', 'longitude': 'lon',
                              'latitude': 'lat', 't_mean': 'T', 't_max': 'Tmax',
                              't_min': 'Tmin', 'rainfall': 'Prec',
                              'radiation': 'Rg', 'hpa': 'h2o', 'lamposumma_v': 'dds',
                              'rainfall_v': 'Prec_a'})
    
    fmi['h2o'] = 1e-1*fmi['h2o']  # hPa-->kPa
    fmi['Rg'] = 1e3 / 86400.0*fmi['Rg']  # kJ/m2/d-1 to Wm-2
    fmi['Par'] = 0.5*fmi['Rg']

    # saturated vapor pressure
    esa = 0.6112*np.exp((17.67*fmi['T']) / (fmi['T'] + 273.16 - 29.66))  # kPa
    vpd = esa - fmi['h2o']  # kPa
    vpd[vpd < 0] = 0.0
    rh = 100.0*fmi['h2o'] / esa
    rh[rh < 0] = 0.0
    rh[rh > 100] = 100.0

    fmi['RH'] = rh
    fmi['esa'] = esa
    fmi['VPD'] = vpd

    fmi['doy'] = fmi.index.dayofyear
    fmi = fmi.drop(['aika'], axis=1)
    # replace nan's in prec with 0.0
    #fmi['Prec'][np.isnan(fmi['Prec'])] = 0.0
    fmi['Prec']= fmi['Prec'].fillna(value=0.0)
    # add CO2 concentration to dataframe
    fmi['CO2'] = float(CO2)
    
    # get desired period
    fmi = fmi[(fmi.index >= start_date) & (fmi.index <= end_date)]
#
    if ID > 0:
        fmi = fmi[fmi['ID'] == ID]
    return fmi

""" get forcing data from climate projections (120 years) """
def read_climate_prj(ID, start_date, end_date, sourcefile='c:\\Datat\\ClimateScenarios\\CanESM2.rcp45.csv'):
    #sourcefile = 'c:\\Datat\\ClimateScenarios\\CanESM2.rcp45.csv'

    # catchment id:climategrid_id
    grid_id = {1: 3211, 2: 5809, 3: 6170, 6: 6069, 7: 3150, 8: 3031, 9: 5349, 10: 7003, 11: 3375,
               12: 3879, 13: 3132, 14: 3268, 15: 2308, 16: 5785, 17: 6038, 18: 4488, 19: 3285,
               20: 6190, 21: 5818, 22: 5236, 23: 4392, 24: 4392, 25: 4960, 26: 1538, 27: 5135,
               28: 2059, 29: 4836, 30: 2438, 31: 2177, 32: 2177, 33: 5810}
               # 100: 6050, 101: 5810, 104: 6050, 105: 5810, 106: 5810}

    ID = int(ID)
    g_id = grid_id[ID]

    c_prj = pd.read_csv(sourcefile, sep=';', header='infer',
                        usecols=['id', 'rday', 'PAR', 'TAir','VPD', 'Precip', 'CO2'], parse_dates=False)                          
    c_prj = c_prj[c_prj['id'] == int(g_id)]
    time = pd.date_range('1/1/1980', '31/12/2099', freq='D')
    index = np.empty(30, dtype=int)
    dtime = np.empty(len(c_prj), dtype='datetime64[s]')
    j = 0; k = 0
    for i, itime in enumerate(time):
        if (itime.month == 2 and itime.day == 29):
            index[j] = i
            j = j + 1
        else: 
            dtime[k] = itime.strftime("%Y-%m-%d")
            k = k + 1
    c_prj.index = dtime
    
    c_prj['PAR'] = (1e6/86400.0/4.56)*c_prj['PAR']  # [mol/m-2/d] to Wm-2
    c_prj['Rg'] = 2.0*c_prj['PAR']
    c_prj['doy'] = c_prj.index.dayofyear

    # now return wanted period and change column names to standard
    dat = c_prj[(c_prj.index >= start_date) & (c_prj.index <= end_date)]
    dat = dat.rename(columns={'PAR': 'Par', 'Precip': 'Prec', 'TAir': 'T'})

    return dat

""" ************ Get Runoffs from SVE catchments ******* """


def read_SVE_runoff(ID, start_date,end_date, sourcefile):
    """
    reads FMI interpolated daily weather data from file
    IN:
        ID - sve catchment ID. str OR list of str (=many catchments)
        start_date - 'yyyy-mm-dd'
        end_date - 'yyyy-mm-dd'
        sourcefile - optional
    OUT:
        roff - pd.dataframe with datetimeindex
            columns: measured runoff (mm/d)
            if ID=str, then column is 'Qm'
            if ID = list of str, then column is catchment ID
            MISSING DATA = np.NaN
    CODE: Samuli Launiainen (Luke, 7.2.2017)
    """
    # Runoffs compiled from Hertta-database (Syke) and Metla/Luke old observations.
    # Span: 1935-2015, missing data=-999.99
    # File columns:
    # pvm;14_Paunulanpuro;15_Katajaluoma;16_Huhtisuonoja;17_Kesselinpuro;18_Korpijoki;20_Vaarajoki;
    # 22_Vaha-Askanjoki;26_Iittovuoma;24_Kotioja;27_Laanioja;1_Lompolojanganoja;10_Kelopuro;
    # 13_Rudbacken;3_Porkkavaara;11_Hauklammenoja;19_Pahkaoja;21_Myllypuro;23_Ylijoki;2_Liuhapuro;
    # 501_Kauheanpuro;502_Korsukorvenpuro;503_Kangasvaaranpuro;504_Kangaslammenpuro;56_Suopuro;
    # 57_Valipuro;28_Kroopinsuo;30_Pakopirtti;31_Ojakorpi;32_Rantainrahka

    # import data
    sourcefile = os.path.join(sourcefile)
    dat = pd.read_csv(sourcefile, sep=';', header='infer', parse_dates=['pvm'], index_col='pvm', na_values=-999)

    # split column names so that they equal ID's
    cols = [x.split("_")[0] for x in dat.columns]
    dat.columns = cols

    # get desired period & rename column ID to Qm
    dat = dat[(dat.index >= start_date) & (dat.index <= end_date)]
    dat = dat[ID]
    if type(ID) is str:
        dat.columns = ['Qm']
    return dat



""" ************************ Forcing data, sitefile ************************** """
def read_FMI_weatherdata(forcfile, fyear,lyear, asdict=False):
    """ 
    reads FMI interpolated daily weather data from file containing single point
    IN: 
        forcfile- filename 
        fyear & lyear - first and last years 
        asdict=True if dict output, else pd.dataframe
    OUT: F -pd.DataFrame with columns (or dict with fields):
        time, doy, Ta, Tmin, Tmax (degC), Prec (mm/d), Rg (Wm-2), VPD (kPa), RH (%), esa (kPa), h2o (kPa), dds (degC, degree-day sum)
        
    """
    
    #OmaTunniste;OmaItä;OmaPohjoinen;Kunta;siteid;vuosi;kk;paiva;longitude;latitude;t_mean;t_max;t_min;
    #rainfall;radiation;hpa;lamposumma_v;rainfall_v;lamposumma;lamposumma_cum
    #-site number
    #-date (yyyy mm dd)
    #-latitude (in KKJ coordinates, metres)
    #-longitude (in KKJ coordinates, metres)
    #-T_mean (degrees celcius)
    #-T_max (degrees celcius)
    #-T_min (degrees celcius)
    #-rainfall (mm)
    #-global radiation (per day in kJ/m2)
    #-H2O partial pressure (hPa)

    from datetime import datetime
    #forcfile='c:\\pyspace\\DATAT\\Topmodel_calibr\\FMI_saa_Porkkavaara.csv'

    #import forcing data
    dat=np.genfromtxt(forcfile,dtype=float,delimiter=';', usecols=(5,6,7,10,11,12,13,14,15,16))

    fi=np.where(dat[:,0]>=fyear); li=np.where(dat[:,0]<=lyear)
    ix=np.intersect1d(fi,li); #del fi, li
    #print min(ix), max(ix), np.shape(ix)
    tvec=dat[ix,0:3] #YYYY MM DD

    dat=dat[ix, 3:] 

    time=[]; doy=[]
    for k in range(0,len(tvec)):
        time.append(datetime( int(tvec[k,0]), int(tvec[k,1]), int(tvec[k,2]), 0, 0) )
        doy.append(time[k].timetuple().tm_yday)
    
    time=np.array(time)
    doy=np.array(doy)
    
    Ta=dat[:,0];Tmax=dat[:,1]; Tmin=dat[:,2]; Prec=dat[:,3]; Rg=1e3*dat[:,4]/86400.0;  Par=Rg*0.5 #from kJ/m2/d-1 to Wm-2 
    e=1e-1*dat[:,5]; #hPa-->kPa
    dds=dat[:,6] #temperature sum

    #saturated vapor pressure    
    esa=0.6112*np.exp((17.67*Ta)/ (Ta +273.16 -29.66))  #kPa
    vpd=esa - e; #kPa   
    vpd[vpd<0]=0.0
    rh=100.0*e/esa;
    rh[rh<0]=0.0; rh[rh>100]=100.0
                
    F={'Ta':Ta, 'Tmin':Tmin, 'Tmax':Tmax, 'Prec':Prec, 'Rg':Rg, 'Par': Par, 'VPD':vpd, 'RH':rh, 'esa':esa, 'h2o':e, 'dds':dds}

    F['time']=time
    F['doy']=doy
    
    ix=np.where(np.isnan(F['Prec'])); 
    F['Prec'][ix]=0.0
    #del dat, fields, n, k, time
    
    if asdict is not True:
        #return pandas dataframe
        F=pd.DataFrame(F)
        cols=['time', 'doy', 'Ta', 'Tmin','Tmax', 'Prec', 'Rg', 'Par',  'VPD', 'RH', 'esa', 'h2o', 'dds']
        F=F[cols]
    return F

def read_HydeDaily(filename, start_date='1900-01-01', end_date='2100-01-01'):

    cols=['time','doy','NEE','GPP','TER','ET','H','NEEflag','ETflag','Hflag','Par','Rnet','Ta','VPD','CO2','PrecSmear','Prec','U','Pamb',
    'SWE0','SWCh','SWCa','SWCb','SWCc', 'Tsh','Tsa','Tsb','Tsc','RnetFlag','Trfall','Snowdepth','Snowdepthstd','SWE','SWEstd','Roff1','Roff2']        
    
    dat=pd.read_csv(filename,sep='\s+',header=None, names=None, parse_dates=[[0,1,2]], keep_date_col=False)
    dat.columns=cols
    dat.index=dat['time']; dat=dat.drop(['time','SWE0'],axis=1)
    
    forc=dat[['doy','Ta','VPD','Prec','Par','U']]; forc['Par']= 1/4.6*forc['Par']; forc['Rg']=2.0*forc['Par']
    forc['VPD'][forc['VPD']<=0]=eps
    forc.rename(columns={'Ta':'T'}, inplace=True)
    #relatively extractable water, Hyde A-horizon
    #poros = 0.45    
    fc = 0.30
    wp = 0.10
    Wliq = dat['SWCa']
    Rew = np.maximum( 0.0, np.minimum( (Wliq-wp)/(fc - wp + eps), 1.0) )
    forc['Rew'] = Rew
    forc['CO2'] = 380.0
    # beta, soil evaporation parameter 
    #forc['beta'] =  Wliq / fc
    
    dat = dat[(dat.index >= start_date) & (dat.index <= end_date)]
    forc = forc[(forc.index >= start_date) & (forc.index <= end_date)]
    
    return dat, forc
