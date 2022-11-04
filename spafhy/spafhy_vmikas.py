# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 16:18:57 2016

@author: slauniai


"""
import os
# import pandas as pd
# import matplotlib.pyplot as plt
# import timeit
import numpy as np
from .canopygrid import CanopyGrid
from .bucketgrid import BucketGrid
from .topmodel import Topmodel_Homogenous as Topmodel
from .spafhy_io import preprocess_soildata

eps = np.finfo(float).eps  # machine epsilon

""" ************** SpaFHy v1.0 ************************************

Simple spatial hydrology and catchment water balance model.

CONSISTS OF THREE CLASSES, defined in separate modules: 
    CanopyGrid - vegetation and snowpack water storages and flows
    BucketGrid - topsoil bucket model (root zone / topsoil water storage)
    Topmodel - integration to catchment scale using Topmodel -concept
HELPER FUNCTIONS:
    spafhy_parameters - parameter definition file
    spafhy_io - utility functions for data input & output
 
MAIN PROGRAM:   
    spafhy_driver is main program, call it as
    outargs = spathy_driver(spathyparamfile, args)
    
    spathyparamfile - path to parameter file, default is 'spathy_default.ini'
    soil type dependent parameters are in 'soilparam.ini'

NEEDS 2D gis rasters in ascii-grid format

CanopyGrid & BucketGrid can be initialized from gis-data or set to be spatially constant

ToDo:
    CanopyGrid:
        -include topographic shading to radiation received at canopy top
        -radiation-based snowmelt coefficient
        -add simple GPP-model; 2-step Farquhar or LUE-based approach
    BucketGrid:
        -make soil hydrologic properties more realistic e.g. using pedotransfer functions
        -kasvupaikkatyyppi (multi-NFI) --> soil properties
        -add soil frost model, simplest would be Stefan equation with coefficients modified based on snow insulation
          --> we need snow density algorithm: SWE <-----> depth
    Topmodel:
        -think of definging 'relative m & to grids' (soil-type & elevation-dependent?) and calibrate 'catchment averages'
        -topmodel gives 'saturated zone storage deficit in [m]'. This can be converted to gwl proxy (?) if: 
        local water retention characteristics are known & hydrostatic equilibrium assumes. 
        Look which water retention model was analytically integrable (Campbell, brooks-corey?)
    
    Graphics and analysis of results:
        -make ready functions


(C) Samuli Launiainen 10/2016-->    

VERSION 05.10.2018 / equations correspond to GMDD paper

"""


def initialize(pgen, pcpy, pbu, soildata, gisdata, cpy_outputs=False, 
                  bu_outputs=False, top_outputs=False, flatten=False):
    """ 
    ******************** sets up SpaFHy  **********************
    
    Normal SpaFHy run without parameter optimization
    1) gets parameters as input arguments
    2) reads GIS-data, here predefined format for Seurantaverkko -cathcments
    
    3) creates CanopyGrid (cpy), BucketGrid (bu) and Topmodel (top) -objects  within Spathy-object (spa) and temporary outputs
    4) creates netCDF -file for outputs if 'ncf' = True
    
    5) returns following outputs:
        spa - spathy object
        outf - filepath to output netCDF file.
        
    IN:
        pgen, pcpy, pbu, psoil - parameter dictionaries
        gisdata - dict of 2d np.arrays containing gis-data with keys:
            cmask - catchment mask; integers within np.Nan outside
            LAI_conif [m2m-2]
            LAI_decid [m2m-2]
            hc, canopy closure [m]
            fc, canopy closure fraction [-]
            soildata - dict
            flowacc - flow accumulation [units]
            slope - local surface slope [units]
            
            cellsize - gridcell size
            lon0 - x-grid
            lat0 - y-grid
        cpy_outputs, bu_, top_ - True saves cpy, bu and top outputs to lists within each object. 
            Use only for testing, memory issue!
        flatten - True flattens 2d arrys to 1d array containing only cells inside catchment
    OUT:
        spa - spathy object
    """

    # start_time = timeit.default_timer()

    # moved as input argument
    # read gis data and create necessary inputs for model initialization
    # gisdata = create_catchment(pgen['catchment_id'], fpath=pgen['gis_folder'],
    #                           plotgrids=False, plotdistr=False)

    # # preprocess soildata --> dict used in BucketModel initialization    
    # soildata = preprocess_soildata(pbu, psoil, gisdata['soilclass'], gisdata['cmask'], pgen['spatial_soil'])

    # inputs for CanopyGrid initialization: update pcpy using spatial data
    cstate = pcpy['state']
    cstate['lai_conif'] = gisdata['LAI_conif'] * gisdata['cmask']
    cstate['lai_decid_max'] = gisdata['LAI_decid'] * gisdata['cmask']
    cstate['cf'] = gisdata['cf'] * gisdata['cmask']
    cstate['hc'] = gisdata['hc'] * gisdata['cmask']
    
    for key in ['w', 'swe']:
        cstate[key] *= gisdata['cmask']

    pcpy['state'] = cstate
    del cstate
    
    """ greate SpatHy object """
    spa = SpaFHy(pgen, pcpy, soildata, gisdata, cpy_outputs=cpy_outputs,
                 bu_outputs=bu_outputs, flatten=flatten)
            
    #print('Loops total [s]: ', timeit.default_timer() - start_time)
    print('********* created SpaFHy instance *********')

    return spa



"""
******************************************************************************
            ----------- SpaFHy model class --------
******************************************************************************
"""


class SpaFHy():
    """
    SpaFHy model class
    """
    def __init__(self, pgen, pcpy, soildata, gisdata, cpy_outputs=False,
                 bu_outputs=False, flatten=False):

        self.dt = pgen['dt']  # s
        self.id = pgen['catchment_id']
        self.spinup_end = pgen['spinup_end']
        self.pgen = pgen
        self.step_nr = 0
        
        #Simulation results in netCDF4 file
        self.ncf_file = pgen['ncf_file']
                
        self.forc_file=pgen['forcing_file']
        self.GisData = gisdata
        self.cmask = self.GisData['cmask']
        self.gridshape = np.shape(self.cmask)
        cmask= self.cmask.copy()

        #flowacc = gisdata['flowacc'].copy()
        #slope = gisdata['slope'].copy()        
        
        """
        flatten=True omits cells outside catchment
        """
        if flatten:
            ix = np.where(np.isfinite(cmask))
            cmask = cmask[ix].copy()
            # sdata = sdata[ix].copy()
            #flowacc = flowacc[ix].copy()
            #slope = slope[ix].copy()
            
            for key in pcpy['state']:
                pcpy['state'][key] = pcpy['state'][key][ix].copy()
                        
            for key in soildata:
                soildata[key] = soildata[key][ix].copy()
                
            self.ix = ix  # indices to locate back to 2d grid

        """--- initialize CanopyGrid ---"""
        self.cpy = CanopyGrid(pcpy, pcpy['state'], outputs=cpy_outputs)

        """--- initialize BucketGrid ---"""
        self.bu = BucketGrid(spara=soildata, outputs=bu_outputs)


    def run_timestep(self, forc, ncf=False, flx=False):
        """ 
        Runs SpaFHy for one timestep starting from current state
        IN:
            forc - dictionary or pd.DataFrame containing forcing values for the timestep
            ncf - netCDF -file handle, for outputs
            flx - returns fluxes to caller as dict
        OUT:
            flx
        """
        doy = forc['doy']
        ta = forc['T']
        vpd = forc['VPD'] + eps
        rg = forc['Rg']
        par = forc['Par'] + eps
        prec = forc['Prec']
        co2 = forc['CO2']
        u = forc['U'] + eps

        swe0 = self.cpy.SWE.copy()
        
        # run CanopyGrid
        potinf, trfall, interc, evap, et, transpi, efloor, mbe, fw, fq, et0 = \
            self.cpy.run_timestep(doy, self.dt, ta, prec, rg, par, vpd, U=u, CO2=co2,
                                  beta=self.bu.Ree, Rew=self.bu.Rew, Psi=self.bu.Psi, P=101300.0)

        # run BucketGrid water balance
        infi, infi_ex, drain, tr, eva, mbes = self.bu.watbal(dt=self.dt, rr=1e-3*potinf, tr=1e-3*transpi,
                                                       evap=1e-3*efloor, retflow=0)



        """ outputs """

        if ncf:
            # writes to netCDF -file at every timestep; bit slow - should
            # accumulate into temporary variables and save every 10 days? 
            # for netCDF output, run in Flatten=True to save memory               
            k = self.step_nr
            
            # canopygrid
            ncf['W'][k,:,:] = self._to_grid(self.cpy.W)
            ncf['SWE'][k,:,:] = self._to_grid(self.cpy.SWE)
            ncf['Trfall'][k,:,:] = self._to_grid(trfall) 
            ncf['Potinf'][k,:,:] = self._to_grid(potinf)
            ncf['ET'][k,:,:] = self._to_grid(et)
            ncf['Transpi'][k,:,:] = self._to_grid(transpi)
            ncf['Efloor'][k,:,:] = self._to_grid(efloor)            
            ncf['Evap'][k,:,:] = self._to_grid(evap)
            ncf['Inter'][k,:,:] = self._to_grid(interc)
            ncf['CMbe'][k,:,:] = self._to_grid(mbe)              
            ncf['fW'][k,:,:] = self._to_grid(fw)
            ncf['fQ'][k,:,:] = self._to_grid(fq)
            
            # for climatological wetness indices
            # non-water limited total ET
            ncf['ET0'][k,:,:] = self._to_grid(et0 + evap)
            
            ncf['effPrec'][k,:,:] = self._to_grid(np.maximum(0.0, prec * self.dt - (self.cpy.SWE - swe0)))
        
            
            # bucketgrid
            ncf['Drain'][k,:,:] = self._to_grid(drain)
            ncf['Infil'][k,:,:] = self._to_grid(infi)
            ncf['Wliq'][k,:,:] = self._to_grid(self.bu.Wliq)
            ncf['Wliq_top'][k,:,:] = self._to_grid(self.bu.Wliq_top)            
            ncf['PondSto'][k,:,:] = self._to_grid(self.bu.PondSto)
            ncf['BMbe'][k,:,:] = self._to_grid(mbes)              
            ncf['Psi'][k,:,:] = self._to_grid(self.bu.Psi)
            ncf['Rew'][k,:,:] = self._to_grid(self.bu.Rew)

        # update step number
        self.step_nr += 1
        
        if flx:
            flx = {'cpy': {'ET': et, 'Transpi': transpi, 'Evap': evap,
                           'Efloor': efloor, 'Inter': interc, 'Trfall': trfall,
                           'Potinf': potinf},
                   'bu': {'Infil': infi, 'Drain': drain},
                  }
            return flx        

                         
    def _to_grid(self, x):
        """
        converts variable x back to original grid for NetCDF outputs
        """
        if self.ix:
            a = np.full(self.gridshape, np.NaN)
            a[self.ix] = x
        else: # for non-flattened, return
            a = x
        return a

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
    ncf.description = 'SpatHy results for VMI-Kas'
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
    W = ncf.createVariable('W', 'f4', ('dtime', 'dlat', 'dlon',))
    W.units = 'canopy storage [mm]'
    SWE = ncf.createVariable('SWE', 'f4', ('dtime', 'dlat', 'dlon',))
    SWE.units = 'snow water equiv. [mm]'
    Trfall = ncf.createVariable('Trfall', 'f4', ('dtime', 'dlat', 'dlon',))
    Trfall.units = 'throughfall [mm]'
    Inter = ncf.createVariable('Inter', 'f4', ('dtime', 'dlat', 'dlon',))
    Inter.units = 'interception [mm]'
    Potinf = ncf.createVariable('Potinf', 'f4', ('dtime', 'dlat', 'dlon',))
    Potinf.units = 'pot. infiltration [mm]'
    ET = ncf.createVariable('ET', 'f4', ('dtime', 'dlat', 'dlon',))
    ET.units = 'dry-canopy et. [mm]'
    Transpi = ncf.createVariable('Transpi', 'f4', ('dtime', 'dlat', 'dlon',))
    Transpi.units = 'transpiration [mm]'
    Efloor = ncf.createVariable('Efloor', 'f4', ('dtime', 'dlat', 'dlon',))
    Efloor.units = 'forest floor evap. [mm]'
    Evap = ncf.createVariable('Evap', 'f4', ('dtime', 'dlat', 'dlon',))
    Evap.units = 'interception evap. [mm]'
    CMbe = ncf.createVariable('CMbe', 'f4', ('dtime', 'dlat', 'dlon',))
    CMbe.units = 'Canopy mass-balance error [mm]'
    fW = ncf.createVariable('fW', 'f4', ('dtime', 'dlat', 'dlon',))
    fW.units = 'Soil-water modifier for transpiration [-]'
    fQ = ncf.createVariable('fQ', 'f4', ('dtime', 'dlat', 'dlon',))
    fQ.units = 'Combined LAI-PAR modifier for transpiration [-]'
    # for climate indices
    ET0 = ncf.createVariable('ET0', 'f4', ('dtime', 'dlat', 'dlon',))
    ET0.units = 'non soil-water restricted actual ET [mm d-1]'
    effPrec = ncf.createVariable('effPrec', 'f4', ('dtime', 'dlat', 'dlon',))
    effPrec.units = 'precpitation including snow storage change [mm d-1]'
    
    # BucketGrid outputs
    Wliq = ncf.createVariable('Wliq', 'f4', ('dtime', 'dlat', 'dlon',))
    Wliq.units = 'root zone vol. water cont. [m3m-3]'
    Wliq_top = ncf.createVariable('Wliq_top', 'f4', ('dtime', 'dlat', 'dlon',))
    Wliq_top.units = 'org. layer vol. water cont. [m3m-3]'
    Psi = ncf.createVariable('Psi', 'f4', ('dtime', 'dlat', 'dlon',))
    Psi.units = 'root zone water potential [MPa]'
    Rew = ncf.createVariable('Rew', 'f4', ('dtime', 'dlat', 'dlon',))
    Rew.units = 'Relative plant available water [-]'
    
    PondSto = ncf.createVariable('PondSto', 'f4', ('dtime', 'dlat', 'dlon',))
    PondSto.units = 'pond storage [mm]'
    Infil = ncf.createVariable('Infil', 'f4', ('dtime', 'dlat', 'dlon',))
    Infil.units = 'infiltration [mm]'
    Drain = ncf.createVariable('Drain', 'f4', ('dtime', 'dlat', 'dlon',))
    Drain.units = 'drainage [mm]'
    BMbe = ncf.createVariable('BMbe', 'f4', ('dtime', 'dlat', 'dlon',))
    BMbe.units = ' Bucket mass-balance error [mm]'

    # topmodel outputs
    # Qt = ncf.createVariable('Qt', 'f4', ('dtime',))
    # Qt.units = 'streamflow[m]'
    # Qb = ncf.createVariable('Qb', 'f4', ('dtime',))
    # Qb.units = 'baseflow [m]'
    # Qr = ncf.createVariable('Qr', 'f4', ('dtime',))
    # Qr.units = 'returnflow [m]'
    # Qs = ncf.createVariable('Qs', 'f4', ('dtime',))
    # Qs.units = 'surface runoff [m]'
    # R = ncf.createVariable('R', 'f4', ('dtime',))
    # R.units = 'average recharge [m]'
    # S = ncf.createVariable('S', 'f4', ('dtime',))
    # S.units = 'average sat. deficit [m]'
    # fsat = ncf.createVariable('fsat', 'f4', ('dtime',))
    # fsat.units = 'saturated area fraction [-]'
    # #This addition is for the saturation map
    # Sloc = ncf.createVariable('Sloc', 'f4', ('dtime','dlat','dlon',))
    # Sloc.units = 'local sat. deficit [m]'
    
    # gisdata
    soilclass = ncf.createVariable('soilclass', 'f4', ('dlat', 'dlon',))
    soilclass.units = 'soil type code [int]'
    sitetype = ncf.createVariable('sitetype', 'f4', ('dlat', 'dlon',))
    sitetype.units = 'sitetype [0 = peat, 1 = herb-rich, 2 = mesic, 3 = sub-xeric, 4 = xeric]'
    twi = ncf.createVariable('twi', 'f4', ('dlat', 'dlon',))
    twi.units = 'twi'
    LAI_conif = ncf.createVariable('LAI_conif', 'f4', ('dlat', 'dlon',))
    LAI_conif.units = 'LAI_conif [m2m-2]'
    LAI_decid = ncf.createVariable('LAI_decid', 'f4', ('dlat', 'dlon',))
    LAI_decid.units = 'LAI_decid [m2m-2]'
    hc = ncf.createVariable('hc', 'f4', ('dlat', 'dlon',))
    hc.units = 'canopy height hc [m]'
    ba = ncf.createVariable('ba', 'f4', ('dlat', 'dlon',))
    ba.units = 'basal area [m2ha-1]'
    Nstems = ncf.createVariable('Nstems', 'f4', ('dlat', 'dlon',))
    Nstems.units = 'trees per ha [ha-1]'
    
    #stream = ncf.createVariable('/gis/stream', 'f4', ('dlat', 'dlon',))
    #stream.units = 'stream mask'
    #dem = ncf.createVariable('/gis/dem', 'f4', ('dlat', 'dlon',))
    #dem.units = 'dem'
    #slope = ncf.createVariable('/gis/slope', 'f4', ('dlat', 'dlon',))
    #slope.units = 'slope'
    #flowacc = ncf.createVariable('/gis/flowacc', 'f4', ('dlat', 'dlon',))
    #flowacc.units = 'flowacc'
    #cmask = ncf.createVariable('/gis/cmask', 'f4', ('dlat', 'dlon',))
    #cmask.units = 'cmask'
    
    print('**** netCDF4 file created ****')
    return ncf, fname
