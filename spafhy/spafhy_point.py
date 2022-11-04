# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 13:13:45 2017

@author: slauniai

spafhy_point: combines canopygrid and bucketgid for solving point-scale water balance and -fluxes
v. 260618 / Samuli
"""
import numpy as np
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

from canopygrid_new import CanopyGrid
from bucketgrid import BucketGrid
from iotools import read_FMI_weather, read_HydeDaily

eps = np.finfo(float).eps  # machine epsilon

def test_at_hyde():
    # run for hyde
    from spafhy_parameters import parameters_hyde
    
    # read parameters
    pgen, pcpy, pbu = parameters_hyde()
    
    fname = r'c:\datat\spathydata\HydeDaily2000-2010.txt'
    dat, FORC = read_HydeDaily(fname)
    FORC['Prec'] = FORC['Prec'] / pgen['dt']  # mms-1
    FORC['T'] = FORC['Ta'].copy()
    cmask = np.ones(1)

    # create model
    model = SpaFHy_point(pgen, pcpy, pbu, cmask, FORC, cpy_outputs=True, bu_outputs=True)
    
    # run model for different parameter combinations, save results into dataframe
    # +/- 15%, 15%, 20%, 30%
    # amax, g1_conif, wmax, wmaxsnow
    p = [[10.0, 2.1, 1.3, 4.5],
         [8.5, 1.7, 1.05, 3.15],
         [11.5, 2.5, 2.6, 5.4]]

    out = []
    for k in range(3):
        a = p[k]
        pcpy['amax'] = a[0]
        pcpy['g1_conif'] = a[1]
        pcpy['g1_decid'] = 1.6*a[1]
        pcpy['wmax'] = a[2]
        pcpy['wmaxsnow'] = a[3]
        
        model = SpaFHy_point(pgen, pcpy, pbu, cmask, FORC, cpy_outputs=True, bu_outputs=True)
        nsteps=len(FORC)
        model._run(0, nsteps)
        
        out.append(model)
        del model
        
    # best model
    Wliq_mod = np.ravel(out[0].bu.results['Wliq'])
    Wliq_low = np.ravel(out[1].bu.results['Wliq'])
    Wliq_high = np.ravel(out[2].bu.results['Wliq'])
    ET_mod = np.ravel(out[0].cpy.results['ET']) 
    ET_low = np.ravel(out[1].cpy.results['ET'])
    ET_high = np.ravel(out[2].cpy.results['ET']) 
    
    E_mod = np.ravel(out[0].cpy.results['Evap']) 
    E_low = np.ravel(out[1].cpy.results['Evap'])
    E_high = np.ravel(out[2].cpy.results['Evap'])

#    SWC_mod = np.ravel(out[0].cpy.results['ET']) 
#    SWC_low = np.ravel(out[1].cpy.results['ET'])
#    SWC_high = np.ravel(out[2].cpy.results['ET'])) 
    
    SWCa = dat['SWCa']
    SWCb = dat['SWCb']
    SWCc = dat['SWCc']
    tvec = dat.index
    et_dry = dat['ET']
    et_dry[dat['Prec']>0.1] = np.NaN
    
    sns.set_style('whitegrid')
    with sns.color_palette('muted'):
        plt.figure()
        
        plt.subplot(2,3,(1,2))
                
        plt.plot(tvec, et_dry, 'o', markersize=4, alpha=0.3, label='meas')
        plt.fill_between(tvec, ET_low, ET_high, facecolor='grey', alpha=0.6, label='range')
        plt.plot(tvec, ET_mod, 'k-', alpha=0.4, lw=0.5, label='mod')
        #plt.xlim([pd.datetime(2003, 10, 1), pd.datetime(2011,1,1)])
        plt.legend(loc=2, fontsize=8)
        plt.setp(plt.gca().get_xticklabels(), fontsize=8)
        plt.setp(plt.gca().get_yticklabels(), fontsize=8)
        plt.ylabel('ET$_{dry}$ (mm d$^{-1}$)', fontsize=8)
        plt.ylim([-0.05, 5.0])
        plt.xlim([pd.datetime(2002, 1, 1), pd.datetime(2011,1,1)])

        # sns.despine()        
        plt.subplot(2,3,(4,5))

        plt.plot(tvec, SWCa, 'o', markersize=4, alpha=0.3,label='meas')
        plt.fill_between(tvec, Wliq_low, Wliq_high, facecolor='grey', alpha=0.6, label='range')
        plt.plot(tvec, Wliq_mod, 'k-',alpha=0.4, lw=0.5, label='mod')        
        #plt.xlim([pd.datetime(2003, 10, 1), pd.datetime(2011,1,1)])
        plt.legend(loc=2, fontsize=8)
        plt.setp(plt.gca().get_xticklabels(), fontsize=8)
        plt.setp(plt.gca().get_yticklabels(), fontsize=8)
        plt.ylabel('$\\theta$ (m$^3$ m$^{-3}$)', fontsize=8)
        plt.xlim([pd.datetime(2002, 1, 1), pd.datetime(2011,1,1)])

        # scatterplot
        plt.subplot(2,3,6)

        meas = np.array(SWCa.values.tolist())
        slope, intercept, r_value, p_value, std_err = stats.linregress(meas, Wliq_mod)
        #print slope, intercept
        xx = np.array([min(meas), max(meas)])
        plt.plot(meas, Wliq_mod, 'o', markersize=5, alpha=0.3)
        plt.plot(xx, slope*xx + intercept, 'k-')
        plt.plot([0.05, 0.45], [0.05, 0.45], 'k--')
        plt.text( 0.07, 0.42, 'y = %.2f x + %.2f' %(slope, intercept), fontsize=8)
        plt.xlim([0.05, 0.45]); plt.ylim([0.05, 0.45])
        
        ax = plt.gca()
        ax.set_yticks([0.1, 0.2, 0.3, 0.4])
        ax.set_xticks([0.1, 0.2, 0.3, 0.4])
        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()
        plt.ylabel('$\\theta$ mod (m$^3$ m$^{-3}$)', fontsize=8)
        
        plt.setp(plt.gca().get_yticklabels(), fontsize=8)
        plt.setp(plt.gca().get_yticklabels(), fontsize=8)
        plt.setp(plt.gca().get_xticklabels(), fontsize=8)
        plt.xlabel('$\\theta$ meas (m$^3$ m$^{-3}$)', fontsize=8)
        
        # scatterplot
        plt.subplot(2,3,3)

        meas = np.array(et_dry.values.tolist())
        ix = np.where(np.isfinite(meas))
        meas=meas[ix].copy()
        mod = ET_mod[ix].copy()
        slope, intercept, r_value, p_value, std_err = stats.linregress(meas, mod)
        xx = np.array([min(meas), max(meas)])
        plt.plot(meas, mod, 'o', markersize=4, alpha=0.3)

        plt.plot(xx, slope*xx + intercept, 'k-')
        plt.plot([0, 5], [0, 5], 'k--')
        plt.text(0.3, 4.2, 'y = %.2f x + %.2f' %(slope, intercept), fontsize=8)

        plt.xlim([-0.01, 5]); plt.ylim([-0.01, 5])
        plt.setp(plt.gca().get_xticklabels(), fontsize=8)
        ax = plt.gca()
        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()
        plt.ylabel('ET$_{dry}$ mod (mm d$^{-1}$)', fontsize=8)
        plt.setp(plt.gca().get_yticklabels(), fontsize=8)
        plt.xlabel('ET$_{dry}$ meas (mm d$^{-1}$)', fontsize=8)
        
        #plt.savefig('Hyde_validate.pdf')
        #plt.savefig('Hyde_validate.png')
        
        # snowpack and throughfall
        
        plt.figure()
        SWE_mod = np.ravel(out[0].cpy.results['SWE'])        
        SWE_low = np.ravel(out[1].cpy.results['SWE'])
        SWE_hi = np.ravel(out[2].cpy.results['SWE'])
        swe_meas = dat['SWE']

        plt.plot(tvec, swe_meas, 'o', markersize=10, alpha=0.3, label='meas')
        plt.fill_between(tvec, SWE_low, SWE_hi, facecolor='grey', alpha=0.6, label='range')
        plt.plot(tvec, SWE_mod, 'k-', alpha=0.4, lw=0.5, label='mod')
        plt.title('SWE'); plt.ylabel('SWE mm')

    return out, dat, FORC
 
class SpaFHy_point():
    """
    SpaFHy for point-scale simulation. Couples Canopygrid and Bucketdrid -classes
    """
    def __init__(self, pgen, pcpy, pbu, cmask, FORC, cpy_outputs=True, bu_outputs=True):

        self.dt = pgen['dt']  # s
        self.id = pgen['catchment_id']
        self.spinup_end = pgen['spinup_end']
        self.pgen = pgen
    
        self.FORC = FORC 
        self.Nsteps = len(self.FORC)

        """--- initialize CanopyGrid and BucketGrid ---"""
        self.cpy = CanopyGrid(pcpy, cmask=cmask, outputs=cpy_outputs)
        self.bu = BucketGrid(pbu, cmask=cmask, outputs=bu_outputs)
                          
    def _run(self, fstep, Nsteps, soil_feedbacks=True):
        """ 
        Runs SpaFHy_point
        IN:
            fstep - index of starting point [int]
            Nsteps - number of timesteps [int]
            soil_feedbacks - False sets REW and REE = 1 and ignores feedback from soil state
        OUT:
            updated state, saves
        """
        dt = self.dt

        for k in range(fstep, fstep + Nsteps):
            print 'k=' + str(k)

            # forcing
            doy = self.FORC['doy'].iloc[k]; ta = self.FORC['T'].iloc[k]
            vpd = self.FORC['VPD'].iloc[k]; rg = self.FORC['Rg'].iloc[k]
            par = self.FORC['Par'].iloc[k]; prec = self.FORC['Prec'].iloc[k]
            co2 = self.FORC['CO2'].iloc[k]; u = self.FORC['U'].iloc[k]
            if not np.isfinite(u):
                u = 2.0            

            if soil_feedbacks:
                beta0 = self.bu.Ree # affects surface evaporation
                rew0 = self.bu.Rew # affects transpiration
            else:
                beta0 = 1.0
                rew0 = 1.0

            # run CanopyGrid
            potinf, trfall, interc, evap, et, transpi, efloor, mbe = \
                self.cpy.run_timestep(doy, dt, ta, prec, rg, par, vpd, U=u, CO2=co2,
                                      beta=beta0, Rew=rew0, P=101300.0)

            # run BucketGrid water balance
            infi, infi_ex, drain, tr, eva, mbes = self.bu.watbal(dt=dt, rr=1e-3*potinf, tr=1e-3*transpi,
                                                           evap=1e-3*efloor, retflow=0.0)


    def _calibration_run(self, fstep, Nsteps, evalvar, evaldates):
        """ 
        Runs SpaFHy_point for parameter optimization
        IN:
            fstep - index of starting point [int]
            Nsteps - number of timesteps [int]
            evalvar, evadates relate to evaluation data
        OUT:
            updated state.
        """
        res = {'ET': [None]*self.Nsteps, 'Efloor': [None]*self.Nsteps, 'Evap': [None]*self.Nsteps,
               'Transpi': [None]*self.Nsteps, 'Trfall': [None]*self.Nsteps, 'SWE':[None]*self.Nsteps,
               'fQ': [None]*self.Nsteps,'fD': [None]*self.Nsteps,
                'fRew': [None]*self.Nsteps, 'fRee': [None]*self.Nsteps, 'Wliq': [None]*self.Nsteps}
        
        dt = self.dt

        for k in range(fstep, fstep + Nsteps):
            # print 'k=' + str(k)

            # forcing
            doy = self.FORC['doy'].iloc[k]; ta = self.FORC['T'].iloc[k]
            vpd = self.FORC['VPD'].iloc[k]; rg = self.FORC['Rg'].iloc[k]
            par = self.FORC['Par'].iloc[k]; prec = self.FORC['Prec'].iloc[k]
            co2 = self.FORC['CO2'].iloc[k]; u = self.FORC['U'].iloc[k]            

            beta0 = self.bu.Ree
            rew0 = self.bu.Rew

            # run CanopyGrid
            potinf, trfall, interc, evap, et, transpi, efloor, mbe = \
                self.cpy.run_timestep(doy, dt, ta, prec, rg, par, vpd, U=u, CO2=co2,
                                      beta=beta0, Rew=rew0, P=101300.0)

            # run BucketGrid water balance
            infi, infi_ex, drain, tr, eva, mbes = self.bu.watbal(dt=dt, rr=1e-3*potinf, tr=1e-3*transpi,
                                                           evap=1e-3*efloor, retflow=0.0)

            res['ET'][k] = et[0]
            res['Wliq'][k] = self.bu.Wliq[0]
            res['Trfall'][k] = trfall[0]
            res['SWE'][k] = self.cpy.SWE[0]
            
    
        res = pd.DataFrame(res)
        res.index = self.FORC.index
        simudata = res[evalvar].ix[evaldates].values.tolist()
        
        return simudata
    

    

#def initialize_netCDF(ID, gis, forc, fpath='results', fname=None):
#    """
#    NOT WORKING NOW !!!
#    SpaFHy netCDF4 format output file initialization
#    IN:
#        ID -catchment id as int or str
#        gis - GisData dict
#        forc - forcing data (pd.dataframe)
#        roff - measured runoff (pd.Series)
#        fpath - path for saving results
#        fname - filename
#    OUT:
#        ncf - netCDF file handle. Initializes data
#        ff - netCDF filename incl. path
#    """
#    from netCDF4 import Dataset, date2num  # , num2date
#    from datetime import datetime
#
#    # dimensions
#    dlat, dlon = np.shape(gis['cmask'])
#    dtime = None
#
#    if fname:
#        ff = os.path.join(spathy_path, fpath, fname)
#        print ff
#    else:
#        ff = os.path.join(spathy_path, fpath, 'Spathy_ch' + str(ID) + '.nc')
#
#    # create dataset & dimensions
#    ncf = Dataset(ff, 'w')
#    ncf.description = 'SpatHy results. Catchment : ' + str(ID)
#    ncf.history = 'created ' + datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#    ncf.source = 'SpatHy -model v.0.99'
#
#    ncf.createDimension('dtime', dtime)
#    ncf.createDimension('dlon', dlon)
#    ncf.createDimension('dlat', dlat)
#
#    # create variables into base and groups 'forc','eval','cpy','bu','top'
#    # call as createVariable(varname,type,(dimensions))
#    time = ncf.createVariable('time', 'f8', ('dtime',))
#    time.units = "days since 0001-01-01 00:00:00.0"
#    time.calendar = 'standard'
#
#    lat = ncf.createVariable('lat', 'f4', ('dlat',))
#    lat.units = 'ETRS-TM35FIN'
#    lon = ncf.createVariable('lon', 'f4', ('dlon',))
#    lon.units = 'ETRS-TM35FIN'
#
#    tvec = [k.to_datetime() for k in forc.index]
#    time[:] = date2num(tvec, units=time.units, calendar=time.calendar)
#    lon[:] = gis['lon0']
#    lat[:] = gis['lat0']
#
#
#    # CanopyGrid outputs
#    W = ncf.createVariable('/cpy/W', 'f4', ('dtime', 'dlat', 'dlon',))
#    W.units = 'canopy storage [mm]'
#    SWE = ncf.createVariable('/cpy/SWE', 'f4', ('dtime', 'dlat', 'dlon',))
#    SWE.units = 'snow water equiv. [mm]'
#    Trfall = ncf.createVariable('/cpy/Trfall', 'f4', ('dtime', 'dlat', 'dlon',))
#    Trfall.units = 'throughfall [mm]'
#    Inter = ncf.createVariable('/cpy/Inter', 'f4', ('dtime', 'dlat', 'dlon',))
#    Inter.units = 'interception [mm]'
#    Potinf = ncf.createVariable('/cpy/Potinf', 'f4', ('dtime', 'dlat', 'dlon',))
#    Potinf.units = 'pot. infiltration [mm]'
#    ET = ncf.createVariable('/cpy/ET', 'f4', ('dtime', 'dlat', 'dlon',))
#    ET.units = 'dry-canopy et. [mm]'
#    Transpi = ncf.createVariable('/cpy/Transpi', 'f4', ('dtime', 'dlat', 'dlon',))
#    Transpi.units = 'transpiration [mm]'
#    Efloor = ncf.createVariable('/cpy/Efloor', 'f4', ('dtime', 'dlat', 'dlon',))
#    Efloor.units = 'forest floor evap. [mm]'
#    Evap = ncf.createVariable('/cpy/Evap', 'f4', ('dtime', 'dlat', 'dlon',))
#    Evap.units = 'interception evap. [mm]'
#    Mbe = ncf.createVariable('/cpy/Mbe', 'f4', ('dtime', 'dlat', 'dlon',))
#    Mbe.units = 'mass-balance error [mm]'
#
#    # BucketGrid outputs
#    Wliq = ncf.createVariable('/bu/Wliq', 'f4', ('dtime', 'dlat', 'dlon',))
#    Wliq.units = 'vol. water cont. [m3m-3]'
#    Wsto = ncf.createVariable('/bu/Wsto', 'f4', ('dtime', 'dlat', 'dlon',))
#    Wsto.units = 'water storage [mm]'
#    SurSto = ncf.createVariable('/bu/SurSto', 'f4', ('dtime', 'dlat', 'dlon',))
#    SurSto.units = 'pond storage [mm]'
#    Infil = ncf.createVariable('/bu/Infil', 'f4', ('dtime', 'dlat', 'dlon',))
#    Infil.units = 'infiltration [mm]'
#    Drain = ncf.createVariable('/bu/Drain', 'f4', ('dtime', 'dlat', 'dlon',))
#    Drain.units = 'drainage [mm]'
#    Mbe = ncf.createVariable('/bu/Mbe', 'f4', ('dtime', 'dlat', 'dlon',))
#    Mbe.units = 'mass-balance error [mm]'
#    Infiex = ncf.createVariable('/bu/Infiex', 'f4', ('dtime',))
#    Infiex.units = 'infiltration excess runoff [mm]'
#    Roff = ncf.createVariable('/bu/Roff', 'f4', ('dtime',))
#    Roff.units = 'total runoff[mm]'
#
#    return ncf, ff
