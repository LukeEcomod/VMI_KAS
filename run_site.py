# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 13:46:03 2022

@author: 03081268
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.colors as mplcolors
import matplotlib.cm as mplcm

import seaborn as sns

from netCDF4 import Dataset#, date2num, num2date
from spafhy import spafhy

from spafhy.spafhy_io import read_FMI_weather#, combine_spafhy_nc

from spafhy.spafhy_parameters import parameters, soil_properties_from_sitetype, soil_properties
from spafhy.spafhy_io import preprocess_soildata, sitetype_to_soilproperties
from spafhy.spafhy_io import create_catchment, read_AsciiGrid, write_AsciiGrid

eps = np.finfo(float).eps

""" paths defined in parameters"""

# read parameters

datafolder = r'data/C14/'
#forcingfile = r'data/SVE_saa.csv'
outfile = r'results/C14_test.nc'
site_id = 14

gisdata = create_catchment(datafolder, plotgrids=True, plotdistr=False)

# test with no decid
#gisdata['LAI_decid'] = np.ones(np.shape(gisdata['LAI_decid']))*0.01

""" set up SpaFHy for the site """

# load parameter dictionaries
(pgen, pcpy, pbu, ptop) = parameters()

# preprocess soildata --> dict used in BucketModel initialization    
# psoil = soil_properties()
#soildata = preprocess_soildata(pbu, psoil, gisdata['soilclass'], gisdata['cmask'], pgen['spatial_soil'])

psoils = soil_properties_from_sitetype()
soildata = sitetype_to_soilproperties(pbu, psoils, gisdata['sitetype'], gisdata['cmask'], pgen['spatial_soil'])

""" read forcing data and catchment runoff file """
FORC = read_FMI_weather(site_id,
                        pgen['start_date'],
                        pgen['end_date'],
                        sourcefile=pgen['forcing_file'])

FORC['Prec'] = FORC['Prec'] / pgen['dt']  # mms-1
FORC['U'] = 2.0 # use constant wind speed ms-1
Nsteps = len(FORC)

#%%
# initialize spafhy

spa = spafhy.initialize(pgen, pcpy, pbu, ptop, soildata.copy(), gisdata.copy(), cpy_outputs=False, 
                        bu_outputs=False, top_outputs=False, flatten=True)
# create netCDF output file

dlat, dlon = np.shape(spa.GisData['cmask'])

ncf, ncf_file = spafhy.initialize_netCDF(ID=spa.id, fname=outfile, lat0=spa.GisData['lat0'], 
                                         lon0=spa.GisData['lon0'], dlat=dlat, dlon=dlon, dtime=None)

# run spafhy for spinup
Nspin = np.where(FORC.index == pgen['spinup_end'])[0][0]

for k in range(0, Nspin):
    forc= FORC[['doy', 'Rg', 'Par', 'T', 'Prec', 'VPD', 'CO2','U']].iloc[k]
    spa.run_timestep(forc, ncf=False)

spa.step_nr = 0

# run spafhy for Nsteps
for k in range(Nspin, Nsteps):
    forc= FORC[['doy', 'Rg', 'Par', 'T', 'Prec', 'VPD', 'CO2','U']].iloc[k]
    
    spa.run_timestep(forc, ncf=ncf)

# append gisdata to ncf
for v in list(ncf['gis'].variables):
    ncf['gis'][v][:,:] = gisdata[v]
    
# close output file
ncf.close()

#del spa, ncf, ncf_file, gisdata

#%%  

from forests_ms import REW, fREW
res = Dataset(ncf_file, 'r')

cres = res['cpy']
bres = res['bu']
forc = FORC.iloc[Nspin:Nsteps]
tvec = forc.index


twi = gisdata['twi']
LAIc = gisdata['LAI_conif']
LAId = gisdata['LAI_decid']
LAIt = LAIc + LAId
soil = gisdata['gtk_soilcode']
sitetype = gisdata['sitetype']
TWI = gisdata['twi']

#sts = np.unique(sitetype)
#sts = sts[sts>0]

# soil type indexes
peat_ix = np.where(soil == 4)
med_ix = np.where(soil == 2)
coarse_ix = np.where(soil == 1)

# indices for high and low twi
htwi_ix = np.where(twi > 12)
ltwi_ix = np.where(twi < 7)

##
Wliq = bres['Wliq'][:,:,:]
Wliq_top = bres['Wliq_top'][:,:,:]

# relative plant available water
Rew = REW(Wliq, soildata['fc'], soildata['wp'])

# transpiration soil moisture modifier
fRew = fREW(Rew, rw=0.20, rwmin=0.05)
ix = np.where((tvec.month >= 5) & (tvec.month <=9))[0]

a = np.array(Rew[ix,:,:])
meanRew = np.mean(a, axis=0)

b = np.array(fRew[ix,:,:])

drisk = np.mean(b, axis=0)

Wm = np.mean(Wliq[ix,:,:], axis=0)
Wtopm = np.mean(Wliq_top[ix,:,:], axis=0)

Wdry = np.percentile(Wliq[ix,:,:], 10.0, axis=0)
Wtop_dry = np.percentile(Wliq_top[ix,:,:], 10.0, axis=0)

plt.figure()
plt.subplot(121); 
plt.imshow(meanRew); plt.colorbar()
plt.subplot(122)
plt.imshow(drisk); plt.colorbar()


#%% plot figure
fig = plt.figure()
fig.set_size_inches(8.0, 10.0)

#cm = plt.get_cmap('tab10', 5)
cm = plt.get_cmap('Paired')
cc = cm.colors[1:6]

cm = mplcolors.ListedColormap(cc)
# maps to lhs of the figure

plt.subplot(321)
sns.heatmap(LAIt, cmap='coolwarm', alpha=1.0, cbar=True, xticklabels=False, yticklabels=False); 
plt.title('LAI (m$^2$m$^{-2}$)')

plt.subplot(322)
cax = sns.heatmap(sitetype, cmap=cm, alpha=1.0, cbar=True, xticklabels=False, yticklabels=False); 
cb1 = cax.collections[0].colorbar
cb1.set_ticks([0.4, 1.2, 2, 2.8, 3.6])
cb1.set_ticklabels(['peatland','herb-rich','mesic','sub-xeric', 'xeric'])
plt.title('sitetype')

plt.subplot(323)
sns.heatmap(drisk, cmap='coolwarm_r', alpha=1.0, cbar=True, xticklabels=False, yticklabels=False); 
plt.title('f_{w} (-)')

plt.subplot(325)
x = np.ravel(LAIt)
y = np.ravel(meanRew)
rr = sitetype.copy()
norm = mplcolors.BoundaryNorm(np.arange(0, 6) - 0.5, 5)
plt.scatter(x, y, c=rr, cmap=cm, norm=norm, s=10, alpha=0.7)
cb1 = plt.colorbar(ticks=np.arange(0, 6))
cb1.ax.set_yticklabels(['peatland','herb-rich','mesic','sub-xeric', 'xeric']) 

plt.subplot(326)
x = np.ravel(LAIt)
y = np.ravel(drisk)
rr = sitetype.copy()
norm = mplcolors.BoundaryNorm(np.arange(1, 7) - 0.5, 5)
plt.scatter(x, y, c=rr, cmap=cm, norm=norm, s=10, alpha=0.7)
cb1 = plt.colorbar(ticks=np.arange(0, 6))
cb1.ax.set_yticklabels(['peatland','herb-rich','mesic','sub-xeric', 'xeric']) 



#%% Make binned boxplot of drought-risk as a function of LAI for different site types

cm = plt.get_cmap('Paired')
cc = cm.colors[1:6]
Lb = np.arange(0,8,1)

fig, ax = plt.subplots()
dd = np.digitize(LAIt, Lb)
for s in [0,1,2,3,4]:
    D = []
    L = []
    for k in np.unique(dd):
        f = np.where((dd == k) & (sitetype == s))
        a = np.ravel(drisk[f])
        #a = a[a > 0]
        l = np.ravel(LAIt[f]) 
        D.append(a)
        L.append(l)
    
    bp = ax.boxplot(D, patch_artist=True, positions=Lb + 0.2*s, widths=0.2, showfliers=False, showcaps=False, 
                    medianprops={'color': 'k', 'linewidth':1.0})
    ax.set_xticks(Lb+0.25)
    ax.set_xticklabels(Lb + 0.5)
    for b in bp['boxes']:
        b.set_facecolor(cc[s])

stypes = ['peatland','herb-rich','mesic','sub-xeric', 'xeric']
y = [0.86, 0.84, 0.82, 0.80, 0.78]
for s in [0,1,2,3,4]:
    ax.plot(0.2, y[s], 'o', markersize=12, color=cc[s])
    ax.text(0.4, y[s]-0.002, stypes[s], fontsize=10)

ax.set_ylabel('f$_{w}$ (-)')
ax.set_xlabel('LAI (m$^2$m$^{-2}$)')
ax.set_xlim([0, 8])

plt.savefig(r'results/Droughtrisk_bins.png', dpi=300)

#%% plot intra-annual variability of soil moisture at each site type
import matplotlib.dates as mdates

slabel = ['a) peatland','b) herb-rich','c) mesic','d) sub-xeric', 'e) xeric']
fc = [0.41, 0.34, 0.28, 0.24, 0.14]
wp = [0.11, 0.11, 0.08, 0.08, 0.04]

cm = plt.get_cmap('Paired')
cc = cm.colors[1:6]

fig, ax = plt.subplots(5,1,figsize=(6.2,8.78)) # A4 portrait 75%

DF = []
L = []
# loop for sitetype
for s in [0,1,2,3,4]:
    ix = np.where(sitetype == s)
    d = Wliq[:,ix[0], ix[1]] #subset
    l = LAIt[ix[0], ix[1]]
    
    df = pd.DataFrame(index = tvec, data = d)
    DF.append(df)
    L.append(l)
    
    # now plot
    grouper = df.groupby(df.index.dayofyear)
    
    wm = grouper.mean()
    
    M, q25, q75, mini, maxi = wm.median(axis=1).values, wm.quantile(0.25,axis=1).values,\
                              wm.quantile(0.75, axis=1).values, wm.quantile(0.025, axis=1).values, wm.quantile(0.975, axis=1).values

    
    t = wm.index
    ax[s].fill_between(t, mini, maxi, color=cc[s], alpha=0.3)
    ax[s].fill_between(t, q25, q75, color=cc[s], alpha=1)
    ax[s].plot(t, M, '-', color='k')
    ax[s].set_ylabel(r'$\theta$ (m$^{3}$m$^{-3}$)')
    
    ax[s].plot(t, fc[s]*np.ones(len(t)), 'k--', linewidth=1)
    ax[s].plot(t, wp[s]*np.ones(len(t)), 'k--', linewidth=1)
    
    ax[s].set_xlim([0,366])
    
    if s == 0:
        ax[s].set_ylim([0.0, 0.9])
    else:
        ax[s].set_ylim([0.0, 0.6])
        
    ax[s].text(0.02, 0.85, slabel[s],
                horizontalalignment='left',
                verticalalignment='center',
                fontsize=12,
                transform = ax[s].transAxes)
    ax[s].tick_params(axis='both', direction='in')
    ax[s].set_xticks([0.0, 30, 60, 90, 120, 150, 180, 210, 230, 260, 290, 330, 360])
    
    if s < 4:
        ax[s].set_xticklabels([])
    
ax[4].set_xticklabels(ax[4].get_xticks(), rotation=45.)    
ax[4].xaxis.set_major_formatter(mdates.DateFormatter('%b'))

plt.tight_layout(pad=0.4, w_pad=0.1, h_pad=0.1)

plt.savefig(r'results/MeanMoisture.png', dpi=300)

#%% plot intra-annual variability of soil moisture at WET/DRY PIXELS AT EACH SITETYPE
import matplotlib.dates as mdates

slabel = ['a) peatland','b) herb-rich','c) mesic','d) sub-xeric', 'e) xeric']
fc = [0.41, 0.34, 0.28, 0.24, 0.14]
wp = [0.11, 0.11, 0.08, 0.08, 0.04]

cm = plt.get_cmap('Paired')
cc = cm.colors[1:6]

fig, ax = plt.subplots(5,1,figsize=(6.2,8.78)) # A4 portrait 75%

DF = []
L = []
# loop for sitetype
for s in [0,1,2,3,4]:
    
    ix = np.where(sitetype == s)
    
    d = np.array(Wliq[:,ix[0], ix[1]]) #subset
    
    ixw = np.where(np.mean(d, axis=0) > np.quantile(np.mean(d, axis=0), 0.95, axis=0))[0][0]
    ixd = np.where(np.mean(d, axis=0) < np.quantile(np.mean(d, axis=0), 0.05, axis=0))[0][0]
    
    dfd = pd.DataFrame(index = tvec, data = d[:,ixd])
    dfw = pd.DataFrame(index = tvec, data = d[:,ixw]) 
    
    # now plot
    gd = dfd.groupby(dfd.index.dayofyear)
    gw = dfw.groupby(dfw.index.dayofyear)
    
    M, q25, q75, mini, maxi = gd.median().values, gd.quantile(0.25).values,\
                              gd.quantile(0.75).values, gd.quantile(0.025).values, gd.quantile(0.975).values

    Mw, q25w, q75w, miniw, maxiw = gw.median().values, gw.quantile(0.25).values,\
                              gw.quantile(0.75).values, gw.quantile(0.025).values, gw.quantile(0.975).values
    
    t = gd.median().index
    # ax[s].fill_between(t, miniw.ravel(), maxiw.ravel(), color='k', alpha=0.1)
    # ax[s].fill_between(t, q25w.ravel(), q75w.ravel(), color='k', alpha=0.3)
    # ax[s].plot(t, Mw, '-', color='k', linewidth=1)
    
    ax[s].fill_between(t, mini.ravel(), maxi.ravel(), color=cc[s], alpha=0.5)
    ax[s].fill_between(t, q25.ravel(), q75.ravel(), color=cc[s], alpha=1)
    ax[s].plot(t, M, '-', color='k')
    ax[s].set_ylabel(r'$\theta$ (m$^{3}$m$^{-3}$)')
    
    ax[s].plot(t, fc[s]*np.ones(len(t)), 'k--', linewidth=1)
    ax[s].plot(t, wp[s]*np.ones(len(t)), 'k--', linewidth=1)
    
    ax[s].set_xlim([0,366])
    
    if s == 0:
        ax[s].set_ylim([0.0, 0.9])
    else:
        ax[s].set_ylim([0.0, 0.6])
        
    ax[s].text(0.02, 0.8, slabel[s],
                horizontalalignment='left',
                verticalalignment='center',
                fontsize=12,
                transform = ax[s].transAxes)
    ax[s].tick_params(axis='both', direction='in')
    ax[s].set_xticks([0.0, 30, 60, 90, 120, 150, 180, 210, 230, 260, 290, 330, 360])
    
    if s < 4:
        ax[s].set_xticklabels([])
    
ax[4].set_xticklabels(ax[4].get_xticks(), rotation=45.)    
ax[4].xaxis.set_major_formatter(mdates.DateFormatter('%b'))

plt.tight_layout(pad=0.4, w_pad=0.1, h_pad=0.1)

plt.savefig(r'results/Moisture_dynamics_extremegridcells.png', dpi=300)

#%% ***** THESE ARE FOR FUEL MOISTURE IDEAS: plot figure on mean soil moisture
fig = plt.figure()
fig.set_size_inches(8.0, 10.0)

#cm = plt.get_cmap('tab10', 5)
cm = plt.get_cmap('Paired')
cc = cm.colors[1:6]

cm = mplcolors.ListedColormap(cc)
# maps to lhs of the figure

plt.subplot(321)
sns.heatmap(LAIt, cmap='coolwarm', alpha=1.0, cbar=True, xticklabels=False, yticklabels=False); 
plt.title('LAI (m$^2$m$^{-2}$)')

plt.subplot(322)
cax = sns.heatmap(sitetype, cmap=cm, alpha=1.0, cbar=True, xticklabels=False, yticklabels=False); 
cb1 = cax.collections[0].colorbar
cb1.set_ticks([0.4, 1.2, 2, 2.8, 3.6])
cb1.set_ticklabels(['peatland','herb-rich','mesic','sub-xeric', 'xeric'])
plt.title('sitetype')


plt.subplot(324)
cax = sns.heatmap(Wtop_dry * 1000 / 200., cmap=cm, alpha=1.0, cbar=True, xticklabels=False, yticklabels=False); 
cb1 = cax.collections[0].colorbar
#cb1.set_ticks([0.4, 1.2, 2, 2.8, 3.6])
#cb1.set_ticklabels(['peatland','herb-rich','mesic','sub-xeric', 'xeric'])
plt.title('Humus layer moisture (g/g), 25th perc.')

plt.subplot(323)
sns.heatmap(Wdry, cmap='coolwarm_r', alpha=1.0, cbar=True, xticklabels=False, yticklabels=False); 
plt.title('soil moisture (m3m-3), 10th perc')

plt.subplot(325)
x = np.ravel(LAIt)
y = np.ravel(Wdry)
rr = sitetype.copy()
norm = mplcolors.BoundaryNorm(np.arange(0, 6) - 0.5, 5)
plt.scatter(x, y, c=rr, cmap=cm, norm=norm, s=10, alpha=0.7)
plt.xlabel('LAI')
plt.title('soil moisture (m3m-3), 10th perc')
cb1 = plt.colorbar(ticks=np.arange(0, 6))
cb1.ax.set_yticklabels(['peatland','herb-rich','mesic','sub-xeric', 'xeric']) 

plt.subplot(326)
x = np.ravel(LAIt)
y = np.ravel(Wtop_dry * 1000 / 200.)
rr = sitetype.copy()
norm = mplcolors.BoundaryNorm(np.arange(1, 7) - 0.5, 5)
plt.scatter(x, y, c=rr, cmap=cm, norm=norm, s=10, alpha=0.7)
cb1 = plt.colorbar(ticks=np.arange(0, 6))
cb1.ax.set_yticklabels(['peatland','herb-rich','mesic','sub-xeric', 'xeric']) 
plt.xlabel('LAI')
plt.title('Humus layer moisture (g/g), 25th perc.')

#%% scatterplots


plt.figure()
plt.subplot(221)
plt.scatter(x=gisdata['age'], y=gisdata['ba'], c=gisdata['p_vol']/gisdata['vol'], alpha=0.5); plt.colorbar()
plt.xlabel('age (yr)'); plt.ylabel('ba (m2ha-1)')

plt.subplot(222)
plt.scatter(x=gisdata['age'], y=gisdata['hc'], c=gisdata['p_vol']/gisdata['vol'], alpha=0.5); plt.colorbar()
plt.xlabel('age (m2ha-1)'); plt.ylabel('height (m)')

plt.subplot(223)
plt.scatter(x=gisdata['age'], y=LAIt, c=gisdata['p_vol']/gisdata['vol'], alpha=0.5); plt.colorbar()
plt.xlabel('age (m2ha-1)'); plt.ylabel('LAI (m2m-2)')


#%%
ig, ax = plt.subplots(1,1, figsize=(11.7, 8.27)) # A4 portrait

DF = []
L = []
# loop for sitetype
for s in [0,1,2,3,4]:
    ix = np.where(sitetype == s)
    d = Wliq[:,ix[0], ix[1]] #subset
    l = LAIt[ix[0], ix[1]]
    
    df = pd.DataFrame(index = tvec, data = d)
    DF.append(df)
    L.append(l)
    
    # now plot
    grouper = df.groupby(df.index.dayofyear)
    
    wm = grouper.mean()
    
    M, q25, q75, mini, maxi = wm.median(axis=1).values, wm.quantile(0.25,axis=1).values,\
                              wm.quantile(0.75, axis=1).values, wm.quantile(0.025, axis=1).values, wm.quantile(0.975, axis=1).values

    
    t = wm.index
    ax.fill_between(t, mini, maxi, color=cc[s], alpha=0.5)
    ax.fill_between(t, q25, q75, color=cc[s], alpha=0.75)
    ax.plot(t, M, '-', color=cc[s], linewidth=2)
    ax.set_ylabel(r'$\theta$ (m$^{2}$m$^{-2}$)')
    ax.set_xlabel('doy')
    ax.set_xlim([1,365])
    #ax.text(0.04, 0.06, slabel[s],
    #            horizontalalignment='center',
    #            verticalalignment='center',
    #            fontsize=14,
    #            transform = ax[s].transAxes)

plt.tight_layout()

#%% save as ascii-grids

# prepare info
info0 = gisdata['info']

out = np.array(drisk)
out[np.where(np.isnan(out))] = -9999.0

outfile = r'results/driskmap.asc'
write_AsciiGrid(outfile, out, info0, fmt='%.3f')

out = np.array(sitetype)
out[np.where(np.isnan(out))] = -9999.0

outfile = r'results/sitetype.asc'
write_AsciiGrid(outfile, out, info0, fmt='%.3f')

out = np.array(LAIt)
out[np.where(np.isnan(out))] = -9999.0

outfile = r'results/LAIt.asc'
write_AsciiGrid(outfile, out, info0, fmt='%.3f')

out = np.array(gisdata['vol'])
out[np.where(np.isnan(out))] = -9999.0

outfile = r'results/vol.asc'
write_AsciiGrid(outfile, out, info0, fmt='%.3f')

#%%
out = np.array(gisdata['ba'])
outfile = r'results/bal.asc'
write_AsciiGrid(outfile, out, info0, fmt='%.3f')
