# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 21:21:08 2022

Module for computing growth-relevant statistics from daily weather data and SpaFHy -simulations


@ Samuli Launiainen
"""

import pandas as pd
import numpy as np
import pickle
import scipy.stats as stats
from scipy.stats import linregress

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm

from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages

num_colors = 41
cmap = cm.get_cmap(name='coolwarm')
colorlist = [cmap(1.*i/num_colors) for i in range(num_colors)]

#del cmap, num_colors

ffile = r'results/southern_finland_1981_2021.pk'
with open(ffile, 'rb') as f:
    res = pickle.load(f)

res = res[res.index.year > 1980]


def make_all_figs(res):
    # compute thermal growing season
    gs = thermal_growingseason(res)
    
    # compute snow statistics
    snow = snowmelt_doy(res)
    
    # compute summary statistics over growing seasons
    gs_stats = compute_period_stats(res, gs, snow, fdoy=None, ldoy=None)
    
    # compute summary statistics over July-August (most prone to drought)
    ja_stats = compute_period_stats(res, gs, snow, fdoy=182, ldoy=243)
    
    # compute summary statistics over Jan-Mar
    jfm_stats = compute_period_stats(res, gs, snow, fdoy=1, ldoy=90)
    
    # compute bi-weekly averages over each year
    # dataframes for each year are in list ave2w
    ave2w, yrs = temporal_averages_by_year(res, window='SMS')
    


    # --- Fig1: seasonal course at 2week intervals
    fig1 = make_seasonal_course_soilmoisture(ave2w, yrs)
    
    # --- Fig2: thermal growing season, snowmelt, growing-season temperature and radiation sum
    cols =  ['gs_start', 'gs_end', 'gs_length', 'T', 'TSum','RgSum', 'PrecSum', 'VPD',
              'snowmelt', 'max_swe', 'Wliq_mesic', 'fW_mesic', 'f_mesic']
    units = ['doy', 'doy', 'days', 'degC', 'degC', 'MJm-2', 'mm', 'kPa' ,'doy', 'mm', 'm3m-3', '-', '-' ]
    
    fig2 = make_timeseries_trend_fig(gs_stats, cols)
    
    # --- Fig3: July-August
    cols = ['T', 'TSum','RgSum', 'PrecSum', 'VPD', 'Wliq_mesic', 'fW_mesic', 'f_mesic']
    
    fig3 = make_timeseries_trend_fig(ja_stats, cols)
    
    # --- Fig 4: Jan-Mar
    #cols = ['T']
    #fig4 = make_timeseries_trend_fig(jfm_stats, cols)
    
    return fig1, fig2, fig3

# -- plot timeseries with linear trend line
def plot_trends(data, cols, ytext, trends=True, means=False):
    """
    Args: 
        data - dataframe
        cols - columns to draw
        ytext - ylabel texts
    Returns:
        fig handle
    """
    N = len(cols)
    c = {'sf': 'r', 'nf': 'b'} # colors
    fig, ax = plt.subplots(N,1, figsize=(8,4*N))

    for k, v in data.items():
        x = v.index.values.astype(float)
        xx = x - x[0]
        n = 0
        for col in cols:
            y = v[col].values.astype(float)
            ax[n].plot(x, y, 'o-', color=c[k], alpha=0.5)

            # add trendline
            if trends:
                ls, ls0, r, ls_p, ls_err = linregress(xx, y)
                f = ls0 + ls*xx
                ax[n].plot(x, f, '-', color=c[k], label= k + ': %.2f units a $^{-1}$ (p=%.2f)' % (ls, ls_p))

            # add mean as dashed line
            if means:
                yave = np.nanmean(y)
                ax[n].plot([min(x), max(x)], [yave, yave], '--', color=c[k], label=k + ' ave=%.2f' %yave)
            n +=1

    # add labels & legend
    for n in range(N):
        ax[n].legend(fontsize=8)
        ax[n].set_ylabel(ytext[n])
    
    return fig

def make_seasonal_course_soilmoisture(D, yrs):
    #-- make plot of soil moisture dynamics
    norm = mpl.colors.Normalize(vmin=0,vmax=40)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
             
    fig1, ax1 = plt.subplots(4,2, figsize=(10, 12))
    #fig1.set_size_inches(0.9*8.27, 0.9*11.69)
    
    for k in range(len(yrs)):
        c = colorlist[k]
        a = 0.8
        lw = 1

        if yrs[k] >= 2018:
            a = 0.8
            lw = 2
            
        t = D[k].index.dayofyear
        ax1[0,0].plot(t,D[k]['Wliq_herb_rich'], '-', color=c, linewidth=lw, alpha=a, label=yrs[k])
        ax1[1,0].plot(t, D[k]['Wliq_mesic'], '-', color=c, linewidth=lw, alpha=a)
        ax1[2,0].plot(t, D[k]['Wliq_sub_xeric'], '-', color=c, linewidth=lw, alpha=a)
        ax1[3,0].plot(t, D[k]['Wliq_xeric'], '-', color=c, linewidth=lw, alpha=a, label=yrs[k])
 
        ax1[0,1].plot(t, D[k]['fW_herb_rich'], '-', color=c, linewidth=lw, alpha=a, label=yrs[k])
        ax1[1,1].plot(t, D[k]['fW_mesic'], '-', color=c, linewidth=lw, alpha=a)
        ax1[2,1].plot(t, D[k]['fW_sub_xeric'], '-', color=c, linewidth=lw, alpha=a)
        ax1[3,1].plot(t, D[k]['fW_xeric'], '-', color=c, linewidth=lw, alpha=a, label=yrs[k])

        # ax1[0,1].plot(t, D[k]['Rew_herb_rich'], '-', color=c, alpha=a, label=yrs[k])
        # ax1[1,1].plot(t, D[k]['Rew_mesic'], '-', color=c, alpha=a)
        # ax1[2,1].plot(t, D[k]['Rew_sub_xeric'], '-', color=c, alpha=a)
        # ax1[3,1].plot(t, D[k]['Rew_xeric'], '-', color=c, alpha=a, label=yrs[k])
    
    for k in range(0,4):
        ax1[k,0].set_ylabel('vol. moisture (m$^3$m$^{-3}$)'); ax1[k,0].set_xlim([0,max(t)])
        ax1[k,1].set_ylabel('fW (-)'); ax1[k,1].set_xlim([0,max(t)])
        #ax1[k,1].set_ylabel('Rew (-)'); ax1[k,1].set_xlim([0,max(t)])
        
    ax1[3,0].set_xlabel('doy')
    ax1[3,1].set_xlabel('doy')
    ax1[0,0].set_title('herb_rich')
    ax1[1,0].set_title('mesic')
    ax1[2,0].set_title('sub_xeric')
    ax1[3,0].set_title('xeric')
    
    plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.8,
                    top=0.9,
                    wspace=0.3,
                    hspace=0.3)
    
    tloc = np.linspace(0,40,21)
    tloc = tloc.astype(int)
    #print(tloc)
    yrs = np.array(yrs)
    tks = yrs[tloc]
    cbar_ax = fig1.add_axes([0.85, 0.15, 0.05, 0.7])
    cbar = plt.colorbar(sm, cax=cbar_ax,ticks=tloc)
    cbar.ax.set_yticklabels(tks)
    
    return fig1

def make_seasonal_course_figure(D,B, yrs, cols, ytext):
    """
    D - list of dataframes (southern finland)
    B - lisf to dataframes (nothern finland)
    yrs - list of years
    cols - list of columns to plot
    ytext - list of ylabels
    """
    #-- make plot of soil moisture dynamics
    norm = mpl.colors.Normalize(vmin=0,vmax=40)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
             
    m = len(cols)
    fig1, ax1 = plt.subplots(m,2, figsize=(10, 3*m))
    #fig1.set_size_inches(0.9*8.27, 0.9*11.69)
    
    for k in range(len(yrs)):
        c = colorlist[k]
        a = 0.8
        lw = 1.0
        if yrs[k] >= 2018:
            a = 0.8
            lw = 2
        t = D[k].index.dayofyear
        j = 0
        for n in cols:
            ax1[j,0].plot(t,D[k][n], '-', color=c, linewidth=lw, alpha=a, label=yrs[k])
            ax1[j,1].plot(t,B[k][n], '-', color=c, linewidth=lw, alpha=a, label=yrs[k])
            j += 1
    
    for k in range(0,m):
        ax1[k,0].set_ylabel('sf ' + ytext[k]); ax1[k,0].set_xlim([0,max(t)])
        ax1[k,1].set_ylabel('nf ' + ytext[k]); ax1[k,1].set_xlim([0,max(t)])
        
    ax1[3,0].set_xlabel('doy')
    ax1[3,1].set_xlabel('doy')
    ax1[0,0].set_title('Southern Finland')
    ax1[0,1].set_title('Northern Finland')
    
    plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.8,
                    top=0.9,
                    wspace=0.3,
                    hspace=0.3)
    
    tloc = np.linspace(0,40,21)
    tloc = tloc.astype(int)
    #print(tloc)
    yrs = np.array(yrs)
    tks = yrs[tloc]
    cbar_ax = fig1.add_axes([0.85, 0.15, 0.05, 0.7])
    cbar = plt.colorbar(sm, cax=cbar_ax,ticks=tloc)
    cbar.ax.set_yticklabels(tks)
    
    return fig1

def make_timeseries_trend_fig(D, cols):
    
    rows = np.ceil(len(cols) / 2).astype(int)
    fig, ax = plt.subplots(rows, 2)
    fig.set_size_inches(0.9*8.27, 0.9*11.69)
    
    #cols = ['gs_start', 'gs_end', 'gs_length', 'snowmelt', 'max_swe', 'T', 'Tsum', 'Rg', 'RgSum', 'PrecSum', 'VPD']
    
    t = D.index.values
    x = t - t[0]
    n=0; m=0
    for c in cols:
        ax[n,m].plot(t, D[c], 'o-'); ax[n,m].set_ylabel(c); ax[n,m].set_xlim([min(t), max(t)])
        n += 1 
        if n == rows:
            n = 0; m += 1
    ax[rows-1,0].set_xlabel('doy')
    return fig

def compute_period_stats(D, gs, fdoy=None, ldoy=None, dt=86400):
    """Computes averages and sums over given periods
    """
    
    gp = D.groupby(D.index.year)
    yrs = list(gp.groups.keys())
    
    x = pd.DataFrame(index=yrs, columns=[
                                         'RgSum', 'TSum', 'PrecSum', 'ETSum', 
                                         'Wliq_herb_rich', 'Wliq_mesic', 'Wliq_sub_xeric', 'Wliq_xeric',
                                         #'Rew_herb_rich', 'Rew_mesic', 'Rew_sub_xeric', 'Rew_xeric',
                                         'fW_herb_rich', 'fW_mesic', 'fW_sub_xeric', 'fW_xeric',
                                         'fQ_herb_rich', 'fQ_mesic', 'fQ_sub_xeric', 'fQ_xeric',
                                         'df_herb_rich', 'df_mesic', 'df_sub_xeric', 'df_xeric',
                                         'f_herb_rich', 'f_mesic', 'f_sub_xeric', 'f_xeric',
                                         'T', 'VPD', 'Rg'])
    
    for yr in yrs:
        dat = gp.get_group(yr)
        if fdoy is None:
            fdoy = gs.loc[yr]['gs_start']
            ldoy = gs.loc[yr]['gs_end']
        
        dat = dat[(dat.index.dayofyear >= fdoy) & (dat.index.dayofyear <= ldoy)]
        
        # for c in ['gs_start', 'gs_end', 'gs_length']:
        #     x[c].loc[yr] = gs.loc[yr][c]
        # for c in ['snowmelt', 'firstsnow', 'max_swe',]:
        #     x[c].loc[yr] = snow.loc[yr][c]
            
        x['RgSum'].loc[yr] = sum(dat['Rg']) * dt * 1e-6 # MJm-2
        T0 = dat['T'] - 5.0; T0[T0<0.0] = 0.0
        x['TSum'].loc[yr] = sum(T0)
        x['PrecSum'].loc[yr] = sum(dat['Prec']) * dt
        x['ETSum'].loc[yr] = sum(dat['ET_herb_rich']) # this is closest to non-restricted ET
        
        acols = ['Wliq_herb_rich', 'Wliq_mesic', 'Wliq_sub_xeric', 'Wliq_xeric',
                 #'Rew_herb_rich', 'Rew_mesic', 'Rew_sub_xeric', 'Rew_xeric',
                 'fW_herb_rich', 'fW_mesic', 'fW_sub_xeric', 'fW_xeric',
                 'fQ_herb_rich', 'fQ_mesic', 'fQ_sub_xeric', 'fQ_xeric',
                 'T', 'VPD', 'Rg']
        
        aves = dat[acols].resample('1A').mean()
        
        for c in acols:
            x[c].loc[yr] = aves[c].values[0]
        
        # fraction of gs days soil moisture modifier < 0.8
        
        for c in ['herb_rich', 'mesic', 'sub_xeric', 'xeric']:
            fw = dat['fW_' + c]
            fq = dat['fQ_' + c]
            #gslen = gs.loc[yr]['gs_length']
            #print(len(fw))
            x['df_' + c].loc[yr] = len(fw[fw<0.8]) / len(fw) * 100 # in percent
            q = dat['fW_' + c] * dat['fQ_' + c]
            #print(fw, fq, q)
            a = q.resample('1A').mean().values
            x['f_' + c].loc[yr] = a[0]
            del fw, fq, q, a
            
    return x
            
def thermal_growingseason(D, tlim=5.0):
    """
    length of thermal growing season following Linderholm et al. 2008 Climatic Change
    returns gs_start, gs_end and gs_length in days
    Args:
        D - dataframe
    Returns
        x - dataframe
    """
    
    gp = D.groupby(D.index.year)
    yrs = list(gp.groups.keys())
    x = pd.DataFrame(data=None, index=yrs, columns=['gs_start', 'gs_end', 'gs_length'])
    #print(yrs)
    for yr in yrs:
        #print(yr)
        dat = gp.get_group(yr)
        
        ta = np.array(dat['T'].values)
        doy = list(dat.index.dayofyear)

        # begining of gs is last day of first 6-day period with ta>5degC
        for d in range(6, 185):
            if all(ta[d-6:d] >= tlim):
                #print(yr, d)
                x['gs_start'].loc[yr]  = doy[d]
                break
        # end of growing season is the last day of first 6-day period with ta <5degC
        for d in range(186, len(ta)):
            if all(ta[d-6:d] <= tlim):
                #gs_end = doy[d]
                x['gs_end'].loc[yr] = doy[d]
                break

    x['gs_length'] = x['gs_end'] - x['gs_start'].values
    
    return x

def snowmelt_doy(D, tlim=5.0):
    """
    returns snowmelt/snowfall days and maximum SWE
    Args:
        D - dataframe
    Returns
        x - dataframe
    """
    snowlim = 5.0 # mm
    
    gp = D.groupby(D.index.year)
    yrs = list(gp.groups.keys())
    x = pd.DataFrame(data=None, index=yrs, columns=['snowmelt', 'firstsnow', 'max_swe', 'doy_max_swe'])
    
    for yr in yrs:
        dat = gp.get_group(yr)
        
        swe = np.array(dat['SWE'].values)
        #print(yr, len(swe), max(swe))
        doy = list(dat.index.dayofyear)

        # snowmelt is the last day when swe <= snowlim
        for d in range(0, 180):
            if all(swe[d:d+60] <= snowlim):
                #print(yr, d)
                x['snowmelt'].loc[yr]  = doy[d]
                break
        # snowfall season is the day of 1st snowfall > snowlimit
        for d in range(186, len(swe)):
            if swe[d] >= snowlim:
                x['firstsnow'].loc[yr] = doy[d]
                break
        # max swe
        f = np.where(swe == max(swe))[0][0]
        x['max_swe'].loc[yr] = swe[f]
        
        d0 = doy[f]
        if d0 > 180:
            d0 -= 365
        x['doy_max_swe'].loc[yr] = d0

    
    return x

def temporal_averages_by_year(D, window='1M', dt=86400.):
    """
    compute temporal averages or sums within windows for each year
    https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
    
    """
    gp = D.groupby(D.index.year)
    yrs = list(gp.groups.keys())
    
    x = []
    for yr in yrs: 
        dat = gp.get_group(yr)
        y = dat.resample(window).mean()
        y['Prec'] = dat['Prec'].resample(window).sum() * dt
        
        x.append(y)
        
    return x, yrs
    
    
def seasonal_trendplots(data, meteo, fyear=2001,
                        resfolder=r'results',
                        compile_pdf=True):
    """
    Monthly trendplots
    """
    import os
    import seaborn as sns
    sns.set(style="whitegrid")
    
    eps = np.finfo(float).eps  # machine epsilon
    
    RAD_TO_DEG = 180.0/np.pi  # conversion rad -->deg
    #L_MOLAR = 44100.0  # J/mol latent heat of vaporization at 20 deg C
    cfact = 0.021618 # umol m-2 s-1 to gC m-2 30min-1
    etfact = 0.0324 # mmmol m-2 s-1 to mm 30min-1

    def lin_reg_range(x, y0, s, e):
        """
        returns slope lines
        """
    
        f = y0 + s*x
        m = np.mean(f) # mean point
        x0 = (m - y0) / s
    
        f1 = m + (s+e)*(x - x0) # hi slope
        f2 = m + (s-e)*(x - x0) # low slope
    
        return f, f1, f2

    units = {'NEP': '(gCm$^{-2}$d$^{-1}$)', 'GPP': '(gCm$^{-2}$d$^{-1}$)', 'Reco': '(gCm$^{-2}$d$^{-1}$)',
             'ET': '(mm d$^{-1}$)', 'U': '(ms$^{-1}$)', 'ust': '(ms$^{-1}$)', 'Par': '(umolm$^{-2}$s$^{-1}$)',
             'Tair': '($^{\circ}$C)', 'VPD': ('kPa'), 'Tsa': '($^{\circ}$C)', 'Wa': '(m$^{3}$m$^{-3}$)', 'CO2': '(ppm)',
             'diff_fr': '(-)', 'Prec': '(mm d$^{-1}$)', 'year': '(-)',
             'LUE': '(mmol mol$^{-1}$)', 'WUE': '(mmol mol$^{-1}$)', 'IWUE': '(mmol mol$^{-1}$)',
             '1-Ci/Ca': '(-)', 'Gsc': '(mol m$^{-2}$s$^{-1}$)'
             }

    # plot options
    n = 5
    col = plt.cm.tab20c(np.linspace(0.01, 0.99, n))

    # form daily datamatrix
    ccols = ['NEP', 'GPP', 'Reco']
    mcols = ['U','Ustar', 'Par','Tair','VPD','Tsa','Wa','CO2', 'RH']


    y = data[ccols].resample('1D').sum()*cfact
    y = pd.concat([y, data['ET'].resample('1D').sum()*etfact], axis=1)
    #y = pd.concat([y, data[['WUE', 'LUE']].resample('1D').mean()], axis=1)
    
    y = pd.concat([y, meteo[mcols].resample('1D').mean()], axis=1)
    y = pd.concat([y, meteo['Prec'].resample('1D').sum()], axis=1)
    y['year'] = y.index.year

    # LUE, WUE, IWUE from daily integrals

    ix = np.where(meteo.Zen * RAD_TO_DEG < 90)[0]
    
    gsc = data['Gsc'].iloc[ix].resample('1D').median()
    diff_fr = meteo['diffRg'].iloc[ix].resample('1D').sum() / meteo['Rg'].iloc[ix].resample('1D').sum()
    lue = 1e3*data['GPP'].iloc[ix].resample('1D').sum() / meteo['Par'].iloc[ix].resample('1D').sum()
    wue = data['GPP'].iloc[ix].resample('1D').sum() / (data['ET'].iloc[ix].resample('1D').sum() + eps)
    wue[wue < 0] = np.NaN
    wue[wue > 1e2] = np.NaN
    y['Gsc'] = gsc.values
    y['diff_fr'] = diff_fr.values
    y['LUE'] = lue.values
    y['WUE'] = wue.values
    del lue, wue, diff_fr

    can_state = meteo['canopy_state'].resample('1D').mean()
    y['canopy_state'] = can_state.copy()
    #y['canopy_state'][can_state >= 0.7] = 1.0
    #y['canopy_state'][can_state <= 0.2] = 0.0


    # make figs
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    #yrs = list(range(2002, 2018))

    cols = y.columns.tolist()
    for k in ['canopy_state', 'year']:
        cols.remove(k)

    y = y[y.index.year >= fyear]
    t = np.unique(y.index.year.values)

    figlist = []
    #cols = ['WUE', 'WUEs', 'LUE', 'LUEs']
    cols = ['NEP', 'GPP', 'Reco', 'ET', 'WUE', 'LUE','Gsc',
            'U','Ustar', 'Par','diff_fr','Tair','Tsa','Wa','VPD','RH']
    
    ylims =  {'GPP':{'bot': 0.0}, 'Reco': {'bot': 0.0}, 'Par': {'bot': 0.0}, 
              'diff_fr': {'top': 1.0}, 'RH': {'top': 100.0},
              'LUE':{'bot': 0.0, 'top': 40.}, 'WUE': {'bot': 0.0, 'top': 15.}, 'Gsc': {'bot': 0.0, 'top': 0.75}}
    
    for col in cols:
        print(col)
        fig, (ax) = plt.subplots(nrows=4, ncols=3)
        fig.set_size_inches(11.69,8.27) # A4L

        n=0; m=0
        for mo in range(1, 13):
            if m == 3:
                m = 0
                n = n + 1

            ix = np.where((y.index.month == mo))[0]
            ixdry = np.where((y.index.month == mo) & (y.canopy_state > 0.8))[0]

            # -- trend all conditions
            M = y[col].iloc[ix].resample('M').mean()
            M = M.dropna(axis=0, how='all')
            xx = M.index.year.values
            Mave = np.mean(M.values)

            # linear lsq trend
            if len(M.values) > 10:
                s, y0, r2, ls_p, err = linregress(xx-xx[0], M.values)

                f, f1, f2 = lin_reg_range(xx-xx[0], y0, s, err)
                tx1 = '%.1f %% (%.2f), r$^2$=%.2f, p=%.2f' %(100*s/abs(Mave), Mave, r2, ls_p)

                del f1, f2, s, y0, r2, ls_p, err, Mave,

                # -- trend dry canopy
                Q = y[col].iloc[ixdry].resample('M').mean()
                Q = Q.dropna(axis=0, how='all')
                xxx = Q.index.year.values
                Qave = np.mean(Q.values)

                # linear lsq trend
                s, y0, r2, ls_p, err = linregress(xxx-xxx[0], Q.values)

                ff, f1, f2 = lin_reg_range(xxx-xxx[0], y0, s, err)
                tx2 = '%.1f %% (%.2f), r$^2$=%.2f, p=%.2f' %(100*s/abs(Qave), Qave, r2, ls_p)
                del f1, f2, s, y0, r2, ls_p, err

                # ---- plot

                sns.violinplot(ax=ax[n,m], x='year', y=col, data=y.iloc[ix,:], cut=0, color='w',
                               linewidth=0.5, scale='width')

                # --- means
                yave = y.iloc[ix].groupby(['year']).mean()
                ax[n,m].plot(range(0, len(yave)),  yave[col], 'bo', markersize=4)
                
                yave1 = y.iloc[ixdry].groupby(['year']).mean()
                yave1 = yave1[col].dropna(axis=0, how='all')
                inx = []
                for k in range(0, len(yave1)):
                    aa = np.where(yave.index.values == yave1.index.values[k])[0]
                    if len(aa) == 1:
                        inx.extend(aa.astype(int))
    
                ax[n,m].plot(inx,  yave1, 'ro', markersize=4)
        
                # ---- trendlines
                ax[n,m].plot(range(0, len(f)), f, 'b-')
                ax[n,m].plot(inx, ff, 'r-')
    
                ax[n,m].text(0.15, 0.05, tx1, color='b', transform=ax[n,m].transAxes, fontsize=8, zorder=10,
                      bbox=dict(facecolor='w', alpha=0.5))
                ax[n,m].text(0.15, 0.85, tx2, color='r', transform=ax[n,m].transAxes, fontsize=8, zorder=10,
                      bbox=dict(facecolor='w', alpha=0.5))
                ax[n,m].set_xticks(np.arange(0, len(t), 2))
                ax[n,m].set_xticklabels([])
                ax[n,m].set_label('')
                ax[n,m].tick_params(axis='both', direction='in')
                ax[n,m].text(0.00, 0.95, months[mo-1], transform=ax[n,m].transAxes, zorder=5,
                  bbox=dict(facecolor='w', alpha=1.0))
    
                if col == 'Prec':
                    ymax = max((y.iloc[ix].groupby(['year']).mean())['Prec']) + 5.0
                    ax[n,m].set_ylim([0, ymax])
                
                #ylims
                if col in ylims:
                    yl = ylims[col]
                    if 'bot' in yl and 'top' in yl:
                        ax[n,m].set_ylim(bottom=yl['bot'], top=yl['top'])
                    elif 'bot' in yl:
                        ax[n,m].set_ylim(bottom=yl['bot'])
                    elif 'top' in yl:
                        ax[n,m].set_ylim(top=yl['top'])                   
                m = m+1

        # ylabel: variable + units
        if col in units:
            u = ' ' + units[col]
        else:
            u = ''
        for n in range(0,4):
            ax[n,0].set_ylabel(col + u, fontsize=10)
            ax[n,1].set_ylabel('')
            ax[n,2].set_ylabel('')
        for k in [0, 1, 2]:
            #ax[n, m].set_xticks()
            ax[3,k].set_xticks(np.arange(0, len(t), 2))
            ax[3,k].set_xticklabels(t[np.arange(0, len(t), 2)], rotation=45.0, fontsize=10)

        # save
        #fn = Path(resfolder) / (col + '_violin.jpg')
        #print(fn)
        fn = os.path.join(resfolder, col + '_violins.jpg')
        #print(fn)
        fig.savefig(fn, dpi=600)

        figlist.append(fig)

    # create multipage pdf
    if compile_pdf:
        from matplotlib.backends.backend_pdf import PdfPages
        fn = os.path.join(resfolder, 'montly_trendplots.pdf')

        with PdfPages(fn) as pdf:
            # As many times as you like, create a figure fig and save it:
            for f in figlist:
                pdf.savefig(f)

def linear_regression(x, y, alpha=0.05):
    """
    Univariate linear least-squares regression with confidence intervals using
    scipy.stats.linregress
    adapted from: http://bagrow.info/dsv/LEC10_notes_2014-02-13.html
    Args: x, y, alpha
    Returns: res (dict)
    """
    
    n = len(x)    
    
    slope, b0, r_value, p_value, serr = stats.linregress(x, y)
    
    # mean squared deviation
    s2 = 1./n * sum([(y[i] - b0 - slope * x[i])**2 for i in range(n)])
    
    #confidence intervals of slope and intercept
    xx = x * x
    c = -1 * stats.t.ppf(alpha/2.,n-2)
    bb1 = c * (s2 / ((n-2) * (xx.mean() - (x.mean())**2)))**.5
    ci_slope = [slope - bb1, slope +bb1]
    
    bb0 = c * ((s2 / (n-2)) * (1 + (x.mean())**2 / (xx.mean() - (x.mean())**2)))**.5
    ci_interc = [b0 - bb0, b0 + bb0]
    
    res = {'model':[slope, b0], 'r': r_value, 'p': p_value, 'ci': [ci_slope, ci_interc]}
    return res

def mann_kendall_test(x, alpha=0.05):
    """
    This function is derived from code originally posted by Sat Kumar Tomer
    (satkumartomer@gmail.com)
    See also: http://vsp.pnnl.gov/help/Vsample/Design_Trend_Mann_Kendall.htm

    The purpose of the Mann-Kendall (MK) test (Mann 1945, Kendall 1975, Gilbert
    1987) is to statistically assess if there is a monotonic upward or downward
    trend of the variable of interest over time. A monotonic upward (downward)
    trend means that the variable consistently increases (decreases) through
    time, but the trend may or may not be linear. The MK test can be used in
    place of a parametric linear regression analysis, which can be used to test
    if the slope of the estimated linear regression line is different from
    zero. The regression analysis requires that the residuals from the fitted
    regression line be normally distributed; an assumption not required by the
    MK test, that is, the MK test is a non-parametric (distribution-free) test.
    Hirsch, Slack and Smith (1982, page 107) indicate that the MK test is best
    viewed as an exploratory analysis and is most appropriately used to
    identify stations where changes are significant or of large magnitude and
    to quantify these findings.

    Input:
        x:   a vector of data
        alpha: significance level (0.05 default)

    Output:
        trend: tells the trend (increasing, decreasing or no trend)
        h: True (if trend is present) or False (if trend is absence)
        p: p value of the significance test
        z: normalized test statistics

    Examples
    --------
      >>> x = np.random.rand(100)
      >>> trend,h,p,z = mk_test(x,0.05)

    """
    n = len(x)

    # calculate S
    s = 0
    for k in range(n-1):
        for j in range(k+1, n):
            s += np.sign(x[j] - x[k])

    # calculate the unique data
    unique_x = np.unique(x)
    g = len(unique_x)

    # calculate the var(s)
    if n == g:  # there is no tie
        var_s = (n*(n-1) *(2*n+5))/18
    else:  # there are some ties in data
        tp = np.zeros(unique_x.shape)
        for i in range(len(unique_x)):
            tp[i] = sum(x == unique_x[i])
        var_s = (n*(n-1)*(2*n+5) - np.sum(tp*(tp-1)*(2*tp+5)))/18

    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else: # s == 0:
        z = 0

    # calculate the p_value
    p = 2*(1 - stats.norm.cdf(abs(z)))  # two tail test
    h = abs(z) > stats.norm.ppf(1-alpha/2)

    if (z < 0) and h:
        trend = 'decreasing'
    elif (z > 0) and h:
        trend = 'increasing'
    else:
        trend = 'no trend'

    return trend, h, p, z
    
def trend_breakpoint(x, y, min_obs=4, figs=True):
    """ 
    Seeks for trend breakpoints using Chow test. Trends are computed using linear
    least-squares estimate
    Args:
        x - independent variable
        y - dependent variable
        min_obs - minimum values for computing fit
    Returns:
        res (dict)
    """
    def find_rss(x, y, model):
        """ sum of squared residuals
        """
        ymod = model[1] + model[0] * x
        return sum((y - ymod)**2), len(x)
    
    def chow_test(x1, y1, x2, y2, conf=0.9):
        
        # linear least-square regressions
        f = linear_regression(np.append(x1, x2), np.append(y1, y2), alpha=0.05)
        f1 = linear_regression(x1, y1, alpha=0.05)
        f2 = linear_regression(x2, y2, alpha=0.05)
        
        # sum of squared residuals
        rss_total, n_total = find_rss(np.append(x1, x2), np.append(y1, y2), f['model'])
        rss_1, n_1 = find_rss(x1, y1, f1['model'])
        rss_2, n_2 = find_rss(x2, y2, f2['model'])
        
        # F-test
        chow_nom = (rss_total - (rss_1 + rss_2)) / 2
        chow_denom = (rss_1 + rss_2) / (n_1 + n_2 - 4)
        
        F = chow_nom / chow_denom
        
        if not F:
            F = 1.0
        
        # Chow-test p-value
        df1 = 2
        df2 = len(x1) + len(x2) - 4
    
        # The survival function (1-cdf) is more precise than using 1-cdf,
        # this helps when p-values are very close to zero.
        # -f.logsf would be another alternative to directly get -log(pval) instead.
        p_val = stats.f.sf(F, df1, df2)
        
        return p_val, f1, f2          
        
    # compute
    
    x0 = x - x[0]
    N = len(x)

    results = {'p_value': [], 'mod_1': [], 'mod_2': []}
    
    k = 0 # - min_obs
    while k < N:
        x1 = x0[0:k]
        y1 = y[0:k]
        x2 = x0[k:]
        y2 = y[k:]
        
        if len(x2) >= min_obs and len(x1) >= min_obs:
            print(k, x1, x2)
            p, f1, f2 = chow_test(x1, y1, x2, y2)
            f1['x'] = x[0:k]
            f2['x'] = x[k:]
        elif len(x2) < min_obs:
            p, f2 = np.NaN, None
            f1 = linear_regression(x1, y1, alpha=0.05)
            f1['x'] = x[0:k]
        elif len(x1) < min_obs:
            p, f1 = np.NaN, None
            f2 = linear_regression(x2, y2, alpha=0.05)
            f2['x'] = x[k:]            
        #print(t[0:k], t[k:N], p)
        
        results['p_value'].append(p)
        results['mod_1'].append(f1)
        results['mod_2'].append(f2)
        k += 1
    # breakpoint is defined as point where p_value == min(p_value) and p_value < 0.05
    min_p = np.nanmin(results['p_value'])
    if min_p <= 0.05:
        #ix = int(np.where(results['p_value'] == min_p) [0])
        ix = int(max(np.where(np.array(results['p_value']) <= 0.05)[0]))

        results['changepoint'] =  float(x[ix])
    else:
        results['changepoint'] = None
    
    if figs:
        plt.figure()
        plt.plot(x, y, 'ro')
        f = linear_regression(x0, y, alpha=0.05)        
        fit = f['model'][0] * x0 + f['model'][1]
        txt = '%.2f (%.2f / %.2f) x, p=%.2f, r2=%.2f' %(f['model'][0], f['ci'][0][0], f['ci'][0][1],
                                                                             f['p'], f['r']**2)
        plt.plot(x, fit, 'k-', label=txt)

        if min_p <= 0.05:
            c1 = results['mod_1'][ix]
            c2 = results['mod_2'][ix]
            
            fit1 = c1['model'][0] * (c1['x'] - x[0]) + c1['model'][1]
            txt1 = '%.2f (%.2f / %.2f) x, p=%.2f, r2=%.2f' %(c1['model'][0], c1['ci'][0][0], c1['ci'][0][1],
                                                                     c1['p'], c1['r']**2)
            fit2 = c2['model'][0] * (c2['x'] - x[0]) + c2['model'][1]
            txt2 = '%.2f (%.2f / %.2f) x, p=%.2f, r2=%.2f' %(c2['model'][0], c2['ci'][0][0], c2['ci'][0][1],
                                                                     c2['p'], c2['r']**2)            
            plt.plot(c1['x'], fit1, 'k--', label=txt1)
            plt.plot(c2['x'], fit2, 'k:', label=txt2)
        
        plt.legend(fontsize=8)
    
    return results


def compute_trends(D, crit=None, figs=False, fname=r'results/Trends'):
    """
    compute trends in timeseries.
    Tests:
    Theil-Sen (theilslopes)
    Mann-Kendall test (mk_test)
    linear regression (linregress)
    
    Args:
        D - pd.dataframe
        yrs - list of years to be used
        figs - True plots
    Returns:
        
    """
        
    crit0 = dict()
    crit0['sign_lev_1'] = 0.01
    crit0['sign_lev_2'] = 0.05
    crit0['sign_lev_3'] = 0.1
    
    if crit is None:
        crit = crit0.copy()

    D = D.replace([np.inf, -np.inf], np.nan)
    
    t = D.index.values.astype(float)
    print(t, type(t))
    x = t - t[0]
    
    # create dataframe for storing trend results     
    cols = ['ts', 'ts0', 'tsl', 'tsu', 'ts_p', 'ls', 'ls0', 'ls_r2', 'ls_p', 'ls_err', 'mk_tr', 'mk_p']
    T = pd.DataFrame(data=None, columns=cols, index=D.columns) 
    
    for key in D.columns:
        m = D[key].dropna(axis=0)
        #t = m.index.values.astype(float)
        print(key)
        #x = t - t[0]
        y = m.values
        print(x, y)
        # theil-sen estimator (median slope between pair of points)
        for a in [crit['sign_lev_1'], crit['sign_lev_2'], crit['sign_lev_3']]:
            ts, ts0, tsl, tsu = stats.theilslopes(y, x, alpha=a)
            ts_a = a
            if np.sign(tsl) == np.sign(tsu): 
                print(key,a); break
        
        # mann-kendall test
        for a in [crit['sign_lev_2'], crit['sign_lev_1']]:
            mk_tr, _, mk_p, _, = mann_kendall_test(y, alpha=a)
            if mk_p <= a: break
    
        # linear least-squares regression
        for a in [crit['sign_lev_2'], crit['sign_lev_1']]:
            ls, ls0, r, ls_p, ls_err = stats.linregress(x, y)

            if ls_p <= a: break
        
        T.loc[key][cols] = [ts, ts0, tsl, tsu,ts_a, ls, ls0, r**2, ls_p, ls_err, mk_tr, mk_p]
            
    # drop rows where no significant trends
    T = T.dropna(axis=0, how='all')
    
    # save to file
    if fname:
        T.to_csv(fname + '.csv', float_format='%.3f', sep='\t')
        #T.to_html(Path(folder) / (fname + '.html'))

    if figs:
        figlist = []

        # plot trends
        for key in list(T.index):
            # plot data and Theil-Sen slopes and linregress slopes
            m = D[key].dropna(axis=0)
            t = m.index.values.astype(float)
            x = t - t[0]
            x0 = np.mean(x)
            xx = x - x0
            m = m.values
            y = T.loc[key]
            
            figname = key
            ttxt = 'TS: [%.3f (%.3f/%.3f)] LR: [%.3f (+/-%.3f); r2=%.2f, p=%.3f]' %(
                    y.ts, y.tsl, y.tsu, y.ls, y.ls_err, y.ls_r2, y.ls_p)
            fig = plt.figure('A_' + key)
            
            f = y.ls0 + y.ls*x
            f0 = y.ls0 + y.ls*x0
    
            fu = f0 + xx*(y.ls + y.ls_err) # * N**0.5)
            fl =  f0 + xx*(y.ls - y.ls_err) # *  N**0.5)

            plt.fill_between(t, fl, fu, color='b', alpha=0.2)
            plt.plot(t, m, 'ko', t, f, 'b-')
            
            # theil-sen intercepts for boundary lines
            ts0l = np.median(m) - y.tsl*np.median(x)
            ts0u = np.median(m) - y.tsu*np.median(x)
            plt.plot(t, y.ts0 + y.ts*x, 'r-', t, ts0l + y.tsl*x, 'r--', t, ts0u + y.tsu*x, 'r--')
            plt.title(ttxt, fontsize=8); plt.ylabel(figname)
            plt.xticks(t, rotation=45.0)
            
            #fn = Path(folder) / (fname + '_' + figname + '.png')
            #fig.savefig(fn, dpi=600)

            figlist.append(fig)

        
        # create multipage pdf
        #from matplotlib.backends.backend_pdf import PdfPages
        
        fn = fname + '.pdf'

        with PdfPages(fn) as pdf:
            # As many times as you like, create a figure fig and save it:
            for f in figlist:
                pdf.savefig(f)
    
    return T

def trend_detection_threshold(n, mu, sigma, pval=0.05, samples=1e4):
    """
    Assuming points are drawn from normal distribution with parameters (mu,sigma),
    estimate propability to detect linear trend from timeseries.
    Args:
        n - nr. of points
        x0 - mean
        xdev - std
        pval - significance threshold
        samples - number of samples for Monte-Carlo
    Returns: 
    """
    mu = abs(mu)
    x = np.arange(0, n)
    x = x - np.mean(x)
    
    def draw_data(x, mu, sigma, tr):
        dy = np.random.normal(0.0, sigma, len(x))
        
        y = (1.0 + tr * x) * mu + dy 
        
        #plt.figure(1)
        #plt.plot(x, y, 'o')
        return y
    
    trend = np.arange(0.0, 0.03, 0.001) # relative to mu
    res = np.zeros((len(trend), 5)) *np.NaN
    res[:,0] = trend
    m = 0
    for tr in trend:
        k = 0
        a = np.zeros((samples, 2)) * np.NaN
        while k < samples:
            y = draw_data(x, mu, sigma, tr)
            slope, b0, r_value, p_value, serr = linregress(x, y)
            a[k,:] = [slope, p_value]
            k += 1
        
        res[m,1] = np.mean(a[:,0])
        res[m,2] = np.quantile(a[:,0], 0.05)
        res[m,3] = np.quantile(a[:,0], 0.95)
        res[m,4] = len(np.where(a[:,0] >0)[0]) / len(a[:,1])
        m += 1
    plt.figure()
    plt.plot(res[:,0]*mu, res[:,3], color='k', alpha=0.5)
    plt.plot(res[:,0]*mu, res[:,2], color='r', alpha=0.5)
    
    plt.figure()
    plt.plot (res[:,0]*mu, res[:,4], 'k-')
    plt.ylabel('propability to detect trend (-)'); plt.xlabel('trend')
    #plt.plot([mu*np.min(res[:,0]), mu*np.max(res[:,0])], [pval, pval], 'r--')
    #plt.plot([mu*np.min(res[:,0]), mu*np.max(res[:,0])], [0.1, 0.1], 'b--')
    plt.ylabel('propability to detect trend (-)'); plt.xlabel('trend gCm-2a-1')
    
    
    return res