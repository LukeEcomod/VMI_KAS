# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 10:41:04 2018

@author: slauniai
"""

#%% this snippet plots some wetness maps
#import pickle
import os
import numpy as np
from spafhy_graphics import moisture_on_map

folder = r'c:\temp\spafhy\figs'

# read basemap (peruskartta) tuple (data, colormap) and other stuff from pickle
ff = open(r'c:\temp\spafhy\data\vihti\peruskartta\vihti_rasters.pk', 'rb')
rasters = pickle.load(ff) # (peruskartta_data, colormap)
ff.close()

pkartta = rasters['peruskartta']  # raster 
colormap = rasters['pkcmap']  # colormap for peruskartta
basemap = (pkartta, colormap)
boundaries = rasters['boundaries']  # calculation boundaries

# open link to resutls 
ncf = Dataset(ncf_file, 'r')


# date as list of strings; should parse from ncg or
datestr = list(FORC.index.astype(str))

# threshold for wetness classification
threshold = [0.03, 0.045]

for k in np.arange(366, 366+330, 10):
    W = ncf['bu']['Wliq'][k,:,:]
    S = ncf['top']['Sloc'][k,:,:]
    S[S <= 0] = 0.0
    
    # categorize S (saturation deficit)
    WX = np.ones(np.shape(S))*np.NaN
    # ix1 = np.where(S < threshold[0])
    ix2 = np.where(S < threshold[0])
    ix3 = np.where(S <= threshold[1])
    WX[ix3] = threshold[1]
    WX[ix2] = threshold[0]
    # WX[ix1] = threshold[0]
    
    # headers
    txt1 = datestr[k] + ': vol. moisture (-)'
    txt2 = datestr[k] + ': sat deficit (m)'
    txt3 = datestr[k] + ': wetness indicator'
    # filepaths
    f1 = os.path.join(folder, 'Vihti-Wliq-' + datestr[k] + '.png' )
    f2 = os.path.join(folder, 'Vihti-S-' + datestr[k] + '.png' )
    f3 = os.path.join(folder, 'Vihti-wetness-' + datestr[k] + '.png' )
    
    # make and save figs
    # h1 = field_on_map(basemap, W, txt1, bounds=boundaries, xcolormap='coolwarm_r', alpha=0.7, vmin=0.1, vmax=0.6, fig_nr=1)
    # h2 = field_on_map(basemap, S, txt2, bounds=boundaries, xcolormap='coolwarm', alpha=0.7, vmin=0.01, vmax=0.05,fig_nr=2)
    h3 = moisture_on_map(basemap, WX, txt3, bounds=boundaries, xcolormap='Blues_r', 
                         alpha=0.6, vmin=0.01, vmax=0.075, fig_nr=3)
    #h1.savefig(f1, dpi=300); plt.close(h1)
    #h2.savefig(f2, dpi=300); plt.close(h2)
    h3.savefig(f3, dpi=500); plt.close(h3)
ncf.close()

def main():
    print("Main")
if __name__ == "__main__":
    main()

