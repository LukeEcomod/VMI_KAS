# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 10:36:02 2022

@author: 03081268
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


from netCDF4 import Dataset
from spafhy import spafhy

EPS = np.finfo(float).eps

def REW(w, fc, wp):
    # relative plant available water
    f = np.minimum((w - wp) / (fc - wp + EPS), 1.0)
    return f

def fREW(rew, rw, rwmin):
    # plant moisture response
    f = np.minimum(1.0, np.maximum(rew / rw, rwmin))
    return f

def lai_binned(lai, y, sitetype):
    # bin data
    plt.figure()
    
    
    for s in [0, 1, 2, 3, 4]:
        d = []
        L = []
        lL = 0.0
        D = []
        for uL in np.arange(0.5, 8.5, 0.5):
            L.append(0.5*(lL + uL))
            ix = np.where((lai>=lL) & (lai<=uL) & (sitetype == s))[0]
            d.append(np.ravel(y[ix]))
            lL = uL
        
        L = np.array(L)
        d = np.array(d)
        D.append(d)
        
        print(D)
        plt.figure()
        plt.boxplot(D)
    

        
    
    