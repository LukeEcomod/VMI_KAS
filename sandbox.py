# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 15:08:51 2022

@author: 03081268
"""
import numpy as np
import matplotlib.pyplot as plt
eps = np.finfo(float).eps

# herb-rich
# vanG_Theta_s = 0.58
# vanG_Theta_r = 0.0
# vanG_alpha = 4.06
# vanG_n = 1.17

# mesic
# vanG_Theta_s = 0.55
# vanG_Theta_r = 0.0
# vanG_alpha = 4.48
# vanG_n = 1.20

# sub-xeric
vanG_Theta_s = 0.53
vanG_Theta_r = 0.0
vanG_alpha = 3.7
vanG_n = 1.24

# xeric
# vanG_Theta_s = 0.55
# vanG_Theta_r = 0.03
# vanG_alpha = 3.80
# vanG_n = 1.42

def theta_psi(Wliq, vanG_Theta_s, vanG_Theta_r, vanG_alpha, vanG_n):
     """
     converts vol. water content (m3m-3) to soil water potential (MPa)
     """
     n = vanG_n
     m = 1 - 1 / n
     # converts water content (m3m-3) to potential (m)
     x = np.minimum(Wliq, vanG_Theta_s)
     x = np.maximum(x, vanG_Theta_r)  # checks limits
     
     s = (vanG_Theta_s - vanG_Theta_r) / ((x - vanG_Theta_r) + eps)
     
     Psi = -1 / vanG_alpha*(s**(1.0 / m) - 1.0)**(1.0 / n)  # kPa
     Psi[np.isnan(Psi)] = 0.0
     return 1e-3*Psi

def psi_theta(psi, vanG_Theta_s, vanG_Theta_r, vanG_alpha, vanG_n):
     """
     converts soil water potential (MPa) to vol. water content (m3m-3)
     """
     x = np.abs(1e3*psi)
     n = vanG_n
     m = 1 - 1/n
    
     y = vanG_Theta_r + (vanG_Theta_s-vanG_Theta_r) / (1 + (vanG_alpha*x)**n)**m
    
     return y
 
def rel_g1(Psi, b=0.63):
    """
    Zhou et al. 2013 AFM eq. 3: f = exp(b * Psi)
    b = 0.63 for gymnosperms, 0.7 for Scots pine (Launiainen et al. 2015 Ecol. Mod)
    Args:
        Psi [MPa]
        b - sensitivity [MPa-1]
    Returns
        f - [-]
    """
    return np.exp(b * Psi) 

def rel_g1_rew(rew, p):
    return np.minimum(1.0, rew/p)

def rew(Wliq, fc, wp):
    r = np.maximum(0.0, np.minimum((Wliq - wp) / (fc - wp + eps), 1.0))    
    return r
    
x = np.linspace(0.0, vanG_Theta_s, 100)

Psi = theta_psi(x, vanG_Theta_s, vanG_Theta_r, vanG_alpha, vanG_n)

y = psi_theta(Psi, vanG_Theta_s, vanG_Theta_r, vanG_alpha, vanG_n)

fc = psi_theta(-10*1e-3, vanG_Theta_s, vanG_Theta_r, vanG_alpha, vanG_n)
wp = psi_theta(-1500*1e-3, vanG_Theta_s, vanG_Theta_r, vanG_alpha, vanG_n)

r = rew(x, fc, wp)

f = rel_g1(Psi, b=0.63)
f2 = rel_g1(Psi, b=0.7)
f3 = rel_g1_rew(r, 0.2)

plt.figure()
plt.subplot(1,2,1)
plt.semilogx(-Psi, x, 'r-')
plt.subplot(1,2,2)
plt.plot(x, f, 'r-')
plt.plot(x, f2, 'g-')
plt.plot(x, f3, 'm-')
