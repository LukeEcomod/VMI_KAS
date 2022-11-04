# -*- coding: utf-8 -*-
"""
PARAMETERS OF SPAFHY

Created on Mon Jun 25 18:34:12 2018

@author: slauniai
"""
import numpy as np

def spafhy_inputs(n, lat, lon):
    """
    equivalent to gisdata reading but now for VMI-kas simulations
    Args: 
        n - number of LAI_conif -bins 
        lat - latitude of region midpoint
        lon - longitude of region midpoint
    Returns:
        pgen
        pcpy
        pbu
        sitedata (dict)
    """
    r = 4 # rows = site types
    
    cmask = np.ones((r,n))
    
    # sitetype coding: 1 = herb-rich, 2 = mesic, 3 = sub-xeric, 4 = xeric
    # site fertility increases from 1st to last row
    sitetype = np.ones((r,n))
    sitetype[1,:] = 2
    sitetype[2,:] = 3
    sitetype[3,:] = 4
    
    # make LAI-grid
    LAI_decid = 0.5 * cmask
    LAI_conif = np.ones((r,n))
    # each row is identical
    for k in range(r):
        LAI_conif[k,:] = np.linspace(1.0, 7.0, n)
    
    hc = 10.0 * cmask
    cf = 0.7 * cmask
    ba = 10.0 * cmask
    
    sitedata = {'lat0': np.ones(r)*lat, 'lon0': np.ones(n)*lon, # arbitrary
                'LAI_conif': LAI_conif, 'LAI_decid': LAI_decid,
                'hc': hc, 'cf': cf, 'ba': ba, 'cmask': cmask,
                'sitetype': sitetype
               }
    
    """
    Model parameters
    """
    pgen = {'catchment_id': 'testregion',
            'forcing_file': r'data/SVE_saa.csv',
            'ncf_file': r'results',
            'start_date': '1980-01-01',
            'end_date': '2021-12-31',
            'spinup_end': '1980-01-01',
            'dt': 86400.0,
            'spatial_cpy': True,
            'spatial_soil': True     
           }
    
    pcpy = {'loc': {'lat': lat, 'lon': lon},
            'flow' : { # flow field
                     'zmeas': 2.0,
                     'zground': 0.5,
                     'zo_ground': 0.01
                     },
            'interc': { # interception
                        'wmax': 1.5, # storage capacity for rain (mm/LAI)
                        'wmaxsnow': 4.5, # storage capacity for snow (mm/LAI),
                        },
            'snow': {
                    # degree-day snow model
                    'kmelt': 2.8934e-05, # melt coefficient in open (mm/s)
                    'kfreeze': 5.79e-6, # freezing coefficient (mm/s)
                    'r': 0.05 # maximum fraction of liquid in snow (-)
                    },
                    
            'physpara': {
                        # canopy conductance
                        'amax': 10.0, # maximum photosynthetic rate (umolm-2(leaf)s-1)
                        'g1_conif': 2.0, # stomatal parameter, conifers
                        'g1_decid': 3.5, # stomatal parameter, deciduous
                        'q50': 50.0, # light response parameter (Wm-2)
                        'kp': 0.6, # light attenuation parameter (-)
                        'g1_beta': 0.63, # decay of stomatal conductance with soil water potential
                        #'rw': 0.20, # critical value for REW (-),
                        #'rwmin': 0.02, # minimum relative conductance (-)
                        # soil evaporation
                        'gsoil': 1e-2 # soil surface conductance if soil is fully wet (m/s)
                        },
            'phenopara': {
                        #seasonal cycle of physiology: smax [degC], tau[d], xo[degC],fmin[-](residual photocapasity)
                        'smax': 18.5, # degC
                        'tau': 13.0, # days
                        'xo': -4.0, # degC
                        'fmin': 0.05, # minimum photosynthetic capacity in winter (-)
                        # deciduos phenology
                        'lai_decid_min': 0.1, # minimum relative LAI (-)
                        'ddo': 45.0, # degree-days for bud-burst (5degC threshold)
                        'ddur': 23.0, # duration of leaf development (days)
                        'sdl': 9.0, # daylength for senescence start (h)
                        'sdur': 30.0, # duration of leaf senescence (days),
                         },
            'state': {
                       'lai_conif': None, # conifer 1-sided LAI (m2 m-2)
                       'lai_decid_max': None, # maximum annual deciduous 1-sided LAI (m2 m-2): 
                       'hc': 16.0, # canopy height (m)
                       'cf': 0.6, # canopy closure fraction (-)
                       #initial state of canopy storage [mm] and snow water equivalent [mm]
                       'w': 0.0, # canopy storage mm
                       'swe': 0.0, # snow water equivalent mm
                       }
            }
    
        
    # BUCKET
    pbu = {'depth': 0.35,  # root zone depth (m)
           # following soil properties are filled based on site_type or soil type
           'poros': np.NaN, # porosity (-)
           'fc': np.NaN, # field capacity (-)
           'wp': np.NaN,	 # wilting point (-)
           'ksat': np.NaN, 
           'beta': np.NaN,
           'alpha': np.NaN,
           'n': np.NaN,
           'wr': np.NaN,
           
           #organic (moss) layer
           'org_depth': 0.04, # depth of organic top layer (m)
           'org_poros': 0.9, # porosity (-)
           'org_fc': 0.31, # field capacity (-)
           'org_rw': 0.24, # critical vol. moisture content (-) for decreasing phase in Ef
           'maxpond': 0.0, # max ponding allowed (m)
           #initial states: rootzone and toplayer soil saturation ratio [-] and pond storage [m]
           'rootzone_sat': 0.6, # root zone saturation ratio (-)
           'org_sat': 1.0, # organic top layer saturation ratio (-)
           'pond_sto': 0.0 # pond storage
           }

    
    # assign values to soil parameters   
    psoil = soil_properties_from_sitetype()
    soildata = sitetype_to_soilproperties(pbu, psoil, sitedata['sitetype'], sitedata['cmask'], spatial=True)
    sitedata['soildata'] = soildata
    
    return pgen, pcpy, pbu, sitedata


def soil_properties_from_sitetype():
    """
    Based on Launiainen et al. 2022 Forests
    soil_id 1 = herb-rich, 2 = mesic, 3 = sub-xeric, 4 = xeric,
    5 = peatland, 6 = humus
    """
    psoil = {
            'humus':
                {
                 'soil_id': 6,
                 'poros': 0.9,
                 'fc': 0.31,
                 'wp': 0.11,
                 'ksat': 1e-06,
                 'beta': 4.0,
                 'alpha': 8.54,
                 'n': 1.32,
                 'wr': 0.10,
                 },
            'herb-rich':
                {
                 'soil_id': 1,
                 'poros': 0.58,
                 'fc': 0.30, # 0.34,
                 'wp': 0.13, # 0.11,
                 'alpha': 4.06,
                 'n': 1.17,
                 'wr': 0.0,
                 'ksat': 1e-6, #1e-06,
                 },
            'mesic':
                {
                 'soil_id': 2,
                 'poros': 0.55,
                 'fc': 0.26, #0.28, # 0.26 from pF-curve
                 'wp': 0.09, #0.08,
                 'alpha': 4.48,
                 'n': 1.20,
                 'wr': 0.0,
                 'ksat': 1e-5, #1e-5,1e-6
                 },
            'sub-xeric':
                {'soil_id': 3,
                 'poros': 0.53,
                 'fc': 0.22, #0.24,
                 'wp': 0.06, #0.08,
                 'alpha': 3.7,
                 'n': 1.24,
                 'wr': 0.0,
                 'ksat': 5e-5, #5e-5,5e-6
                 },
                
            'xeric':
                {'soil_id': 4,
                 'poros': 0.48,
                 'fc': 0.14,
                 'wp': 0.04,
                 'alpha': 3.8,
                 'n': 1.42,
                 'wr': 0.03,
                 'ksat': 1e-4,
                },
            'Peat':
                {'soil_id': 0,
                 'poros': 0.9,
                 'fc': 0.41,
                 'wp': 0.11,
                 'alpha': 1.23,
                 'n': 1.28,
                 'wr': 0.00,
                 'ksat': 1e-5,
                },
            }      

    return psoil

def soil_properties():
    """ 
    defines class-PTF for obtaining soil properties from soil type
    """
    psoil = {
             'FineTextured': 
                 {'airentry': 34.2,
                  'alpha': 0.018,
                  'beta': 7.9,
                  'fc': 0.34,
                  'ksat': 1e-06,
                  'n': 1.16,
                  'poros': 0.5,
                  'soil_id': 3.0,
                  'wp': 0.25,
                  'wr': 0.07,
                 },

             'MediumTextured': 
                 {'airentry': 20.8,
                  'alpha': 0.024,
                  'beta': 4.7,
                  'fc': 0.33,
                  'ksat': 1e-05,
                  'n': 1.2,
                  'poros': 0.43,
                  'soil_id': 2.0,
                  'wp': 0.13,
                  'wr': 0.05,
                 },

            'CoarseTextured':
                 {'airentry': 14.7,
                  'alpha': 0.039,
                  'beta': 3.1,
                  'fc': 0.21,
                  'ksat': 0.0001,
                  'n': 1.4,
                  'poros': 0.41,
                  'soil_id': 1.0,
                  'wp': 0.1,
                  'wr': 0.05,
                 },

             'Peat':
                 {'airentry': 29.2,
                  'alpha': 0.123,
                  'beta': 6.0,
                  'fc': 0.414,
                  'ksat': 5e-05,
                  'n': 1.28,
                  'poros': 0.9,
                  'soil_id': 4.0,
                  'wp': 0.11,
                  'wr': 0.0,
                 },
              'Humus':
                 {'airentry': 29.2,
                  'alpha': 0.123,
                  'beta': 6.0,
                  'fc': 0.35,
                  'ksat': 8e-06,
                  'n': 1.28,
                  'poros': 0.85,
                  'soil_id': 5.0,
                  'wp': 0.15,
                  'wr': 0.01,
                 },
            }
    return psoil

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
            print(key)
            ix = np.where(sitetype_id == c)
            data['poros'][ix] = psoil[key]['poros']
            data['fc'][ix] = psoil[key]['fc']
            data['wp'][ix] = psoil[key]['wp']
            data['ksat'][ix] = psoil[key]['ksat']
            data['alpha'][ix] = psoil[key]['alpha']
            data['n'][ix] = psoil[key]['n']
            data['wr'][ix] = psoil[key]['wr']
            del ix

        data['soilcode'] = sitetype_id
    return data

# def parameters():
#     pgen = {#'catchment_id': None,
#             #'gis_folder': r'data/C14/',
#             'forcing_file': r'data/SVE_saa.csv',
#             'ncf_file': r'results',
#             'start_date': '2000-01-01',
#             'end_date': '2016-12-31',
#             'spinup_end': '2000-12-31',
#             'dt': 86400.0,
#             'spatial_cpy': True,
#             'spatial_soil': True     
#            }
    
#     # canopygrid
#     pcpy = {'loc': {'lat': 61.4, 'lon': 23.7},
#             'flow' : { # flow field
#                      'zmeas': 2.0,
#                      'zground': 0.5,
#                      'zo_ground': 0.01
#                      },
#             'interc': { # interception
#                         'wmax': 1.5, # storage capacity for rain (mm/LAI)
#                         'wmaxsnow': 4.5, # storage capacity for snow (mm/LAI),
#                         },
#             'snow': {
#                     # degree-day snow model
#                     'kmelt': 2.8934e-05, # melt coefficient in open (mm/s)
#                     'kfreeze': 5.79e-6, # freezing coefficient (mm/s)
#                     'r': 0.05 # maximum fraction of liquid in snow (-)
#                     },
                    
#             'physpara': {
#                         # canopy conductance
#                         'amax': 10.0, # maximum photosynthetic rate (umolm-2(leaf)s-1)
#                         'g1_conif': 2.1, # stomatal parameter, conifers
#                         'g1_decid': 3.5, # stomatal parameter, deciduous
#                         'q50': 50.0, # light response parameter (Wm-2)
#                         'kp': 0.6, # light attenuation parameter (-)
#                         'rw': 0.20, # critical value for REW (-),
#                         'rwmin': 0.02, # minimum relative conductance (-)
#                         # soil evaporation
#                         'gsoil': 1e-2 # soil surface conductance if soil is fully wet (m/s)
#                         },
#             'phenopara': {
#                         #seasonal cycle of physiology: smax [degC], tau[d], xo[degC],fmin[-](residual photocapasity)
#                         'smax': 18.5, # degC
#                         'tau': 13.0, # days
#                         'xo': -4.0, # degC
#                         'fmin': 0.05, # minimum photosynthetic capacity in winter (-)
#                         # deciduos phenology
#                         'lai_decid_min': 0.1, # minimum relative LAI (-)
#                         'ddo': 45.0, # degree-days for bud-burst (5degC threshold)
#                         'ddur': 23.0, # duration of leaf development (days)
#                         'sdl': 9.0, # daylength for senescence start (h)
#                         'sdur': 30.0, # duration of leaf senescence (days),
#                          },
#             'state': {
#                        'lai_conif': 3.5, # conifer 1-sided LAI (m2 m-2)
#                        'lai_decid_max': 0.5, # maximum annual deciduous 1-sided LAI (m2 m-2): 
#                        'hc': 16.0, # canopy height (m)
#                        'cf': 0.6, # canopy closure fraction (-)
#                        #initial state of canopy storage [mm] and snow water equivalent [mm]
#                        'w': 0.0, # canopy storage mm
#                        'swe': 0.0, # snow water equivalent mm
#                        }
#             }

        
#     # BUCKET
#     pbu = {'depth': 0.4,  # root zone depth (m)
#            # following soil properties are used if spatial_soil = False
#            'poros': 0.43, # porosity (-)
#            'fc': 0.33, # field capacity (-)
#            'wp': 0.13,	 # wilting point (-)
#            'ksat': 2.0e-6, 
#            'beta': 4.7,
#            #organic (moss) layer
#            'org_depth': 0.04, # depth of organic top layer (m)
#            'org_poros': 0.9, # porosity (-)
#            'org_fc': 0.31, # field capacity (-)
#            'org_rw': 0.24, # critical vol. moisture content (-) for decreasing phase in Ef
#            'maxpond': 0.0, # max ponding allowed (m)
#            #initial states: rootzone and toplayer soil saturation ratio [-] and pond storage [m]
#            'rootzone_sat': 0.6, # root zone saturation ratio (-)
#            'org_sat': 1.0, # organic top layer saturation ratio (-)
#            'pond_sto': 0.0 # pond storage
#            }
    
#     return pgen, pcpy, pbu