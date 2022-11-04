# -*- coding: utf-8 -*-
"""
PARAMETERS OF SPAFHY

Created on Mon Jun 25 18:34:12 2018

@author: slauniai
"""
#pathlib.Path converts '/' to '\' on windows,
#i.e. one can always use '/' in file path
#import pathlib
#str(pathlib.Path('../SpaFHyData/Runoff/Runoffs_SVEcatchments_mmd.csv')),

def parameters():
    pgen = {'catchment_id': None,
            'gis_folder': r'data/C14/',
            'forcing_file': r'data/SVE_saa.csv',
            'runoff_file': r'data/Runoffs_SVEcatchments_mmd.csv',
            'pickle_file': None,
            'ncf_file': r'results',
            'start_date': '2000-01-01',
            'end_date': '2016-12-31',
            'spinup_end': '2000-12-31',
            'dt': 86400.0,
            'spatial_cpy': True,
            'spatial_soil': True     
           }
    
    # canopygrid
    pcpy = {'loc': {'lat': 61.4, 'lon': 23.7},
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
                        'g1_conif': 2.1, # stomatal parameter, conifers
                        'g1_decid': 3.5, # stomatal parameter, deciduous
                        'q50': 50.0, # light response parameter (Wm-2)
                        'kp': 0.6, # light attenuation parameter (-)
                        'rw': 0.20, # critical value for REW (-),
                        'rwmin': 0.02, # minimum relative conductance (-)
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
                       'lai_conif': 3.5, # conifer 1-sided LAI (m2 m-2)
                       'lai_decid_max': 0.5, # maximum annual deciduous 1-sided LAI (m2 m-2): 
                       'hc': 16.0, # canopy height (m)
                       'cf': 0.6, # canopy closure fraction (-)
                       #initial state of canopy storage [mm] and snow water equivalent [mm]
                       'w': 0.0, # canopy storage mm
                       'swe': 0.0, # snow water equivalent mm
                       }
            }

        
    # BUCKET
    pbu = {'depth': 0.4,  # root zone depth (m)
           # following soil properties are used if spatial_soil = False
           'poros': 0.43, # porosity (-)
           'fc': 0.33, # field capacity (-)
           'wp': 0.13,	 # wilting point (-)
           'ksat': 2.0e-6, 
           'beta': 4.7,
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
    
    # TOPMODEL
    ptop = {'dt': 86400.0, # timestep (s)
            'm': 0.007, # Paunulanpuro # generic is 0.01, # scaling depth (m)
            'ko': 0.001, # transmissivity parameter (ms-1)
            'twi_cutoff': 99.5,  # cutoff of cumulative twi distribution (%)
            'so': 0.05 # initial saturation deficit (m)
           }
    
    return pgen, pcpy, pbu, ptop

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
                 'fc': 0.34,
                 'wp': 0.11,
                 'ksat': 1e-06,
                 'beta': 4.0,
                 'alpha': 4.06,
                 'n': 1.17,
                 'wr': 0.0,
                 },
            'mesic':
                {
                 'soil_id': 2,
                 'poros': 0.55,
                 'fc': 0.28,
                 'wp': 0.08,
                 'ksat': 1e-5,
                 'beta': 4.0,
                 'alpha': 4.48,
                 'n': 1.20,
                 'wr': 0.0,
                 },
            'sub-xeric':
                {'soil_id': 3,
                 'poros': 0.53,
                 'fc': 0.24,
                 'wp': 0.08,
                 'ksat': 5e-5,
                 'beta': 4.0,
                 'alpha': 3.7,
                 'n': 1.24,
                 'wr': 0.0,
                 },
                
            'xeric':
                {'soil_id': 4,
                 'poros': 0.48,
                 'fc': 0.14,
                 'wp': 0.04,
                 'ksat': 5e-04,
                 'beta': 4.0,
                 'alpha': 3.8,
                 'n': 1.42,
                 'wr': 0.03,
                },
            'Peat':
                {'soil_id': 0,
                 'poros': 0.9,
                 'fc': 0.41,
                 'wp': 0.11,
                 'ksat': 5e-5,
                 'beta': 4.0,
                 'n': 1.28,
                 'wr': 0.00,
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
