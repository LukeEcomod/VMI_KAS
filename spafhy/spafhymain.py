import os
import pickle
import numpy as np
import argparse
from netCDF4 import Dataset
import pandas as pd
import spafhy
from spafhy_io import read_FMI_weather

#Vihti-specific
from spafhy_parameters import soil_properties
from spafhy_parameters import parameters_vihti as parameters
from spafhy_io import create_vihti_catchment as create_catchment
""" set up SpaFHy for the catchment """

class SpaFHyRun:
    def __init__(self):
        """
        SpaFHy parameters and catchment data for initialization
        are now in python functions. Each catchment area (e,g, Vihti) 
        in its own function. But for generality imported names are 
        always 'parameters' for the parameters and 'create_catchment' for 
        catchment data. See above
        """
        #SpaFHy model
        self.spa = None
        #Catchment representation
        self.gisdata = None
        #The original full FMI data at Luke
        self.FORC = None
        #Sliced FMI data e.g. for SpaFHuy runs
        self.FORC_current=None
        #All these above are next intialized
        self.spafhy_initialize()
    def first_date_forc(self):
        return self.FORC.index[0]
    def last_date_forc(self):
        length=len(self.FORC.index)
        return self.FORC.index[length-1]
    def first_date_forc_current(self):
        return self.FORC_current.index[0]
    def last_date_forc_current(self):
        length=self.length_forc_current()
        return self.FORC_current.index[length-1]
    def dimensions_forc(self):
        return np.shape(self.FORC)
    def dimensions_forc_current(self):
        return np.shape(self.FORC_current)
    def length_forc(self):
        return self.dimensions_forc()[0]
    def length_forc_current(self):
        return self.dimensions_forc_current()[0]
    def slice_forc(self,first_date,last_date):
        """
        Gracefully slice the FMI weather data between the two dates
        Give dates as type Timestamp (see FORC index values)
        """
        #t1 = pd.to_datetime(first_date,format='%Y%/m%d',errors='coerce')
        #t2 = pd.to_datetime(last_date,format='%Y%/m%d',errors='coerce')
        self.FORC_current = self.FORC[(self.FORC.index >= first_date) & (self.FORC.index <= last_date)]
    def spafhy_initialize(self):
        """
        The very first initialization based on parameters
        """
        #Load parameter dictionaries
        (pgen, pcpy, pbu, ptop) = parameters()
        psoil = soil_properties()

        #Read gis data and create necessary inputs for model initialization
        self.gisdata = create_catchment(pgen['catchment_id'], fpath=pgen['gis_folder'],
                                        plotgrids=False, plotdistr=False)
        #Initialize spafhy
        self.spa = spafhy.initialize(pgen, pcpy, pbu, ptop, psoil, self.gisdata, cpy_outputs=False, 
                                     bu_outputs=False, top_outputs=False, flatten=True)
        #Read forcing data, i.e. weather (available at Luke).
        #This should read all data for a long enough time period
        self.FORC = read_FMI_weather(pgen['catchment_id'],'19600101','21000101',
                                     sourcefile=pgen['forcing_file'])
        self.FORC['Prec'] = self.FORC['Prec'] / self.spa.dt  # mms-1
        self.FORC['U'] = 2.0 # use constant wind speed ms-1
        self.FORC_current = self.FORC

    def spafhy_initialize_from_pickle(self):
        """
        SpaFHy has been run and saved to pickle file.
        Now, initialize as for the first time but in the end
        spathy model will be read from the pickle file.
        """
        self.spafhy_initialize()
        self.spafhy_from_pickle()
    def spafhy_to_pickle_forc(self):
        """
        Utility method to pickle the weather data 
        """
        forc_file=self.spa.forc_file
        (root,ext)=os.path.splitext(forc_file)
        forc_pk_file=root+'.pk'
        print("FORC",forc_pk_file)
        with open(forc_pk_file,'wb') as f:
            pickle.dump(self.FORC,f)
    def spafhy_from_pickle_forc(self):
        """
        Utility method to set the weather data to match
        the end of the previous simulation and the status
        of the SpaFHy model. Use this e.g., when creating
        figures for the www page.
        """
        forc_file=self.spa.forc_file
        (root,ext)=os.path.splitext(forc_file)
        forc_pk_file=root+'.pk'
        with open(forc_pk_file,'rb') as f:
            self.spa.FORC =  pickle.load(f)
    def spafhy_to_pickle(self):
        with open(self.spa.pickle_file,'wb') as f:
            pickle.dump(self.spa,f)
        self.spafhy_to_pickle_forc()
    def spafhy_from_pickle(self):
        """
        Read the SpaFHy model from the pickle file.
        Note the weather data (FORC) will be set to 
        most recent FMI data available at Luke
        """
        with open(self.spa.pickle_file,'rb') as f:
            self.spa = pickle.load(f)
        self.spafhy_from_pickle_forc()
    def spafhy_first_run(self,from_date,to_date):
        """
        The vey first first run: Initialize netCDF file, slice FORC from 'from_date' to 'to_date',
        give the dates as Timestamp.
        Run spafhy saving results for each step in netCDF file and finally close the netCDF file.
        """ 
        #netCDF4 output file
        dlat, dlon = np.shape(self.spa.GisData['cmask'])
        (ncf, ncf_file) = spafhy.initialize_netCDF(ID=self.spa.id, fname=self.spa.ncf_file, lat0=self.spa.GisData['lat0'], 
                                                   lon0=self.spa.GisData['lon0'], dlat=dlat, dlon=dlon, dtime=None)
        self.slice_forc(from_date,to_date)
        for k in range(0,self.length_forc_current()):
            forc_line = self.FORC_current[['doy', 'Rg', 'Par', 'T', 'Prec', 'VPD', 'CO2','U']].iloc[k]
            self.spa.run_timestep(forc_line,ncf=ncf)
        ncf.close()
        
    def spafhy_run(self,from_date,to_date):
        """
        Subsequent runs following 'spafhy_first_run': continue running SpaFHy, e.g. 
        from the end date of the spafhy_first_run to the end of available weather data. 
        Slice FORC according to dates and run spafhy. Give dates as Timestamp.
        """
        ncf = Dataset(self.spa.ncf_file, 'a')
        self.slice_forc(from_date,to_date)
        for k in range(0,self.length_forc_current()):
            forc_line = self.FORC_current[['doy', 'Rg', 'Par', 'T', 'Prec', 'VPD', 'CO2','U']].iloc[k]
            self.spa.run_timestep(forc_line,ncf=ncf)
        #The 'with' clause should automatically close files when using 'open' function
        #I am not sure with 'Dataset' function.
        ncf.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d1','--from,type=str',dest='d1',help="From date as 'YYYYMMDD'")
    parser.add_argument('-d2','--to,type=str',dest='d2',help="End date as 'YYYYMMDD'")
    args=parser.parse_args() 
    spafhy_run = SpaFHyRun()
    spafhy_run.spafhy_first_run(spafhy_run.first_date_forc_current(),
                                spafhy_run.last_date_forc_current())
    spafhy_run.spafhy_to_pickle()
                                
