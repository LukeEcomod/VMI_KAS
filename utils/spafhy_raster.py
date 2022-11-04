import pickle
import pathlib
from spafhy_graphics import field_on_map
vihti_raster_path=str(pathlib.Path("../SpaFHyData/rasters_vihti/"))
vihti_raster_file=vihti_raster_path+str(pathlib.Path("vihti_rasters.pk"))

def read_basemap(raster_file):
    """
    raster_file: pickle file containing base (background) map information
    read map (peruskartta),  its colormap and map boundaries from the rasterfile
    make 2-tuple basemap=(map,colormap)
    return 2-tuple (basemap,boundaries) 
    """
    f = open(str(raster_file)), 'rb')
    rasters = pickle.load(f)
    map = rasters['peruskartta']
    colormap = rasters['pkcmap']
    basemap = (map,colormap)
    boundaries = rasters['boundaries']
    f.close()
    return (basemap,boundaries)
    
