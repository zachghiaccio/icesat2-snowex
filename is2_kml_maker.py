# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 15:44:53 2022

@author: zfair
"""

import h5py
import simplekml
import numpy as np

from osgeo import ogr

def is2_kml_maker(is2_file, beam):
    """
    A quick script to convert ICESat-2 lat/lon coordinates into a KML file.
    Needs a beam specification.

    Parameters
    ----------
    is2_file : str
        The .h5 file containing ICESat-2 data. Currently only works with
        ATL08 data.
    beam: str
        The beam data to be read in. Accepts: gt1r, gt1l, gt2r, gt2l, gt3r, gt3l

    Returns
    -------
    is2_kml : KML object
        KML of desired ICESat-2 track/beam to be saved. The KML is already
        saved as part of the function - the returned variable is for backup.

    """
    
    # Read in the ICESat-2 file, with the beam specification
    beam_dir = '/%s/land_segments/' % (beam)
    with h5py.File(is2_file, 'r') as f:
        lat = f[beam_dir+'latitude'][:]
        lon = f[beam_dir+'longitude'][:]
        
    # Add the coordinates to an array
    coords = np.empty([len(lat), 2])
    coords[:,0], coords[:,1] = lon, lat
    
    # Setup the KML
    is2_kml = simplekml.Kml()
    ls = is2_kml.newlinestring(name='%s' %(beam))
    
    # Add the coordinates to the KML
    for row in coords:
        ls.coords.addcoordinates([(row[0], row[1])])
        
    rgt = is2_file[91:95]
    is2_kml.save('rgt%s_%s.kml' %(rgt, beam))
    
    return is2_kml

#\---------------------------------------------------------/#
def kml_to_coords(kml_file):
    
    ds = ogr.Open(kml_file)
    
    coords = []
    for lyr in ds:
        for feat in lyr:
            geom = feat.GetGeometryRef()
            if geom != None:
                for i in range(0, geom.GetPointCount()):
                    coords.append(geom.GetPoint(i))
                    
    coords = np.array(coords)
    
    kml_lat = coords[:,1]
    kml_lon = coords[:,0]
    
    return kml_lat, kml_lon

#\---------------------------------------------------------/#
def make_new_rgts(kml_file):
    
    kml_lat,kml_lon = kml_to_coords(kml_file)
    
    dellon = 3390./(np.cos(np.radians(kml_lat))*111000)
    lon_3l = kml_lon + dellon
    lon_1l = kml_lon - dellon
    #raise ValueError('debug')
    kml_gt1l = simplekml.Kml()
    kml_gt3l = simplekml.Kml()
    
    ls3l = kml_gt3l.newlinestring(name='gt3l')
    ls1l = kml_gt1l.newlinestring(name='gt1l')
    
    coords_gt3l = np.empty([len(lon_3l), 2])
    coords_gt1l = np.empty([len(lon_1l), 2])
    
    coords_gt3l[:,0],coords_gt3l[:,1] = lon_3l,kml_lat
    coords_gt1l[:,0],coords_gt1l[:,1] = lon_1l,kml_lat
    
    #raise ValueError('debug')
    for row in coords_gt3l:
        ls3l.coords.addcoordinates([(row[0], row[1])])
        
    for row in coords_gt1l:
        ls1l.coords.addcoordinates([(row[0], row[1])])
    
    rgt = kml_file[102:106]
    try:
    
        kml_gt1l.save('C:/Users/zfair/OneDrive - NASA/Documents/Python/icesat2-snow/rgt%s_gt1l' %(rgt))
        kml_gt3l.save('C:/Users/zfair/OneDrive - NASA/Documents/Python/icesat2-snow/rgt%s_gt3l' %(rgt))
    except:
        rgt = kml_file[113:117]
        
        kml_gt1l.save('C:/Users/zfair/OneDrive - NASA/Documents/Python/icesat2-snow/rgt%s_gt1l' %(rgt))
        kml_gt3l.save('C:/Users/zfair/OneDrive - NASA/Documents/Python/icesat2-snow/rgt%s_gt3l' %(rgt))
    
    return kml_gt3l, kml_gt1l