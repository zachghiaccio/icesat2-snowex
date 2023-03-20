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
    Converts ICESat-2 lat/lon coordinates into a KML file.
    Needs a beam specification.
    
    **Requires the "simplekml" Python package.***

    Parameters
    ----------
    is2_file : str
        The .h5 file containing ICESat-2 data. Currently only works with
        ATL08 data.
    beam: str
        The beam data to be read. Accepts: gt1r, gt1l, gt2r, gt2l, gt3r, gt3l

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
        
    # Save the new KML file
    rgt = is2_file[91:95]
    is2_kml.save('rgt%s_%s.kml' %(rgt, beam))
    
    return is2_kml

#\---------------------------------------------------------/#
def kml_to_coords(kml_file):
    """
    Takes the lat/lon coordinates from an ICESat-2 KML file
    and places them into numpy arrays.
    
    Parameters
    ----------
    kml_file: str
        A KML file containing ICESat-2 coordinate information.
        KMLs from the above function and from the ICESat-2
        website are both valid.
        
    Returns
    -------
    kml_lat: numpy array
        Array of latitudes for ICESat-2 track.
    kml_lon: numpy array
        Array of longitudes for ICESat-2 track.
        
    """
    
    # Open the KML file with osgeo
    ds = ogr.Open(kml_file)
    
    # Loop through the points within the KML file, and
    # add them to an array
    coords = []
    for lyr in ds:
        for feat in lyr:
            geom = feat.GetGeometryRef()
            if geom != None:
                for i in range(0, geom.GetPointCount()):
                    coords.append(geom.GetPoint(i))
                    
    coords = np.array(coords)
    
    # Separate the latitudes and longitudes into individual arrays
    kml_lat = coords[:,1]
    kml_lon = coords[:,0]
    
    return kml_lat, kml_lon

#\---------------------------------------------------------/#
def make_new_rgts(kml_file):
    """
    Takes KMLs from the ICESat-2 website, or center beams 
    generated from is2_kml_maker, to approximate the left
    and right beam paths. The paths are saved into separate
    KML files.
    
    **Requires the "simplekml" Python package.***
    
    Parameters
    ----------
    kml_file: str
        A KML file containing ICESat-2 coordinate information.
        KMLs from the above function and from the ICESat-2
        website are both valid.
        
    Returns
    ----------
    kml_left: KML file
        KML track for left beam(s).
    kml_right: KML file
        KML track for right beam(s).
        
    """
    
    # Convert KML coordinates to numpy arrays
    kml_lat,kml_lon = kml_to_coords(kml_file)
    
    # Estimate location of left and right beams, given
    # a beam pair spacing of ~3.39 km
    dellon = 3390./(np.cos(np.radians(kml_lat))*111000)
    lon_right = kml_lon + dellon
    lon_left = kml_lon - dellon
 
    # Initialize new KML files
    kml_right = simplekml.Kml()
    kml_left = simplekml.Kml()
    
    # Assign labels to KMLs
    ls_right = kml_right.newlinestring(name='left beam')
    ls_left = kml_left.newlinestring(name='right beam')
    
    # Assign coordinates to 2D arrays
    coords_right = np.empty([len(lon_left), 2])
    coords_left = np.empty([len(lon_right), 2])
    
    coords_right[:,0],coords_right[:,1] = lon_right,kml_lat
    coords_left[:,0],coords_left[:,1] = lon_left,kml_lat
    
    # Add lat/lon coordinates to KMLs
    for row in coords_right:
        ls_right.coords.addcoordinates([(row[0], row[1])])
        
    for row in coords_left:
        ls_left.coords.addcoordinates([(row[0], row[1])])
    

    # Save the new KML files
    kml_right.save('approx_right_beams.kml')
    kml_left.save('approx_left_beams.kml)
    
    return kml_right, kml_left
                  
#\---------------------------------------------------------/#
def make_too_track(kml_file, lat0, lon0, latn, lonn):
    """
    Approximates the location of a TOO track tasked for ICESat-2.
    Requires the original ICESat-2 track in KML form, the coordinates
    of the requested TOO, and approximate coordinates from the
    original track.
    
    Parameters
    ----------
    kml_file: str
        A KML file containing ICESat-2 coordinate information.
        KMLs from "is2_kml_maker" and from the ICESat-2
        website are both valid.
    lat0, lon0: float
        Lat/lon coordinates for a point along the original ICESat-2
        track. Should be close to the TOO coordinates.
    latn, lonn: float
        Lat/lon coordinates from the TOO tasking order.
        
    Returns
    ----------
    too_center: KML file
        KML track for TOO center beam(s).
    too_right: KML file
        KML track for TOO right beam(s).
    too_left: KML file
        KML track for TOO left beam(s).
    """
    
    # Calculate the change in coordinates
    dellat = latn - lonn
    dellon = lonn - lon0
    
    # Shift the original KML coordinates
    kml_lat, kml_lon = kml_to_coords(kml_file)
    kml_lat += dellat
    kml_lon += dellon
                  
    # Generate new KML for shifted track
    is2_kml = simplekml.Kml()
    ls = is2_kml.newlinestring(name='gt2l')
                  
    coords = np.empty([len(kml_lat), 2])
    coords[:,0], coords[:,1] = kml_lon, kml_lat
    for row in coords:
        ls.coords.addcoordinates([(row[0], row[1])])
        
    # Approximate the left and right beams, and save the new KMLs
    too_center = 'too_center_track.kml'
    is2_kml.save(too_center)
    too_right, too_left = make_new_rgts(too_center)
    
    return too_center, too_right, too_left