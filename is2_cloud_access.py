import earthaccess
import ipywidgets as widgets
import logging
import concurrent.futures
import time
from datetime import datetime
import h5py
import lidar_processing as lp
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import pickle
import py3dep
from pyproj import Transformer, CRS
import s3fs
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
from sliderule import icesat2
from sliderule import sliderule, ipysliderule, io

import xarray as xr
import geoviews as gv
import geoviews.feature as gf
from geoviews import dim, opts
import geoviews.tile_sources as gts
from bokeh.models import HoverTool
import hvplot.pandas

import icepyx as ipx
gv.extension('bokeh')

"""
A series of functions designed to streamline cloud data access. Originally designed for three ICESat-2 products (SlideRule, ATL06, and ATL08), it is being expanded to include complimentary datasets, including 3DEP, ArcticDEM (through SlideRule), and UAF airborne lidar.

The sliderule, earthaccess, and py3dep packages are required.
"""

#\---------------------------------------------------------/#
def atl03q(field_id, date_range, rgt, version='006',
           cnf_surface=4, atl08_class=4,
           ats=5.0, segment_length=20.0, res=10.0, 
           maxi=5, arctic_dem=False):    
    """
    Generates SlideRule data using user-defined inputs. A series of default parameters are pre-defined.

    Parameters
    ----------
    field_id: string
        SnowEx field site identifier:
            * "cffl": Creamer's Field/Farmer's Loop
            * "bcef": Bonanza Creek Experimental Forest
            * "cpcrw": Caribou/Poker Creek Research Watershed
            * "utk": Upper Kuparuk/Toolik
            * "acp": Arctic Coastal Plain
    date_range: string list
        A list containing a start date and end date, in string format.
        Ex: ["2023-03-01", "2023-03-31"]
    rgt: string
        ICESat-2 reference ground track(s) (RGTs), for track-specific analyses.
        Use "all" if all RGTs are desired
        Ex: "1356"
    version: string
        ICESat-2 version number. Use if ICESat-2 data is transitioning to a new version (i.e., v007).
        Default: "006"
    cnf_surface: int
        Numeric identifier for the level of confidence of the ATL03 photons used by SlideRule.
            * All photons, including noise: 0
            * Signal/background, no noise: 1
            * Low/medium/high signal: 2
            * Medium/high signal: 3
            * High signal only: 4
        Default: 4 (high confidence photons only)
    atl08_class: int
        Numeric identifier for the ATL08 filtering scheme implemented by SlideRule.
            * No classification: -1
            * "atl08_unclassified": 0
            * "atl08_noise": 1
            * "atl08_canopy": 2
            * "atl08_top_of_canopy": 3
            * "atl08_ground": 4
        Default: 4 (ground photons only)
    ats: float/int
        The minimum spread/uncertainty in meters in along-track photons used for height estimation.
        Default: 5.0
    segment_length: float/int
        Length of along-track segments used to derive surface elevation, in meters.
        Default: 20.0
    res: float/int
        Distance between two successive along-track segments, in meters.
        Default: 10.0
    maxi: int
        Maximum number of iterations used for the photon refinement algorithm.
        Default: 5
    arctic_dem: boolean
        If True, adds ArcticDEM mosaic to the SlideRule request with a series of statistics.
        Default: False

    Returns
    -------
    atl03: geodataframe
        SlideRule output given the above input parameters.
    """
    
    icesat2.init('slideruleearth.io', verbose=False)
    
    # Load geoJSON for field site of interest
    if field_id == 'cpcrw':
        # Caribou/Poker Creek, AK
        region = sliderule.toregion('jsons-shps/cpcrw_lidar_box.geojson')['poly']
    elif field_id == 'cffl':
        # Creamer's Field/Farmer's Loop, AK
        region = sliderule.toregion('jsons-shps/cffl_lidar_box.geojson')['poly']
    elif field_id == 'bcef':
        # Bonanza Creek, AK
        region = sliderule.toregion('jsons-shps/bcef_lidar_box.geojson')['poly']
    elif field_id == 'acp':
        # Arctic Coastal Plain, AK
        region = sliderule.toregion('jsons-shps/acp_lidar_box.geojson')['poly']
    elif field_id == 'utk':
        # Toolik Station, AK
        region = sliderule.toregion('jsons-shps/toolik_lidar_boxes.geojson')['poly']
    else:
        raise ValueError('Field ID not recognized, or not implemented yet.')
    
    # Convert user-defined ATL08 class ID to string readable by SlideRule
    atl08_ids = {-1: 'None',
                 0: 'atl08_unclassified',
                 1: 'atl08_noise',
                 2: 'atl08_canopy',
                 3: 'atl08_top_of_canopy',
                 4: 'atl08_ground'}
    
    time_root = 'T00:00:00Z'

    parms = {
             "poly": region,
             "srt": icesat2.SRT_LAND,
             "cnf": cnf_surface,
             "ats": ats,
             "len": segment_length,
             "res": res,
             "maxi": maxi,
             "t0": date_range[0]+time_root,
             "t1": date_range[1]+time_root
            }

    if rgt != "all":
        parms["rgt"] = rgt
        print(f"Subsetting to only include ICESat-2 RGT {rgt}.")

    if atl08_ids.get(atl08_class) != "None":
        parms["atl08_class"] = atl08_ids.get(atl08_class)
        print("Subsetting by selected ATL08 filter...")

    if arctic_dem:
        parms["samples"] = {"mosaic": {"asset": "arcticdem-mosaic", "radius": 10.0, "zonal_stats": True}}
        print("Adding ArcticDEM mosaic to SlideRule request...")

    atl03 = icesat2.atl06p(parms)
    
    return atl03

#\---------------------------------------------------------/#
def atl06q(field_id, date_range, rgt, version='006'):
    """
    Calls icepyx to retrieve ATL06 data using user-defined inputs.

    Parameters
    ----------
    field_id: string
        SnowEx field site identifier:
            * "cffl": Creamer's Field/Farmer's Loop
            * "bcef": Bonanza Creek Experimental Forest
            * "cpcrw": Caribou/Poker Creek Research Watershed
            * "utk": Upper Kuparuk/Toolik
            * "acp": Arctic Coastal Plain
    date_range: string list
        A list containing a start date and end date, in string format.
        Ex: ["2023-03-01", "2023-03-31"]
    rgt: string
        ICESat-2 reference ground track(s) (RGTs), for track-specific analyses.
        Ex: "1356"
    version: string
        ICESat-2 version number. Use if ICESat-2 data is transitioning to a new version (i.e., v007).
        Default: "006"

    Returns
    -------
    atl06: geodataframe
        ATL06 data retrieved with icepyx given the above input parameters.
    """
    
    # Specify the ICESat-2 product
    short_name = 'ATL06'
    
    # Define the spatial extent using a pre-generated bounding box
    with open('/home/jovyan/icesat2-snowex/jsons-shps/snowex_sites_for_icepyx.pkl', 'rb') as f:
        coordinates = pickle.load(f)
        spatial_extent = coordinates['alaska']
        
    # Generate the query object
    try:
        if rgt == 'all':
            region = ipx.Query(short_name, spatial_extent, date_range)
        else:
            region = ipx.Query(short_name, spatial_extent, date_range, tracks=rgt)
        
        region.earthdata_login('zhfair', 'zhfair@umich.edu', s3token=True)
        credentials = region._session.get("https://data.nsidc.earthdatacloud.nasa.gov/s3credentials").json()
        s3 = s3fs.S3FileSystem(key=credentials['accessKeyId'],
                               secret=credentials['secretAccessKey'],
                               token=credentials['sessionToken'])
        gran_ids = region.avail_granules(ids=True, cloud=True)
    except:
        if rgt == 'all':
            region = ipx.Query(short_name, spatial_extent, date_range, version=version)
        else:
            region = ipx.Query(short_name, spatial_extent, date_range, tracks=rgt)
    
        # Set up s3 cloud access - currently in a transition phase for the authentication
        credentials = region.s3login_credentials
        s3 = s3fs.S3FileSystem(key=credentials['accessKeyId'],
                               secret=credentials['secretAccessKey'],
                               token=credentials['sessionToken'])
        gran_ids = region.avail_granules(ids=True, cloud=True)
    
    # Access the data through an s3 url
    s3url = gran_ids[1][0]
    f = s3.open(s3url, 'rb')
    f = [f]
    
    # Process the data into a DataFrame
    atl06 = lp.beam_cycle_concat(f, 'ATL06')
    
    return atl06

#\---------------------------------------------------------/#
def atl08q(field_id, date_range, rgt, version='006'): 
    """
    Calls icepyx to retrieve ATL08 data using user-defined inputs.

    Parameters
    ----------
    field_id: string
        SnowEx field site identifier:
            * "cffl": Creamer's Field/Farmer's Loop
            * "bcef": Bonanza Creek Experimental Forest
            * "cpcrw": Caribou/Poker Creek Research Watershed
            * "utk": Upper Kuparuk/Toolik
            * "acp": Arctic Coastal Plain
    date_range: string list
        A list containing a start date and end date, in string format.
        Ex: ["2023-03-01", "2023-03-31"]
    rgt: string
        ICESat-2 reference ground track(s) (RGTs), for track-specific analyses.
        Ex: "1356"
    version: string
        ICESat-2 version number. Use if ICESat-2 data is transitioning to a new version (i.e., v007).
        Default: "006"

    Returns
    -------
    atl08: geodataframe
        ATL08 data retrieved with icepyx given the above input parameters.
    """
    
    # Specify the ICESat-2 data product
    short_name = "ATL08"

    # Define the spatial extent using a pre-generated bounding box
    with open('/home/jovyan/icesat2-snowex/jsons-shps/snowex_sites_for_icepyx.pkl', 'rb') as f:
        coordinates = pickle.load(f)
        spatial_extent = coordinates['alaska']
        
    # Generate the query object
    try:
        if rgt == 'all':
            region = ipx.Query(short_name, spatial_extent, date_range)
        else:
            region = ipx.Query(short_name, spatial_extent, date_range, tracks=rgt)
        
        # Set up s3 cloud access - currently in a transition phase for the authentication
        region.earthdata_login('zhfair', 'zhfair@umich.edu', s3token=True)
        credentials = region._session.get("https://data.nsidc.earthdatacloud.nasa.gov/s3credentials").json()
        s3 = s3fs.S3FileSystem(key=credentials['accessKeyId'],
                               secret=credentials['secretAccessKey'],
                               token=credentials['sessionToken'])
        gran_ids = region.avail_granules(ids=True, cloud=True)
    except:
        if rgt == 'all':
            region = ipx.Query(short_name, spatial_extent, date_range, version=version)
        else:
            region = ipx.Query(short_name, spatial_extent, date_range, tracks=rgt, version=version)
    
        # Set up s3 cloud access - currently in a transition phase for the authentication
        credentials = region.s3login_credentials
        s3 = s3fs.S3FileSystem(key=credentials['accessKeyId'],
                               secret=credentials['secretAccessKey'],
                               token=credentials['sessionToken'])
        gran_ids = region.avail_granules(ids=True, cloud=True)
    
    # Access the data through an s3 url
    s3url = gran_ids[1][0]
    f = s3.open(s3url, 'rb')
    f = [f]
  
    # Process the data into a DataFrame
    atl08 = lp.beam_cycle_concat(f, 'ATL08')
    
    return atl08

#\---------------------------------------------------------/#
def threedepq(field_id):
    """
    Calls py3dep to retrieve 3DEP data over a SnowEx site of interest.

    Parameters
    ----------
    field_id: string
        SnowEx field site identifier:
            * "cffl": Creamer's Field/Farmer's Loop
            * "bcef": Bonanza Creek Experimental Forest
            * "cpcrw": Caribou/Poker Creek Research Watershed
            * "utk": Upper Kuparuk/Toolik
            * "acp": Arctic Coastal Plain

    Returns
    -------
    three_dep: Xarray
        3DEP data array over a field site of interest.
    """
    
    # Load geoJSON for field site of interest
    if field_id == 'cpcrw':
        # Caribou/Poker Creek, AK
        region = gpd.read_file('jsons-shps/cpcrw_lidar_box.geojson').geometry[0]
    elif field_id == 'cffl':
        # Creamer's Field/Farmer's Loop, AK
        region = gpd.read_file('jsons-shps/cffl_lidar_box.geojson').geometry[0]
    elif field_id == 'bcef':
        # Bonanza Creek, AK
        region = gpd.read_file('jsons-shps/bcef_lidar_box.geojson').geometry[0]
    elif field_id == 'acp':
        # Arctic Coastal Plain, AK
        region = gpd.read_file('jsons-shps/acp_lidar_box.geojson').geometry[0]
    elif field_id == 'utk':
        # Toolik Station, AK (extra step needed to merge three ROI polygons)
        tmp = gpd.read_file('jsons-shps/toolik_lidar_boxes.geojson').geometry
        region = unary_union(tmp.geometry.values)
    else:
        raise ValueError('Field ID not recognized, or not implemented yet.')

    # Access 3DEP data using the py3dep package
    three_dep = py3dep.get_map("DEM", region, resolution=10, geo_crs=4326)
    
    return three_dep