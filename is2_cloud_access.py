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
from pyproj import Transformer, CRS
import s3fs
from shapely.geometry import Polygon, Point
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

#\---------------------------------------------------------/#
def atl03q(field_id, date_range, rgt, version='006',
           cnf_surface=4, atl08_class=4,
           ats=5.0, segment_length=20.0, res=10.0, maxi=5):
    
    icesat2.init('slideruleearth.io', verbose=False)
    
    # Load geoJSON for field site of interest
    if field_id == 'cpcrw':
        # Caribou/Poker Creek, AK
        region = icesat2.toregion('jsons-shps/cpcrw_lidar_box.geojson')['poly']
    elif field_id == 'cffl':
        # Creamer's Field/Farmer's Loop, AK
        region = icesat2.toregion('jsons-shps/cffl_lidar_box.geojson')['poly']
    elif field_id == 'bcef':
        # Bonanza Creek, AK
        region = icesat2.toregion('jsons-shps/bcef_lidar_box.geojson')['poly']
    elif field_id == 'acp':
        # Arctic Coastal Plain, AK
        region = sliderule.toregion('jsons-shps/acp_lidar_box.geojson')['poly']
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
    
    if atl08_ids.get(atl08_class) == 'None':
        parms = {
            "poly": region,
            "rgt": rgt,
            "srt": icesat2.SRT_LAND,
            "cnf": cnf_surface,
            "ats": ats,
            "len": segment_length,
            "res": res,
            "maxi": maxi,
            "t0": date_range[0]+time_root,
            "t1": date_range[1]+time_root
        }
    else:
        parms = {
            "poly": region,
            "rgt": rgt,
            "srt": icesat2.SRT_LAND,
            "cnf": cnf_surface,
            "atl08_class": atl08_ids.get(atl08_class),
            "ats": ats,
            "len": segment_length,
            "res": res,
            "maxi": maxi,
            "t0": date_range[0]+time_root,
            "t1": date_range[1]+time_root
        }
    
    atl03 = icesat2.atl06p(parms)
    
    return atl03

#\---------------------------------------------------------/#
def atl06q(field_id, date_range, rgt, version='006'):
    
    # Specify the ICESat-2 product
    short_name = 'ATL06'
    
    # Define the spatial extent using a pre-generated bounding box
    with open('/home/jovyan/icesat2-snowex/jsons-shps/snowex_sites_for_icepyx.pkl', 'rb') as f:
        coordinates = pickle.load(f)
        spatial_extent = coordinates['alaska']
        
    # Generate the query object
    try:
        region = ipx.Query(short_name, spatial_extent, date_range, tracks=rgt)
        
        region.earthdata_login('zhfair', 'zhfair@umich.edu', s3token=True)
        #s3 = earthaccess.get_s3fs_session(daac='NSIDC', provider=region._s3login_credentials)
        credentials = region._session.get("https://data.nsidc.earthdatacloud.nasa.gov/s3credentials").json()
        s3 = s3fs.S3FileSystem(key=credentials['accessKeyId'],
                               secret=credentials['secretAccessKey'],
                               token=credentials['sessionToken'])
        gran_ids = region.avail_granules(ids=True, cloud=True)
    except:
        region = ipx.Query(short_name, spatial_extent, date_range, tracks=rgt, version=version)
    
        # Set up s3 cloud access - currently in a transition phase for the authentication
        region.earthdata_login('zhfair', 'zhfair@umich.edu', s3token=True)
        #s3 = earthaccess.get_s3fs_session(daac='NSIDC', provider=region._s3login_credentials)
        credentials = region._session.get("https://data.nsidc.earthdatacloud.nasa.gov/s3credentials").json()
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
    
    # Specify the ICESat-2 product
    short_name = 'ATL08'
    
    # Define the spatial extent using a pre-generated bounding box
    with open('/home/jovyan/icesat2-snowex/jsons-shps/snowex_sites_for_icepyx.pkl', 'rb') as f:
        coordinates = pickle.load(f)
        spatial_extent = coordinates['alaska']
        
    # Generate the query object
    try:
        region = ipx.Query(short_name, spatial_extent, date_range, tracks=rgt)
        
        # Set up s3 cloud access - currently in a transition phase for the authentication
        region.earthdata_login('zhfair', 'zhfair@umich.edu', s3token=True)
        credentials = region._session.get("https://data.nsidc.earthdatacloud.nasa.gov/s3credentials").json()
        s3 = s3fs.S3FileSystem(key=credentials['accessKeyId'],
                               secret=credentials['secretAccessKey'],
                               token=credentials['sessionToken'])
        gran_ids = region.avail_granules(ids=True, cloud=True)
    except:
        region = ipx.Query(short_name, spatial_extent, date_range, tracks=rgt, version=version)
    
        # Set up s3 cloud access - currently in a transition phase for the authentication
        region.earthdata_login('zhfair', 'zhfair@umich.edu', s3token=True)
        credentials = region._session.get("https://data.nsidc.earthdatacloud.nasa.gov/s3credentials").json()
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