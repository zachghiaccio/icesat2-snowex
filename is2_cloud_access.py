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
def atl03q(field_id):
    icesat2.init('slideruleearth.io', verbose=False)
    
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
        region = icesat2.toregion('jsons-ships/acp_lidar_box.geojson')['poly']
    else:
        raise ValueError('Field ID not recognized, or not implemented yet.')
    
    parms = {
        "poly": region,
        "srt": icesat2.SRT_LAND,
        "cnf": icesat2.CNF_SURFACE_HIGH,
        "atl08_class": ["atl08_ground"],
        "ats": 5.0,
        "len": 20.0,
        "res": 10.0,
        "maxi": 5
    }
    
    atl03 = icesat2.atl06p(parms, 'nsidc-s3')
    
    return atl03

#\---------------------------------------------------------/#
def atl06q(field_id, date_range, rgt):
    
    # Specify the ICESat-2 product
    short_name = 'ATL06'
    
    # Define the spatial extent using a pre-generated bounding box
    with open('snowex_sites_for_icepyx.pkl', 'rb') as f:
        coordinates = pickle.load(f)
        spatial_extent = coordinates['alaska']
        
    # Generate the query object
    region = ipx.Query(short_name, spatial_extent, date_range, tracks=rgt)
    
    # Set up s3 cloud access
    region.earthdata_login('example', 'query@example.edu', s3token=True)
    credentials = reg._s3login_credentials
    s3 = s3fs.S3FileSystem(key=credentials['accessKeyId'],
                           secret=credentials['secretAccessKey'],
                           token=credentials['sessionToken'])
    
    # Access the data through an s3 url
    gran_ids = region.avail_granules(ids=True, cloud=True)
    s3url = gran_ids[1][0]
    f = h5py.File(s3.open(s3url, 'rb'), 'r')
    
    # Process the data into a DataFrame
    atl06 = lp.beam_cycle_concat(f, 'ATL06')
    
    return atl06

#\---------------------------------------------------------/#
def atl08q(field_id, date_range, rgt):
    
    # Specify the ICESat-2 product
    short_name = 'ATL08'
    
    # Define the spatial extent using a pre-generated bounding box
    with open('snowex_sites_for_icepyx.pkl', 'rb') as f:
        coordinates = pickle.load(f)
        spatial_extent = coordinates['alaska']
        
    # Generate the query object
    region = ipx.Query(short_name, spatial_extent, date_range, tracks=rgt)
    
    # Set up s3 cloud access
    region.earthdata_login('example', 'query@example.edu', s3token=True)
    credentials = reg._s3login_credentials
    s3 = s3fs.S3FileSystem(key=credentials['accessKeyId'],
                           secret=credentials['secretAccessKey'],
                           token=credentials['sessionToken'])
    
    # Access the data through an s3 url
    gran_ids = region.avail_granules(ids=True, cloud=True)
    s3url = gran_ids[1][0]
    f = h5py.File(s3.open(s3url, 'rb'), 'r')
    
    # Process the data into a DataFrame
    atl06 = lp.beam_cycle_concat(f, 'ATL08')
    
    return atl08