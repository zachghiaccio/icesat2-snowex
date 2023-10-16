"""
Created on Wed Aug 16 16:03:00 2023

@author: zfair


Description
--------------
This script is a condensed form of is2_snowex_ak.ipynb, designed to estimate snow depth and produce output files without the hassle of running individual cells repeatedly. In summary, this script uses cloud computing capabilities in SlideRule and icepyx to derive ICESat-2 snow depths over SnowEx field sites in Alaska. Currently, only ICESat-2 and UAF airborne lidar are considered.

Work in progress: allowing users to specify SlideRule parameters.
"""

#---------------#
# General packages
from cartopy import crs
import geopandas as gpd
import numpy as np
import os
import pandas as pd
from pyproj import Proj, transform, Transformer, CRS
import rioxarray as rio
from shapely import wkt
from shapely.geometry import Polygon, Point
import sys

# Self-made packages/functions
import ground_data_processing as gdp
from is2_cloud_access import atl03q, atl06q, atl08q
import lidar_processing as lp

# Data query packages
import icepyx as ipx
from sliderule import icesat2, sliderule
#---------------#

#---------------#
# User Input
#---------------#
"""
Acceptable field site IDs over Alaska are:
    * 'cffl': Creamer's Field/Farmer's Loop
    * 'cpcrw': Caribou/Poker Creek Experimental Watershed
    * 'bcef': Bonanza Creek Experimental Forest
    * 'acp': Arctic Coastal Plain
    
    Note: SnowEx data was also gathered for Toolik Research Station, but airborne data is not available at this time. CPCRW has data available for March 2022, but not October 2022.
    
Acceptable IDs for Sliderule ATL08 class (use numeric ID):
    * No classification: -1
    * 'atl08_unclassified': 0
    * 'atl08_noise': 1
    * 'atl08_canopy': 2
    * 'atl08_top_of_canopy': 3
    * 'atl08_ground': 4
"""

# Field site ID
field_id = 'acp'

# Snow-on (True) or snow-off (False) analysis
snow_on = False

# Use March UAF data ('mar') or October depths ('oct')
uaf_depths = 'mar'

# Base data path
path = '/home/jovyan/icesat2-snowex'

# Desired RGT and date range for data queries
date_range = ['2022-03-01', '2023-09-31']
rgt = '1097'

# SlideRule parameters (optional)
cnf_surface = 4
atl08_class = 4
segment_length = 20
res = 10


#---------------#
# Read UAF Lidar Data
#---------------#
"""
Note that the snow-off lidar DEM is needed to estimate ICESat-2 snow depths, so it is loaded even for snow-on analyses.
"""

if field_id == 'cffl':
    f_snow_off = f'{path}/lidar-dems/farmersloop_2022may28_dtm_3m.tif'
    
    if uaf_depths == 'mar':
        f_snow_on = f'{path}/lidar-dems/farmersloop_2022mar11_snowdepth_3m.tif'
    elif uaf_depths == 'oct':
        f_snow_on = f'{path}/lidar-dems/farmersloop_2022oct24_snowdepth_3m.tif'
    else:
        print('Not data found for given field_id/uaf_depths combination.')
        
elif field_id == 'cpcrw':
    f_snow_off = '%s/lidar-dems/caribou_2022may29_dtm_3m.tif' %(path)
    
    if uaf_depths == 'mar':
        f_snow_on = '%s/lidar-dems/caribou_2022mar11_snowdepth_3m.tif' %(path)
    elif uaf_depths == 'oct':
        print('UAF lidar data currently not available for October 2022.')
        
elif field_id == 'bcef':
    f_snow_off = '%s/lidar-dems/bonanza_2022may28_dtm_3m.tif' %(path)
    
    if uaf_depths == 'mar':
        f_snow_on = '%s/lidar-dems/bonanza_2022mar11_snowdepth_3m.tif' %(path)
    elif uaf_depths == 'oct':
        f_snow_on = '%s/lidar-dems/bonanza_2022oct24_snowdepth_3m.tif' %(path)
        
elif field_id == 'acp':
    f_snow_off = '%s/lidar-dems/coastalplain_2022aug31_dtm_3m.tif' %(path)
    
    if uaf_depths == 'mar':
        f_snow_on = '%s/lidar-dems/coastalplain_2022mar12_snowdepth_3m.tif' %(path)
    elif uaf_depths == 'oct':
        f_snow_on = '%s/lidar-dems/coastalplain_2022oct26_snowdepth_3m.tif' %(path)
        
# Read lidar TIFFs into rioxarray format
lidar_snow_off = rio.open_rasterio(f_snow_off)
lidar_snow_on = rio.open_rasterio(f_snow_on)


#---------------#
# Read ICESat-2 Data
#---------------#
"""
icepyx will be used to access ATL06 and ATL08 data, whereas SlideRule will process ATL03 data with a specific set of parameters. See is2_cloud_access.py for more information on the parameters used.

User-defined SlideRule parameter reading is a work in progress.
"""

# Query the ICESat-2 data products
atl06sr = atl03q(field_id, date_range, rgt,
                 cnf_surface=cnf_surface,
                 atl08_class=atl08_class,
                 segment_length=segment_length,
                 res=res)
atl06 = atl06q(field_id, date_range, rgt)
atl08 = atl08q(field_id, date_range, rgt)


#---------------#
# Process ICESat-2 Data
#---------------#
"""
The queried icepyx data does not align exactly with our SlideRule product (work in progress to make them match up), so here we make sure they are within the same bounds. We also add easting/northing coordinates to each DataFrame and convert to a GeoDataFrame.
"""

# Limit ATL06/08 to SlideRule bounds
atl06 = atl06[(atl06.lat.values>atl06sr.geometry.y.min()) &
              (atl06.lat.values<atl06sr.geometry.y.max())]
atl08 = atl08[(atl08.lat.values>atl06sr.geometry.y.min()) &
              (atl08.lat.values<atl06sr.geometry.y.max())]

# Convert ATL06SR to geodataframe in EPSG:32606
atl06sr['lon'], atl06sr['lat'] = atl06sr.geometry.x, atl06sr.geometry.y
atl06sr_gdf = atl06sr.to_crs('epsg:32606')

# Rename ATL06SR height column, for consistency with other dataframes
atl06sr_gdf = atl06sr_gdf.rename(columns={'h_mean': 'height'})

# Convert ATL06/08 to geodataframes
atl06_gdf = gpd.GeoDataFrame(atl06, geometry=gpd.points_from_xy(atl06.lon, atl06.lat), crs='EPSG:4326')
atl06_gdf = atl06_gdf.to_crs('epsg:32606')

atl08_gdf = gpd.GeoDataFrame(atl08, geometry=gpd.points_from_xy(atl08.lon, atl08.lat), crs='EPSG:4326')
atl08_gdf = atl08_gdf.to_crs('epsg:32606')

# Remove filler/messy data from ATL06/08
upper = atl06_gdf['height'].mean() + 3*atl06_gdf['height'].std()
atl06_gdf = atl06_gdf.loc[atl06_gdf.height<upper]

upper = atl08_gdf['height'].mean() + 3*atl08_gdf['height'].std()
atl08_gdf = atl08_gdf.loc[atl08_gdf.height<upper]


#---------------#
# Co-register UAF and ICESat-2
#---------------#
"""
This section co-registers UAF lidar with each of the ICESat-2 data products. If snow_on==True, then the 'residual' column is renamed to 'is2_snow_depth', and a snow depth residual is calculated.
"""

# IS2/UAF co-registration
atl06sr_uaf = lp.coregister_is2(lidar_snow_off, lidar_snow_on, atl06sr_gdf)
atl06_uaf = lp.coregister_is2(lidar_snow_off, lidar_snow_on, atl06_gdf)
atl08_uaf = lp.coregister_is2(lidar_snow_off, lidar_snow_on, atl08_gdf)

# If snow on, then rename residual column and calculate snow depth residuals
if snow_on:
    atl06sr_uaf['snow_depth_residual'] = atl06sr_uaf['residual'] - atl06sr_uaf['lidar_snow_depth']
    atl06_uaf['snow_depth_residual'] = atl06_uaf['residual'] - atl06_uaf['lidar_snow_depth']
    atl08_uaf['snow_depth_residual'] = atl08_uaf['residual'] - atl08_uaf['lidar_snow_depth']
    

#---------------#
# Co-register UAF and ICESat-2
#---------------#
"""
The interpolation included some messy data or filler values, so only data from the 10th-90th percentiles will be included.
"""

# ATL06-SR
upper = atl06sr_uaf['residual'].quantile(0.9)
lower = atl06sr_uaf['residual'].quantile(0.1)
atl06sr_uaf = atl06sr_uaf[(atl06sr_uaf['residual']>=lower) &
                          (atl06sr_uaf['residual']<=upper) &
                          (atl06sr_uaf['lidar_snow_depth']>0)]

# ATL06
upper = atl06_uaf['residual'].quantile(0.9)
lower = atl06_uaf['residual'].quantile(0.1)
atl06_uaf = atl06_uaf[(atl06_uaf['residual']>=lower) &
                      (atl06_uaf['residual']<=upper) &
                      (atl06_uaf['lidar_snow_depth']>0)]

# ATL08
upper = atl08_uaf['residual'].quantile(0.9)
lower = atl08_uaf['residual'].quantile(0.1)
atl08_uaf = atl08_uaf[(atl08_uaf['residual']>=lower) &
                      (atl08_uaf['residual']<=upper) &
                      (atl08_uaf['lidar_snow_depth']>0)]


#---------------#
# Save co-registered data
#---------------#
"""
The data will be saved as a CSV. This might be updated to a geoJSON in the near-future.
"""

# Set up time string
time_str = str(atl06sr.index.year[0]) +\
           str(atl06sr.index.month[0]) +\
           str(atl06sr.index.day[0])

# Save data, dependent on snow-on or snow-off. Also renames 'residual' to 'is2_snow_depth' in snow_on cases

if snow_on:    
    atl06sr_uaf = atl06sr_uaf.rename(columns={'residual': 'is2_snow_depth'})
    atl06_uaf = atl06_uaf.rename(columns={'residual': 'is2_snow_depth'})
    atl08_uaf = atl08_uaf.rename(columns={'residual': 'is2_snow_depth'})
    
    atl06sr_uaf.to_csv(f'{path}/snow-depth-data/{field_id}/atl06sr_snowdepth_rgt{rgt}'
                       f'_{field_id}_{time_str}_{cnf_surface}{atl08_class}{segment_length}'
                       f'{res}.csv'
                      )
    atl06_uaf.to_csv(f'{path}/snow-depth-data/{field_id}/atl06_snowdepth_rgt{rgt}'
                       f'_{field_id}_{time_str}.csv')
    atl08_uaf.to_csv(f'{path}/snow-depth-data/{field_id}/atl08_snowdepth_rgt{rgt}'
                       f'_{field_id}_{time_str}.csv')
elif not snow_on:
    atl06sr_uaf.to_csv(f'{path}/snow-depth-data/{field_id}/atl06sr_snowoff_rgt{rgt}'
                       f'_{field_id}_{time_str}_{cnf_surface}{atl08_class}{segment_length}'
                       f'{res}.csv'
                      )
    atl06_uaf.to_csv(f'{path}/snow-depth-data/{field_id}/atl06_snowoff_rgt{rgt}'
                       f'_{field_id}_{time_str}.csv')
    atl08_uaf.to_csv(f'{path}/snow-depth-data/{field_id}/atl08_snowoff_rgt{rgt}'
                       f'_{field_id}_{time_str}.csv')
