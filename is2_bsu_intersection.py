# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 11:20:00 2022

@author: zfair

A script to co-register lidar data with ICESat-2, then save the results as 3
CSV files. Each file pertains to one of the ICESat-2 strong beams.

Currently only configured for the Boise State lidar over Grand Mesa.
"""

import os
import h5py
import pandas as pd
import geopandas as gpd
import numpy as np
import ulmo
import xarray as xr
from shapely import wkt
from shapely.geometry import Point
import seaborn as sns
from datetime import datetime, timedelta
from ground_data_processing import snotel_fetch, add_dowy
from pyproj import Proj, transform
import lidar_processing as lp
import matplotlib.pyplot as plt


# User Input
#---------------#

# ICESat-2 RGT information
rgt = '737'
year = '2020-' # Dash added to prevent random time strings from being included
month = '02-'  # Desired month to plot, optional
month2 = '02' # Temporary measure because datetime objects are annoying

# SnowEx data
path = 'C:/Users/zfair/OneDrive - NASA/Documents/Python/icesat2-snow/Data/'
bsu1 = '%slas_laz/bsu_fl2c_sep2019.asc' %(path) # Snow-off
bsu2 = '%slas_laz/bsu_fl2c_feb2020.asc' %(path) # Snow-on
bsu_gpr = '%scsvs/SNEX20_BSU_GPR_pE_01282020_01292020_02042020_downsampled.csv' %(path)
unm_gpr = '%scsvs/SNEX20_UNM_GPR.csv' %(path)
csu_gpr = '%scsvs/SNEX20_GM_CSU_GPR_1GHz_v01.csv' %(path)
community_probes = '%scsvs/SnowEx2020_SnowDepths_COGM_alldepths_v01.csv' %(path)
#aso_snow_on = ('%sASO_GrandMesa_mosaic_2020Feb1-2_AllData_and_Reports/'
#               'ASO_GrandMesa_2020Feb1-2_snowdepth_3m.tif' %(path))
#aso_snow_off = ('%sASO_snowfree_GrandMesa_2016Sep26_fsrdemvf_mos_3m/'
#                'ASO_snowfree_GrandMesa_2016Sep26_fsrdemvf_mos_3m.tif' %(path))
aso_snow_on = '%sGrandMesa13Feb_snow_depth_3M.tif' %(path)
aso_snow_off = '%sGrandMesa13Feb_DEMs_mosaic_3m_WGS84.tif' %(path)
cover_file = '%sNLCD_XInQtyxggRVLWdOXTCS1/NLCD_2019_Land_Cover_L48_20210604_XInQtyxggRVLWdOXTCS1.tiff' %(path)

# Flag to split COGM community probe data by method
split_cogm = False

# ICESat-2 file paths
atl03sr_files = os.listdir(path + 'is2/sliderule/')
atl06_files = os.listdir(path + 'is2/ATL06/grand-mesa/')
atl08_files = os.listdir(path + 'is2/ATL08/grand-mesa/')

#-----------------------------------------------------------------------------#
# Load ground-based lidar data (BSU TLS)
bsu_snow_off = xr.open_dataset(bsu1).load()
bsu_snow_on = xr.open_dataset(bsu2).load()

# Round x/y coordinates to nearest meter
bsu_snow_off['x'] = np.floor(bsu_snow_off['x'])
bsu_snow_off['y'] = np.floor(bsu_snow_off['y'])
bsu_snow_on['x'] = np.floor(bsu_snow_on['x'])
bsu_snow_on['y'] = np.floor(bsu_snow_on['y'])

# Subset snow-off DEM to align with snow-on. Will possibly implement code for
# the contrary case if needed.
bsu_snow_off = bsu_snow_off.where((bsu_snow_off.x>bsu_snow_on.x.min()) &
                                  (bsu_snow_off.x<bsu_snow_on.x.max()) &
                                  (bsu_snow_off.y>bsu_snow_on.y.min()) &
                                  (bsu_snow_off.y<bsu_snow_on.y.max()), drop=True)

# Calculate BSU snow depth
bsu_snow_depth = bsu_snow_on - bsu_snow_off

# Plot snow depth raster to make sure it looks good
if False:
    bsu_snow_depth.band_data.plot()
#-----------------------------------------------------------------------------#
# Load ground-based GPR data from Boise State (BSU GPR)
br = pd.read_csv(bsu_gpr, header=0)

#br = br[br['Easting']>744500]
#br = br[br['Northing']<=4322638]

if False:
    plt.scatter(br['Easting'], br['Northing'], c=br['ElevationWGS84'])

#-----------------------------------------------------------------------------#
# Load ground-based GPR data from UNM (UNM GPR)
nmr = pd.read_csv(unm_gpr, header=0)

nmr = nmr.rename(columns={'ELEV_m': 'Elevation',
                          'DEPTH_m': 'Depth',
                          'DATE_dd_mmm_yy': 'UTCdoy',
                          'EASTING': 'Easting',
                          'NORTHING': 'Northing'})

#nmr = nmr[nmr['Easting']<744940]
#nmr = nmr[nmr['Northing']<=4323365]

#-----------------------------------------------------------------------------#
# Load ground-based GPR data from CSU (CSU GPR)
cr = pd.read_csv(csu_gpr, header=0)

cr = cr.rename(columns={'ElevationWGS84 [mae]': 'Elevation',
                        'Depth [cm]': 'Depth',
                        'Date [mmddyy]': 'UTCdoy',
                        'Easting [m]': 'Easting',
                        'Northing [m]': 'Northing'})

#cr = cr[cr['Easting']>744500]
#cr = cr[cr['Northing']<=4322638]

#-----------------------------------------------------------------------------#
# Load ground-based community probe data (COGM)
cogm = pd.read_csv(community_probes, header=0)

#cogm = cogm[cogm['Easting']>744500]
#cogm = cogm[cogm['Northing']<=4322638]

# Rename the columns for better consistency
cogm = cogm.rename(columns={'elevation (m)': 'Elevation',
                            'Depth (cm)': 'Depth',
                            'Date (yyyymmdd)': 'UTCdoy'})

# Split the probe data by instrument, if desired
if split_cogm:
    mp = cogm[cogm['ID']<200000]
    m2 = cogm[(cogm['ID']>=200000) & (cogm['ID']<300000)]
    pr = cogm[cogm['ID']>=300000]

# Only look at a single snow pit location. Temporary solution until all pits
# can be considered at once.
pit = np.unique(cogm.PitID)[2]
#cogm = cogm.loc[cogm.PitID == pit]

# Derive aggregate statistics by pit ID
stat_list = ['count','min','max','mean','std','median','mad']
cogm_stats = cogm.groupby('PitID')['Longitude', 'Latitude',
                             'Easting', 'Northing',
                             'UTCdoy', 'Depth',
                             'Elevation'].agg({'Longitude': 'mean',
                                               'Latitude': 'mean',
                                               'Easting': 'mean',
                                               'Northing': 'mean',
                                               'UTCdoy': 'mean',
                                               'Depth': stat_list,
                                               'Elevation': stat_list})

if False:
    plt.scatter(cogm['Easting'], cogm['Northing'], c=cogm['Elevation'])
    raise ValueError('testing')

#-----------------------------------------------------------------------------#
# Load airborne lidar data (ASO snow depth, DEM in future)
aso_snow_depth = xr.open_rasterio(aso_snow_on)
aso_dsm = xr.open_rasterio(aso_snow_off)

# Plot snow depth raster to make sure it looks good
if False:
    aso_snow_depth.plot(vmin=0, vmax=1.5,
                        cbar_kwargs={'label': 'Snow depth [m]'})
#-----------------------------------------------------------------------------#
## Read land cover data
land_cover = xr.open_rasterio(cover_file)

# Replace NaNs with filler values
land_cover.rio.write_nodata(241, inplace=True)

# Reproject to EPSG:32612
land_cover = land_cover.rio.reproject('EPSG:32612')

#-----------------------------------------------------------------------------#
## Load ICESat-2 data (currently ATL06/08, will expand to ATL03SR)
atl03sr_files = [file for file in atl03sr_files if ('grandmesa' in file) and
                 (rgt in file)]
atl06_files = [file for file in atl06_files if ('ATL06' in file) and 
               (rgt in file) and (year[0:4] in file)]
atl08_files = [file for file in atl08_files if ('ATL08' in file) and 
               (rgt in file) and (year[0:4] in file)]

# Add a check for years with multiple files
# if (rgt=='737') & (year=='2020-') & (snow_on):
#     atl06_files = atl06_files[0]
#     atl08_files = atl08_files[0]
# elif (rgt=='737') & (year=='2020-') & (not snow_on):
#     atl06_files = atl06_files[1]
#     atl00_files = atl06_files[1]

## Read the ATL03SR data
atl03sr_files = [path+'is2/sliderule/'+f for f in atl03sr_files]
atl03sr = pd.read_csv(atl03sr_files[0])
atl03sr['geometry'] = atl03sr['geometry'].apply(wkt.loads)

# Briefly switch over to a GeoDataFrame to change the projection
tmp_gdf = gpd.GeoDataFrame(atl03sr, geometry='geometry', crs='4326')
tmp_gdf = tmp_gdf.to_crs(epsg=32612)

# Add columns for the new x/y coordinates
atl03sr['x'] = tmp_gdf.geometry.x
atl03sr['y'] = tmp_gdf.geometry.y

# Include only strong beam data. Note that the strong beams are when the column
# "spot" == [1,3,5], independent of sc_orient (NEEDS TESTING).
strong_spots = ['1', '3', '5']
atl03sr['spot'] = atl03sr['spot'].apply(str)
atl03sr = atl03sr[atl03sr['spot'].isin(strong_spots)]

# Change height column name to be consistent with other products
atl03sr = atl03sr.rename(columns={'h_mean': 'height'})

# Filter data to be for year of interest. Interannual  capatibility is a
# work in progress.
atl03sr = atl03sr.loc[atl03sr.time.str.contains(year, case='False')]

# Spatial subset for where ground data is clustered
#atl03sr = atl03sr[atl03sr.x>744500]
#atl03sr = atl03sr[atl03sr.y<=4322638]


# Initialize projection change for ATL06/08
inp = Proj(init='epsg:4326')
outp = Proj(init='epsg:32612')

# Identify the strong beams
with h5py.File(path+'is2/ATL08/grand-mesa/'+atl08_files[0]) as f:
    sc_orient = f['orbit_info/sc_orient'][0]
    
strong_beams,strong_ids = lp.strong_beam_finder(sc_orient)


## Read the ATL06 data
atl06_files = [path+'is2/ATL06/grand-mesa/'+f for f in atl06_files]

# Concatenate ATL06 data into a continuous DataFrame
atl06 = lp.beam_cycle_concat(atl06_files, 'ATL06')

# Convert coordinates to easting/northing
atl06['x'], atl06['y'] = transform(inp, outp, atl06.lon, atl06.lat)

# Filter filler values out of data
atl06.loc[atl06.height>1e38, 'height'] = np.nan
upper = atl06.height.mean() + 3*atl06.height.std()
atl06 = atl06.loc[atl06.height<upper]

#atl06 = atl06[atl06.x>744500]
#atl06 = atl06[atl06.y<=4322638]


## Read the ATL08 data
atl08_files = [path+'is2/ATL08/grand-mesa/'+f for f in atl08_files]

# Concatenate ATL08 data into a continuous DataFrame
atl08 = lp.beam_cycle_concat(atl08_files, 'ATL08')

# Convert coordinates to easting/northing
atl08['x'], atl08['y'] = transform(inp, outp, atl08.lon, atl08.lat)

# Remove filler/messy data
atl08.loc[atl08.height>1e38, 'height'] = np.nan
upper = atl08.height.mean() + 3*atl08.height.std()
atl08 = atl08.loc[atl08.height<upper]

# Restrict analysis to ASO bounds
atl08 = atl08[(atl08.y.values>aso_snow_depth.y.values.min()) & 
              (atl08.y.values<aso_snow_depth.y.values.max())]

#atl08 = atl08[atl08.x>744500]
#atl08 = atl08[atl08.y<=4322638]
#-----------------------------------------------------------------------------#
## Co-register the SnowEx data with ICESat-2
# Select a month to plot in 2D, if desired
atl03sr = atl03sr[atl03sr.time.str.contains(month)]
atl06 = atl06[(atl06.time.dt.month == float(month2)) & (atl06.time.dt.year==2020)]
atl08 = atl08[(atl08.time.dt.month==float(month2)) & (atl08.time.dt.year==2020)]

atl03sr_aso = lp.coregister_is2(aso_dsm, aso_snow_depth, atl03sr, strong_ids)
atl06_aso = lp.coregister_is2(aso_dsm, aso_snow_depth, atl06, strong_ids)
atl08_aso = lp.coregister_is2(aso_dsm, aso_snow_depth, atl08, strong_ids)

## Apply filters to ASO data
# Remove filler values in Quantum data
upper = atl03sr_aso['is2_height'].mean() + 2.5*atl03sr_aso['is2_height'].std()
lower = atl03sr_aso['is2_height'].min() - 2.5*atl03sr_aso['is2_height'].std()
atl03sr_aso = atl03sr_aso[(atl03sr_aso['lidar_height']>=lower) &
                          (atl03sr_aso['lidar_height']<=upper)]
atl06_aso = atl06_aso[(atl06_aso['lidar_height']>=lower) &
                          (atl06_aso['lidar_height']<=upper)]
atl08_aso = atl08_aso[(atl08_aso['lidar_height']>=lower) &
                          (atl08_aso['lidar_height']<=upper)]

# Apply vertical datum corrections
atl03sr_aso['residual'] += 27.24
atl06_aso['residual'] += 27.24
atl08_aso['residual'] += 27.24
atl03sr_aso['lidar_height'] -= 27.24
atl06_aso['lidar_height'] -= 27.24
atl08_aso['lidar_height'] -= 27.24
#atl03sr_aso['lidar_height'] -= 17.29
#atl06_aso['lidar_height'] -= 17.29
#atl08_aso['lidar_height'] -= 17.29

# Remove filler values in snow depth data
# if month2 == '02':
#     atl03sr_aso = atl03sr_aso[(atl03sr_aso['lidar_snow_depth']>=0) &
#                               (atl03sr_aso['residual']>=0)]
#     atl06_aso = atl06_aso[(atl06_aso['lidar_snow_depth']>=0) &
#                           (atl06_aso['residual']>=0)]
#     atl08_aso = atl08_aso[(atl08_aso['lidar_snow_depth']>=0) &
#                           (atl08_aso['residual']>=0)]
# else:
#     atl03sr_aso = atl03sr_aso[atl03sr_aso['lidar_snow_depth']>=0]
#     atl06_aso = atl06_aso[atl06_aso['lidar_snow_depth']>=0]
#     atl08_aso = atl08_aso[atl08_aso['lidar_snow_depth']>=0]
    
atl03sr_aso = atl03sr_aso[atl03sr_aso['lidar_snow_depth']>=0]
atl06_aso = atl06_aso[atl06_aso['lidar_snow_depth']>=0]
atl08_aso = atl08_aso[atl08_aso['lidar_snow_depth']>=0]

# Calculate snow depth residuals
atl03sr_aso['snow_depth_residual'] = atl03sr_aso.residual - atl03sr_aso.lidar_snow_depth
atl06_aso['snow_depth_residual'] = atl06_aso.residual - atl06_aso.lidar_snow_depth
atl08_aso['snow_depth_residual'] = atl08_aso.residual - atl08_aso.lidar_snow_depth

# Update CSU GPR coordinate system to epsg:32612
cr['Easting'], cr['Northing'] = transform(inp, outp, cr['Longitude [DD]'], cr['Latitude [DD]'])
#cr = cr[cr['Easting']>744330]
#cr = cr[cr['Northing']<=4322800]

# Co-register ground-based data with ICESat-2
bsu_aso = lp.coregister_point_data(aso_dsm, aso_snow_depth, br)
unm_aso = lp.coregister_point_data(aso_dsm, aso_snow_depth, nmr)
csu_aso = lp.coregister_point_data(aso_dsm, aso_snow_depth, cr)

# # Co-register community probe data, together or separately
if split_cogm:
    mp_aso = lp.coregister_point_data(aso_dsm, aso_snow_depth, mp)
    m2_aso = lp.coregister_point_data(aso_dsm, aso_snow_depth, m2)
    pr_aso = lp.coregister_point_data(aso_dsm, aso_snow_depth, pr)
    
    mp_aso['snow_depth_residual'] = mp_aso['lidar_snow_depth'] - mp_aso['ground_snow_depth']/100
    m2_aso['snow_depth_residual'] = m2_aso['lidar_snow_depth'] - m2_aso['ground_snow_depth']/100
    pr_aso['snow_depth_residual'] = pr_aso['lidar_snow_depth'] - pr_aso['ground_snow_depth']/100
else:
    cogm_aso = lp.coregister_point_data(aso_dsm, aso_snow_depth, cogm)
    cogm_aso['snow_depth_residual'] = cogm_aso['lidar_snow_depth'] - cogm_aso['ground_snow_depth']/100

# # Calculate snow depth residuals
bsu_aso['snow_depth_residual'] = bsu_aso['lidar_snow_depth'] - bsu_aso['ground_snow_depth']/100
csu_aso['snow_depth_residual'] = csu_aso['lidar_snow_depth'] - csu_aso['ground_snow_depth']/100
unm_aso['snow_depth_residual'] = unm_aso['lidar_snow_depth'] - unm_aso['ground_snow_depth']

# # Apply (very basic) correction factors to ground data elevations
# bsu_aso['residual'] += 7.83
# bsu_aso['lidar_height'] -= 7.83
# unm_aso['residual'] += 5.14
# unm_aso['lidar_height'] -= 5.14
# mp_aso['residual'] += 9.85
# mp_aso['lidar_height'] -= 9.85
# m2_aso['residual'] += 6.59
# m2_aso['lidar_height'] -= 6.59
# pr_aso['residual'] += 10.86
# pr_aso['lidar_height'] -= 10.86
# csu_aso['residual'][csu_aso.doy==20620] += 27.89
# csu_aso['lidar_height'][csu_aso.doy==20620] -= 27.89
# csu_aso['residual'][csu_aso.doy==20720] += 27.97
# csu_aso['lidar_height'][csu_aso.doy==20720] -= 27.97
# csu_aso['residual'][csu_aso.doy==20820] += 6.85
# csu_aso['lidar_height'][csu_aso.doy==20820] -= 6.85
# csu_aso['residual'][csu_aso.doy==20920] += 25.99
# csu_aso['lidar_height'][csu_aso.doy==20920] -= 25.99


# Plot the flight tracks to make sure they align with ASO
if False:
    plt.plot(atl06.x[atl06['gt']=='5'], atl06.y[atl06['gt']=='5'],
             '.', label='ATL08')
    plt.plot(atl06.x[atl06['gt']=='3'], atl06.y[atl06['gt']=='3'], '.')
    plt.plot(atl06.x[atl06['gt']=='1'], atl06.y[atl06['gt']=='1'], '.')
    plt.xlabel('easting [m]', fontsize=18)
    plt.ylabel('northing [m]', fontsize=18)
    plt.title('RGT %s' %(rgt), fontsize=18)
    plt.tight_layout()
    plt.show()
    
if False:
    if cogm_aso['lidar_height'].max() < 20:
        fig = plt.figure()
        m, b = np.polyfit(cogm_aso['ground_snow_depth']/100,
                          cogm_aso['lidar_height']-9.9, 1)
        plt.plot(cogm_aso['ground_snow_depth']/100,
                 cogm_aso['lidar_height']-9.9, 'o')
        plt.plot(cogm_aso['ground_snow_depth']/100,
                 m*cogm_aso['ground_snow_depth']/100+b, 'k')
        plt.xlabel('Probe snow depth [m]')
        plt.ylabel('ASO snow depth [m]')
        plt.title('PitID = {0}'.format(pit))
        plt.tight_layout()
        plt.show()
        
        fig = plt.figure()
        m, b = np.polyfit(bsu_aso['ground_snow_depth']/100,
                          bsu_aso['lidar_height']-9.9, 1)
        plt.plot(bsu_aso['ground_snow_depth']/100,
                 bsu_aso['lidar_height']-9.9, 'o')
        plt.plot(bsu_aso['ground_snow_depth']/100,
                 m*bsu_aso['ground_snow_depth']/100+b, 'k')
        plt.xlabel('BSU GPR snow depth [m]')
        plt.ylabel('ASO snow depth [m]')
        plt.tight_layout()
        plt.show()
        
        fig = plt.figure()
        m, b = np.polyfit(unm_aso['ground_snow_depth']/100,
                          unm_aso['lidar_height']-9.9, 1)
        plt.plot(unm_aso['ground_snow_depth']/100,
                 unm_aso['lidar_height']-9.9, 'o')
        plt.plot(unm_aso['ground_snow_depth']/100,
                 m*unm_aso['ground_snow_depth']/100+b, 'k')
        plt.xlabel('UNM GPR snow depth [m]')
        plt.ylabel('ASO snow depth [m]')
        plt.tight_layout()
        plt.show()
        
        fig = plt.figure()
        m, b = np.polyfit(csu_aso['ground_snow_depth']/100,
                          csu_aso['lidar_height']-9.95, 1)
        plt.plot(csu_aso['ground_snow_depth']/100,
                 csu_aso['lidar_height']-9.95, 'o')
        plt.plot(csu_aso['ground_snow_depth']/100,
                 m*csu_aso['ground_snow_depth']+b, 'k')
        plt.xlabel('CSU GPR snow depth [m]')
        plt.ylabel('ASO snow depth [m]')
        plt.tight_layout()
        plt.show()
    else:
        fig = plt.figure()
        m, b = np.polyfit(cogm_aso['ground_height'],
                          cogm_aso['lidar_height']+9.9, 1)
        plt.plot(cogm_aso['ground_height'],
                 cogm_aso['lidar_height']+9.9, 'o')
        plt.plot(cogm_aso['ground_height'],
                 m*cogm_aso['ground_height']+b, 'k')
        plt.xlabel('Probe elevation [m]')
        plt.ylabel('Quantum lidar elevation [m]')
        plt.title('PitID = {0}'.format(pit))
        plt.tight_layout()
        plt.show()
        
        fig = plt.figure()
        m, b = np.polyfit(bsu_aso['ground_height'],
                          bsu_aso['lidar_height']+9.9, 1)
        plt.plot(bsu_aso['ground_height'],
                 bsu_aso['lidar_height']+9.9, 'o')
        plt.plot(bsu_aso['ground_height'],
                 m*bsu_aso['ground_height']+b, 'k')
        plt.xlabel('BSU GPR elevation [m]')
        plt.ylabel('Quantum lidar elevation [m]')
        plt.tight_layout()
        plt.show()
        
        fig = plt.figure()
        m, b = np.polyfit(unm_aso['ground_height'],
                          unm_aso['lidar_height']+9.9, 1)
        plt.plot(unm_aso['ground_height'],
                 unm_aso['lidar_height']+9.9, 'o')
        plt.plot(unm_aso['ground_height'],
                 m*unm_aso['ground_height']+b, 'k')
        plt.xlabel('UNM GPR elevation [m]')
        plt.ylabel('Quantum lidar elevation [m]')
        plt.tight_layout()
        plt.show()
        
        fig = plt.figure()
        m, b = np.polyfit(csu_aso['ground_height'],
                          csu_aso['lidar_height']+9.95, 1)
        plt.plot(csu_aso['ground_height'],
                 csu_aso['lidar_height']+9.95, 'o')
        plt.plot(csu_aso['ground_height'],
                 m*csu_aso['ground_height']+b, 'k')
        plt.xlabel('CSU GPR elevation [m]')
        plt.ylabel('Quantum lidar elevation [m]')
        plt.tight_layout()
        plt.show()
    #raise ValueError('testing')
#-----------------------------------------------------------------------------#
## Co-register land cover data with ICESat-2


#-----------------------------------------------------------------------------#
## Read SNOTEL data from ulmo
wsdlurl = 'https://hydroportal.cuahsi.org/Snotel/cuahsi_1_1.asmx?WSDL'
sites = ulmo.cuahsi.wof.get_sites(wsdlurl)
sites_df = pd.DataFrame.from_dict(sites, orient='index').dropna()

# Change lat/lons and elevation data into more useful formats
sites_df['geometry'] = [Point(float(loc['longitude']), float(loc['latitude'])) for loc in sites_df['location']]
sites_df = sites_df.drop(columns='location')
sites_df = sites_df.astype({"elevation_m":float})

# Convert to geopandas dataframe. The coordinate reference system is there for
# completeness, and shouldn't be needed
sites_gdf_all = gpd.GeoDataFrame(sites_df, crs='EPSG:4326')

# Look at only the site at Grand Mesa (technically Mesa Lake)
mesa_lake_gdf_conus = sites_gdf_all[(sites_gdf_all.index.str.contains('622'))]

# Fetch SNOTEL data from the selected site
start_date = datetime(1950,1,1)
end_date = datetime.today()
values_df = snotel_fetch(mesa_lake_gdf_conus.index[-1], 'SNOTEL:SNWD_D',
                         start_date, end_date)
add_dowy(values_df)

# Aggregate SNOTEL data into statistics
stat_list = ['count','min','max','mean','std','median']
doy_stats = values_df.groupby('doy').agg(stat_list)['value']

#-----------------------------------------------------------------------------#
## Elevation plots

# # Left beam
fig1 = plt.figure(figsize=[8,5])
plt.plot(atl03sr_aso.y[atl03sr_aso.beam=='5'], atl03sr_aso.is2_height[atl03sr_aso.beam=='5'],
          '.', label='ATL03SR')
plt.plot(atl06_aso.y[atl06_aso.beam=='5'], atl06_aso.is2_height[atl06_aso.beam=='5'],
          '.', label='ATL06')
plt.plot(atl08_aso.y[atl08_aso.beam=='5'], atl08_aso.is2_height[atl08_aso.beam=='5'],
          '.', label='ATL08')
plt.plot(atl03sr_aso.y[atl03sr_aso.beam=='5'], atl03sr_aso.lidar_height[atl03sr_aso.beam=='5']-9.95,
          '.', label='ASO')
plt.xlabel('Northing [m]', fontsize=18)
plt.ylabel('Elevation [m]', fontsize=18)
plt.legend()
plt.title('Left strong beam', fontsize=18)
plt.tight_layout()
plt.show()

# Central beam
fig2 = plt.figure(figsize=[8,5])
plt.plot(atl03sr_aso.y[atl03sr_aso.beam=='3'], atl03sr_aso.lidar_height[atl03sr_aso.beam=='3']-9.95,
          '.', label='Quantum lidar', markersize=12)
plt.plot(atl03sr_aso.y[atl03sr_aso.beam=='3'], atl03sr_aso.is2_height[atl03sr_aso.beam=='3'],
          '.', label='ATL03SR')
plt.plot(atl06_aso.y[atl06_aso.beam=='3'], atl06_aso.is2_height[atl06_aso.beam=='3'],
          '.', label='ATL06')
plt.plot(atl08_aso.y[atl08_aso.beam=='3'], atl08_aso.is2_height[atl08_aso.beam=='3'],
          '.', label='ATL08')
plt.xlabel('Northing [m]', fontsize=18)
plt.ylabel('Elevation [m]', fontsize=18)
plt.legend(fontsize=12)
plt.title('Central strong beam', fontsize=18)
plt.tight_layout()
plt.show()

# Right beam
fig3 = plt.figure(figsize=[8,5])
plt.plot(atl03sr_aso.y[atl03sr_aso.beam=='1'], atl03sr_aso.is2_height[atl03sr_aso.beam=='1'],
          '.', label='ATL03SR')
plt.plot(atl06_aso.y[atl06_aso.beam=='1'], atl06_aso.is2_height[atl06_aso.beam=='1'],
          '.', label='ATL06')
plt.plot(atl08_aso.y[atl08_aso.beam=='1'], atl08_aso.is2_height[atl08_aso.beam=='1'],
          '.', label='ATL08')
plt.plot(atl03sr_aso.y[atl03sr_aso.beam=='1'], atl03sr_aso.lidar_height[atl03sr_aso.beam=='1']-9.95,
          '.', label='ASO')
plt.xlabel('Northing [m]', fontsize=18)
plt.ylabel('Elevation [m]', fontsize=18)
plt.legend()
plt.title('Right strong beam', fontsize=18)
plt.tight_layout()
plt.show()
# #-----------------------------------------------------------------------------#
# ## Bias/residual plots, relative to ASO

# # Left beam
# fig4 = plt.figure(figsize=[8,5])
# plt.plot(atl03sr_aso.y[atl03sr_aso.beam=='5'], atl03sr_aso.residual[atl03sr_aso.beam=='5'],
#          '.', label='ATL03SR')
# plt.plot(atl06_aso.y[atl06_aso.beam=='5'], atl06_aso.residual[atl06_aso.beam=='5'],
#          '.', label='ATL06')
# plt.plot(atl08_aso.y[atl08_aso.beam=='5'], atl08_aso.residual[atl08_aso.beam=='5'],
#          '.', label='ATL08')
# plt.xlabel('Northing [m]', fontsize=18)
# plt.ylabel('Elevation residual [m]', fontsize=18)
# plt.legend()
# plt.title('IS2-ASO, Left strong beam', fontsize=18)
# plt.tight_layout()
# plt.show()

fig5 = plt.figure(figsize=[8,5])
plt.plot(atl03sr_aso.y[atl03sr_aso.beam=='3'], atl03sr_aso.residual[atl03sr_aso.beam=='3'],
          '.', label='ATL03SR')
plt.plot(atl06_aso.y[atl06_aso.beam=='3'], atl06_aso.residual[atl06_aso.beam=='3'],
          '.', label='ATL06')
plt.plot(atl08_aso.y[atl08_aso.beam=='3'], atl08_aso.residual[atl08_aso.beam=='3'],
          '.', label='ATL08')
plt.xlabel('Northing [m]', fontsize=18)
plt.ylabel('Elevation residual [m]', fontsize=18)
plt.legend()
plt.title('IS2-ASO, Central strong beam', fontsize=18)
plt.tight_layout()
plt.show()

# fig6 = plt.figure(figsize=[8,5])
# plt.plot(atl03sr_aso.y[atl03sr_aso.beam=='1'], atl03sr_aso.residual[atl03sr_aso.beam=='1'],
#          '.', label='ATL03SR')
# plt.plot(atl06_aso.y[atl06_aso.beam=='1'], atl06_aso.residual[atl06_aso.beam=='1'],
#          '.', label='ATL06')
# plt.plot(atl08_aso.y[atl08_aso.beam=='1'], atl08_aso.residual[atl08_aso.beam=='1'],
#          '.', label='ATL08')
# plt.xlabel('Northing [m]', fontsize=18)
# plt.ylabel('Elevation residual [m]', fontsize=18)
# plt.legend()
# plt.title('IS2-ASO, Left strong beam', fontsize=18)
# plt.tight_layout()
# plt.show()

# #-----------------------------------------------------------------------------#
# ## Snow depth plots (currently ASO only)

# # Left depths
# fig7 = plt.figure(figsize=[8,5])
# plt.plot(atl03sr_aso.y[atl03sr_aso.beam=='5'],
#          atl03sr_aso.lidar_snow_depth[atl03sr_aso.beam=='5'],
#          '.', label='ASO')
# plt.plot(atl03sr_aso.y[atl03sr_aso.beam=='5'],
#          atl03sr_aso.residual[atl03sr_aso.beam=='5'],
#          '.', label='ATL03-SR')
# plt.plot(atl06_aso.y[atl06_aso.beam=='5'],
#          atl06_aso.residual[atl06_aso.beam=='5'],
#          '.', label='ATL06')
# plt.plot(atl08_aso.y[atl08_aso.beam=='5'],
#          atl08_aso.residual[atl08_aso.beam=='5'],
#          '.', label='ATL08')
# plt.xlabel('Northing [m]', fontsize=18)
# plt.ylabel('Snow depth [m]', fontsize=18)
# plt.legend()
# plt.title('Left strong beam', fontsize=18)
# plt.tight_layout()
# plt.show()

# # Central depths
fig8 = plt.figure(figsize=[8,5])
plt.plot(atl03sr_aso.y[atl03sr_aso.beam=='3'],
          atl03sr_aso.lidar_snow_depth[atl03sr_aso.beam=='3'],
          '.', label='ASO', markersize=12)
plt.plot(atl03sr_aso.y[atl03sr_aso.beam=='3'],
          atl03sr_aso.residual[atl03sr_aso.beam=='3']-0.332,
          '.', label='ATL03SR')
plt.plot(atl06_aso.y[atl06_aso.beam=='3'],
          atl06_aso.residual[atl06_aso.beam=='3']-0.332,
          '.', label='ATL06')
plt.plot(atl08_aso.y[atl08_aso.beam=='3'],
          atl08_aso.residual[atl08_aso.beam=='3']-0.332,
          '.', label='ATL08')
plt.xlabel('Northing [m]', fontsize=18)
plt.ylabel('Snow depth [m]', fontsize=18)
plt.legend()
#plt.title('Central strong beam', fontsize=18)
plt.tight_layout()
plt.show()

# # Right depths
# fig9 = plt.figure(figsize=[8,5])
# plt.plot(atl03sr_aso.y[atl03sr_aso.beam=='1'],
#          atl03sr_aso.lidar_snow_depth[atl03sr_aso.beam=='1'],
#          '.', label='ASO')
# plt.plot(atl03sr_aso.y[atl03sr_aso.beam=='1'],
#          atl03sr_aso.residual[atl03sr_aso.beam=='1'],
#          '.', label='ATL03-SR')
# plt.plot(atl06_aso.y[atl06_aso.beam=='1'],
#          atl06_aso.residual[atl06_aso.beam=='1'],
#          '.', label='ATL06')
# plt.plot(atl08_aso.y[atl08_aso.beam=='1'],
#          atl08_aso.residual[atl08_aso.beam=='1'],
#          '.', label='ATL08')
# plt.xlabel('Northing [m]', fontsize=18)
# plt.ylabel('Snow depth [m]', fontsize=18)
# plt.legend()
# plt.title('Right strong beam', fontsize=18)
# plt.tight_layout()
# plt.show()

# #-----------------------------------------------------------------------------#
# ## Surface height correlations

# # ICESat-2, ATL03-SlideRule
# fig10, ax10 = plt.subplots(3, 3)
# sns.regplot(x=atl03sr_aso.is2_height[atl03sr_aso.beam=='5'],
#             y=atl03sr_aso.lidar_height[atl03sr_aso.beam=='5']-9.95, ax=ax10[0,0])
# ax10[0,0].set_xlabel(' ')
# ax10[0,0].set_ylabel('Quantum lidar elevation [m]')
# sns.regplot(x=atl03sr_aso.is2_height[atl03sr_aso.beam=='3'],
#             y=atl03sr_aso.lidar_height[atl03sr_aso.beam=='3']-9.95, ax=ax10[0,1])
# ax10[0,1].set_xlabel(' ')
# ax10[0,1].set_ylabel(' ')
# sns.regplot(x=atl03sr_aso.is2_height[atl03sr_aso.beam=='1'],
#             y=atl03sr_aso.lidar_height[atl03sr_aso.beam=='1']-9.95, ax=ax10[0,2])
# ax10[0,2].set_xlabel(' ')
# ax10[0,2].set_ylabel(' ')
    
# # ICESat-2, ATL06
# sns.regplot(x=atl06_aso.is2_height[atl06_aso.beam=='5'],
#             y=atl06_aso.lidar_height[atl06_aso.beam=='5']-9.95, ax=ax10[1,0])
# ax10[1,0].set_xlabel(' ')
# ax10[1,0].set_ylabel('Quantum lidar elevation [m]')
# sns.regplot(x=atl06_aso.is2_height[atl06_aso.beam=='3'],
#             y=atl06_aso.lidar_height[atl06_aso.beam=='3']-9.95, ax=ax10[1,1])

# ax10[1,1].set_xlabel(' ')
# ax10[1,1].set_ylabel(' ')
# sns.regplot(x=atl06_aso.is2_height[atl06_aso.beam=='1'],
#             y=atl06_aso.lidar_height[atl06_aso.beam=='1']-9.95, ax=ax10[1,2])
# ax10[1,2].set_xlabel(' ')
# ax10[1,2].set_ylabel(' ')
    
# # ICESat-2, ATL08
# sns.regplot(x=atl08_aso.is2_height[atl08_aso.beam=='5'],
#             y=atl08_aso.lidar_height[atl08_aso.beam=='5']-9.95, ax=ax10[2,0])
# ax10[2,0].set_xlabel('ICESat-2 elevation [m]')
# ax10[2,0].set_ylabel('Quantum lidar elevation [m]')
# sns.regplot(x=atl08_aso.is2_height[atl08_aso.beam=='3'],
#             y=atl08_aso.lidar_height[atl08_aso.beam=='3']-9.95, ax=ax10[2,1])
# ax10[2,1].set_xlabel('ICESat-2 elevation [m]')
# ax10[2,1].set_ylabel(' ')
# sns.regplot(x=atl08_aso.is2_height[atl08_aso.beam=='1'],
#             y=atl08_aso.lidar_height[atl08_aso.beam=='1']-9.95, ax=ax10[2,2])
# ax10[2,2].set_xlabel('ICESat-2 elevation [m]')
# ax10[2,2].set_ylabel(' ')

# cols = ['Left strong', 'Central strong', 'Right strong']
# for ax, col in zip(ax10[0], cols):
#     ax.set_title(col)
# for ax in ax10.flat: # Include x-/y-labels for outer axes only
#     ax.set(xlim=[2950, 3150], ylim=[2950, 3150])
    
# #-----------------------------------------------------------------------------#
# ## Snow depth correlations (ICESat-2 and ASO)

# # ICESat-2, ATL03-SlideRule
# #atl03sr_aso = atl03sr_aso[(atl03sr_aso.residual>0) & (atl03sr_aso.lidar_snow_depth.abs()<2.5)]
# fig10, ax10 = plt.subplots(3, 3)
# sns.regplot(x=atl03sr_aso.residual[atl03sr_aso.beam=='5'],
#             y=atl03sr_aso.lidar_snow_depth[atl03sr_aso.beam=='5'], 
#             ax=ax10[0,0])
# ax10[0,0].set_xlabel(' ')
# ax10[0,0].set_ylabel('ASO snow depth [m]')
# sns.regplot(x=atl03sr_aso.residual[atl03sr_aso.beam=='3'],
#             y=atl03sr_aso.lidar_snow_depth[atl03sr_aso.beam=='3'], 
#             ax=ax10[0,1])
# ax10[0,1].set_xlabel(' ')
# ax10[0,1].set_ylabel(' ')
# sns.regplot(x=atl03sr_aso.residual[atl03sr_aso.beam=='1'],
#             y=atl03sr_aso.lidar_snow_depth[atl03sr_aso.beam=='1'], 
#             ax=ax10[0,2])
# ax10[0,2].set_xlabel(' ')
# ax10[0,2].set_ylabel(' ')
    
# # ICESat-2, ATL06
# #atl06_aso = atl06_aso[(atl06_aso.residual>0) & (atl06_aso.lidar_snow_depth.abs()<2.5)]
# sns.regplot(x=atl06_aso.residual[atl06_aso.beam=='5'],
#             y=atl06_aso.lidar_snow_depth[atl06_aso.beam=='5'], 
#             ax=ax10[1,0])
# ax10[1,0].set_xlabel(' ')
# ax10[1,0].set_ylabel('ASO snow depth [m]')
# sns.regplot(x=atl06_aso.residual[atl06_aso.beam=='3'],
#             y=atl06_aso.lidar_snow_depth[atl06_aso.beam=='3'], 
#             ax=ax10[1,1])
# ax10[1,1].set_xlabel(' ')
# ax10[1,1].set_ylabel(' ')
# sns.regplot(x=atl06_aso.residual[atl06_aso.beam=='1'],
#             y=atl06_aso.lidar_snow_depth[atl06_aso.beam=='1'], 
#             ax=ax10[1,2])
# ax10[1,2].set_xlabel(' ')
# ax10[1,2].set_ylabel(' ')
    
# # ICESat-2, ATL08
# #atl08_aso = atl08_aso[(atl08_aso.residual>0) & (atl08_aso.lidar_snow_depth.abs()<2.5)]
# sns.regplot(x=atl08_aso.residual[atl08_aso.beam=='5'],
#             y=atl08_aso.lidar_snow_depth[atl08_aso.beam=='5'], 
#             ax=ax10[2,0])
# ax10[2,0].set_xlabel('ICESat-2 snow depth [m]')
# ax10[2,0].set_ylabel('ASO snow_depth [m]')
# sns.regplot(x=atl08_aso.residual[atl08_aso.beam=='3'],
#             y=atl08_aso.lidar_snow_depth[atl08_aso.beam=='3'], 
#             ax=ax10[2,1])
# ax10[2,1].set_xlabel('ICESat-2 snow depth [m]')
# ax10[2,1].set_ylabel(' ')
# sns.regplot(x=atl08_aso.residual[atl08_aso.beam=='1'],
#             y=atl08_aso.lidar_snow_depth[atl08_aso.beam=='1'], ax=ax10[2,2])
# ax10[2,2].set_xlabel('ICESat-2 snow depth [m]')
# ax10[2,2].set_ylabel(' ')

#-----------------------------------------------------------------------------#
## ICESat-2 elevation comparisons

# Make filtered estimates for each residual
upper = atl03sr_aso.residual.quantile(0.9)
lower = atl03sr_aso.residual.quantile(0.1)
atl03_residual = atl03sr_aso[(atl03sr_aso.residual>=lower) &
                             (atl03sr_aso.residual<=upper)]

upper = atl06_aso.residual.quantile(0.9)
lower = atl06_aso.residual.quantile(0.1)
atl06_residual = atl06_aso[(atl06_aso.residual>=lower) &
                           (atl06_aso.residual<=upper)]

upper = atl08_aso.residual.quantile(0.9)
lower = atl08_aso.residual.quantile(0.1)
atl08_residual = atl08_aso[(atl08_aso.residual>=lower) &
                           (atl08_aso.residual<=upper)]


# # Make shaded relief map from Quantum lidar data
aso_hillshade, aso_slope = lp.generate_lidar_hillshade(aso_dsm)

# ICESat-2/Quantum map comparison
fig = plt.figure(figsize=[8,5])
aso_slope.plot(cmap='Greys', vmin=0, vmax=50)
plt.scatter(atl03sr_aso.x, atl03sr_aso.y,
            c=atl03sr_aso.residual,
            vmin=-1.5, vmax=1.5)
cbar = plt.colorbar()
cbar.set_label('IS2-Quantum residual [m]')
plt.xlabel('Easting [m]', fontsize=18)
plt.ylabel('Northing [m]', fontsize=18)
plt.xlim([735000, 760000])
plt.ylim([4.32e6, 4.33e6])
plt.title(' ')
plt.tight_layout()
plt.show()
plt.savefig('is2_quantum_grand_mesa_elevation_map.png', dpi=500)

# ICESat-2/Quantum along track comparison
fig = plt.figure(figsize=[8,5])
#plt.plot(atl03sr_aso.y[atl03sr_aso.beam=='3'], atl03sr_aso[atl03sr_aso.beam=='3'].lidar_height,
#         '#440154', marker='.', linestyle='None', markersize=12,
#         alpha=0.3, label='Quantum lidar')
plt.plot(atl03_residual[atl03_residual.beam=='3'].y, atl03_residual[atl03_residual.beam=='3'].residual,
         '#3b528b', marker='.', linestyle='None', markersize=12,
         alpha=0.5, label='ATL03SR')
plt.plot(atl06_residual[atl06_residual.beam=='3'].y, atl06_residual[atl06_residual.beam=='3'].residual,
         '#21918c', marker='.', linestyle='None', markersize=16,
         alpha=0.5, label='ATL06')
plt.plot(atl08_residual[atl08_residual.beam=='3'].y, atl08_residual[atl08_residual.beam=='3'].residual,
         '#5ec962', marker='.', linestyle='None', markersize=20,
         alpha=0.5, label='ATL08')
plt.xlabel('Northing [m]', fontsize=18)
plt.ylabel('Elevation residual [m]', fontsize=18)
plt.legend()
plt.tight_layout()
plt.show()
plt.savefig('is2_quantum_grand_mesa_elevation_comparison.png', dpi=500)

# Scatter plot
fig = plt.figure(figsize=[8,5])
plt.scatter(atl03sr_aso.lidar_height-9.95, atl03sr_aso.is2_height,
            c =atl03sr_aso.residual,
            vmin=-1.5, vmax=1.5, alpha=0.3)
cbar = plt.colorbar()
cbar.set_label('IS2-Quantum residual [m]')
cbar.set_alpha(1)
cbar.draw_all()
plt.xlabel('Quantum lidar height [m]', fontsize=18)
plt.ylabel('ICESat-2 height [m]', fontsize=18)
plt.tight_layout()
plt.show()
plt.savefig('is2_quantum_grand_mesa_elevation_snowon_scatter.png', dpi=500)

#-----------------------------------------------------------------------------#
## ICESat-2 snow depth comparisons

# Make filtered estimates for each residual
upper = atl03sr_aso.snow_depth_residual.quantile(0.9)
lower = atl03sr_aso.snow_depth_residual.quantile(0.1)
atl03_residual = atl03sr_aso[(atl03sr_aso.snow_depth_residual>=lower) &
                             (atl03sr_aso.snow_depth_residual<=upper)]

upper = atl06_aso.snow_depth_residual.quantile(0.9)
lower = atl06_aso.snow_depth_residual.quantile(0.1)
atl06_residual = atl06_aso[(atl06_aso.snow_depth_residual>=lower) &
                           (atl06_aso.snow_depth_residual<=upper)]

upper = atl08_aso.snow_depth_residual.quantile(0.9)
lower = atl08_aso.snow_depth_residual.quantile(0.1)
atl08_residual = atl08_aso[(atl08_aso.snow_depth_residual>=lower) &
                           (atl08_aso.snow_depth_residual<=upper)]

# ICESat-2 snow depth residual map
fig = plt.figure(figsize=[8,5])
aso_slope.plot(cmap='Greys', vmin=0, vmax=50)
plt.scatter(atl03sr_aso.x, atl03sr_aso.y,
            c=atl03sr_aso.snow_depth_residual,
            vmin=-1.5, vmax=1.5)
cbar = plt.colorbar()
cbar.set_label('IS2-ASO depth residual [m]')
plt.xlabel('Easting [m]', fontsize=18)
plt.ylabel('Northing [m]', fontsize=18)
plt.xlim([735000, 760000])
plt.ylim([4.32e6, 4.33e6])
plt.title(' ')
plt.tight_layout()
plt.show()
plt.savefig('is2_aso_grand_mesa_snowdepth_map.png', dpi=500)

# Along track comparison
fig = plt.figure(figsize=[8,5])
#plt.plot(atl03sr_aso[atl03sr_aso.beam=='3'].y, atl03sr_aso[atl03sr_aso.beam=='3'].lidar_snow_depth,
#         '#440154', marker='.', linestyle='None', markersize=12,
#         alpha=0.3, label='ASO')
plt.plot(atl03_residual[atl03_residual.beam=='3'].y, atl03_residual[atl03_residual.beam=='3'].snow_depth_residual,
         '#3b528b', marker='.', linestyle='None', markersize=12,
         alpha=0.6, label='ATL03SR')
plt.plot(atl06_residual[atl06_residual.beam=='3'].y, atl06_residual[atl06_residual.beam=='3'].snow_depth_residual,
         '#21918c', marker='.', linestyle='None', markersize=16,
         alpha=0.6, label='ATL06')
plt.plot(atl08_residual[atl08_residual.beam=='3'].y, atl08_residual[atl08_residual.beam=='3'].snow_depth_residual,
         '#5ec962', marker='.', linestyle='None', markersize=20,
         alpha=0.6, label='ATL08')
plt.xlabel('Northing [m]', fontsize=18)
plt.ylabel('Snow depth residual [m]', fontsize=18)
#plt.ylim([0, 4])
plt.legend()
plt.tight_layout()
plt.show()
plt.savefig('is2_aso_grand_mesa_snowdepth_comparison.png', dpi=500)

# Scatter plot
fig = plt.figure(figsize=[8,5])
plt.scatter(atl06_aso.lidar_snow_depth, atl06_aso.residual,
            c =atl06_aso.snow_depth_residual,
            vmin=-1.5, vmax=1.5, alpha=0.3)
cbar = plt.colorbar()
cbar.set_label('IS2-ASO depth residual [m]')
cbar.set_alpha(1)
cbar.draw_all()
plt.xlabel('ASO snow depth [m]', fontsize=18)
plt.ylabel('ICESat-2 snow depth [m]', fontsize=18)
plt.tight_layout()
plt.xlim([0, 4])
plt.ylim([0, 4])
plt.show()
plt.savefig('is2_aso_grand_mesa_snowdepth_scatter.png', dpi=500)

#-----------------------------------------------------------------------------#
## Ground-based snow depth hexbins

if False:
    fig,ax = plt.subplots(2, 2)
    ax[0,0].hexbin(cogm_aso.lidar_snow_depth, cogm_aso.ground_snow_depth/100, bins='log',
                 vmin=1, vmax=100, cmap='BuGn')
    ax[0,0].set_ylabel('Community probe snow depth [m]', fontsize=18)
    ax[0,0].set_xlim([0,2])
    ax[0,0].set_ylim([0,2])
    
    ax[0,1].hexbin(csu_aso.lidar_snow_depth, csu_aso.ground_snow_depth/100, bins='log',
                 vmin=1, vmax=100, cmap='BuGn')
    ax[0,1].set_ylabel('CSU GPR snow depth [m]', fontsize=18)
    ax[0,1].set_xlim([0,2])
    ax[0,1].set_ylim([0,2])
    
    ax[1,0].hexbin(bsu_aso.lidar_snow_depth, bsu_aso.ground_snow_depth/100, bins='log',
                 vmin=1, vmax=100, cmap='BuGn')
    ax[1,0].set_ylabel('BSU GPR snow depth [m]', fontsize=18)
    ax[1,0].set_xlim([0,2])
    ax[1,0].set_ylim([0,2])
    
    hb = ax[1,1].hexbin(unm_aso.lidar_snow_depth, unm_aso.ground_snow_depth, bins='log',
                 vmin=1, vmax=100, cmap='BuGn')
    ax[1,1].set_ylabel('UNM GPR snow depth [m]', fontsize=18)
    ax[1,1].set_xlim([0,2])
    ax[1,1].set_ylim([0,2])
    
    ax[1,0].set_xlabel('ASO snow depth [m]', fontsize=18)
    ax[1,1].set_xlabel('ASO snow depth [m]', fontsize=18)
    cbar = fig.colorbar(hb, ax=ax)
    cbar.set_label('log10(N)')
    plt.show()
    plt.savefig('ground_data_grand_mesa_snowdepth.png', dpi=500)
    

# COGM
fig = plt.figure(figsize=[8,5])
plt.hexbin(cogm_aso.lidar_snow_depth, cogm_aso.ground_snow_depth/100, bins='log',
           vmin=1, vmax=100, cmap='BuGn')
cbar = plt.colorbar()
cbar.set_label('log10(N)')
plt.xlabel('ASO snow depth [m]', fontsize=18)
plt.ylabel('Community probe snow depth [m]', fontsize=18)
plt.xlim([0.15, 2])
plt.ylim([0.15, 2])
plt.tight_layout()
plt.show()
plt.savefig('cogm_aso_grand_mesa_snowdepth.png', dpi=500)

# CSU GPR
fig = plt.figure(figsize=[8,5])
plt.hexbin(csu_aso.lidar_snow_depth, csu_aso.ground_snow_depth/100, bins='log',
           vmin=1, vmax=100, cmap='BuGn')
cbar = plt.colorbar()
cbar.set_label('log10(N)')
plt.xlabel('ASO snow depth [m]', fontsize=18)
plt.ylabel('CSU GPR snow depth [m]', fontsize=18)
plt.xlim([0.1, 1.5])
plt.ylim([0.1, 1.5])
plt.tight_layout()
plt.show()
plt.savefig('csu_aso_grand_mesa_snowdepth.png', dpi=500)

# BSU GPR
fig = plt.figure(figsize=[8,5])
plt.hexbin(bsu_aso.lidar_snow_depth, bsu_aso.ground_snow_depth/100, bins='log',
           vmin=1, vmax=100, cmap='BuGn')
cbar = plt.colorbar()
cbar.set_label('log10(N)')
plt.xlabel('ASO snow depth [m]', fontsize=18)
plt.ylabel('BSU GPR snow depth [m]', fontsize=18)
plt.xlim([0.4, 1.75])
plt.ylim([0.4, 1.75])
plt.tight_layout()
plt.show()
plt.savefig('bsu_aso_grand_mesa_snowdepth.png', dpi=500)

# UNM GPR
fig = plt.figure(figsize=[8,5])
plt.hexbin(unm_aso.lidar_snow_depth, unm_aso.ground_snow_depth, bins='log',
           vmin=1, vmax=1000, cmap='BuGn')
cbar = plt.colorbar()
cbar.set_label('log10(N)')
plt.xlabel('ASO snow depth [m]', fontsize=18)
plt.ylabel('UNM GPR snow depth [m]', fontsize=18)
plt.xlim([0, 2])
plt.ylim([0, 2])
plt.tight_layout()
plt.show()
plt.savefig('unm_aso_grand_mesa_snowdepth.png', dpi=500)

#-----------------------------------------------------------------------------#
## SNOTEL time series
curr_y = datetime.strptime('2018-01-01', '%Y-%m-%d').year
df_wy = values_df.loc[f'{curr_y}-01-01':f'{curr_y}-12-31'].dropna()

f, ax = plt.subplots(figsize=[8,5])
ax.plot(doy_stats.index, doy_stats['mean']*0.0254, label='SNOTEL Mean')
ax.fill_between(doy_stats.index, (doy_stats['mean']-doy_stats['std'])*0.0254,
                (doy_stats['mean']+doy_stats['std'])*0.0254, 
                color='lightgrey', label='1-std')
ax.plot(df_wy['doy'], df_wy['value']*0.0254, marker='.', color='k', ls='none',
        label='SNOTEL 2020')
ax.errorbar(43, atl03sr_aso['residual'].mean(), yerr=atl03sr_aso['residual'].std(),
            fmt='o', color='#3b528b', ecolor='#3b528b', elinewidth=3, 
            capsize=5, label='ICESat-2')
ax.errorbar(44, atl03sr_aso['lidar_snow_depth'].mean(), 
            yerr=atl03sr_aso['lidar_snow_depth'].std(),
            fmt='o', color='#440154', ecolor='#440154', elinewidth=3, 
            capsize=5, label='ASO')
ax.legend()
ax.set_xlabel('Day of Year, after Jan 1', fontsize=18)
ax.set_ylabel('Snow depth [m]', fontsize=18)
ax.set_xlim(0, 366)
ax.set_ylim(bottom=0)
f.tight_layout()
f.show()

#-----------------------------------------------------------------------------#
# # CSU GPR
# fig = plt.subplots(figsize=(12,8))
# aso_slope.plot(cmap='Greys', vmax=50, add_colorbar=False)
# plt.scatter(csu_aso.x, csu_aso.y, c=csu_aso.residual,
#             vmin=csu_aso.residual.quantile(0.1),
#             vmax=csu_aso.residual.quantile(0.95),
#             s=2, marker='s', cmap='Spectral_r')
# plt.xlabel('Easting [m]')
# plt.ylabel('Northing [m]')
# plt.title('CSU GPR')
# cbar = plt.colorbar()
# cbar.set_label('ASO - CSU [m]')
# plt.tight_layout()
# plt.show()

# # BSU GPR
# fig = plt.subplots(figsize=(12,8))
# aso_slope.plot(cmap='Greys', vmax=50, add_colorbar=False)
# plt.scatter(bsu_aso.x, bsu_aso.y, c=bsu_aso.residual,
#             vmin=bsu_aso.residual.quantile(0.1),
#             vmax=bsu_aso.residual.quantile(0.95),
#             s=2, marker='s', cmap='Spectral_r')
# plt.xlabel('Easting [m]')
# plt.ylabel('Northing [m]')
# plt.title('BSU GPR')
# cbar = plt.colorbar()
# cbar.set_label('ASO - BSU [m]')
# plt.tight_layout()
# plt.show()

# # UNM GPR
# fig = plt.subplots(figsize=(12,8))
# aso_slope.plot(cmap='Greys', vmax=50, add_colorbar=False)
# plt.scatter(unm_aso.x, unm_aso.y, c=unm_aso.residual,
#             vmin=unm_aso.residual.quantile(0.1),
#             vmax=unm_aso.residual.quantile(0.95),
#             s=2, marker='s', cmap='Spectral_r')
# plt.xlabel('Easting [m]')
# plt.ylabel('Northing [m]')
# plt.title('UNM GPR')
# cbar = plt.colorbar()
# cbar.set_label('ASO - UNM [m]')
# plt.tight_layout()
# plt.show()

# # COGM Probes
# fig = plt.subplots(figsize=(12,8))
# aso_slope.plot(cmap='Greys', vmax=50, add_colorbar=False)
# if split_cogm:
#     plt.scatter(pr_aso.x, pr_aso.y, c=pr_aso.residual,
#                 vmin=pr_aso.residual.quantile(0.1),
#                 vmax=pr_aso.residual.quantile(0.95),
#                 s=2, marker='s', cmap='Spectral_r')
# else:
#     plt.scatter(cogm_aso.x, cogm_aso.y, c=cogm_aso.residual,
#                 vmin=cogm_aso.residual.quantile(0.1),
#                 vmax=cogm_aso.residual.quantile(0.95),
#                 s=2, marker='s', cmap='Spectral_r')
# plt.xlabel('Easting [m]')
# plt.ylabel('Northing [m]')
# plt.title('COGM Probes')
# cbar = plt.colorbar()
# cbar.set_label('ASO - Probes [m]')
# plt.tight_layout()
# plt.show()

# Map Plot of all data sets
# fig,ax = plt.subplots(figsize=(12,8))
# aso_slope.plot(cmap='Greys', vmax=50, add_colorbar=False)
# plt.plot(unm_aso.x[::10], unm_aso.y[::10], '.', alpha=0.5, label='UNM GPR')
# plt.plot(csu_aso.x, csu_aso.y, '.', alpha=0.5, label='CSU GPR')
# plt.plot(bsu_aso.x, bsu_aso.y, '.', alpha=0.5, label='BSU GPR')
# plt.plot(mp_aso.x, mp_aso.y, '.', alpha=0.5, label='Magnaprobes')
# plt.plot(atl08_aso.x, atl08_aso.y, '.', alpha=0.5, label='ICESat-2')
# plt.title('SnowEx 2020: Grand Mesa', fontsize=18)
# plt.xlabel('Easting [m]', fontsize=18)
# ax.tick_params(axis='both', which='major', labelsize=12)
# plt.ylabel('Northing [m]', fontsize=18)
# plt.legend(fontsize=12, markerscale=2)
# plt.show()