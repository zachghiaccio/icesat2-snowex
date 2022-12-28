# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 16:22:28 2022

@author: zfair

A script to co-register lidar data with ICESat-2, then save the results as 3
CSV files. Each file pertains to one of the ICESat-2 strong beams.

Currently only configured for the NEON lidar with RGT 1356.
"""

import os
import h5py
import pandas as pd
import geopandas as gpd
import numpy as np
import xarray as xr
import seaborn as sns
import ground_data_processing as gdp
from shapely import wkt
from pyproj import Proj, transform
import lidar_processing as lp
import matplotlib.pyplot as plt

# User Input
#---------------#

rgt = '1356'
year = '2022-' # Dash added to prevent random time strings from being included
path = 'C:/Users/zfair/OneDrive - NASA/Documents/Python/icesat2-snow/Data/'

# NEON lidar data (Caribou Creek only)
if rgt == '1356':
    neon = '%sNEON_DEM_471000_480000.tif' %(path)

# Field data and Chris Larsen's lidar data (only one at a time)
if rgt == '1356':
    # Field data
    cpcrw_field = '%s/txts/03112022_UAFP48035_CPCRW.txt' %(path)
    field_snow = pd.read_csv(cpcrw_field, header=1)
    field_bounds = [-147.7, 65.1, -147.3, 65.3]
    
    # Lidar tif files
    cpcrw = '%scaribou_dtm.tif' %(path)
    cpcrw_snow = '%scaribou_snowdepth.tif' %(path)
    
    # Read tiffs into xarrays
    larsen_tif = xr.open_rasterio(cpcrw)
    larsen_snow = xr.open_rasterio(cpcrw_snow)
    
    # Get the snow depth/DTM resolutions to match up
    larsen_tif['x'],larsen_tif['y'] = larsen_tif.x.round(),larsen_tif.y.round()
    larsen_snow['x'],larsen_snow['y'] = larsen_snow.x.round(),larsen_snow.y.round()
    
    # ICESat-2 correction factors
    is2_correction_factor = 9.25
    larsen_tif -= is2_correction_factor
    
    vm = 50
    cm = 'Greys_r'
    site = 'cpcrw'
    
elif rgt == '266':
    # Field data
    cffl_field = '%s/csvs/CRRELA_magnaprobe_OperatorView1_CFFL.csv' %(path)
    field_snow = pd.read_csv(cffl_field, header=1)
    field_bounds = [-147.76, 64.84, -147.66, 64.9]
    
    # Lidar tiff files
    cffl = '%sfarmersloop_dtm.tif' %(path)
    cffl_snow = '%sfarmersloop_snow_depth.tif' %(path)
    
    # SNOTEL site data
    snotel_cffl = '%s/csvs/1302_STAND_YEAR=2022.csv' %(path)
    snotel = pd.read_csv(snotel_cffl, header=1)
    snotel['snow_depth_meters'] = snotel['SNWD.I-1 (in) '] * 0.0254
    snotel['snow_depth_meters'][snotel['snow_depth_meters']<0] = 0
    
    # Read tiffs into xarrays
    larsen_tif = xr.open_rasterio(cffl)
    larsen_snow = xr.open_rasterio(cffl_snow)
    
    # ICESat-2 correction factors
    is2_correction_factor = 9.77
    larsen_tif -= is2_correction_factor
    
    vm = 50
    cm = 'Greys'
    site = 'cffl'
    month = '-07-'
    
elif rgt == '472':
    # Field data
    bcef_field = '%s/txts/03092022_UAFP48068_BCEF.txt' %(path)
    field_snow = pd.read_csv(bcef_field, header=1)
    field_bounds = [-148.34, 64.68, -148.24, 64.76]
    
    # Lidar tiff files
    bonanza = '%sbonanza_dtm.tif' %(path)
    bonanza_snow = '%sbonanza_snowdepth.tif' %(path)
    
    # Read tiffs into xarrays
    larsen_tif = xr.open_rasterio(bonanza)
    larsen_snow = xr.open_rasterio(bonanza_snow)
    
    # ICESat-2 correction factors
    is2_correction_factor = 9.9
    larsen_tif -= is2_correction_factor
    
    vm = 90
    cm = 'Greys'
    site = 'bcef'
    
else:
    raise ValueError('No lidar data available for this RGT.')

#-----------------------------------------------------------------------------#
## Choose the files of interest

# Identify the ATL03SL files
if rgt == '1356':
    atl03sl_file = path + 'is2/sliderule/is2_atl03sl_cpcrw_rgt1356.csv'
elif rgt == '266':
    atl03sl_file = path + 'is2/sliderule/is2_atl03sl_cffl_rgt266.csv'
elif rgt == '472':
    atl03sl_file = path + 'is2/sliderule/is2_atl03sl_bcef_rgt472.csv'
else:
    raise ValueError('ATL03-SlideRule file missing.')

# Identify the ATL06 files
atl06_files = os.listdir(path + 'is2/ATL06/alaska')
atl06_files = [file for file in atl06_files if ('ATL06' in file) and (rgt in file)]
        
# Identify the ATL08 files
atl08_files = os.listdir(path + 'is2/ATL08/alaska')
atl08_files = [file for file in atl08_files if ('ATL08' in file) and (rgt in file)]
#atl08_files = [file for file in atl08_files if 'processed' not in file]

#-----------------------------------------------------------------------------#
### Read ICESat-2 data products into DataFrames

# Initialize projection change for ATL06/08
inp = Proj(init='epsg:4326')
outp = Proj(init='epsg:32606')

# Identify the strong beams
#with h5py.File(path+'is2/ATL06/alaska/'+atl06_files[0]) as f:
#    sc_orient = f['orbit_info/sc_orient'][0]
    
#strong_beams,strong_ids = lp.strong_beam_finder(sc_orient)

## Read the ATL03SL data
atl03sl = pd.read_csv(atl03sl_file)
atl03sl['geometry'] = atl03sl['geometry'].apply(wkt.loads)

# Briefly switch over to a GeoDataFrame to change the projection
tmp_gdf = gpd.GeoDataFrame(atl03sl, geometry='geometry', crs='4326')
tmp_gdf = tmp_gdf.to_crs(epsg=32606)

# Add columns for the new x/y coordinates
atl03sl['x'] = tmp_gdf.geometry.x
atl03sl['y'] = tmp_gdf.geometry.y

# Include only strong beam data. Note that the strong beams are when the column
# "spot" == [1,3,5], independent of sc_orient (NEEDS TESTING).
strong_spots = ['1', '3', '5']
atl03sl['spot'] = atl03sl['spot'].apply(str)
atl03sl = atl03sl[atl03sl['spot'].isin(strong_spots)]

# Change height column name to be consistent with other products
atl03sl = atl03sl.rename(columns={'h_mean': 'height'})

# Filter data to be for year of interest. Interannual  capatibility is a
# work in progress.
atl03sl = atl03sl.loc[atl03sl.time.str.contains(year, case='False')]

# Do the same thing, but for the month. Currently only needed for Creamer's Field
if (site == 'cffl') & (year == '2021-'):
    atl03sl = atl03sl.loc[atl03sl.time.str.contains(month, case='False')]
elif (site == 'cffl') & (year == '2022-'):
    atl03sl = atl03sl.loc[atl03sl.time.str.contains('-01-', case='False')]


## Read the ATL06 data
atl06_files = [path+'is2/ATL06/alaska/'+f for f in atl06_files]

# Concatenate ATL06 data into a continuous DataFrame
atl06 = lp.beam_cycle_concat(atl06_files, 'ATL06')

# Convert coordinates to easting/northing
atl06['x'], atl06['y'] = transform(inp, outp, atl06.lon, atl06.lat)

# Filter filler values out of data
atl06.loc[atl06.height>1e38, 'height'] = np.nan
upper = atl06.height.mean() + 3*atl06.height.std()
atl06 = atl06.loc[atl06.height<upper]

# Restrict analysis to ATL03SL bounds
atl06 = atl06[(atl06.y.values>atl03sl.y.min()) & (atl06.y.values<atl03sl.y.max())]

# Filter data to be for year of interest. Interannual  capatibility is a
# work in progress.
atl06 = atl06.loc[atl06.time.astype(str).str.contains(year, case='False')]


## Read the ATL08 data
atl08_files = [path+'is2/ATL08/alaska/'+f for f in atl08_files]

# Concatenate ATL08 data into a continuous DataFrame
atl08 = lp.beam_cycle_concat(atl08_files, 'ATL08')

# Convert coordinates to easting/northing
atl08['x'], atl08['y'] = transform(inp, outp, atl08.lon, atl08.lat)

# Remove filler/messy data
atl08.loc[atl08.height>1e38, 'height'] = np.nan
upper = atl08.height.mean() + 3*atl08.height.std()
atl08 = atl08.loc[atl08.height<upper]

# Restrict analysis to ATL03SL bounds
atl08 = atl08[(atl08.y.values>atl03sl.y.min()) & (atl08.y.values<atl03sl.y.max())]

# Filter data to be for year of interest. Interannual  capatibility is a
# work in progress.
atl08 = atl08.loc[atl08.time.astype(str).str.contains(year, case='False')]

# Subset CPCRW data to a small region, where the field data is expected to be collected
[7.224e6, 474000, 7.227e6, 477000]
if site == 'cpcrw':
    atl03sl = atl03sl[(atl03sl.x>=474000) & (atl03sl.x<=477000) &
                      (atl03sl.y>=7.225e6) & (atl03sl.y<=7.226e6)]
    atl06 = atl06[(atl06.x>=474000) & (atl06.x<=477000) &
                      (atl06.y>=7.225e6) & (atl06.y<=7.226e6)]
    atl08 = atl08[(atl08.x>=474000) & (atl08.x<=477000) &
                      (atl08.y>=7.225e6) & (atl08.y<=7.226e6)]

#-----------------------------------------------------------------------------#
## Read the NEON lidar data

# if neon:
#     tif = xr.open_rasterio(neon)

strong_ids = np.unique(atl06['gt'])
#     # Rescale NEON data to ICESat-2 product resolutions
#     atl03sl_neon = lp.coregister_is2(tif, [], atl03sl, strong_ids)
#     atl06_neon = lp.coregister_is2(tif, [], atl06, strong_ids)
#     atl08_neon = lp.coregister_is2(tif, [], atl08, strong_ids)

#-----------------------------------------------------------------------------#
## Read Chris Larsen's lidar data (where applicable)

# Subset DTM to size of snow depth maps
larsen_tif = larsen_tif.where((larsen_tif.x>=larsen_snow.x.min()) &
                              (larsen_tif.x<=larsen_snow.x.max()) &
                              (larsen_tif.y>=larsen_snow.y.min()) &
                              (larsen_tif.y<=larsen_snow.y.max()),
                              drop=True)

# Coarsen data to reduce memory usage
if rgt == '1356':
    larsen_tif = larsen_tif.coarsen(x=3, y=3, boundary='trim').mean()
    larsen_snow = larsen_snow.coarsen(x=6, y=6, boundary='trim').mean()
else:
    larsen_tif = larsen_tif.coarsen(x=6, y=6, boundary='trim').mean()
    larsen_snow = larsen_snow.coarsen(x=6, y=6, boundary='trim').mean()

# Co-register with ICESat-2 products
atl03sl_larsen = lp.coregister_is2(larsen_tif, larsen_snow, atl03sl, strong_ids)
atl06_larsen = lp.coregister_is2(larsen_tif, larsen_snow, atl06, strong_ids)
atl08_larsen = lp.coregister_is2(larsen_tif, larsen_snow, atl08, strong_ids)

# Calculate snow depth residuals
atl03sl_larsen['snow_depth_residual'] = atl03sl_larsen.residual - atl03sl_larsen.lidar_snow_depth
atl06_larsen['snow_depth_residual'] = atl06_larsen.residual - atl06_larsen.lidar_snow_depth
atl08_larsen['snow_depth_residual'] = atl08_larsen.residual - atl08_larsen.lidar_snow_depth

# Filter data where negative snow depths exist
atl03sl_larsen = atl03sl_larsen[(atl03sl_larsen['residual']>=0) & 
                                (atl03sl_larsen['lidar_snow_depth']>=0)]
atl06_larsen = atl06_larsen[(atl06_larsen['residual']>=0) & 
                              (atl06_larsen['lidar_snow_depth']>=0)]
atl08_larsen = atl08_larsen[(atl08_larsen['residual']>=0) & 
                              (atl08_larsen['lidar_snow_depth']>=0)]
atl03sl_larsen = atl03sl_larsen[atl03sl_larsen['lidar_snow_depth']>=0]
atl06_larsen = atl06_larsen[atl06_larsen['lidar_snow_depth']>=0]
atl08_larsen = atl08_larsen[atl08_larsen['lidar_snow_depth']>=0]

#-----------------------------------------------------------------------------#
## Filter ground-based data by location, and coregister with the airborne lidar

# Process the raw field data
field_snow_processed = gdp.process_raw_data(field_snow)

# Subset field data to regions of interest, where applicable
field_snow_processed = field_snow_processed[(field_snow_processed.Longitude>=field_bounds[0]) &
                        (field_snow_processed.Longitude<=field_bounds[2]) &
                        (field_snow_processed.Latitude>=field_bounds[1]) &
                        (field_snow_processed.Latitude<=field_bounds[3])]

# Co-register field data with the lidar
field_larsen = lp.coregister_point_data(larsen_tif, larsen_snow, field_snow_processed)

# Calculate snow depth residuals
field_larsen['snow_depth_residuals'] = field_larsen.lidar_snow_depth - field_larsen.ground_snow_depth


#-----------------------------------------------------------------------------#
## Add easting/northing to SNOTEL data
snotel_coords = [-147.733, 64.867]
snx, sny = transform(inp, outp, snotel_coords[0], snotel_coords[1])

#-----------------------------------------------------------------------------#
# Elevation plots (Larsen)

#atl03sl_larsen['lidar_height'] -= 9.08
#atl06_larsen['lidar_height'] -= 9.08
#atl08_larsen['lidar_height'] -= 9.08

# Left beam
fig = plt.figure(figsize=[8,5])
plt.plot(atl03sl_larsen.y[atl03sl_larsen.beam=='5'], atl03sl_larsen.is2_height[atl03sl_larsen.beam=='5'],
         '.', label='ATL03SR')
plt.plot(atl06_larsen.y[atl06_larsen.beam=='5'], atl06_larsen.is2_height[atl06_larsen.beam=='5'],
         '.', label='ATL06')
plt.plot(atl08_larsen.y[atl08_larsen.beam=='5'], atl08_larsen.is2_height[atl08_larsen.beam=='5'],
         '.', label='ATL08')
plt.plot(atl03sl_larsen.y[atl03sl_larsen.beam=='5'], atl03sl_larsen.lidar_height[atl03sl_larsen.beam=='5'],
         '.', label='Larsen')
plt.xlabel('Northing [m]', fontsize=18)
plt.ylabel('Elevation [m]', fontsize=18)
plt.legend()
plt.title('Left strong beam', fontsize=18)
plt.tight_layout()
plt.show()

# Central beam
fig = plt.figure(figsize=[8,5])
plt.plot(atl03sl_larsen.y[atl03sl_larsen.beam=='3'], atl03sl_larsen.lidar_height[atl03sl_larsen.beam=='3'],
         '.', label='UAF', markersize=12)
plt.plot(atl03sl_larsen.y[atl03sl_larsen.beam=='3'], atl03sl_larsen.is2_height[atl03sl_larsen.beam=='3'],
         '.', label='ATL03SR')
plt.plot(atl06_larsen.y[atl06_larsen.beam=='3'], atl06_larsen.is2_height[atl06_larsen.beam=='3'],
         '.', label='ATL06')
plt.plot(atl08_larsen.y[atl08_larsen.beam=='3'], atl08_larsen.is2_height[atl08_larsen.beam=='3'],
         '.', label='ATL08')
plt.xlabel('Northing [m]', fontsize=18)
plt.ylabel('Elevation [m]', fontsize=18)
plt.legend()
plt.title('Central strong beam', fontsize=18)
plt.tight_layout()
plt.show()

# Right beam
fig = plt.figure(figsize=[8,5])
plt.plot(atl03sl_larsen.y[atl03sl_larsen.beam=='1'], atl03sl_larsen.lidar_height[atl03sl_larsen.beam=='1'],
         '.', label='UAF', markersize=12)
plt.plot(atl03sl_larsen.y[atl03sl_larsen.beam=='1'], atl03sl_larsen.is2_height[atl03sl_larsen.beam=='1'],
         '.', label='ATL03SR')
plt.plot(atl06_larsen.y[atl06_larsen.beam=='1'], atl06_larsen.is2_height[atl06_larsen.beam=='1'],
         '.', label='ATL06')
plt.plot(atl08_larsen.y[atl08_larsen.beam=='1'], atl08_larsen.is2_height[atl08_larsen.beam=='1'],
         '.', label='ATL08')
plt.xlabel('Northing [m]', fontsize=18)
plt.ylabel('Elevation [m]', fontsize=18)
plt.legend()
plt.title('Right strong beam', fontsize=18)
plt.tight_layout()
plt.show()

#-----------------------------------------------------------------------------#
## ICESat-2 surface heights (ICESat-2 and UAF)

# Snow depth residual map
larsen_hillshade, larsen_slope = lp.generate_lidar_hillshade(larsen_tif)

fig = plt.figure(figsize=[8,5])
larsen_slope.plot(cmap=cm, vmin=0, vmax=vm)
plt.scatter(atl03sl_larsen.x, atl03sl_larsen.y,
            c=atl03sl_larsen.residual,
            vmin=-1.5, vmax=1.5)
cbar = plt.colorbar()
cbar.set_label('IS2-UAF residual [m]')
plt.xlabel('Easting [m]', fontsize=18)
plt.ylabel('Northing [m]', fontsize=18)
#plt.xlim([4.365e5, 4.4061e5])
#plt.ylim([7.17418e6, 7.18117e6])
plt.xticks([464500, 465500, 466500, 467500, 468500])
plt.title(' ')
plt.tight_layout()
plt.show()
plt.savefig('is2_uaf_%s_elevation_map.png' %(site), dpi=500)

# Along track comparison
fig = plt.figure(figsize=[8,5])
plt.plot(atl03sl_larsen.y[atl03sl_larsen.beam=='3'], atl03sl_larsen[atl03sl_larsen.beam=='3'].lidar_height,
         '#440154', marker='.', linestyle='None', markersize=12,
         alpha=0.3, label='UAF')
plt.plot(atl03sl_larsen[atl03sl_larsen.beam=='3'].y, atl03sl_larsen[atl03sl_larsen.beam=='3'].is2_height,
         '#3b528b', marker='.', linestyle='None', markersize=12,
         alpha=0.3, label='ATL03SR')
plt.plot(atl06_larsen[atl06_larsen.beam=='3'].y, atl06_larsen[atl06_larsen.beam=='3'].is2_height,
         '#21918c', marker='.', linestyle='None', markersize=12,
         alpha=0.3, label='ATL06')
plt.plot(atl08_larsen[atl08_larsen.beam=='3'].y, atl08_larsen[atl08_larsen.beam=='3'].is2_height,
         '#5ec962', marker='.', linestyle='None', markersize=12,
         alpha=0.3, label='ATL08')
plt.xlabel('Northing [m]', fontsize=18)
plt.ylabel('Elevation [m]', fontsize=18)
plt.legend()
plt.tight_layout()
plt.show()
plt.savefig('is2_uaf_%s_elevation_comparison.png' %(site), dpi=500)

# Scatter plot
fig = plt.figure(figsize=[8,5])
plt.scatter(atl03sl_larsen.lidar_height, atl03sl_larsen.is2_height,
            c =atl03sl_larsen.residual,
            vmin=-1.5, vmax=1.5, alpha=0.3)
cbar = plt.colorbar()
cbar.set_label('IS2-UAF residual [m]')
cbar.set_alpha(1)
cbar.draw_all()
#plt.xlim([0, 4])
#plt.ylim([0, 4])
plt.xlabel('UAF lidar height [m]', fontsize=18)
plt.ylabel('ICESat-2 height [m]', fontsize=18)
plt.tight_layout()
plt.show()
plt.savefig('is2_uaf_%s_elevation_scatter.png' %(site), dpi=500)

#-----------------------------------------------------------------------------#
## ICESat-2 snow depth (ICESat-2 and UAF)

# Snow depth residual map
#larsen_hillshade, larsen_slope = lp.generate_lidar_hillshade(larsen_tif)

fig = plt.figure(figsize=[8,5])
larsen_slope.plot(cmap=cm, vmin=0, vmax=vm)
plt.scatter(atl03sl_larsen.x, atl03sl_larsen.y,
            c=atl03sl_larsen.snow_depth_residual,
            vmin=-1.5, vmax=1.5)
cbar = plt.colorbar()
cbar.set_label('IS2-UAF depth residual [m]')
plt.xlabel('Easting [m]', fontsize=18)
plt.ylabel('Northing [m]', fontsize=18)
#plt.xlim([4.365e5, 4.4061e5])
#plt.ylim([7.17418e6, 7.18117e6])
#plt.xticks([436500, 437500, 438500, 439500, 440500])
plt.xticks([464500, 465500, 466500, 467500, 468500])
plt.title(' ')
plt.tight_layout()
plt.show()
plt.savefig('is2_uaf_%s_snowdepth_map.png' %(site), dpi=500)

# Along track comparison
fig = plt.figure(figsize=[8,5])
plt.plot(atl03sl_larsen[atl03sl_larsen.beam=='3'].y, atl03sl_larsen[atl03sl_larsen.beam=='3'].lidar_snow_depth,
         '#440154', marker='.', linestyle='None', markersize=12,
         alpha=0.3, label='UAF')
plt.plot(atl03sl_larsen[atl03sl_larsen.beam=='3'].y, atl03sl_larsen[atl03sl_larsen.beam=='3'].residual,
         '#3b528b', marker='.', linestyle='None', markersize=12,
         alpha=0.3, label='ATL03SR')
plt.plot(atl06_larsen[atl06_larsen.beam=='3'].y, atl06_larsen[atl06_larsen.beam=='3'].residual,
         '#21918c', marker='.', linestyle='None', markersize=12,
         alpha=0.3, label='ATL06')
plt.plot(atl08_larsen[atl08_larsen.beam=='3'].y, atl08_larsen[atl08_larsen.beam=='3'].residual,
         '#5ec962', marker='.', linestyle='None', markersize=12,
         alpha=0.3, label='ATL08')
plt.xlabel('Northing [m]', fontsize=18)
plt.ylabel('Snow depth [m]', fontsize=18)
plt.legend()
plt.tight_layout()
plt.show()
plt.savefig('is2_uaf_%s_snowdepth_comparison.png' %(site), dpi=500)

# Scatter plot
fig = plt.figure(figsize=[8,5])
plt.scatter(atl03sl_larsen.lidar_snow_depth, atl03sl_larsen.residual,
            c =atl03sl_larsen.snow_depth_residual,
            vmin=-1.5, vmax=1.5, alpha=0.3)
cbar = plt.colorbar()
cbar.set_label('IS2-UAF depth residual [m]')
cbar.set_alpha(1)
cbar.draw_all()
plt.xlim([0, 3])
plt.ylim([0, 3])
plt.xlabel('UAF snow depth [m]', fontsize=18)
plt.ylabel('ICESat-2 snow depth [m]', fontsize=18)
plt.tight_layout()
plt.show()
plt.savefig('is2_uaf_%s_snowdepth_scatter.png' %(site), dpi=500)


#-----------------------------------------------------------------------------#
## Ground-based snow depth (Ground data and UAF)

fig = plt.figure(figsize=[8,5])
plt.hexbin(field_larsen.lidar_snow_depth, field_larsen.ground_snow_depth,
           gridsize=50, bins='log', vmin=1, vmax=10)
cbar = plt.colorbar()
cbar.set_label('log10(N)')
plt.xlabel('UAF snow depth [m]', fontsize=18)
plt.ylabel('Probe snow depth [m]', fontsize=18)
plt.tight_layout()
plt.show()
plt.savefig('probes_uaf_%s_snowdepth.png' %(site), dpi=500)

#-----------------------------------------------------------------------------#
## Field site data (currently only SNOTEL over Creamer's Field)
try:
    fig = plt.figure(figsize=[8,5])
    plt.plot(snotel.Date, snotel.snow_depth_meters, label='SNOTEL')
    plt.plot(pd.Series('2022-01-08'), atl03sl_larsen.residual.iloc[72],
             '.', color='#3b528b', markersize=12, label='ICESat-2')
    plt.plot(pd.Series('2022-03-15'), 0.711252, 
                '.', color='#440154', markersize=12, label='UAF')
    plt.legend()
    plt.xticks(['2022-01-15', '2022-02-15', '2022-03-15', '2022-04-01',
                '2022-05-01', '2022-06-01', '2022-07-01', '2022-08-01'])
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Snow depth [m]', fontsize=18)
    plt.xlim('2022-01-01', '2022-03-31')
    plt.ylim(0.5, 0.8)
    plt.tight_layout()
    plt.show()
    plt.savefig('snotel_%s_snowdepth.png' %(site), dpi=500)
except:
    print('No field site data available yet.')
    

#-----------------------------------------------------------------------------#
## Elevation plots (NEON)

# # Left beam
# fig1 = plt.figure(figsize=[8,5])
# plt.plot(atl03sl_neon.y[atl03sl_neon.beam=='5'], atl03sl_neon.is2_height[atl03sl_neon.beam=='5'],
#          '.', label='ATL03SR')
# plt.plot(atl06_neon.y[atl06_neon.beam=='5'], atl06_neon.is2_height[atl06_neon.beam=='5'],
#          '.', label='ATL06')
# plt.plot(atl08_neon.y[atl08_neon.beam=='5'], atl08_neon.is2_height[atl08_neon.beam=='5'],
#          '.', label='ATL08')
# plt.plot(atl03sl_neon.y[atl03sl_neon.beam=='5'], atl03sl_neon.neon_height[atl03sl_neon.beam=='5'],
#          '.', label='NEON')
# plt.xlabel('Northing [m]', fontsize=18)
# plt.ylabel('Elevation [m]', fontsize=18)
# plt.legend()
# plt.title('Left strong beam', fontsize=18)
# plt.tight_layout()
# plt.show()

# # Central beam
# fig2 = plt.figure(figsize=[8,5])
# plt.plot(atl03sl_neon.y[atl03sl_neon.beam=='3'], atl03sl_neon.is2_height[atl03sl_neon.beam=='3'],
#          '.', label='ATL03SR')
# plt.plot(atl06_neon.y[atl06_neon.beam=='3'], atl06_neon.is2_height[atl06_neon.beam=='3'],
#          '.', label='ATL06')
# plt.plot(atl08_neon.y[atl08_neon.beam=='3'], atl08_neon.is2_height[atl08_neon.beam=='3'],
#          '.', label='ATL08')
# plt.plot(atl03sl_neon.y[atl03sl_neon.beam=='3'], atl03sl_neon.neon_height[atl03sl_neon.beam=='3'],
#          '.', label='NEON')
# plt.xlabel('Northing [m]', fontsize=18)
# plt.ylabel('Elevation [m]', fontsize=18)
# plt.legend()
# plt.title('Central strong beam', fontsize=18)
# plt.tight_layout()
# plt.show()

# # Right beam
# fig3 = plt.figure(figsize=[8,5])
# plt.plot(atl03sl_neon.y[atl03sl_neon.beam=='1'], atl03sl_neon.is2_height[atl03sl_neon.beam=='1'],
#          '.', label='ATL03SR')
# plt.plot(atl06_neon.y[atl06_neon.beam=='1'], atl06_neon.is2_height[atl06_neon.beam=='1'],
#          '.', label='ATL06')
# plt.plot(atl08_neon.y[atl08_neon.beam=='1'], atl08_neon.is2_height[atl08_neon.beam=='1'],
#          '.', label='ATL08')
# plt.plot(atl03sl_neon.y[atl03sl_neon.beam=='1'], atl03sl_neon.neon_height[atl03sl_neon.beam=='1'],
#          '.', label='NEON')
# plt.xlabel('Northing [m]', fontsize=18)
# plt.ylabel('Elevation [m]', fontsize=18)
# plt.legend()
# plt.title('Right strong beam', fontsize=18)
# plt.tight_layout()
# plt.show()

# #-----------------------------------------------------------------------------#
# ## Bias/residual plots (NEON)

# # Left bias
# fig4 = plt.figure(figsize=[8,5])
# plt.plot(atl03sl_neon.y[atl03sl_neon.beam=='5'], atl03sl_neon.residual[atl03sl_neon.beam=='5'],
#          '.', label='ATL03SR')
# plt.plot(atl06_neon.y[atl06_neon.beam=='5'], atl06_neon.residual[atl06_neon.beam=='5'],
#          '.', label='ATL06')
# plt.plot(atl08_neon.y[atl08_neon.beam=='5'], atl08_neon.residual[atl08_neon.beam=='5'],
#          '.', label='ATL08')
# plt.xlabel('Northing [m]', fontsize=18)
# plt.ylabel('Elevation residual [m]', fontsize=18)
# plt.legend()
# plt.title('IS2-NEON, Left strong beam', fontsize=18)
# plt.tight_layout()
# plt.show()

# # Central bias
# fig5 = plt.figure(figsize=[8,5])
# plt.plot(atl03sl_neon.y[atl03sl_neon.beam=='3'], atl03sl_neon.residual[atl03sl_neon.beam=='3'],
#          '.', label='ATL03SR')
# plt.plot(atl06_neon.y[atl06_neon.beam=='3'], atl06_neon.residual[atl06_neon.beam=='3'],
#          '.', label='ATL06')
# plt.plot(atl08_neon.y[atl08_neon.beam=='3'], atl08_neon.residual[atl08_neon.beam=='3'],
#          '.', label='ATL08')
# plt.xlabel('Northing [m]', fontsize=18)
# plt.ylabel('IS2-NEON [m]', fontsize=18)
# plt.legend()
# plt.title('IS2-NEON, Central strong beam', fontsize=18)
# plt.tight_layout()
# plt.show()

# # Right bias
# fig6 = plt.figure(figsize=[8,5])
# plt.plot(atl03sl_neon.y[atl03sl_neon.beam=='1'], atl03sl_neon.residual[atl03sl_neon.beam=='1'],
#          '.', label='ATL03SR')
# plt.plot(atl06_neon.y[atl06_neon.beam=='1'], atl06_neon.residual[atl06_neon.beam=='1'],
#          '.', label='ATL06')
# plt.plot(atl08_neon.y[atl08_neon.beam=='1'], atl08_neon.residual[atl08_neon.beam=='1'],
#          '.', label='ATL08')
# plt.xlabel('Northing [m]', fontsize=18)
# plt.ylabel('IS2-NEON [m]', fontsize=18)
# plt.legend()
# plt.title('IS2-NEON, Right strong beam', fontsize=18)
# plt.tight_layout()
# plt.show()

# #-----------------------------------------------------------------------------#
# ## Slope/residual correlations (NEON)

# # ATL03SR
# fig7, ax7 = plt.subplots(1, 3, sharey=True)
# plt.suptitle('ATL03SR')
# sns.regplot(x=atl03sl_neon.slope[atl03sl_neon.beam=='5'], y=atl03sl_neon.residual[atl03sl_neon.beam=='5'], ax=ax7[0])
# ax7[0].set_title('Left strong beam')
# sns.regplot(x=atl03sl_neon.slope[atl03sl_neon.beam=='3'], y=atl03sl_neon.residual[atl03sl_neon.beam=='3'], ax=ax7[1])
# ax7[1].set_title('Central strong beam')
# sns.regplot(x=atl03sl_neon.slope[atl03sl_neon.beam=='1'], y=atl03sl_neon.residual[atl03sl_neon.beam=='1'], ax=ax7[2])
# ax7[2].set_title('Right strong beam')
# for ax in ax7.flat: # Include x-/y-labels for outer axes only
#     ax.set(xlabel='Along-track slope', ylabel='IS2-NEON [m]')
#     ax.label_outer()

# # ATL06
# fig8, ax8 = plt.subplots(1, 3, sharey=True)
# plt.suptitle('ATL06')
# sns.regplot(x=atl06_neon.slope[atl06_neon.beam=='5'], y=atl06_neon.residual[atl06_neon.beam=='5'], ax=ax8[0])
# ax8[0].set_title('Left strong beam')
# sns.regplot(x=atl06_neon.slope[atl06_neon.beam=='3'], y=atl06_neon.residual[atl06_neon.beam=='3'], ax=ax8[1])
# ax8[1].set_title('Central strong beam')
# sns.regplot(x=atl06_neon.slope[atl06_neon.beam=='1'], y=atl06_neon.residual[atl06_neon.beam=='1'], ax=ax8[2])
# ax8[2].set_title('Right strong beam')
# for ax in ax8.flat: # Include x-/y-labels for outer axes only
#     ax.set(xlabel='Along-track slope', ylabel='IS2-NEON [m]')
#     ax.label_outer()
    
# # ATL08
# fig9, ax9 = plt.subplots(1, 3, sharey=True)
# plt.suptitle('ATL08')
# sns.regplot(x=atl08_neon.slope[atl08_neon.beam=='5'], y=atl08_neon.residual[atl08_neon.beam=='5'], ax=ax9[0])
# ax9[0].set_title('Left strong beam')
# sns.regplot(x=atl08_neon.slope[atl08_neon.beam=='3'], y=atl08_neon.residual[atl08_neon.beam=='3'], ax=ax9[1])
# ax9[1].set_title('Central strong beam')
# sns.regplot(x=atl08_neon.slope[atl08_neon.beam=='1'], y=atl08_neon.residual[atl08_neon.beam=='1'], ax=ax9[2])
# ax9[2].set_title('Right strong beam')
# for ax in ax9.flat: # Include x-/y-labels for outer axes only
#     ax.set(xlabel='Along-track slope', ylabel='IS2-NEON [m]')
#     ax.label_outer()