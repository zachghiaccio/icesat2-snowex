# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 13:50:22 2022

@author: zfair
"""

import h5py
import datetime
import contextily as cx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.interpolate import RectBivariateSpline
from xrspatial import hillshade, slope

#---------------#
def strong_beam_finder(sc_orient):
    """
    Identifies the strong beam ground tracks, based on space craft orientation.

    Parameters
    ----------
    sc_orient : int
        Identifier for spacecraft orientation.
            If sc_orient = 0 (BACKWARD), then the strong beams are the LEFT beams.
            If sc_orient = 1 (FORWARD), then the strong beams are the RIGHT beams.

    Returns
    -------
    strong_beams: list
        List of strings for ATL06/08 identifying the strong beam ground tracks.
    strongs_ids: list
        Same as strong_beams, but for ATL03SL

    """
    
    if sc_orient == 0:
        strong_beams = ['gt1l', 'gt2l', 'gt3l']
        strong_spots = ['1', '3', '5']
    elif sc_orient == 1:
        strong_beams = ['gt1r', 'gt2r', 'gt3r']
        strong_spots = ['5', '3', '1']
    else:
        raise ValueError('Invalid orientation - spacecraft may have been in' +
                         'transition phase')
    
    return strong_beams, strong_spots

#---------------#
def beam_cycle_concat(is2_files, product_id):
    """
    Concatenates ICESat-2 data into 3 DataFrames - one for each product.
    Individual beams (GT1/GT2/GT3) are concatenated sequentially before the next
    repeat track is added.
    
    Note that ATL03SL is already set up properly, so this function is only for
    ATL06 and ATL08.

    Parameters
    ----------
    is2_files : list
        List of ATL06/08 files that are stored as strings.
    product_id: str
        Specifies whether the data is ATL06 or ATL08.
    strong_beams: list
        String identifiers for the strong beams, as found in strong_beam_finder()

    Returns
    -------
    concat_pd: DataFrame
        A concatenated DataFrame containing all repeat tracks and strong beams.

    """
    
    concat_pd = pd.DataFrame()
    for file in is2_files:
        with h5py.File(file, 'r') as f:
            # Grab only the strong beam data
            sc_orient = f['orbit_info/sc_orient'][0]
            strong_beams,strong_ids = strong_beam_finder(sc_orient)
            for idx,beam in enumerate(strong_beams):
                if product_id == 'ATL06':
                    try:
                        tmp = pd.DataFrame(data={'lat': f[beam+'/land_ice_segments/latitude'][:],
                                                 'lon': f[beam+'/land_ice_segments/longitude'][:],
                                                 'height': f[beam+'/land_ice_segments/h_li'][:],
                                                 'delta_time': f[beam+'/land_ice_segments/delta_time'][:],
                                                 'gt': strong_ids[idx]})
                    except:
                        print('Beam %s missing in data. Skipping concatenation...' %(beam))
                        tmp = pd.DataFrame()
                elif product_id == 'ATL08':
                    try:
                        tmp = pd.DataFrame(data={'lat': f[beam+'/land_segments/latitude'][:],
                                                 'lon': f[beam+'/land_segments/longitude'][:],
                                                 'height': f[beam+'/land_segments/terrain/h_te_best_fit'][:],
                                                 'delta_time': f[beam+'/land_segments/delta_time'][:],
                                                 'gt': strong_ids[idx]})
                    except:
                        print('Beam %s missing in data. Skipping concatenation...' %(beam))
                        tmp = pd.DataFrame()
                    
                concat_pd = pd.concat([concat_pd, tmp])
            
            
    
    # Generate the time column
    atlas_epoch = np.datetime64(datetime.datetime(2018,1,1))
    delta_time = (concat_pd['delta_time']*1e9).astype('timedelta64[ns]')
    concat_pd['time'] = gpd.pd.to_datetime(atlas_epoch + delta_time)
    
    # Set the datetime as the index
    #concat_pd.set_index('time', inplace=True)
    #concat_pd.sort_index(inplace=True)
    
    return concat_pd

#---------------#
def polyf(seri):
    """
    Generates a polynomial to estimate along-track slope. Slope is based on NEON heights.
    """
    
    return np.polyfit(seri.index.values, seri.values, 1)[0]

#---------------#
def coregister_is2(lidar_height, lidar_snow_depth, is2_pd, strong_ids):
    """
    Co-registers NEON data with ICESat-2 data with a rectangular bivariate
    spline. The data is also corrected for geoid differences.
    This function likely be updated when other lidar or DEMs are considered.

    Parameters
    ----------
    lidar_height : Xarray
        Lidar DEM/DSM in Xarray format.
    lidar_snow_depth : Xarray
        Lidar-derived snow depth in Xarray format.
    is2_pd : DataFrame
        DataFrame for the ICESat-2 product of interest.
    strong_ids: list
        List of strings identifying the strong beams. Needed to co-register
        NEON with each beam.

    Returns
    -------
    is2_neon_pd : DataFrame
        Contains the coordinate and elevation data that matches best with
        ICESat-2.

    """
    
    # Correction factor to reproject lidar data to WGS84. Currently only
    # have a value for NEON.
    # A GEOID CORRECTION FOR ASO IS NEEDED ASAP
    geoid_correction = 9.95
    
    # Surface elevation coordinates
    x0 = np.array(lidar_height.x)
    y0 = np.array(lidar_height.y)
    
    # Apply spline to NEON data
    dem_heights = np.array(lidar_height.sel(band=1))[::-1,:]
    dem_heights[np.isnan(dem_heights)] = -9999
    interpolator = RectBivariateSpline(np.array(y0)[::-1], 
                                       np.array(x0),
                                       dem_heights)
    
    snow_depths = np.array(lidar_snow_depth.sel(band=1))[::-1,:]
    snow_depths[np.isnan(snow_depths)] = -9999
    interpolator2 = RectBivariateSpline(np.array(y0)[::-1],
                                       np.array(x0),
                                       snow_depths)
    
    # Use the constructed spline to align NEON with ICESat-2. This is done for
    # all three strong beams.
    is2_neon_pd = pd.DataFrame()
    for spot in strong_ids:
        if not 'spot' in is2_pd.columns:
            is2_tmp = is2_pd.loc[is2_pd['gt']==spot]
            
            xn = is2_tmp['x'].values
            yn = is2_tmp['y'].values
            
            #Define indices within x/y bounds of DEM
            i1 = (xn>np.min(x0)) & (xn<np.max(x0))
            i1 &= (yn>np.min(y0)) & (yn<np.max(y0))
            
            # Set x/y coordinates, NEON heights, and corresponding IS-2 heights
            x, y = xn[i1], yn[i1]
            lidar_h = interpolator(yn[i1], xn[i1], grid=False)
            lidar_d = interpolator2(yn[i1], xn[i1], grid=False)
            is2_height = is2_tmp['height'][i1]
            time = is2_tmp['time'][i1]
            beam = is2_tmp['gt'][i1]
        else:
            is2_tmp = is2_pd.loc[is2_pd['spot']==spot]
            
            xn = is2_tmp['x'].values
            yn = is2_tmp['y'].values
            
            #Define indices within x/y bounds
            i1 = (xn>np.min(x0)) & (xn<np.max(x0))
            i1 &= (yn>np.min(y0)) & (yn<np.max(y0))
            
            # Set x/y coordinates, NEON heights, and corresponding IS-2 heights
            x, y = xn[i1], yn[i1]
            lidar_h = interpolator(yn[i1], xn[i1], grid=False)
            lidar_d = interpolator2(yn[i1], xn[i1], grid=False)
            is2_height = is2_tmp['height'][i1]
            time = is2_tmp['time'][i1]
            beam = is2_tmp['spot'][i1]
        
    
    # Construct co-registered dataframe (NEEDS TO INCLUDE ALL BEAMS AND TIMES)
        tmp = pd.DataFrame(data={'time':time,
                                 'x': x,
                                 'y': y,
                                 'lidar_height': lidar_h,
                                 'lidar_snow_depth': lidar_d,
                                 'is2_height': is2_height,
                                 'beam': beam})
        
        is2_neon_pd = pd.concat([is2_neon_pd, tmp])
        
    # Apply correction factor to NEON data
    is2_neon_pd['lidar_height'] += geoid_correction
    is2_neon_pd['residual'] = is2_neon_pd['is2_height'] - is2_neon_pd['lidar_height']
    is2_neon_pd['slope'] = is2_neon_pd['is2_height'].rolling(10, min_periods=2).apply(polyf, raw=False)
    
    return is2_neon_pd

#---------------#
def coregister_point_data(lidar_tif, lidar_snow_depth, ground_pd):
    """
    Same as coregister_neon (which needs a more generic name), but matches
    a lidar DEM with ground-based data instead. Currently configured for BSU GPR
    and ASO over Grand Mesa.

    Parameters
    ----------
    lidar_tif : Xarray
        The lidar in Xarray format. Can be surface elevation or snow depth.
    ground_pd : DataFrame
        DataFrame for ground-based data of interest. Should include easting/northing
        coordinates, surface elevation, and/or snow depth.

    Returns
    -------
    ground_lidar_pd : DataFrame
        Co-registered lidar and ground-based data, using spline interpolation.

    """
    
    # Correction factor to reproject lidar data to WGS84. Currently only
    # have a value for NEON.
    # A GEOID CORRECTION FOR ASO IS NEEDED ASAP
    geoid_correction = 9.95
    
    x0 = np.array(lidar_tif.x)
    y0 = np.array(lidar_tif.y)
    
    # Apply spline to NEON data
    dem_heights = np.array(lidar_tif.sel(band=1))[::-1,:]
    dem_heights[np.isnan(dem_heights)] = -9999
    interpolator = RectBivariateSpline(np.array(y0)[::-1], 
                                       np.array(x0),
                                       dem_heights)
    
    snow_depths = np.array(lidar_snow_depth.sel(band=1))[::-1,:]
    snow_depths[np.isnan(snow_depths)] = -9999
    interpolator2 = RectBivariateSpline(np.array(y0)[::-1],
                                       np.array(x0),
                                       snow_depths)
    
    
    # Use the constructed spline to align NEON with ICESat-2. This is done for
    # all three strong beams.
    ground_lidar_pd = pd.DataFrame()
    if 'PitID' in ground_pd.columns:
        for ID in np.unique(ground_pd['PitID']):
            tmp_pd = ground_pd.loc[ground_pd.PitID==ID]
            
            xn = tmp_pd['Easting'].values
            yn = tmp_pd['Northing'].values
            
            i1 = (xn>np.min(x0)) & (xn<np.max(x0))
            i1 &= (yn>np.min(y0)) & (yn<np.max(y0))
            
            x, y = xn[i1], yn[i1]
            lidar_height = interpolator(yn[i1], xn[i1], grid=False)
            lidar_snow_depth = interpolator2(yn[i1], xn[i1], grid=False)
            ground_height = tmp_pd['Elevation'][i1]
            doy = tmp_pd['UTCdoy'][i1]
            ground_snow_depth = tmp_pd['Depth'][i1]
            
            # Construct co-registered dataframe (NEEDS TO INCLUDE ALL BEAMS AND TIMES)
            tmp = pd.DataFrame(data={'doy':doy,
                                     'x': x,
                                     'y': y,
                                     'lidar_height': lidar_height,
                                     'lidar_snow_depth': lidar_snow_depth,
                                     'ground_height': ground_height,
                                     'ground_snow_depth': ground_snow_depth})
            
            ground_lidar_pd = pd.concat([ground_lidar_pd, tmp])
    else:
        xn = ground_pd['Easting'].values
        yn = ground_pd['Northing'].values
        
        #Define indices within x/y bounds
        i1 = (xn>np.min(x0)) & (xn<np.max(x0))
        i1 &= (yn>np.min(y0)) & (yn<np.max(y0))
        
        # Set x/y coordinates, NEON heights, and corresponding IS-2 heights
        x, y = xn[i1], yn[i1]
        lidar_height = interpolator(yn[i1], xn[i1], grid=False)
        lidar_snow_depth = interpolator2(yn[i1], xn[i1], grid=False)
        ground_height = ground_pd['Elevation'][i1]
        doy = ground_pd['UTCdoy'][i1]
        ground_snow_depth = ground_pd['Depth'][i1]

        
        # Construct co-registered dataframe (NEEDS TO INCLUDE ALL BEAMS AND TIMES)
        tmp = pd.DataFrame(data={'doy':doy,
                                 'x': x,
                                 'y': y,
                                 'lidar_height': lidar_height,
                                 'lidar_snow_depth': lidar_snow_depth,
                                 'ground_height': ground_height,
                                 'ground_snow_depth': ground_snow_depth})
        
        ground_lidar_pd = pd.concat([ground_lidar_pd, tmp])
    
        
    # Apply correction factor to NEON data
    ground_lidar_pd['lidar_height'] += geoid_correction
    ground_lidar_pd['residual'] = ground_lidar_pd['ground_height'] - ground_lidar_pd['lidar_height']
    ground_lidar_pd['slope'] = ground_lidar_pd['ground_height'].rolling(10, min_periods=2).apply(polyf, raw=False)
    
    return ground_lidar_pd

#---------------#
def coregister_land_cover_data(is2_pd, land_cover_map, strong_ids):
    
    # Surface elevation coordinates
    x0 = np.array(land_cover_map.x)
    y0 = np.array(land_cover_map.y)
    
    # Apply spline to NEON data
    lc = np.array(land_cover_map.sel(band=1))
    interpolator = RectBivariateSpline(np.array(y0)[::-1], 
                                       np.array(x0),
                                       lc)
    
    # Use the constructed spline to align NEON with ICESat-2. This is done for
    # all three strong beams.
    lc_pd = pd.DataFrame()
    for spot in strong_ids:
        if not 'spot' in is2_pd.columns:
            is2_tmp = is2_pd.loc[is2_pd['gt']==spot]
            
            xn = is2_tmp['x'].values
            yn = is2_tmp['y'].values
            
            #Define indices within x/y bounds of DEM
            i1 = (xn>np.min(x0)) & (xn<np.max(x0))
            i1 &= (yn>np.min(y0)) & (yn<np.max(y0))
            
            # Set x/y coordinates, NEON heights, and corresponding IS-2 heights
            x, y = xn[i1], yn[i1]
            land_cover = interpolator(yn[i1], xn[i1], grid=False)
        else:
            is2_tmp = is2_pd.loc[is2_pd['spot']==spot]
            
            xn = is2_tmp['x'].values
            yn = is2_tmp['y'].values
            
            #Define indices within x/y bounds
            i1 = (xn>np.min(x0)) & (xn<np.max(x0))
            i1 &= (yn>np.min(y0)) & (yn<np.max(y0))
            
            # Set x/y coordinates, NEON heights, and corresponding IS-2 heights
            x, y = xn[i1], yn[i1]
            land_cover = interpolator(yn[i1], xn[i1], grid=False)
            
            # Construct co-registered dataframe (NEEDS TO INCLUDE ALL BEAMS AND TIMES)
            tmp = pd.DataFrame(data={'x': x,
                                     'y': y,
                                     'land_cover': land_cover})
                
            lc_pd = pd.concat([lc_pd, tmp])
            
    return lc_pd


#---------------#
def make_snow_map(ground_lidar_pd, bbox, epsg_code):
    """
    Uses contextily to make a map of snow depth or data differences from the
    ground data.

    Parameters
    ----------
    ground_lidar_pd : DataFrame
        Co-registered ground/ASO/Quantum data.
    bbox : list
        The SW/NE coordinates, in [SW_lon, SW_lat, NE_lon, NE_lat] format.
    epsg_code : float/int
        EPSG number for desired horizontal projection.

    Returns
    -------
    snow_map : axis object
        Mapped ground data.

    """
    
    # Set bounding box
    west,south,east,north = (bbox[0],
                             bbox[1],
                             bbox[2],
                             bbox[3])
    
    # Construct contextily map
    img, shp = cx.bounds2img(west,
                             south,
                             east,
                             north,
                             ll=True,
                             source=cx.providers.USGS.USImageryTopo)
    
    # Convert to desired projection
    img, shp = cx.warp_tiles(img, shp, 'epsg:{}'.format(epsg_code))
    
    snow_map = plt.imshow(img, extent=shp)
    
    return snow_map

#---------------#
def generate_lidar_hillshade(lidar_tif):
    """
    Takes a lidar DEM/DSM/DTM and generates a shaded relief map. Applicable
    to any lidar data saved as a tif and loaded into Xarray.

    Parameters
    ----------
    lidar_tif : DataArray
        The lidar elevation data, given as a DEM/DTM/DSM. Horizontal
        and vertical datum transformations should not be necessary. 

    Returns
    -------
    lidar_hillshade : DataArray
        The shaded relief map, using the hillshade technique.

    """
    
    lidar_hillshade = hillshade(lidar_tif.isel(band=0))
    lidar_slope = slope(lidar_tif.isel(band=0))
    
    return lidar_hillshade, lidar_slope

#---------------#
# def coregister_point_data(ground_x, ground_y, z, is2_x, is2_y):
#     """
#     Co-registers ICESat-2 data with ground-based data. Ideally will be applicable
#     to all available point-source data.

#     Parameters
#     ----------
#     ground_x : float
#         Easting values for ground-based data of interest.
#         Units of meters.
#     ground_y : float
#         Northing values for ground-based data of interest.
#         Units of meters.
#     z : float
#         Surface heights or snow depths from ground-based data.
#         Units of meters.
#     is2_x : float
#         Easting values from ICESat-2 track.
#         Units of meters.
#     is2_y : float
#         Easting values from ICESat-2 track.
#         Units of meters.

#     Returns
#     -------
#     zi : float
#         Weighted heights of depths, based on distance between ICESat-2 and the
#         ground-based data.
#         Units of meters.

#     """
    
#     dist = distance_matrix(ground_x, ground_y, is2_x, is2_y)
    
#     weights = 1.0 / dist
    
#     zi = np.dot(weights.T, z)
    
#     return zi
# #---------------#
# def distance_matrix(x0, y0, x1, y1):
#     """
    
#     Helper function for coregister_point_data.
    
#     """
    
#     x0 = x0[(y0>y1.min()) & (y0<y1.max())]
#     y0 = y0[(y0>y1.min()) & (y0<y1.max())]
    
#     obs = np.vstack((x0, y0)).T
#     interp = np.vstack((x1, y1)).T
    
#     d0 = np.subtract.outer(obs[:,0], interp[:,0])
#     d1 = np.subtract.outer(obs[:,1], interp[:,1])
    
#     return np.hypot(d0, d1)