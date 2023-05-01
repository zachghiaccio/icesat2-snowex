# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 13:39:20 2022

@author: zfair
"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from datetime import datetime
from pyproj import Proj, transform
import ulmo

#---------------#
def process_raw_data(df):
    """
    Takes unprocessed site data and makes it more easily readable for further
    analysis. Currently works for the magnaprobe data gathered from Alaska 2022.

    Parameters
    ----------
    txt_file : str
        .txt file containing lat/lon coordinates and snow depth data from one of
        the SnowEx Alaska sites.

    Returns
    -------
    df : DataFrame
        DataFrame containing processed magnaprobe data, including snow depth (in meters),
        condensed lat/lon coordinates, and easting/northing estimates.

    """
    
    #df = pd.read_csv(txt_file, header=1)
    
    # Include only rows with data
    df = df.iloc[2:]
    
    # Convert numeric data to floats
    df['DepthCm'] = pd.to_numeric(df['DepthCm'])
    df['altitudeB'] = pd.to_numeric(df['altitudeB'])
    df[['latitude_a', 'LatitudeDDDDD']] = df[['latitude_a', 'LatitudeDDDDD']].apply(pd.to_numeric)
    df[['Longitude_a', 'LongitudeDDDDD']] = df[['Longitude_a', 'LongitudeDDDDD']].apply(pd.to_numeric)
    
    # Combine coordinates into single columns
    df['Latitude'] = df['latitude_a'] + df['LatitudeDDDDD']
    df['Longitude'] = df['Longitude_a'] + df['LongitudeDDDDD']
    
    # Convert snow depth to meters
    df['Depth'] = df['DepthCm'] / 100.
    
    # Estimate easting/northing coordinates
    inp = Proj(init='epsg:4326')
    outp = Proj(init='epsg:32606')
    
    df['Easting'], df['Northing'] = transform(inp, outp, 
                                              df['Longitude'], df['Latitude'])
    
    # Rename time and elevation columns for consistency with other scripts
    df = df.rename(columns={'TIMESTAMP': 'UTCdoy',
                            'altitudeB': 'Elevation'})
    
    # Plotting, for debugging/quick views
    if True:
        plt.scatter(df['Easting'], df['Northing'], c=df['Depth'])
        plt.xlabel('easting [m]')
        plt.ylabel('northing [m]')
        plt.show()
    
    return df

#---------------#
def snotel_fetch(sitecode, variablecode='SNOTEL:SNWD_D', start_date='1950-10-01', end_date='2020-12-31'):
    """
    Accesses SNOTEL data from the cloud using the CUAHSI HydroPortal.
    
    Parameters
    ----------
    sitecode: str
        Numeric ID for SNOTEL site, in string format.
    variablecode: str
        Dataset to be accessed. 
        Default: SNOTEL snow depth.
    start_date: str
        Start date of SNOTEL time series, in YYYY-MM-DD format.
        Default: 1950-10-01
    end_date: str
        End date of SNOTEL time series, in YYYY-MM-DD format.
        Default: 2020-12-31
        
    Returns
    -------
    values_df: DataFrame
        DataFrame containing desired SNOTEL dataset.
    
    """
    
    #print(sitecode, variablecode, start_date, end_date)
    values_df = None
    wsdlurl = 'https://hydroportal.cuahsi.org/Snotel/cuahsi_1_1.asmx?WSDL'
    try:
        #Request data from the server
        site_values = ulmo.cuahsi.wof.get_values(wsdlurl, sitecode, variablecode, start=start_date, end=end_date)
        #Convert to a Pandas DataFrame   
        values_df = pd.DataFrame.from_dict(site_values['values'])
        #Parse the datetime values to Pandas Timestamp objects
        values_df['datetime'] = pd.to_datetime(values_df['datetime'], utc=True)
        #Set the DataFrame index to the Timestamps
        values_df = values_df.set_index('datetime')
        #Convert values to float and replace -9999 nodata values with NaN
        values_df['value'] = pd.to_numeric(values_df['value']).replace(-9999, np.nan)
        #Remove any records flagged with lower quality
        values_df = values_df[values_df['quality_control_level_code'] == '1']
    except:
        print("Unable to fetch %s" % variablecode)

    return values_df
#---------------#
def add_dowy(df, col=None):
    """
    Companion function to snotel_fetch. Converts days of year to days of water year.
    
    Parameters
    ----------
    df: Dataframe
        DataFrame that contains SNOTEL data (values_df in snotel_fetch).
    col: Pandas column
        DataFrame column that contains day-of-year information. If passed as None, then attempts to access day of year from the indeces.
        Default: None
    
    Returns
    -------
    None
    
    """
    
    if col is None:
        df['doy'] = df.index.dayofyear
    else:
        df['doy'] = df[col].dayofyear
    # Sept 30 is doy 273
    df['dowy'] = df['doy'] - 273
    df.loc[df['dowy'] <= 0, 'dowy'] += 365
#---------------#
def coregister_nlcd_data(is2_pd, land_cover_tif, txt_name):
    """
    Uses the nearest-neighbor approach to coregister National Land Cover Data (NLCD) with ICESat-2 data. The nearest-neighbor approach is needed to preserve land cover values that would be lost with spline interpolation.
    
    Unlike the lidar coregistration, matched land cover data is saved to a txt file to speed up the process for subsequent code runs.
    
    Parameters
    ----------
    is2_pd: DataFrame
        DataFrame containing ICESat-2 data, processed using the lidar_processing scripts.
    land_cover_tif: Xarray tiff
        Xarray object containing land cover values across Alaska.
    txt_name: str
        Name of text file that will store coregistered land cover information.
        
    Returns
    -------
    None
    
    """
    # Initialize the text file
    nlcd_file = open(txt_name, 'w')
    
    # Use nearest-neighbor approach to match land cover with ATL03 segments
    for i in np.arange(len(is2_pd)):
        x = is2_pd['x'][i]
        y = is2_pd['y'][i]
        
        dx = x - land_cover_tif.x
        dy = y - land_cover_tif.y
        d = np.sqrt(dx**2 + dy**2)
        
        x_idx = d.argmin(dim='x')[0].values
        y_idx = d.argmin(dim='y')[0].values
        
        tmp = land_cover_tif[:, x_idx, y_idx].values
        
        # Write land cover data to text file iteratively
        nlcd_file.write('%f\n' %tmp)
    
    # Close text file
    nlcd_file.close()
#---------------#
def process_nlcd_data(is2_pd, txt_name):
    """
    Takes coregistered land cover values and translates them to more descriptive labels. Also corrects misclassified values that were found over the ACP site in Alaska.
    
    Parameters
    ----------
    is2_pd: DataFrame
        DataFrame containing ICESat-2 data, processed using the lidar_processing scripts.
        
    txt_name: str
        Name of text file that contains coregistered land cover information.
        
    Returns
    -------
    is2_pd: DataFrame
        An updated ICESt-2 dataframe that contains both land cover values and the corresponding labels.
        
    """
    
    # Add land cover data to DataFrame
    is2_pd['land_cover_value'] = pd.read_table(txt_name, header=None).values
    
    path = '/home/jovyan/icesat2-snowex/jsons-shps/land-cover-maps/'
    # Translate values into land cover classification
    nlcd_legend = pd.read_csv(f'{path}/NLCD_landcover_legend_2018_12_17_39kiznbMV7t0juCIU61D.csv', header=0)
    
    is2_pd['land_cover'] = ''
    # Apply legend monikers to new column. Also converts strings to something that is easier to type.
    for i,value in enumerate(is2_pd['land_cover_value']):
        # Account for misclassified values
        if (value==51.0) | (value==52.0):
            is2_pd['land_cover'][i] = 'Shrub'
        elif (value == 72.0) | (value==71.0):
            is2_pd['land_cover'][i] = 'Herbaceous'
        else:
            is2_pd['land_cover'][i] = nlcd_legend['Legend'].loc[value]
            
    return is2_pd