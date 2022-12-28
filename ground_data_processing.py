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
    if col is None:
        df['doy'] = df.index.dayofyear
    else:
        df['doy'] = df[col].dayofyear
    # Sept 30 is doy 273
    df['dowy'] = df['doy'] - 273
    df.loc[df['dowy'] <= 0, 'dowy'] += 365