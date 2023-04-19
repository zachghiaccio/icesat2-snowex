# Preliminary ICESat-2 Snow Depths: Data Dictionary
This directory contains preliminary snow depth data derived from ICESat-2 ATL03, ATL06, and ATL08 products. The CSV files contain variables that may not be immediately obvious to a potential user, so this document outlines each variable.

Note that the ATL03 products contain more information than the ATL06/ATL08 products, which are identical beyond the resolution and pre-processing methods.

Spatial Resolution for Each Product:
* ATL03: 10 m*
* ATL06: 40 m^
* ATL08: 100 m^

*ATL03 data is processed at 10 m resolution, but aggregates photons within 20 m segments using high-confidence ground photons. Ground photons are identified using the ATL08 vegetation classification scheme.
^Both ATL06 and ATL08 DataFrames have an unnamed first column that represents the row ID in the original data file. Planned to be removed.

## ATL03
| Label     | Long Name | Data Type | Description | 
| ----------- | ----------- | ----------- | ----------- |
| time     | Acquisition date/time | pandas.Timestamp | Date and time of received photons |
| x   | Easting (meters) | float64 | Easting coordinate in CRS EPSG:32606 |
| y | Northing (meters) | float64 | Northing coordinate in CRS EPSG:32606 |
| lidar_height | Lidar surface elevation (meters) | float64 | Surface elevation observed by UAF airborne lidar |
| lidar_snow_depth | Lidar snow depth (meters) | float64 | Snow depth derived from UAF airborne lidar |
| is2_height | ICESat-2 surface elevation (meters) | float64 | Surface elevation observed by ICESat-2 |
| h_sigma | ATL03 uncertainty (meters) | float64 | Uncertainty in elevation estimate, based on spread of ATL03 photons |
| beam | ICESat-2 beam | string | ICESat-2 beam identifier. 1, 3, 5 are strong beams (right, center, left); 2, 4, 6 are weak beams (right, center, left) |
| residual | IS2-UAF height residual (meters) | float64 | Elevation difference between co-registered ICESat-2 and UAF data. **This is the ICESat-2 snow depth estimate during the snow-on season.** |
| snow_depth_residual | IS2-UAF snow depth residual (meters) | float64 | Snow depth residual between ICESat-2 and UAF estimates |
| lon | Longitude | float64 | Longitude coordinate |
| lat | Latitude | float64 | Latitude coordinate |
| geometry | lat/lon point geometry | shapely.geometry | Lat/lon coordinate for data mapping |
| land_cover_value | Land cover ID | float64 | NLCD numeric identifier for land cover type |
| land_cover | Land cover type | string | NLCD 2019 land cover classification |

## ATL06/08
| Label     | Long Name | Data Type | Description | 
| ----------- | ----------- | ----------- | ----------- |
| x | Easting (meters) | float64 | Easting coordinate in CRS EPSG:32606 |
| y | Northing (meters) | float64 | Northing coordinate in CRS EPSG:32606 |
| lidar_height | Lidar surface elevation (meters) | float64 | Surface elevation observed by UAF airborne lidar |
| lidar_snow_depth | Lidar snow depth (meters) | float64 | Snow depth derived from UAF airborne lidar |
| is2_height | ICESat-2 surface elevation (meters) | float64 | Surface elevation observed by ICESat-2 |
| beam | ICESat-2 beam | string | ICESat-2 beam identifier. 1, 3, 5 are strong beams (right, center, left); 2, 4, 6 are weak beams (right, center, left) |
| residual | IS2-UAF height residual (meters) | float64 | Elevation difference between co-registered ICESat-2 and UAF data. **This is the ICESat-2 snow depth estimate during the snow-on season.** |
| snow_depth_residual | IS2-UAF snow depth residual (meters) | float64 | Snow depth residual between ICESat-2 and UAF estimates |
| lon | Longitude | float64 | Longitude coordinate |
| lat | Latitude | float64 | Latitude coordinate |
| geometry | lat/lon point geometry | shapely.geometry | Lat/lon coordinate for data mapping |