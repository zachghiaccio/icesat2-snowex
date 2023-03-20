# icesat2-snowex
Code and data to compare ICESat-2 data with SnowEx 2020 and 2023.

Only takes specific datasets as input, but future versions will include more generic coding.

Currently working on integrating these scripts into the cloud, ideally focusing on data access without the need to download large numbers of files.

## Scripts
**ground_data_processing.py** - (1) Takes unprocessed magnaprobe data from the Alaska 2022 campaign and makes it readable for further analysis. Latitude and longitude are converted to a more readable format, and variable labels are given more generic names. Snow depth is converted to meters. Easting and northing coordinates are derived from the lat/lon coordinates, at projection EPSG: 32606 (UTM zone 6N; Alaska). (2) Accesses SNOTEL data from the cloud (requires ulmo package). (3) If SNOTEL data is accessed, then the day of year (DOY) and day of water year (DOWY) is added to dataframe loaded from SNOTEL.

**is2_bsu_intersection.py** - Coregisters ICESat-2 with ASO/Quantum lidar data over Grand Mesa, CO. Comparisons are made for the SnowEx 2020 campaign. Requires the "ground_data_processing" and "lidar_processing" modules.

**is2_kml_maker.py** - (1) Loads an ICESat-2 .h5 file and converts the given lat/lon coordinates into a KML file (requires simplekml and osgeo). (2) Extracts latitude and longitude from KML files. (3) Using a pre-existing ICESat-2 KML track for the central beam, derives new KML files for the left and right beams.

**is2_neon_coregistration.py** - Same as "is2_bsu_intersection", but for the SnowEx 2022/2023 campaign in Alaska.

**lidar_processing.py** - Contains several functions to process ICESat-2 and airborne lidar data. (1) Identifies the strong beams in a read ICESat-2 file. (2) Concatenates ICESat-2 surface height data into dataframes. (3) Coregisters ICESat-2 dataframes with airborne lidar DEMs. (4) Coregisters airborne lidar DEMs with ground-based data sets. (5) Coregisters ICESat-2 data with USGS land cover maps. (6) Generates slope and shaded relief maps from airborne lidar DEMs (requires xrspatial package).

**sliderule_query.ipynb** - An interactive Jupyter Notebook for querying ICESat-2 data with the SlideRule tool.

**snowex_rgt_finder.ipynb** - Uses the icepyx software package to query ICESat-2 data over a specified region (or regions) of interest. Can also be used to identify reference ground tracks (RGTs) that pass through the ROI.