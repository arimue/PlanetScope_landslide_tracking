# Tutorial 3: Offset tracking and DEM building with L1B data

Working with L1B data has the advantage that you as the user gain control over the reference DEM that is used for orthorectification. If you project your data onto a DEM that closely resembles the topography at the time of acquisition of the PlanetScope data, you can greatly reduce the orthorectification error and thus correlate image pairs, even if they were acquired from opposite look directions. This tutorial outlines all steps from data search to velocity calculation for L1B data and also describes how to generate DEMs from unprojected PlanetScope data itself. Many steps are similar to the processing of L3B data, so also refer to tutorials [1](./tutorial/Tutorial1_Data_Search.md) and [2](./tutorial/Tutorial2_Offset_Tracking_L3B.md) for details on these.  

## Step 1: Get some data

For data search, you may use the tools from `planet_search_functions.py`, but you can also just go the Planet Explorer directly as you are not limited to common view and satellite azimuth angles. Filter and download suitable acquisitions that are cloud-free and cover the full area of interest (AOI). To download scenes in L1B format, go to "Show more" below the recified assets and select "Basic analytic radiance (TOAR) - 4 band" from the unrectified assets. Note: for the unrecified assets, there is no clipping option available, so you will have to download full scenes. This will consume more quota than L3B cropped to the AOI.

## Step 2: Isolate green band

As before, you will need to extract a single band from the muli-band scene for correlation:

``` python
import preprocessing_functions as preprocessing
import glob

work_dir = "./tutorial"

#get a list of all multiband TIFF files that you downloaded
files = glob.glob("/path/to/downloaded/scenes/jobname_psscene_basic_analytic_udm2/PSScene/*1B_AnalyticMS.tif") 
preprocessing.get_single_band(files, out_path = work_dir, band_nr = 2)
```

## Step 3: Orthorectify L1B data
To orthorectify the raw L1B scenes based on a new DEM, use `orthorectify_L1B` from `preprocessing_functions.py`. This will do a few things: (1)


