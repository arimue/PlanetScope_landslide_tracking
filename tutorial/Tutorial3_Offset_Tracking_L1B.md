# Tutorial 3: Offset tracking with L1B data

Working with L1B data has the advantage that you as the user gain control over the reference DEM that is used for orthorectification. If you project your data onto a DEM that closely resembles the topography at the time of acquisition of the PlanetScope data, you can greatly reduce the orthorectification error and thus correlate image pairs, even if they were acquired from opposite look directions. This tutorial outlines all steps from data search to velocity calculation for L1B data. Many steps are similar to the processing of L3B data, so also refer to tutorials [1](./tutorial/Tutorial1_Data_Search.md) and [2](./tutorial/Tutorial2_Offset_Tracking_L3B.md) for details on these.  

## Step 1: Get some data

For data search, you may use the tools from `planet_search_functions.py`, but you can also just go the Planet Explorer directly as you are not limited to common view and satellite azimuth angles. Filter and download suitable acquisitions that are cloud-free and cover the full area of interest (AOI). To download scenes in L1B format, go to "Show more" below the recified assets and select "Basic analytic radiance (TOAR) - 4 band" from the unrectified assets. Note: for the unrecified assets, there is no clipping option available, so you will have to download full scenes. This will consume more quota than L3B cropped to the AOI.

## Step 2: Isolate green band

As before, you will need to extract a single band from the muli-band scene for correlation:

``` python
import preprocessing_functions as preprocessing
import glob

work_dir = "./tutorial/L1B"

#get a list of all multiband TIFF files that you downloaded
files = glob.glob("/path/to/downloaded/scenes/jobname_psscene_basic_analytic_udm2/PSScene/*1B_AnalyticMS.tif") 
preprocessing.get_single_band(files, out_path = work_dir, band_nr = 2)
```

## Step 3: Orthorectify L1B data
To orthorectify the raw L1B scenes based on a new DEM, use `orthorectify_L1B` from `preprocessing_functions.py`. This will do a few things: (1) create a rough rectangular clip of the L1B data based on the extents of the given AOI. This speeds up the orthorectification. (2) Orthorectify the data using the [mapproject](https://stereopipeline.readthedocs.io/en/latest/tools/mapproject.html) tool from [Ames Stereo Pipeline](https://stereopipeline.readthedocs.io/en/latest/index.html), and (3) clip the orthorectified data to the AOI again because the intial clip is only an approximation. Make sure to provide the path to your ASP installation, a list of files, the DEM you want to use for orthorectification, the AOI in for of a GeoJSON file and the EPSG code of the CRS you want your output data to have. Choose a projected CRS.

``` python
amespath = "/your/path/StereoPipeline-version-date-x86_64-Linux/bin"
aoi = "./tutorial/test_aoi.geojson"
demname = "/path/to/a/DEM.tif"

preprocessing.orthorectify_L1B(amespath, files, demname, aoi, epsg = 32720)
```

All output files will be stored in the same directiory as your input image files and will have the file extension `*_clip_mp_clip.tif`. 

## Step 4: Find correlation pairs
Now you can follow the same workflow as for the L3B data, just make sure to set the correct file extension. Create a matchfile by matching all orthorectified and clipped L1B (now technically L3B) scenes:
``` python
matches = preprocessing.match_all(work_dir, ext = "_b2_clip_mp_clip.tif", dt_min = 180)
```
## Step 5: Correlate data
``` python
import asp_helper_functions as asp

dmaps = asp.correlate_asp_wrapper(amespath, matches, sp_mode = 2, corr_kernel = 35, prefix_ext = "_L1B")
```
## Step 6: Apply polynomial fit
``` python
dmaps_pfit = opt.apply_polyfit(matches, prefix_ext= "_L1B", order = 2, demname = demname)
```
## Step 7: Calculate velocity
``` python
import postprocessing_functions as postprocessing

vels = postprocessing.calc_velocity_wrapper(matches, prefix_ext = "_L1B_polyfit")
```

For the Del Medio landslide, we found that the orthorectification error was already significantly reduced when projecting the L1B images onto the Copernicus DEM instead of the SRTM DEM. To obtain an even more up-to-date topography, you can also attempt to use PlanetScope L1B data to build a smooth, low-resolution DEM that can then be used as reference during orthorectification. The DEM generation is documented in [Tutorial 4](./tutorial/Tutorial4_DEM_Building.md).

