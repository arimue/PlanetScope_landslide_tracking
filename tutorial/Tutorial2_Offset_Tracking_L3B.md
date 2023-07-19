# Tutorial 2: Offset tracking with L3B data

In this tutorial, we will use the downloaded PlanetScope L3B data to retrieve disparity maps and estimate landslide velocity. The necessary processing steps are (1) isolate bands, (2) determine correlation pairs, (3) correlate data, (4) apply polynomial fit, and (5) calculate velocity. 

## Step 1: Isolate green band

The correlation with ASP can only be carried out for single-band images. Technically, a pseudo-panchromatic image could be generated from all RGB bands, however, due to inter-band misalignment we recommend to work with a single band only. To isolate bands use the functions collected unter `preprocessing_functions.py`:

``` python
import preprocessing_functions as preprocessing
import glob

work_dir = "./tutorial/"

#get a list of all multiband TIFF files that you downloaded
files = glob.glob("/path/to/downloaded/scenes/jobname_psscene_analytic_sr_udm2/PSScene/*3B_AnalyticMS_SR_clip.tif") 
preprocessing.get_single_band(files, out_path = work_dir, band_nr = 2)
```

Here, I select all scenes in the folder containing the downloaded PlanetScope data (be sure to modify the path accordingly) and then extract the green band (band 2) from the multi-band raster. All output images will be saved in my working directory in a new folder called `b2`.

## Step 2: Find correlation pairs 

