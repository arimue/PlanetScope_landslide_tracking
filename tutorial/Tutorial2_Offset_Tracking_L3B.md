# Tutorial 2: Offset tracking with L3B data

In this tutorial, we will use the downloaded PlanetScope L3B data to retrieve disparity maps and estimate landslide velocity. The necessary processing steps are (1) isolate bands, (2) determine correlation pairs, (3) correlate data, (4) apply polynomial fit, and (5) calculate velocity. 

## Step 1: Isolate green band

The correlation with ASP can only be carried out for single-band images. Technically, a pseudo-panchromatic image could be generated from all RGB bands, however, due to inter-band misalignment we recommend to work with a single band only. To isolate bands use the functions collected unter `preprocessing_functions.py`:

``` python
import preprocessing_functions as preprocessing
import glob

work_dir = "./tutorial"

#get a list of all multiband TIFF files that you downloaded
files = glob.glob("/path/to/downloaded/scenes/jobname_psscene_analytic_sr_udm2/PSScene/*3B_AnalyticMS_SR_clip.tif") 
preprocessing.get_single_band(files, out_path = work_dir, band_nr = 2)
```

Here, I select all scenes in the folder containing the downloaded PlanetScope data (be sure to modify the path accordingly) and then extract the green band (band 2) from the multi-band raster. All output images will be saved in my working directory.

## Step 2: Find correlation pairs 

You have a couple of options to generate a matchfile (file storing correlation pairs of reference and secondary scenes). The easiest option is to provide the path to a directory were you have stored the data you want to correlate. Then you can match all files in there like this: 


``` python
matches = preprocessing.match_all(work_dir, ext = "_b2.tif", dt_min = 180)
```

This function collects all files with the extension *_b2.tif in the provided directory and forms correlation pairs. To ensure sufficient displacement for detection, a minimum temporal baseline of 180 days is required between the acquisition of both scenes. The specific value depends on the velocity of the investigated target. It's important to note that the function only matches older scenes with newer acquisitions, preventing any duplicate pairs where both A B and B A are considered. Also, this function does not consider true view angle difference, so you need to ensure that only scenes acquired from a common perspective are in the provided directory.

Alternatively, you can also use the pandas DataFrames obtained from searching the Planet catalog (see [Tutorial 1](./Tutorial1_Data_Search.md).
