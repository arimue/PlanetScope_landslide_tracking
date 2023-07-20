# Tutorial 4: DEM building with PlanetScope L1B data and Ames Stereo Pipeline

This tutorial explains how you can use PlanetScope L1B data to generate a low-resolution (30 m) digital elevation model (DEM). A PlanetScope DEM can help to better model the Earth's surface at time of image acquisition and thus reduce the orthorectification errors in disparity maps derived from scenes acquired from variable perspectives. Tools for stereo correlation are part of the [Ames Stereo Pipeline](https://stereopipeline.readthedocs.io/en/latest/index.html).

## Step 1: Data selection

Contrary to before, it is essential to select an image pair with large perspective difference to maximize the parallax in the obtained disparity maps. This means you want to search for images with a large true view angle difference (ideally 10°). Also ensure that the images were acquired as close in time as possible, so you can minimize the bias introduced by changing terrain (e.g. through vegetation) or illumination conditions (different seasons). Also, I recommend to not use PSB.SD acquisitions from early 2020, as these may be affected by sub-frame misalignment resulting in significant stripe artifacts. I have not yet implemented a tool that suggests suitable image pairs, however, you can still use the tools from `planet_search_functions.py` to filter relevant scenes and view their metadata.

## Step 2: DEM generation

Once you have downloaded two suitable scenes, you can perform stereo correlation and triangulation. The whole process, including bundle adjustment, initial DEM building, mapprojection and final stereo correlation can be executed by running `dem_building()` from `asp_helper_functions.py`. As input, you will need to provide the path to your ASP installation, two images with large perspective differences and an EPSG code for the target CRS. Warning: Stereo correlation takes long, especially since it is run twice and using Bayes EM weighting for sub-pixel refinement to ensure highest quality. You can run it locally, but if you have access to a server with many cores, I recommend to move the process there. To reduce the runtime a little bit, constrain the stereo correlation to your AOI by providing a GeoJSON. Make sure that the DEM aoi is slightly larger that your original AOI, because pixels along the image margins often match with lower accuracy and are eroded during the DEM generation process. When providing an AOI for clipping, you will also need to point the system to a reference DEM so that the extent of the AOI can be transferred into image space via the RPCs:

``` python
import asp_helper_functions as asp

amespath = "/your/path/StereoPipeline-version-date-x86_64-Linux/bin"
demname = "/path/to/a/DEM.tif"
aoi = "./tutorial/dem_aoi.geojson"
img1 = "./tutorial/L1B/20220907_140709_64_24a3_1B_AnalyticMS_b2.tif"
img2 = "./tutorial/L1B/20220912_141056_91_2486_1B_AnalyticMS_b2.tif"

asp.dem_building(amespath, img1, img2, epsg = 32720, aoi = aoi, refdem = demname)

```

## Step 3: DEM alignment

As no ground control is used, the resulting DEM may be offset in all X Y and Z directions with regards to the terrain it is modeling. 
