# Tutorial 1: Searching for PlanetScope data

This tutorial walks you through the data selection process. We will access and search the PlanetScope online catalog via the [Planet Software Development Kit (SDK) for Python](https://github.com/planetlabs/planet-client-python) and refine the search to form groups of images with common satellite perspective. Please make sure that you have authenticated with the Planet server using your Planet account (see [Documentation](https://planet-sdk-for-python-v2.readthedocs.io/en/latest/get-started/quick-start-guide/#step-4-sign-on-to-your-account). Note: The result will be a recommendation of scenes to download. You can download these automatically via the Planet API, however, I do recommend to check all imagery visually in the Planet Explorer, to make sure your target is not cloud covered.

## Step 1: Draw your area of interest (AOI)
Create a polygon that constrains your area of interest and store it as a GeoJSON. You can use QGIS for this. Make sure to use EPSG:4326 as a coordinate reference system.  

## Step 2: Conduct a rough search
All functions related to data search are stored under `planet_search_functions.py`. Here is an example for a first-order search for PlanetScope data acquired between March 2020 and June 2023 by PSB.SD instruments over the given AOI:
``` python
import planet_search_functions as search
aoi = "./tutorial/test_aoi.geojson"

searchfile = search.search_planet_catalog(instrument = "PSB.SD", aoi = aoi, cloud_cover_max=0.1, date_start = "2020-03-01", date_stop = "2023-06-30")
```
Executing this code will provide you with a new GeoJSON file (search.geojson) that stores the footprints and metadata of all scenes that match your filter criteria. You can open this in QGIS:

![AOI and footprints of data that fit search criteria.](./figures/search1.jpeg =250x)

