# Tutorial 1: Searching the PlanetScope catalog for suitable data

This tutorial walks you through the data selection process. We will access and search the PlanetScope online catalog via the [Planet Software Development Kit (SDK) for Python](https://github.com/planetlabs/planet-client-python) and refine the search to form groups of images with common satellite perspective. Please make sure that you have authenticated with the Planet server using your Planet account (see [Documentation](https://planet-sdk-for-python-v2.readthedocs.io/en/latest/get-started/quick-start-guide/#step-4-sign-on-to-your-account).

##Step 1: Draw your area of interest (AOI)
Create a polygon that constrains your area of interest and store it as a GeoJSON. You can use QGIS for this. Make sure to use EPSG:4326 as a coordinate reference system.  

##Step 2: Conduct a rough search
``` python
import planet_search_functions as search
aoi = "./tutorial/test_aoi.geojson"

searchfile = search.search_planet_catalog(instrument = "PSB.SD", aoi = aoi, cloud_cover_max=0.1, date_start = "2020-03-01", date_stop = "2023-06-30")

```
