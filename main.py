#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 15:41:47 2023

@author: ariane
"""
import planet_search_functions as search
import preprocessing_functions as preprocessing
import postprocessing_functions as postprocessing
import core_functions as core
import asp_helper_functions as asp
import pandas as pd
import glob, os
import numpy as np
import helper_functions as helper

work_dir = "./tutorial/"
aoi = os.path.join(work_dir,"test_aoi.geojson") #TODO: check that AOI is in EPSG:4326, else reproject
instrument = "PSB.SD"

searchfile = search.search_planet_catalog(instrument = instrument, aoi = aoi, cloud_cover_max=0.3, date_start = "2022-11-01", date_stop = "2023-05-30")
df = search.refine_search_and_convert_to_csv(searchfile, aoi = aoi, min_overlap = 99)
groups = search.find_common_perspectives(df, va_diff_thresh = 0.8, min_group_size = 3, min_dt = 1)