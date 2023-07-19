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
work_dir = '/home/ariane/Downloads/test'
work_dir = "/home/ariane/Documents/PlanetScope/delMedio/L3B/group3/"
aoi = os.path.join("./tutorial/","test_aoi.geojson") #TODO: check that AOI is in EPSG:4326, else reproject
instrument = "PSB.SD"

# searchfile = search.search_planet_catalog(instrument = instrument, aoi = aoi, cloud_cover_max=0.1, date_start = "2020-03-01", date_stop = "2023-06-30")
# df = search.refine_search_and_convert_to_csv(searchfile, aoi = aoi, min_overlap = 99)
# groups = search.find_common_perspectives(df, va_diff_thresh = 0.6, min_group_size = 5, min_dt = 30, searchfile = searchfile)

files = glob.glob("/home/ariane/Downloads/Siguas_L3B_fillGaps_psscene_analytic_sr_udm2/PSScene/*3B_AnalyticMS_SR_clip.tif")
#preprocessing.get_single_band(files, out_path = "/home/ariane/Downloads/test/", band_nr = 2)
#matches = preprocessing.match_all(work_dir, ext = "_b2.tif", dt_min = 180)
matches = preprocessing.generate_matchfile_from_search(df, dt_min = 180, path = work_dir, check_existence=True)
matches = preprocessing.generate_matchfile_from_groups(groups, dt_min = 180, path = work_dir, check_existence=True)