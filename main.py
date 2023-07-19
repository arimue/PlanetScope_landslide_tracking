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
amespath = "/home/ariane/Downloads/StereoPipeline-3.1.1-alpha-2022-10-16-x86_64-Linux/bin"
# searchfile = search.search_planet_catalog(instrument = instrument, aoi = aoi, cloud_cover_max=0.1, date_start = "2020-03-01", date_stop = "2023-06-30")
# scenes = search.refine_search_and_convert_to_csv(searchfile, aoi = aoi, min_overlap = 99)
# groups = search.find_common_perspectives(scenes, va_diff_thresh = 0.6, min_group_size = 5, min_dt = 30, searchfile = searchfile)

files = glob.glob("/home/ariane/Downloads/Siguas_L3B_fillGaps_psscene_analytic_sr_udm2/PSScene/*3B_AnalyticMS_SR_clip.tif")
#preprocessing.get_single_band(files, out_path = "/home/ariane/Downloads/test/", band_nr = 2)
#matches = preprocessing.match_all(work_dir, ext = "_b2.tif", dt_min = 180)
#matches = preprocessing.generate_matchfile_from_search(scenes, dt_min = 180, path = work_dir, check_existence=True)
matches = preprocessing.generate_matchfile_from_groups(groups, dt_min = 600, path = work_dir, check_existence=True)
matches = matches.iloc[0:2]

dmaps = asp.correlate_asp_wrapper(amespath, matches, sp_mode = 1, corr_kernel = 35, prefix_ext = "_L3B", overwrite=True)

dx = helper.read_file(maps[0])
dy = helper.read_file(maps[0], 2)
mask = helper.read_file(maps[0], 3)
import matplotlib.gridspec as gridspec
fig, ax = plt.subplots(1, 3, figsize=(12, 4))
im0 = ax[0].imshow(dx, vmin=-2, vmax=2, cmap="coolwarm")
im1 = ax[1].imshow(dy, vmin=-2, vmax=2, cmap="coolwarm")
im2 = ax[2].imshow(mask, vmin=0, vmax=1, cmap="Greys")

# Add colorbars
cbar0 = fig.colorbar(im0, ax=ax[0], aspect=40)
cbar1 = fig.colorbar(im1, ax=ax[1], aspect=40)
cbar2 = fig.colorbar(im2, ax=ax[2], aspect=40)

plt.savefig("./tutorial/figures/disp_map.png", dpi = 300)

