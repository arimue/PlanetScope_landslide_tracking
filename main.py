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
import pandas as pd
import glob
from helper_functions import get_scene_id

#TODO: write wrapper for DEM error finding
#TODO: write wrapper for applying correlation etc to files

# # #search for Planet Data in the AOI
# gj = search.search_planet_catalog(instrument = "PS2", geom = "/home/ariane/Documents/PlanetScope/jujuy_final_mask_dovec.geojson", cloud_cover_max=0.5)
# #refine search to get scenes the fully overlap AOI polygons and share the same orbit
# df = search.refine_search_and_convert_to_csv(gj, refPoly = "/home/ariane/Documents/PlanetScope/jujuy_final_mask_dovec.geojson", minOverlap = 99)
# #get a suggestion for a reference scene. Inspect this visually in the Planet Explorer!
# stable = search.suggest_reference_and_stable_pair(df)

# preprocess downloaded scenes (isolate green band)
# get a list of all downloaded scenes
# files = glob.glob("/home/ariane/Downloads/Dove_C_fill*/files/PSScene/*/basic_analytic_udm2/*AnalyticMS.tif")
# all_files = preprocessing.preprocess_scenes(files, outpath = "/home/ariane/Documents/PlanetScope/Dove-C_Jujuy_all/L1B/", bandNr = 2)
# all_files = glob.glob("./Dove-C_Jujuy_all/L1B/*_b2.tif")
# reference = "./Dove-C_Jujuy_all/L1B/20190626_140959_103c_1B_Analytic_b2.tif"
# df = preprocessing.generate_matchfile(all_files, reference, checkOverlap = True, refPoly = "/home/ariane/Documents/PlanetScope/polygon_DoveC.geojson", minOverlap = 95)

#after remapping correlate these images
#file = matf.build_remapped_match_file_crossref('/home/ariane/Documents/PlanetScope/SD_Jujuy_Nadir/sd_matches_crossangle.csv')

#file = preprocessing.build_remapped_match_file_crossref('/home/ariane/Documents/PlanetScope/Dove-C_Jujuy_all/L1B/matches_final.csv')

ul_lon = -65.61782
ul_lat = -23.88517

ysize = 1600
xsize = 3200

path = "./Dove-C_Jujuy_all/L1B/"

amespath = "/raid-manaslu/amueting/StereoPipeline-3.1.1-alpha-2022-07-29-x86_64-Linux/bin/"
dem = "./data/DEM/EGM96/demcoreg_alos/CopernicusDEM_EasternCordillera_EGM96_clip_AW3D30_NWArg_nuth_x+10.19_y-0.36_z+2.36_align_3m.tif"
dem_err_x = f"{path}/stereo/dem_error_dx_mean_mgm.tif"
dem_err_y = f"{path}/stereo/dem_error_dy_mean_mgm.tif"

# df = pd.read_csv("./Dove-C_Jujuy_all/L1B/matches_final.csv")#(file)
# for i in range(len(df)):
#     core.raw_correlate_and_correct(df.ref[i], df.sec[i], dem, amespath, ul_lon, ul_lat, xsize = xsize, ysize = ysize, zgrid = None, dem_err_x = dem_err_x, dem_err_y = dem_err_y, reduce = 5, first_fit_order = 1, ext = "_Err", overwrite = False, plot = False)
    
# df = pd.read_csv(path+"matches_remapped_crossref.csv")

# for i in range(len(df)):
#     core.correlate_remapped_img_pairs(df.ref[i], df.sec[i], amespath)
    
    
    
matchfile = "./Dove-C_Jujuy_all/L1B/matches_remapped_crossref.csv"
img_with_rpc = "./Dove-C_Jujuy_all/L1B/20190626_140959_103c_1B_Analytic_b2_clip.tif"
pad = 5
xcoord = 2170
ycoord = 1347
postprocessing.mapproject_and_calc_velocity(amespath, matchfile, dem, img_with_rpc)
postprocessing.generate_timeline(matchfile, xcoord, ycoord, pad = pad, resolution = 3, take_velocity = True, plot = True)