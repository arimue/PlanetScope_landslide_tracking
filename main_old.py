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
from helper_functions import get_scene_id, read_file, read_meta, size_from_aoi, fixed_val_scaler

###################################LOSMEDANOS###########################################################################
# gj = search.search_planet_catalog(instrument = "PSB.SD", geom = "/home/ariane/Documents/PlanetScope/LasZanjas/las_zanjas_aoi.geojson", cloud_cover_max=0.1)
# df = search.refine_search_and_convert_to_csv(gj, refPoly = "/home/ariane/Documents/PlanetScope/LasZanjas/las_zanjas_aoi.geojson", minOverlap = 99)

# stable = search.suggest_reference_and_stable_pair(df, max_day_diff = 15, angle_lim_ref = 0.1)

####################################MINAPURNA############################################################################
amespath = "/raid-manaslu/amueting/StereoPipeline-3.1.1-alpha-2022-07-29-x86_64-Linux/bin/"

#amespath = "/home/ariane/Downloads/StereoPipeline-3.1.1-alpha-2022-10-16-x86_64-Linux/bin/"
demname = "/raid-manaslu/amueting/PhD/Project1/ImageTransformation/MinaPurna/DEM/COP_aligned_to_ALOS_3m_MPclip.tif"

# 

# gj = search.search_planet_catalog(instrument = "PSB.SD", geom = "/home/ariane/Documents/PlanetScope/delMedio/del_medio_aoi.geojson", cloud_cover_max=0.1, date_start = "2020-01-01", date_stop = "2023-05-30")
# df = search.refine_search_and_convert_to_csv(gj, refPoly = "/home/ariane/Documents/PlanetScope/delMedio/del_medio_aoi.geojson", minOverlap = 99)

# gj = search.search_planet_catalog(instrument = "PSB.SD", geom = "/home/ariane/Documents/PlanetScope/Siguas/siguas_aoi.geojson", cloud_cover_max=0.2, date_start = "2020-01-01", date_stop = "2023-06-01")
# df = search.refine_search_and_convert_to_csv(gj, refPoly = "/home/ariane/Documents/PlanetScope/Siguas/siguas_aoi.geojson", minOverlap = 99) 


# matches = preprocessing.match_all(df, path = "./Siguas/L3B/randomPairs/", dt = 180, checkExistence=True)

# matches = match_all(df, path = "./delMedio/L1B/group4/", ext = "_1B_AnalyticMS_b2_clip_mp_clip.tif",  level = 1, dt = 180, checkExistence=True)
#df_elements = df.sample(n=10)
# matches = preprocessing.match_all(df_elements, path = "./Siguas/L3B/")
#matches = preprocessing.match_all(df, path = "./delMedio/L3B/group2/", checkExistence=True, dt = 180)
# matches = pd.read_csv("./Siguas/L3B/matches_stable.csv")
# scores = preprocessing.rate_match(df, matches)
#scores.to_csv("/home/ariane/Documents/PlanetScope/Siguas/L3B/matches_stable_scores_with_true_va.csv", index = False)

# groups = preprocessing.find_best_matches(df, mindt = 30, minGroupSize = 5)
# groups = groups.loc[groups.group_id.isin([5,9])].reset_index(drop = True)


# files = glob.glob("/home/ariane/Documents/PlanetScope/Siguas/L3B/*_b2.tif")
# df = pd.read_csv("Siguas/L3B/matches_stable.csv")
# refids = [get_scene_id(sec) for sec in df.ref]
# secids = [get_scene_id(sec) for sec in df.sec]

# refids.extend(secids)

# ids = np.unique(refids)

# ids = [get_scene_id(f) for f in files]
# groups = df.loc[df.ids.isin(ids)].reset_index(drop = True)
# groups.to_csv("/home/ariane/Documents/PlanetScope/Siguas/L3B/stable_groupinfo.csv", index = False)

# groupsm = pd.DataFrame()
# i = 1
# for group in ["group1", "group2", "group3", "random"]:
#     files = glob.glob(f"/home/ariane/Documents/PlanetScope/delMedio/L3B/{group}/*_b2.tif")
#     ids = [get_scene_id(f,level = 3) for f in files]
#     groups = df.loc[df.ids.isin(ids)].reset_index(drop = True)
#     groups["type"] = f"Group {i}"
#     i += 1

#     groupsm = pd.concat([groupsm, groups])
# groupsm.to_csv("/home/ariane/Documents/PlanetScope/delMedio/L3B/groupinfo.csv", index = False)



#matches = preprocessing.generate_matchfile_from_groups(groups, path = "./delMedio/L3B/", checkExistence = True)
#matches = preprocessing.match_all(df, path = "./Siguas/L3B/", checkExistence = True)
# infodf = pd.read_csv("/raid-manaslu/amueting/PhD/Project1/ImageTransformation/Siguas/L3B/info_siguas_scenes.csv")
# matches = pd.read_csv("/raid-manaslu/amueting/PhD/Project1/ImageTransformation/Siguas/L3B/all_matches.csv")
# scores = preprocessing.rate_match(infodf, matches)
#result = preprocessing.add_offset_from_mask_to_rated_matches(scores, stereopath ="/raid-manaslu/amueting/PhD/Project1/ImageTransformation/Siguas/L3B/stereo/" , mask = "/raid-manaslu/amueting/PhD/Project1/ImageTransformation/Siguas/siguas_dist_area.tif")
#result.to_csv("/raid-manaslu/amueting/PhD/Project1/ImageTransformation/Siguas/L3B/mask_stats.csv", index = False)
#result = preprocessing.add_offset_variance_to_rated_matches(scores, stereopath = "/raid-manaslu/amueting/PhD/Project1/ImageTransformation/Siguas/L3B/stereo/")
#result.to_csv("/raid-manaslu/amueting/PhD/Project1/ImageTransformation/Siguas/L3B/var_stats.csv", index = False)

# all_files = glob.glob("./Siguas/L3B/*_b2.tif")
# match = preprocessing.generate_matchfile(all_files, reference = "./Siguas/L3B/20220823_145618_97_2407_3B_AnalyticMS_SR_clip_b2.tif")
# preprocessing.build_remapped_match_file_crossref("./Siguas/L3B/matches.csv", level = 3, dt_min = 0)
# search.download_xml_metadata(list(df.ids), out_dir = "/home/ariane/Documents/PlanetScope/test_ang_calc/")
# df = df.loc[df.ids != "20221101_140904_78_2490"]

# stable = search.suggest_reference_and_stable_pair(df, max_day_diff = 20)

# files = glob.glob("/home/ariane/Downloads/Siguas_L1B_pt2*/PSScene/*AnalyticMS.tif")
# # files = glob.glob("/home/ariane/Downloads/delMedio_addGroup*/PSScene/*AnalyticMS_SR_clip.tif")
# all_files = preprocessing.preprocess_scenes(files, outpath = "/home/ariane/Documents/PlanetScope/Siguas/L1B/group4/", bandNr = 2)

# #all_files = glob.glob("./MinaPurna/L1B/*_b2.tif")
#reference = "./MinaPurna/L1B/20220823_133636_33_2465_1B_AnalyticMS_b2.tif"
#df = preprocessing.generate_matchfile(all_files, reference, checkOverlap = False, refPoly = "/home/ariane/Documents/PlanetScope/polygon_DoveC.geojson", minOverlap = 95)


aoi = "/raid-manaslu/amueting/PhD/Project1/ImageTransformation/MinaPurna/mina_purna_aoi.geojson"
#ul_lon, ul_lat, xsize, ysize = size_from_aoi(aoi)
ul_lon = -65.71608534638614
ul_lat = -24.276889492561747
xsize = 4150
ysize = 3300
#df = pd.read_csv("/raid-manaslu/amueting/PhD/Project1/ImageTransformation/MinaPurna/L1B/matches_stable.csv")

#correlate stable pairs and subtract 3rd order fit only
# for i in range(len(df)):
#     core.raw_correlate_and_correct(df.ref[i], df.sec[i], demname, amespath, ul_lon, ul_lat, xsize = xsize, ysize = ysize, reduce = 1, first_fit_order = 3, ext = "_Err", overwrite = False, plot = False)

# files = glob.glob("/home/ariane/Documents/PlanetScope/MinaPurna/L1B/stereo/bm_ck35/*clip_dx_corrected.tif")
# core.estimate_topo_signal(files, direction = "x", method = "mean", plot = True)

# files = glob.glob("/home/ariane/Documents/PlanetScope/MinaPurna/L1B/stereo/bm_ck35/*clip_dy_corrected.tif")
# core.estimate_topo_signal(files, direction = "y", method = "mean", plot = True)

#correlate unstable pairs
# df = pd.read_csv("/raid-manaslu/amueting/PhD/Project1/ImageTransformation/MinaPurna/L1B/matches.csv")
# df = df.reindex(index=df.index[::-1]).reset_index(drop = True)

# dem_err_x = "/raid-manaslu/amueting/PhD/Project1/ImageTransformation/MinaPurna/L1B/stable/dem_error_dx_mean_bm_ck35.tif"
# dem_err_y = "/raid-manaslu/amueting/PhD/Project1/ImageTransformation/MinaPurna/L1B/stable/dem_error_dy_mean_bm_ck35.tif"
# for i in range(len(df)):
#     core.raw_correlate_and_correct(df.ref[i], df.sec[i], demname, amespath, ul_lon, ul_lat, xsize = xsize, ysize = ysize, dem_err_x = dem_err_x, dem_err_y = dem_err_y, reduce = 1, first_fit_order = 1, ext = "_Err", overwrite = False, plot = False)

#file = preprocessing.build_remapped_match_file_crossref('./MinaPurna/L1B/matches.csv', dt_min = 183)


#correlate L3B data
###################################################################################################################
# df = pd.read_csv("/raid-manaslu/amueting/PhD/Project1/ImageTransformation/Siguas/L1B/group4/all_matches.csv")
# df = df.reindex(index=df.index[::-1]).reset_index(drop = True)

# for i in range(len(df)):
#     path,_ = os.path.split(df.ref[i])
#     id1 = "_".join(df.ref[i].split("/")[-1].split("_")[0:4])
#     id2 = "_".join(df.sec[i].split("/")[-1].split("_")[0:4])

#     prefix = id1 + "_" + id2 + "_L1B"
#     if not os.path.isfile(path+"/stereo/"+prefix+"-F.tif"):

#         stereopath = asp.correlate_asp(amespath, df.ref[i], df.sec[i], prefix = prefix, session = "rpc", sp_mode = 2, method = "asp_bm", nodata_value = None, corr_kernel = 35)
#         asp.clean_asp_files(stereopath, prefix)
    
####################################################################################################################
#stack
# postprocessing.calc_velocity_L3B("/raid-manaslu/amueting/PhD/Project1/ImageTransformation/delMedio/L3B/group1/all_matches.csv", prefixext="L3B")
# postprocessing.calc_velocity_L3B("/raid-manaslu/amueting/PhD/Project1/ImageTransformation/delMedio/L3B/group2/all_matches.csv", prefixext="L3B")
# postprocessing.calc_velocity_L3B("/raid-manaslu/amueting/PhD/Project1/ImageTransformation/delMedio/L3B/group3/all_matches.csv", prefixext="L3B")
# postprocessing.calc_velocity_L3B("/raid-manaslu/amueting/PhD/Project1/ImageTransformation/delMedio/L3B/random/all_matches.csv", prefixext="L3B")

# for group in [2,8,10]:
    # postprocessing.stack_rasters_weightfree(f"/raid-manaslu/amueting/PhD/Project1/ImageTransformation/Siguas/L3B/matches_group{group}.csv", what = "dx", prefixext = "L3B_polyfit")
    # postprocessing.stack_rasters_weightfree(f"/raid-manaslu/amueting/PhD/Project1/ImageTransformation/Siguas/L3B/matches_group{group}.csv", what = "dy", prefixext = "L3B_polyfit")
    # postprocessing.stack_rasters_weightfree(f"/raid-manaslu/amueting/PhD/Project1/ImageTransformation/Siguas/L3B/matches_group{group}.csv", what = "velocity", prefixext = "L3B_polyfit")
    # postprocessing.stack_rasters_weightfree(f"/raid-manaslu/amueting/PhD/Project1/ImageTransformation/Siguas/L3B/matches_group{group}.csv", what = "direction", prefixext = "L3B_polyfit")

# postprocessing.stack_rasters_weightfree("/raid-manaslu/amueting/PhD/Project1/ImageTransformation/delMedio/L3B/random/all_matches.csv", what = "dx")
# postprocessing.stack_rasters_weightfree("/raid-manaslu/amueting/PhD/Project1/ImageTransformation/delMedio/L3B/random/all_matches.csv", what = "dy")
# postprocessing.stack_rasters_weightfree("/raid-manaslu/amueting/PhD/Project1/ImageTransformation/delMedio/L3B/random/all_matches.csv", what = "velocity")

# for g in [1,2,3]:
#     postprocessing.stack_rasters_weightfree(f"/raid-manaslu/amueting/PhD/Project1/ImageTransformation/delMedio/L3B/group{g}/all_matches.csv", what = "dx")
    #postprocessing.stack_rasters_weightfree(f"/raid-manaslu/amueting/PhD/Project1/ImageTransformation/delMedio/L3B/group{g}/all_matches.csv", what = "dy")
    #postprocessing.stack_rasters_weightfree(f"/raid-manaslu/amueting/PhD/Project1/ImageTransformation/delMedio/L3B/group{g}/all_matches.csv", what = "velocity")

# postprocessing.stack_rasters_weightfree("/raid-manaslu/amueting/PhD/Project1/ImageTransformation/delMedio/L3B/matches_group6.csv", what = "dx")
# postprocessing.stack_rasters_weightfree("/raid-manaslu/amueting/PhD/Project1/ImageTransformation/delMedio/L3B/matches_group6.csv", what = "dy")
# postprocessing.stack_rasters_weightfree("/raid-manaslu/amueting/PhD/Project1/ImageTransformation/delMedio/L3B/matches_group6.csv", what = "velocity")

# postprocessing.stack_rasters_weightfree("/raid-manaslu/amueting/PhD/Project1/ImageTransformation/Siguas/L3B/randomPairs/matches_randomPairs.csv", what = "dx", prefixext = "L3B_polyfit")
# postprocessing.stack_rasters_weightfree("/raid-manaslu/amueting/PhD/Project1/ImageTransformation/Siguas/L3B/randomPairs/matches_randomPairs.csv", what = "dy", prefixext = "L3B_polyfit")
# postprocessing.stack_rasters_weightfree("/raid-manaslu/amueting/PhD/Project1/ImageTransformation/Siguas/L3B/randomPairs/matches_randomPairs.csv", what = "velocity", prefixext = "L3B_polyfit")
# postprocessing.stack_rasters_weightfree("/raid-manaslu/amueting/PhD/Project1/ImageTransformation/Siguas/L3B/randomPairs/matches_randomPairs.csv", what = "direction", prefixext = "L3B_polyfit")

#######################################################################################################################



# df = pd.read_csv("./MinaPurna/L1B/matches_remapped_crossref.csv")


# for i in range(len(df)):
#     core.correlate_remapped_img_pairs(df.ref[i], df.sec[i], amespath)


###################################################

# matchfile = "./Siguas/L3B/randomPairs/matches_randomPairs.csv"
# #aoi = "./Siguas/siguas_landslide_outline_utm.geojson"
# aoi = "./delMedio/landslide_mask.geojson"
aoi = "./Siguas/siguas_aoi.geojson"
#demname = "./DEMdata/20220907_140709_64_24a3_20220912_141056_91_2486-DEM_output_FdemnameNASADEM_epsg32720_align_copy.tif"
demname = "./DEMdata/Siguas/final_aligned_PlanetDEM_epsg32718.tif"
# amespath = "/home/ariane/Downloads/StereoPipeline-3.1.1-alpha-2022-10-16-x86_64-Linux/bin/"

# files = glob.glob("./Siguas/L1B/group4/*_b2.tif")
# preprocessing.orthorectify_L1B(files, demname, aoi, epsg = 32718, amespath = amespath, pad = 400)
#matches = preprocessing.match_all(df, path = "./Siguas/L1B/group4/", ext = "_1B_AnalyticMS_b2_clip_mp_clip.tif", dt = 180, checkExistence=True)

    
#out = postprocessing.compare_two_aois(matchfile, aoi, level = 3, prefixext = "L3B")
# aoi = "./delMedio/landslide_mask.geojson"


#for group in ["group1","group2","group3", "random"]:
# for group in ["random"]:
# #     # files = glob.glob(f"./delMedio/L3B/{group}/stereo/*F_polyfit.tif")
# #     # for f in files:
# #     #     os.rename(f, f[:-14]+"_polyfit-F.tif")
#     if group == "random":
#         #matchfile = f"./delMedio/L3B/{group}/all_matches_no2803.csv"
#         matchfile = "./Siguas/L3B/randomPairs/matches_randomPairs.csv"
#     else:
#         matchfile = f"./delMedio/L3B/{group}/all_matches.csv"
#     core.apply_polyfit(matchfile, prefix_ext= "L3B", order = 2, demname = demname, plimlow = 20, plimup = 80)
#     postprocessing.calc_velocity_wrapper(matchfile, prefixext="L3B_polyfit", overwrite= True,  medShift = False)
#     postprocessing.stack_rasters_weightfree(matchfile, what = "velocity", prefixext = "L3B_polyfit")
#     postprocessing.stack_rasters_weightfree(matchfile, what = "dx", prefixext = "L3B_polyfit")
#     postprocessing.stack_rasters_weightfree(matchfile, what = "dy", prefixext = "L3B_polyfit")
#     postprocessing.stack_rasters_weightfree(matchfile, what = "direction", prefixext = "L3B_polyfit")

#     #postprocessing.get_stats_in_aoi(matchfile, level = 3, aoi = aoi,  prefixext = "L3B", max_dt = 10000, take_velocity = True)

# matchfile = "./delMedio/L1B/group4/all_matches_no2803.csv"
# postprocessing.stack_rasters_weightfree(matchfile, prefixext="_L1B_polyfit", what = "velocity")
# postprocessing.stack_rasters_weightfree(matchfile, prefixext="_L1B_polyfit", what = "dx", medShift = False)
# postprocessing.stack_rasters_weightfree(matchfile, prefixext="_L1B_polyfit", what = "dy", medShift = False)

#matchfile = "./delMedio/L3B/random/all_matches.csv"
# postprocessing.calc_velocity_L3B(matchfile, prefixext="L3B_polyfit", medShift = False)
# postprocessing.stack_rasters_weightfree(matchfile, prefixext="L3B_polyfit", what = "velocity")
#postprocessing.stack_rasters_weightfree(matchfile, prefixext="L3B_polyfit", what = "dx", medShift = False)
# postprocessing.stack_rasters_weightfree(matchfile, prefixext="L3B_polyfit", what = "dy", medShift = False)
#matchfile = "./delMedio/L3B/random/all_matches_no2803.csv"
# postprocessing.stack_rasters_weightfree(matchfile, prefixext="L3B_polyfit", what = "velocity")
#postprocessing.stack_rasters_weightfree(matchfile, prefixext="L3B_polyfit", what = "dx", medShift = False)
# postprocessing.stack_rasters_weightfree(matchfile, prefixext="L3B_polyfit", what = "dy", medShift = False)


#postprocessing.calc_velocity_L3B(matchfile, prefixext="_L1B_polyfit", medShift = False)

##################compare std to sun position#######################################
# out = pd.DataFrame()

# for group in ["group1", "group2", "group3", "random"]:
#     matchfile = f"./delMedio/L3B/{group}/all_matches.csv"
#     df = pd.read_csv(matchfile)
#     df["type"] = group
#     out = pd.concat([out,df])

    
# out.to_csv("./delMedio/L3B/allpairs.csv")

# postprocessing.get_std_iqr("./delMedio/L3B/allpairs.csv", prefixext = "L3B")
# postprocessing.get_std_iqr("./Siguas/L3B/allpairs.csv", prefixext = "L3B")
# infodf = df.copy()
# df = pd.read_csv("delMedio/L3B/allpairs_offset_std.csv")
# merged_df = df.merge(infodf[['ids', 'sun_azimuth', 'sun_elevation']], left_on='id_ref', right_on='ids', how='left')
# merged_df = merged_df.merge(infodf[['ids', 'sun_azimuth', 'sun_elevation']], left_on='id_sec', right_on='ids', how='left')
# merged_df.to_csv("delMedio/L3B/allpairs_suninfo.csv", index = False)
#######################################################################################


    # core.apply_polyfit(matchfile, prefix_ext= "L3B", order = 2, demname = demname)
    #postprocessing.calc_velocity_wrapper(matchfile, prefixext="L3B_polyfit", overwrite= True,  medShift = False)
    

# matchfile = f"./Siguas/L3B/randomPairs/matches_randomPairs.csv"    
# postprocessing.calc_velocity_wrapper(matchfile, prefixext="L3B_polyfit", overwrite= True,  medShift = False)

aoi = "delMedio/landslide_mask_epsg32718.geojson"
# aoi = "Siguas/landslide_mask_upper.geojson"
# #matchfile = "Siguas/L3B/allpairs.csv"
matchfile = "delMedio/L1B/group4/all_matches_no2803.csv"

#postprocessing.get_stats_in_aoi(matchfile, aoi = aoi,  prefixext = "_L1B_polyfit", max_dt = 10000, take_velocity = True)

# matchfile = "./delMedio/L1B/group4/all_matches_no2803.csv"
#core.apply_polyfit(matchfile, prefix_ext= "_L1B", order = 2, demname = demname)

#matchfile = "./delMedio/L1B/group4/all_matches_no2803.csv"
#postprocessing.calc_velocity_wrapper(matchfile, prefixext="_L1B_polyfit",overwrite = True, medShift = False)
# postprocessing.stack_rasters_weightfree(matchfile, what = "dx", prefixext = "_L1B_polyfit")
# postprocessing.stack_rasters_weightfree(matchfile, what = "dy", prefixext = "_L1B_polyfit")
# postprocessing.stack_rasters_weightfree(matchfile, what = "velocity", prefixext = "_L1B_polyfit")
# postprocessing.stack_rasters_weightfree(matchfile, what = "direction", prefixext = "_L1B_polyfit")


#postprocessing.get_stats_in_aoi(matchfile, aoi = aoi,  prefixext = "_L1B_polyfit", max_dt = 10000, take_velocity = True)
matchfile = "./Siguas/L1B/group4/all_matches.csv"
postprocessing.get_stats_for_entire_raster(matchfile, prefixext="_L1B_polyfit", take_velocity=False)
#matchfile = "./delMedio/L3B/random/all_matches_no2803.csv"
matchfile = "./Siguas/L3B/matches_randomPairs.csv"
postprocessing.get_stats_for_entire_raster(matchfile, prefixext="L3B_polyfit", take_velocity=False)
postprocessing.get_stats_for_entire_raster(matchfile, prefixext="L3B", take_velocity=False)

matchfile = "./delMedio/L1B/group4/all_matches_no2803.csv"
postprocessing.get_stats_for_entire_raster(matchfile, prefixext="_L1B_polyfit", take_velocity=False)
matchfile = "./delMedio/L3B/random/all_matches_no2803.csv"
postprocessing.get_stats_for_entire_raster(matchfile, prefixext="L3B_polyfit", take_velocity=False)
postprocessing.get_stats_for_entire_raster(matchfile, prefixext="L3B", take_velocity=False)


#########################################################################################################################

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
# files = glob.glob("/home/ariane/Downloads/delMedio_L1B_may2022_psscene_basic_analytic_udm2/PSScene/*AnalyticMS.tif")
# all_files = preprocessing.preprocess_scenes(files, outpath = "/home/ariane/Documents/PlanetScope/delMedio/L1B/group4/", bandNr = 2)
# all_files = glob.glob("./Dove-C_Jujuy_all/L1B/*_b2.tif")
# reference = "./Dove-C_Jujuy_all/L1B/20190626_140959_103c_1B_Analytic_b2.tif"
# df = preprocessing.generate_matchfile(all_files, reference, checkOverlap = False, refPoly = "/home/ariane/Documents/PlanetScope/polygon_DoveC.geojson", minOverlap = 95)

#after remapping correlate these images

#file = preprocessing.build_remapped_match_file_crossref('/home/ariane/Documents/PlanetScope/SD_Jujuy_Nadir/sd_matches_crossangle.csv', dt_min = 183)
#file = preprocessing.build_remapped_match_file_crossref('/home/ariane/Documents/PlanetScope/Dove-C_Jujuy_all/L1B/matches.csv', dt_min = 183)

#Dove-C
# ul_lon = -65.61782
# ul_lat = -23.88517

# ysize = 1600
# xsize = 3200

#Super Dove
# ul_lon = -65.61782
# ul_lat = -23.88517
# xsize = 3200
# ysize = 2200

# path = "./SD_Jujuy_Nadir/"

# #path = "./Dove-C_Jujuy_all/L1B/"

# amespath = "/raid-manaslu/amueting/StereoPipeline-3.1.1-alpha-2022-07-29-x86_64-Linux/bin/"
# dem = "./data/DEM/EGM96/demcoreg_alos/CopernicusDEM_EasternCordillera_EGM96_clip_AW3D30_NWArg_nuth_x+10.19_y-0.36_z+2.36_align_3m.tif"
# # dem_err_x = f"{path}/stereo/dem_error_dx_mean_mgm.tif"
# # dem_err_y = f"{path}/stereo/dem_error_dy_mean_mgm.tif"
# cutline = "polygon_SuperDove.geojson"
# #cutline = "new_polygon_DoveC.geojson"


# dem_err_x = "./SD_Jujuy_all/stereo/dem_error_dx_mean_bm.tif"
# dem_err_y = "./SD_Jujuy_all/stereo/dem_error_dy_mean_bm.tif"

#df = pd.read_csv(path+"matches.csv")#(file)
#df = pd.read_csv(path+"sd_matches_crossangle.csv")#(file)
#df = df.iloc[13:14].reset_index(drop = True)
# #df = df.reindex(index=df.index[::-1]).reset_index(drop = True)

# df = pd.read_csv(path + "matches_remapped_crossref_mp.csv")
# df = df.reindex(index=df.index[::-1]).reset_index(drop = True)
# df = df.iloc[4:5].reset_index(drop = True)

# for i in range(len(df)):
# #       core.raw_correlate_and_correct(df.ref[i], df.sec[i], dem, amespath, ul_lon, ul_lat, xsize = xsize, ysize = ysize, zgrid = "estimate", dem_err_x = dem_err_x, dem_err_y = dem_err_y, reduce = 5, first_fit_order = 1, ext = "_Err", overwrite = False, plot = False)

#     core.mp_correlate(df.ref[i], df.sec[i], dem, amespath, crop_before_mp = False, cutline = cutline, plot = False)

# #path = "./SD_Jujuy_Nadir/"

# df = pd.read_csv(path+"matches_remapped_crossref.csv")
# df = df.reindex(index=df.index[::-1]).reset_index(drop = True)

# for i in [4]:#range(len(df)):
#     core.correlate_remapped_img_pairs(df.ref[i], df.sec[i], amespath)

# aoi = "landslide_mask.geojson"
 
# matchfile = "./SD_Jujuy_Nadir/matches_remapped_crossref.csv"
# img_with_rpc = "./SD_Jujuy_Nadir/20220924_134300_71_2212_1B_AnalyticMS_b2_clip.tif"
# postprocessing.generate_timeline(matchfile, aoi = aoi, take_velocity = True, weigh_by_dt=False)

#postprocessing.stack_rasters(matchfile, take_velocity=True)

#postprocessing.get_stats_for_allpairs(matchfile, take_velocity = True)

    
# matchfile = "./Dove-C_Jujuy_all/L1B/matches_remapped_crossref_mp.csv"
# #matchfile = "./Dove-C_Jujuy_all/L1B/matches_mp_originalRPCs.csv"
# img_with_rpc = "./Dove-C_Jujuy_all/L1B/20190626_140959_103c_1B_Analytic_b2_clip.tif"
## fixed res PSBSD = 3.670189073838629
## fixed res DoveC = 3.91910337761
# path = "./SD_Jujuy_Nadir/"
# matchfile = path+"matches_remapped_crossref.csv"
# img_with_rpc = "./SD_Jujuy_Nadir/20220924_134300_71_2212_1B_AnalyticMS_b2_clip.tif"
#postprocessing.mapproject_and_calc_velocity(amespath, matchfile, dem, fixed_res = 3, img_with_rpc = img_with_rpc, velocity_only=True)

#postprocessing.generate_timeline(matchfile, aoi = aoi, take_velocity = False, weigh_by_dt=False)
#postprocessing.generate_timeline(matchfile, aoi = aoi, take_velocity = True, weigh_by_dt=False)

#postprocessing.get_stats_for_allpairs(matchfile, take_velocity = True)


# pad = 5
# xcoord = 2170
# ycoord = 1347


# pad = 5
# xcoord = 2200
# ycoord = 1387

#postprocessing.mapproject_and_calc_velocity(amespath, matchfile, dem, img_with_rpc)
# postprocessing.generate_timeline(matchfile, xcoord = xcoord, ycoord = ycoord, pad = pad, take_velocity = True, plot = True)
#postprocessing.generate_timeline(matchfile, aoi = aoi, take_velocity = True)
# postprocessing.stack_rasters(matchfile, take_velocity=False)
# postprocessing.calculate_average_direction(path+"average_dx_dy_mp.tif")



#########################DEM generation#####################################
# aoi = './DEMgen/siguas_dem_aoi.geojson'
# # # aoi = './DEMgen/dem_aoi.geojson'

# # # img1 = "./DEMgen/20220907_140709_64_24a3_1B_AnalyticMS_b2.tif"            
# # # img2 = "./DEMgen/20220916_141026_69_2461_1B_AnalyticMS_b2.tif" 
# # # img2 = "./DEMgen/20220912_141056_91_2486_1B_AnalyticMS_b2.tif"  
# # # img2 = "./DEMgen/20220908_141037_39_2479_1B_AnalyticMS_b2.tif"

# img1 = "./DEMgen/Siguas/20220702_145351_89_240c_1B_AnalyticMS_b2.tif"
# img2 = "./DEMgen/Siguas/20220706_144107_59_24a3_1B_AnalyticMS_b2.tif"
# # # # #amespath = "/home/ariane/Downloads/StereoPipeline-3.1.1-alpha-2022-10-16-x86_64-Linux/bin/"
# amespath = "/raid-manaslu/amueting/StereoPipeline-3.1.1-alpha-2022-07-29-x86_64-Linux/bin/"
# id1 = get_scene_id(img1)
# id2 = get_scene_id(img2)

# epsg = 32718      
# prefix = f"{id1}_{id2}_ck45"
# refdem = "./DEMdata/Siguas/20220702_145351_89_240c_20220706_144107_59_24a3_ck65-DEM_NASADEM_utm_clip_align_epsg32718_res30_copy_epsg4326.tif"
# # refdem = "./DEMdata/siguas_COP30_wgs.tif"
# # #refdem = "./DEMdata/output_COP30_epsg32720_dem_align/output_COP30_epsg32720_output_NASADEM_epsg32720_nuth_x+15.80_y+2.09_z-1.01_align.tif"
# demcoregpath = "/raid-manaslu/amueting/PhD/Project1/demcoreg/demcoreg/"
# # #potentially need to install pygeotools pip install pygeotools https://github.com/dshean/pygeotools for demcoreg to work
# asp.dem_pipeline(amespath,  img1, img2, refdem, prefix = prefix, aoi = aoi, epsg = 32718, overwrite = True)

#######################DEM alignment###################################

# refdem = "./DEMgen/Siguas/COP30_utm_clip_dem_align/COP30_utm_clip_NASADEM_utm_clip_align.tif"
# demname = "./DEMgen/Siguas/20220702_145351_89_240c_20220706_144107_59_24a3_ck65-DEM.tif"
# amespath = "/home/ariane/Downloads/StereoPipeline-3.1.1-alpha-2022-10-16-x86_64-Linux/bin/"
# path = "/home/ariane/Documents/PlanetScope"
# img1 = path + "/Siguas/L1B/20220707_144112_41_247c_1B_AnalyticMS_b2_clip.tif"  
# img2 = path + "/Siguas/L1B/20220717_143827_66_2470_1B_AnalyticMS_b2_clip.tif"
# core.disparity_based_DEM_alignment(amespath, img1, img2, demname, refdem, epsg = 32718, iterations = 3)
