#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 15:50:14 2023

@author: ariane
"""



#project clipped L1B images on different reference DEMs
import helper_functions as helper
import asp_helper_functions as asp
import os
import core_functions as core

path = "."
#path = "/home/ariane/Documents/PlanetScope"
img1 = path + "/Siguas/L1B/20220707_144112_41_247c_1B_AnalyticMS_b2.tif"  
img2 = path + "/Siguas/L1B/20220717_143827_66_2470_1B_AnalyticMS_b2.tif"
aoi = path + "/Siguas/siguas_aoi.geojson"
epsg = 32718
refdem = path + "/DEMdata/Siguas/output_NASADEM_clip_output_COP30_clip_nuth_x-16.83_y-2.02_z+3.65_align.tif"
amespath = "/raid-manaslu/amueting/StereoPipeline-3.1.1-alpha-2022-07-29-x86_64-Linux/bin/"
#amespath = "/home/ariane/Downloads/StereoPipeline-3.1.1-alpha-2022-10-16-x86_64-Linux/bin/"

#make sure to find aoi with same refdem
ul_lon, ul_lat, xsize, ysize = helper.size_from_aoi(aoi, epsg = epsg)
img1 = helper.clip_raw(img1, ul_lon, ul_lat, xsize, ysize, refdem)
ul_lon, ul_lat, xsize, ysize = helper.size_from_aoi(aoi, epsg = epsg)
img2 = helper.clip_raw(img2, ul_lon, ul_lat, xsize, ysize, refdem)

#[path +"/DEMdata/Siguas/20220702_145351_89_240c_20220706_144107_59_24a3_ck65-DEM_NASADEM_utm_clip_align_epsg32718_res30.tif"]:
for refdem in [path +"/DEMdata/Siguas/20220702_145351_89_240c_20220706_144107_59_24a3_ck65-DEM_NASADEM_utm_clip_align_epsg32718_res30.tif"]:#[path +"/DEMdata/Siguas/COP30_utm_clip_NASADEM_utm_clip_align.tif"]:

    #img2r, newdem = core.improve_L1B_geolocation(amespath, img1, img2, refdem, epsg = epsg, order = 2, add_elev = True)
    
    # mp1 = asp.mapproject(amespath, img1, newdem)
    # mp2 = asp.mapproject(amespath, img2r, newdem)
    mp1 = asp.mapproject(amespath, img1, refdem)
    mp2 = asp.mapproject(amespath, img2, refdem)
    
    mp1 = helper.clip_mp_cutline(mp1, aoi)
    mp2 = helper.clip_mp_cutline(mp2, aoi)

    id1 = helper.get_scene_id(img1)
    id2 = helper.get_scene_id(img2)
    
    _,demfn = os.path.split(refdem)
#    _,demfn = os.path.split(newdem)

    prefix = f"{id1}_{id2}_noshift_{demfn[:-4]}"
    stereopath = asp.correlate_asp(amespath, mp1, mp2, prefix = prefix, session = "rpc", sp_mode = 2, method = "asp_bm", nodata_value = None, corr_kernel = 35)
    asp.clean_asp_files(stereopath, prefix)
    
    
# img1 = path + "/DEMgen/stereo/20220907_140709_64_24a3_1B_AnalyticMS_b2_clip.tif"  
# img2 = path + "/DEMgen/stereo/20220924_134300_71_2212_1B_AnalyticMS_b2_clip.tif"
# mp1 = path+"/delMedio/L3B/20220907_140709_64_24a3_3B_AnalyticMS_SR_clip_b2.tif"
# mp2 = path+"/delMedio/L3B/20220924_134300_71_2212_3B_AnalyticMS_SR_clip_b2.tif"
# prefix = "20220907_140709_64_24a3_20220924_134300_71_2212_L3B"
# stereopath = asp.correlate_asp(amespath, mp1, mp2, prefix = prefix, session = "rpc", sp_mode = 2, method = "asp_bm", nodata_value = None, corr_kernel = 35)
# asp.clean_asp_files(stereopath, prefix)