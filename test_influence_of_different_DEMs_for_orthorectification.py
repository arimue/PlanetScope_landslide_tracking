#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 23 13:07:18 2023

@author: ariane
"""

#project clipped L1B images on different reference DEMs
import helper_functions as helper
import asp_helper_functions as asp
import os
import core_functions as core

path = "."
img1 = path + "/DEMgen/20220907_140709_64_24a3_1B_AnalyticMS_b2.tif"  
img2 = path + "/DEMgen/test_mapprojection_differences_DEMs/20220924_134300_71_2212_1B_AnalyticMS_b2.tif"
aoi = path + "/delMedio/del_medio_aoi.geojson"
epsg = 32720
refdem = path + "/DEMdata/output_COP30_epsg32720_NASADEMaligned.tif"
amespath = "/raid-manaslu/amueting/StereoPipeline-3.1.1-alpha-2022-07-29-x86_64-Linux/bin/"

#make sure to find aoi with same refdem
ul_lon, ul_lat, xsize, ysize = helper.size_from_aoi(aoi, epsg = epsg)
img1 = helper.clip_raw(img1, ul_lon, ul_lat, xsize+50, ysize+50, refdem)
ul_lon, ul_lat, xsize, ysize = helper.size_from_aoi(aoi, epsg = epsg)
img2 = helper.clip_raw(img2, ul_lon, ul_lat, xsize+400, ysize+500, refdem)


for refdem in [path +"/DEMdata/output_AW3D30_epsg32720_NASADEMaligned.tif", path +"/DEMdata/output_COP30_epsg32720_NASADEMaligned.tif",path +"/DEMdata/output_NASADEM_epsg32720.tif", "./DEMdata/run-DEM_NASADEMaligned.tif"]:

    img2r = core.improve_L1B_geolocation(img1, img2, refdem, order = 3)
    
    mp1 = asp.mapproject(amespath, img1, refdem)
    mp2 = asp.mapproject(amespath, img2r, refdem)
    
    mp1 = helper.clip_mp_cutline(mp1, aoi)
    mp2 = helper.clip_mp_cutline(mp2, aoi)

    id1 = helper.get_scene_id(img1)
    id2 = helper.get_scene_id(img2)
    
    _,demfn = os.path.split(refdem)
    prefix = f"{id1}_{id2}_remapped_3rdorder_{demfn[:-4]}"
    stereopath = asp.correlate_asp(amespath, mp1, mp2, prefix = prefix, session = "rpc", sp_mode = 2, method = "asp_bm", nodata_value = None, corr_kernel = 35)
    asp.clean_asp_files(stereopath, prefix)
    
    
# mp1 = path+"/delMedio/L3B/20220907_140709_64_24a3_3B_AnalyticMS_SR_clip_b2.tif"
# mp2 = path+"/delMedio/L3B/20220924_134300_71_2212_3B_AnalyticMS_SR_clip_b2.tif"
# prefix = "20220907_140709_64_24a3_20220924_134300_71_2212_L3B"
# stereopath = asp.correlate_asp(amespath, mp1, mp2, prefix = prefix, session = "rpc", sp_mode = 2, method = "asp_bm", nodata_value = None, corr_kernel = 35)
# asp.clean_asp_files(stereopath, prefix)