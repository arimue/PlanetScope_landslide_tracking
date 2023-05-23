#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 15:23:13 2023

@author: ariane
"""

import glob, subprocess, os, shutil
from helper_functions import clip_raw, size_from_aoi, clip_mp_cutline

def correlate_asp(amespath, img1, img2, prefix = "run", session = "rpc", sp_mode = 1, method = "asp_bm", nodata_value = None, corr_kernel = 25):
    
    #run ASP stereo correlation in correlator mode using the input parameters
    #provide a path to your asp installation
    
    folder = img1.replace(img1.split("/")[-1], "")
    print(f"Data will be saved under {folder}stereo/")
    
    if method == "asp_bm":
    #this can also be changed to parallel_stereo
        cmd = f"{amespath}stereo {img1} {img2} {folder}stereo/{prefix} --correlator-mode -t {session} --datum Earth --skip-rough-homography --stereo-algorithm {method} --subpixel-mode {sp_mode} --corr-kernel {corr_kernel} {corr_kernel} --subpixel-kernel {corr_kernel+10} {corr_kernel+10} --threads 0" 
        if nodata_value is not None: 
            cmd = f"{cmd} --nodata-value {nodata_value}"
    else:
        print(corr_kernel)
        if (corr_kernel > 9) or (corr_kernel%2 == 0):
            print("Correlation kernel size is not suitable for mgm. Pick an odd kernel size <= 9!")
            return
        cmd = f"{amespath}parallel_stereo {img1} {img2} {folder}stereo/{prefix} --correlator-mode -t {session} --datum Earth --skip-rough-homography --stereo-algorithm {method} --corr-kernel 9 9 --subpixel-mode {sp_mode} --subpixel-kernel {corr_kernel*2+1} {corr_kernel*2+1} --threads 0" 

        if nodata_value is not None: 
            cmd = f"{cmd} --nodata-value {nodata_value}"
            
    subprocess.run(cmd, shell = True)
    
    return f"{folder}stereo/"


def mapproject(amespath, img, dem, img_with_rpc = None, ba_prefix = None, ext = "mp",  epsg = "32720", resolution = 3):
    
    # mapproject raw image data onto DEM
    # requires the image to have RPCs in the header. These can be added with copy_rpc if missing or just simply providing the image with rpc metadata
    #TODO: switch from _single to mapproject although mp single works much better in my case
    
    if img_with_rpc is not None:
        cmd = f"{amespath}mapproject_single {dem} {img} {img_with_rpc} {img[:-4]}_{ext}.tif --threads 0 -t rpc --t_srs epsg:{epsg} --tr {resolution} --no-bigtiff --tif-compress Deflate --nodata-value -9999"
    else: 
        cmd = f"{amespath}mapproject_single {dem} {img} {img[:-4]}_{ext}.tif -t rpc --threads 0 --t_srs epsg:{epsg} --tr {resolution} --no-bigtiff --tif-compress Deflate --nodata-value -9999"
 
    if ba_prefix is not None: 
        cmd = f"{cmd} --bundle-adjust-prefix {ba_prefix}"

    subprocess.run(cmd, shell = True)
    return f"{img[:-4]}_{ext}.tif"
    
def clean_asp_files(path, prefix):
    
    #cleans up behind ASP to remove unneccessary files 
    #will only keep the filtered disparity file ("*-F.tif")
    
    files = glob.glob(f"{path}{prefix}-*")
    disp  = glob.glob(f"{path}{prefix}-F.tif")
    remove = set(files)-set(disp)
    
    for file in remove:
        try:
            os.remove(file)
        except IsADirectoryError: #if parallel_stereo is used, also remove folders
            shutil.rmtree(file)
            
# aoi = './DEMgen/dem_aoi.geojson'
# img1 = "./DEMgen/20220907_140709_64_24a3_1B_AnalyticMS_b2.tif"            
# img2 = "./DEMgen/20220908_141037_39_2479_1B_AnalyticMS_b2.tif"  
# #amespath = "/home/ariane/Downloads/StereoPipeline-3.1.1-alpha-2022-10-16-x86_64-Linux/bin/"
# amespath = "/raid-manaslu/amueting/StereoPipeline-3.1.1-alpha-2022-07-29-x86_64-Linux/bin/"

# epsg = 32720      
# prefix = "run"
# refdem = "./data/DEM/EGM96/demcoreg_alos/CopernicusDEM_EasternCordillera_EGM96_clip_AW3D30_NWArg_nuth_x+10.19_y-0.36_z+2.36_align.tif"


def dem_pipeline(amespath, demcoregpath, img1, img2, refdem, aoi = None, epsg = 32720, prefix = "run"):
    
#TODO: implement buffer for aoi
    if aoi is not None:
        #TODO: implement epsg finder
        #TODO: implement GSD finder
        ul_lon, ul_lat, xsize, ysize = size_from_aoi(aoi, epsg = epsg)
        img1 = clip_raw(img1, ul_lon, ul_lat, xsize, ysize, refdem)
        ul_lon, ul_lat, xsize, ysize = size_from_aoi(aoi, epsg = epsg)
        img2 = clip_raw(img2, ul_lon, ul_lat, xsize, ysize, refdem)
        
    path, fn1 = os.path.split(img1)
    _, fn2 = os.path.split(img2)
        
    cmd = f"{amespath}bundle_adjust -t rpc {img1} {img2} -o {path}/bundle_adjust/{prefix}"
    subprocess.run(cmd, shell = True)
    
    cmd = f"{amespath}stereo {img1} {img2} {path}/stereo_run1/{prefix} -t rpc --datum Earth --bundle-adjust-prefix {path}/bundle_adjust/{prefix} --stereo-algorithm asp_bm --subpixel-mode 2 --threads 0" 
    subprocess.run(cmd, shell = True)
    
    cmd = f"{amespath}point2dem {path}/stereo_run1/{prefix}-PC.tif --tr 90 --t_srs EPSG:{epsg} -o {path}/point2dem_run1/{prefix}" 
    subprocess.run(cmd, shell = True)
    
    #need to use the actual mapproject command, not mapproject_single to keep the rpc information 
    cmd = f"{amespath}mapproject {path}/point2dem_run1/{prefix}-DEM.tif {img1} {img1[:-4]}_mp.tif -t rpc --threads 0 --t_srs epsg:{epsg} --tr 3 --no-bigtiff --tif-compress Deflate --nodata-value -9999"
    subprocess.run(cmd, shell = True)
    cmd = f"{amespath}mapproject {path}/point2dem_run1/{prefix}-DEM.tif {img2} {img2[:-4]}_mp.tif -t rpc --threads 0 --t_srs epsg:{epsg} --tr 3 --no-bigtiff --tif-compress Deflate --nodata-value -9999"
    subprocess.run(cmd, shell = True)
    
    mp1 = img1[:-4]+"_mp.tif"
    mp2 = img2[:-4]+"_mp.tif"
    
    
    #need to copy bundle adjusted files, because the program doesnt find it anymore due to name changes
    shutil.copyfile(f"{path}/bundle_adjust/{prefix}-{fn1[:-4]}.adjust", f"{path}/bundle_adjust/{prefix}-{fn1[:-4]}_mp.adjust")
    shutil.copyfile(f"{path}/bundle_adjust/{prefix}-{fn2[:-4]}.adjust", f"{path}/bundle_adjust/{prefix}-{fn2[:-4]}_mp.adjust")
    
    cmd = f"{amespath}stereo {mp1} {mp2} -t rpcmaprpc --datum Earth --bundle-adjust-prefix {path}/bundle_adjust/{prefix} {path}/stereo_run2/{prefix} {path}/point2dem_run1/{prefix}-DEM.tif --stereo-algorithm asp_bm --subpixel-mode 2 " 
    subprocess.run(cmd, shell = True)
    
    cmd = f"{amespath}point2dem {path}/stereo_run2/{prefix}-PC.tif --tr 30 --t_srs EPSG:{epsg} --dem-hole-fill-len 10 -o {path}/point2dem_run2/{prefix}" 
    subprocess.run(cmd, shell = True)

    cmd = f"{demcoregpath}dem_align.py {refdem} {path}/point2dem_run2/{prefix}-DEM.tif -mode nuth -max_dz 1000 -max_offset 500"
    subprocess.run(cmd, shell = True)
    #TODO: improve max offset constraints (initial guess?) catch errors better so that process doesnt have to start from the beginning, i.e. implement overwrite check


    print("Done!")