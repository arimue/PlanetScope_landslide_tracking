#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 15:23:13 2023

@author: ariane
"""

import glob, subprocess, os, shutil
from helper_functions import clip_raw, size_from_aoi, clip_mp_cutline, get_scene_id, get_epsg, warp
import pandas as pd

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


def mapproject(amespath, img, dem, epsg, img_with_rpc = None, ba_prefix = None, ext = "mp", resolution = 3):
    
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
            

def dem_pipeline(amespath, img1, img2, refdem, aoi = None, epsg = 32720, prefix = None, overwrite = False):
    
    #TODO implement check that the epsg is always a projected EPSG
    if prefix is None: 
        id1 = get_scene_id(img1)
        id2 = get_scene_id(img2)

        prefix = f"{id1}_{id2}"
        
    print(f"All outputs will be saved with the prefix {prefix}.")
#TODO: implement buffer for aoi
    if aoi is not None:
        #TODO: implement epsg finder
        #TODO: implement GSD finder
        ul_lon, ul_lat, xsize, ysize = size_from_aoi(aoi, epsg = epsg, gsd = 4)
        img1 = clip_raw(img1, ul_lon, ul_lat, xsize, ysize, refdem)
        ul_lon, ul_lat, xsize, ysize = size_from_aoi(aoi, epsg = epsg, gsd = 4)
        img2 = clip_raw(img2, ul_lon, ul_lat, xsize, ysize, refdem)
        
    path, fn1 = os.path.split(img1)
    _, fn2 = os.path.split(img2)
        
    if not (os.path.isfile(f"bundle_adjust/{prefix}-{fn1[:-4]}.adjust") and os.path.isfile(f"bundle_adjust/{prefix}-{fn2[:-4]}.adjust")) or overwrite:
        cmd = f"{amespath}bundle_adjust -t rpc {img1} {img2} -o {path}/bundle_adjust/{prefix}"
        subprocess.run(cmd, shell = True)
    else:
        print("Using existing bundle adjustment files.")
        
    if not os.path.isfile(f"{path}/stereo_run1/{prefix}-PC.tif") or overwrite:
        cmd = f"{amespath}stereo {img1} {img2} {path}/stereo_run1/{prefix} -t rpc --datum Earth --bundle-adjust-prefix {path}/bundle_adjust/{prefix} --stereo-algorithm asp_bm --subpixel-mode 2 --threads 0 --corr-kernel 65 65 --subpixel-kernel 75 75" 
        subprocess.run(cmd, shell = True)
    else:
        print(f"Using triangulated points from existing file {path}/stereo_run1/{prefix}-PC.tif")
    
    if not os.path.isfile(f"{path}/point2dem_run1/{prefix}-DEM.tif") or overwrite:
        cmd = f"{amespath}point2dem {path}/stereo_run1/{prefix}-PC.tif --tr 90 --t_srs EPSG:{epsg} -o {path}/point2dem_run1/{prefix}" 
        subprocess.run(cmd, shell = True)
    else:
        print(f"Using existing DEM {path}/point2dem_run1/{prefix}-DEM.tif")
    
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
    
    if not os.path.isfile(f"{path}/stereo_run2/{prefix}-PC.tif") or overwrite:
        cmd = f"{amespath}stereo {mp1} {mp2} -t rpcmaprpc --datum Earth --bundle-adjust-prefix {path}/bundle_adjust/{prefix} {path}/stereo_run2/{prefix} {path}/point2dem_run1/{prefix}-DEM.tif --stereo-algorithm asp_bm --subpixel-mode 2 --corr-kernel 65 65 --subpixel-kernel 75 75" 
        subprocess.run(cmd, shell = True)
    else:
        print(f"Using triangulated points from existing file {path}/stereo_run2/{prefix}-PC.tif")
    
    if not os.path.isfile(f"{path}/point2dem_run2/{prefix}-DEM.tif") or overwrite:
        cmd = f"{amespath}point2dem {path}/stereo_run2/{prefix}-PC.tif --tr 30 --t_srs EPSG:{epsg} --dem-hole-fill-len 10 -o {path}/point2dem_run2/{prefix}" 
        subprocess.run(cmd, shell = True)
    else:
        print(f"Using existing DEM {path}/point2dem_run2/{prefix}-DEM.tif")
        
        
    #alignment with demcoreg. Sometimes not working so well, so I prefer to use the disparity based alignment approach (see core functions)
    # print(f"Aligning DEM to {refdem}...")
    
    # #RefDEM will need to be in UTM coordinates
    
    # epsg = get_epsg(refdem)
    
    
    # if epsg == 4326:
    #     print("Reprojecting the reference DEM to a projected CRS...")
    #     refdem = warp(refdem, epsg = epsg)
        
    # cmd = f"{demcoregpath}dem_align.py {refdem} {path}/point2dem_run2/{prefix}-DEM.tif -mode nuth -max_dz 1000 -max_offset 500"
    # subprocess.run(cmd, shell = True)
    #TODO: improve max offset constraints (initial guess?) catch errors better so that process doesnt have to start from the beginning, i.e. implement overwrite check


    print("Done!")
    
    
def image_align_asp(amespath, img1, img2, prefix = None):
    
    #run ASP image_align with disparity derived from running the correlation
    
    folder = img1.replace(img1.split("/")[-1], "")
    print(f"Data will be saved under {folder}image_align/")
    if prefix: 
        cmd = f"{amespath}image_align {img1} {img2} -o {img2[:-4]}_aligned.tif --output-prefix {folder}image_align/{prefix} --alignment-transform affine --disparity-params '{folder}stereo/{prefix}-F.tif 10000' --inlier-threshold 100" 
    else:
        cmd = f"{amespath}image_align {img1} {img2} -o {img2[:-4]}_aligned.tif --output-prefix {folder}image_align/{prefix} --alignment-transform affine  --inlier-threshold 100" 

    subprocess.run(cmd, shell = True)

def parse_match_asp(amespath, img1, img2, prefix = "run"):
    
    #turn a .match file (output from image_align) into readable .txt format 
    
    folder = img1.replace(img1.split("/")[-1], "")+"image_align/"
    matchfile = glob.glob(f"{folder}{prefix}-*-clean.match")
    if len(matchfile)>1:
        print("More that one matching file found. Please check if prefixes were used more that once...")
        return
    matchfile = matchfile[0]
    cmd = f"python {amespath}parse_match_file.py {matchfile} {matchfile[:-6]}.txt"
    subprocess.run(cmd, shell = True)
    return f"{matchfile[:-6]}.txt"

def read_match(matchfile):
    
    #convert a match.txt file to better readable df 
    
    df = pd.read_csv(matchfile, skiprows = 1, header = None, sep = " ")
    nrIPs = pd.read_csv(matchfile, nrows = 1, header = None, sep = " ")

    df1 = df.head(nrIPs[0][0]).reset_index(drop = True)
    df2 = df.tail(nrIPs[1][0]).reset_index(drop = True)

    df = pd.DataFrame({"x_img1":df1[0], "y_img1":df1[1],"x_img2":df2[0], "y_img2":df2[1]})
    
    return df