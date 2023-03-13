#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 15:23:13 2023

@author: ariane
"""

import glob, subprocess, os, shutil

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
        if (corr_kernel > 9) or (corr_kernel%2 != 0):
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