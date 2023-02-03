#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 14:01:48 2022

@author: ariane
"""
import rasterio, os, fnmatch

def list_files(dir, pattern):
    r = []
    for root, dirs, files in os.walk(dir):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                r.append(os.path.join(root, name))
    return r

def read_file(file, b=1):
    with rasterio.open(file) as src:
        return(src.read(b))

def read_transform(file):
    with rasterio.open(file) as src:
        return(src.transform)
    
def read_meta(file):
    with rasterio.open(file) as src:
        meta = src.meta
        return meta

def get_epsg(file):
    meta = read_meta(file)
    if meta["crs"].is_epsg_code:
        code = int(meta["crs"]['init'].lstrip('epsg:'))
        return code
    else:
        print("No EPSG found. Check your data.")
        return
    
def get_corners(meta):
    
    xmin = meta["transform"][2]
    ymax = meta["transform"][5]
    
    xmax = xmin + meta["transform"][0] * meta["width"]
    ymin = ymax + meta["transform"][4] * meta["height"]
    
    return xmin, ymin, xmax, ymax
    
# save arrays to array
def save_file(img, ref, outname, out_dir= "", b2 = None, b3 = None, b4 = None, extra = ""):
    #get metadata from reference file 
    with rasterio.open(ref) as src:
        meta = src.meta
    meta.update(dtype=rasterio.float32)
    
    meta.update(count=(sum(x is not None for x in [img, b2, b3, b4])))

    
    with rasterio.open(f"{out_dir}{outname}", 'w', **meta) as dst:
        dst.write(img.astype(rasterio.float32), 1)
        
        #optionally create additional bands
        if b2 is not None:
            dst.write(b2.astype(rasterio.float32), 2)
        if b3 is not None:
            dst.write(b3.astype(rasterio.float32), 3)
        if b4 is not None:
            dst.write(b4.astype(rasterio.float32), 4)
            
            
    print(f"I have written {outname}!")
    

