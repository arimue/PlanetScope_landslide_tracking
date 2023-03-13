#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 14:01:48 2022

@author: ariane
"""
import rasterio, os, fnmatch, datetime, subprocess, glob
import numpy as np
from osgeo import gdal, gdalconst
import pandas as pd
from scipy.ndimage import label, binary_dilation
from rpcm.rpc_model import rpc_from_geotiff


def list_files(dir, pattern):
    r = []
    for root, dirs, files in os.walk(dir):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                r.append(os.path.join(root, name))
    return r

#raster stuff
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
    
def get_extent(file):
    meta = read_meta(file)

    xmin = meta["transform"][2]
    ymax = meta["transform"][5]
    
    xmax = xmin + meta["transform"][0] * meta["width"]
    ymin = ymax + meta["transform"][4] * meta["height"]
    
    return xmin, ymin, xmax, ymax
    

def save_file(bands, ref, outname, out_dir= ""):
    #save raste rusing metadata information from a reference file. 
    #get metadata from reference file 
    with rasterio.open(ref) as src:
        meta = src.meta
    meta.update(dtype=rasterio.float32)
    
    meta.update(count= len(bands))

    
    with rasterio.open(f"{out_dir}{outname}", 'w', **meta, compress="DEFLATE") as dst:
        
        for i, band in enumerate(bands): 
            
            dst.write(band.astype(rasterio.float32), i+1)
        
    print(f"I have written {outname}!")
    
def rasterValuesToPoint(xcoords, ycoords, rastername):
    coords = [(x,y) for x, y in zip(xcoords,ycoords)]
    with rasterio.open(rastername) as src:
        meta = src.meta
        
    epsg = int(meta["crs"]['init'].lstrip('epsg:'))
    if epsg != 4326:
        print("Reprojecting raster to EPSG 4326 ...")
        cmd = f"gdalwarp -t_srs epsg:4326 -r bilinear -overwrite -co COMPRESS=DEFLATE -co PREDICTOR=2 -co ZLEVEL=9 {rastername} {rastername[:-4]}_epsg4326.tif"
        subprocess.run(cmd, shell = True)
        rastername = f"{rastername[:-4]}_epsg4326.tif"

    src = rasterio.open(rastername)
    
    out = np.zeros([len(xcoords),meta["count"]])
    for count in range(meta["count"]):
        out[:,count] = [x[count] for x in src.sample(coords)]
        
    if meta["count"] == 1:
        out = out.flatten()
    return out

def min_max_scaler(x):
    if len(x)>1:
        return (x-min(x))/(max(x)-min(x))
    elif len(x) == 1: 
        return np.array([1])
    else: 
        return np.array([])
    
def impute(arr, max_fillsize = 1000):
    #fill holes in disparity raster with mean of surroundings
    #after https://stackoverflow.com/questions/41550979/fill-holes-with-majority-of-surrounding-values-python
    imputed_array = np.copy(arr)

    mask = np.isnan(arr)
    labels, count = label(mask) #label holes individually
    
    for idx in range(1, count + 1):
        hole = labels == idx #select current hole
        if hole.sum() <= max_fillsize: #make sure to not fill the borders. These can be filtered as they include a large number of nan values
            surrounding_values = arr[binary_dilation(hole) & ~hole] #enlarge hole with dilation to get surroundings (those who are pixels in padded but not actual hole)
            #fill with mean of surroundings
            imputed_array[hole] = np.nanmean(surrounding_values)

    return imputed_array

def clip_raw(img, ul_lon, ul_lat, xsize, ysize, demname):
    rpc = rpc_from_geotiff(img)
    x,y = rpc.projection([ul_lon], [ul_lat], rasterValuesToPoint([ul_lon], [ul_lat], demname))
  
    # #alternative gdal
    # ds = gdal.Open(img)
    # #create transformer
    # tr = gdal.Transformer(ds, None, ["METHOD=RPC"])
    # _,pnt = tr.TransformPoint(1, ul_lon, ul_lat, rasterValuesToPoint([ul_lon], [ul_lat], demname)[0])
    # ds = tr = None
    cmd = f"gdal_translate -co COMPRESS=DEFLATE -co ZLEVEL=9 -co PREDICTOR=2 -srcwin {x[0]} {y[0]} {xsize} {ysize} {img} {img[:-4]}_clip.tif"
    result = subprocess.run(cmd, shell = True,  stdout=subprocess.PIPE, stderr=subprocess.PIPE, text = True)
    if result.stderr != "":
        print(result.stderr)
    return f"{img[:-4]}_clip.tif"

def clip_mp_projwin(img, ul_lon, ul_lat, xsize, ysize):
    #clip mapprojected images (no transformation to image_coords required)

    cmd = f"gdal_translate -co COMPRESS=DEFLATE -co ZLEVEL=9 -co PREDICTOR=2 -projwin {ul_lon} {ul_lat} {xsize} {ysize} {img} {img[:-4]}_clip.tif"
    result = subprocess.run(cmd, shell = True,  stdout=subprocess.PIPE, stderr=subprocess.PIPE, text = True)
    if result.stderr != "":
        print(result.stderr)
    return f"{img[:-4]}_clip.tif"

def clip_mp_cutline(img, cutline):
    #clip mapprojected images (no transformation to image_coords required)
    res = read_transform(img)[0]
    cmd = f"gdalwarp -co COMPRESS=DEFLATE -co ZLEVEL=9 -co PREDICTOR=2 -cutline {cutline} -overwrite -crop_to_cutline -tr {res} {res} {img} {img[:-4]}_clip.tif"
    result = subprocess.run(cmd, shell = True,  stdout=subprocess.PIPE, stderr=subprocess.PIPE, text = True)
    if result.stderr != "":
        print(result.stderr)
    return f"{img[:-4]}_clip.tif"

def copy_rpcs(rpc_fn, non_rpc_fn):
    #copy RPC header from one file to the other
    #after  https://github.com/uw-cryo/skysat_stereo/blob/31508598757f1b9be3a85b381fea50ccf9d398ae/skysat_stereo/skysat.py
    rpc_img = gdal.Open(rpc_fn, gdalconst.GA_ReadOnly)
    non_rpc_img = gdal.Open(non_rpc_fn, gdalconst.GA_Update)
    rpc_data = rpc_img.GetMetadata('RPC')
    non_rpc_img.SetMetadata(rpc_data,'RPC')

    del(rpc_img)
    del(non_rpc_img)
    
#PS scene specific
def get_scene_id(fn):
    #extract the scene id from a PS scene filename
    #assumes the filename still begins with the scene ID (should be default when downloading data)
    
    #make sure to remove the path if still part of the image
    _, fn = os.path.split(fn)
    if fn.split("_").index("1B") == 4: #PSB.SD case
        scene_id = "_".join(fn.split("_")[0:4])
    elif fn.split("_").index("1B") == 3: #PS2 case
        scene_id = "_".join(fn.split("_")[0:3])
    else: 
        print("Couldn't guess the instrument type. Have you modifies filenames?")
        return
    return scene_id
        
def get_date(scene_id):
    #strip the time from th PS scene id
    return datetime.datetime.strptime(scene_id[0:8], "%Y%m%d")

