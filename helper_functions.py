#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 14:01:48 2022

@author: ariane
"""
import rasterio, os, fnmatch, datetime, subprocess, glob, json
import numpy as np
from osgeo import gdal, gdalconst
import pandas as pd
from scipy.ndimage import label, binary_dilation
from pyproj import Transformer, CRS

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
    #replace infinite values with 0. They will fall outside of the DEM and will be assigned the nodata value. 
    xcoords = np.array(xcoords)
    ycoords = np.array(ycoords)
    xcoords[~np.isfinite(xcoords)] = 0
    ycoords[~np.isfinite(ycoords)] = 0

    coords = [(x,y) for x, y in zip(xcoords,ycoords)]
    with rasterio.open(rastername) as src:
        meta = src.meta
        
    epsg = int(meta["crs"]['init'].lstrip('epsg:'))
    #TODO: implement check for EPSG but do not automatically reproject everything to 4326
    # if epsg != 4326:
    #     print("Reprojecting raster to EPSG 4326 ...")
    #     cmd = f"gdalwarp -t_srs epsg:4326 -r bilinear -overwrite -co COMPRESS=DEFLATE -co PREDICTOR=2 -co ZLEVEL=9 {rastername} {rastername[:-4]}_epsg4326.tif"
    #     subprocess.run(cmd, shell = True)
    #     rastername = f"{rastername[:-4]}_epsg4326.tif"

    src = rasterio.open(rastername)
    
    out = np.zeros([len(xcoords),meta["count"]])
    for count in range(meta["count"]):
        out[:,count] = [x[count] for x in src.sample(coords)]
        
    if meta["count"] == 1:
        out = out.flatten()
    return out

def min_max_scaler(x):
    if len(x)>1:
        return (x-np.nanmin(x))/(np.nanmax(x)-np.nanmin(x))
    elif len(x) == 1: 
        return np.array([1])
    else: 
        return np.array([])
    
def fixed_val_scaler(x, xmin, xmax):
    return (x-xmin)/(xmax-xmin)

    
def size_from_aoi(aoi, gsd, epsg):

    #AOI has to be a rectangle 
    with open(aoi) as f:
         gj = json.load(f)
         
    coords = list(set([tuple(c) for c in gj["features"][0]["geometry"]["coordinates"][0]]))
    
    if len(coords) !=4:
        print("AOI has to be a rectangle!")
        #return
    #get corner coords of polygon by splitting in upper and lower half
    yc = np.unique([c[1] for c in coords])

    upper = [c for c in coords if c[1]>=yc[-2]]
    upper.sort()
    upperleft = upper[0]
    upperright = upper[1]
    
    lower = [c for c in coords if c[1]<yc[-2]]
    lower.sort()
    lowerleft = lower[0]
    
    ul_lon = upperleft[0]
    ul_lat = upperleft[1]

    #calculate distances (in m) from corner to corner to get size of the aoi
    sizecoords = [lowerleft, upperleft, upperright]
    transformer = Transformer.from_crs(CRS("EPSG:4326"), CRS("EPSG:"+str(epsg)), always_xy=True)  #! need always_xy = True otherwise does strange things
    coords_proj = [transformer.transform(c[0],c[1]) for c in sizecoords]
    
    dists = []
    for i in range(len(coords_proj)-1):
        d = np.sqrt((coords_proj[i+1][0]-coords_proj[i][0])**2+(coords_proj[i+1][1]-coords_proj[i][1])**2)
        dists.append(d)
    
    xsize = int(dists[1]/gsd)
    ysize = int(dists[0]/gsd)

    return ul_lon, ul_lat, xsize, ysize

def warp(img, epsg, res = None):
    
    if res is not None: 
        outname = f"{img[:-4]}_epsg{epsg}_res{res}.tif"
        cmd = f"gdalwarp -t_srs EPSG:{epsg} -tr {res} {res} -co COMPRESS=DEFLATE -r bilinear -overwrite -co ZLEVEL=9 -co PREDICTOR=2 {img} {outname}"
    else: #let gdal guess resolution
        outname = f"{img[:-4]}_epsg{epsg}.tif"
        cmd = f"gdalwarp -t_srs EPSG:{epsg} -co COMPRESS=DEFLATE -r bilinear -overwrite -co ZLEVEL=9 -co PREDICTOR=2 {img} {outname}"
    subprocess.run(cmd, shell = True)
    return outname

def clip_raw(img, ul_lon, ul_lat, xsize, ysize, demname):
    #gets approximate dimensions for clipping raw data from aoi
    #check that CRS = epsg:4326 otherwise the RPC projection will result in wrong estimages
    epsg = get_epsg(demname)
    
    if epsg != 4326:
        print("Reprojecting the given DEM to geographic CRS...")
        demname = warp(demname, epsg = 4326)
    
    ds = gdal.Open(img)
    #create transformer
    tr = gdal.Transformer(ds, None, ["METHOD=RPC",f"RPC_DEM={demname}"])
    _,pnt = tr.TransformPoint(1, ul_lon, ul_lat)
    ds = tr = None
    cmd = f"gdal_translate -co COMPRESS=DEFLATE -co ZLEVEL=9 -co PREDICTOR=2 -srcwin {pnt[0]} {pnt[1]} {xsize} {ysize} {img} {img[:-4]}_clip.tif"
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
    
    #determine processing level of scenes
    if "_1B_" in fn:
        level = 1
    elif "_3B_" in fn:
        level = 3
    else:
        print("Could not determine processing level of the data. Make sure that either _1B_ or _3B_ is included in the filename of your scene.")
        return
    
    if fn.split("_").index(f"{level}B") == 4: #PSB.SD case
        scene_id = "_".join(fn.split("_")[0:4])
    elif fn.split("_").index(f"{level}B") == 3: #PS2 case
        scene_id = "_".join(fn.split("_")[0:3])
    else: 
        print("Couldn't guess the instrument type. Have you modifies filenames?")
        return
    return scene_id
        
def get_date(scene_id):
    #strip the time from th PS scene id
    return datetime.datetime.strptime(scene_id[0:8], "%Y%m%d")

def match_raster_size_and_res(r1, r2):
    epsg = get_epsg(r1)
    xmin, ymin, xmax, ymax = get_extent(r1)
    res = read_transform(r1)[0]
    
    cmd = f"gdalwarp -te {xmin} {ymin} {xmax} {ymax} -t_srs EPSG:{epsg} -tr {res} {res} -r cubic -overwrite -co COMPRESS=DEFLATE -co ZLEVEL=9 -co PREDICTOR=2 {r2} {r2[:-4]}_matched_size.tif"
    result = subprocess.run(cmd, shell = True,  stdout=subprocess.PIPE, stderr=subprocess.PIPE, text = True)
    if result.stderr != "":
        print(result.stderr)
    return f"{r2[:-4]}_matched_size.tif"
