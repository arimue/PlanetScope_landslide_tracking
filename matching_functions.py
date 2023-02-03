#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 15:07:02 2022

@author: ariane
"""
import os
import pandas as pd
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import rasterio, subprocess
from rpcm.rpc_model import rpc_from_geotiff
import glob, shutil
import json
from shapely.geometry import Polygon, Point
from mpl_toolkits.axes_grid1 import make_axes_locatable
from basic_functions import read_file, read_meta


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

def isolateBand(img, bandNr=2):
    out_dir, img_fn = os.path.split(img)
    if not os.path.exists(f"{out_dir}/b{bandNr}/"):
        print(f"Generating directory {out_dir}/b{bandNr}/")
        os.makedirs(f"{out_dir}/b{bandNr}/")
    out_img = f"{out_dir}/b{bandNr}/{img_fn[:-4]}_b{bandNr}.tif"
    cmd = f"gdal_translate -co COMPRESS=DEFLATE -co ZLEVEL=9 -co PREDICTOR=2 -b {bandNr} {img} {out_img}"
    subprocess.run(cmd, shell = True,  stdout=subprocess.PIPE, stderr=subprocess.PIPE, text = True)
    return out_img

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

def clip_mp(img, ul_lon, ul_lat, xsize, ysize, demname):

    cmd = f"gdal_translate -co COMPRESS=DEFLATE -co ZLEVEL=9 -co PREDICTOR=2 -projwin {ul_lon} {ul_lat} {xsize} {ysize} {img} {img[:-4]}_clip.tif"
    result = subprocess.run(cmd, shell = True,  stdout=subprocess.PIPE, stderr=subprocess.PIPE, text = True)
    if result.stderr != "":
        print(result.stderr)
    return f"{img[:-4]}_clip.tif"

def get_vlims(x):
    plow = np.nanpercentile(x, 2)
    pup = np.nanpercentile(x, 98)
    return np.max([abs(plow), abs(pup)])

def plot_offset_results(files, band = 1, width = 7.5, height = 5, outfile = "", meanshift = False):
    ysize = int(np.ceil(len(files)/3))
    xsize = int(np.ceil(len(files)/ysize))
        
    fig, ax = plt.subplots(ysize, xsize, figsize = (width*xsize, height*ysize))
    k = 0
    for i in range(ysize):
        for j in range(xsize):
            try:
                _,fn = os.path.split(files[k])
                ident = fn.removesuffix('-F.tif')
                dat = read_file(files[k], b=band)
                meta = read_meta(files[k])
                #check if mask exists
                if meta["count"] ==3:   
                    mask = read_file(files[k], b = 3)
                    dat[mask == 0] = np.nan
                #remove meanshift
                if meanshift:
                    dat = dat - np.nanmean(dat)
                p = ax[i,j].imshow(dat, vmin = -2, vmax = 2, cmap = "coolwarm")#vmin = -get_vlims(dat), vmax = get_vlims(dat)
                ax[i,j].set_title(f"{ident}", fontsize=11)
                divider = make_axes_locatable(ax[i,j])
                cax = divider.append_axes("right", size="5%", pad=0.05)
                fig.colorbar(p, cax=cax, label='Offset [pix]')  
                k+=1
            except IndexError:
                pass
    plt.tight_layout()
    if outfile != "":
        plt.savefig(outfile, dpi = 300)
    plt.show()
    
    
def plot_single_results(fnX, fnY):
    
    dx = read_file(fnX, 1)
    dy = read_file(fnY, 1)
    
    fig, ax = plt.subplots(2,2, figsize = (14,8))
    ax[0,0].imshow(dx, vmin = -2, vmax = 2, cmap = "coolwarm")
    ax[0,1].imshow(dy, vmin = -2, vmax = 2, cmap = "coolwarm")
    ax[1,0].hist(dx.flatten(), bins = 100, range = (-4,4))
    ax[1,0].axvline(0, color="black", linestyle="--")
    ax[1,1].hist(dy.flatten(), bins = 100, range = (-4,4))
    ax[1,1].axvline(0, color="black", linestyle="--")

    plt.show()


def get_matching_hillshade(img1, img2, searchfile, demname, instrument = "PSB.SD"):
    with open(searchfile) as f:
        gj = json.load(f)
    if instrument== "PSB.SD":
        ids = ["_".join(os.path.split(img)[1].split("_")[0:4]) for img in [img1,img2]]
        features = [gj["features"][i] for i in range(len(gj["features"])) if gj["features"][i]["properties"]["id"] in ids]
    elif instrument == "PS2":
        ids = ["_".join(os.path.split(img)[1].split("_")[0:3]) for img in [img1,img2]]
        #TODO: add feature filter for PS2
        
    sun_az = features[0]["properties"]["sun_azimuth"]
    sun_el = features[0]["properties"]["sun_elevation"]
    
    cmd = f"gdaldem hillshade -az {sun_az} -alt {sun_el} {demname} {demname}_hillshade_temp.tif"
    result = subprocess.run(cmd, shell = True,  stdout=subprocess.PIPE, stderr=subprocess.PIPE, text = True)
    if result.stderr != "":
        print(result.stderr)
    hs1 = read_file(f"{demname}_hillshade_temp.tif")
    shade1 = np.where(hs1<100,1,0)
    
    sun_az = features[1]["properties"]["sun_azimuth"]
    sun_el = features[1]["properties"]["sun_elevation"]
    
    cmd = f"gdaldem hillshade -az {sun_az} -alt {sun_el} {demname} {demname}_hillshade_temp.tif"
    result = subprocess.run(cmd, shell = True,  stdout=subprocess.PIPE, stderr=subprocess.PIPE, text = True)
    if result.stderr != "":
        print(result.stderr)
    hs2 = read_file(f"{demname}_hillshade_temp.tif")
    shade2 = np.where(hs2<100,1,0)

    shade_diff = shade2-shade1
    shade_diff[shade_diff<0] = 1
    plt.figure()
    plt.imshow(shade_diff)
    
    _,img1_name = os.path.split(img1)
    _,img2_name = os.path.split(img2)

    plt.title(f"{img1_name} {img2_name}")


def correlate_asp(img1, img2, prefix = "run", session = "rpc", sp_mode = 1, method = "asp_bm", nodata_value = None, corr_kernel = 25, amespath="/home/ariane/Downloads/StereoPipeline-3.1.1-alpha-2022-06-16-x86_64-Linux/bin/"):
    #run ASP stereo correlation in correlator mode
    folder = img1.replace(img1.split("/")[-1], "")
    print(f"Data will be saved under {folder}stereo/")
    
    if method == "asp_bm":
        cmd = f"{amespath}stereo {img1} {img2} {folder}stereo/{prefix} --correlator-mode -t {session} --datum Earth --skip-rough-homography --stereo-algorithm {method} --subpixel-mode {sp_mode} --corr-kernel {corr_kernel} {corr_kernel} --subpixel-kernel {corr_kernel+10} {corr_kernel+10} --threads 0" 
        if nodata_value is not None: 
            cmd = f"{cmd} --nodata-value {nodata_value}"
    else:
        if corr_kernel == 25:
            corr_kernel == 7
        cmd = f"{amespath}parallel_stereo {img1} {img2} {folder}stereo/{prefix} --correlator-mode -t {session} --datum Earth --skip-rough-homography --stereo-algorithm {method} --corr-kernel 9 9 --subpixel-mode {sp_mode} --threads 0" 

        if nodata_value is not None: 
            cmd = f"{cmd} --nodata-value {nodata_value}"
            
    subprocess.run(cmd, shell = True)
    
    return f"{folder}stereo/"
    
def image_align_asp(img1, img2, prefix = None, amespath="/home/ariane/Downloads/StereoPipeline-3.1.1-alpha-2022-06-16-x86_64-Linux/bin/"):
    #run ASP image_align with disparity derived from running the correlation
    folder = img1.replace(img1.split("/")[-1], "")
    print(f"Data will be saved under {folder}image_align/")
    if prefix: 
        cmd = f"{amespath}image_align {img1} {img2} -o {img2[:-4]}_aligned.tif --output-prefix {folder}image_align/{prefix} --alignment-transform affine --disparity-params '{folder}stereo/{prefix}-F.tif 10000' --inlier-threshold 100" 
    else:
        cmd = f"{amespath}image_align {img1} {img2} -o {img2[:-4]}_aligned.tif --output-prefix {folder}image_align/{prefix} --alignment-transform affine  --inlier-threshold 100" 

    subprocess.run(cmd, shell = True)

def parse_match_asp(img1, img2, prefix = "run", amespath="/home/ariane/Downloads/StereoPipeline-3.1.1-alpha-2022-06-16-x86_64-Linux/bin/"):
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
    df = pd.read_csv(matchfile, skiprows = 1, header = None, sep = " ")
    nrIPs = pd.read_csv(matchfile, nrows = 1, header = None, sep = " ")

    df1 = df.head(nrIPs[0][0]).reset_index(drop = True)
    df2 = df.tail(nrIPs[1][0]).reset_index(drop = True)

    df = pd.DataFrame({"x_img1":df1[0], "y_img1":df1[1],"x_img2":df2[0], "y_img2":df2[1]})
    
    return df

def revert_match(match_df, path, img1, img2, prefix, amespath):
    nrIPs = len(match_df)
    img1 = img1.split("/")[-1]
    img2 = img2.split("/")[-1]
    x = pd.concat([match_df.x_img1, match_df.x_img2], axis = 0).reset_index(drop = True)
    y = pd.concat([match_df.y_img1, match_df.y_img2], axis = 0).reset_index(drop = True)
    out_df = pd.DataFrame([x,y,x,y,np.zeros(x.shape),np.ones(x.shape),np.zeros(x.shape),np.zeros(x.shape),np.zeros(x.shape),np.zeros(x.shape),np.zeros(x.shape)]).transpose()
    for col in [2,3,7,8,9,10]:
        out_df[col] = out_df[col].astype(int)
    out_df.to_csv(f"{path}{prefix}_{img1[:-4]}__{img2[:-4]}-clean.txt", index = False, sep = " ", header = [nrIPs, nrIPs, "", "", "", "", "", "", "", "", ""])
    #not adding -clean extension at the end so that bundle_adjust finds the correct file
    cmd = f"python {amespath}parse_match_file.py -rev {path}{prefix}_{img1[:-4]}__{img2[:-4]}-clean.txt {path}{prefix}_{img1[:-4]}__{img2[:-4]}.match"
    subprocess.run(cmd, shell = True)
    return f"{path}{prefix}_{img1[:-4]}__{img2[:-4]}.match"
    
    
def plot_matches(img1, img2, match_df, title = ""):
    cvimg1 = cv.imread(img1, cv.IMREAD_UNCHANGED) 
    cvimg1 = cvimg1/np.max(cvimg1)*255
    cvimg1 = cvimg1.astype(np.uint8)
    cvimg2 = cv.imread(img2, cv.IMREAD_UNCHANGED)
    cvimg2 = cvimg2/np.max(cvimg2)*255
    cvimg2 = cvimg2.astype(np.uint8)

    #equalize histogram 
    eqimg1 = cv.equalizeHist(cvimg1)
    eqimg2 = cv.equalizeHist(cvimg2)

    fig, ax = plt.subplots(1,2)
    ax[0].imshow(eqimg1, cmap = "Greys")
    ax[0].scatter(match_df.x_img1,match_df.y_img1, s = 0.1, c = "r")
    ax[1].imshow(eqimg2, cmap = "Greys")
    ax[1].scatter(match_df.x_img2,match_df.y_img2, s = 0.1, c = "r")
    plt.title(title)
    plt.show()
    
def localize_matches(img1, img2, match_df, matchfile, demname, approxElev = 4000, save = False):
    
    #convert tiepoints from image coordinates to lon lat for plotting
    
    #load rpc models from files
    rpc1 = rpc_from_geotiff(img1)
    rpc2 = rpc_from_geotiff(img2)

    #get approximate lon lat
    lon1,lat1 = rpc1.localization(match_df.x_img1, match_df.y_img1,approxElev)
    #get DEM values at point (will be precise enough due to 30m resolution of DEM)
    elev = rasterValuesToPoint(lon1, lat1, demname)

    #recalculate lat and lon using actual elevation values
    lon1,lat1 = rpc1.localization(match_df.x_img1, match_df.y_img1,elev)

    #get approximate lon lat
    lon2,lat2 = rpc2.localization(match_df.x_img2, match_df.y_img2,approxElev)
    #get DEM values at point (will be precise enough due to 30m resolution of DEM)
    elev = rasterValuesToPoint(lon2, lat2, demname)
    #recalculate lat and lon using actual elevation values
    lon2,lat2 = rpc2.localization(match_df.x_img2, match_df.y_img2,elev)
    
    df = pd.DataFrame({"lon_img1":lon1, "lat_img1": lat1, "lon_img2":lon2, "lat_img2":lat2})

    if save: 
        df.to_csv(f"{matchfile[:-4]}_lonlat.csv", index = False)
        
    return df

def remove_unstable_matches(img1, img2, match_df, matchfile, outline, demname,approxElev = 4000, plot = False):
    loc = localize_matches(img1, img2, match_df, matchfile, demname)
   
    with open(outline) as f:
        gj = json.load(f)
        
    #generate polygon from gj
    geom = gj['features'][0]["geometry"]["coordinates"][0]
    poly = Polygon([tuple(coords) for coords in geom])
    
    if plot:
        plt.figure()
        plt.scatter(loc.lon_img1, loc.lat_img1, s = 0.1) 
        plt.plot(*poly.exterior.xy, c = "red")
        plt.show()
          
    ind = []
    for i, row in loc.iterrows():    
        p = Point(row.lon_img1, row.lat_img1)
        if p.intersects(poly):
            ind.append(i)
            
    match_df = match_df.drop(ind)

    match_df = match_df.reset_index(drop = True)

    if plot:
        loc = localize_matches(img1, img2, match_df, matchfile, demname)
        plt.figure()
        plt.scatter(loc.lon_img1, loc.lat_img1, s = 0.1)
        plt.plot(*poly.exterior.xy, c = "red")
        plt.show()
    
    return match_df

def apply_udm_mask(img1, img2, match_df, matchfile, demname, plot = False):
    loc = localize_matches(img1, img2, match_df, matchfile, demname)
    udm_fn = img2.replace("b2/","")
    udm_fn = udm_fn.replace("_b2", "_DN_udm")
    vals = rasterValuesToPoint(match_df.x_img2, match_df.y_img2, udm_fn)
    
    match_df = match_df.drop(np.where(np.array(vals)!=0)[0])
    match_df = match_df.reset_index(drop = True)
    
    if plot:
        plt.figure()
        plt.scatter(loc.lon_img1, loc.lat_img1, s = 0.1, c = vals, cmap = "coolwarm")
        plt.show()
    return match_df
        
def mapproject(img, dem, ext = "mp", ba_prefix = None, amespath = ""):
    
    #generate orthoimages with Ames
    if ba_prefix: 
        cmd = f"{amespath}mapproject {dem} {img} {img[:-4]}_{ext}.tif -t rpc --t_srs epsg:32720 --tr 3 --no-bigtiff --tif-compress Deflate --bundle-adjust-prefix {ba_prefix}"
    else: 
        cmd = f"{amespath}mapproject {dem} {img} {img[:-4]}_{ext}.tif -t rpc --t_srs epsg:32720 --tr 3 --no-bigtiff --tif-compress Deflate"
       
    subprocess.run(cmd, shell = True)
    return f"{img[:-4]}_{ext}.tif"
    
def clean_asp_files(path, prefix):
    #cleans up behind ASP to remove unneccessary files
    files = glob.glob(f"{path}{prefix}-*")
    disp  = glob.glob(f"{path}{prefix}-F.tif")
    remove = set(files)-set(disp)
    
    for file in remove:
        try:
            os.remove(file)
        except IsADirectoryError: #if parallel_stereo is used, also remove folders
            shutil.rmtree(file)