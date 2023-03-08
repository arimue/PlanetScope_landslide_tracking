#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions from an earlier approach relying on matching points from ASP image_align.  
"""

import pandas as pd
import numpy as np
import subprocess, glob, json
from helper_functions import rasterValuesToPoint
import cv2 as cv
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point
from rpcm.rpc_model import rpc_from_geotiff # see https://github.com/centreborelli/rpcm for installation


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

def revert_match(match_df, path, img1, img2, prefix, amespath):
    
    #turn a df back into a .match file
    
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
    
    #plot matches retrieved from image_align
    
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
    #remove matching points inside landslide mask
    
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
    #remove matches inside areas based on udm mask from planet
    
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