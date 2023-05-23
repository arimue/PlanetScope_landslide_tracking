#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 17:54:21 2023

@author: ariane
"""
from helper_functions import read_file, impute, rasterValuesToPoint, clip_raw, get_scene_id, copy_rpcs, save_file, min_max_scaler
import numpy as np
import matplotlib.pyplot as plt
import asp_helper_functions as asp
from rpcm.rpc_model import rpc_from_geotiff
import multiprocessing
import scipy.optimize
import pandas as pd
import scipy.ndimage
import cv2 as cv
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
from scipy.stats import linregress
import shutil
from plotting_functions import get_vlims
import subprocess, os, sys
from osgeo import gdal

def polyXY1(X, a, b, c):
    x,y = X
    out = a*x + b*y + c
    return out

def polyXY2(X, a, b, c, d, e, f):
    x,y = X
    out = a*x**2 +b*y**2 +c*x*y +d*x + e*y + f
    return out

def polyXY3(X, a, b, c, d, e, f, g, h, i, j, k):
    x,y = X
    out = a*x**3 + b*y**3 + c*x**2*y + d*x*y**2 + e*x**2 +f*y**2 +g*x*y +h*x + i*y + k
    return out


def find_tiepoints_SIFT(img1, img2, min_match_count = 100, plot = False):

    # image1 = cv.imread(img1) 
    # image2 = cv.imread(img2) 
    # gray1 = cv.cvtColor(image1, cv.COLOR_BGR2GRAY)
    # gray2 = cv.cvtColor(image2, cv.COLOR_BGR2GRAY)
    
    image1 = cv.imread(img1, cv.IMREAD_UNCHANGED) 
    image1 = min_max_scaler(image1)*255
    image1 = image1.astype(np.uint8)
    image2 = cv.imread(img2, cv.IMREAD_UNCHANGED)
    image2 = min_max_scaler(image2)*255
    image2 = image2.astype(np.uint8)
        
    # Histogram stretching helps A LOT with tiepoint detection
    gray1 = cv.equalizeHist(image1)
    gray2 = cv.equalizeHist(image2)
    

    # Find the key points and descriptors with SIFT

    sift = cv.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

    if len(good) > min_match_count:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        # Reduce dimension
        src_pts = src_pts[:, 0, :]
        dst_pts = dst_pts[:, 0, :]
    else:
        sys.exit("Not enough tiepoints were found. Is there sufficient overlap between your scenes?")

    df = pd.DataFrame({"x_img1": src_pts[:,0], "y_img1": src_pts[:,1], "x_img2": dst_pts[:,0], "y_img2": dst_pts[:,1]})
    
    if plot:
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))

        ax[0].imshow(gray1, cmap="gray")
        ax[0].scatter(df.x_img1, df.y_img1, c= "red", s = 0.1)
        ax[0].set_title("Tiepoints Image 1")

        ax[1].imshow(gray2, cmap="gray")
        ax[1].scatter(df.x_img2, df.y_img2, c= "red", s = 0.1)
        ax[1].set_title("Tiepoints Image 2")
        plt.show()

    
    return df


def improve_L3B_geolocation(img1, img2, order = 3, plot = False):
    df = find_tiepoints_SIFT(img1, img2, plot = plot)

    #!!!! assumes that images are clipped to the same aoi
    #TODO: catch this
    image2 = read_file(img2)


    df["xdiff"] = df.x_img2 - df.x_img1
    df["ydiff"] = df.y_img2 - df.y_img1
    
    #remove matches with really large distances
    pxup = np.nanpercentile(df.xdiff, 99)
    pyup = np.nanpercentile(df.ydiff, 99)
    pxlow = np.nanpercentile(df.xdiff, 1)
    pylow = np.nanpercentile(df.ydiff, 1)
    
    df = df.loc[df.xdiff >= pxlow]
    df = df.loc[df.xdiff <= pxup]
    df = df.loc[df.ydiff >= pylow]
    df = df.loc[df.ydiff <= pyup]
    df = df.reset_index(drop = True)
    
    xgrid, ygrid = np.meshgrid(np.arange(0,image2.shape[1], 1), np.arange(0, image2.shape[0], 1))

    
    if order == 1:
        xcoeffs1, xcov1 = scipy.optimize.curve_fit(polyXY1, xdata = (df.x_img2,df.y_img2), ydata = df.xdiff)
        xcoeffs2, xcov2 = scipy.optimize.curve_fit(polyXY1, xdata = (df.x_img2,df.y_img2), ydata = df.ydiff)
            
        dgx = polyXY1((xgrid.flatten(),ygrid.flatten()), *xcoeffs1)
        dgy = polyXY1((xgrid.flatten(),ygrid.flatten()), *xcoeffs2)
        
    elif order == 2:
 
        xcoeffs1, xcov1 = scipy.optimize.curve_fit(polyXY2, xdata = (df.x_img2,df.y_img2), ydata = df.xdiff)
        xcoeffs2, xcov2 = scipy.optimize.curve_fit(polyXY2, xdata = (df.x_img2,df.y_img2), ydata = df.ydiff)
            
        dgx = polyXY2((xgrid.flatten(),ygrid.flatten()), *xcoeffs1)
        dgy = polyXY2((xgrid.flatten(),ygrid.flatten()), *xcoeffs2)
        
    elif order == 3:
 
        xcoeffs1, xcov1 = scipy.optimize.curve_fit(polyXY3, xdata = (df.x_img2,df.y_img2), ydata = df.xdiff)
        xcoeffs2, xcov2 = scipy.optimize.curve_fit(polyXY3, xdata = (df.x_img2,df.y_img2), ydata = df.ydiff)
            
        dgx = polyXY3((xgrid.flatten(),ygrid.flatten()), *xcoeffs1)
        dgy = polyXY3((xgrid.flatten(),ygrid.flatten()), *xcoeffs2)
    
    else:
        print("Maximum order of polynomial fit is 3. Please pick a valid number!")
        return
    
    if plot:
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))

        p1 = ax[0].imshow(dgx.reshape(xgrid.shape), cmap="magma")
        ax[0].set_title("X fit")

        p2 = ax[1].imshow(dgy.reshape(xgrid.shape), cmap="magma")
        ax[1].set_title("Y fit")
        fig.colorbar(p1, ax=ax[0])
        fig.colorbar(p2, ax=ax[1])
        plt.show()
        
    dgx = (xgrid+dgx.reshape(xgrid.shape)).astype(np.float32)
    dgy = (ygrid +dgy.reshape(xgrid.shape)).astype(np.float32)
    
    image2_remap = cv.remap(image2, dgx, dgy, interpolation = cv.INTER_LINEAR)
    
    save_file([image2_remap], img2, img2[:-4]+"_remapped.tif")
    
    return img2[:-4]+"_remapped.tif"


def improve_L1B_geolocation(img1, img2, demname, order =3, plot = False):
    df = find_tiepoints_SIFT(img1, img2, plot = plot)
    image2 = read_file(img2)

    #localize SIFT features in object space using RPCs from img1
    ds = gdal.Open(img1)
    tr = gdal.Transformer(ds, None, ["METHOD=RPC", f"RPC_DEM={demname}"])
    pts_obj,_ = tr.TransformPoints(0, list(zip(df.x_img1, df.y_img1)))
    ds = tr = None
    
    #now project these points into the second image using RPCs from img2
    
    ds = gdal.Open(img2)
    tr = gdal.Transformer(ds, None, ["METHOD=RPC", f"RPC_DEM={demname}"])
    pts_img,_ = tr.TransformPoints(1, list(zip([p[0] for p in pts_obj],[p[1] for p in pts_obj])))
    ds = tr = None
    
    df["x_img2_should"] = [p[0] for p in pts_img]
    df["y_img2_should"] = [p[1] for p in pts_img]
    #if ref DEM has holes, points can become inf
    df = df[np.isfinite(df).all(1)]
    
    df["xdiff"] = df.x_img2 - df.x_img2_should
    df["ydiff"] = df.y_img2 - df.y_img2_should
    
    #remove matches with really large distances
    pxup = np.nanpercentile(df.xdiff, 99)
    pyup = np.nanpercentile(df.ydiff, 99)
    pxlow = np.nanpercentile(df.xdiff, 1)
    pylow = np.nanpercentile(df.ydiff, 1)
    
    df = df.loc[df.xdiff >= pxlow]
    df = df.loc[df.xdiff <= pxup]
    df = df.loc[df.ydiff >= pylow]
    df = df.loc[df.ydiff <= pyup]
    df = df.reset_index(drop = True)
    
    xgrid, ygrid = np.meshgrid(np.arange(0,image2.shape[1], 1), np.arange(0, image2.shape[0], 1))

    if order == 1:
        xcoeffs1, xcov1 = scipy.optimize.curve_fit(polyXY1, xdata = (df.x_img2,df.y_img2), ydata = df.xdiff)
        xcoeffs2, xcov2 = scipy.optimize.curve_fit(polyXY1, xdata = (df.x_img2,df.y_img2), ydata = df.ydiff)
            
        dgx = polyXY1((xgrid.flatten(),ygrid.flatten()), *xcoeffs1)
        dgy = polyXY1((xgrid.flatten(),ygrid.flatten()), *xcoeffs2)
        
    elif order == 2:

        xcoeffs1, xcov1 = scipy.optimize.curve_fit(polyXY2, xdata = (df.x_img2,df.y_img2), ydata = df.xdiff)
        xcoeffs2, xcov2 = scipy.optimize.curve_fit(polyXY2, xdata = (df.x_img2,df.y_img2), ydata = df.ydiff)
            
        dgx = polyXY2((xgrid.flatten(),ygrid.flatten()), *xcoeffs1)
        dgy = polyXY2((xgrid.flatten(),ygrid.flatten()), *xcoeffs2)
        
    elif order == 3:

        xcoeffs1, xcov1 = scipy.optimize.curve_fit(polyXY3, xdata = (df.x_img2,df.y_img2), ydata = df.xdiff)
        xcoeffs2, xcov2 = scipy.optimize.curve_fit(polyXY3, xdata = (df.x_img2,df.y_img2), ydata = df.ydiff)
            
        dgx = polyXY3((xgrid.flatten(),ygrid.flatten()), *xcoeffs1)
        dgy = polyXY3((xgrid.flatten(),ygrid.flatten()), *xcoeffs2)
    
    else:
        print("Maximum order of polynomial fit is 3. Please pick a valid number!")
        return
            
        
    if plot:
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))

        p1 = ax[0].imshow(dgx.reshape(xgrid.shape), cmap="magma")
        ax[0].set_title("X fit")

        p2 = ax[1].imshow(dgy.reshape(xgrid.shape), cmap="magma")
        ax[1].set_title("Y fit")
        fig.colorbar(p1, ax=ax[0])
        fig.colorbar(p2, ax=ax[1])
        plt.show()
        
    dgx = (xgrid+dgx.reshape(xgrid.shape)).astype(np.float32)
    dgy = (ygrid +dgy.reshape(xgrid.shape)).astype(np.float32)
    
    image2_remap = cv.remap(image2, dgx, dgy, interpolation = cv.INTER_LINEAR)
    cv.imwrite(img2[:-4]+"_remapped.tif", image2_remap)
    
    copy_rpcs(img2, img2[:-4]+"_remapped.tif")
    
    return img2[:-4]+"_remapped.tif"
