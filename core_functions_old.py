#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 17:54:21 2023

@author: ariane
"""
from helper_functions import read_file,rasterValuesToPoint, warp, clip_raw, get_epsg, get_scene_id, copy_rpcs, save_file, min_max_scaler
import numpy as np
import matplotlib.pyplot as plt
import asp_helper_functions as asp
import multiprocessing
import scipy.optimize
import pandas as pd
import scipy.ndimage
import cv2 as cv
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import subprocess, os, sys, shutil
from osgeo import gdal
from pyproj import Transformer, CRS


def polyXY1(X, a, b, c):
    x,y = X
    out = a*x + b*y + c
    return out

def polyXYZ1(X, a, b, c, d):
    x,y,z = X
    out = a*x + b*y + c*z +d
    return out

def polyXY2(X, a, b, c, d, e, f):
    x,y = X
    out = a*x**2 +b*y**2 +c*x*y +d*x + e*y + f
    return out

def polyXYZ2(X, a, b, c, d, e, f, g, h, i, j):
    x,y,z = X
    out = a*x**2 +b*y**2 +c*z**2 +d*x*y + e*x*z + f*y*z + g*x +h*y +z*i + j
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

def shift_dem(params, demname, img2, east_img1, north_img1, x_img2, y_img2, proj_tr):
    a,b = params
    print(f"{a} {b}")
    if os.path.isfile(demname[:-4]+"_copy.tif"):
        os.remove(demname[:-4]+"_copy.tif")
    if os.path.isfile(demname[:-4]+"_copy.tif.aux.xml"):
        os.remove(demname[:-4]+"_copy.tif.aux.xml")
    shutil.copyfile(demname, demname[:-4]+"_copy.tif")
    demds = gdal.Open(demname[:-4]+"_copy.tif")
    gt = list(demds.GetGeoTransform())
    gt[0] +=a
    gt[3] +=b
    demds.SetGeoTransform(tuple(gt))
    demds = None
    
    ds = gdal.Open(img2)
    tr = gdal.Transformer(ds, None, ["METHOD=RPC", f"RPC_DEM={demname[:-4]}_copy.tif"])

    pts_obj,_ = tr.TransformPoints(0, list(zip(x_img2, y_img2)))
    ds = tr = None

    coords_proj = [proj_tr.transform(c[0],c[1]) for c in pts_obj]
    
    east_diff = abs(east_img1 - np.array([c[0] for c in coords_proj]))
    north_diff = abs(north_img1 - np.array([c[1] for c in coords_proj]))

    # plt.figure()
    # plt.scatter(x_img2, y_img2, c = east_diff, vmin = -10, vmax = 10, cmap = "coolwarm")
    # plt.title(f"a = {a}, b = {b}")
    
    #penalize inf values
    east_diff[~np.isfinite(east_diff)] = 100
    north_diff[~np.isfinite(north_diff)] = 100
    
    print(east_diff.sum()+north_diff.sum())
    return east_diff.sum()+north_diff.sum()

    #penalize infinite values
    east_diff[~np.isfinite(east_diff)] = 100#np.nanmax(east_diff)
    north_diff[~np.isfinite(north_diff)] = 100#np.nanmax(north_diff)

    print(east_diff.quantile(0.8)+north_diff.quantile(0.8))
    return east_diff.quantile(0.8)+north_diff.quantile(0.8)


def opt_xpos_o1(params, east_img1, north_img1, x_img2, y_img2, tr, proj_tr):
    # Extract the adjustment parameters
    a,b,c = params
    # Adjust the x and y coordinates of img2
    
    
    x_img2_adjusted = x_img2*a +y_img2*b +c
    y_img2_adjusted = y_img2 

    # Localize SIFT features in object space using adjusted coordinates
    pts_obj, _ = tr.TransformPoints(0, list(zip(x_img2_adjusted, y_img2_adjusted)))

    # Calculate the differences between the projected positions and img1 coordinates
    
    coords_proj = [proj_tr.transform(c[0],c[1]) for c in pts_obj]
    
    east_diff = east_img1 - np.array([c[0] for c in coords_proj])


    #replace infinite values
    east_diff[~np.isfinite(east_diff)] = 0
    

    return east_diff

def opt_ypos_o1(params, east_img1, north_img1, x_img2, y_img2, tr, proj_tr):
    # Extract the adjustment parameters
    a,b,c = params
    # Adjust the x and y coordinates of img2
    x_img2_adjusted = x_img2 
    y_img2_adjusted = x_img2*a +y_img2*b +c

    # Localize pts in object space using adjusted coordinates
    pts_obj, _ = tr.TransformPoints(0, list(zip(x_img2_adjusted, y_img2_adjusted)))

    # Calculate the differences between the projected positions and img1 coordinates
    
    coords_proj = [proj_tr.transform(c[0],c[1]) for c in pts_obj]
    
    north_diff = north_img1 - np.array([c[1] for c in coords_proj])

    #replace infinite values
    north_diff[~np.isfinite(north_diff)] = 0

    return north_diff


def opt_xpos_o1_z(params, east_img1, north_img1, elev_img2, x_img2, y_img2, tr, proj_tr):
    # Extract the adjustment parameters
    a,b,c,d = params
    # Adjust the x and y coordinates of img2
    
    
    x_img2_adjusted = x_img2*a +y_img2*b +elev_img2*c +d
    y_img2_adjusted = y_img2 

    # Localize SIFT features in object space using adjusted coordinates
    pts_obj, _ = tr.TransformPoints(0, list(zip(x_img2_adjusted, y_img2_adjusted)))

    # Calculate the differences between the projected positions and img1 coordinates
    
    coords_proj = [proj_tr.transform(c[0],c[1]) for c in pts_obj]
    
    east_diff = east_img1 - np.array([c[0] for c in coords_proj])

    #replace infinite values
    east_diff[~np.isfinite(east_diff)] = 0
    

    return east_diff

def opt_ypos_o1_z(params, east_img1, north_img1, elev_img2, x_img2, y_img2, tr, proj_tr):
    # Extract the adjustment parameters
    a,b,c,d = params
    # Adjust the x and y coordinates of img2
    x_img2_adjusted = x_img2 
    y_img2_adjusted = x_img2*a +y_img2*b +elev_img2*c +d

    # Localize SIFT features in object space using adjusted coordinates
    pts_obj, _ = tr.TransformPoints(0, list(zip(x_img2_adjusted, y_img2_adjusted)))

    # Calculate the differences between the projected positions and img1 coordinates
    
    coords_proj = [proj_tr.transform(c[0],c[1]) for c in pts_obj]
    
    north_diff = north_img1 - np.array([c[1] for c in coords_proj])

    #replace infinite values
    north_diff[~np.isfinite(north_diff)] = 0

    return north_diff


def opt_xpos_o2(params, east_img1, north_img1, x_img2, y_img2, tr, proj_tr):
    # Extract the adjustment parameters
    a,b,c,d,e,f = params
    # Adjust the x and y coordinates of img2
    
    x_img2_adjusted = a*x_img2**2 + b*y_img2**2 +c*x_img2*y_img2 + d*x_img2 +e*y_img2 +f 
    y_img2_adjusted = y_img2 

    # Localize SIFT features in object space using adjusted coordinates
    pts_obj, _ = tr.TransformPoints(0, list(zip(x_img2_adjusted, y_img2_adjusted)))

    # Calculate the differences between the projected positions and img1 coordinates
    
    coords_proj = [proj_tr.transform(c[0],c[1]) for c in pts_obj]
    
    east_diff = east_img1 - np.array([c[0] for c in coords_proj])

    #replace infinite values
    east_diff[~np.isfinite(east_diff)] = 0
    

    return east_diff

def opt_ypos_o2(params, east_img1, north_img1, x_img2, y_img2, tr, proj_tr):
    # Extract the adjustment parameters
    a,b,c,d,e,f = params
    # Adjust the x and y coordinates of img2
    x_img2_adjusted = x_img2 
    y_img2_adjusted = a*x_img2**2 + b*y_img2**2 +c*x_img2*y_img2 + d*x_img2 +e*y_img2 +f 

    # Localize SIFT features in object space using adjusted coordinates
    pts_obj, _ = tr.TransformPoints(0, list(zip(x_img2_adjusted, y_img2_adjusted)))

    # Calculate the differences between the projected positions and img1 coordinates
    
    coords_proj = [proj_tr.transform(c[0],c[1]) for c in pts_obj]
    
    north_diff = north_img1 - np.array([c[1] for c in coords_proj])

    #replace infinite values
    north_diff[~np.isfinite(north_diff)] = 0

    return north_diff

def opt_xpos_o2_z(params, east_img1, north_img1, elev_img2, x_img2, y_img2, tr, proj_tr):
    # Extract the adjustment parameters
    a,b,c,d,e,f,g,h,i,j = params
    # Adjust the x and y coordinates of img2
    
    x_img2_adjusted = a*x_img2**2 + b*y_img2**2 +c*elev_img2**2 + d*x_img2*y_img2 + e*x_img2*elev_img2 + f*y_img2*elev_img2 +g*x_img2 +h*y_img2 +i*elev_img2 +j 
    y_img2_adjusted = y_img2 

    # Localize SIFT features in object space using adjusted coordinates
    pts_obj, _ = tr.TransformPoints(0, list(zip(x_img2_adjusted, y_img2_adjusted)))

    # Calculate the differences between the projected positions and img1 coordinates
    
    coords_proj = [proj_tr.transform(c[0],c[1]) for c in pts_obj]
    
    east_diff = east_img1 - np.array([c[0] for c in coords_proj])

    #replace infinite values
    east_diff[~np.isfinite(east_diff)] = 0


    return east_diff

def opt_ypos_o2_z(params, east_img1, north_img1,elev_img2, x_img2, y_img2, tr, proj_tr):
    # Extract the adjustment parameters
    a,b,c,d,e,f,g,h,i,j = params
    # Adjust the x and y coordinates of img2
    x_img2_adjusted = x_img2 
    y_img2_adjusted = a*x_img2**2 + b*y_img2**2 +c*elev_img2**2 + d*x_img2*y_img2 + e*x_img2*elev_img2 + f*y_img2*elev_img2 +g*x_img2 +h*y_img2 +i*elev_img2 +j 

    # Localize SIFT features in object space using adjusted coordinates
    pts_obj, _ = tr.TransformPoints(0, list(zip(x_img2_adjusted, y_img2_adjusted)))

    # Calculate the differences between the projected positions and img1 coordinates
    
    coords_proj = [proj_tr.transform(c[0],c[1]) for c in pts_obj]
    
    north_diff = north_img1 - np.array([c[1] for c in coords_proj])

    #replace inf values with zero. cannot just be removed because the optimization wount work if output has different lengths
    #TODO: think about this
    north_diff[~np.isfinite(north_diff)] = 0
    #print(len(north_diff))
    return north_diff

    
def improve_L1B_geolocation(amespath, img1, img2, demname, epsg, order = 2, plot = False, add_elev = True):
    #df = find_tiepoints_SIFT(img1, img2, plot = plot)
    
    id1 = get_scene_id(img1)
    id2 = get_scene_id(img2)

    prefix = f"{id1}_{id2}_L1B"
    path, _ = os.path.split(img1)
    
    if not os.path.isfile(path+"/stereo/"+prefix+"-F.tif"):
        #TODO: allow adjustment of ames parameters
        print("Generating L1B disparity map to find tiepoints across entire scene...")

        stereopath = asp.correlate_asp(amespath, img1, img2, prefix = prefix, session = "rpc", sp_mode = 2, method = "asp_bm", nodata_value = None, corr_kernel = 35)
        asp.clean_asp_files(stereopath, prefix)
        
    else:
        print("Disparity file exists. Loading existing file to find tiepoints...")
        
    asp.image_align_asp(amespath, img1, img2, prefix = f"{id1}_{id2}_L1B")
    txt = asp.parse_match_asp(amespath, img1, img2, prefix = f"{id1}_{id2}_L1B")
    df = asp.read_match(txt)
    
    image2 = read_file(img2)

    #localize SIFT features in object space using RPCs from img1
    ds = gdal.Open(img1)
    tr = gdal.Transformer(ds, None, ["METHOD=RPC", f"RPC_DEM={demname}"])
    pts_obj,_ = tr.TransformPoints(0, list(zip(df.x_img1, df.y_img1)))
    ds = tr = None
    
    #transform to UTM to have differences in m
    proj_tr = Transformer.from_crs(CRS("EPSG:4326"), CRS("EPSG:"+str(epsg)), always_xy=True)  #! need always_xy = True otherwise does strange things

    coords_proj = [proj_tr.transform(c[0],c[1]) for c in pts_obj]
    
    df["east_img1"] = [c[0] for c in coords_proj]
    df["north_img1"] = [c[1] for c in coords_proj]
    

    #calculate the initial distances in bject space to remove points that are far off
    ds = gdal.Open(img2)
    tr = gdal.Transformer(ds, None, ["METHOD=RPC", f"RPC_DEM={demname}"])
    # Localize SIFT features in object space using adjusted coordinates
    pts_obj,_ = tr.TransformPoints(0, list(zip(df.x_img2, df.y_img2)))
    
    coords_proj = [proj_tr.transform(c[0],c[1]) for c in pts_obj]
    
    df["east_img2"] = [c[0] for c in coords_proj]
    df["north_img2"] = [c[1] for c in coords_proj]
    df["east_diff_init"] = df.east_img1 - df.east_img2
    df["north_diff_init"] = df.north_img1 - df.north_img2
    
 
#######################
    dist_thresh = 10
    
    df = df.loc[df.east_diff_init <= dist_thresh]
    df = df.loc[df.north_diff_init <= dist_thresh]
    df = df.loc[df.east_diff_init >= -dist_thresh]
    df = df.loc[df.north_diff_init >= -dist_thresh]
    
    #if ref DEM has holes or are outside image frame, points can become inf
    df = df[np.isfinite(df).all(1)]
    df = df.reset_index(drop = True)

    # Perform least squares minimization
    #get meshgrid for remapping the image later on
    xgrid, ygrid = np.meshgrid(np.arange(0,image2.shape[1], 1), np.arange(0, image2.shape[0], 1))
    
    if add_elev:
        if not os.path.isfile(f"{img2[:-4]}_zgrid.npy"):

            print("Generating zgrid. This will take a moment...")
            pts_obj_e, _ = tr.TransformPoints(0, list(zip(xgrid.flatten(), ygrid.flatten())))
            coords_proj_e = [proj_tr.transform(c[0],c[1]) for c in pts_obj_e]
            easting = np.array([c[0] for c in coords_proj_e])
            northing = np.array([c[1] for c in coords_proj_e])
    
            dem_epsg = get_epsg(demname)
            if dem_epsg != epsg:
                print("Reprojecting given reference DEM to match the given CRS...")
                demname = warp(demname, epsg = epsg)
                
            zgrid = rasterValuesToPoint(easting, northing, demname)
            zgrid[zgrid<0] = np.nan
            zgrid = zgrid.reshape(xgrid.shape)
            np.save(f"{img2[:-4]}_zgrid.npy", zgrid)
        else: 
            print("Loading existing zgrid...")
            zgrid = np.load(f"{img2[:-4]}_zgrid.npy")         

    print("Adjusting secondary image...")    
    if order == 1:
        if add_elev: 
            #extracting z position using initial RPCs to stay consistent with zgrid generation, slight offsets should not be a problem at 30 m res
            df["elev_img2"] = rasterValuesToPoint(df.east_img2, df.north_img2, demname)
            resultx = scipy.optimize.least_squares(opt_xpos_o1_z, [1,0,0,0], args=(df.east_img1, df.north_img1, df.elev_img2, df.x_img2, df.y_img2, tr, proj_tr))
            resulty = scipy.optimize.least_squares(opt_ypos_o1_z, [0,1,0,0], args=(df.east_img1, df.north_img1, df.elev_img2, df.x_img2, df.y_img2, tr, proj_tr))
            print(resultx.x)
            print(resulty.x)
            df["east_diff_fit"] = opt_xpos_o1_z(resultx.x, df.east_img1, df.north_img1,df.elev_img2, df.x_img2, df.y_img2, tr, proj_tr)
            df["north_diff_fit"] = opt_ypos_o1_z(resulty.x, df.east_img1, df.north_img1, df.elev_img2, df.x_img2, df.y_img2, tr, proj_tr)
            dgx = polyXYZ1((xgrid,ygrid,zgrid),*resultx.x).astype(np.float32) 
            dgy = polyXYZ1((xgrid,ygrid,zgrid),*resulty.x).astype(np.float32) 
        else:
            resultx = scipy.optimize.least_squares(opt_xpos_o1, [1,0,0], args=(df.east_img1, df.north_img1, df.x_img2, df.y_img2, tr, proj_tr))
            resulty = scipy.optimize.least_squares(opt_ypos_o1, [0,1,0], args=(df.east_img1, df.north_img1, df.x_img2, df.y_img2, tr, proj_tr))
            print(resultx.x)
            print(resulty.x)
            df["east_diff_fit"] = opt_xpos_o1(resultx.x, df.east_img1, df.north_img1, df.x_img2, df.y_img2, tr, proj_tr)
            df["north_diff_fit"] = opt_ypos_o1(resulty.x, df.east_img1, df.north_img1, df.x_img2, df.y_img2, tr, proj_tr)

            dgx = polyXY1((xgrid,ygrid),*resultx.x).astype(np.float32) 
            dgy = polyXY1((xgrid,ygrid),*resulty.x).astype(np.float32) 
            
    elif order == 2:
        if add_elev: 
            df["elev_img2"] = rasterValuesToPoint(df.east_img2, df.north_img2, demname)
            resultx = scipy.optimize.least_squares(opt_xpos_o2_z, [0,0,0,0,0,0,1,0,0,0], args=(df.east_img1, df.north_img1, df.elev_img2,df.x_img2, df.y_img2, tr, proj_tr))
            resulty = scipy.optimize.least_squares(opt_ypos_o2_z, [0,0,0,0,0,0,0,1,0,0], args=(df.east_img1, df.north_img1, df.elev_img2,df.x_img2, df.y_img2, tr, proj_tr))
            print(resultx.x)
            print(resulty.x)
            df["east_diff_fit"] = opt_xpos_o2_z(resultx.x, df.east_img1, df.north_img1,df.elev_img2, df.x_img2, df.y_img2, tr, proj_tr)
            df["north_diff_fit"] = opt_ypos_o2_z(resulty.x, df.east_img1, df.north_img1,df.elev_img2, df.x_img2, df.y_img2, tr, proj_tr)
            dgx = polyXYZ2((xgrid,ygrid, zgrid),*resultx.x).astype(np.float32) 
            dgy = polyXYZ2((xgrid,ygrid, zgrid),*resulty.x).astype(np.float32) 
        
        else: 
            resultx = scipy.optimize.least_squares(opt_xpos_o2, [0,0,0,1,0,0], args=(df.east_img1, df.north_img1, df.x_img2, df.y_img2, tr, proj_tr))
            resulty = scipy.optimize.least_squares(opt_ypos_o2, [0,0,0,0,1,0], args=(df.east_img1, df.north_img1, df.x_img2, df.y_img2, tr, proj_tr))
            print(resultx.x)
            print(resulty.x)
            df["east_diff_fit"] = opt_xpos_o2(resultx.x, df.east_img1, df.north_img1, df.x_img2, df.y_img2, tr, proj_tr)
            df["north_diff_fit"] = opt_ypos_o2(resulty.x, df.east_img1, df.north_img1, df.x_img2, df.y_img2, tr, proj_tr)
            dgx = polyXY2((xgrid,ygrid),*resultx.x).astype(np.float32)
            dgy = polyXY2((xgrid,ygrid),*resulty.x).astype(np.float32)
            

    if plot:
        fig, ax = plt.subplots(2, 2, figsize=(12, 10))

        ax[0,0].imshow(image2, cmap="gray")
        p1 = ax[0,0].scatter(df.x_img2, df.y_img2, c= df.east_diff_init, s = 0.5, vmin = -3, vmax = 3, cmap = "coolwarm")
        ax[0,0].set_title("Tiepoint distance obj space init")

        ax[0,1].imshow(image2, cmap="gray")
        p2 = ax[0,1].scatter(df.x_img2, df.y_img2, c= df.east_diff_fit, s = 0.5, vmin = -3, vmax = 3, cmap = "coolwarm")
        ax[0,1].set_title("Tiepoint distance obj space after fit")
        
        
        ax[1,0].imshow(image2, cmap="gray")
        p3 = ax[1,0].scatter(df.x_img2, df.y_img2, c= df.north_diff_init, s = 0.5, vmin = -3, vmax = 3, cmap = "coolwarm")
        ax[1,0].set_title("Tiepoint distance obj space init")

        ax[1,1].imshow(image2, cmap="gray")
        p4 = ax[1,1].scatter(df.x_img2, df.y_img2, c= df.north_diff_fit, s = 0.5, vmin = -3, vmax = 3, cmap = "coolwarm")
        ax[1,1].set_title("Tiepoint distance obj space after fit")
        
        fig.colorbar(p1, ax=ax[0,0])
        fig.colorbar(p2, ax=ax[0,1])
        fig.colorbar(p3, ax=ax[1,0])
        fig.colorbar(p4, ax=ax[1,1])
        plt.show()
        

    ds = tr = None
    
    #fit (difference between dxg and xgrid) needs to be subtracted to be propperly in place
    dgx = (xgrid -(dgx-xgrid)).astype(np.float32)
    dgy = (ygrid -(dgy-ygrid)).astype(np.float32)

    image2_remap = cv.remap(image2, dgx, dgy, interpolation = cv.INTER_LINEAR)
    cv.imwrite(img2[:-4]+"_remapped.tif", image2_remap)
    
    copy_rpcs(img2, img2[:-4]+"_remapped.tif")
    
    return img2[:-4]+"_remapped.tif"

