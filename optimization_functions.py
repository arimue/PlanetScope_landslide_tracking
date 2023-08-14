#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import helper_functions as helper
import numpy as np
import matplotlib.pyplot as plt
import asp_helper_functions as asp
import scipy.optimize
import pandas as pd
import scipy.ndimage
import cv2 as cv
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import os, sys, shutil
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

def polyXYZ3(X, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t):
    x,y,z = X
    out = a*x**3 + b*y**3 + c*z**3 + d*x*y**2 + e*x*z**2 + f*y*x**2 + g*y*z**2 + h*z*x**2 + i*z*y**2 + j*x*y*z + k*x**2 + l*y**2 + m*z**2 + n*x*y + o*x*z + p*y*z + q*x + r*y + s*z + t
    return out


def find_tiepoints_SIFT(img1, img2, min_match_count = 100, plot = False):
    # alternative to full image correlation. Faster but irregular spaced tiepoints 
    # image1 = cv.imread(img1) 
    # image2 = cv.imread(img2) 
    # gray1 = cv.cvtColor(image1, cv.COLOR_BGR2GRAY)
    # gray2 = cv.cvtColor(image2, cv.COLOR_BGR2GRAY)
    
    image1 = cv.imread(img1, cv.IMREAD_UNCHANGED) 
    image1 = helper.min_max_scaler(image1)*255
    image1 = image1.astype(np.uint8)
    image2 = cv.imread(img2, cv.IMREAD_UNCHANGED)
    image2 = helper.min_max_scaler(image2)*255
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


def shift_dem(params, demname, img1, img2, x_img1, y_img1, x_img2, y_img2, proj_tr, cross_track_weight = 10):
    
    #optimization function for disparity based DEM alignment
    
    a,b = params

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
    
    #project img1
    ds = gdal.Open(img1)
    tr = gdal.Transformer(ds, None, ["METHOD=RPC", f"RPC_DEM={demname[:-4]}_copy.tif"])

    pts_obj,_ = tr.TransformPoints(0, list(zip(x_img1, y_img1)))
    ds = tr = None

    coords_proj_img1 = [proj_tr.transform(c[0],c[1]) for c in pts_obj]
    
    #project img2
    ds = gdal.Open(img2)
    tr = gdal.Transformer(ds, None, ["METHOD=RPC", f"RPC_DEM={demname[:-4]}_copy.tif"])

    pts_obj,_ = tr.TransformPoints(0, list(zip(x_img2, y_img2)))
    ds = tr = None

    coords_proj_img2 = [proj_tr.transform(c[0],c[1]) for c in pts_obj]
    
    east_diff = abs(np.array([c[0] for c in coords_proj_img1]) - np.array([c[0] for c in coords_proj_img2]))
    north_diff = abs(np.array([c[1] for c in coords_proj_img1]) - np.array([c[1] for c in coords_proj_img2]))
    
    #penalize inf values
    east_diff[~np.isfinite(east_diff)] = 100
    north_diff[~np.isfinite(north_diff)] = 100
    
    return cross_track_weight * east_diff.sum() + north_diff.sum()

  
def disparity_based_DEM_alignment(amespath, img1, img2, demname, refdem, epsg, aoi = None, iterations = 1):
    
        
    """
    Perform disparity-based DEM alignment by minimizing the distances between tiepoints found in two images
    when projected into object space using the DEM.
    
    Parameters:
        amespath (str): Path to the directory containing the ASP executables.
        img1 (str): Path to the reference image.
        img2 (str): Path to the secondary image.
        demname (str): Path to the DEM that should be aligned.
        refdem (str): Path to the reference DEM raster file (will only be used for Z component). 
        epsg (int): EPSG code specifying a projected CRS.
        aoi (str, optional): Path to the AOI (Area of Interest) file. Defaults to None and assumes that input images are clipped then.
        iterations (int, optional): Number of iterations to run the process. Defaults to 1.
        
    Returns:
        Path to aligned DEM
    """

    #df = find_tiepoints_SIFT(img1, img2, plot = plot)
    
    if aoi is not None:
        #will clip images if AOI is provided. Else assumes that clipped images are passed
        ul_lon, ul_lat, xsize, ysize = helper.size_from_aoi(aoi, epsg = epsg, gsd = 4)
        img1 = helper.clip_raw(img1, ul_lon, ul_lat, xsize, ysize, refdem)
        ul_lon, ul_lat, xsize, ysize = helper.size_from_aoi(aoi, epsg = epsg, gsd = 4)
        img2 = helper.clip_raw(img2, ul_lon, ul_lat, xsize, ysize, refdem)
    else:
        print("Assuming that provided images are clipped...")
        
    for i in range(iterations):
        id1 = helper.get_scene_id(img1)
        id2 = helper.get_scene_id(img2)
    
        prefix = f"{id1}_{id2}_L1B"
        path, _ = os.path.split(img1)
        
        #usually, the PlanetDEM is located quite well in XY direction, just the elevation is off and tilted
        #therefore, the elevation difference between it and a reference DEM is calculated, modelled with a 1st order polyfit and subtracted
        
        refdemclip = helper.match_raster_size_and_res(demname, refdem)
        dem1 = helper.read_file(demname)
        dem2 = helper.read_file(refdemclip)
        meta = helper.read_meta(demname)
        dem1[dem1 == meta["nodata"]] = np.nan
        dem2[dem1 == meta["nodata"]] = np.nan
        
        demdiff = dem1-dem2
        xgrid, ygrid = np.meshgrid(np.arange(0,dem1.shape[1], 1), np.arange(0, dem1.shape[0], 1))
    
        data = np.c_[xgrid.flatten(), ygrid.flatten(), demdiff.flatten()]
        data = data[~np.isnan(data).any(axis=1)]
    
        xcoeffs, xcov = scipy.optimize.curve_fit(polyXY1, xdata = (data[:,0], data[:,1]), ydata = data[:,2])
        dg = polyXY1((xgrid.flatten(), ygrid.flatten()), *xcoeffs).reshape(dem1.shape)
        
        dem1 = dem1-dg
        
        if i > 0: #naming will be off if more then 10 iterations 
            helper.save_file([dem1], demname, demname[:-19]+f"_zaligned_it{i}.tif")
            demname = demname[:-19]+f"_zaligned_it{i}.tif"
            
        else:
            helper.save_file([dem1], demname, demname[:-4]+f"_zaligned_it{i}.tif")
            demname = demname[:-4]+f"_zaligned_it{i}.tif"
        
        if not os.path.isfile(path+"/disparity_maps/"+prefix+"-F.tif"):
            #TODO: allow adjustment of ames parameters
            print("Generating L1B disparity map to find tiepoints across entire scene...")
    
            stereopath = asp.correlate_asp(amespath, img1, img2, prefix = prefix, session = "rpc", sp_mode = 2, method = "asp_bm", nodata_value = None, corr_kernel = 35)
            asp.clean_asp_files(stereopath, prefix)
            
        else:
            print("Disparity file exists. Loading existing file to find tiepoints...")
            
        asp.image_align_asp(amespath, img1, img2, prefix = f"{id1}_{id2}_L1B")
        txt = asp.parse_match_asp(amespath, img1, img2, prefix = f"{id1}_{id2}_L1B")
        df = asp.read_match(txt)
            
        #localize tiepoints in object space using RPCs from img1
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
    
        
        #shift NEW
        pts_img,_ = tr.TransformPoints(1, [(c[0],c[1]) for c in pts_obj])
        df["x_img2_should"] = [c[0] for c in pts_img]
        df["y_img2_should"] = [c[1] for c in pts_img]
        
        df["x_diff"] = df.x_img2 - df.x_img2_should
        df["y_diff"] = df.y_img2 - df.y_img2_should
            
        
        #if ref DEM has holes or are outside image frame, points can become inf
        df = df[np.isfinite(df).all(1)]
        df = df.reset_index(drop = True)
        
        if len(df) == 0:
            print("Error! There are no valid matches in your dataframe left. This means that the tiepoints could not properly be projected. Check demname and input images.")
            return
        
        #remove median shift. these are related to imprecise cuts when working with the raw data only
        df["x_img2_new"] = df.x_img2 - df.x_diff.median()
        df["y_img2_new"] = df.y_img2 - df.y_diff.median()
        
        
        dist_thresh = 5 #TODO: let user adjust this
    
        #remove unreliable matches remaining after median shift
        df = df.loc[abs(df.x_img2_new - df.x_img2_should) <= dist_thresh]
        df = df.loc[abs(df.y_img2_new - df.y_img2_should) <= dist_thresh]
    
        print("Finding optimal DEM shift ...")
        result = scipy.optimize.minimize(shift_dem, [0,0], args=(demname, img1, img2, df.x_img1, df.y_img1, df.x_img2_new, df.y_img2_new, proj_tr))
        print(f"Adjusting DEM position: xshift = {result.x[0]}, yshift = {result.x[1]}")
    
        #apply final shift to DEM
        if os.path.isfile(demname[:-4]+"_copy.tif"):
            os.remove(demname[:-4]+"_copy.tif")
        if os.path.isfile(demname[:-4]+"_copy.tif.aux.xml"):
            os.remove(demname[:-4]+"_copy.tif.aux.xml")
        shutil.copyfile(demname, demname[:-17]+f"_xyzaligned_it{i}.tif")
        demds = gdal.Open(demname[:-17]+f"_xyzaligned_it{i}.tif")
        gt = list(demds.GetGeoTransform())
        gt[0] +=result.x[0]
        gt[3] +=result.x[1]
        demds.SetGeoTransform(tuple(gt))
        demds = tr = ds = None
        
        demname = demname[:-17]+f"_xyzaligned_it{i}.tif"
        
    return demname

def percentile_cut(dat, plow = 5, pup = 95, replace = np.nan):
    
    #removes high disparities based on percentile cut
    
    perc1 = np.nanpercentile(dat, plow)
    perc2 = np.nanpercentile(dat, pup)
    
    dat[dat < perc1] = replace
    dat[dat > perc2] = replace

    return dat

def apply_polyfit(matches, prefix_ext= "", order = 2, demname = None, plimlow = 5, plimup = 95, save_remapped_sec = False):
    """
    Apply polynomial fit to the disparity maps from the provided matches.
    
    Parameters
        matches (str or pd.DataFrame): Path to the matchfile or pandas DataFrame containing the match information.
        prefix_ext (str, optional): Prefix extension for filenames. Defaults to an empty string.
        order (int, optional): Order of the polynomial fit. Defaults to 2.
        demname (str, optional): Path to the DEM raster file. If provided, an elevation component will be incoporated in the polyfit.
        plimlow (float, optional): Lower percentile for outlier removal. Defaults to 5.
        plimup (float, optional): Upper percentile for outlier removal.. Defaults to 95.
    
    Returns:
        list: List of filenames of the disparity maps with the applied polynomial fit.
    """
    if type(matches) == str:
        try:
            df = pd.read_csv(matches)

        except FileNotFoundError:
            print("Could not find the provided matchfile.")
            return
    elif type(matches) == pd.core.frame.DataFrame:
        df = matches.copy()
    else:
        print("Matches must be either a string indicating the path to a matchfile or a pandas DataFrame.")
        return
    
    out = []
    for idx, row in df.iterrows():
        id1 = helper.get_scene_id(row.ref)
        id2 = helper.get_scene_id(row.sec)
        prefix = f"{id1}_{id2}{prefix_ext}"
        
        path,_ = os.path.split(row.ref)
        dispfn = os.path.join(path, "disparity_maps", prefix+"-F.tif")
        if os.path.isfile(dispfn):
            print(dispfn)
            dx = helper.read_file(dispfn, b = 1)
            dy = helper.read_file(dispfn, b = 2)
            mask = helper.read_file(dispfn, b = 3)
            
            dx[mask == 0] = np.nan
            dy[mask == 0] = np.nan
            
            #TODO: add plotting option
            # fix,ax = plt.subplots(1,2)
            # ax[0].imshow(dx, vmin = -3, vmax = 3, cmap = "coolwarm")
            # ax[1].imshow(dy, vmin = -3, vmax = 3, cmap = "coolwarm")
            
            dxc = percentile_cut(dx.copy(), plow = plimlow, pup = plimup)
            dyc = percentile_cut(dy.copy(), plow = plimlow, pup = plimup)
            
                        
            xgrid, ygrid = np.meshgrid(np.arange(0,dx.shape[1], 1), np.arange(0, dx.shape[0], 1))

            fit_data = helper.min_max_scaler(xgrid.flatten())
            fit_data = np.c_[fit_data, helper.min_max_scaler(ygrid.flatten()), dxc.flatten(), dyc.flatten()]
            
            if demname is not None: 
                print("Adding elevation to the polynomial fit...")
                dem_matched = helper.match_raster_size_and_res(dispfn, demname)
                zgrid = helper.read_file(dem_matched)
                #make sure to remove nodata (any negative values)
                zgrid[zgrid < 0] = np.nan            
                fit_data = np.c_[fit_data,helper.min_max_scaler(zgrid.flatten())]

            fit_data = fit_data[~np.isnan(fit_data).any(axis=1)]
            
            if order == 1:
                
                if demname is None:
                    xcoeffs, xcov = scipy.optimize.curve_fit(polyXY1, xdata = (fit_data[:,0],fit_data[:,1]), ydata = fit_data[:,2])
                    ycoeffs, ycov = scipy.optimize.curve_fit(polyXY1, xdata = (fit_data[:,0],fit_data[:,1]), ydata = fit_data[:,3])
                        
                    dgx = polyXY1((helper.min_max_scaler(xgrid.flatten()),helper.min_max_scaler(ygrid.flatten())), *xcoeffs)
                    dgy = polyXY1((helper.min_max_scaler(xgrid.flatten()),helper.min_max_scaler(ygrid.flatten())), *ycoeffs)
                    
                else:
                    xcoeffs, xcov = scipy.optimize.curve_fit(polyXYZ1, xdata = (fit_data[:,0],fit_data[:,1],fit_data[:,4]), ydata = fit_data[:,2])
                    ycoeffs, ycov = scipy.optimize.curve_fit(polyXYZ1, xdata = (fit_data[:,0],fit_data[:,1],fit_data[:,4]), ydata = fit_data[:,3])
                        
                    dgx = polyXYZ1((helper.min_max_scaler(xgrid.flatten()),helper.min_max_scaler(ygrid.flatten()), helper.min_max_scaler(zgrid.flatten())), *xcoeffs)
                    dgy = polyXYZ1((helper.min_max_scaler(xgrid.flatten()),helper.min_max_scaler(ygrid.flatten()), helper.min_max_scaler(zgrid.flatten())), *ycoeffs)
                
            elif order == 2:
                if demname is None:
                    xcoeffs, xcov = scipy.optimize.curve_fit(polyXY2, xdata = (fit_data[:,0],fit_data[:,1]), ydata = fit_data[:,2])
                    ycoeffs, ycov = scipy.optimize.curve_fit(polyXY2, xdata = (fit_data[:,0],fit_data[:,1]), ydata = fit_data[:,3])
                                   
                    dgx = polyXY2((helper.min_max_scaler(xgrid.flatten()),helper.min_max_scaler(ygrid.flatten())), *xcoeffs)
                    dgy = polyXY2((helper.min_max_scaler(xgrid.flatten()),helper.min_max_scaler(ygrid.flatten())), *ycoeffs)
                    
                else:
                    xcoeffs, xcov = scipy.optimize.curve_fit(polyXYZ2, xdata = (fit_data[:,0],fit_data[:,1],fit_data[:,4]), ydata = fit_data[:,2])
                    ycoeffs, ycov = scipy.optimize.curve_fit(polyXYZ2, xdata = (fit_data[:,0],fit_data[:,1],fit_data[:,4]), ydata = fit_data[:,3])
                        
                    dgx = polyXYZ2((helper.min_max_scaler(xgrid.flatten()),helper.min_max_scaler(ygrid.flatten()), helper.min_max_scaler(zgrid.flatten())), *xcoeffs)
                    dgy = polyXYZ2((helper.min_max_scaler(xgrid.flatten()),helper.min_max_scaler(ygrid.flatten()), helper.min_max_scaler(zgrid.flatten())), *ycoeffs)
                    
                                        
            elif order == 3:
                if demname is None:

                    xcoeffs, xcov = scipy.optimize.curve_fit(polyXY3, xdata = (fit_data[:,0],fit_data[:,1]), ydata = fit_data[:,2])
                    ycoeffs, ycov = scipy.optimize.curve_fit(polyXY3, xdata = (fit_data[:,0],fit_data[:,1]), ydata = fit_data[:,3])
                        
                    dgx = polyXY3(helper.min_max_scaler(xgrid.flatten()),helper.min_max_scaler(ygrid.flatten()), *xcoeffs)
                    dgy = polyXY3(helper.min_max_scaler(xgrid.flatten()),helper.min_max_scaler(ygrid.flatten()), *ycoeffs)
                    
                else:
                    xcoeffs, xcov = scipy.optimize.curve_fit(polyXYZ3, xdata = (fit_data[:,0],fit_data[:,1],fit_data[:,4]), ydata = fit_data[:,2])
                    ycoeffs, ycov = scipy.optimize.curve_fit(polyXYZ3, xdata = (fit_data[:,0],fit_data[:,1],fit_data[:,4]), ydata = fit_data[:,3])
                        
                    dgx = polyXYZ3((helper.min_max_scaler(xgrid.flatten()),helper.min_max_scaler(ygrid.flatten()), helper.min_max_scaler(zgrid.flatten())), *xcoeffs)
                    dgy = polyXYZ3((helper.min_max_scaler(xgrid.flatten()),helper.min_max_scaler(ygrid.flatten()), helper.min_max_scaler(zgrid.flatten())), *ycoeffs)
                
            dx = dx - dgx.reshape(dx.shape)
            dy = dy - dgy.reshape(dy.shape)
            
            # fix,ax = plt.subplots(1,2)
            # ax[0].imshow(dx, vmin = -3, vmax = 3, cmap = "coolwarm")
            # ax[1].imshow(dy, vmin = -3, vmax = 3, cmap = "coolwarm")
            
            if save_remapped_sec: #remapping only makes sense for image pairs with a common reference scene
                sec = helper.read_file(row.sec)
                dgx = (xgrid + dgx.reshape(xgrid.shape)).astype(np.float32)
                dgy = (ygrid + dgy.reshape(xgrid.shape)).astype(np.float32)
                remap = cv.remap(sec, dgx, dgy, interpolation = cv.INTER_LINEAR)
                helper.save_file([remap], row.sec, outname = row.sec[:-4]+"_remap.tif")
            
            helper.save_file([dx,dy], dispfn, dispfn[:-6]+"_polyfit-F.tif")
            out.append(dispfn[:-6]+"_polyfit-F.tif")

    return out
            
            
  

