#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 17:54:21 2023

@author: ariane
"""
from helper_functions import read_file, impute, rasterValuesToPoint, clip_raw, get_scene_id
import numpy as np
import matplotlib.pyplot as plt
import asp_helper_functions as asp
from rpcm.rpc_model import rpc_from_geotiff
import multiprocessing
import scipy.optimize
import pandas as pd
import cv2 as cv
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
from scipy.stats import linregress
import shutil
from plotting_functions import get_vlims
import subprocess, os

def get_topo_info_grid(row, length, rpc, demname, approxElev = 4000):
    print(row)

    lon,lat = rpc.localization(np.arange(0,length, 1), np.repeat(row,length), approxElev)
    zvals = rasterValuesToPoint(lon, lat, demname)
    #finetune RPC locatlization with dem values
    lon,lat = rpc.localization(np.arange(0,length, 1), np.repeat(row,length), zvals)
        
    z = rasterValuesToPoint(lon, lat, demname)

    return z

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

def polyXYZ(X, a, b, c, d, e, f, g, h, i, j):
    x,y,z = X
    out = a*x**2 + b*y**2 + c*z**2 + d*x*y + e*x*z + f*y*z + g*x + h*y + i*z + j
    return out

def polyXYZ3(X, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t):
    x,y,z = X
    out = a*x**3 + b*y**3 + c*z**3 + d*x*y**2 + e*x*z**2 + f*y*x**2 + g*y*z**2 + h*z*x**2 + i*z*y**2 + j*x*y*z + k*x**2 + l*y**2 + m*z**2 + n*x*y + o*x*z + p*y*z + q*x + r*y + s*z + t
    return out

def polyXYZE(X, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o):
    x,y,z,err = X
    out = a*x**2 + b*y**2 + c*z**2 + d*err**2 + e*x*y + f*x*z + g*x*err + h*y*z + i*y*err + j*z*err + k*x + l*y + m*z + n*err + o
    return out


def fit(fit_points, direction, xgrid, ygrid, dgrid, mask, zgrid = None, egrid = None, prefix = "", max_iterations = 10, max_iqr_diff = 0.01, first_fit_order = 1, plot = False):
    
    xgridf = xgrid.flatten()
    ygridf = ygrid.flatten()
    
    if first_fit_order == 1:
        xcoeffs1, xcov1 = scipy.optimize.curve_fit(polyXY1, xdata = (fit_points.x, fit_points.y), ydata = fit_points[f"d{direction}"])
        fit_points["d_fit1"] = fit_points[f"d{direction}"] - polyXY1((fit_points.x, fit_points.y), *xcoeffs1) 
        dg1 = polyXY1((xgridf,ygridf), *xcoeffs1)
    elif first_fit_order ==2:
        xcoeffs1, xcov1 = scipy.optimize.curve_fit(polyXY2, xdata = (fit_points.x, fit_points.y), ydata = fit_points[f"d{direction}"])
        fit_points["d_fit1"] = fit_points[f"d{direction}"] - polyXY2((fit_points.x, fit_points.y), *xcoeffs1) 
        dg1 = polyXY2((xgridf,ygridf), *xcoeffs1)
    elif first_fit_order ==3:
        xcoeffs1, xcov1 = scipy.optimize.curve_fit(polyXY3, xdata = (fit_points.x, fit_points.y), ydata = fit_points[f"d{direction}"])
        fit_points["d_fit1"] = fit_points[f"d{direction}"] - polyXY3((fit_points.x, fit_points.y), *xcoeffs1) 
        dg1 = polyXY3((xgridf,ygridf), *xcoeffs1)
    else:
        print("Please provide a polynomial order <=3!")
        return
    
    #clean upper and lower percentiles to avoid the impact of abnormally high values
    dup = np.percentile(fit_points["d_fit1"], 95)
    dlow = np.percentile(fit_points["d_fit1"], 5)
    
    fit_points = fit_points.loc[(fit_points["d_fit1"] >= dlow) & (fit_points["d_fit1"] <= dup)]

    #correct grid
    disp_corrected1 = dgrid-dg1.reshape(xgrid.shape)
    disp_corrected1[mask==0] = np.nan
    
    #in case elevation fit is not carried out, but topography comes from DEM error, use first fit values
    dg2 = np.zeros(dg1.shape)
    fit_points["d_fit2"] = fit_points["d_fit1"]
    disp_corrected2 = disp_corrected1
    
    if zgrid is not None:
        zvals = fit_points.z
        zgridf = zgrid.flatten()
        #second fit
        xcoeffs2, xcov2 = scipy.optimize.curve_fit(polyXYZ3, xdata = (fit_points.x, fit_points.y, zvals), ydata = fit_points["d_fit1"])
        dg2 = polyXYZ3((xgridf,ygridf,zgridf), *xcoeffs2)
        fit_points["d_fit2"] = fit_points["d_fit1"] - polyXYZ3((fit_points.x, fit_points.y, zvals), *xcoeffs2) 

        disp_corrected2 = disp_corrected1-dg2.reshape(xgrid.shape)
        disp_corrected2[mask==0] = np.nan      
    
    dg3 = np.zeros(dg2.shape)
    #third fit if dem err given
    
    if f"err{direction}" in fit_points.columns:
        print("Using given DEM error raster to correct disparity...")
        
        it = 1
        iqr_diff = 100 #some arbitrary high number
        iqr_now = np.nanpercentile(disp_corrected2.flatten(), 75)-np.nanpercentile(disp_corrected2.flatten(),25)
       
        while iqr_diff > max_iqr_diff and it <= max_iterations:
            xcoeffs3, xcov3 = scipy.optimize.curve_fit(polyXYZ3, xdata = (fit_points.x, fit_points.y, fit_points[f"err{direction}"]), ydata = fit_points["d_fit2"])
            dg3 = polyXYZ3((xgridf,ygridf,egrid.flatten()), *xcoeffs3)
            fit_points["d_fit3"] = fit_points["d_fit2"] - polyXYZ3((fit_points.x, fit_points.y, fit_points[f"err{direction}"]), *xcoeffs3) 
    
            disp_corrected3 = disp_corrected2-dg3.reshape(xgrid.shape)
            disp_corrected3[mask==0] = np.nan
            
            #filter points that have a disparity greater than 1 pixel (optional)
            
            dup = np.percentile(fit_points["d_fit3"], 95)
            dlow = np.percentile(fit_points["d_fit3"], 5)
    
            fit_points = fit_points.loc[(fit_points["d_fit3"] >= dlow) & (fit_points["d_fit3"] <= dup)]
            
            
            iqr_before = iqr_now
            iqr_now = np.nanpercentile(disp_corrected3.flatten(), 75)-np.nanpercentile(disp_corrected3.flatten(),25)
            iqr_diff = iqr_before - iqr_now
            print(f"Iteration: {it}, IQR now: {iqr_now}, IQR change: {iqr_diff}")
            it+=1
        
    #plotting
    if plot: 
        dgrid[mask==0] = np.nan

        if zgrid is not None and egrid is not None:
            fig, ax = plt.subplots(2,2, figsize = (15,10))
            p00 = ax[0,0].imshow(dgrid, vmin = -get_vlims(dgrid), vmax = get_vlims(dgrid), cmap = "coolwarm")
            ax[0,0].set_title(f"d{direction} original")
            divider = make_axes_locatable(ax[0,0])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(p00, cax=cax, label='Offset [pix]')    
            p01 = ax[0,1].imshow(disp_corrected1, vmin = -get_vlims(disp_corrected1), vmax = get_vlims(disp_corrected1), cmap = "coolwarm")
            ax[0,1].set_title( f"d{direction} after polyXYing / unwarping")
            divider = make_axes_locatable(ax[0,1])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(p01, cax=cax, label='Offset [pix]')  
            p10 = ax[1,0].imshow(disp_corrected2, vmin = -get_vlims(disp_corrected2), vmax = get_vlims(disp_corrected2), cmap = "coolwarm")
            ax[1,0].set_title(f"d{direction} after removal of topographic signal")
            divider = make_axes_locatable(ax[1,0])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(p10, cax=cax, label='Offset [pix]')  
            p11 = ax[1,1].imshow(disp_corrected3, vmin = -get_vlims(disp_corrected3), vmax = get_vlims(disp_corrected3), cmap = "coolwarm")
            ax[1,1].set_title(f"d{direction} after removal of DEM error")
            divider = make_axes_locatable(ax[1,1])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(p11, cax=cax, label='Offset [pix]')  
            
            
        elif zgrid is not None: 
            fig, ax = plt.subplots(1,3, figsize = (22,5))
            p00 = ax[0].imshow(dgrid, vmin = -get_vlims(dgrid), vmax = get_vlims(dgrid), cmap = "coolwarm")
            ax[0].set_title(f"d{direction} original")
            divider = make_axes_locatable(ax[0])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(p00, cax=cax, label='Offset [pix]')    
            p01 = ax[1].imshow(disp_corrected1, vmin = -get_vlims(disp_corrected1), vmax = get_vlims(disp_corrected1), cmap = "coolwarm")
            ax[1].set_title( f"d{direction} after polyXYing / unwarping")
            divider = make_axes_locatable(ax[1])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(p01, cax=cax, label='Offset [pix]')  
            p10 = ax[2].imshow(disp_corrected2, vmin = -get_vlims(disp_corrected2), vmax = get_vlims(disp_corrected2), cmap = "coolwarm")
            ax[2].set_title(f"d{direction} after removal of topographic signal")
            divider = make_axes_locatable(ax[2])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(p10, cax=cax, label='Offset [pix]')  
            
            
        elif egrid is not None: 
            fig, ax = plt.subplots(1,3, figsize = (22,5))
            p00 = ax[0].imshow(dgrid, vmin = -get_vlims(dgrid), vmax = get_vlims(dgrid), cmap = "coolwarm")
            ax[0].set_title(f"d{direction} original")
            divider = make_axes_locatable(ax[0])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(p00, cax=cax, label='Offset [pix]')    
            p01 = ax[1].imshow(disp_corrected1, vmin = -get_vlims(disp_corrected1), vmax = get_vlims(disp_corrected1), cmap = "coolwarm")
            ax[1].set_title( f"d{direction} after polyXYing / unwarping")
            divider = make_axes_locatable(ax[1])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(p01, cax=cax, label='Offset [pix]')  
            p10 = ax[2].imshow(disp_corrected3, vmin = -get_vlims(disp_corrected3), vmax = get_vlims(disp_corrected3), cmap = "coolwarm")
            ax[2].set_title(f"d{direction} after removal of DEM error")
            divider = make_axes_locatable(ax[2])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(p10, cax=cax, label='Offset [pix]')  
            
            
        else: 
            fig, ax = plt.subplots(1,2, figsize = (15,5))
            p00 = ax[0].imshow(dgrid, vmin = -get_vlims(dgrid), vmax = get_vlims(dgrid), cmap = "coolwarm")
            ax[0].set_title(f"d{direction} original")
            divider = make_axes_locatable(ax[0])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(p00, cax=cax, label='Offset [pix]')    
            p01 = ax[1].imshow(disp_corrected1, vmin = -get_vlims(disp_corrected1), vmax = get_vlims(disp_corrected1), cmap = "coolwarm")
            ax[1].set_title( f"d{direction} after polyXYing / unwarping")
            divider = make_axes_locatable(ax[1])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(p01, cax=cax, label='Offset [pix]')  
            
        plt.tight_layout()
        plt.savefig(f"{prefix}_d{direction}_correction_steps_new.png", dpi = 300)
        plt.show()
        
    return (dg1+dg2+dg3).reshape(xgrid.shape)


def raw_correlate_and_correct(fn_img1, fn_img2, demname, amespath, ul_lon, ul_lat, xsize = 3200, ysize = 2000, zgrid = None, dem_err_x = "", dem_err_y = "", ext = "", reduce = 1, first_fit_order = 1, overwrite = False, plot = True):    
        
    #TODO:Improve remapped filenaming
    
    ident1 = get_scene_id(fn_img1)
    ident2 = get_scene_id(fn_img2)
    prefix = f"{ident1}_{ident2}_clip"
    ref_prefix = f"{ident1}_clip"
    path,_ = os.path.split(fn_img1)
    path = path +"/"
    stereopath = path + "stereo/"
    disp_fn = f"{stereopath}{prefix}-F.tif"

    
    if os.path.isfile(fn_img2[:-4]+"_clip_remap"+ext+".tif") and os.path.isfile(disp_fn[:-6]+"_dx_corrected"+ext+".tif") and os.path.isfile(disp_fn[:-6]+"_dy_corrected"+ext+".tif") and not overwrite:
        print("Remapped secondary image and corrected disparities exist. Skipping the fit...")
        return
    
    
    #Clipping
    print("Clipping to same extent...")

    ref_img_fn = clip_raw(fn_img1, ul_lon, ul_lat, xsize, ysize, demname)
    sec_img_fn = clip_raw(fn_img2, ul_lon, ul_lat, xsize, ysize, demname)
    

    #important: check for all zero columns and remove these
    ref_img = read_file(ref_img_fn)
    
    #easy clip, loosing RPCs
    #ref_img = ref_img[ref_img.any(axis=1),:]
    #cv.imwrite(ref_img_fn, ref_img)

    #complicated clip, keeping RPCs    
    first_row = np.argmax(ref_img.any(axis=1))
    first_col = np.argmax(ref_img.any(axis=0))
    last_row = np.max(np.argwhere(ref_img.any(axis=1)))
    last_col = np.max(np.argwhere(ref_img.any(axis=0)))
    
    if first_row != 0 or first_col != 0 or last_row < (ref_img.shape[1]-1) or last_col < (ref_img.shape[0]-1):

        shutil.copyfile(ref_img_fn, f"{ref_img_fn[:-4]}_temp.tif")
        cmd = f"gdal_translate -srcwin {first_col} {first_row} {last_col - first_col +1} {last_row - first_row +1} {ref_img_fn[:-4]}_temp.tif {ref_img_fn}"
        os.system(cmd)
        #subprocess.run(cmd, shell = True,  stdout=subprocess.PIPE, stderr=subprocess.PIPE, text = True)
    
        os.remove(f"{ref_img_fn[:-4]}_temp.tif")
    

    #run correlation
    if not os.path.isfile(disp_fn) or overwrite:
        stereopath = asp.correlate_asp(amespath, ref_img_fn, sec_img_fn, sp_mode = 2, method= "asp_bm", prefix = prefix)
        asp.clean_asp_files(stereopath, prefix)
    else:
        print("Disparity file exists. Skipping correlation.")
    

    dispX = read_file(disp_fn, 1)
    dispY = read_file(disp_fn, 2) 
    mask = read_file(disp_fn, 3)
    
    xgrid, ygrid = np.meshgrid(np.arange(0,dispX.shape[1], 1), np.arange(0, dispX.shape[0], 1))#
    
    fit_points = pd.DataFrame({"x":xgrid.flatten(), "y":ygrid.flatten(), "dx": dispX.flatten(),"dy": dispY.flatten(), "mask":mask.flatten()})
    if zgrid == "estimate":
        #get elevation with rpcs from img1 if not existent yet
        if not os.path.isfile(f"{path}{ref_prefix}.npy"):
            rpc1 = rpc_from_geotiff(ref_img_fn)
            zgrid = np.zeros(dispX.shape)
            
            pool = multiprocessing.Pool(processes=multiprocessing.cpu_count()-2)
            results = [pool.apply_async(get_topo_info_grid, [row, xgrid.shape[1], rpc1, demname, 4000]) for row in range(xgrid.shape[0])]
            for row, zvals in enumerate(results):
                zgrid[row,:] = zvals.get()
            
            np.save(f"{path}{ref_prefix}.npy", zgrid) 
        else: 
            zgrid = np.load(f"{path}{ref_prefix}.npy")         
        fit_points["z"] = zgrid.flatten()

    if dem_err_x != "":
        errx = read_file(dem_err_x)
        fit_points["errx"] = errx.flatten()
        egrid = errx
    if dem_err_y != "":
        erry = read_file(dem_err_y)
        fit_points["erry"] = erry.flatten()
        
    fit_points = fit_points[fit_points["mask"] != 0]
    fit_points = fit_points.dropna()

    fit_points.reset_index(inplace = True, drop = True)
    fit_points = fit_points.iloc[::reduce, :]

    egrid = None
    if dem_err_x != "":
        egrid = errx
    correction_x = fit(fit_points, "x", xgrid, ygrid, dispX, mask, zgrid, egrid, prefix = prefix, first_fit_order=first_fit_order, plot = plot)
    egrid = None
    if dem_err_y != "":
        egrid = erry
    correction_y = fit(fit_points, "y", xgrid, ygrid, dispY, mask, zgrid, egrid, prefix = prefix,first_fit_order=first_fit_order, plot = plot)
    
    xfin = (xgrid+correction_x).astype(np.float32)
    yfin = (ygrid+correction_y).astype(np.float32)
    
    dispX_corrected = dispX-correction_x
    dispX_corrected[mask==0]=np.nan
    dispY_corrected = dispY-correction_y
    dispY_corrected[mask==0]=np.nan
    
    dispX[mask==0]=np.nan
    dispY[mask==0]=np.nan

    if plot: 
        
        fig, ax = plt.subplots(1,2, figsize = (15,5))
        p00 = ax[0].imshow(dispX_corrected, vmin = -get_vlims(dispX_corrected), vmax = get_vlims(dispX_corrected), cmap = "coolwarm")
        ax[0].set_title("dx raw corrected")
        divider = make_axes_locatable(ax[0])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(p00, cax=cax, label='Offset [pix]')    
        p01 = ax[1].imshow(dispY_corrected, vmin = -get_vlims(dispY_corrected), vmax = get_vlims(dispY_corrected), cmap = "coolwarm")
        ax[1].set_title("dy raw corrected")
        divider = make_axes_locatable(ax[1])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(p01, cax=cax, label='Offset [pix]')  

        plt.tight_layout()
        plt.savefig(path+prefix+"_correction_new.png", dpi = 300)
        plt.show()
        
    img = read_file(sec_img_fn)
    remap_img = cv.remap(img, xfin, yfin, interpolation = cv.INTER_LINEAR)
    
    #
    ext = ""
    if zgrid is not None: 
        ext = ext + "_DEM"
    if egrid is not None:
        ext = ext + "_Err"
    cv.imwrite(sec_img_fn[:-4]+"_remap"+ext+".tif", remap_img)
    cv.imwrite(disp_fn[:-6]+"_dx_corrected"+ext+".tif", dispX_corrected)
    cv.imwrite(disp_fn[:-6]+"_dy_corrected"+ext+".tif", dispY_corrected)
    
    return sec_img_fn[:-4]+"_remap"+ext+".tif", disp_fn[:-6]+"_dx_corrected"+ext+".tif", disp_fn[:-6]+"_dy_corrected"+ext+".tif"
   

def estimate_dem_error(files, direction, method = "median", asp_alg = "bm", kernelsize = 9, plot = False):
    array_list = [read_file(x) for x in files]
    for i in range(len(array_list)):
        #if can happen, that some of the disparities is inverted therefore check relation with first scene:
        df = pd.DataFrame({"x":array_list[0].flatten(), "y":array_list[i].flatten()})
        df = df.dropna()
        results = linregress(df.x, df.y)
        
        #if they are negatively correlated, invert values
        if results[0]<0:
            print("Inverting disparity...")
            array_list[i]= array_list[i]*-1
    
    if method == "median": 
        m_arr = np.nanmedian(array_list, axis=0)
        m_arr = impute(m_arr) #fill small holes
    if method == "mean": 
        m_arr = np.nanmean(array_list, axis=0)
        m_arr = impute(m_arr) #fill small holes

    nanmask = np.where(np.isnan(m_arr), 0, 1)
    m_arr[np.isnan(m_arr)] = 0
    
    m_arr = cv.blur(m_arr, (kernelsize,kernelsize))
    m_arr[nanmask==0] = np.nan
    if plot: 
        plt.figure()
        plt.imshow(m_arr, vmin = -get_vlims(m_arr), vmax = get_vlims(m_arr), cmap = "coolwarm")
        plt.colorbar()
        
    
    path,_ = os.path.split(files[0])
    cv.imwrite(f"{path}/dem_error_d{direction}_{method}_{asp_alg}.tif", m_arr)
    return f"{path}/dem_error_d{direction}_{method}_{asp_alg}.tif"

def correlate_remapped_img_pairs(img1_fn, img2_fn, amespath):
     
    ident1 = get_scene_id(img1_fn)
    ident2 = get_scene_id(img2_fn)
    prefix = f"{ident1}_{ident2}_remap"
    
    path,_ = os.path.split(img1_fn)
    path = path +"/"
    stereopath = path + "stereo/"
    disp_fn = f"{stereopath}{prefix}-F.tif"


    #run correlation
    if not os.path.isfile(disp_fn):
        stereopath = asp.correlate_asp(amespath, img1_fn, img2_fn, sp_mode = 2, method= "asp_bm", nodata_value = 0, prefix = prefix)
        asp.clean_asp_files(stereopath, prefix)
    else:
        print("Disparity file exists. Skipping correlation.")
        
def mp_correlate(fn_img1, fn_img2, demname, amespath, ul_lon = None, ul_lat = None, xsize = 3200, ysize = 2000, crop_before_mp = False, cutline = None, plot = False):
    
    path,_ = os.path.split(fn_img1)
    
    if crop_before_mp:
        assert ul_lon is not None and ul_lat is not None, "Need to provide the coordinates of the upper left corner of the desired clip!"

        print("Clipping to same extent...")

        ref_img_fn = clip_raw(fn_img1, ul_lon, ul_lat, xsize, ysize, demname)
        sec_img_fn = clip_raw(fn_img2, ul_lon, ul_lat, xsize, ysize, demname)
        
        #important: check for all zero columns and remove these
        ref_img = read_file(ref_img_fn)
        
        #complicated clip, keeping RPCs    
        first_row = np.argmax(ref_img.any(axis=1))
        first_col = np.argmax(ref_img.any(axis=0))
        last_row = np.max(np.argwhere(ref_img.any(axis=1)))
        last_col = np.max(np.argwhere(ref_img.any(axis=0)))
        
        if first_row != 0 or first_col != 0 or last_row < (ref_img.shape[1]-1) or last_col < (ref_img.shape[0]-1):

            shutil.copyfile(ref_img_fn, f"{ref_img_fn[:-4]}_temp.tif")
            cmd = f"gdal_translate -srcwin {first_col} {first_row} {last_col - first_col +1} {last_row - first_row +1} {ref_img_fn[:-4]}_temp.tif {ref_img_fn}"
            subprocess.run(cmd, shell = True,  stdout=subprocess.PIPE, stderr=subprocess.PIPE, text = True)
        
            os.remove(f"{ref_img_fn[:-4]}_temp.tif")
            
    else: 
        assert cutline is not None, "Need to provide a cutline if clipping takes place after mp!"
        
        ref_img_fn = fn_img1
        sec_img_fn = fn_img2
        
    ident1 = get_scene_id(ref_img_fn)
    ident2 = get_scene_id(sec_img_fn)
    
    if not os.path.isfile(ref_img_fn[:-4]+"_mp.tif"):
        asp.mapproject(amespath, ref_img_fn, demname)
    if not os.path.isfile(sec_img_fn[:-4]+"_mp.tif"):
        asp.mapproject(amespath, sec_img_fn, demname)

    ref_img_fn_mp = ref_img_fn[:-4]+"_mp.tif"
    sec_img_fn_mp = sec_img_fn[:-4]+"_mp.tif"
    
    
    if not crop_before_mp:
        
        print("Clipping to same extent...")
        
        cmd = f"gdalwarp -tr 3 3 -r bilinear -cutline {cutline} -crop_to_cutline {ref_img_fn_mp} {ref_img_fn_mp[:-4]}_clip.tif -co COMPRESS=DEFLATE -co ZLEVEL=9 -co PREDICTOR=2"
        subprocess.run(cmd, shell = True,  stdout=subprocess.PIPE, stderr=subprocess.PIPE, text = True)
        
        ref_img_fn_mp = f"{ref_img_fn_mp[:-4]}_clip.tif"
        
        cmd = f"gdalwarp -tr 3 3 -r bilinear -cutline {cutline} -crop_to_cutline {sec_img_fn_mp} {sec_img_fn_mp[:-4]}_clip.tif -co COMPRESS=DEFLATE -co ZLEVEL=9 -co PREDICTOR=2"
        subprocess.run(cmd, shell = True,  stdout=subprocess.PIPE, stderr=subprocess.PIPE, text = True)
        
        sec_img_fn_mp = f"{sec_img_fn_mp[:-4]}_clip.tif"


    prefix = f"{ident1}_{ident2}_clip"
    stereopath = path + "stereo/"
    prefix_mp = prefix + "_mp"
    
    disp_fn = f"{stereopath}{prefix_mp}-F.tif"
    
    if not os.path.isfile(disp_fn):
        stereopath = asp.correlate_asp(amespath, ref_img_fn_mp, sec_img_fn_mp, sp_mode = 2, prefix = prefix_mp, method = "asp_bm")
        asp.clean_asp_files(stereopath, prefix_mp)
    else:
        print("Disparity file exists. Skipping correlation.")
        
        
    if plot: 
        dispXmp = read_file(disp_fn, 1)
        dispYmp = read_file(disp_fn, 2)
        maskmp = read_file(disp_fn, 3)
        dispXmp[maskmp==0]=np.nan
        dispYmp[maskmp==0]=np.nan
        
        fig, ax = plt.subplots(1,2, figsize = (15,10))
        p00 = ax[0].imshow(dispXmp, vmin = -2, vmax = 2, cmap = "coolwarm")
        ax[0].set_title("dx mapprojected")
        divider = make_axes_locatable(ax[0])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(p00, cax=cax, label='Offset [pix]')    
        p01 = ax[1].imshow(dispYmp, vmin = -25, vmax = 25, cmap = "coolwarm")
        ax[1].set_title("dy mapprojected")
        divider = make_axes_locatable(ax[1])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(p01, cax=cax, label='Offset [pix]')  
        plt.tight_layout()
        plt.savefig(path+"/"+prefix_mp+"_correlation.png", dpi = 300)
        plt.show()