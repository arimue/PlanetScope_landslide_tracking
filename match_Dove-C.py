#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 17:10:52 2023

@author: ariane
"""



from basic_functions import read_file 
import numpy as np
import matplotlib.pyplot as plt
import matching_functions as matf
from rpcm.rpc_model import rpc_from_geotiff
import multiprocessing
import scipy.optimize
import pandas as pd
import os, glob, json
import cv2 as cv
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
from scipy.stats import linregress
import datetime, rasterio
import shutil
import subprocess

def get_topo_info_grid(row, length, rpc, demname, approxElev = 4000):
    print(row)

    lon,lat = rpc.localization(np.arange(0,length, 1), np.repeat(row,length), approxElev)
    zvals = matf.rasterValuesToPoint(lon, lat, demname)
    #finetune RPC locatlization with dem values
    
    lon,lat = rpc.localization(np.arange(0,length, 1), np.repeat(row,length), zvals)

        
    z = matf.rasterValuesToPoint(lon, lat, demname)

    return z

def polyXY(X, a, b, c):
    x,y = X
    out = a*x + b*y + c
    return out

def polyXY3(X, a, b, c, d, e, f, g, h, i, j, k):
    x,y = X
    out = a*x**3 + b*y**3 + c*x**2*y + d*x*y**2 + e*x**2 +f*y**2 +g*x*y +h*x + i*y + k
    return out

def polyXYZ(X, a, b, c, d, e, f, g, h, i, j):#, k, l, m, n, o, p, q, r, s, t):
    x,y,z = X
    #out = a*x**3 + b*y**3 + c*z**3 + d*x*y**2 + e*x*z**2 + f*y*x**2 + g*y*z**2 + h*z*x**2 + i*z*y**2 + j*x*y*z + k*x**2 + l*y**2 + m*z**2 + n*x*y + o*x*z + p*y*z + q*x + r*y + s*z + t
    out = a*x**2 + b*y**2 + c*z**2 + d*x*y + e*x*z + f*y*z + g*x + h*y + i*z + j

    return out

def polyXYZE(X, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o):#, k, l, m, n, o, p, q, r, s, t):
    x,y,z,err = X
    #out = a*x**3 + b*y**3 + c*z**3 + d*x*y**2 + e*x*z**2 + f*y*x**2 + g*y*z**2 + h*z*x**2 + i*z*y**2 + j*x*y*z + k*x**2 + l*y**2 + m*z**2 + n*x*y + o*x*z + p*y*z + q*x + r*y + s*z + t
    out = a*x**2 + b*y**2 + c*z**2 + d*err**2 + e*x*y + f*x*z + g*x*err + h*y*z + i*y*err + j*z*err + k*x + l*y + m*z + n*err + o

    return out


def polyXYZ3(X, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t):
    x,y,z = X
    out = a*x**3 + b*y**3 + c*z**3 + d*x*y**2 + e*x*z**2 + f*y*x**2 + g*y*z**2 + h*z*x**2 + i*z*y**2 + j*x*y*z + k*x**2 + l*y**2 + m*z**2 + n*x*y + o*x*z + p*y*z + q*x + r*y + s*z + t
    #out = k*x**3 + l*y**3 + m*x**2*y + n*x*y**2 + a*x**2 + b*y**2 + c*z**2 + d*x*y + e*x*z + f*y*z + g*x + h*y + i*z + j

    return out

def fit(fit_points, direction, xgrid, ygrid, dgrid, mask, zgrid = None, egrid = None, prefix = "", max_iterations = 10, max_iqr_diff = 0.1, plot = False):
    
    #TODO: make sure to use poly XY3 when estimating the DEM error
    xgridf = xgrid.flatten()
    ygridf = ygrid.flatten()
    
    xcoeffs1, xcov1 = scipy.optimize.curve_fit(polyXY, xdata = (fit_points.x, fit_points.y), ydata = fit_points[f"d{direction}"])
    fit_points["d_fit1"] = fit_points[f"d{direction}"] - polyXY((fit_points.x, fit_points.y), *xcoeffs1) 
    dg1 = polyXY((xgridf,ygridf), *xcoeffs1)
    
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
        # xcoeffs2, xcov2 = scipy.optimize.curve_fit(polyXYZE, xdata = (fit_points.x, fit_points.y, zvals, fit_points[f"err{direction}"]), ydata = fit_points["d_fit1"])
        # dg2 = polyXYZE((xgridf,ygridf,zgridf, egrid.flatten()), *xcoeffs2)
        # d_fit2 = fit_points["d_fit1"] - polyXYZE((fit_points.x, fit_points.y, zvals, fit_points[f"err{direction}"]), *xcoeffs2) 

        disp_corrected2 = disp_corrected1-dg2.reshape(xgrid.shape)
        disp_corrected2[mask==0] = np.nan
        
    #set removal of DEM error to all zeros if no DEM error correction takes place
    
    # fig, ax = plt.subplots(1,2, figsize = (14,5))
    # p0 = ax[0].imshow(disp_corrected2, vmin = -10, vmax = 10, cmap = "coolwarm")
    # divider = make_axes_locatable(ax[0])
    # cax = divider.append_axes("right", size="5%", pad=0.05)
    # fig.colorbar(p0, cax=cax, label='Offset [pix]')  
    # ax[1].hist(disp_corrected2.flatten(), bins = 100, range =(-10, 10))
    # ax[1].set_xlabel("Offset [pix]")
    # #plt.title(f"run {iterations}, dlow = {dlow}, dup = {dup}")
    # asp = np.diff(ax[1].get_xlim())[0] / np.diff(ax[1].get_ylim())[0]
    # asp /= np.abs(np.diff(ax[0].get_xlim())[0] / np.diff(ax[0].get_ylim())[0])
    # ax[1].set_aspect(asp)
    
    # ax[1].annotate(text = "", xy=(np.nanpercentile(disp_corrected2.flatten(), 25),np.diff(ax[1].get_ylim())[0]/2), xytext=(np.nanpercentile(disp_corrected2.flatten(), 75),np.diff(ax[1].get_ylim())[0]/2), arrowprops=dict(arrowstyle='<->'))
    # ax[1].annotate(text = f"IQR = {np.round(np.nanpercentile(disp_corrected2.flatten(), 75)-np.nanpercentile(disp_corrected2.flatten(), 25),3)}", xy=(np.nanpercentile(disp_corrected2.flatten(), 50),(np.diff(ax[1].get_ylim())[0]/2)+np.diff(ax[1].get_ylim())[0]/20), ha='center')

    # plt.suptitle("Disparity after initial deramping", fontsize=14)
    # plt.tight_layout()
    # plt.savefig(f"correction_after_deramping_d{direction}.png", dpi = 300)
    # plt.show()
    
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
            
            # fig, ax = plt.subplots(1,2, figsize = (14,5))
            # p0 = ax[0].imshow(disp_corrected3, vmin = -2, vmax = 2, cmap = "coolwarm")
            # divider = make_axes_locatable(ax[0])
            # cax = divider.append_axes("right", size="5%", pad=0.05)
            # fig.colorbar(p0, cax=cax, label='Offset [pix]')  
            # ax[1].hist(disp_corrected3.flatten(), bins = 100, range =(-2, 2))
            # ax[1].set_xlabel("Offset [pix]")
            # asp = np.diff(ax[1].get_xlim())[0] / np.diff(ax[1].get_ylim())[0]
            # asp /= np.abs(np.diff(ax[0].get_xlim())[0] / np.diff(ax[0].get_ylim())[0])
            # ax[1].set_aspect(asp)
            
            # ax[1].annotate(text = "", xy=(np.nanpercentile(disp_corrected3.flatten(), 25),np.diff(ax[1].get_ylim())[0]/2), xytext=(np.nanpercentile(disp_corrected3.flatten(), 75),np.diff(ax[1].get_ylim())[0]/2), arrowprops=dict(arrowstyle='<->'))
            # ax[1].annotate(text = f"IQR = {np.round(np.nanpercentile(disp_corrected3.flatten(), 75)-np.nanpercentile(disp_corrected3.flatten(), 25),3)}", xy=(np.nanpercentile(disp_corrected3.flatten(), 50),(np.diff(ax[1].get_ylim())[0]/2)+np.diff(ax[1].get_ylim())[0]/20), ha='center')

            # plt.suptitle(f"Disparity after round {it} of point elimination", fontsize=14)
            # plt.tight_layout()
            # plt.savefig(f"correction_after_fit{it}_d{direction}.png", dpi = 300)
            # plt.show()
            
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
            p00 = ax[0,0].imshow(dgrid, vmin = -matf.get_vlims(dgrid), vmax = matf.get_vlims(dgrid), cmap = "coolwarm")
            ax[0,0].set_title(f"d{direction} original")
            divider = make_axes_locatable(ax[0,0])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(p00, cax=cax, label='Offset [pix]')    
            p01 = ax[0,1].imshow(disp_corrected1, vmin = -matf.get_vlims(disp_corrected1), vmax = matf.get_vlims(disp_corrected1), cmap = "coolwarm")
            ax[0,1].set_title( f"d{direction} after polyXYing / unwarping")
            divider = make_axes_locatable(ax[0,1])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(p01, cax=cax, label='Offset [pix]')  
            p10 = ax[1,0].imshow(disp_corrected2, vmin = -matf.get_vlims(disp_corrected2), vmax = matf.get_vlims(disp_corrected2), cmap = "coolwarm")
            ax[1,0].set_title(f"d{direction} after removal of topographic signal")
            divider = make_axes_locatable(ax[1,0])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(p10, cax=cax, label='Offset [pix]')  
            p11 = ax[1,1].imshow(disp_corrected3, vmin = -matf.get_vlims(disp_corrected3), vmax = matf.get_vlims(disp_corrected3), cmap = "coolwarm")
            ax[1,1].set_title(f"d{direction} after removal of DEM error")
            divider = make_axes_locatable(ax[1,1])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(p11, cax=cax, label='Offset [pix]')  
            
            # plt.tight_layout()
            # plt.show()
            # egrid[mask==0] = np.nan
            # fig, ax = plt.subplots(1,2, figsize = (15,5))
            # p00 = ax[0].imshow(egrid, vmin = -matf.get_vlims(egrid), vmax = matf.get_vlims(egrid), cmap = "coolwarm")
            # ax[0].set_title(f"Estimated DEM error (median) in {direction} direction")
            # divider = make_axes_locatable(ax[0])
            # cax = divider.append_axes("right", size="5%", pad=0.05)
            # fig.colorbar(p00, cax=cax, label='Offset [pix]')    
            # p01 = ax[1].imshow(disp_corrected2, vmin = -matf.get_vlims(disp_corrected2), vmax = matf.get_vlims(disp_corrected2), cmap = "coolwarm")
            # ax[1].set_title( f"d{direction} after removal of topographic signal")
            # divider = make_axes_locatable(ax[1])
            # cax = divider.append_axes("right", size="5%", pad=0.05)
            # fig.colorbar(p01, cax=cax, label='Offset [pix]')  

            
        elif zgrid is not None: 
            fig, ax = plt.subplots(1,3, figsize = (22,5))
            p00 = ax[0].imshow(dgrid, vmin = -matf.get_vlims(dgrid), vmax = matf.get_vlims(dgrid), cmap = "coolwarm")
            ax[0].set_title(f"d{direction} original")
            divider = make_axes_locatable(ax[0])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(p00, cax=cax, label='Offset [pix]')    
            p01 = ax[1].imshow(disp_corrected1, vmin = -matf.get_vlims(disp_corrected1), vmax = matf.get_vlims(disp_corrected1), cmap = "coolwarm")
            ax[1].set_title( f"d{direction} after polyXYing / unwarping")
            divider = make_axes_locatable(ax[1])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(p01, cax=cax, label='Offset [pix]')  
            p10 = ax[2].imshow(disp_corrected2, vmin = -matf.get_vlims(disp_corrected2), vmax = matf.get_vlims(disp_corrected2), cmap = "coolwarm")
            ax[2].set_title(f"d{direction} after removal of topographic signal")
            divider = make_axes_locatable(ax[2])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(p10, cax=cax, label='Offset [pix]')  
            
            
        elif egrid is not None: 
            fig, ax = plt.subplots(1,3, figsize = (22,5))
            p00 = ax[0].imshow(dgrid, vmin = -matf.get_vlims(dgrid), vmax = matf.get_vlims(dgrid), cmap = "coolwarm")
            ax[0].set_title(f"d{direction} original")
            divider = make_axes_locatable(ax[0])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(p00, cax=cax, label='Offset [pix]')    
            p01 = ax[1].imshow(disp_corrected1, vmin = -matf.get_vlims(disp_corrected1), vmax = matf.get_vlims(disp_corrected1), cmap = "coolwarm")
            ax[1].set_title( f"d{direction} after polyXYing / unwarping")
            divider = make_axes_locatable(ax[1])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(p01, cax=cax, label='Offset [pix]')  
            p10 = ax[2].imshow(disp_corrected3, vmin = -matf.get_vlims(disp_corrected3), vmax = matf.get_vlims(disp_corrected3), cmap = "coolwarm")
            ax[2].set_title(f"d{direction} after removal of DEM error")
            divider = make_axes_locatable(ax[2])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(p10, cax=cax, label='Offset [pix]')  
            
            
        else: 
            fig, ax = plt.subplots(1,2, figsize = (15,5))
            p00 = ax[0].imshow(dgrid, vmin = -matf.get_vlims(dgrid), vmax = matf.get_vlims(dgrid), cmap = "coolwarm")
            ax[0].set_title(f"d{direction} original")
            divider = make_axes_locatable(ax[0])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(p00, cax=cax, label='Offset [pix]')    
            p01 = ax[1].imshow(disp_corrected1, vmin = -matf.get_vlims(disp_corrected1), vmax = matf.get_vlims(disp_corrected1), cmap = "coolwarm")
            ax[1].set_title( f"d{direction} after polyXYing / unwarping")
            divider = make_axes_locatable(ax[1])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(p01, cax=cax, label='Offset [pix]')  
            
        plt.tight_layout()
        plt.savefig(f"{prefix}_d{direction}_correction_steps_new.png", dpi = 300)
        plt.show()
        
    return (dg1+dg2+dg3).reshape(xgrid.shape)

    
def mp_correlate(fn_img1, fn_img2, demname, ul_lon, ul_lat, xsize = 3200, ysize = 2000, plot = False):
    print("Clipping to same extent...")
    ref_img_fn = matf.clip_raw(fn_img1, ul_lon, ul_lat, xsize, ysize, demname)
    src_img_fn = matf.clip_raw(fn_img2, ul_lon, ul_lat, xsize, ysize, demname)
    
    ident1 = "_".join(os.path.split(ref_img_fn)[1].split("_")[0:4])
    ident2 = "_".join(os.path.split(src_img_fn)[1].split("_")[0:4])
    
    matf.mapproject(ref_img_fn, demname, amespath=amespath)
    ref_img_fn_mp = ref_img_fn[:-4]+"_mp.tif"
    matf.mapproject(src_img_fn, demname, amespath=amespath)
    src_img_fn_mp = src_img_fn[:-4]+"_mp.tif"
    
    prefix = f"{ident1}_{ident2}_clip"
    stereopath = path + "stereo/"
    prefix_mp = prefix + "_mp"
    
    disp_fn = f"{stereopath}{prefix_mp}-F.tif"
    
    if not os.path.isfile(disp_fn):
        stereopath = matf.correlate_asp(ref_img_fn_mp, src_img_fn_mp, sp_mode = 2, amespath = amespath, prefix = prefix_mp, method = "asp_bm")
        matf.clean_asp_files(stereopath, prefix_mp)
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
        
    
def raw_correlate_and_correct(fn_img1, fn_img2, demname, ul_lon, ul_lat, xsize = 3200, ysize = 2000, demres = 3, zgrid = None, dem_err_x = "", dem_err_y = "", reduce = 1, plot = True):    
    
    print("Clipping to same extent...")

    ref_img_fn = matf.clip_raw(fn_img1, ul_lon, ul_lat, xsize, ysize, demname)
    src_img_fn = matf.clip_raw(fn_img2, ul_lon, ul_lat, xsize, ysize, demname)
    
    
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

        
    ident1 = "_".join(os.path.split(ref_img_fn)[1].split("_")[0:3])
    ident2 = "_".join(os.path.split(src_img_fn)[1].split("_")[0:3])
    prefix = f"{ident1}_{ident2}_clip"
    ref_prefix = f"{ident1}_clip"
    
    
    path,_ = os.path.split(ref_img_fn)
    path = path +"/"
    stereopath = path + "stereo/"
    disp_fn = f"{stereopath}{prefix}-F.tif"

    #run correlation
    if not os.path.isfile(disp_fn):
        stereopath = matf.correlate_asp(ref_img_fn, src_img_fn, sp_mode = 2, method= "asp_bm", amespath = amespath, prefix = prefix)
        matf.clean_asp_files(stereopath, prefix)
    else:
        print("Disparity file exists. Skipping correlation.")
    

    dispX = read_file(disp_fn, 1)
    dispY = read_file(disp_fn, 2) 
    mask = read_file(disp_fn, 3)
    
    xgrid, ygrid = np.meshgrid(np.arange(0,dispX.shape[1], 1), np.arange(0, dispX.shape[0], 1))#
    
    fit_points = pd.DataFrame({"x":xgrid.flatten(), "y":ygrid.flatten(), "dx": dispX.flatten(),"dy": dispY.flatten(), "mask":mask.flatten()})
    if zgrid == "estimate":
        #get elevation with rpcs from img1 if not existent yet
        if not os.path.isfile(f"{path}{ref_prefix}_{demres}m.npy"):
            rpc1 = rpc_from_geotiff(ref_img_fn)
            zgrid = np.zeros(dispX.shape)
            
            pool = multiprocessing.Pool(processes=multiprocessing.cpu_count()-2)
            results = [pool.apply_async(get_topo_info_grid, [row, xgrid.shape[1], rpc1, demname, 4000]) for row in range(xgrid.shape[0])]
            for row, zvals in enumerate(results):
                zgrid[row,:] = zvals.get()
            
            np.save(f"{path}{ref_prefix}_{demres}m.npy", zgrid) 
        else: 
            zgrid = np.load(f"{path}{ref_prefix}_{demres}m.npy")         
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
    correction_x = fit(fit_points, "x", xgrid, ygrid, dispX, mask, zgrid, egrid, prefix = prefix,  plot = plot)
    egrid = None
    if dem_err_y != "":
        egrid = erry
    correction_y = fit(fit_points, "y", xgrid, ygrid, dispY, mask, zgrid, egrid, prefix = prefix, plot = plot)
    
    xfin = (xgrid+correction_x).astype(np.float32)
    yfin = (ygrid+correction_y).astype(np.float32)
    
    dispX_corrected = dispX-correction_x
    dispX_corrected[mask==0]=np.nan
    dispY_corrected = dispY-correction_y
    dispY_corrected[mask==0]=np.nan
    
    dispX[mask==0]=np.nan
    dispY[mask==0]=np.nan

    if plot: 
        
        # if dem_err_x != "" and dem_err_y != "":
        #     fig, ax = plt.subplots(1,2, figsize = (15,5))
        #     p00 = ax[0].imshow(errx,  vmin = -matf.get_vlims(errx), vmax = matf.get_vlims(errx), cmap = "coolwarm")
        #     ax[0].set_title("DEM error x direction")
        #     divider = make_axes_locatable(ax[0])
        #     cax = divider.append_axes("right", size="5%", pad=0.05)
        #     fig.colorbar(p00, cax=cax, label='Offset [pix]') 
        #     p01 = ax[1].imshow(erry,  vmin = -matf.get_vlims(erry), vmax = matf.get_vlims(erry), cmap = "coolwarm")
        #     ax[1].set_title("DEM error y direction")
        #     divider = make_axes_locatable(ax[1])
        #     cax = divider.append_axes("right", size="5%", pad=0.05)
        #     fig.colorbar(p01, cax=cax, label='Offset [pix]') 
        #     plt.tight_layout()
        #     plt.show()
        # fig, ax = plt.subplots(2,2, figsize = (15,10))
        # p00 = ax[0,0].imshow(dispX, vmin = -matf.get_vlims(dispX), vmax = matf.get_vlims(dispX), cmap = "coolwarm")
        # ax[0,0].set_title("dx raw")
        # divider = make_axes_locatable(ax[0,0])
        # cax = divider.append_axes("right", size="5%", pad=0.05)
        # fig.colorbar(p00, cax=cax, label='Offset [pix]')    
        # p01 = ax[0,1].imshow(dispY, vmin = -matf.get_vlims(dispY), vmax = matf.get_vlims(dispY), cmap = "coolwarm")
        # ax[0,1].set_title("dy raw")
        # divider = make_axes_locatable(ax[0,1])
        # cax = divider.append_axes("right", size="5%", pad=0.05)
        # fig.colorbar(p01, cax=cax, label='Offset [pix]')  
        # p10 = ax[1,0].imshow(dispX_corrected, vmin = -matf.get_vlims(dispX_corrected), vmax = matf.get_vlims(dispX_corrected), cmap = "coolwarm")
        # ax[1,0].set_title("dx raw corrected")
        # divider = make_axes_locatable(ax[1,0])
        # cax = divider.append_axes("right", size="5%", pad=0.05)
        # fig.colorbar(p10, cax=cax, label='Offset [pix]')  
        # p11 = ax[1,1].imshow(dispY_corrected, vmin = -matf.get_vlims(dispY_corrected), vmax = matf.get_vlims(dispY_corrected), cmap = "coolwarm")
        # ax[1,1].set_title("dy raw corrected")
        # divider = make_axes_locatable(ax[1,1])
        # cax = divider.append_axes("right", size="5%", pad=0.05)
        # fig.colorbar(p11, cax=cax, label='Offset [pix]')  
        fig, ax = plt.subplots(1,2, figsize = (15,5))
        p00 = ax[0].imshow(dispX_corrected, vmin = -matf.get_vlims(dispX_corrected), vmax = matf.get_vlims(dispX_corrected), cmap = "coolwarm")
        ax[0].set_title("dx raw corrected")
        divider = make_axes_locatable(ax[0])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(p00, cax=cax, label='Offset [pix]')    
        p01 = ax[1].imshow(dispY_corrected, vmin = -matf.get_vlims(dispY_corrected), vmax = matf.get_vlims(dispY_corrected), cmap = "coolwarm")
        ax[1].set_title("dy raw corrected")
        divider = make_axes_locatable(ax[1])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(p01, cax=cax, label='Offset [pix]')  

        plt.tight_layout()
        plt.savefig(path+prefix+"_correction_new.png", dpi = 300)
        plt.show()
        
    img = read_file(src_img_fn)
    remap_img = cv.remap(img, xfin, yfin, interpolation = cv.INTER_LINEAR)
    
    #
    ext = ""
    if zgrid is not None: 
        ext = ext + "_DEM"
    if egrid is not None:
        ext = ext + "_Err"
    cv.imwrite(src_img_fn[:-4]+"_remap"+ext+".tif", remap_img)
    cv.imwrite(disp_fn[:-6]+"_dx_corrected"+ext+".tif", dispX_corrected)
    cv.imwrite(disp_fn[:-6]+"_dy_corrected"+ext+".tif", dispY_corrected)
    
    return src_img_fn[:-4]+"_remap.tif", disp_fn[:-6]+"_dx_corrected.tif", disp_fn[:-6]+"_dy_corrected.tif"
   
from scipy.ndimage import label, binary_dilation

def impute(arr, max_fillsize = 1000):
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

def estimate_dem_error(files, direction, method = "median", kernelsize = 9, plot = False):
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
        
    kernel = np.ones((5,5),np.uint8)
    kernel[2,2] = 0
    
    #fill holes 
    #array_list = [impute(arr, max_fillsize = 1000) for arr in array_list]
    if method == "median": 
        m_arr = np.nanmedian(array_list, axis=0)
        m_arr = impute(m_arr, max_fillsize = 1000) #fill small holes
    if method == "mean": 
        m_arr = np.nanmean(array_list, axis=0)
        m_arr = impute(m_arr, max_fillsize = 1000) #fill small holes

    nanmask = np.where(np.isnan(m_arr), 0, 1)
    m_arr[np.isnan(m_arr)] = 0
    
    m_arr = cv.blur(m_arr, (kernelsize,kernelsize))
    m_arr[nanmask==0] = np.nan
    if plot: 
        plt.figure()
        plt.imshow(m_arr, vmin = -matf.get_vlims(m_arr), vmax = matf.get_vlims(m_arr), cmap = "coolwarm")
        plt.colorbar()
        
    
    path,_ = os.path.split(files[0])
    cv.imwrite(f"{path}/dem_error_d{direction}_{method}_bm.tif", m_arr)
    return f"{path}/dem_error_d{direction}_{method}_bm.tif"

    
def correlate_remapped_img_pairs(img1_fn, img2_fn, amespath):
     
    ident1 = "_".join(os.path.split(img1_fn)[1].split("_")[0:4])
    ident2 = "_".join(os.path.split(img2_fn)[1].split("_")[0:4])
    prefix = f"{ident1}_{ident2}_remap"
    
    path,_ = os.path.split(img1_fn)
    path = path +"/"
    stereopath = path + "stereo/"
    disp_fn = f"{stereopath}{prefix}-F.tif"


    #run correlation
    if not os.path.isfile(disp_fn):
        stereopath = matf.correlate_asp(img1_fn, img2_fn, sp_mode = 2, method= "asp_bm", nodata_value = 0, amespath = amespath, prefix = prefix)
        matf.clean_asp_files(stereopath, prefix)
    else:
        print("Disparity file exists. Skipping correlation.")
    
        
def plot_sun_angle_diff_sd(df, searchfile):
    with open(searchfile) as f:
        gj = json.load(f)
    
    ref_id = "_".join(os.path.split(df.ref[0])[1].split("_")[0:4])
    sec_ids = ["_".join(os.path.split(df.sec[i])[1].split("_")[0:4]) for i in range(len(df.sec))]
    
    ref= [gj["features"][i] for i in range(len(gj["features"])) if gj["features"][i]["properties"]["id"] in ref_id] #change to ["features"]["properties"] for SD
    secs = [gj["features"][i] for i in range(len(gj["features"])) if gj["features"][i]["properties"]["id"] in sec_ids] #change to ["features"]["properties"] for SD

    sun_elev_diff = [ref[0]["properties"]["sun_elevation"]- s["properties"]["sun_elevation"] for s in secs]
    sun_az_diff = [ref[0]["properties"]["sun_azimuth"]- s["properties"]["sun_azimuth"] for s in secs]
    dates = [datetime.datetime.strptime(s["properties"]["acquired"],"%Y/%m/%d %H:%M:%S") for s in secs]
    
    plt.figure()
    plt.scatter(dates, sun_az_diff)
    plt.hlines(0, min(dates), max(dates), "gray", "--")
    plt.show()

def calc_velocity(fn, base_fn):
    
    _,fnsp = os.path.split(fn)
    d1 = fnsp.split("_")[0]
    d2 = fnsp.split("_")[4]
    
    date1 = datetime.datetime.strptime(d1, "%Y%m%d")
    date2 = datetime.datetime.strptime(d2, "%Y%m%d")

    date_dt_base = date2-date1
    tdiff = date_dt_base.days
    print(f"time difference: {tdiff} days")
            

   
    # load autoRIFT output
    with rasterio.open(fn) as src:
        # get raster resolution from metadata
        meta = src.meta
        res = meta["transform"][0]
        #print(res)
        # first band is offset in x direction, second band in y
        dx = src.read(1)
        dy = src.read(2)
        valid = src.read(3)
        
    if tdiff < 0: #invert velocity if negative time difference (ref younger than sec)
        dx = dx*-1
        dy = dy*-1

    with rasterio.open(base_fn) as src:

        bm = src.read(1)
    
    dx[valid == 0] = np.nan
    dy[valid == 0] = np.nan
    
    #calculate total offset (length of vector)
    v = np.sqrt((dx**2+dy**2))
    #convert to meter
    v = v * res
    #convert to meter/year (year)
    v = (v/abs(tdiff))*365
    
    #calculate angle to north
    north = np.array([0,1])
    #stack x and y offset to have a 3d array with vectors along axis 2
    vector_2 = np.dstack((dx,dy))
    unit_vector_1 = north / np.linalg.norm(north)
    unit_vector_2 = vector_2 / np.linalg.norm(vector_2, axis = 2, keepdims = True)
    #here np.tensordot is needed (instead of np.dot) because of the multiple dimensions of the input arrays
    dot_product = np.tensordot(unit_vector_1,unit_vector_2, axes=([0],[2]))

    direction = np.rad2deg(np.arccos(dot_product))
    
    #as always the smallest angle to north is given, values need to be substracted from 360 if x is negative
    subtract = np.zeros(dx.shape)
    subtract[dx<0] = 360
    direction = abs(subtract-direction)
    
    ##axtract necessary points for quiver plot
    xgrid, ygrid = np.meshgrid(np.arange(0,dx.shape[1], 1), np.arange(0, dx.shape[0], 1))#
    
    df = pd.DataFrame({"x":xgrid.flatten(), "y":ygrid.flatten(), "mask":valid.flatten(), "dx":dx.flatten(), "dy":dy.flatten(), "v":v.flatten()})
    df = df.loc[df['x'].mod(2).eq(0)]
    df = df.loc[df['y'].mod(2).eq(0)]

    #df = df.iloc[::10]
    df = df.loc[df["mask"]==1]
    # df = df.loc[df["dx"]<= np.nanpercentile(dx, 99.9)]
    # df = df.loc[df["dx"]>= np.nanpercentile(dx, 0.1)]
    # df = df.loc[df["dy"]<= np.nanpercentile(dy, 99.9)]
    # df = df.loc[df["dy"]>= np.nanpercentile(dy, 0.1)]
    df = df.loc[df["v"]>=0.3]
    df = df.loc[df["v"]<= 2]
    df = df.reset_index(drop = True)

    df = df.iloc[::100]


    ##plot
    
    fig, ax = plt.subplots(2,2, figsize = (12,8))
    p1 = ax[0,0].imshow(dx, cmap = "coolwarm", vmin = -2, vmax = 2)
    ax[0,0].set_title(f"DX, {d1} - {d2}, dt = {tdiff} days")
    divider = make_axes_locatable(ax[0,0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(p1, cax=cax, label='Disparity [pix]')

    p2 = ax[0,1].imshow(dy, cmap = "coolwarm", vmin = -2, vmax = 2)
    ax[0,1].set_title(f"DY, {d1} - {d2}, dt = {tdiff} days" )
    divider = make_axes_locatable(ax[0,1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(p2, cax=cax, label='Disparity [pix]')
    
    p3 = ax[1,0].imshow(v, cmap = "Oranges", vmin = 0, vmax = 1)
    ax[1,0].set_title(f"Velocity, {d1} - {d2}" )
    divider = make_axes_locatable(ax[1,0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(p3, cax=cax, label='Velocity [m/yr]')
    
    ax[1,1].imshow(bm, cmap = "Greys")
    ax[1,1].set_title(f"Vector field, {d1} - {d2}" )
    p4 = ax[1,1].quiver(df.x, df.y, df.dx, df.dy, df.v, cmap = "magma")
    divider = make_axes_locatable(ax[1,1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(p4, cax=cax, label='Velocity [m/yr]')
    
    plt.tight_layout()
    
demres = 3
path = "./Dove-C_Jujuy_all/L1B/"
#path = "/raid-manaslu/amueting/PhD/Project1/ImageTransformation/SD_Jujuy_Nadir/"
demname = f"./data/DEM/EGM96/demcoreg_alos/CopernicusDEM_EasternCordillera_EGM96_clip_AW3D30_NWArg_nuth_x+10.19_y-0.36_z+2.36_align_{demres}m.tif"
amespath = "/home/ariane/Downloads/StereoPipeline-3.1.1-alpha-2022-10-16-x86_64-Linux/bin/"
#amespath = "/raid-manaslu/amueting/StereoPipeline-3.1.1-alpha-2022-07-29-x86_64-Linux/bin/"
dem_err_x = f"{path}/stereo/dem_error_dx_mean_mgm.tif"
dem_err_y = f"{path}/stereo/dem_error_dy_mean_mgm.tif"
#searchfile = "all_scenes_SD.geojson"

# dem_err_x = f"./SD_Jujuy_all/stereo/dem_error_dx_mean_resid_single_scene.tif"
# dem_err_y = f"./SD_Jujuy_all/stereo/dem_error_dy_mean_resid_single_scene.tif"
#df = pd.read_csv(path+"sd_matches_stable.csv")
# ul_lon = -65.59392
# ul_lat = -23.89596
# xsize = 2500
# ysize = 1500
ul_lon = -65.61782
ul_lat = -23.88517

df = pd.read_csv(path+"Dove-C_matches.csv")

# corrfx = []
# corrfy = []
# for i in range(len(df)):
#     _, xname, yname = raw_correlate_and_correct(df.ref[i], df.sec[i], demname, ul_lon, ul_lat, demres = 3,  plot = False)   
#     corrfx.append(xname)
#     corrfy.append(yname)

# corrfx = glob.glob(f"{path}stereo/*_dx_corrected.tif")
# corrfy = glob.glob(f"{path}stereo/*_dy_corrected.tif")

    
# dem_err_x = estimate_dem_error(corrfx, "x",method = "mean", kernelsize = 21, plot = True)
# dem_err_y = estimate_dem_error(corrfy, "y",method = "mean", kernelsize = 21, plot = True)
# dem_err_x = estimate_dem_error(corrfx, "x",method = "median", kernelsize = 21, plot = True)
# dem_err_y = estimate_dem_error(corrfy, "y",method = "median", kernelsize = 21, plot = True)

#df = pd.read_csv(path+"Dove-C_matches.csv")
# # #df = pd.read_csv(path+"sd_remapped_matches.csv")

for i in range(35, len(df)):
    # i = 9
    raw_correlate_and_correct(df.ref[i], df.sec[i], demname, ul_lon, ul_lat, demres = 3, dem_err_x = dem_err_x, dem_err_y = dem_err_y, plot = False, reduce = 5)    
    
# df = pd.read_csv(path+"sd_remapped_matches.csv")
# for i in range(len(df)):
# #i = 9
#     correlate_remapped_img_pairs(df.ref[i], df.sec[i], amespath)

# #     raw_correlate_and_correct(df.ref[i], df.sec[i], demname, ul_lon, ul_lat, zgrid = "estimate", demres = 3, plot = True)    
     #mp_correlate(df.ref[i], df.sec[i],  demname, ul_lon, ul_lat)
    
#TODO: function to derive ul x and y and sizes from input geojson
#TODO: automatically derive demres?! even if in epsg 4326 (or cut demres)
#TODO: catch input errors
#TODO: catch file not found in functional


# files = sorted(glob.glob(path+"stereo/*_dx_corrected_Err.tif"))
# matf.plot_offset_results(files, outfile="DoveC_dx_raw_corrected.png")


# files = sorted(glob.glob(path+"stereo/*_dy_corrected_Err.tif"))
# matf.plot_offset_results(files, outfile="DoveC_dy_raw_corrected.png")

# # for f in files:
# #     matf.mapproject(f, dem = demname, amespath = amespath)
# files = sorted(glob.glob("/home/ariane/Documents/PlanetScope/SD_Jujuy_Nadir/stereo/*_remap-F.tif"))
# #matf.plot_offset_results(files, band = 1, outfile="dx_corrected_iterative_reduced_bm_mean_remap.png", meanshift = False)
# for f in files: 
#     calc_velocity(f, "./SD_Jujuy_Nadir/20220924_134300_71_2212_1B_AnalyticMS_b2_clip.tif")
# # refimg = cv.imread(ref_img_fn, cv.IMREAD_UNCHANGED)
# cv.imwrite(df.ref[i][:-4]+"_raw.tif", refimg)

# refimg = cv.imread(df.ref[i][:-4]+"_raw.tif", cv.IMREAD_UNCHANGED)
# t = np.where(~refimg.any(axis=1))[0]

# files = sorted(glob.glob("/home/ariane/Documents/PlanetScope/SD_Jujuy_5deg/*_remap.tif"))

# for f in files:#[df.ref[i][:-4]+"_raw.tif"]:#files: 
#     cmd = f"gdal_translate -srcwin 0 278 3200 1722 {f} {f[:-4]}_clippedBorder.tif"
#     os.system(cmd)
