#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 16:02:02 2023

@author: ariane
"""
import pandas as pd
import helper_functions as helper
import os
import cv2 as cv
import numpy as np
from optimization_functions import percentile_cut, polyXY1, polyXY2, polyXY3, polyXYZ1, polyXYZ2, polyXYZ3
import scipy.optimize

def run_opt_flow(matches, flip = False, overwrite = False, prefix_ext = "", kernel_size = 35):
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
    
    if flip: # flipping dataframe. makes sense if running correlation on multiple machines, 
        df = df.reindex(index=df.index[::-1]).reset_index(drop = True)

    df["id_ref"] = df.ref.apply(helper.get_scene_id)
    df["id_sec"] = df.sec.apply(helper.get_scene_id)
    df["path"] =  df["ref"].apply(lambda x: os.path.split(x)[0])
    
    out = []
    for _,row in df.iterrows():
        if not os.path.isdir(row.path + "/opt_flow"):
            os.mkdir(row.path + "/opt_flow")
        print(f"Disparity map will be stored under {row.path}/opt_flow ...")
        prefix = row.id_ref + "_" + row.id_sec + prefix_ext
        if (not os.path.isfile(os.path.join(row.path,"opt_flow",prefix+".tif"))) or overwrite:
            img1 = cv.imread(row.ref, cv.IMREAD_UNCHANGED)
            img2 = cv.imread(row.sec, cv.IMREAD_UNCHANGED)

            flow = cv.calcOpticalFlowFarneback(img1, img2, None, 0.5, 10, kernel_size, 5, 5, 1.2, 0)
            dx = flow[:,:,0]
            dy = flow[:,:,1]
            dx[img1 == 0] = np.nan
            dy[img1 == 0] = np.nan

            helper.save_file([dx,dy], ref = row.ref, outname = os.path.join(row.path,"opt_flow",prefix+".tif"))
        else: 
            print("Disparity map exists. Skipping correlation...")
        out.append(os.path.join(row.path,"opt_flow",prefix+".tif"))
        
def apply_polyfit(matches, prefix_ext= "", order = 2, demname = None, plimlow = 5, plimup = 95, save_remapped_sec = False, overwrite = True):
  
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
        dispfn = os.path.join(path, "opt_flow", prefix+".tif")
        if os.path.isfile(dispfn):
            #print(dispfn)
            if overwrite or (not os.path.isfile(dispfn[:-4]+"_polyfit.tif")):
                dx = helper.read_file(dispfn, b = 1)
                dy = helper.read_file(dispfn, b = 2)

                
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
                    if len(np.unique(zgrid)) == 1: 
                        print("Only NoData values found in zgrid. Make sure the reference DEM covers the extent of the disparity maps!")
                        return
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
                
                helper.save_file([dx,dy], dispfn, dispfn[:-4]+"_polyfit.tif")
            else:
                print(dispfn[:-4]+"_polyfit.tif exists. Skipping polyfit...")
            out.append(dispfn[:-4]+"_polyfit.tif")

    return out
            
            
  