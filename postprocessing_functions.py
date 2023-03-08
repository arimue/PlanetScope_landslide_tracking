#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 22:48:36 2023

@author: ariane
"""


import pandas as pd
import asp_helper_functions as asp
from helper_functions import get_scene_id, get_date, read_file, copy_rpcs, save_file
import datetime, os, subprocess
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
import rasterio
from scipy.stats import circmean, circstd


def calc_velocity(fn, dt):
    

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
        
    if dt.days < 0: #invert velocity if negative time difference (ref younger than sec)
        dx = dx*-1
        dy = dy*-1
    
    dx[valid == 0] = np.nan
    dy[valid == 0] = np.nan
    
    #calculate total offset (length of vector)
    v = np.sqrt((dx**2+dy**2))
    #convert to meter
    v = v * res
    #convert to meter/year (year)
    v = (v/abs(dt.days))*365
    
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
    
    return v, direction
    
def mapproject_and_calc_velocity(amespath, matchfile, dem, img_with_rpc, resolution = 3, epsg = "32720", ext = "mp", overwrite = False):
    df = pd.read_csv(matchfile)
    path,_ = os.path.split(matchfile)

    df["id_ref"] = df.ref.apply(get_scene_id)
    df["id_sec"] = df.sec.apply(get_scene_id)
    df["date_ref"] = df.id_ref.apply(get_date)
    df["date_sec"] = df.id_sec.apply(get_date)
    
    df["dt"]  = df.date_sec - df.date_ref

    #extract statistics from disparity files
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        disp = f"{path}/stereo/{row.id_ref}_{row.id_sec}_remap-F.tif"
        #copy_rpcs(img_with_rpc, disp)
        if os.path.isfile(disp):
            if not os.path.isfile(f"{disp[:-4]}_{ext}.tif") or overwrite:
                #if the mapprojected result doesnt show in QGIS, make sure to remove Band 4 as the alpha band from the transparency settings
                output = asp.mapproject(amespath, disp, dem, img_with_rpc, ext = ext, resolution = resolution, epsg = epsg)       
            else:
                print("Mapprojected disparity existis. Skipping mapprojection...")
                output = f"{disp[:-4]}_{ext}.tif"
        else:
            print(f"Warning! Disparity file {disp} not found.")
            pass
        
        if not os.path.isfile(disp[:-4]+"_velocity.tif") or overwrite: 
            v, direction = calc_velocity(output, row["dt"])
            save_file([v,direction], output, outname = disp[:-4]+"_velocity.tif")
        else:
            print("Mapprojected disparity existis. Skipping mapprojection...")


def offset_stats(r, xcoord, ycoord, pad = 0, resolution = None, dt = None, take_velocity = True, angles = False):
    if not take_velocity:
        if dt is None or resolution is None: 
            print("Need to provide a time difference and raster resolution when getting stats for dx/dy.")
            return
        #calculating displacement in m/yr to make things comparable
        r = ((r*resolution)/dt)*365

    sample = r[ycoord-pad:ycoord+pad+1, xcoord-pad:xcoord+pad+1]
    
    if angles: #calculate circular mean for direction
        mean = np.rad2deg(circmean(np.deg2rad(sample)))
        p75 = np.nan
        p25 = np.nan
        std = np.rad2deg(circstd(np.deg2rad(sample)))
    else: 
        mean = np.nanmean(sample)
        p75 = np.nanpercentile(r, 75)
        p25 = np.nanpercentile(r, 25)
        std = np.nanstd(r)
    return mean, p25, p75, std


def generate_timeline(matchfile, xcoord, ycoord, pad = 0, resolution = 3, take_velocity = True, plot = False):
    df = pd.read_csv(matchfile)
    path,_ = os.path.split(matchfile)

    df["id_ref"] = df.ref.apply(get_scene_id)
    df["id_sec"] = df.sec.apply(get_scene_id)
    df["date_ref"] = df.id_ref.apply(get_date)
    df["date_sec"] = df.id_sec.apply(get_date)
    
    df["dt"]  = df.date_sec - df.date_ref
    timeline = pd.concat([df.date_ref, df.date_sec]).drop_duplicates().reset_index(drop = True)

    #extract statistics from disparity files
    if take_velocity:
        print("Using velocity to generate timeline...")
        ext = "_velocity"
        colnames = ["vel", "vel_p25", "vel_p75", "vel_std", "ang", "ang_std"]
        angles = True
        timeline_stats = np.zeros([len(timeline), 6])

    else: 
        print("Using mapprojected dx/dy to generate timeline...") #use the mapprojjected version to make sure that raster res is exactly 3 m 
        ext = "_mp"
        colnames = ["dx", "dx_p25", "dx_p75", "dx_std", "dy", "dy_p25", "dy_p75", "dy_std"]
        angles = False
        timeline_stats = np.zeros([len(timeline), 8])

    stats = np.zeros((len(df), 8))
    timeline_stats[:] = np.nan

    for index, row in tqdm(df.iterrows(), total=df.shape[0]):

        disp = f"{path}/stereo/{row.id_ref}_{row.id_sec}_remap-F{ext}.tif"

        if os.path.isfile(disp):

            #get mean in sample region and iqr (weight) for dx
            stats[index,0], stats[index,1], stats[index,2], stats[index,3]  = offset_stats(read_file(disp, 1), xcoord, ycoord, pad, resolution = resolution, dt = row["dt"].days, take_velocity = take_velocity, angles = False)
            #same for dy
            stats[index,4], stats[index,5], stats[index,6], stats[index,7]  = offset_stats(read_file(disp, 2), xcoord, ycoord, pad, resolution = resolution, dt = row["dt"].days, take_velocity = take_velocity, angles = angles)
        else:
            print(f"Warning! Disparity file {disp} not found.")
                              
    #generate timeline assuming linear velocity between first and second image and performing a weighted average for each day of change
    
    #removing columns if disparity is not available
    df = df[np.sum(stats, axis = 1)!=0].reset_index(drop = True)
    stats = stats[np.sum(stats, axis = 1)!=0]
    
    
    for i, date in enumerate(timeline): 
        #print(date)
        active = np.zeros(len(df)).astype(bool)
        active[(df.date_ref<=date) & (df.date_sec >= date)] = True #filter image pairs that have correlation results at this date
        
        if len(stats[active]) > 0:
            if take_velocity:
                
                timeline_stats[i,0] = np.average(stats[active, 0], weights = 1/(stats[active, 2])) #weigh only by 1/P75
                timeline_stats[i,1] = np.average(stats[active, 1], weights = 1/(stats[active, 2]))
                timeline_stats[i,2] = np.average(stats[active, 2], weights = 1/(stats[active, 2]))
                timeline_stats[i,3] = np.average(stats[active, 3], weights = 1/(stats[active, 2]))
                timeline_stats[i,4] = np.average(stats[active, 4], weights = 1/(stats[active, 2]))
                timeline_stats[i,5] = np.average(stats[active, 5], weights = 1/(stats[active, 2]))

    
            else:
                timeline_stats[i,0] = np.average(stats[active, 0], weights = 1/(stats[active, 2]-stats[active, 1])) #weighted average of velocities, weights based on IQR
                timeline_stats[i,1] = np.average(stats[active, 1], weights = 1/(stats[active, 2]-stats[active, 1])) #weighted average of percentiles
                timeline_stats[i,2] = np.average(stats[active, 2], weights = 1/(stats[active, 2]-stats[active, 1]))
                timeline_stats[i,3] = np.average(stats[active, 3], weights = 1/(stats[active, 2]-stats[active, 1]))
                timeline_stats[i,4] = np.average(stats[active, 4], weights = 1/(stats[active, 5]-stats[active, 4]))
                timeline_stats[i,5] = np.average(stats[active, 5], weights = 1/(stats[active, 5]-stats[active, 4]))
                timeline_stats[i,6] = np.average(stats[active, 6], weights = 1/(stats[active, 5]-stats[active, 4]))
                timeline_stats[i,7] = np.average(stats[active, 7], weights = 1/(stats[active, 5]-stats[active, 4]))

    if plot:
        fig, ax = plt.subplots(1,2, figsize = (15,5))
        ax[0].scatter(timeline, timeline_stats[:,0])
        ax[0].set_title("velocity x")
        ax[1].scatter(timeline, timeline_stats[:,3])
        ax[1].set_title("velocity y")
        plt.show()
        
        img = read_file(df.ref[0])
        rect = patches.Rectangle((xcoord-pad, ycoord-pad), pad*2+1, pad*2+1, linewidth=1, edgecolor='r', facecolor='none')
        fig, ax = plt.subplots()
        ax.imshow(img, cmap = "Greys")
        ax.add_patch(rect)
        plt.savefig("show_area.png", dpi = 400)
        plt.show()
        
    out = pd.DataFrame(timeline_stats, columns = colnames)
    out["date"] = timeline
    out.to_csv(f"{path}/timeline_x{xcoord}_y{ycoord}_pad{pad}{ext}.csv", index = False)

    


        
        
