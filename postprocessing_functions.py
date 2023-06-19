#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 22:48:36 2023

@author: ariane
"""


import pandas as pd
import asp_helper_functions as asp
from helper_functions import get_scene_id, get_date, read_file, copy_rpcs, save_file, read_transform, get_extent, min_max_scaler
import datetime, os, subprocess
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
import rasterio
from scipy.stats import circmean, circstd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
import cv2


def calc_velocity(fn, dt, fixed_res = None, medShift = False):
    
    #NOTE: if the temporal baseline is short, the background noise of typically +-1-2 pixels will result in abnormally high velocities
    #therefore, only use these pairs if the landslide is fast moving
    
    # load autoRIFT output
    with rasterio.open(fn) as src:
        # get raster resolution from metadata
        if fixed_res is None:
            meta = src.meta
            res = meta["transform"][0]
        else:
            res = fixed_res
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
    
    if medShift: 
        dx = dx - np.nanmedian(dx)
        dy = dy - np.nanmedian(dy)

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
    #there np.tensordot is needed (instead of np.dot) because of the multiple dimensions of the input arrays
    dot_product = np.tensordot(unit_vector_1,unit_vector_2, axes=([0],[2]))

    direction = np.rad2deg(np.arccos(dot_product))
    
    #as always the smallest angle to north is given, values need to be substracted from 360 if x is negative
    subtract = np.zeros(dx.shape)
    subtract[dx<0] = 360
    direction = abs(subtract-direction)
    
    return v, direction


def calc_velocity_L3B(matchfile, prefixext="L3B", overwrite = False, medShift = True): 
    df = pd.read_csv(matchfile)
    path,_ = os.path.split(matchfile)

    df["id_ref"] = df.ref.apply(get_scene_id, level = 3)
    df["id_sec"] = df.sec.apply(get_scene_id, level = 3)
    df["date_ref"] = df.id_ref.apply(get_date)
    df["date_sec"] = df.id_sec.apply(get_date)
    
    df["dt"]  = df.date_sec - df.date_ref
    
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        disp = f"{path}/stereo/{row.id_ref}_{row.id_sec}{prefixext}-F.tif"
        #disp = f"{path}/stereo/{row.id_ref}_{row.id_sec}_clip_mp-F.tif"
        if os.path.isfile(disp):
            if not os.path.isfile(disp[:-4]+"_velocity.tif") or overwrite: 
                v, direction = calc_velocity(disp, row["dt"], medShift=medShift)

                save_file([v], disp, outname = disp[:-4]+"_velocity.tif")
        else:
            print(f"Warning: Disparity file {disp} not found. Skipping velocity calculation...")
    

def offset_stats_pixel(r, xcoord, ycoord, pad = 0, resolution = None, dt = None, take_velocity = True, angles = False):
    r[r==-9999] = np.nan
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

def offset_stats_aoi(r, mask, resolution, dt = None, take_velocity = True, angles = False):
    r[r==-9999] = np.nan

    if not take_velocity:
        if dt is None or resolution is None: 
            print("Need to provide a time difference and raster resolution when getting stats for dx/dy.")
            return
        #calculating displacement in m/yr to make things comparable
        r = ((r*resolution)/dt)*365
    
    try:    
        sample = r[mask == 1]
    except IndexError:
        print("There seems to be a problem with your input scene. Likely the dimensions do not fit the rest of the data. Have you altered your x/ysize or coordinates of the upper left corner when correlating scenes?")
        return np.nan, np.nan, np.nan, np.nan
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


def compare_surroundings_to_aoi(matchfile, aoi, level = 3, prefixext = "L3B"):
    
    df = pd.read_csv(matchfile)
    path,_ = os.path.split(matchfile)

    df["id_ref"] = df.ref.apply(get_scene_id, level = level)
    df["id_sec"] = df.sec.apply(get_scene_id, level = level)
    df["date_ref"] = df.id_ref.apply(get_date)
    df["date_sec"] = df.id_sec.apply(get_date)
    
    if os.path.isfile("./temp.tif"):
        os.remove("./temp.tif")
        
    diffsdx = []
    diffsdy = []
        
    for idx,row in tqdm(df.iterrows(), total=df.shape[0]):
        fn = os.path.join(path, f"stereo/{row.id_ref}_{row.id_sec}{prefixext}-F.tif")
        diffdx = np.nan
        diffdy = np.nan
        
        if os.path.isfile(fn):
            
            dx = read_file(fn,1)
            dy = read_file(fn,2)
            dmask = read_file(fn,3)
            dx[dmask == 0] = np.nan
            dy[dmask == 0] = np.nan
            
            if not os.path.isfile("./temp.tif"):
                #only calculating the mask once - all images should have the same extent
                #rasterize aoi to find the pixels inside
                extent = get_extent(fn)
                resolution = read_transform(fn)[0]
                #TODO: catch AOI having a different CRS that mapprojected rasters!
                cmd = f"gdal_rasterize -tr {resolution} {resolution} -burn 1 -a_nodata 0 -ot Int16 -of GTiff -te {' '.join(map(str,extent))} {aoi} ./temp.tif"
                subprocess.run(cmd, shell = True)

            mask = read_file("./temp.tif")
              
            dx_aoi = dx.copy()
            dy_aoi = dy.copy()
            
            dx_aoi[mask == 0] = np.nan
            dy_aoi[mask == 0] = np.nan
            
            mdx_aoi = np.nanmean(dx_aoi)
            mdy_aoi = np.nanmean(dy_aoi)
            
            diffdx = abs(mdx_aoi)-abs(np.nanmean(dx))
            diffdy = abs(mdy_aoi)-abs(np.nanmean(dy))

        diffsdx.append(diffdx)
        diffsdy.append(diffdy)
        
    df["diff_dx"] = diffsdx
    df["diff_dy"] = diffsdy
    
    
    df.to_csv(matchfile[:-4]+"_offset_aoi_diff.csv", index = False)
    
    return df

def compare_two_aois(matchfile, aoi, level=3, prefixext="L3B"):
    # aoi should have two polygons with an attribute field called "id" containing the values 1 and 2
    df = pd.read_csv(matchfile)
    path, _ = os.path.split(matchfile)

    df["id_ref"] = df.ref.apply(get_scene_id, level=level)
    df["id_sec"] = df.sec.apply(get_scene_id, level=level)
    df["date_ref"] = df.id_ref.apply(get_date)
    df["date_sec"] = df.id_sec.apply(get_date)

    if os.path.isfile("./temp.tif"):
        os.remove("./temp.tif")

    diffs_dx_mean = []    
    diffs_dy_mean = []    
    diffs_dx_median = []  
    diffs_dy_median = []

    for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
        fn = os.path.join(path, f"stereo/{row.id_ref}_{row.id_sec}{prefixext}-F.tif")
        diffdx_mean = np.nan
        diffdy_mean = np.nan
        diffdx_median = np.nan
        diffdy_median = np.nan

        if os.path.isfile(fn):
            dx = read_file(fn, 1)
            dy = read_file(fn, 2)
            dmask = read_file(fn, 3)
            dx[dmask == 0] = np.nan
            dy[dmask == 0] = np.nan

            if not os.path.isfile("./temp.tif"):
                # only calculating the mask once - all images should have the same extent
                # rasterize aoi to find the pixels inside
                extent = get_extent(fn)
                resolution = read_transform(fn)[0]
                # TODO: catch AOI having a different CRS that mapprojected rasters!
                cmd = f"gdal_rasterize -tr {resolution} {resolution} -a id -a_nodata 0 -ot Int16 -of GTiff -te {' '.join(map(str,extent))} {aoi} ./temp.tif"
                subprocess.run(cmd, shell=True)

            mask = read_file("./temp.tif")

            dx_aoi1 = dx.copy()
            dy_aoi1 = dy.copy()

            dx_aoi1[mask != 1] = np.nan
            dy_aoi1[mask != 1] = np.nan

            dx_aoi2 = dx.copy()
            dy_aoi2 = dy.copy()

            dx_aoi2[mask != 2] = np.nan
            dy_aoi2[mask != 2] = np.nan

            diffdx_mean = np.nanmean(dx_aoi2) - np.nanmean(dx_aoi1)
            diffdy_mean = np.nanmean(dy_aoi2) - np.nanmean(dy_aoi1)

            diffdx_median = np.nanmedian(dx_aoi2) - np.nanmedian(dx_aoi1)
            diffdy_median = np.nanmedian(dy_aoi2) - np.nanmedian(dy_aoi1)

        diffs_dx_mean.append(diffdx_mean)
        diffs_dy_mean.append(diffdy_mean)
        diffs_dx_median.append(diffdx_median)
        diffs_dy_median.append(diffdy_median)

    df["diff_dx_mean"] = diffs_dx_mean
    df["diff_dy_mean"] = diffs_dy_mean
    df["diff_dx_median"] = diffs_dx_median
    df["diff_dy_median"] = diffs_dy_median

    df.to_csv(matchfile[:-4] + "_offset_two_aois_compared.csv", index=False)

    return df



def get_variance(matchfile, aoi = None, inverse = False, level = 3, prefixext = "L3B"):
    
    df = pd.read_csv(matchfile)
    path,_ = os.path.split(matchfile)

    df["id_ref"] = df.ref.apply(get_scene_id, level = level)
    df["id_sec"] = df.sec.apply(get_scene_id, level = level)
    df["date_ref"] = df.id_ref.apply(get_date)
    df["date_sec"] = df.id_sec.apply(get_date)
    
    if os.path.isfile("./temp.tif"):
        os.remove("./temp.tif")
        
    varsdx = []
    varsdy = []
    
    for idx,row in tqdm(df.iterrows(), total=df.shape[0]):
        fn = os.path.join(path, f"stereo/{row.id_ref}_{row.id_sec}{prefixext}-F.tif")
        vardx = np.nan
        vardy = np.nan
        if os.path.isfile(fn):
            
            dx = read_file(fn,1)
            dy = read_file(fn,2)
            dmask = read_file(fn,3)
            dx[dmask == 0] = np.nan
            dy[dmask == 0] = np.nan
            
            if aoi is not None: 
                if not os.path.isfile("./temp.tif"):
                    #only calculating the mask once - all images should have the same extent
                    #rasterize aoi to find the pixels inside
                    extent = get_extent(fn)
                    resolution = read_transform(fn)[0]
                    #TODO: catch AOI having a different CRS that mapprojected rasters!
                    cmd = f"gdal_rasterize -tr {resolution} {resolution} -burn 1 -a_nodata 0 -ot Int16 -of GTiff -te {' '.join(map(str,extent))} {aoi} ./temp.tif"
                    if inverse: 
                        cmd += " -i"
                    subprocess.run(cmd, shell = True)
    
                mask = read_file("./temp.tif")
                
                dx[mask == 0] = np.nan
                dy[mask == 0] = np.nan
            
            vardx = np.nanvar(dx)
            vardy = np.nanvar(dy)

        varsdx.append(vardx)
        varsdy.append(vardy)
        
    df["var_dx"] = varsdx
    df["var_dy"] = varsdy
    
    df.to_csv(matchfile[:-4]+"_offset_variance.csv", index = False)
    
    return df
    

def generate_timeline(matchfile, aoi = None, xcoord = None, ycoord = None, pad = 0, max_dt = 861, weigh_by_dt = True, take_velocity = True):
    
    assert aoi is not None or (xcoord is not None and ycoord is not None), "Please provide either an AOI (vector dataset) or x and y coordinates!"
   
    if aoi is not None:
        print("Calculating velocity inside AOI...")
    else:
        print(f"Calculating at pixel value {xcoord} {ycoord} with a padding of {pad} pixels...")

    df = pd.read_csv(matchfile)
    path,_ = os.path.split(matchfile)
    if os.path.isfile("./temp.tif"):
        os.remove("./temp.tif")

    df["id_ref"] = df.ref.apply(get_scene_id)
    df["id_sec"] = df.sec.apply(get_scene_id)
    df["date_ref"] = df.id_ref.apply(get_date)
    df["date_sec"] = df.id_sec.apply(get_date)
    
    df["dt"]  = df.date_sec - df.date_ref
    #introduce upper timelimit
    df = df[df.dt <= datetime.timedelta(days=max_dt)].reset_index(drop = True)
    timeline = pd.concat([df.date_ref, df.date_sec]).drop_duplicates().reset_index(drop = True)

    #extract statistics from disparity files
    if take_velocity:
        print("Using velocity to generate timeline...")
        ext = "_imgspace_velocity_mp"
        colnames = ["vel", "vel_p25", "vel_p75", "vel_std", "ang", "ang_std"]
        angles = True
        timeline_stats = np.zeros([len(timeline), 7])

    else: #TODO: tis needs refinement since I am only mapprojecting the velocity
        print("Using mapprojected dx/dy to generate timeline...") #use the mapprojjected version to make sure that raster res is exactly 3 m 
        ext = "_mp"
        colnames = ["dx", "dx_p25", "dx_p75", "dx_std", "dy", "dy_p25", "dy_p75", "dy_std"]
        angles = False
        timeline_stats = np.zeros([len(timeline), 10])

    stats = np.zeros((len(df), 9))
    stats[:] = np.nan
    timeline_stats[:] = np.nan

    for index, row in tqdm(df.iterrows(), total=df.shape[0]):

        disp = f"{path}/stereo/{row.id_ref}_{row.id_sec}_remap-F{ext}.tif"
        #disp = f"{path}/stereo/{row.id_ref}_{row.id_sec}_clip_mp-F_velocity.tif"


        if os.path.isfile(disp):
           # print(disp)
            
            if aoi is not None:
                
                if not os.path.isfile("./temp.tif"):
                    #only calculating the mask once - all images should have the same extent
                    #rasterize aoi to find the pixels inside
                    extent = get_extent(disp)
                    resolution = read_transform(disp)[0]
                    #TODO: catch AOI having a different CRS that mapprojected rasters!
                    cmd = f"gdal_rasterize -tr {resolution} {resolution} -burn 1 -a_nodata 0 -ot Int16 -of GTiff -te {' '.join(map(str,extent))} {aoi} ./temp.tif"
                    subprocess.run(cmd, shell = True)
    
                    mask = read_file("./temp.tif")
                
                #get mean in sample region and iqr/p75 (weight) for dx or velocity
                stats[index,0], stats[index,1], stats[index,2], stats[index,3]  = offset_stats_aoi(read_file(disp, 1), mask, resolution = resolution, dt = row["dt"].days, take_velocity = take_velocity, angles = False)
                #same for dy or direction
                #stats[index,4], stats[index,5], stats[index,6], stats[index,7]  = offset_stats_aoi(read_file(disp, 2), mask, resolution = resolution, dt = row["dt"].days, take_velocity = take_velocity, angles = angles)
                
                if stats[index,0] > 100:
                    print(f"Warning! {disp} exceeds 100 m.")
            else:
                #get mean in sample region and iqr/p75 (weight) for dx or velocity
                stats[index,0], stats[index,1], stats[index,2], stats[index,3]  = offset_stats_pixel(read_file(disp, 1), xcoord, ycoord, pad, resolution = resolution, dt = row["dt"].days, take_velocity = take_velocity, angles = False)
                #same for dy or direction
                stats[index,4], stats[index,5], stats[index,6], stats[index,7]  = offset_stats_pixel(read_file(disp, 2), xcoord, ycoord, pad, resolution = resolution, dt = row["dt"].days, take_velocity = take_velocity, angles = angles)
                
            stats[index, 8]  = row["dt"].days
            
        else:
          print(f"Warning! Disparity file {disp} not found.")
    
    #TODO: improve quick fix for not having the direction in the mapprojected scene: fill rows to keep them
    
    stats[:,4] = 1
    stats[:,5] = 1
    #deleting temporary mask raster
    os.remove("./temp.tif")
    #generate timeline assuming linear velocity between first and second image and performing a weighted average for each day of change
    
    #removing columns if disparity is not available
    df = df[~np.all(np.isnan(stats), axis=1)].reset_index(drop = True)

    #drop nan columns
    stats = stats[~np.all(np.isnan(stats), axis=1)]
    stats = stats[:,~np.all(np.isnan(stats), axis=0)]

    #for storing all data, not just the average
    timeline_alldata = np.empty((0,stats.shape[1]))
    count = np.zeros((len(timeline)))
    for i, date in enumerate(timeline): 
        #print(date)
        active = np.zeros(len(df)).astype(bool)
        active[(df.date_ref<=date) & (df.date_sec >= date)] = True #filter image pairs that have correlation results at this date
        
        if len(stats[active]) > 0:
            if take_velocity: #velocity case
                
                if weigh_by_dt:
                    dt_weights = min_max_scaler(stats[active, -1])
                    disp_weights = min_max_scaler(1/stats[active,2])
                    weights = dt_weights + disp_weights
                else: 
                    weights = 1/stats[active,2]
                    
                timeline_stats[i,:-1] = np.average(stats[active,:6], axis=0, weights= weights)
                timeline_stats[i,-1]  = np.sqrt(np.cov(stats[active,0], aweights=weights)) #weighted standard deviation

            else: #dx/dy separate case
            
                if weigh_by_dt:
                    dt_weights = min_max_scaler(stats[active, -1])
                    disp_weights1 = min_max_scaler(1/(stats[active, 2]-stats[active, 1]))
                    disp_weights2 = min_max_scaler(1/(stats[active, 5]-stats[active, 4]))
                    weights1 = dt_weights + disp_weights1
                    weights2 = dt_weights + disp_weights2

                else:
                    weights1 = 1/(stats[active, 2]-stats[active, 1]) #IQR
                    weights2 = 1/(stats[active, 6]-stats[active, 5])
                    
                                               
                #separate weights for dx and dy
                timeline_stats[i,:4] = np.average(stats[active, :4], weights = weights1, axis=0)
                timeline_stats[i,4:-2] = np.average(stats[active, 4:-1], weights = weights2, axis=0)
                
                timeline_stats[i,-2]  = np.sqrt(np.cov(stats[active,0], aweights=weights1)) #weighted standard deviation
                timeline_stats[i,-1]  = np.sqrt(np.cov(stats[active,4], aweights=weights2)) #weighted standard deviation

        timeline_alldata = np.concatenate((timeline_alldata,stats[active, :]), axis = 0)
        #print(len(stats[active, :]))
        count[i] = len(stats[active,:])
        
    if take_velocity:
        out = pd.DataFrame(timeline_stats, columns =  [*colnames, "mean_vel_avg_std"])
    else: 
        out = pd.DataFrame(timeline_stats, columns =  [*colnames, "mean_vel_avg_std_dx", "mean_vel_avg_std_dy"])

    out["date"] = timeline
    out["count"] = count
    
    
    colnames.append("dt")
    timeline_alldata = pd.DataFrame.from_records(timeline_alldata, columns = colnames)
    timeline_alldata["date"] = timeline.repeat(count.astype(int)).reset_index(drop = True)
    
        
    if aoi is not None:
        timeline_alldata.to_csv(f"{path}/timeline_alldata_aoi{ext}_new.csv", index = False)
        out.to_csv(f"{path}/timeline_averaged_aoi{ext}_new.csv", index = False)
    else:
        
        timeline_alldata.to_csv(f"{path}/timeline_alldata_x{xcoord}_y{ycoord}_pad{pad}{ext}_new.csv", index = False)
        out.to_csv(f"{path}/timeline_averaged_x{xcoord}_y{ycoord}_pad{pad}{ext}_new.csv", index = False)

    
        
        # img = read_file(df.ref[0])
        # rect = patches.Rectangle((xcoord-pad, ycoord-pad), pad*2+1, pad*2+1, linewidth=1, edgecolor='r', facecolor='none')
        # fig, ax = plt.subplots()
        # ax.imshow(img, cmap = "Greys")
        # ax.add_patch(rect)
        # #plt.savefig("show_area.png", dpi = 400)
        # plt.show()
        
def get_stats_for_allpairs(matchfile, take_velocity = True):
    #good for heatmaps
    df = pd.read_csv(matchfile)
    path,_ = os.path.split(matchfile)
    if os.path.isfile("./temp.tif"):
        os.remove("./temp.tif")

    df["id_ref"] = df.ref.apply(get_scene_id)
    df["id_sec"] = df.sec.apply(get_scene_id)
    df["date_ref"] = df.id_ref.apply(get_date)
    df["date_sec"] = df.id_sec.apply(get_date)
    
    df["dt"]  = df.date_sec - df.date_ref

    #extract statistics from disparity files
    if take_velocity:
        print("Getting velocity stats...")
        ext = "_imgspace_velocity_mp"
        stats = np.zeros((len(df), 3))
        stats[:] = np.nan
        colnames = ["v_median", "v_p25", "v_p75"]

    else: 
        print("Getting mapprojected dx/dy stats...") #use the mapprojjected version to make sure that raster res is exactly 3 m 
        ext = "_mp"
        stats = np.zeros((len(df), 6))
        stats[:] = np.nan
        colnames = ["dx_median", "dx_p25", "dx_p75", "dy_median", "dy_p25", "dy_p75"]


    for index, row in tqdm(df.iterrows(), total=df.shape[0]):

        disp = f"{path}/stereo/{row.id_ref}_{row.id_sec}_remap-F{ext}.tif"
        #disp = f"{path}/stereo/{row.id_ref}_{row.id_sec}_clip_mp-F.tif"

        if os.path.isfile(disp):
            if take_velocity:
                vel = read_file(disp)
                stats[index, 0] = np.nanmedian(vel)
                stats[index, 1] = np.nanpercentile(vel, 25)
                stats[index, 2] = np.nanpercentile(vel, 75)
            else:
                dx = read_file(disp)
                stats[index, 0] = np.nanmedian(dx)
                stats[index, 1] = np.nanpercentile(dx, 25)
                stats[index, 2] = np.nanpercentile(dx, 75)
                dy = read_file(disp)
                stats[index, 3] = np.nanmedian(dy)
                stats[index, 4] = np.nanpercentile(dy, 25)
                stats[index, 5] = np.nanpercentile(dy, 75)
            
    out = pd.DataFrame(stats, columns = colnames)
    out = pd.concat([df, out], axis = 1)

    out.to_csv(f"{path}/stats{ext}.csv", index = False)
    
    



def stack_rasters_weightfree(matchfile, prefixext = "L3B", what = "velocity"):
    
    df = pd.read_csv(matchfile)
    path,fn = os.path.split(matchfile) #! Assumes that all files are stored in the directory of the matchfile. Might refine that in the future.
    
    df["id_ref"] = df.ref.apply(get_scene_id, level = 3)
    df["id_sec"] = df.sec.apply(get_scene_id, level = 3)
    df["date_ref"] = df.id_ref.apply(get_date)
    df["date_sec"] = df.id_sec.apply(get_date)
    df["dt"]  = df.date_sec - df.date_ref
        
    if what == "velocity": 
        df["filenames"]  = path+"/stereo/"+df.id_ref+"_"+df.id_sec+prefixext+"-F_velocity.tif"
        array_list = [np.ma.masked_invalid(read_file(x,1)) for x in df.filenames if os.path.isfile(x)]
        
    elif what == "dx":
        df["filenames"]  = path+"/stereo/"+df.id_ref+"_"+df.id_sec+prefixext+"-F.tif"
        
        array_list = [read_file(x,1) for x in df.filenames if os.path.isfile(x)]
        mask_list = [read_file(x,3) for x in df.filenames if os.path.isfile(x)]

        dt = [df["dt"][i].days for i in range(len(df)) if os.path.isfile(df.filenames[i])]
        resolution = [read_transform(df.filenames[i])[0] for i in range(len(df)) if os.path.isfile(df.filenames[i])]
        
        for i in range(len(dt)):
            #masking
            masked = np.where(mask_list[i] == 1, array_list[i], np.nan)
            #median shift 
            med = np.nanmedian(masked)
            masked = masked - med
            
            ####TODO: remove this
            #siguas
            # if (np.nanmedian(masked[1300:1500, 1200:1450])<0):
            #     masked = masked * -1
            # if (np.nanmedian(masked[1000:1200,900:1100])<0):
            #     masked = masked * -1
                
            ####  
            #dx in m/yr
            array_list[i] = np.ma.masked_invalid(((masked*resolution[i])/dt[i])*365)
            

    elif what == "dy":
        df["filenames"]  = path+"/stereo/"+df.id_ref+"_"+df.id_sec+prefixext+"-F.tif"
        
        array_list = [read_file(x,2) for x in df.filenames if os.path.isfile(x)]
        mask_list = [read_file(x,3) for x in df.filenames if os.path.isfile(x)]

        dt = [df["dt"][i].days for i in range(len(df)) if os.path.isfile(df.filenames[i])]
        resolution = [read_transform(df.filenames[i])[0] for i in range(len(df)) if os.path.isfile(df.filenames[i])]

        for i in range(len(dt)):
            #masking
            masked = np.where(mask_list[i] == 1, array_list[i], np.nan)
            #median shift 
            med = np.nanmedian(masked)
            masked = masked - med

            #dy in m/yr
            array_list[i] = np.ma.masked_invalid(((masked*resolution[i])/dt[i])*365)

    else:
        print("No valid what option provided.")
        return
    
    average_vals = np.ma.average(array_list, axis=0)
    variance_vals = np.ma.var(array_list, axis = 0)
    save_file([average_vals, variance_vals], df.filenames[0], os.path.join(path,fn[:-4] + f"_average_{what}.tif"))

    


def stack_rasters(matchfile, take_velocity = True, max_dt = 861):
    df = pd.read_csv(matchfile)
    path,_ = os.path.split(matchfile) #! Assumes that all files are stored in the directory of the matchfile. Might refine that in the future.
    
    df["id_ref"] = df.ref.apply(get_scene_id)
    df["id_sec"] = df.sec.apply(get_scene_id)
    df["date_ref"] = df.id_ref.apply(get_date)
    df["date_sec"] = df.id_sec.apply(get_date)
    df["dt"]  = df.date_sec - df.date_ref
    #introduce upper timelimit
    df = df[df.dt <= datetime.timedelta(days=max_dt)].reset_index(drop = True)
    
    if take_velocity: 
        print("Stacking velocity...")
        
        ext = "imgspace_velocity_mp"
        df["filenames"]  = path+"/stereo/"+df.id_ref+"_"+df.id_sec+"_remap-F_"+ext+".tif"
        #TODO: check which one is the nodata value
        #array_list = [np.ma.masked_invalid(read_file(x,1)) for x in df.filenames if os.path.isfile(x)]
        array_list = [np.ma.masked_equal(read_file(x,1), -9999) for x in df.filenames if os.path.isfile(x)]

        weights = [1/np.nanpercentile(a.data,75) for a in array_list]
        average_velocity = np.ma.average(array_list, axis=0, weights=weights)
        save_file([average_velocity], df.filenames[0], os.path.join(path,"average_velocity.tif"))
   
    else: 
        
        df["date_ref"] = df.id_ref.apply(get_date)
        df["date_sec"] = df.id_sec.apply(get_date)
        
        df["dt"]  = df.date_sec - df.date_ref
        
        ext = "mp"
        df["filenames"]  = path+"/stereo/"+df.id_ref+"_"+df.id_sec+"_remap-F_"+ext+".tif"
        
        array_list_dx = [read_file(x,1) for x in df.filenames if os.path.isfile(x)]
        array_list_dy = [read_file(x,2) for x in df.filenames if os.path.isfile(x)]

        dt = [df["dt"][i].days for i in range(len(df)) if os.path.isfile(df.filenames[i])]
        resolution = [read_transform(df.filenames[i])[0] for i in range(len(df)) if os.path.isfile(df.filenames[i])]

        for i in range(len(dt)):
            array_list_dx[i] = ((array_list_dx[i]*resolution[i])/dt[i])*365
            array_list_dy[i] = ((array_list_dy[i]*resolution[i])/dt[i])*365

        array_list_dx = [np.ma.masked_invalid(a) for a in array_list_dx]
        array_list_dy = [np.ma.masked_invalid(a) for a in array_list_dy]

        print("Averaging dx...")
        weights = [1/(np.nanpercentile(a.data,75)-np.nanpercentile(a.data,25)) for a in array_list_dx]
        weights = np.nan_to_num(weights, posinf = 0)
        average_dx = np.ma.average(array_list_dx, axis=0, weights=weights)
        
        print("Averaging dy...")
        weights = [1/(np.nanpercentile(a.data,75)-np.nanpercentile(a.data,25)) for a in array_list_dy]
        weights = np.nan_to_num(weights, posinf = 0)
        average_dy = np.ma.average(array_list_dy, axis=0, weights=weights)
        
        pot_ref = [file for file in df.filenames if os.path.isfile(file)]
        save_file([average_dx, average_dy], pot_ref[0], os.path.join(path,"average_dx_dy.tif"))

