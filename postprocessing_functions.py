#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 22:48:36 2023

@author: ariane
"""


import pandas as pd
import asp_helper_functions as asp
from helper_functions import get_scene_id, get_date, read_file, copy_rpcs, save_file, read_transform, get_extent, min_max_scaler, read_meta
import datetime, os, subprocess
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
import rasterio
from scipy.stats import circmean, circstd, circvar
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
import cv2


def calc_velocity(fn, dt, fixed_res = None, medShift = False):
    """
    Calculate velocity and direction of displacement between two images.
 
    Parameters:
    fn (str): Path to the input raster file.
    dt (timedelta): Temporal baseline between the two images.
    fixed_res (float, optional): Fixed raster resolution (default: None, read from metadata).
    med_shift (bool, optional): Apply median shift to displacements (default: False).
 
    Returns:
    v (numpy.ndarray): Velocity in meters per year.
    direction (numpy.ndarray): Direction of displacement in degrees with respect to north.
    """

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
        
        if meta["count"] == 3:
           # print("Interpreting the third band as good pixel mask.")
            valid = src.read(3)
            dx[valid == 0] = np.nan
            dy[valid == 0] = np.nan
        
    if dt.days < 0: #invert velocity if negative time difference (ref younger than sec)
        dx = dx*-1
        dy = dy*-1
    

    
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


def calc_velocity_wrapper(matches, prefix_ext = "", overwrite = False, medShift = True): 
    
    """
    Calculate velocity and direction of displacement for multiple image pairs.
    
    Parameters:
    matches (str or pandas.DataFrame): Path to the matchfile or a pandas DataFrame with match information.
    prefix_ext (str, optional): Prefix extension for the output files (default: "").
    overwrite (bool, optional): Overwrite existing velocity files (default: False).
    med_shift (bool, optional): Apply median shift to displacements (default: True).
    
    Returns:
    out (list): List of paths to the calculated velocity files.
    """
    
    if type(matches) == str:
        try:
            df = pd.read_csv(matches)
            path,_ = os.path.split(matches)

        except FileNotFoundError:
            print("Could not find the provided matchfile.")
            return
    elif type(matches) == pd.core.frame.DataFrame:
        df = matches.copy()
        path,_ = os.path.split(df.ref.iloc[0])
    else:
        print("Matches must be either a string indicating the path to a matchfile or a pandas DataFrame.")
        return
    
    df["id_ref"] = df.ref.apply(get_scene_id)
    df["id_sec"] = df.sec.apply(get_scene_id)
    df["date_ref"] = df.id_ref.apply(get_date)
    df["date_sec"] = df.id_sec.apply(get_date)
    
    df["dt"]  = df.date_sec - df.date_ref
    
    
    out = []
    
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        disp = os.path.join(path, "disparity_maps", f"{row.id_ref}_{row.id_sec}{prefix_ext}-F.tif")
        if os.path.isfile(disp):
            if overwrite or (not os.path.isfile(disp[:-4]+"_velocity.tif")): 
                v, direction = calc_velocity(disp, row["dt"], medShift=medShift)
                save_file([v,direction], disp, outname = disp[:-4]+"_velocity.tif")
                out.append(disp[:-4]+"_velocity.tif")
        else:
            print(f"Warning: Disparity file {disp} not found. Skipping velocity calculation...")
    
    return out

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
        p75 = np.nanpercentile(sample, 75)
        p25 = np.nanpercentile(sample, 25)
        std = np.nanstd(sample)
    return mean, p25, p75, std

def offset_stats_aoi(r, mask, resolution, dt = None, take_velocity = True):
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

    mean = np.nanmean(sample)
    median = np.nanmedian(sample)
    p75 = np.nanpercentile(sample, 75)
    p25 = np.nanpercentile(sample, 25)
    std = np.nanstd(sample)
    return mean, std, median, p25, p75


def compare_surroundings_to_aoi(matchfile, aoi, prefixext = "L3B"):
    
    df = pd.read_csv(matchfile)
    path,_ = os.path.split(matchfile)

    df["id_ref"] = df.ref.apply(get_scene_id)
    df["id_sec"] = df.sec.apply(get_scene_id)
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



def get_std_iqr(matchfile, aoi = None, inverse = False, prefixext = "L3B"):
    
    df = pd.read_csv(matchfile)
    path,_ = os.path.split(matchfile)

    df["id_ref"] = df.ref.apply(get_scene_id)
    df["id_sec"] = df.sec.apply(get_scene_id)
    df["date_ref"] = df.id_ref.apply(get_date)
    df["date_sec"] = df.id_sec.apply(get_date)
    df["path"] =  df["ref"].apply(lambda x: os.path.split(x)[0])
    
    if os.path.isfile("./temp.tif"):
        os.remove("./temp.tif")
        
    stdsdx = []
    stdsdy = []
    iqrsdx = []
    iqrsdy = []
    
    for idx,row in tqdm(df.iterrows(), total=df.shape[0]):
        fn = os.path.join(row.path, f"stereo/{row.id_ref}_{row.id_sec}{prefixext}-F.tif")
        stddx = np.nan
        stddy = np.nan
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
            
            stddx = np.nanstd(dx)
            stddy = np.nanstd(dy)
            iqrx = np.nanpercentile(dx,75)-np.nanpercentile(dx,25)
            iqry = np.nanpercentile(dy,75)-np.nanpercentile(dy,25)
        else:
            print(f"File {fn} not found. Skipping ...")

        stdsdx.append(stddx)
        stdsdy.append(stddy)
        iqrsdx.append(iqrx)
        iqrsdy.append(iqry)
        
    df["std_dx"] = stdsdx
    df["std_dy"] = stdsdy
    df["iqr_dx"] = iqrsdx
    df["iqr_dy"] = iqrsdy
    
    df = df.drop(columns = ["path"])
    df.to_csv(matchfile[:-4]+"_offset_std.csv", index = False)
    
    return df
    

def get_stats_in_aoi(matchfile, aoi = None, xcoord = None, ycoord = None, pad = 0, prefixext = "", max_dt = 10000, take_velocity = True):
    
    assert aoi is not None or (xcoord is not None and ycoord is not None), "Please provide either an AOI (vector dataset) or x and y coordinates!"
   
    if aoi is not None:
        print("Calculating velocity inside AOI...")
    else:
        print(f"Calculating at pixel value {xcoord} {ycoord} with a padding of {pad} pixels...")

    df = pd.read_csv(matchfile)
    path,matchfn = os.path.split(matchfile)
    if os.path.isfile("./temp.tif"):
        os.remove("./temp.tif")

    df["id_ref"] = df.ref.apply(get_scene_id)
    df["id_sec"] = df.sec.apply(get_scene_id)
    df["date_ref"] = df.id_ref.apply(get_date)
    df["date_sec"] = df.id_sec.apply(get_date)
    df["path"] =  df["ref"].apply(lambda x: os.path.split(x)[0])

    df["dt"]  = df.date_sec - df.date_ref
    #introduce upper timelimit
    df = df[df.dt <= datetime.timedelta(days=max_dt)].reset_index(drop = True)

    #extract statistics from disparity files
    if take_velocity:
        print("Using velocity to generate timeline...")
        ext = "_velocity"
        colnames = ["vel_mean", "vel_std", "vel_median", "vel_p25", "vel_p75"]
        stats = np.zeros([len(df), 5])

    # else: #TODO: tis needs refinement since I am only mapprojecting the velocity
    #     print("Using mapprojected dx/dy to generate timeline...") #use the mapprojjected version to make sure that raster res is exactly 3 m 
    #     ext = "_mp"
    #     colnames = ["dx", "dx_p25", "dx_p75", "dx_std", "dy", "dy_p25", "dy_p75", "dy_std"]
    #     angles = False
    #     timeline_stats = np.zeros([len(timeline), 10])

    stats[:] = np.nan

    for index, row in tqdm(df.iterrows(), total=df.shape[0]):

        disp = f"{row.path}/stereo/{row.id_ref}_{row.id_sec}{prefixext}-F{ext}.tif"
        #disp = f"{path}/stereo/{row.id_ref}_{row.id_sec}_clip_mp-F_velocity.tif"


        if os.path.isfile(disp):

            
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
                stats[index,0], stats[index,1], stats[index,2], stats[index,3], stats[index,4]  = offset_stats_aoi(read_file(disp, 1), mask, resolution = resolution, dt = row["dt"].days, take_velocity = take_velocity)
                #same for dy or direction
                #stats[index,4], stats[index,5], stats[index,6], stats[index,7]  = offset_stats_aoi(read_file(disp, 2), mask, resolution = resolution, dt = row["dt"].days, take_velocity = take_velocity, angles = angles)
                
                if stats[index,0] > 100:
                    print(f"Warning! {disp} exceeds 100 m.")
            else:
                #get mean in sample region and iqr/p75 (weight) for dx or velocity
                stats[index,0], stats[index,1], stats[index,2], stats[index,3], stats[index,4]  = offset_stats_pixel(read_file(disp, 1), xcoord, ycoord, pad, resolution = resolution, dt = row["dt"].days, take_velocity = take_velocity)

        else:
          print(f"Warning! Disparity file {disp} not found.")
        
    statsdf = pd.DataFrame(stats, columns = colnames)
    df = pd.concat([df, statsdf], axis = 1)
    
    df.to_csv(f"{path}/velocity_in_aoi_{matchfn[:-4]}.csv", index = False)


        
def get_stats_for_entire_raster(matchfile, prefixext = "L3B", take_velocity = True):
    #good for heatmaps
    df = pd.read_csv(matchfile)
    path,_ = os.path.split(matchfile)
    if os.path.isfile("./temp.tif"):
        os.remove("./temp.tif")

    df["id_ref"] = df.ref.apply(get_scene_id)
    df["id_sec"] = df.sec.apply(get_scene_id)
    df["date_ref"] = df.id_ref.apply(get_date)
    df["date_sec"] = df.id_sec.apply(get_date)
    df["path"] =  df["ref"].apply(lambda x: os.path.split(x)[0])

    df["dt"]  = df.date_sec - df.date_ref

    #extract statistics from disparity files
    if take_velocity:
        print("Getting velocity stats...")
        prefixext += "_velocity"
        stats = np.zeros((len(df), 5))
        stats[:] = np.nan
        colnames = ["v_median", "v_p25", "v_p75", "v_mean", "v_std"]

    else: 
        print("Getting dx/dy stats...") #use the mapprojected version to make sure that raster res is exactly 3 m 
        stats = np.zeros((len(df), 10))
        stats[:] = np.nan
        colnames = ["dx_median", "dx_p25", "dx_p75", "dx_mean", "dx_std", "dy_median", "dy_p25", "dy_p75", "dy_mean", "dy_std",]


    for index, row in tqdm(df.iterrows(), total=df.shape[0]):

        disp = os.path.join(row.path, f"stereo/{row.id_ref}_{row.id_sec}{prefixext}-F.tif")

        if os.path.isfile(disp):
            if take_velocity:
                vel = read_file(disp)
                stats[index, 0] = np.nanmedian(vel)
                stats[index, 1] = np.nanpercentile(vel, 25)
                stats[index, 2] = np.nanpercentile(vel, 75)
                stats[index, 3] = np.nanpmean(vel)
                stats[index, 4] = np.nanstd(vel)
            else:
                count = read_meta(disp)["count"]
                
                dx = read_file(disp)
                dy = read_file(disp, b = 2)
                
                if count == 3:
                    mask = read_file(disp, b = 3)
                    dx[mask == 0] = np.nan
                    dy[mask == 0] = np.nan
                stats[index, 0] = np.nanmedian(dx)
                stats[index, 1] = np.nanpercentile(dx, 25)
                stats[index, 2] = np.nanpercentile(dx, 75)
                stats[index, 3] = np.nanmean(dx)
                stats[index, 4] = np.nanstd(dx)
                
                stats[index, 5] = np.nanmedian(dy)
                stats[index, 6] = np.nanpercentile(dy, 25)
                stats[index, 7] = np.nanpercentile(dy, 75)
                stats[index, 8] = np.nanmean(dy)
                stats[index, 9] = np.nanstd(dy)

            
    out = pd.DataFrame(stats, columns = colnames)
    out = pd.concat([df, out], axis = 1)

    out.to_csv(f"{df.path.iloc[0]}/stats{prefixext}.csv", index = False)
    



def stack_rasters(matches, prefix_ext = "", what = "velocity", medShift = False):
    
    #TODO: make this more laptop friendly
    if type(matches) == str:
        try:
            df = pd.read_csv(matches)
            path,fn = os.path.split(matches)
   
        except FileNotFoundError:
            print("Could not find the provided matchfile.")
            return
    elif type(matches) == pd.core.frame.DataFrame:
        df = matches.copy()
        path,_ = os.path.split(df.ref.iloc[0])
        fn = "matches.csv"
    else:
        print("Matches must be either a string indicating the path to a matchfile or a pandas DataFrame.")
        return
     
    
    df["id_ref"] = df.ref.apply(get_scene_id)
    df["id_sec"] = df.sec.apply(get_scene_id)
    df["date_ref"] = df.id_ref.apply(get_date) 
    df["date_sec"] = df.id_sec.apply(get_date)
    df["dt"]  = df.date_sec - df.date_ref
    df["path"] =  df["ref"].apply(lambda x: os.path.split(x)[0])

        
    if what == "velocity": 
        df["filenames"] = df.path+"/disparity_maps/"+df.id_ref+"_"+df.id_sec+ prefix_ext+"-F_velocity.tif"

        array_list = [np.ma.masked_invalid(read_file(x,1)) for x in df.filenames if os.path.isfile(x)]
        
    elif what == "direction": 
        df["filenames"] = df.path+"/disparity_maps/"+df.id_ref+"_"+df.id_sec+ prefix_ext+"-F_velocity.tif"
        array_list = [np.deg2rad(read_file(x,2)) for x in df.filenames if os.path.isfile(x)]
        
    else: 
        df["filenames"] = df.path+"/disparity_maps/"+df.id_ref+"_"+df.id_sec+ prefix_ext+"-F.tif"

        if what == "dx":
        
            array_list = [read_file(x,1) for x in df.filenames if os.path.isfile(x)]
        
        elif what == "dy":
            array_list = [read_file(x,2) for x in df.filenames if os.path.isfile(x)]

        else: 
            print("Please provide a valid input for what should be stacked [dx/dy/velocity/direction].")
            return
        
        dt = [df["dt"][i].days for i in range(len(df)) if os.path.isfile(df.filenames[i])]
        resolution = [read_transform(df.filenames[i])[0] for i in range(len(df)) if os.path.isfile(df.filenames[i])]
        
        
        #if polyfitting has been applied, there is no more need for masking and disparity maps only have two bands
        bands = [read_meta(df.filenames[i])["count"] for i in range(len(df)) if os.path.isfile(df.filenames[i])]
        mask_list = [read_file(df.filenames[i],3) for i in range(len(df)) if (os.path.isfile(df.filenames[i]) and bands[i] == 3)]
        
        
        for i in range(len(dt)):
            #masking
            
            if bands[i] == 3:
                
                print("Interpreting the third band as good pixel mask.")
                masked = np.where(mask_list[i] == 1, array_list[i], np.nan)
        
            masked = array_list[i]

            #median shift 
            if medShift: 
                med = np.nanmedian(masked)
                masked = masked - med
            
            #dx in m/yr
            array_list[i] = np.ma.masked_invalid(((masked*resolution[i])/dt[i])*365)
            

    if what != "direction":
        average_vals = np.ma.average(array_list, axis=0)
        variance_vals = np.ma.std(array_list, axis = 0)
    else: #need to use circmean and circvar for angles
        average_vals = np.rad2deg(circmean(array_list, axis=0, nan_policy="omit"))
        variance_vals = np.rad2deg(circstd(array_list, axis=0, nan_policy="omit"))
        
    save_file([average_vals, variance_vals], df.filenames[0], os.path.join(path,fn[:-4] + f"_average_{what}{prefix_ext}.tif"))

    

    
