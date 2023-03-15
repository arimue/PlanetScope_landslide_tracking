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
    
def mapproject_and_calc_velocity(amespath, matchfile, dem, img_with_rpc, resolution = 3, epsg = "32720", ext = "mp", overwrite = False, velocity_only = False):
    df = pd.read_csv(matchfile)
    #df = df.reindex(index=df.index[::-1])
    #df = df.iloc[40:, :]
    path,_ = os.path.split(matchfile)

    df["id_ref"] = df.ref.apply(get_scene_id)
    df["id_sec"] = df.sec.apply(get_scene_id)
    df["date_ref"] = df.id_ref.apply(get_date)
    df["date_sec"] = df.id_sec.apply(get_date)
    
    df["dt"]  = df.date_sec - df.date_ref

    #extract statistics from disparity files
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        disp = f"{path}/stereo/{row.id_ref}_{row.id_sec}_remap-F.tif"
        #disp = f"{path}/stereo/{row.id_ref}_{row.id_sec}_clip_mp-F.tif"

        #copy_rpcs(img_with_rpc, disp)
        if os.path.isfile(disp):
            if not velocity_only:
                #mapproject
                if not os.path.isfile(f"{disp[:-4]}_{ext}.tif") or overwrite:
                    #if the mapprojected result doesnt show in QGIS, make sure to remove Band 4 as the alpha band from the transparency settings
                    output = asp.mapproject(amespath, disp, dem, img_with_rpc, ext = ext, resolution = resolution, epsg = epsg)       
                else:
                    print("Mapprojected disparity exists. Skipping mapprojection...")
                    output = f"{disp[:-4]}_{ext}.tif"
            else: 
                output = disp
            if not os.path.isfile(disp[:-4]+"_velocity.tif") or overwrite: 
                v, direction = calc_velocity(output, row["dt"])
                save_file([v,direction], output, outname = disp[:-4]+"_velocity.tif")
            else:
                print("Velocity file exists. Skipping velocity calculation...")
        else:
            print(f"Warning! Disparity file {disp} not found.")


def offset_stats_pixel(r, xcoord, ycoord, pad = 0, resolution = None, dt = None, take_velocity = True, angles = False):
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
    if not take_velocity:
        if dt is None or resolution is None: 
            print("Need to provide a time difference and raster resolution when getting stats for dx/dy.")
            return
        #calculating displacement in m/yr to make things comparable
        r = ((r*resolution)/dt)*365
    
    try:    
        sample = r[mask == 1]
    except IndexError:
        print(f"There seems to be a problem with your input scene. Likely the dimensions do not fit the rest of the data. Have you altered your x/ysize or coordinates of the upper left corner when correlating scenes?")
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

    stats = np.zeros((len(df), 9))
    stats[:] = np.nan
    timeline_stats[:] = np.nan

    for index, row in tqdm(df.iterrows(), total=df.shape[0]):

        disp = f"{path}/stereo/{row.id_ref}_{row.id_sec}_remap-F{ext}.tif"

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
                stats[index,0], stats[index,1], stats[index,2], stats[index,3]  = offset_stats_aoi(read_file(disp, 1), mask, resolution = resolution, dt = row["dt"].days, take_velocity = take_velocity, angles = False)
                #same for dy or direction
                stats[index,4], stats[index,5], stats[index,6], stats[index,7]  = offset_stats_aoi(read_file(disp, 2), mask, resolution = resolution, dt = row["dt"].days, take_velocity = take_velocity, angles = angles)

            else:
                #get mean in sample region and iqr/p75 (weight) for dx or velocity
                stats[index,0], stats[index,1], stats[index,2], stats[index,3]  = offset_stats_pixel(read_file(disp, 1), xcoord, ycoord, pad, resolution = resolution, dt = row["dt"].days, take_velocity = take_velocity, angles = False)
                #same for dy or direction
                stats[index,4], stats[index,5], stats[index,6], stats[index,7]  = offset_stats_pixel(read_file(disp, 2), xcoord, ycoord, pad, resolution = resolution, dt = row["dt"].days, take_velocity = take_velocity, angles = angles)
                
            stats[index, 8]  = row["dt"].days
            
        else:
          print(f"Warning! Disparity file {disp} not found.")
    
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
                    
                timeline_stats[i,:] = np.average(stats[active,:6], axis=0, weights= weights)
    
            else: #dx/dy separate case
            
                if weigh_by_dt:
                    dt_weights = min_max_scaler(stats[active, -1])
                    disp_weights1 = min_max_scaler(1/(stats[active, 2]-stats[active, 1]))
                    disp_weights2 = min_max_scaler(1/(stats[active, 5]-stats[active, 4]))
                    weights1 = dt_weights + disp_weights1
                    weights2 = dt_weights + disp_weights2

                else:
                    weights1 = 1/(stats[active, 2]-stats[active, 1])
                    weights2 = 1/(stats[active, 5]-stats[active, 4])
                    
                                               
                #separate weights for dx and dy
                timeline_stats[i,:4] = np.average(stats[active, :4], weights = weights1, axis=0)
                timeline_stats[i,4:6] = np.average(stats[active, 4:6], weights = weights2, axis=0)
            
        timeline_alldata = np.concatenate((timeline_alldata,stats[active, :]), axis = 0)
        #print(len(stats[active, :]))
        count[i] = len(stats[active,:])
        
    
    out = pd.DataFrame(timeline_stats, columns = colnames)
    out["date"] = timeline
    out["count"] = count
    
    
    colnames.append("dt")
    timeline_alldata = pd.DataFrame.from_records(timeline_alldata, columns = colnames)
    timeline_alldata["date"] = timeline.repeat(count.astype(int)).reset_index(drop = True)
    
        
    if aoi is not None:
        timeline_alldata.to_csv(f"{path}/timeline_alldata_aoi{ext}_maxDT.csv", index = False)
        out.to_csv(f"{path}/timeline_averaged_aoi{ext}_maxDT.csv", index = False)
    else:
        
        timeline_alldata.to_csv(f"{path}/timeline_alldata_x{xcoord}_y{ycoord}_pad{pad}{ext}_maxDT.csv", index = False)
        out.to_csv(f"{path}/timeline_averaged_x{xcoord}_y{ycoord}_pad{pad}{ext}_maxDT.csv", index = False)

    
        
        # img = read_file(df.ref[0])
        # rect = patches.Rectangle((xcoord-pad, ycoord-pad), pad*2+1, pad*2+1, linewidth=1, edgecolor='r', facecolor='none')
        # fig, ax = plt.subplots()
        # ax.imshow(img, cmap = "Greys")
        # ax.add_patch(rect)
        # #plt.savefig("show_area.png", dpi = 400)
        # plt.show()
        


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
        
        ext = "velocity"
        df["filenames"]  = path+"/stereo/"+df.id_ref+"_"+df.id_sec+"_remap-F_"+ext+".tif"
        array_list = [np.ma.masked_invalid(read_file(x,1)) for x in df.filenames if os.path.isfile(x)]
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


def calculate_average_direction(average_fn):
    path,_ = os.path.split(average_fn)
    dx = read_file(average_fn, 1)
    dy = read_file(average_fn, 2)
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
    save_file([direction], average_fn, os.path.join(path,"average_direction.tif"))

def find_clusters(path):
    #TODO: improve this!
    path = "./Dove-C_Jujuy_all/L1B/"
    vel = read_file(os.path.join(path, "average_velocity.tif"))
    direct = read_file(os.path.join(path, "average_direction.tif"))
    dx = read_file(os.path.join(path, "average_dx_dy.tif"), 1)
    dy = read_file(os.path.join(path, "average_dx_dy.tif"), 2)

    direct[np.isnan(direct)] = -9999
    #x, y = np.meshgrid(np.arange(vel.shape[1]), np.arange(vel.shape[0]))
    
     #Stack the image data into a 2D array
    data = np.column_stack((vel.flatten(), direct.flatten(), dx.flatten(), dy.flatten()))    
    # Define the number of clusters and the spatial weight
    K = 15
    
    kmeans = KMeans(n_clusters=K, random_state=0)
    #weight = np.array([2,2,2,1,0.5, 0.5])
    # introduce weights by multiplying data matrix by weight matrix
    #weighted_data = data * weight[np.newaxis, :]
    
    # fit the weighted data to the k-means algorithm
    clusters = kmeans.fit(data).labels_
    
    # reshape the cluster labels to match the original image shape
    cluster_img = clusters.reshape(vel.shape)
    ls = cluster_img
    ls[ls!=1] = 0
    # visualize the clustered image
    plt.imshow(ls)
    plt.show()

