#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from helper_functions import get_scene_id, get_date, read_file, save_file, read_transform, get_extent, read_meta, min_max_scaler, get_epsg
import datetime, os, subprocess
import numpy as np
from osgeo import ogr
import matplotlib.pyplot as plt
from tqdm import tqdm
import rasterio
from scipy.stats import circmean, circstd
from skimage import exposure
import cv2 as cv

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

def offset_stats_pixel(r, xcoord, ycoord, pad = 0, resolution = None, dt = None, take_velocity = True):
    r[r==-9999] = np.nan
    if not take_velocity:
        if dt is None or resolution is None: 
            print("Need to provide a time difference and raster resolution when getting stats for dx/dy.")
            return
        #calculating displacement in m/yr to make things comparable
        r = ((r*resolution)/dt)*365

    sample = r[ycoord-pad:ycoord+pad+1, xcoord-pad:xcoord+pad+1]
    
    mean = np.nanmean(sample)
    median = np.nanmedian(sample)
    p75 = np.nanpercentile(sample, 75)
    p25 = np.nanpercentile(sample, 25)
    std = np.nanstd(sample)
    
    return mean, std, median, p25, p75

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


def get_std_iqr(matchfile, aoi = None, inverse = False, prefix_ext = "L3B"):
    
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
        fn = os.path.join(row.path, f"disparity_maps/{row.id_ref}_{row.id_sec}{prefix_ext}-F.tif")
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
    

def get_stats_in_aoi(matchfile, aoi = None, xcoord = None, ycoord = None, pad = 0, prefix_ext = "", max_dt = 10000, take_velocity = True):
    
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

    else: 
        print("Using disparity maps to generate timeline...")
        ext = ""
        colnames = ["dx_mean", "dx_std", "dx_median", "dx_p25", "dx_p75", "dy_mean", "dy_std", "dy_median", "dy_p25", "dy_p75"]
        stats = np.zeros([len(df), 10])
        

    stats[:] = np.nan

    for index, row in tqdm(df.iterrows(), total=df.shape[0]):

        disp = os.path.join(row.path, "disparity_maps", f"{row.id_ref}_{row.id_sec}{prefix_ext}-F{ext}.tif")


        if os.path.isfile(disp):

            if aoi is not None: #stats in geojson
                
                if not os.path.isfile("./temp.tif"):
                    #check aoi crs
                    driver = ogr.GetDriverByName("GeoJSON")
                    ds = driver.Open(aoi)
                    lyr = ds.GetLayerByIndex(0)
                    epsg_aoi = lyr.GetSpatialRef().GetAuthorityCode(None)
                    driver = ds = lyr = None
                    
                    #get epsg of disparity raster. assumes that all disp maps will have the same epsg
                    epsg_disp = get_epsg(disp)
                    
                    if int(epsg_disp) != int(epsg_aoi):
                        print("Reprojecting input GeoJSON to match EPSG of disparity maps...")
                        cmd = f"ogr2ogr -f 'GeoJSON' {aoi[:-8]}_EPSG{epsg_disp}.geojson {aoi} -s_srs EPSG:{epsg_aoi} -t_srs EPSG:{epsg_disp} "
                        subprocess.run(cmd, shell = True)
                        aoi = f"{aoi[:-8]}_EPSG{epsg_disp}.geojson"

                    #only calculating the mask once - all images should have the same extent
                    #rasterize aoi to find the pixels inside
                    extent = get_extent(disp)
                    resolution = read_transform(disp)[0]
                    cmd = f"gdal_rasterize -tr {resolution} {resolution} -burn 1 -a_nodata 0 -ot Int16 -of GTiff -te {' '.join(map(str,extent))} {aoi} ./temp.tif"
                    subprocess.run(cmd, shell = True)
    
                    mask = read_file("./temp.tif")
                
                if take_velocity:
                    stats[index,0], stats[index,1], stats[index,2], stats[index,3], stats[index,4]  = offset_stats_aoi(read_file(disp, 1), mask, resolution = resolution, dt = row["dt"].days, take_velocity = take_velocity)
                else:
                    stats[index,0], stats[index,1], stats[index,2], stats[index,3], stats[index,4]  = offset_stats_aoi(read_file(disp, 1), mask, resolution = resolution, dt = row["dt"].days, take_velocity = take_velocity)
                    stats[index,5], stats[index,6], stats[index,7], stats[index,8], stats[index,9]  = offset_stats_aoi(read_file(disp, 2), mask, resolution = resolution, dt = row["dt"].days, take_velocity = take_velocity)

            else: #stats in sample region
            
                if take_velocity:
                    stats[index,0], stats[index,1], stats[index,2], stats[index,3], stats[index,4]  = offset_stats_pixel(read_file(disp, 1), xcoord, ycoord, pad, resolution = resolution, dt = row["dt"].days, take_velocity = take_velocity)
                else:
                    stats[index,0], stats[index,1], stats[index,2], stats[index,3], stats[index,4]  = offset_stats_pixel(read_file(disp, 1), xcoord, ycoord, pad, resolution = resolution, dt = row["dt"].days, take_velocity = take_velocity)
                    stats[index,5], stats[index,6], stats[index,7], stats[index,8], stats[index,9]  = offset_stats_pixel(read_file(disp, 2), xcoord, ycoord, pad, resolution = resolution, dt = row["dt"].days, take_velocity = take_velocity)

        else:
          print(f"Warning! Disparity file {disp} not found.")
        
    statsdf = pd.DataFrame(stats, columns = colnames)
    df = pd.concat([df, statsdf], axis = 1)
    
    df.to_csv(f"{path}/stats_in_aoi_{matchfn[:-4]}{ext}.csv", index = False)
    return df



def stack_rasters(matches, prefix_ext = "", what = "velocity", medShift = False):
    """
    Stack velocity or disparity rasters from all correlation pairs.
    
    Parameters:
    matches (str or pandas.DataFrame): Path to the matchfile or a pandas DataFrame with match information.
    prefix_ext (str, optional): Prefix extension for the output files (default: "").
    what (str, optional): Type of rasters to stack [dx/dy/velocity/direction] (default: "velocity").
    med_shift (bool, optional): Apply median shift to displacements (default: False).
    
    Returns:
    path to averaged raster
    """
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
        std_vals = np.ma.std(array_list, axis = 0)
    else: #need to use circmean and circvar for angles
        average_vals = np.rad2deg(circmean(array_list, axis=0, nan_policy="omit"))
        std_vals = np.rad2deg(circstd(array_list, axis=0, nan_policy="omit"))
        
    save_file([average_vals, std_vals], df.filenames[0], os.path.join(path,fn[:-4] + f"_average_{what}{prefix_ext}.tif"))
    return os.path.join(path,fn[:-4] + f"_average_{what}{prefix_ext}.tif")


def shape_even(array):
    #ensure image has even dimension, otherwise ffmpeg will complain

    if array.shape[0]%2 != 0:
        array = array[:-1,:]
    if array.shape[1]%2 != 0:
        array = array[:,:-1]
    return array

def adjust_to_uint16(array):
    #ffmpeg wants uint16 images, so stretch gray values between 0 and 2**16
    
    img = array.astype(np.float32)
    img[img == 0] = np.nan
    
    img = min_max_scaler(img)
    img = img * (2**16-1)
    img[np.isnan(img)] = 0

    return img.astype(np.uint16)

    
def make_video(matches, video_name = "out.mp4", ext = "_remap", crop = 0):
    """
    Create a video and GIF from a sequence of PlanetScope scenes.
   
    Parameters:
    matches (str or DataFrame): Path to the matchfile or pandas DataFrame.
    video_name (str, optional): Name of the output video file. Defaults to "out.mp4".
    ext (str, optional): Extension of the secondary images. Defaults to "_remap".
    crop (int, optional): Number of pixels to crop from image edges. Defaults to 0.
    
    Returns:
    str: Path to the generated GIF file.
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

    matches.sec = matches.sec.str.replace(".tif", ext+".tif", regex=True)
    
    all_files = [matches.ref.unique(), list(matches.sec)]
    all_files = [item for sublist in all_files for item in sublist]
    all_files = sorted(all_files)
    
           
    #match histograms to have similar brightness    
    final_files = []    
    ref_img = cv.imread(all_files[0], cv.IMREAD_UNCHANGED)
    
    if crop > 0:
        ref_img = ref_img[crop:-crop,crop:-crop]
    
    ref_img = adjust_to_uint16(shape_even(ref_img))
    font = cv.FONT_HERSHEY_DUPLEX
    date = get_date(get_scene_id(all_files[0])).strftime('%Y-%m-%d')
    ref_img = cv.putText(ref_img,date,(int(ref_img.shape[0]*0.05), int(ref_img.shape[0]*0.05)),font,2,(2**16,2**16,2**16),3)  
    
    cv.imwrite(os.path.join(path, all_files[0][:-4]+"_forGIF.tif"), ref_img)
    final_files.append(os.path.join(path, all_files[0][:-4]+"_forGIF.tif"))
    
    for f in all_files[1:]:
        img = cv.imread(f, cv.IMREAD_UNCHANGED)
        img = adjust_to_uint16(shape_even(img))
        if crop > 0: 
            img = img[crop:-crop,crop:-crop]
        matched = exposure.match_histograms(img, ref_img)
        matched = matched.astype(np.uint16)
        date = get_date(get_scene_id(f)).strftime('%Y-%m-%d')
        matched = cv.putText(matched,date,(int(ref_img.shape[0]*0.05), int(ref_img.shape[0]*0.05)),font,2,(2**16,2**16,2**16),3)  
        cv.imwrite(os.path.join(path, f[:-4]+"_forGIF.tif"), matched)
        final_files.append(os.path.join(path, f[:-4]+"_forGIF.tif"))
        
    with open(os.path.join(path, 'file_list.txt'), 'w') as fl:
        for line in final_files:
            fl.write(f"file {line}\n")

    #cannot use option -framerate with concat, see https://superuser.com/questions/1671523/ffmpeg-concat-input-txt-set-frame-rate
    cmd = f"ffmpeg -f concat -safe 0 -i {os.path.join(path, 'file_list.txt')} -y -vf 'settb=AVTB,setpts=N/2/TB,fps=2' -c:v libx264 -pix_fmt yuv420p {os.path.join(path, video_name)}"
    subprocess.run(cmd, shell = True)

    cmd = f"ffmpeg -y -i {os.path.join(path, video_name)} -vf 'fps=5,scale=1080:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse' -loop 0 {os.path.join(path, video_name[:-4]+'.gif')}"  
    subprocess.run(cmd, shell = True)
    
    return os.path.join(path, video_name[:-4]+'.gif')

    
