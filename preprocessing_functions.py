#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import subprocess 
import shutil
import glob
import pandas as pd
import helper_functions as helper
from shapely.geometry import Polygon
import asp_helper_functions as asp

def isolate_band(img, band_nr=2):
    """
    Isolate a specific band from an image and save it as a separate TIFF file.

    Args:
        img (str): Path to the input image.
        band_nr (int): Band number to isolate (default: 2, i.e. green band).

    Returns:
        str: Path to the output isolated band image.

    """
    out_dir, img_fn = os.path.split(img)
    band_dir = f"{out_dir}/b{band_nr}/"
    if not os.path.exists(band_dir):
        print(f"Generating directory {band_dir}")
        os.makedirs(band_dir)
    out_img = f"{band_dir}{img_fn[:-4]}_b{band_nr}.tif"
    cmd = f"gdal_translate -co COMPRESS=DEFLATE -co ZLEVEL=9 -co PREDICTOR=2 -b {band_nr} {img} {out_img}"
    subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return out_img

def get_single_band(files, out_path = "./", band_nr = 2, apply_udm_mask = False): #TODO: fetch work_dir globally
    """
    Preprocess a list of input files by isolating a specific band and copying it to the output directory.
    
    Args:
        files (list): List of input file paths.
        out_path (str): Output directory path (default: "./").
        band_nr (int): Band number to isolate (default: 2).
    
    Returns:
        list: List of output file paths.
    
    """
    out = []
    for file in files:
        out_image = isolate_band(file, band_nr)
        fn = os.path.basename(out_image)
        if apply_udm_mask:
            p = os.path.dirname(file)
            sid = helper.get_scene_id(fn)
            udm_fn = glob.glob(f"{p}/{sid}*_udm2*.tif")
            if len(udm_fn) == 0:
                print(f"Could not find matching udm mask for scene {sid}.")
            elif len(udm_fn) > 1:
                print(f"Found too many potential udm files: {udm_fn}")
            else:
                r = helper.read_file(out_image)
                udm = helper.read_file(udm_fn[0]) #first band stores clean pixels
                r[udm == 0] = -9999
                helper.save_file([r], out_image, os.path.join(out_path,fn))
        else:
            shutil.copyfile(out_image, os.path.join(out_path,fn))
                
        out.append(os.path.join(out_path,fn))
    print("Isolated bands can now be found in " + out_path)
    
    return out


def rate_match(infodf, matchdf):
    """
    Rates existing matches.
 
    Args:
        infodf (pandas.DataFrame): DataFrame containing scene information.
        matchdf (pandas.DataFrame): DataFrame containing matched scenes.
 
    Returns:
        pandas.DataFrame: DataFrame containing true view angle difference and overlap between scenes.
    """
      
    #sort by reference if not already to be able to calculate ratings in batches
    matchdf = matchdf.sort_values("ref").reset_index(drop = True)
    ratings = []
    for ref in matchdf.ref.unique():
        
        #print(ref)
        refid = helper.get_scene_id(ref)
        secids = [helper.get_scene_id(sec) for sec in matchdf.sec.loc[matchdf.ref == ref]]
        refinfo = infodf.loc[infodf.ids == refid]
        secinfo = infodf.loc[infodf.ids.isin(secids)]
        
        refPoly = Polygon([tuple(coords) for coords in refinfo.footprint.iloc[0]])
        overlap = []
        
        for idx, row in secinfo.iterrows():
            secPoly = Polygon([tuple(coords) for coords in row.footprint])
            intersection = refPoly.intersection(secPoly)
            overlap.append(intersection.area/refPoly.area*100)
            
             
        ratings.append({
            "refid": refid,
            "secid": secinfo.ids, 
            "overlap": overlap,
            "true_va_diff":abs(secinfo.true_view_angle- refinfo.true_view_angle.iloc[0])})

        
    ratings = pd.DataFrame.from_records(ratings).explode(["secid","overlap","true_va_diff"]).reset_index(drop = True)
    return(ratings)

def generate_matchfile_from_search(df, dt_min = None, path = "./",  ext = "_b2.tif", check_existence = False):
    
    """
    Matches PlanetScope scenes from the provided dataframe.

    Args:
        df (pandas.DataFrame): DataFrame containing scene information.
        dt_min (int): Minimum number of days between reference and secondary scenes (default: None).
        path (str): Path to the scenes.
        ext (str): File extension (default: "_b2.tif").
        check_existence (bool): Check if the scenes exist in the given path (default: False).

    Returns:
        pandas.DataFrame: DataFrame containing matched scenes.

    """

    if check_existence:
        files = glob.glob(f"{path}/*{ext}")
        if len(files) == 0:
            print(f"No files found in {path} that match the pattern {ext}.")
            return
        ids = [helper.get_scene_id(f) for f in files]
        exists = [i in ids for i in df.ids]
        df = df.loc[exists].reset_index(drop = True)
        file_ext = [files[i].split("/")[-1].replace(ids[i], "") for i in range(len(files))]
        if len(set(file_ext)) != 1:
            print(f"Found variable file extents: {list(set(file_ext))}, but I need these to be equal. Are you working with data from different sensors?")
            return
    matches = []
    df = df.sort_values("ids").reset_index(drop = True)
    for i in range(len(df)-1):
        matches.append({
            "ref": df.ids.iloc[i],
            "sec": list(df.ids.iloc[i+1:])})
        
    matches = pd.DataFrame.from_records(matches).explode("sec")
    
    if dt_min is not None:
        date_ref = [helper.get_date(i) for i in matches.ref]
        date_sec = [helper.get_date(i) for i in matches.sec]
        good_dt = [i for i in range(len(date_ref)) if (date_sec[i]-date_ref[i]).days > dt_min]
        matches = matches.iloc[good_dt].reset_index(drop = True)
        
        if len(matches) == 0:
            print("It looks like there are no more suitable matches if I apply your minimal temporal baseline. Try to lower it.")
            return
        
    if check_existence:
        matches.ref = matches.ref.apply(lambda row: os.path.join(path, row+file_ext[0]))
        matches.sec = matches.sec.apply(lambda row: os.path.join(path, row+file_ext[0]))
        matches.to_csv(os.path.join(path, "matches_from_search.csv"), index = False)
        print(f"All matches were stored under {os.path.join(path, 'matches_from_search.csv')}.")

    else: 
        print("Only returning IDs of potential matches since you have not pointed by to any directory storing PlanetScope data." )
        

    print(f"I have found a total of {len(matches)} correlation pairs.")

    return matches

def generate_matchfile_from_groups(groups, dt_min = None, path = "./",  ext = "_b2.tif", check_existence = False):
    
    """
    Matches PlanetScope scenes in groups with a common perspective based on the provided dataframe.

    Args:
        groups (pandas.DataFrame): DataFrame containing grouped scenes.
        dt_min (int): Minimum number of days between reference and secondary scenes (default: None).
        path (str): Path to the scenes (default: "./").
        ext (str): File extension (default: "_b2.tif").
        check_existence (bool): Whether to check if the scenes exist in the given path (default: False).

    Returns:
        pandas.DataFrame: DataFrame containing matched scenes.

    """

    if check_existence:
        files = glob.glob(f"{path}/*{ext}")
        if len(files) == 0:
            print(f"No files found in {path} that match the pattern {ext}.")
            return
        ids = [helper.get_scene_id(f) for f in files]
        exists = [i in ids for i in groups.ids]
        groups = groups.loc[exists].reset_index(drop = True)
        file_ext = [files[i].split("/")[-1].replace(ids[i], "") for i in range(len(files))]
        if len(set(file_ext)) != 1:
            print(f"Found variable file extents: {set(file_ext)}, but I need these to be equal. Are you working with data from different sensors?")
            return
    matches = []
    for group in groups.group_id.unique():
        gdf = groups.loc[groups.group_id == group].sort_values("ids").reset_index(drop = True)
        for i in range(len(gdf)-1):
            matches.append({
                "ref": gdf.ids.iloc[i],
                "sec": list(gdf.ids.iloc[i+1:]), 
                "group": group})
            
    matches = pd.DataFrame.from_records(matches).explode(["sec"])
    
    if dt_min is not None:
        date_ref = [helper.get_date(i) for i in matches.ref]
        date_sec = [helper.get_date(i) for i in matches.sec]
        good_dt = [i for i in range(len(date_ref)) if (date_sec[i]-date_ref[i]).days > dt_min]
        matches = matches.iloc[good_dt].reset_index(drop = True)
        
        if len(matches) == 0:
            print("It looks like there are no more suitable matches if I apply your minimal temporal baseline. Try to lower it.")
            return
        
    matches = matches.drop_duplicates(subset=['ref', 'sec'])

    if check_existence:
        matches.ref = matches.ref.apply(lambda row: os.path.join(path, row+file_ext[0]))
        matches.sec = matches.sec.apply(lambda row: os.path.join(path, row+file_ext[0]))
        matches.to_csv(os.path.join(path, "matches_by_group.csv"), index = False)
        print(f" All matches were stored under {os.path.join(path, 'matches_by_group.csv')}.")

    else: 
        print("Only returning IDs of potential matches since you have not pointed by to any directory storing PlanetScope data." )
        
    print(f"I have found a total of {len(matches)} correlation pairs.")
    return matches


def match_all(path, ext = "_b2.tif", dt_min = None):
    """
    Matches all PlanetScope scenes stored in the provided directory that match the given pattern (file extension).

    Args:
        df (pandas.DataFrame): DataFrame containing scene information.
        path (str): Path to the scenes.
        ext (str): File extension (default: "_b2.tif").
        dt_min (int): Minimum number of days between reference and secondary scenes (default: None).

    Returns:
        pandas.DataFrame: DataFrame containing matched scenes.
    """
    files = glob.glob(f"{path}/*{ext}")
    
    if len(files) < 1: 
        print("I could not find any matching scenes. Check if the provided path and file extent is correct.")
        return
    elif len(files) <2:
        print("Only one matching scene found. I need at least two for matching.")
        return
    
    ids = [helper.get_scene_id(f) for f in files]
    file_ext = [files[i].split("/")[-1].replace(ids[i], "") for i in range(len(files))]
    if len(set(file_ext)) != 1:
        print(f"Found variable file extents: {set(file_ext)}, but I need these to be equal. Are you working with data from different sensors?")
        return
    ids = sorted(ids)
    matches = []
        
    for i in range(len(files)-1):
        matches.append({
            "ref": ids[i],
            "sec": list(ids[i+1:])})
                
            
    matches = pd.DataFrame.from_records(matches).explode("sec").reset_index(drop = True)
    matches.ref = matches.ref.apply(lambda row: os.path.join(path, row+file_ext[0]))
    matches.sec = matches.sec.apply(lambda row: os.path.join(path, row+file_ext[0]))
    
    
    if dt_min is not None:
        date_ref = [helper.get_date(helper.get_scene_id(i)) for i in matches.ref]
        date_sec = [helper.get_date(helper.get_scene_id(i)) for i in matches.sec]
        good_dt = [i for i in range(len(date_ref)) if (date_sec[i]-date_ref[i]).days > dt_min]
        
        matches = matches.iloc[good_dt].reset_index(drop = True)
        
        if len(matches) == 0:
            print("It looks like there are no more suitable matches if I apply your minimal temporal baseline. Try to lower it.")
            return
    matches.to_csv(os.path.join(path, "all_matches.csv"), index = False)
    print(f"I have found {len(matches)} correlation pairs. All matches were stored under {os.path.join(path, 'all_matches.csv')}.")
    
    return matches

def match_to_one_ref(path, ext = "_b2.tif"):
 
    files = glob.glob(f"{path}/*{ext}")
    
    if len(files) < 1: 
        print("I could not find any matching scenes. Check if the provided path and file extent is correct.")
        return
    elif len(files) <2:
        print("Only one matching scene found. I need at least two for matching.")
        return
    
    ids = [helper.get_scene_id(f) for f in files]
    file_ext = [files[i].split("/")[-1].replace(ids[i], "") for i in range(len(files))]
    if len(set(file_ext)) != 1:
        print(f"Found variable file extents: {set(file_ext)}, but I need these to be equal. Are you working with data from different sensors?")
        return
    ids = sorted(ids)
    matches = []

    matches.append({
        "ref": ids[0],
        "sec": list(ids[1:])})
                
            
    matches = pd.DataFrame.from_records(matches).explode("sec").reset_index(drop = True)
    matches.ref = matches.ref.apply(lambda row: os.path.join(path, row+file_ext[0]))
    matches.sec = matches.sec.apply(lambda row: os.path.join(path, row+file_ext[0]))
    

    matches.to_csv(os.path.join(path, "matches_one_ref.csv"), index = False)
    
    return matches


def match_common_perspectives_and_illumination(df, va_diff_thresh = 0.3, sun_az_thresh = 5, sun_elev_thresh = 5, min_group_size = 5, dt_min = 1):
    """
    Group scenes that fullfill search criteria into groups with a common satellite perspective and illumination conditions.
    Note that not all scenes within a group can be matched, as the illumination diff is always with regards to the first scene.
    """
    
    out = pd.DataFrame()
    if min_group_size < 2:
        print("Resetting min_group_size to 2. Smaller groups are not allowed.")
    for idx, row in df.iterrows():
        mask = (
            (df["true_view_angle"].between(row["true_view_angle"] - va_diff_thresh, row["true_view_angle"] + va_diff_thresh)) &
            (df["sun_azimuth"].between(row["sun_azimuth"] - sun_az_thresh, row["sun_azimuth"] + sun_az_thresh)) &
            (df["sun_elevation"].between(row["sun_elevation"] - sun_elev_thresh, row["sun_elevation"] + sun_elev_thresh))
        )
    
        selected = df.loc[mask]
        

        if len(selected) >= (min_group_size): 
            other_ids = [i for i in selected.ids if i not in row.ids]
            match = pd.DataFrame({"ref": [row.ids]*len(other_ids), "sec": other_ids})
            match["date0"] = match.ref.apply(helper.get_date)
            match["date1"] = match.sec.apply(helper.get_date)
            match["dt"] = (match.date1 - match.date0).dt.days
            match = match.loc[match["dt"] >= dt_min]
            out = pd.concat([match.loc[:, ["ref", "sec"]], out])
    out = out.reset_index(drop = True)
    return out

def orthorectify_L1B(amespath, files, demname, aoi, epsg, pad = 100):
    """
    Orthorectifies provided L1B data based on the given DEM.

    Args:
        files (list): List of input files.
        demname (str): Name of the DEM file.
        aoi (str): name and path tp AOI geometry (GeoJSON).
        epsg (int): EPSG code for the coordinate system.
        amespath (str): Ames Stereo Pipeline path.
        pad (int): Padding value for clipping raw data based on RPCs (default: 0).

    """
    #TODO: implement cleanup
    ul_lon, ul_lat, xsize, ysize = helper.size_from_aoi(aoi, epsg = epsg, gsd = 4)

    for f in files: 
        #TODO: size estimation works not 100% well yet
        #its best to roughly clip before mapprojection, otherwise the process takes long
        clip = helper.clip_raw(f, ul_lon, ul_lat, xsize+pad, ysize+pad, demname)
        mp = asp.mapproject(amespath, clip, demname, epsg = epsg)
        #finetune clip, because clipping with RPCs is not always super exact
        _ = helper.clip_mp_cutline(mp, aoi)
    