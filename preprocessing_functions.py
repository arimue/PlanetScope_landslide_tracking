#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import subprocess 
import shutil
import glob
import pandas as pd
import helper_functions as helper
import numpy as np
from shapely.geometry import Polygon
from xml.dom import minidom
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
    out_img = f"{band_dir}/{img_fn[:-4]}_b{band_nr}.tif"
    cmd = f"gdal_translate -co COMPRESS=DEFLATE -co ZLEVEL=9 -co PREDICTOR=2 -b {band_nr} {img} {out_img}"
    subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return out_img

def get_single_band(files, out_path = "./", band_nr = 2): #TODO: fetch work_dir globally
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
        _,fn = os.path.split(out_image)
        shutil.copyfile(out_image, os.path.join(out_path,fn))
        out.append(os.path.join(out_path,fn))
    print("Isolated bands can now be found in " + out_path)
    
    return out


def rate_match(infodf, matchdf):
    #rate existing matches, do not suggest
    #sort by reference if not already to be able to calculate scores in batches
    matchdf = matchdf.sort_values("ref").reset_index(drop = True)
    scores = []
    for ref in matchdf.ref.unique():
        
        #print(ref)
        refid = helper.get_scene_id(ref)
        secids = [helper.get_scene_id(sec) for sec in matchdf.sec.loc[matchdf.ref == ref]]
        refinfo = infodf.loc[infodf.ids == refid]
        secinfo = infodf.loc[infodf.ids.isin(secids)]
        sum_va_scaled = helper.fixed_val_scaler(refinfo.view_angle.iloc[0]+secinfo.view_angle, 0, 10)
        diff_va_scaled = helper.fixed_val_scaler(abs(refinfo.view_angle.iloc[0]-secinfo.view_angle), 0, 5)
        sat_az_diff = abs(refinfo.sat_az.iloc[0]-secinfo.sat_az)
        sat_az_diff[sat_az_diff>180] = abs(360-sat_az_diff[sat_az_diff>180])
        sat_az_diff_scaled = helper.fixed_val_scaler(sat_az_diff, 0,180)
        score =  1.5 -(sum_va_scaled * sat_az_diff_scaled + diff_va_scaled )
        
        refPoly = Polygon([tuple(coords) for coords in refinfo.footprint.iloc[0]])
        overlap = []
        
        for idx, row in secinfo.iterrows():
            secPoly = Polygon([tuple(coords) for coords in row.footprint])
            intersection = refPoly.intersection(secPoly)
            overlap.append(intersection.area/refPoly.area*100)
            
        #add xml info
        
        # angs = []
        # for i, row in df.iterrows():
        #     xml_file = glob.glob(f"/home/ariane/Documents/PlanetScope/test_ang_calc/{row.ids}*metadata.xml")
            
        #     if len(xml_file) == 1:
        #         xmldoc = minidom.parse(xml_file[0])
        #         north_ang = float(xmldoc.getElementsByTagName("ps:azimuthAngle")[0].firstChild.data)
        #         angs.append(north_ang)
                
        #     else:
        #         angs.append(np.nan)
                
        # df["az_ang"] = angs
        # df["az_diff"] = df.sat_az-df.az_ang
        # df.az_diff[df.az_diff >180] = df.az_diff -180
        #TODO: remove this
        xml_file = glob.glob(f"/home/ariane/Documents/PlanetScope/test_ang_calc/{refid}*metadata.xml")
        
        if len(xml_file)  == 1:
            
            xmldoc = minidom.parse(xml_file[0])
            ang_to_north_ref = float(xmldoc.getElementsByTagName("ps:azimuthAngle")[0].firstChild.data)
            
            print(ang_to_north_ref)
            limangle_ref = 180+ang_to_north_ref
        else:
            ang_diff_ref = np.nan
            
        ang_diff_secs = []
        limangle_secs = []
        for secid in secids:

            xml_file = glob.glob(f"/home/ariane/Documents/PlanetScope/test_ang_calc/{secid}*metadata.xml")
            if len(xml_file) == 1:
                
                xmldoc = minidom.parse(xml_file[0])
                inc_ang = float(xmldoc.getElementsByTagName("eop:incidenceAngle")[0].firstChild.data)
                view_ang = float(xmldoc.getElementsByTagName("ps:spaceCraftViewAngle")[0].firstChild.data)
                ang_to_north_sec = float(xmldoc.getElementsByTagName("ps:azimuthAngle")[0].firstChild.data)

                ang_diff_secs.append(abs(inc_ang-view_ang))
                
                print(ang_to_north_sec)
                limangle_secs.append(180+ang_to_north_sec)
            else:
                ang_diff_secs.append(np.nan)
                
        if refinfo.sat_az.iloc[0] > limangle_ref:
            ref_va = -1*refinfo.view_angle.iloc[0]
        else:
            ref_va = refinfo.view_angle.iloc[0]
            
        secinfo["limang"] = limangle_secs
        secinfo["true_va"] = secinfo.view_angle
        secinfo.true_va[secinfo.sat_az > secinfo.limang] = secinfo.true_va*-1
        
        #TODO:also remove true_va stuff here
        scores.append({
            "refid": refid,
            "secid": secinfo.ids, 
            "score": score, 
            "overlap": overlap,
            "va_sum":refinfo.view_angle.iloc[0]+secinfo.view_angle,
            "va_diff": abs(refinfo.view_angle.iloc[0]-secinfo.view_angle),
            "az_diff": sat_az_diff,
            "true_va_diff":abs(ref_va-secinfo.true_va)})
            #"ang_diff_diff": [a-ang_diff_ref for a in ang_diff_secs] })
        
    scores = pd.DataFrame.from_records(scores).explode(["secid","score", "overlap", "va_sum", "va_diff", "az_diff", "true_va_diff"]).reset_index(drop = True)
    return(scores)

def generate_matchfile_from_search(df, path = "./",  ext = "_b2.tif", check_existence = False):
    
    """
    Matches PlanetScope scenes from the provided dataframe.

    Args:
        df (pandas.DataFrame): DataFrame containing scene information.
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
        
    if check_existence:
        matches.ref = matches.ref.apply(lambda row: os.path.join(path, row+file_ext[0]))
        matches.sec = matches.sec.apply(lambda row: os.path.join(path, row+file_ext[0]))
        matches.to_csv(os.path.join(path, "matches_from_search.csv"), index = False)
        print(f"All matches were stored under {os.path.join(path, 'matches_from_search.csv')}.")

    else: 
        print("Only returning IDs of potential matches since you have not pointed by to any directory storing PlanetScope data." )
        

    print(f"I have found a total of {len(matches)} correlation pairs.")

    return matches

def generate_matchfile_from_groups(groups, path = "./",  ext = "_b2.tif", check_existence = False):
    
    """
    Matches PlanetScope scenes in groups with a common perspective based on the provided dataframe.

    Args:
        groups (pandas.DataFrame): DataFrame containing grouped scenes.
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
            
    matches = pd.DataFrame.from_records(matches).explode("sec", "group")
        
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


def orthorectify_L1B(files, demname, aoi, epsg, amespath, pad = 0):
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
    ul_lon, ul_lat, xsize, ysize = helper.size_from_aoi(aoi, epsg = epsg, gsd = 4)

    for f in files: 
        #its best to roughly clip before mapprojection, otherwise the process takes long
        clip = helper.clip_raw(f, ul_lon, ul_lat, xsize+pad, ysize+pad, demname)
        mp = asp.mapproject(amespath, clip, demname, epsg = epsg)
        #finetune clip, because clipping with RPCs is not always super exact
        _ = helper.clip_mp_cutline(mp, aoi)
    