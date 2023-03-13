#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 15:55:06 2023

@author: ariane
"""

import os, subprocess, shutil, glob, datetime
import pandas as pd
import planet_search_functions as search
from helper_functions import get_date, get_scene_id


def isolateBand(img, bandNr=2):
    out_dir, img_fn = os.path.split(img)
    if not os.path.exists(f"{out_dir}/b{bandNr}/"):
        print(f"Generating directory {out_dir}/b{bandNr}/")
        os.makedirs(f"{out_dir}/b{bandNr}/")
    out_img = f"{out_dir}/b{bandNr}/{img_fn[:-4]}_b{bandNr}.tif"
    cmd = f"gdal_translate -co COMPRESS=DEFLATE -co ZLEVEL=9 -co PREDICTOR=2 -b {bandNr} {img} {out_img}"
    subprocess.run(cmd, shell = True,  stdout=subprocess.PIPE, stderr=subprocess.PIPE, text = True)
    return out_img

def preprocess_scenes(files, outpath = "./", bandNr = 2):
    out = []
    for file in files:
        out_image = isolateBand(file, bandNr)
        _,fn = os.path.split(out_image)
        shutil.copyfile(out_image, os.path.join(outpath,fn))
        out.append(os.path.join(outpath,fn))
    print("Isolated bands can now be found in " + outpath)
    
    return out

def generate_matchfile(all_files, reference, outname = "matches.csv", checkOverlap = False, refPoly = None, minOverlap = 80):
    
    all_files = [f for f in all_files if f != reference]
    
    if checkOverlap:
        if refPoly is None:
            print("Please provide the AOI as geojson so that I can check the scene overlap.")
            return
        print("Checking overlap with AOI")
        print("Guessing instrument from filenames..")
        
        if len(os.path.split(all_files[0])[1].split("_")) == 6:
            instrument = "PS2"
            ids = ["_".join(os.path.split(file)[1].split("_")[0:3]) for file in all_files]

        elif len(os.path.split(all_files[0])[1].split("_")) == 7:
            instrument = "PSB.SD"
            ids = ["_".join(os.path.split(file)[1].split("_")[0:4]) for file in all_files]

        gj = search.search_planet_catalog(instrument = instrument, ids = ids)
        refined = search.refine_search_and_convert_to_csv(gj, refPoly = refPoly, minOverlap = minOverlap)
        

        all_files = [f for f in all_files if any(test in f for test in refined.ids)]
        all_files.sort()

    df = pd.DataFrame({"ref":reference, "sec":all_files})
    path,_ = os.path.split(reference)
    df.to_csv(os.path.join(path, "matches.csv"), index = False)
    return df

def build_remapped_match_file_crossref(matchfile, dt_min = 365, dt_max = 861):
    #match remapped images among each other if they are further than a min time difference apart
    #only matching older ref with younger sec image
    
    df = pd.read_csv(matchfile)
    
    info = []
    
    for index, row in df.iterrows():
        path,fn = os.path.split(row.sec)
        remapped = glob.glob(row.sec[:-4]+"*_remap_Err.tif")
        scene_id = get_scene_id(fn)    
        date = get_date(scene_id)
        
        if len(remapped) == 0:
            print("Warning: No matching remapped file found.")
            info.append({
            "remapped_file": "",
            "scene_id": scene_id,
            "date": date})
        else:
            info.append({
            "remapped_file": remapped[0],
            "scene_id": scene_id,
            "date": date})
    df = pd.concat([df, pd.DataFrame(info)], axis = 1)
    
    #append reference, assuming that all images have the same refernce
    remapped = df.ref[0][:-4]+"_clip.tif"
    scene_id = get_scene_id(df.ref[0])    
    date = get_date(scene_id)
    
    ref_fill = pd.DataFrame(columns = df.columns, index = [0])
    ref_fill.remapped_file[0] = remapped
    ref_fill.scene_id[0] = scene_id
    ref_fill.date[0] = date
    
    df = pd.concat([df, ref_fill], axis = 0)
    df = df.sort_values(by = ["date"]).reset_index(drop = True)
    
    new_matches = []

    for index, row in df.iterrows():
        dt = pd.to_datetime(df.date)-row.date
        
        #filter by min and max timedifference
        dt = dt[dt > datetime.timedelta(dt_min)]
        dt = dt[dt < datetime.timedelta(dt_max)]

        if len(dt) > 0: 
            matches = df.loc[dt.index]
            new_matches.append({
                "ref": row.remapped_file,
                "sec": matches.remapped_file.tolist()})
            
    new_matches = pd.DataFrame.from_records(new_matches)
    new_matches = new_matches.explode("sec")
        
    new_matches.to_csv(path+"/matches_remapped_crossref.csv", index = False)
    print("I have stored the remapped matches under "+ path+"/matches_remapped_crossref.csv")
    return path+"/matches_remapped_crossref.csv"