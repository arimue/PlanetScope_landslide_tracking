#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 15:55:06 2023

@author: ariane
"""

import os, subprocess, shutil, glob, datetime
import pandas as pd
import planet_search_functions as search
from helper_functions import get_date, get_scene_id, fixed_val_scaler, read_file
import numpy as np
from tqdm import tqdm

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

def find_best_matches(df, minGroupSize = 10, mindt = 1):
    pd.options.mode.chained_assignment = None 
    
    groups = pd.DataFrame(columns = ['ids', 'view_angle', 'gsd', 'sat_az', 'quality', 'datetime', 'date','group_id'])
    group_id = 0
    
    df = df.reset_index(drop = True)
    for idx in range(len(df)):
        scene = df.iloc[[idx]]
    
        comp = df.copy()
        comp = comp.loc[~comp.ids.isin(scene.ids)]
        comp = comp.loc[~comp.ids.isin((groups.ids))].reset_index(drop = True)
        comp["sum_va_scaled"] = fixed_val_scaler(comp.view_angle+scene.view_angle.iloc[0], 0, 10)
        comp["diff_va_scaled"] = fixed_val_scaler(abs(comp.view_angle-scene.view_angle.iloc[0]), 0, 5)
        comp["sat_az_diff"] = abs(comp.sat_az - scene.sat_az.iloc[0])
        comp.sat_az_diff[comp.sat_az_diff>180] = abs(360-comp.sat_az_diff[comp.sat_az_diff>180])
        comp["sat_az_diff_scaled"] = fixed_val_scaler(comp.sat_az_diff, 0,180)
        comp["score"] = (comp.sum_va_scaled * comp.sat_az_diff_scaled + comp.diff_va_scaled) 
        
        good_matches = comp.loc[comp.score <= 0.1]
        good_matches = good_matches[df.columns]
        

        if len(good_matches)>minGroupSize:             
            good_matches = pd.concat([good_matches, scene])
            good_matches["group_id"] = group_id
            group_id += 1
            groups = pd.concat([groups,good_matches], ignore_index=True)
            
        
    if mindt > 1:
        old_groups = groups.copy()
        groups = pd.DataFrame(columns = ['ids', 'view_angle', 'gsd', 'sat_az', 'quality', 'datetime', 'date','group_id'])

        for gi in range(group_id):
            #print(gi)
            group = old_groups.loc[old_groups.group_id == gi]
            group = group.sort_values("datetime").reset_index(drop = True)
        #discard scenes to comply with dt lim
            dt = group.datetime.diff()
            dt.iloc[0] = datetime.timedelta(days = mindt)
            group = group.loc[dt >= datetime.timedelta(days = mindt)]
            groups = pd.concat([groups,group], ignore_index=True)
            
    

    return groups

def rate_match(infodf, matchdf, level = 3):
    #rate existing matches, do not suggest
    #sort by reference if not already to be able to calculate scores in batches
    matchdf = matchdf.sort_values("ref").reset_index(drop = True)
    scores = []
    for ref in matchdf.ref.unique():
        
        #print(ref)
        refid = get_scene_id(ref, level = level)
        secids = [get_scene_id(sec, level = level) for sec in matchdf.sec.loc[matchdf.ref == ref]]
        refinfo = infodf.loc[infodf.ids == refid]
        secinfo = infodf.loc[infodf.ids.isin(secids)]
        sum_va_scaled = fixed_val_scaler(refinfo.view_angle.iloc[0]+secinfo.view_angle, 0, 10)
        diff_va_scaled = fixed_val_scaler(abs(refinfo.view_angle.iloc[0]-secinfo.view_angle), 0, 5)
        sat_az_diff = abs(refinfo.sat_az.iloc[0]-secinfo.sat_az)
        sat_az_diff[sat_az_diff>180] = abs(360-sat_az_diff[sat_az_diff>180])
        sat_az_diff_scaled = fixed_val_scaler(sat_az_diff, 0,180)
        score =  (sum_va_scaled * sat_az_diff_scaled + diff_va_scaled )
        
        scores.append({
            "refid": refid,
            "secid": secinfo.ids, 
            "score": score})
    scores = pd.DataFrame.from_records(scores).explode(["secid","score"]).reset_index(drop = True)
    return(scores)

def add_offset_variance_to_rated_matches(scores, stereopath, prefix_ext = "L3B"):
    
    scores = scores.reset_index(drop = True)
    dvar = np.zeros([len(scores),2])
    dvar[:] = np.nan
    
    for i, row in tqdm(scores.iterrows(), total=scores.shape[0]):
        prefix = row.refid + "_" + row.secid + "L3B"
        
        if os.path.isfile(stereopath+prefix+"-F.tif"):
            dx = read_file(stereopath+prefix+"-F.tif",1)
            dy = read_file(stereopath+prefix+"-F.tif", 2)
            mask = read_file(stereopath+prefix+"-F.tif",3)
            dx[mask == 0] = np.nan
            dy[mask == 0] = np.nan
            
            dvar[i,0] = np.nanvar(dx)
            dvar[i,1] = np.nanvar(dy)
        else: 
            print(f"File {stereopath}{prefix}-F.tif not found...")
            
    scores["dx_var"] = dvar[:,0]
    scores["dy_var"] = dvar[:,1]

    return scores
    
def add_offset_from_mask_to_rated_matches(scores, stereopath, mask, prefix_ext = "L3B"):
    
    scores = scores.reset_index(drop = True)
    dvar = np.zeros([len(scores),2])
    dvar[:] = np.nan
    aoi_mask = read_file(mask)
    colnames  =  [d+"_"+str(i) for i in np.unique(aoi_mask) for d in ["dx", "dy"]]

    out = np.zeros([len(scores), len(np.unique(aoi_mask))*2])
    out[:] = np.nan
    for idx, row in tqdm(scores.iterrows(), total=scores.shape[0]):
        prefix = row.refid + "_" + row.secid + "L3B"
        
        if os.path.isfile(stereopath+prefix+"-F.tif"):
            dx = read_file(stereopath+prefix+"-F.tif",1)
            dy = read_file(stereopath+prefix+"-F.tif", 2)
            dmask = read_file(stereopath+prefix+"-F.tif",3)
            dx[dmask == 0] = np.nan
            dy[dmask == 0] = np.nan
                        
            ii = 0
            for maskval in np.unique(aoi_mask):
                out[idx, ii] = np.nanmean(dx[aoi_mask == maskval])
                ii+=1
                out[idx, ii] = np.nanmean(dy[aoi_mask == maskval])
                ii+=1
            
            
        else: 
            print(f"File {stereopath}{prefix}-F.tif not found...")
            
    outdf = pd.DataFrame(out, columns = colnames)
    scores = pd.concat([scores, outdf], axis = 1)
    
    return(scores)

def generate_matchfile_from_groups(groups, path, ext = "_3B_AnalyticMS_SR_clip_b2.tif", checkExistence = True):
    
    if checkExistence:
        exists = [os.path.isfile(path+i+ext) for i in groups.ids]
        groups = groups.loc[exists].reset_index(drop = True)

    matches = []
    for group in groups.group_id.unique():
        gdf = groups.loc[groups.group_id == group].sort_values("ids").reset_index(drop = True)
        for i in range(len(gdf)-1):
            matches.append({
                "ref": gdf.ids.iloc[i],
                "sec": list(gdf.ids.iloc[i+1:])})
            
    matches = pd.DataFrame.from_records(matches).explode("sec")
    
    matches.ref = path + matches.ref+ ext
    matches.sec = path + matches.sec+ ext
    
    matches.to_csv(path+"matches.csv", index = False)
    
    return matches


def match_all(df, path, ext = "_3B_AnalyticMS_SR_clip_b2.tif", checkExistence = False):
    
    matches = []
    df = df.sort_values("ids").reset_index(drop = True)
    if checkExistence: 
        
        exists = [os.path.isfile(path+ i+ ext) for i in df.ids]
        df = df.loc[exists].reset_index(drop = True)
        
    for i in range(len(df)-1):
        matches.append({
            "ref": df.ids.iloc[i],
            "sec": list(df.ids.iloc[i+1:])})
            
    matches = pd.DataFrame.from_records(matches).explode("sec")
    matches.ref = path + matches.ref+ ext
    matches.sec = path + matches.sec+ ext
    
    matches.to_csv(path+"all_matches.csv", index = False)
    
    return matches
    


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

def build_remapped_match_file_crossref(matchfile, dt_min = 365, dt_max = 861, level = 1, pattern = "_remap_Err.tif"):
    #match remapped images among each other if they are further than a min time difference apart
    #only matching older ref with younger sec image
    
    df = pd.read_csv(matchfile)
    
    info = []
    
    for index, row in df.iterrows():
        path,fn = os.path.split(row.sec)
        remapped = glob.glob(row.sec[:-4]+"*"+pattern)
        scene_id = get_scene_id(fn, level = level)    
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
    scene_id = get_scene_id(df.ref[0], level = level)    
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