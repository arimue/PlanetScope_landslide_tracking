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
from pyproj import Transformer, CRS
from shapely.geometry import Polygon
from xml.dom import minidom


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

# def shortest_distance_to_line(point, line_start, line_end):
#     #see https://stackoverflow.com/questions/39840030/distance-between-point-and-a-line-from-two-points
#     point = np.array(point)
#     line_start = np.array(line_start)
#     line_end = np.array(line_end)

#     # Calculate shortest distance
#     shortest_distance = np.cross(point - line_start, line_end - line_start) / np.linalg.norm(line_end - line_start)

#     return shortest_distance

# def get_poi_ydistance(scene, poi, epsg):
#     #TODO: implement epsg guesser and check in which CRS POI coords are given
#     #transform to UTM to have differences in m
#     proj_tr = Transformer.from_crs(CRS("EPSG:4326"), CRS("EPSG:"+str(epsg)), always_xy=True)  #! need always_xy = True otherwise does strange things
#     poi_proj = proj_tr.transform(*poi)
#     upt1_proj = proj_tr.transform(*scene.upper_pt1)
#     upt2_proj = proj_tr.transform(*scene.upper_pt2)

#     dist_to_upper_line = shortest_distance_to_line(poi_proj, upt1_proj, upt2_proj)
    
#     return dist_to_upper_line

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
        comp["score"] = 1.5 - (comp.sum_va_scaled * comp.sat_az_diff_scaled + comp.diff_va_scaled) 
        
        good_matches = comp.loc[comp.score >= 1.4]
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
        xml_file = glob.glob(f"/home/ariane/Documents/PlanetScope/test_ang_calc/{refid}*metadata.xml")
        if len(xml_file)  == 1:
            
            xmldoc = minidom.parse(xml_file[0])
            ang_to_north_ref = float(xmldoc.getElementsByTagName("ps:azimuthAngle")[0].firstChild.data)
        else:
            ang_diff_ref = np.nan
            
        ang_diff_secs = []
        for secid in secids:

            xml_file = glob.glob(f"/home/ariane/Documents/PlanetScope/test_ang_calc/{secid}*metadata.xml")
            if len(xml_file) == 1:
                
                xmldoc = minidom.parse(xml_file[0])
                inc_ang = float(xmldoc.getElementsByTagName("eop:incidenceAngle")[0].firstChild.data)
                view_ang = float(xmldoc.getElementsByTagName("ps:spaceCraftViewAngle")[0].firstChild.data)
                       
                ang_diff_secs.append(abs(inc_ang-view_ang))
            else:
                ang_diff_secs.append(np.nan)
                
            
        scores.append({
            "refid": refid,
            "secid": secinfo.ids, 
            "score": score, 
            "overlap": overlap,
            "va_sum":refinfo.view_angle.iloc[0]+secinfo.view_angle,
            "va_diff": abs(refinfo.view_angle.iloc[0]-secinfo.view_angle),
            "az_diff": sat_az_diff})
            #"ang_diff_diff": [a-ang_diff_ref for a in ang_diff_secs] })
        
    scores = pd.DataFrame.from_records(scores).explode(["secid","score", "overlap", "va_sum", "va_diff", "az_diff"]).reset_index(drop = True)
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


def match_all(df, path, ext = "_3B_AnalyticMS_SR_clip_b2.tif", dt = None, checkExistence = False):
    
    matches = []
    df = df.sort_values("ids").reset_index(drop = True)
    if checkExistence: 
        
        exists = [os.path.isfile(path+ i+ ext) for i in df.ids]
        df = df.loc[exists].reset_index(drop = True)
        
    for i in range(len(df)-1):
        matches.append({
            "ref": df.ids.iloc[i],
            "sec": list(df.ids.iloc[i+1:])})
                
            
    matches = pd.DataFrame.from_records(matches).explode("sec").reset_index(drop = True)
    matches.ref = path + matches.ref+ ext
    matches.sec = path + matches.sec+ ext
    
    
    if dt is not None:
        date_ref = [get_date(get_scene_id(i, level = 3)) for i in matches.ref]
        date_sec = [get_date(get_scene_id(i, level = 3)) for i in matches.sec]
        good_dt = [i for i in range(len(date_ref)) if (date_sec[i]-date_ref[i]).days > dt]
        
        matches = matches.iloc[good_dt].reset_index(drop = True)
        
    matches.to_csv(path+"all_matches.csv", index = False)
    
    return matches
    