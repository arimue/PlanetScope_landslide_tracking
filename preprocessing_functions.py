#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import subprocess 
import shutil
import glob
import pandas as pd
import helper_functions as helper
import numpy as np
from tqdm import tqdm
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

def preprocess_scenes(files, out_path = "./", band_nr = 2): #TODO: fetch work_dir globally
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

def add_offset_variance_to_rated_matches(scores, stereopath, prefix_ext = "L3B"):
    
    scores = scores.reset_index(drop = True)
    dvar = np.zeros([len(scores),2])
    dvar[:] = np.nan
    
    for i, row in tqdm(scores.iterrows(), total=scores.shape[0]):
        prefix = row.refid + "_" + row.secid + "L3B"
        
        if os.path.isfile(stereopath+prefix+"-F.tif"):
            dx = helper.read_file(stereopath+prefix+"-F.tif",1)
            dy = helper.read_file(stereopath+prefix+"-F.tif", 2)
            mask = helper.read_file(stereopath+prefix+"-F.tif",3)
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
    aoi_mask = helper.read_file(mask)
    colnames  =  [d+"_"+str(i) for i in np.unique(aoi_mask) for d in ["dx", "dy"]]

    out = np.zeros([len(scores), len(np.unique(aoi_mask))*2])
    out[:] = np.nan
    for idx, row in tqdm(scores.iterrows(), total=scores.shape[0]):
        prefix = row.refid + "_" + row.secid + "L3B"
        
        if os.path.isfile(stereopath+prefix+"-F.tif"):
            dx = helper.read_file(stereopath+prefix+"-F.tif",1)
            dy = helper.read_file(stereopath+prefix+"-F.tif", 2)
            dmask = helper.read_file(stereopath+prefix+"-F.tif",3)
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
        if len(df) < 1: 
            print("I could not find any matching scenes. Check if the provided path and file extent is correct.")
            return
        
    for i in range(len(df)-1):
        matches.append({
            "ref": df.ids.iloc[i],
            "sec": list(df.ids.iloc[i+1:])})
                
            
    matches = pd.DataFrame.from_records(matches).explode("sec").reset_index(drop = True)
    matches.ref = path + matches.ref+ ext
    matches.sec = path + matches.sec+ ext
    
    
    if dt is not None:
        date_ref = [helper.get_date(helper.get_scene_id(i)) for i in matches.ref]
        date_sec = [helper.get_date(helper.get_scene_id(i)) for i in matches.sec]
        good_dt = [i for i in range(len(date_ref)) if (date_sec[i]-date_ref[i]).days > dt]
        
        matches = matches.iloc[good_dt].reset_index(drop = True)
        
    matches.to_csv(path+"all_matches.csv", index = False)
    
    return matches


def orthorectify_L1B(files, demname, aoi, epsg, amespath, pad = 0):
    ul_lon, ul_lat, xsize, ysize = helper.size_from_aoi(aoi, epsg = epsg, gsd = 4)

    for f in files: 
        #its best to roughly clip before mapprojection, otherwise the process takes long
        clip = helper.clip_raw(f, ul_lon, ul_lat, xsize+pad, ysize+pad, demname)
        mp = asp.mapproject(amespath, clip, demname, epsg = epsg)
        #finetune clip, bc clipping with RPCs is not always super exact
        _ = helper.clip_mp_cutline(mp, aoi)
    