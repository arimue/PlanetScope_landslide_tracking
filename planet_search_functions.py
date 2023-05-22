#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import itertools, json
import pandas as pd
from shapely.geometry import Polygon
import datetime, subprocess
from datetime import datetime
import re, sys,os
from pathlib import Path
import geopandas

def check_overlap(features, refPoly, minOverlap = 99):
    #check the overlap percentage between a list of scenes and a given reference polygon
    
    
    with open(refPoly) as f:
         gj = json.load(f)
         refPoly = Polygon([tuple(coords) for coords in gj["features"][0]["geometry"]["coordinates"][0]])

    keep = np.full(len(features), False, dtype = bool)
    
    for i, feat in enumerate(features):
        geom = feat["geometry"]["coordinates"][0]
        p = Polygon([tuple(coords) for coords in geom])
        intersection = refPoly.intersection(p)
        overlap = intersection.area/refPoly.area*100

        if overlap >= minOverlap:
            keep[i] = True

    return list(itertools.compress(features, keep))
   
def get_orbit(features, direction = "NE"):
    #for some reason, I am not able to filter by orbit given the feature properties from planet
    #thus I am naively finding the angle to north and sort accordingly
    #only useful for Dove-C
    keep = np.full(len(features), False, dtype = bool)
    for i, feat in enumerate(features): 
        geom = feat["geometry"]["coordinates"][0][0:-1] #remove last entry as it is the same as the first to close the poly
        
        #corners are unfortunately not always in the right order
        lon = [c[0] for c in geom]
        lat = [c[1] for c in geom]

        left = np.argsort(lon)[0:2]
        leftlat = np.array(lat)[left]
        #now which of these two points has the lowest latitude
        
        if np.argmin(leftlat) == 0 and direction == "NE": #NE case
            keep[i] = True
            
        elif np.argmin(leftlat) == 1 and direction == "NW": #NW case
            keep[i] = True

    return list(itertools.compress(features, keep))


def search_planet_catalog(instrument, geom = None, ids = None, date_start = "2010-01-01", date_stop = "2040-01-01", view_ang_min = 0, view_ang_max = 20, cloud_cover_max = 1, path = "./"):
    #search the planet catalog for scenes based on provided criteria
    #make sure to provide either a list of ids or a geojson that outlines the AOI
    #date_start = date_start + "T00:00:00.00Z"
    if ids is not None: 
        print("Searching for scenes based on the provided ids...") #add --std-quality 
        #search for potential scenes
        search = f"planet data filter  --date-range acquired gte {date_start} --range cloud_cover lte {cloud_cover_max} --date-range acquired lte {date_stop} --string-in instrument {instrument} --string-in id {','.join(ids)} --range view_angle gte {view_ang_min} --range view_angle lte {view_ang_max} --string-in ground_control true > {path}filter.json"
        
    elif geom is not None: 
        print("Searching for scenes based on the provided geometry...")
        search = f"planet data filter  --date-range acquired gte {date_start} --range cloud_cover lte {cloud_cover_max} --date-range acquired lte {date_stop} --string-in instrument {instrument} --geom {geom} --string-in ground_control true --range view_angle gte {view_ang_min} --range view_angle lte {view_ang_max} > {path}filter.json"
        
    else: 
        print("Please provide either a list of ids or a valid geometry to constrain the search.")
        return 
    
    result = subprocess.run(search, shell = True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.stderr != "":
        print(result.stderr)
    search = f"planet data search PSScene --limit 0 --filter {path}filter.json > {path}search.geojson"

    result = subprocess.run(search, shell = True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.stderr != "":
        print(result.stderr)
    
    return f"{path}search.geojson"

def refine_search_and_convert_to_csv(searchfile, refPoly, instrument = "PS2", orbit = "NE", minOverlap = 99):
    #filter the result from search_planet_catalog to have images that overlap the AOI polygon by min X percent and only get one orbit (only relevant for Dove-C)
    gj = [json.loads(line) for line in open(searchfile, "r")]

    if instrument == "PS2":
        features = get_orbit(gj, orbit)
    else:
        features = gj
        
    features = check_overlap(features, refPoly = refPoly, minOverlap = minOverlap)
    
    df = pd.DataFrame({"ids":[f["id"] for f in features], "view_angle":[f["properties"]["view_angle"] for f in features], "gsd":[f["properties"]["gsd"] for f in features],"sat_az":[f["properties"]["satellite_azimuth"] for f in features],"quality":[f["properties"]["quality_category"] for f in features],
                       "datetime": [datetime.strptime(f["properties"]["acquired"],"%Y-%m-%dT%H:%M:%S.%fZ") for f in features]})
    df["date"] = df['datetime'].dt.date
    
    try: #remove old searchfile
        os.remove(searchfile)
    except OSError:
        pass
    
    
    print("Updating searchfile...")
    with open(searchfile, 'a') as outfile:
        for f in features:
            #print(f)
            json.dump(f, outfile, indent = None)
            outfile.write('\n')
    
    print(f"Found {len(features)} scenes covering min. {minOverlap}% of the given AOI.")
    return df
    
def suggest_reference_and_stable_pair(df, angle_lim_ref = 0.1, angle_lim_stable = 4.5, max_day_diff= 30):
    
    #find a potential reference scene (low view angle) and corresponding stable pairs that preferrable have a high view angle difference
    #to better depict topography based on the result from refine_search_and_convert_to_csv
    #make sure to visually check images in the Planet Explorer to make sure the AOI is free of clouds
    
    pot_ref = df[(df.angle <= angle_lim_ref) & (df.quality == "standard")].reset_index(drop = True)
    pot_stable = df[df.angle >= angle_lim_stable]    
    pot_ref["matches"] = 0
    for index, row in pot_ref.iterrows():
        date_diff = abs(pot_stable["date"] - row["date"])
        date_diff = date_diff[date_diff <= datetime.timedelta(max_day_diff)]
        pot_ref["matches"][index] = len(date_diff)
    
    best_ref = pot_ref[pot_ref.matches == np.max(pot_ref.matches)].reset_index(drop = True)
    print("I can recommend the following scenes as reference scenes:")
    for index, row in best_ref.iterrows():
        print(f"{index}: {row.ids} – {row.matches} potential matches")
    
    if len(best_ref) > 1:
        choice = input(f"Please select which reference scene to choose {list(range(len(best_ref)))}\n")
        best_ref = best_ref.iloc[int(choice)]
        
        print("Returning a dataframe with the selected reference scene and corresponding stable pairs...")
        best_ref = (pd.DataFrame(best_ref.drop("matches")).T).reset_index(drop = True)
    else: 
        best_ref = best_ref.drop(["matches"], axis = 1)
    
    best_ref["type"] = "reference"
    date_diff = abs(pot_stable["date"] - best_ref["date"][0])
    date_diff = date_diff[date_diff <= datetime.timedelta(max_day_diff)]
    
    best_stable = pot_stable.loc[date_diff.index]
    best_stable["type"] = "stable pair"
    
    return pd.concat([best_ref, best_stable], axis = 0).reset_index(drop = True)

def download_xml_metadata(ids, out_dir = None):
    
    if out_dir is None:
        out_dir= str(Path.home())
    
    while True: 
        user_in = input(f"Your request will download {len(ids)} metadata files. Continue? [y/n]")
        if user_in == "n":
            print("Exiting...")
            return
        elif user_in == "y":
            break
        else: 
            print("Please provide a valid input!")
    
    #only one asset can be activated at once
    #more info: https://planet-sdk-for-python-v2.readthedocs.io/en/latest/cli/cli-data/#stats
    for i in ids:
        cmd = f"planet data asset-activate PSScene {i} basic_analytic_4b_xml"
        subprocess.run(cmd, shell = True)
    print("Waiting for assets to activate...")
    
    for i in ids:
        cmd =  f"planet data asset-wait PSScene {i} basic_analytic_4b_xml && \
                planet data asset-download PSScene {i} basic_analytic_4b_xml --directory {out_dir}"
        subprocess.run(cmd, shell = True)
    

    print(f"Downloaded files are stored under {out_dir}")
    