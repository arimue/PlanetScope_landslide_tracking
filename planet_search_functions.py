#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import itertools
import json
import pandas as pd
from shapely.geometry import Polygon
import datetime
import subprocess
import os
from pathlib import Path

def check_overlap(features, aoi, min_overlap=99):
    """
    Check the overlap percentage between a list of scenes and a given reference polygon.

    Args:
        features (list): List of scenes.
        aoi (str): Path to the reference polygon file.
        min_overlap (int or float) Minimum overlap percentage (default: 99).

    Returns:
        list: List of scenes that satisfy the minimum overlap.

    """
    with open(aoi) as f:
        gj = json.load(f)
        aoi = Polygon([tuple(coords) for coords in gj["features"][0]["geometry"]["coordinates"][0]])

    keep = np.full(len(features), False, dtype=bool)

    for i, feat in enumerate(features):
        geom = feat["geometry"]["coordinates"][0]
        p = Polygon([tuple(coords) for coords in geom])
        intersection = aoi.intersection(p)
        overlap = intersection.area / aoi.area * 100

        if overlap >= min_overlap:
            keep[i] = True

    return list(itertools.compress(features, keep))

   
def get_orbit(features, direction="NE"):
    """
    Filter features by orbit direction.

    It is not possible to filter by orbit given the feature properties from Planet.
    Thus, the function naively finds the angle to north and sorts accordingly.
    Only relevant for PS2 (Dove-C) data.

    Args:
        features (list): List of features to filter.
        direction (str): Orbit direction ("NE" for Northeast, "NW" for Northwest) (default: "NE").

    Returns:
        list: Filtered list of features.

    """
    keep = np.full(len(features), False, dtype=bool)

    for i, feat in enumerate(features):
        geom = feat["geometry"]["coordinates"][0][0:-1]  # Remove last entry as it is the same as the first to close the poly

        # Corners are unfortunately not always in the right order
        lon = [c[0] for c in geom]
        lat = [c[1] for c in geom]

        left = np.argsort(lon)[0:2]
        leftlat = np.array(lat)[left]
        
        # Now find which of these two points has the lowest latitude
        if np.argmin(leftlat) == 0 and direction == "NE":  # NE case
            keep[i] = True

        elif np.argmin(leftlat) == 1 and direction == "NW":  # NW case
            keep[i] = True

    return list(itertools.compress(features, keep))



def search_planet_catalog(instrument, aoi = None, ids = None, date_start = "2010-01-01", date_stop = "2040-01-01", view_ang_min = 0, view_ang_max = 20, cloud_cover_max = 1, path = "./"):
    """
    Search the planet catalog for scenes based on provided criteria.
    Make sure to provide either a list of ids or a GeoJSON that outlines the AOI.

    Args:
        instrument (str): Instrument name (PSB.SD, PS2.SD or PS2)
        aoi (str): Path to the GeoJSON file outlining the AOI (default: None).
        ids (list): List of scene ids (default: None).
        date_start (str): Start date in the format "YYYY-MM-DD" (default: "2010-01-01").
        date_stop (str): Stop date in the format "YYYY-MM-DD" (default: "2040-01-01").
        view_ang_min (int): Minimum view angle (default: 0).
        view_ang_max (int): Maximum view angle (default: 20).
        cloud_cover_max (int): Maximum cloud cover (default: 1).
        path (str): Path to store the search results (default: "./").

    Returns:
        str: Path to the generated search GeoJSON file.

    """
    
    if ids is not None: 
        print("Searching for scenes based on the provided ids...") # add --std-quality to exclude test scenes
        #search for potential scenes
        search = f"planet data filter  --date-range acquired gte {date_start} --range cloud_cover lte {cloud_cover_max} --date-range acquired lte {date_stop} --string-in instrument {instrument} --string-in id {','.join(ids)} --range view_angle gte {view_ang_min} --range view_angle lte {view_ang_max} --string-in ground_control true > {path}filter.json"
        
    elif aoi is not None: 
        print("Searching for scenes based on the provided geometry...")
        search = f"planet data filter  --date-range acquired gte {date_start} --range cloud_cover lte {cloud_cover_max} --date-range acquired lte {date_stop} --string-in instrument {instrument} --geom {aoi} --string-in ground_control true --range view_angle gte {view_ang_min} --range view_angle lte {view_ang_max} > {path}filter.json"
        
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


def refine_search_and_convert_to_csv(searchfile, aoi, instrument = "PSB.SD", orbit = "NE", min_overlap = 99, az_angle = 10):
    """
    Filter the result from search_planet_catalog to have images that overlap the AOI polygon by at least a minimum percentage.
    Filter data by orbit (only relevant for Dove-C).
    Calculate true view angle considering look direction. 
    
    Args:
        searchfile (str): Path to the search file.
        aoi (str): Path to the reference polygon GeoJSON file.
        instrument (str): Instrument name. Not relevant if not PS2. (default: "PSB.SD"). 
        orbit (str): Orbit direction. Not relevant if not PS2. ("NE" for Northeast, "NW" for Northwest) (default: "NE").
        min_overlap (int or float): Minimum overlap percentage (default: 99).
        az_angle (int or float): Estimated azimuth angle (azAngle in XML metadata), i.e. flight line angle to north (default: 10) 

    Returns:
        pandas.DataFrame: DataFrame containing the filtered information.

    """
    gj = [json.loads(line) for line in open(searchfile, "r")]

    if instrument == "PS2":
        features = get_orbit(gj, orbit)
    else:
        features = gj
        
    features = check_overlap(features, aoi = aoi, min_overlap = min_overlap)
    
    df = pd.DataFrame({"ids":[f["id"] for f in features], "view_angle":[f["properties"]["view_angle"] for f in features], "gsd":[f["properties"]["gsd"] for f in features],"sat_az":[f["properties"]["satellite_azimuth"] for f in features],"quality":[f["properties"]["quality_category"] for f in features],
                       "datetime": [datetime.datetime.strptime(f["properties"]["acquired"],"%Y-%m-%dT%H:%M:%S.%fZ") for f in features], "footprint":[f["geometry"]["coordinates"][0] for f in features], "sun_azimuth":[f["properties"]["sun_azimuth"] for f in features],"sun_elevation":[f["properties"]["sun_elevation"] for f in features]})
    df["date"] = df['datetime'].dt.date
    
    #calculating true view angle  
    # to distinguish right from left looking, we assume an azimuth angle (scan line angle to north) of 10 degrees (typically ranges between 7 and 12 degrees)
    #for precise info, xml metadata would need to be downloaded, but using 10 degrees shound be sufficient in 99% of cases
  
    df["true_view_angle"] = df.view_angle
    df.true_view_angle[(df.sat_az>(180+az_angle)) | (df.sat_az<(az_angle))] = df.true_view_angle*-1
    
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
    
    print(f"Found {len(features)} scenes covering min. {min_overlap}% of the given AOI.")
    return df


def find_common_perspectives(df, va_diff_thresh = 0.5, min_group_size = 5, min_dt = 1):
    """
    Group scenes that fulfill search criteria into groups with a common satellite perspective.

    Args:
        df (pandas.DataFrame): DataFrame containing scene information.
        va_diff_thresh (float): True view angle threshold for considering scenes to have a common perspective (default: 0.5).
        min_group_size (int): Minimum number of scenes required to form a group (default: 5).
        min_dt (int): Minimum temporal baseline between scenes in a group (default: 1).

    Returns:
        pandas.DataFrame: DataFrame containing grouped scenes.

    """
    pd.options.mode.chained_assignment = None 
    
    groups = pd.DataFrame(columns = [*df.columns,'group_id'])
    group_id = 0
    
    df = df.reset_index(drop = True)
    for idx in range(len(df)):
        scene = df.iloc[[idx]]
    
        comp = df.copy()
        comp = comp.loc[~comp.ids.isin(scene.ids)]
        comp = comp.loc[~comp.ids.isin((groups.ids))].reset_index(drop = True) #do not double assign scenes to groups
        comp["va_diff"] = abs(comp.true_view_angle - scene.true_view_angle.iloc[0])
        good_matches = comp.loc[comp.va_diff <= va_diff_thresh]
        good_matches = good_matches[df.columns] #remove va_diff col

        if len(good_matches) >= min_group_size:     

            good_matches = pd.concat([good_matches, scene])
            good_matches["group_id"] = group_id
            group_id += 1
            groups = pd.concat([groups,good_matches], ignore_index=True)
            
    if min_dt > 1:
        old_groups = groups.copy()
        groups = pd.DataFrame(columns =  [*df.columns,'group_id'])

        for gi in range(group_id):

            group = old_groups.loc[old_groups.group_id == gi]
            group = group.sort_values("datetime").reset_index(drop = True)
            #discard scenes to stay within dt limits
            dt = group.datetime.diff()
            dt.iloc[0] = datetime.timedelta(days = min_dt)
            group = group.loc[dt >= datetime.timedelta(days = min_dt)]
            groups = pd.concat([groups,group], ignore_index=True)
            
    #TODO: could update the search.json file to only keep relevant scenes and add group attribute
            
    return groups
    
def download_xml_metadata(ids, out_dir = None):
    """
   Download XML metadata files for the provided scene ids.

   Args:
       ids (list): List of scene ids.
       out_dir (str): Output directory to store the downloaded files (default: Home directory).

   """
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
    