#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob, subprocess, os, shutil
from helper_functions import clip_raw, size_from_aoi, get_scene_id
import pandas as pd


def correlate_asp(amespath, img1, img2, prefix = "run", session = "rpc", sp_mode = 1, method = "asp_bm", nodata_value = None, corr_kernel = 35):
    
    """
    Run ASP stereo correlation in correlator mode using the input parameters.

    Args:
        amespath (str): Path to the ASP installation.
        img1 (str): Path to the reference image.
        img2 (str): Path to the secondary image.
        prefix (str): Prefix for output files (default: "run").
        session (str): Session type (default: "rpc").
        sp_mode (int): Subpixel mode (default: 1).
        method (str): Stereo algorithm method (default: "asp_bm").
        nodata_value (float or None): Nodata value for output disparity maps (default: None).
        corr_kernel (int): Correlation kernel size (default: 35).

    Returns:
        str: Path to the folder where disparity maps are saved.

    """
    
    folder,_ = os.path.split(img1)
    print(f"Data will be saved under {os.path.join(folder, 'disparity_maps')}")
    
    if method == "asp_bm":
        cmd = f"{os.path.join(amespath, 'parallel_stereo')} {img1} {img2} {os.path.join(folder, 'disparity_maps', prefix)} --correlator-mode -t {session} --datum Earth --skip-rough-homography --stereo-algorithm {method} --subpixel-mode {sp_mode} --corr-kernel {corr_kernel} {corr_kernel} --subpixel-kernel {corr_kernel+10} {corr_kernel+10} --threads 0" 
        if nodata_value is not None: 
            cmd = f"{cmd} --nodata-value {nodata_value}"
    else:
        print(corr_kernel)
        if (corr_kernel > 9) or (corr_kernel%2 == 0):
            print("Correlation kernel size is not suitable for mgm. Pick an odd kernel size <= 9!")
            return
        cmd = f"{os.path.join(amespath, 'parallel_stereo')} {img1} {img2} {os.path.join(folder, 'disparity', prefix)} --correlator-mode -t {session} --datum Earth --skip-rough-homography --stereo-algorithm {method} --corr-kernel 9 9 --subpixel-mode {sp_mode} --subpixel-kernel {corr_kernel*2+1} {corr_kernel*2+1} --threads 0" 

        if nodata_value is not None: 
            cmd = f"{cmd} --nodata-value {nodata_value}"
            
    subprocess.run(cmd, shell = True)
    
    return os.path.join(folder, 'disparity_maps')


def clean_asp_files(path, prefix):
    
    """
    Remove unnecessary ASP files, keeping only the filtered disparity maps.

    Args:
        path (str): Path to the folder containing the ASP files.
        prefix (str): Prefix used in the ASP file names.

    """
    
    files = glob.glob(f"{path}/{prefix}-*")
    disp  = glob.glob(f"{path}/{prefix}-F.tif")
    remove = set(files)-set(disp)
    
    for file in remove:
        try:
            os.remove(file)
        except IsADirectoryError: #if parallel_stereo is used, also remove folders
            shutil.rmtree(file)


def correlate_asp_wrapper(amespath, matches, prefix_ext = "", sp_mode = 2, corr_kernel = 35, flip = False, overwrite = False):
    """
     Wrapper function for performing ASP correlation on multiple image pairs based on provided matches.
    
     Args:
         amespath (str): Path to the ASP installation.
         matches (str or pd.core.frame.DataFrame): Path to the matchfile or DataFrame containing the matches.
         prefix_ext (str): Prefix extension for output files (default: "").
         sp_mode (int): Subpixel mode (default: 2).
         corr_kernel (int): Correlation kernel size (default: 35).
         flip (bool): Flag to indicate reversing the order of matches (default: False).
         overwrite (bool): Flag to indicate overwriting existing disparity maps (default: False).
    
     Returns:
         list: List of output disparity map paths.
    
     """
    if type(matches) == str:
        try:
            df = pd.read_csv(matches)
        except FileNotFoundError:
            print("Could not find the provided matchfile.")
            return
    elif type(matches) == pd.core.frame.DataFrame:
        df = matches.copy()
    else:
        print("Matches must be either a string indicating the path to a matchfile or a pandas DataFrame.")
        return
    
    if flip: # flipping dataframe. makes sense if running correlation on multiple machines, 
        df = df.reindex(index=df.index[::-1]).reset_index(drop = True)

    df["id_ref"] = df.ref.apply(get_scene_id)
    df["id_sec"] = df.sec.apply(get_scene_id)
    df["path"] =  df["ref"].apply(lambda x: os.path.split(x)[0])
    
    out = []
    for _,row in df.iterrows():
    
        prefix = row.id_ref + "_" + row.id_sec + prefix_ext
        if (not os.path.isfile(os.path.join(row.path,"disparity_maps",prefix+"-F.tif"))) or overwrite:
            outpath = correlate_asp(amespath, row.ref, row.sec, prefix = prefix, session = "rpc", sp_mode = sp_mode, method = "asp_bm", nodata_value = None, corr_kernel = corr_kernel)
            clean_asp_files(outpath, prefix)
        else: 
            print("Disparity map exists. Skipping correlation...")
        out.append(os.path.join(row.path,"disparity_maps",prefix+"-F.tif"))
    
    return(out)


def mapproject(amespath, img, dem, epsg, img_with_rpc = None, ba_prefix = None, ext = "mp", resolution = 3):
    
    """
    Map-project (orthorectify) raw image data onto DEM.

    Parameters:
    amespath (str): Path to the Ames Stereo Pipeline installation.
    img (str): Path to the input image.
    dem (str): Path to the Digital Elevation Model (DEM).
    epsg (int): EPSG code of the target spatial reference system.
    img_with_rpc (str, optional): Path to the image with RPC metadata (default: None).
    ba_prefix (str, optional): Prefix for bundle adjustment (default: None).
    ext (str, optional): Output file extension (default: "mp").
    resolution (int, optional): Target resolution (default: 3).

    Returns:
    str: Path to the mapprojected image.
    """
    # requires the image to have RPCs in the header. These can be added with copy_rpc if missing or just simply providing the image with rpc metadata
    
    if img_with_rpc is not None:
        cmd = f"{os.path.join(amespath, 'mapproject_single')} {dem} {img} {img_with_rpc} {img[:-4]}_{ext}.tif --threads 0 -t rpc --t_srs epsg:{epsg} --tr {resolution} --no-bigtiff --tif-compress Deflate --nodata-value -9999"
    else: 
        cmd = f"{os.path.join(amespath, 'mapproject_single')} {dem} {img} {img[:-4]}_{ext}.tif -t rpc --threads 0 --t_srs epsg:{epsg} --tr {resolution} --no-bigtiff --tif-compress Deflate --nodata-value -9999"
 
    if ba_prefix is not None: 
        cmd = f"{cmd} --bundle-adjust-prefix {ba_prefix}"

    subprocess.run(cmd, shell = True)
    return f"{img[:-4]}_{ext}.tif"
    
            

def dem_building(amespath, img1, img2, epsg, aoi = None, refdem = None, prefix = None, corr_kernel = 35, overwrite = False):
    
    #TODO implement check that the epsg is always a projected EPSG
    if prefix is None: 
        id1 = get_scene_id(img1)
        id2 = get_scene_id(img2)

        prefix = f"{id1}_{id2}"
        
    print(f"All outputs will be saved with the prefix {prefix}.")
    #TODO: implement buffer for aoi
    if aoi is not None:
        assert refdem is not None, "Need to provide a reference DEM when working with AOI to guess coodinates."
        #TODO: implement epsg finder
        #TODO: implement GSD finder
        ul_lon, ul_lat, xsize, ysize = size_from_aoi(aoi, epsg = epsg, gsd = 4)
        img1 = clip_raw(img1, ul_lon, ul_lat, xsize, ysize, refdem)
        ul_lon, ul_lat, xsize, ysize = size_from_aoi(aoi, epsg = epsg, gsd = 4)
        img2 = clip_raw(img2, ul_lon, ul_lat, xsize, ysize, refdem)
        
    path, fn1 = os.path.split(img1)
    _, fn2 = os.path.split(img2)
        
    if (not (os.path.isfile(f"{path}/bundle_adjust/{prefix}-{fn1[:-4]}.adjust") and os.path.isfile(f"{path}/bundle_adjust/{prefix}-{fn2[:-4]}.adjust"))) or overwrite:
        cmd = f"{os.path.join(amespath, 'bundle_adjust')} -t rpc {img1} {img2} -o {path}/bundle_adjust/{prefix}"
        subprocess.run(cmd, shell = True)
    else:
        print("Using existing bundle adjustment files.")
        
    if (not os.path.isfile(f"{path}/stereo_run1/{prefix}-PC.tif")) or overwrite:
        cmd = f"{os.path.join(amespath, 'stereo')} {img1} {img2} {path}/stereo_run1/{prefix} -t rpc --datum Earth --bundle-adjust-prefix {path}/bundle_adjust/{prefix} --stereo-algorithm asp_bm --subpixel-mode 2 --threads 0 --corr-kernel {corr_kernel} {corr_kernel} --subpixel-kernel {corr_kernel+10} {corr_kernel+10}" 
        subprocess.run(cmd, shell = True)
    else:
        print(f"Using triangulated points from existing file {path}/stereo_run1/{prefix}-PC.tif")
    
    if (not os.path.isfile(f"{path}/point2dem_run1/{prefix}-DEM.tif")) or overwrite:
        cmd = f"{os.path.join(amespath, 'point2dem')} {path}/stereo_run1/{prefix}-PC.tif --tr 90 --t_srs EPSG:{epsg} -o {path}/point2dem_run1/{prefix}" 
        subprocess.run(cmd, shell = True)
    else:
        print(f"Using existing DEM {path}/point2dem_run1/{prefix}-DEM.tif")
    
    #need to use the actual mapproject command, not mapproject_single to keep the rpc information 
    
    cmd = f"{os.path.join(amespath, 'mapproject')} {path}/point2dem_run1/{prefix}-DEM.tif {img1} {img1[:-4]}_mp.tif -t rpc --threads 0 --t_srs epsg:{epsg} --tr 3 --no-bigtiff --tif-compress Deflate --nodata-value -9999"
    subprocess.run(cmd, shell = True)
    cmd = f"{os.path.join(amespath, 'mapproject')} {path}/point2dem_run1/{prefix}-DEM.tif {img2} {img2[:-4]}_mp.tif -t rpc --threads 0 --t_srs epsg:{epsg} --tr 3 --no-bigtiff --tif-compress Deflate --nodata-value -9999"
    subprocess.run(cmd, shell = True)
    
    mp1 = img1[:-4]+"_mp.tif"
    mp2 = img2[:-4]+"_mp.tif"
    
    
    #need to copy bundle adjusted files, because the program doesnt find it anymore due to name changes
    shutil.copyfile(f"{path}/bundle_adjust/{prefix}-{fn1[:-4]}.adjust", f"{path}/bundle_adjust/{prefix}-{fn1[:-4]}_mp.adjust")
    shutil.copyfile(f"{path}/bundle_adjust/{prefix}-{fn2[:-4]}.adjust", f"{path}/bundle_adjust/{prefix}-{fn2[:-4]}_mp.adjust")
    
    if (not os.path.isfile(f"{path}/stereo_run2/{prefix}-PC.tif")) or overwrite:
        cmd = f"{os.path.join(amespath, 'stereo')} {mp1} {mp2} -t rpcmaprpc --datum Earth --bundle-adjust-prefix {path}/bundle_adjust/{prefix} {path}/stereo_run2/{prefix} {path}/point2dem_run1/{prefix}-DEM.tif --stereo-algorithm asp_bm --subpixel-mode 2 --corr-kernel 65 65 --subpixel-kernel 75 75" 
        subprocess.run(cmd, shell = True)
    else:
        print(f"Using triangulated points from existing file {path}/stereo_run2/{prefix}-PC.tif")
    
    if (not os.path.isfile(f"{path}/point2dem_run2/{prefix}-DEM.tif")) or overwrite:
        cmd = f"{os.path.join(amespath, 'point2dem')} {path}/stereo_run2/{prefix}-PC.tif --tr 30 --t_srs EPSG:{epsg} --dem-hole-fill-len 10 -o {path}/point2dem_run2/{prefix}" 
        subprocess.run(cmd, shell = True)
    else:
        print(f"Using existing DEM {path}/point2dem_run2/{prefix}-DEM.tif")
        
        
    #alignment with demcoreg. Sometimes not working so well, so I prefer to use the disparity based alignment approach (see core functions)
    # print(f"Aligning DEM to {refdem}...")
    
    # #RefDEM will need to be in UTM coordinates
    
    # epsg = get_epsg(refdem)
    
    
    # if epsg == 4326:
    #     print("Reprojecting the reference DEM to a projected CRS...")
    #     refdem = warp(refdem, epsg = epsg)
        
    # cmd = f"{demcoregpath}dem_align.py {refdem} {path}/point2dem_run2/{prefix}-DEM.tif -mode nuth -max_dz 1000 -max_offset 500"
    # subprocess.run(cmd, shell = True)
    #TODO: improve max offset constraints (initial guess?) catch errors better so that process doesnt have to start from the beginning, i.e. implement overwrite check


    print("Done!")
    
    
def image_align_asp(amespath, img1, img2, prefix = None):
    
    #run ASP image_align with disparity derived from running the correlation
    
    folder = img1.replace(img1.split("/")[-1], "")
    print(f"Data will be saved under {folder}image_align/")
    if prefix: 
        cmd = f"{amespath}image_align {img1} {img2} -o {img2[:-4]}_aligned.tif --output-prefix {folder}image_align/{prefix} --alignment-transform affine --disparity-params '{folder}stereo/{prefix}-F.tif 10000' --inlier-threshold 100" 
    else:
        cmd = f"{amespath}image_align {img1} {img2} -o {img2[:-4]}_aligned.tif --output-prefix {folder}image_align/{prefix} --alignment-transform affine  --inlier-threshold 100" 

    subprocess.run(cmd, shell = True)

def parse_match_asp(amespath, img1, img2, prefix = "run"):
    
    #turn a .match file (output from image_align) into readable .txt format 
    
    folder = img1.replace(img1.split("/")[-1], "")+"image_align/"
    matchfile = glob.glob(f"{folder}{prefix}-*-clean.match")
    if len(matchfile)>1:
        print("More that one matching file found. Please check if prefixes were used more that once...")
        return
    matchfile = matchfile[0]
    cmd = f"python {amespath}parse_match_file.py {matchfile} {matchfile[:-6]}.txt"
    subprocess.run(cmd, shell = True)
    return f"{matchfile[:-6]}.txt"

def read_match(matchfile):
    
    #convert a match.txt file to better readable df 
    
    df = pd.read_csv(matchfile, skiprows = 1, header = None, sep = " ")
    nrIPs = pd.read_csv(matchfile, nrows = 1, header = None, sep = " ")

    df1 = df.head(nrIPs[0][0]).reset_index(drop = True)
    df2 = df.tail(nrIPs[1][0]).reset_index(drop = True)

    df = pd.DataFrame({"x_img1":df1[0], "y_img1":df1[1],"x_img2":df2[0], "y_img2":df2[1]})
    
    return df