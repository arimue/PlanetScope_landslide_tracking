#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from helper_functions import read_file, read_meta, min_max_scaler, get_date, get_scene_id
import os
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cv2 as cv

def get_vlims(x):
    #get limits for colorbar based on percentile
    
    plow = np.nanpercentile(x, 2)
    pup = np.nanpercentile(x, 98)
    return np.max([abs(plow), abs(pup)])

def plot_all_offset_results(files, band = 1, width = 7.5, height = 5, outfile = "", meanshift = False):   
    #plot all disparity rasters together
    
    ysize = int(np.ceil(len(files)/3))
    xsize = int(np.ceil(len(files)/ysize))
        
    fig, ax = plt.subplots(ysize, xsize, figsize = (width*xsize, height*ysize))
    k = 0
    for i in range(ysize):
        for j in range(xsize):
            try:
                _,fn = os.path.split(files[k])
                ident = fn.removesuffix('-F.tif')
                dat = read_file(files[k], b=band)
                meta = read_meta(files[k])
                #check if mask exists
                if meta["count"] ==3:   
                    mask = read_file(files[k], b = 3)
                    dat[mask == 0] = np.nan
                #remove meanshift
                if meanshift:
                    dat = dat - np.nanmean(dat)
                p = ax[i,j].imshow(dat, vmin = -2, vmax = 2, cmap = "coolwarm")#vmin = -get_vlims(dat), vmax = get_vlims(dat)
                ax[i,j].set_title(f"{ident}", fontsize=11)
                divider = make_axes_locatable(ax[i,j])
                cax = divider.append_axes("right", size="5%", pad=0.05)
                fig.colorbar(p, cax=cax, label='Offset [pix]')  
                k+=1
            except IndexError:
                pass
    plt.tight_layout()
    if outfile != "":
        plt.savefig(outfile, dpi = 300)
    plt.show()
    
    
def plot_single_results(fn1, fn2 = None, plot_velocity = False, limP = 2, limH = 4, limV = 4):
    #plot image and histogram of dx and dy
    
    if fn2 is not None: #get x and y from different files
        dx = read_file(fn1, 1)
        dy = read_file(fn2, 1)
    else: #get x and y from different bands
        dx = read_file(fn1, 1)
        dy = read_file(fn1, 2)
        mask = read_file(fn1, 3)
        dx[mask == 0] = np.nan
        dy[mask == 0] = np.nan
        
    if plot_velocity:
        #add velocity plot
        vel = read_file(fn1[:-4]+"_velocity.tif", 1)
        fig, ax = plt.subplots(3,2, figsize = (14,12))
        ax[2,0].imshow(vel, vmin = 0, vmax = limV, cmap = "Reds")
        ax[2,1].hist(vel.flatten(), bins = 100, range = (0, limV))
        ax[2,1].axvline(np.nanpercentile(vel, 75), color="red", linestyle="-")
        ax[2,1].text(.9, .9, f"P75 = {np.round(np.nanpercentile(vel, 75),3)}", color = "red" , ha='right', va='top', transform = ax[2,1].transAxes)
        

    else:
        fig, ax = plt.subplots(2,2, figsize = (14,8))
    ax[0,0].imshow(dx, vmin = -limP, vmax = limP, cmap = "coolwarm")
    ax[0,1].imshow(dy, vmin = -limP, vmax = limP, cmap = "coolwarm")
    ax[1,0].hist(dx.flatten(), bins = 100, range = (-limH, limH))
    ax[1,0].axvline(0, color="black", linestyle="--")
    ax[1,0].axvline(np.nanpercentile(dx, 25), color="red", linestyle="-")
    ax[1,0].axvline(np.nanpercentile(dx, 75), color="red", linestyle="-")
    ax[1,0].text(.9, .9, f"IQR = {np.round(np.nanpercentile(dx, 75)-np.nanpercentile(dx, 25),3)}", color = "red" , ha='right', va='top', transform = ax[1,0].transAxes)
    ax[1,1].hist(dy.flatten(), bins = 100, range = (-limH, limH))
    ax[1,1].axvline(0, color="black", linestyle="--")
    ax[1,1].axvline(np.nanpercentile(dy, 25), color="red", linestyle="-")
    ax[1,1].axvline(np.nanpercentile(dy, 75), color="red", linestyle="-")
    ax[1,1].text(.9, .9, f"IQR = {np.round(np.nanpercentile(dy, 75)-np.nanpercentile(dy, 25),3)}", color = "red" , ha='right', va='top', transform = ax[1,1].transAxes)
    
    #add 5th and 95th percentile
    ax[1,0].axvline(np.nanpercentile(dx, 5), color="blue", linestyle="-")
    ax[1,0].axvline(np.nanpercentile(dx, 75), color="blue", linestyle="-")
    ax[1,0].text(.9, .8, f"P95-P5: = {np.round(np.nanpercentile(dx, 75)-np.nanpercentile(dx, 5),3)}", color = "blue" , ha='right', va='top', transform = ax[1,0].transAxes)

    ax[1,1].axvline(np.nanpercentile(dy, 5), color="blue", linestyle="-")
    ax[1,1].axvline(np.nanpercentile(dy, 95), color="blue", linestyle="-")
    ax[1,1].text(.9, .8, f"P95-P5: = {np.round(np.nanpercentile(dy, 95)-np.nanpercentile(dy, 5),3)}", color = "blue" , ha='right', va='top', transform = ax[1,1].transAxes)
 
    plt.show()
    
    
def make_video(matchfile, video_name = "out.mp4", refvalue = 7916, refposx = 70, refposy = 214):
    #TODO: Improve this!!!
    df = pd.read_csv(matchfile)
    
    df["remapped"] = df.sec.str.rstrip('.tif') + "_clip_remap_Err.tif"
    df["id_ref"] = df.ref.apply(get_scene_id)
    df["id_sec"] = df.sec.apply(get_scene_id)
    df["date_ref"] = df.id_ref.apply(get_date)
    df["date_sec"] = df.id_sec.apply(get_date)

    refposx = 145
    refposy = 275
    refvalue = 4923
    
    dates = list(df.date_sec)
    dates.append(df.date_ref[0])
    dates.sort()
    files = list(df.remapped)
    files.append(df.ref[0][:-4]+"_clip.tif")
    files.sort()
    
    timeline = pd.date_range(min(dates), max(dates),freq='d')
    timeline_files = []
    
    for time in timeline:
        dt = [abs((time-d).days) for d in dates]
        idx = dt.index(min(dt)) 
        timeline_files.append(files[idx])
    for f in files: 

        img = cv.imread(f, cv.IMREAD_UNCHANGED)
        #img = img[600:900, 1450:1900]
        img = img[700:1100,1550:2300]
        img = img.astype(np.float32)
        img[img == 0] = np.nan
        
        diff_at_refpos = refvalue - img[refposy, refposx]
        
        img = img + diff_at_refpos
        
        #img[img > np.nanpercentile(img, 99)] = np.nanpercentile(img, 99)
        img = min_max_scaler(img)
        #img = img/np.nanmax(img)*255
        img = img * (2**16-1)
        img[np.isnan(img)] = 0

        img = img.astype(np.uint16)
        

        #equalize histogram 
        #img = cv.equalizeHist(img)
        
        cv.imwrite(f[:-4]+ "_eq.tif", img)
        
    path = "/home/ariane/Documents/PlanetScope/Dove-C_Jujuy_all/L1B/"
    timeline_files = [path + "20161205_134509_0e1f_1B_AnalyticMS_b2_clip_remap_Err.tif", path + "20171212_135346_1022_1B_AnalyticMS_b2_clip_remap_Err.tif", path + "20181001_140308_0f35_1B_Analytic_b2_clip_remap_Err.tif", path + "20201012_141343_1012_1B_AnalyticMS_b2_clip_remap_Err.tif", path + "20220412_140659_103b_1B_AnalyticMS_b2_clip_remap_Err.tif"]
    
    path = "/home/ariane/Documents/PlanetScope/SD_Jujuy_5deg/"
    timeline_files = [path + "20200516_134707_21_2277_1B_AnalyticMS_b2_clip_remap_Err.tif", path + "20220912_141056_91_2486_1B_AnalyticMS_b2_clip_remap_Err.tif", path + "20200516_134707_21_2277_1B_AnalyticMS_b2_clip_remap_Err.tif", path + "20220912_141056_91_2486_1B_AnalyticMS_b2_clip_remap_Err.tif"]
    
    
    with open('file_list.txt', 'w') as fl:
        for line in timeline_files:
            fl.write(f"file {line[:-4]}_eq.tif\n")
    
    cmd = f"ffmpeg -r 1 -safe 0 -y -f concat -i file_list.txt -vf 'pad=ceil(iw/2)*2:ceil(ih/2)*2' -r 10 -c:v libx264 -pix_fmt yuv420p /home/ariane/Documents/PlanetScope/SuperDove_selected_images.mp4"
    os.system(cmd)

    cmd = f"ffmpeg -y -i /home/ariane/Documents/PlanetScope/SuperDove_selected_images.mp4 -vf 'fps=10,scale=1080:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse' -loop 0 SuperDove_selected_images.gif"  
    os.system(cmd)
    # frame = cv.imread(files[0], cv.IMREAD_UNCHANGED)
    # height, width = frame.shape
    
    # video = cv.VideoWriter(video_name, 0, 1, (width,height))
    
    # for f in files:
    #     img = cv.imread(f, cv.IMREAD_UNCHANGED)
    #     img =img/np.max(img)*255
    #     img = img.astype(np.uint8)
    #     video.write(img)
    
    # cv.destroyAllWindows()
    # video.release()
    