#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from helper_functions import read_file, read_meta
import os
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
    
    