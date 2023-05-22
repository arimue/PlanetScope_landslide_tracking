#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 12:19:59 2023

@author: ariane
"""
import cv2
from helper_functions import read_file
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing

path = "/home/ariane/Documents/PlanetScope/testProjective/"
#path = "/raid-manaslu/amueting/PhD/Project1/ImageTransformation/testProjective/"
img_src = cv2.imread(path+"20220823_133636_33_2465_1B_AnalyticMS_b2_clip.tif", cv2.IMREAD_UNCHANGED)
img_dst = cv2.imread(path+"20210820_134119_90_2450_1B_AnalyticMS_b2_clip.tif", cv2.IMREAD_UNCHANGED)
dx = read_file(path+"20220823_133636_33_2465_20210820_134119_90_2450_clip-F.tif", b=1)
dy = read_file(path+"20220823_133636_33_2465_20210820_134119_90_2450_clip-F.tif", b=2)
mask = read_file(path+"20220823_133636_33_2465_20210820_134119_90_2450_clip-F.tif", b=3)

dx[mask==0] = np.nan
dy[mask==0] = np.nan

#dx = dx[0:200,:]
#dy = dy[0:200,:]
xgrid, ygrid = np.meshgrid(np.arange(0,dx.shape[1], 1), np.arange(0, dx.shape[0], 1))

Ms = np.zeros([dx.shape[0],9])
pad = 50

out = np.zeros(dx.shape)

def getM(row):
    M = np.zeros(9)
    print(row)
    if row < pad: # upper edge case
        lolim = 0
        uplim = row+pad
    if row > dx.shape[0]-pad: #lower edge case
        lolim = row-pad
        uplim = dx.shape[0]
    else:
        lolim = row-pad
        uplim = row+pad
                
    src_coords = np.array([xgrid[lolim:uplim,:].flatten(), ygrid[lolim:uplim,:].flatten()]).transpose()
    dst_coords = np.array([xgrid[lolim:uplim,:].flatten()+dx[lolim:uplim,:].flatten(), ygrid[lolim:uplim,:].flatten()+dy[lolim:uplim,:].flatten()]).transpose()
    
    # #remove nan entries
    src_coords =  src_coords[~np.isnan(dx[lolim:uplim,:].flatten())]
    dst_coords =  dst_coords[~np.isnan(dx[lolim:uplim,:].flatten())]

    
    if len(src_coords) > 1000:
        M, mask = cv2.findHomography(src_coords, dst_coords, 0, cv2.RANSAC,5.0)
        if M is not None:
            M = M.flatten() 
    else:
        pass
        #print(f"Not sufficient tie points for row {row} with a padding of {pad} pixels.")
    
    return M
    # plt.figure()
    # plt.imshow(trans_dst)
    
    # cv2.imwrite(path+"src.tif", img_src)
    # cv2.imwrite(path+"dst.tif", img_dst)
pool = multiprocessing.Pool(processes=multiprocessing.cpu_count()-2)
results = [pool.apply_async(getM, [row]) for row in range(xgrid.shape[0])]
for row, res in enumerate(results):
    Ms[row,:] = res.get()
    
for i in range(Ms.shape[0]):
    M = Ms[i,:].reshape(3,3)   
    if not np.max(M) == np.max(M) == 0:
        trans_dst = cv2.warpPerspective(img_dst, M, (img_dst.shape[1], img_dst.shape[0]))
        # plt.figure()
        # plt.imshow(trans_dst)
        out[i,:] = trans_dst[i,:]

cv2.imwrite(path+"trans.tif", out)
