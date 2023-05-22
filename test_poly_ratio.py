#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 15:55:33 2023

@author: ariane
"""
import numpy as np
import pandas as pd
from helper_functions import read_file
import scipy.optimize
import matplotlib.pyplot as plt

def polyXYZ3(X, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t):
    x,y,z = X
    out = a*x**3 + b*y**3 + c*z**3 + d*x*y**2 + e*x*z**2 + f*y*x**2 + g*y*z**2 + h*z*x**2 + i*z*y**2 + j*x*y*z + k*x**2 + l*y**2 + m*z**2 + n*x*y + o*x*z + p*y*z + q*x + r*y + s*z + t
    return out

# def polyXYZ32(X, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t):
#     x,y,z,firstfit = X
#     out = firstfit/(a*x**3 + b*y**3 + c*z**3 + d*x*y**2 + e*x*z**2 + f*y*x**2 + g*y*z**2 + h*z*x**2 + i*z*y**2 + j*x*y*z + k*x**2 + l*y**2 + m*z**2 + n*x*y + o*x*z + p*y*z + q*x + r*y + s*z + t)
#     return out


# def polyXYZ3(X, a, b, c, d, e, f, g, h, i, j):
#     x,y,z = X
#     out = a*x**2 + b*y**2 + c*z**2 + d*x*y + e*x*z + f*y*z + g*x + h*y + i*z + j
#     return out
# def polyXYZ3(X, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, a1, b1, c1, d1, e1, f1, g1, h1, i1, j1, k1, l1, m1, n1, o1, p1, q1, r1, s1, t1):
#     x,y,z = X
#     out = (a*x**3 + b*y**3 + c*z**3 + d*x*y**2 + e*x*z**2 + f*y*x**2 + g*y*z**2 + h*z*x**2 + i*z*y**2 + j*x*y*z + k*x**2 + l*y**2 + m*z**2 + n*x*y + o*x*z + p*y*z + q*x + r*y + s*z + t)/\
#           (a1*x**3 + b1*y**3 + c1*z**3 + d1*x*y**2 + e1*x*z**2 + f1*y*x**2 + g1*y*z**2 + h1*z*x**2 + i1*z*y**2 + j1*x*y*z + k1*x**2 + l1*y**2 + m1*z**2 + n1*x*y + o1*x*z + p1*y*z + q1*x + r1*y + s1*z + t1)
#     return out

path = "/home/ariane/Documents/PlanetScope/testProjective/"
dem_err_x = "/home/ariane/Documents/PlanetScope/MinaPurna/L1B/stereo/bm_ck35/dem_error_dx_mean_bm_ck35.tif"
dem_err_y = "/home/ariane/Documents/PlanetScope/MinaPurna/L1B/stereo/bm_ck35/dem_error_dy_mean_bm_ck35.tif"
disp_fn = path+"20220823_133636_33_2465_20210820_134119_90_2450_clip-F.tif"
reduce = 100
direction = "x"
dispX = read_file(disp_fn, 1)
dispY = read_file(disp_fn, 2) 
mask = read_file(disp_fn, 3)

xgrid, ygrid = np.meshgrid(np.arange(0,dispX.shape[1], 1), np.arange(0, dispX.shape[0], 1))#
xgridf = xgrid.flatten()
ygridf = ygrid.flatten()
fit_points = pd.DataFrame({"x":xgrid.flatten(), "y":ygrid.flatten(), "dx": dispX.flatten(),"dy": dispY.flatten(), "mask":mask.flatten()})


errx = read_file(dem_err_x)
fit_points["errx"] = errx.flatten()
egrid = errx

erry = read_file(dem_err_y)
fit_points["erry"] = erry.flatten()
    
fit_points = fit_points[fit_points["mask"] != 0]
fit_points = fit_points.iloc[::reduce, :]
fit_points = fit_points.dropna()


if direction == "x":
    dgrid = dispX
elif direction == "y":
    dgrid = dispY

#clean upper and lower percentiles to avoid the impact of abnormally high values
dup = np.percentile(fit_points[f"d{direction}"], 90)
dlow = np.percentile(fit_points[f"d{direction}"], 10)

fit_points = fit_points.loc[(fit_points[f"d{direction}"] >= dlow) & (fit_points[f"d{direction}"] <= dup)]


dup = np.percentile(fit_points[f"err{direction}"], 95)
dlow = np.percentile(fit_points[f"err{direction}"], 5)

fit_points = fit_points.loc[(fit_points[f"err{direction}"] >= dlow) & (fit_points[f"err{direction}"] <= dup)]


it = 1
iqr_diff = 100 #some arbitrary high number
max_iterations = 10
max_iqr_diff = 0.01
iqr_now = np.nanpercentile(dgrid.flatten(), 75)-np.nanpercentile(dgrid.flatten(),25)

while iqr_diff > max_iqr_diff and it <= max_iterations:
    xcoeffs3, xcov3 = scipy.optimize.curve_fit(polyXYZ3, xdata = (fit_points.x, fit_points.y, fit_points[f"err{direction}"]), ydata = fit_points[f"d{direction}"])
    dg3 = polyXYZ3((xgridf,ygridf,egrid.flatten()), *xcoeffs3)
    fit_points["d_fit3"] = fit_points[f"d{direction}"] - polyXYZ3((fit_points.x, fit_points.y, fit_points[f"err{direction}"]), *xcoeffs3) 

    disp_corrected = dgrid-dg3.reshape(xgrid.shape)
    disp_corrected [mask==0] = np.nan
    
    #filter points that have a disparity greater than 1 pixel (optional)
    
    dup = np.percentile(fit_points["d_fit3"], 95)
    dlow = np.percentile(fit_points["d_fit3"], 5)

    fit_points = fit_points.loc[(fit_points["d_fit3"] >= dlow) & (fit_points["d_fit3"] <= dup)]
    
    iqr_before = iqr_now
    iqr_now = np.nanpercentile(disp_corrected.flatten(), 75)-np.nanpercentile(disp_corrected.flatten(),25)
    iqr_diff = iqr_before - iqr_now
    print(f"Iteration: {it}, IQR now: {iqr_now}, IQR change: {iqr_diff}")
    it+=1
    
 
# dup = np.percentile(fit_points[f"err{direction}"], 95)
# dlow = np.percentile(fit_points[f"err{direction}"], 5)

# fit_points = fit_points.loc[(fit_points[f"err{direction}"] >= dlow) & (fit_points[f"err{direction}"] <= dup)]

    
# xcoeffs32, xcov32 = scipy.optimize.curve_fit(polyXYZ3, xdata = (fit_points.x, fit_points.y, fit_points[f"err{direction}"]), ydata = fit_points["d_fit3"])
# dg32 = polyXYZ3((xgridf,ygridf,egrid.flatten()), *xcoeffs32)
# fit_points["d_fit32"] = fit_points[f"d{direction}"] - polyXYZ3((fit_points.x, fit_points.y, fit_points["d_fit3"]), *xcoeffs3) 

disp_corrected = dgrid-dg3.reshape(xgrid.shape)#-dg32.reshape(xgrid.shape)
disp_corrected [mask==0] = np.nan    
    

plt.figure()    
plt.imshow(disp_corrected, vmin = -1, vmax = 1)
