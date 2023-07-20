#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 15:41:47 2023

@author: ariane
"""
import planet_search_functions as search
import preprocessing_functions as preprocessing
import postprocessing_functions as postprocessing
import optimization_functions as opt
import asp_helper_functions as asp
import pandas as pd
import glob, os
import numpy as np
import helper_functions as helper
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


# work_dir = "./tutorial/"
# work_dir = '/home/ariane/Downloads/test'
# work_dir = "/home/ariane/Documents/PlanetScope/delMedio/L3B/group3/"
# aoi = os.path.join("./tutorial/","test_aoi.geojson") #TODO: check that AOI is in EPSG:4326, else reproject
# instrument = "PSB.SD"
# amespath = "/home/ariane/Downloads/StereoPipeline-3.1.1-alpha-2022-10-16-x86_64-Linux/bin"
# # searchfile = search.search_planet_catalog(instrument = instrument, aoi = aoi, cloud_cover_max=0.1, date_start = "2020-03-01", date_stop = "2023-06-30")
# # scenes = search.refine_search_and_convert_to_csv(searchfile, aoi = aoi, min_overlap = 99)
# # groups = search.find_common_perspectives(scenes, va_diff_thresh = 0.6, min_group_size = 5, min_dt = 30, searchfile = searchfile)

# files = glob.glob("/home/ariane/Downloads/Siguas_L3B_fillGaps_psscene_analytic_sr_udm2/PSScene/*3B_AnalyticMS_SR_clip.tif")
# #preprocessing.get_single_band(files, out_path = "/home/ariane/Downloads/test/", band_nr = 2)
# #matches = preprocessing.match_all(work_dir, ext = "_b2.tif", dt_min = 180)
# #matches = preprocessing.generate_matchfile_from_search(scenes, dt_min = 180, path = work_dir, check_existence=True)
# matches = preprocessing.generate_matchfile_from_groups(groups, dt_min = 600, path = work_dir, check_existence=True)
# matches = matches.iloc[0:2]

# dmaps = asp.correlate_asp_wrapper(amespath, matches, sp_mode = 1, corr_kernel = 35, prefix_ext = "_L3B", overwrite=True)

# fn = dmaps[0]
# dx = helper.read_file(fn)
# dy = helper.read_file(fn, 2)
# mask = helper.read_file(fn, 3)
# fig, ax = plt.subplots(1, 3, figsize=(12, 4))
# im0 = ax[0].imshow(dx, vmin=-2, vmax=2, cmap="coolwarm")
# im1 = ax[1].imshow(dy, vmin=-2, vmax=2, cmap="coolwarm")
# im2 = ax[2].imshow(mask, vmin=0, vmax=1, cmap="Greys")
# ax[0].set_title("dx")
# ax[1].set_title("dy")

# ax[2].set_title("mask")

# # Add colorbars
# divider0 = make_axes_locatable(ax[0])
# cax0 = divider0.append_axes("right", size="5%", pad=0.05)
# cbar0 = fig.colorbar(im0, cax=cax0, label='Offset [pix]')

# divider1 = make_axes_locatable(ax[1])
# cax1 = divider1.append_axes("right", size="5%", pad=0.05)
# cbar1 = fig.colorbar(im1, cax=cax1, label='Offset [pix]')

# divider2 = make_axes_locatable(ax[2])
# cax2 = divider2.append_axes("right", size="5%", pad=0.05)
# cbar2 = fig.colorbar(im2, cax=cax2, label='Mask value')
# plt.tight_layout()

# # plt.savefig("./tutorial/figures/disp_map.png", dpi = 300)
# demname = "/home/ariane/Documents/PlanetScope/DEMcomp/final_with_shifted_PlanetDEM/delMedio/final_aligned_PlanetDEM.tif"

# dmaps_pfit = opt.apply_polyfit(matches, prefix_ext= "L3B", order = 2, demname = demname)
# fn = dmaps_pfit[0]
# dx = helper.read_file(fn)
# dy = helper.read_file(fn, 2)
# fig, ax = plt.subplots(1, 2, figsize=(8, 4))
# im0 = ax[0].imshow(dx, vmin=-2, vmax=2, cmap="coolwarm")
# im1 = ax[1].imshow(dy, vmin=-2, vmax=2, cmap="coolwarm")

# ax[0].set_title("dx")
# ax[1].set_title("dy")


# # Add colorbars
# divider0 = make_axes_locatable(ax[0])
# cax0 = divider0.append_axes("right", size="5%", pad=0.05)
# cbar0 = fig.colorbar(im0, cax=cax0, label='Offset [pix]')

# divider1 = make_axes_locatable(ax[1])
# cax1 = divider1.append_axes("right", size="5%", pad=0.05)
# cbar1 = fig.colorbar(im1, cax=cax1, label='Offset [pix]')

# plt.tight_layout()
# plt.savefig("./tutorial/figures/pfit_2nd_order_elev.png", dpi = 300)


# vels = postprocessing.calc_velocity_wrapper(matches, prefix_ext = "L3B_polyfit", overwrite = True)
# fn = vels[0]
# v = helper.read_file(fn)
# d = helper.read_file(fn, 2)
# fig, ax = plt.subplots(1, 2, figsize=(8, 4))
# im0 = ax[0].imshow(v, vmin=0, vmax=5, cmap="Reds")
# im1 = ax[1].imshow(d, vmin=0, vmax=360, cmap="twilight")

# ax[0].set_title("Velocity")
# ax[1].set_title("Direction")


# # Add colorbars
# divider0 = make_axes_locatable(ax[0])
# cax0 = divider0.append_axes("right", size="5%", pad=0.05)
# cbar0 = fig.colorbar(im0, cax=cax0, label='Velocity [m/yr]')

# divider1 = make_axes_locatable(ax[1])
# cax1 = divider1.append_axes("right", size="5%", pad=0.05)
# cbar1 = fig.colorbar(im1, cax=cax1, label='Direction [°]')

# plt.tight_layout()
# plt.savefig("./tutorial/figures/velocity_direction.png", dpi = 300)

# postprocessing.stack_rasters(matches, prefix_ext = "L3B_polyfit", what = "velocity")


#####################################
#L1B
# aoi = "/home/ariane/Documents/PlanetScope/delMedio/del_medio_aoi.geojson"
# files = glob.glob("/home/ariane/Documents/PlanetScope/delMedio/L1B/group4/*_b2.tif")
# files = files[0:2]
# preprocessing.orthorectify_L1B(amespath, files, demname, aoi, epsg = 32720, pad = 200)
# matches = preprocessing.match_all("/home/ariane/Documents/PlanetScope/delMedio/L1B/group4/", ext = "_b2_clip_mp_clip.tif", dt_min = 180)


####
#DEM gen
amespath = "/raid-manaslu/amueting/StereoPipeline-3.1.1-alpha-2022-07-29-x86_64-Linux/bin/"
img1 = "./delMedio/L1B/test/20220907_140709_64_24a3_1B_AnalyticMS_b2.tif"
img2 = "./delMedio/L1B/test/20220912_141056_91_2486_1B_AnalyticMS_b2.tif"
aoi = "./delMedio/L1B/test/dem_aoi.geojson"

#asp.dem_building(amespath, img1, img2, epsg = 32720, aoi = aoi, refdem = "./delMedio/L1B/test/output_COP30.tif")
opt.disparity_based_DEM_alignment(amespath, img1, img2, "./delMedio/L1B/test/point2dem_run2/20220907_140709_64_24a3_20220912_141056_91_2486-DEM_final.tif", "./delMedio/L1B/test/output_COP30.tif", epsg = 32720, iterations = 3)