#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
#make sure to add the path to the PlanetScope_landslide_tracking folder so that the system can locate the modules
sys.path.append("/home/ariane/Documents/PlanetScope/PlanetScope_landslide_tracking")

import planet_search_functions as search
import preprocessing_functions as preprocessing
import postprocessing_functions as postprocessing
import optimization_functions as opt
import asp_helper_functions as asp
import glob
import helper_functions as helper
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

"""
Exemplary python script for data search, DEM generation and L1B and L3B workflow.
Paths, parameters and filenames need to be adjusted.
For details check Tutorials on GitHub.
"""
###########PARAMETERS##########################################################################################
amespath = "/home/ariane/Downloads/StereoPipeline-3.2.0-2023-01-01-x86_64-Linux/bin"
instrument = "PSB.SD"
aoi = "test_aoi.geojson"
cop_dem = "Copernicus_DEM_NWArg.tif"
epsg = 32720

###########DATA#SEARCH#########################################################################################
searchfile = search.search_planet_catalog(instrument = instrument, aoi = aoi, cloud_cover_max=0.1, date_start = "2020-03-01", date_stop = "2023-06-30")
scenes = search.refine_search_and_convert_to_csv(searchfile, aoi = aoi, instrument = instrument, min_overlap = 99)
groups = search.find_common_perspectives(scenes, va_diff_thresh = 0.3, min_group_size = 5, min_dt = 30, searchfile = searchfile)

###########L3B#WORKFLOW########################################################################################
files = glob.glob("/home/ariane/Downloads/test_psscene_analytic_sr_udm2/PSScene/*3B_AnalyticMS_SR_clip.tif")
work_dir = "./L3B"
preprocessing.get_single_band(files, out_path = work_dir, band_nr = 2)
matches = preprocessing.match_all(work_dir, ext = "_b2.tif", dt_min = 180)

dmaps = asp.correlate_asp_wrapper(amespath, matches, sp_mode = 2, corr_kernel = 35, prefix_ext = "_L3B", overwrite=True)
dmaps_pfit = opt.apply_polyfit(matches, prefix_ext= "_L3B", order = 2, demname = cop_dem)

#plotting example
fn = dmaps_pfit[0]
dx = helper.read_file(fn)
dy = helper.read_file(fn, 2)
fig, ax = plt.subplots(1, 2, figsize=(8, 4))
im0 = ax[0].imshow(dx, vmin=-2, vmax=2, cmap="coolwarm")
im1 = ax[1].imshow(dy, vmin=-2, vmax=2, cmap="coolwarm")

ax[0].set_title("dx")
ax[1].set_title("dy")

# Add colorbars
divider0 = make_axes_locatable(ax[0])
cax0 = divider0.append_axes("right", size="5%", pad=0.05)
cbar0 = fig.colorbar(im0, cax=cax0, label='Offset [pix]')

divider1 = make_axes_locatable(ax[1])
cax1 = divider1.append_axes("right", size="5%", pad=0.05)
cbar1 = fig.colorbar(im1, cax=cax1, label='Offset [pix]')

plt.tight_layout()
plt.savefig("./figures/pfit_2nd_order_elev.png", dpi = 300)

vels = postprocessing.calc_velocity_wrapper(matches, prefix_ext = "_L3B_polyfit", overwrite = True)

postprocessing.stack_rasters(matches, prefix_ext = "_L3B_polyfit", what = "velocity")
stats = postprocessing.get_stats_in_aoi(matches, aoi = aoi, prefix_ext = "_L3B", take_velocity=True)
stats = postprocessing.get_stats_in_aoi(matches, xcoord = 200, ycoord = 300, pad = 3, prefix_ext = "L3B", take_velocity=False)

###########DEM#GENERATION#####################################################################################
dem_aoi = "dem_aoi.geojson"

#optional: get a suggestion for scene pairs to use:
searchfile = search.search_planet_catalog(instrument = instrument, aoi = dem_aoi, cloud_cover_max=0.1, date_start = "2020-03-01", date_stop = "2023-06-30")
scenes = search.refine_search_and_convert_to_csv(searchfile, aoi = dem_aoi, instrument = instrument, min_overlap = 99)
pairs = search.suggest_dem_pairs(scenes, min_va = 5, max_dt = 30)

img1 = "./test/20220907_140709_64_24a3_1B_AnalyticMS_b2.tif"
img2 = "./test/20220912_141056_91_2486_1B_AnalyticMS_b2.tif"

planet_dem = asp.dem_building(amespath, img1, img2, epsg = epsg, aoi = dem_aoi, refdem = cop_dem)
planet_dem_aligned = opt.disparity_based_DEM_alignment(amespath, img1, img2, planet_dem, cop_dem, epsg = epsg, aoi = aoi, iterations = 1)

###########L1B#WORKFLOW#######################################################################################
files = glob.glob("/home/ariane/Downloads/test_psscene_basic_analytic_udm2/PSScenes/*_AnalyticMS.tif")
work_dir = "./L1B"

preprocessing.get_single_band(files, out_path = work_dir, band_nr = 2)
preprocessing.orthorectify_L1B(amespath, files, planet_dem_aligned, aoi, epsg = epsg, pad = 200)
matches = preprocessing.match_all(work_dir, ext = "_b2_clip_mp_clip.tif", dt_min = 180)

dmaps = asp.correlate_asp_wrapper(amespath, matches, sp_mode = 2, corr_kernel = 35, prefix_ext = "_L1B", overwrite=True)
dmaps_pfit = opt.apply_polyfit(matches, prefix_ext= "_L1B", order = 2, demname = cop_dem)
vels = postprocessing.calc_velocity_wrapper(matches, prefix_ext = "_L1B_polyfit", overwrite = True)
postprocessing.stack_rasters(matches, prefix_ext = "_L1B_polyfit", what = "velocity")


#########REMAPPING############################################################################################
matches = preprocessing.match_to_one_ref("/home/ariane/Documents/PlanetScope/Siguas/L3B/group1")
dmaps_pfit = opt.apply_polyfit(matches, prefix_ext= "_L3B", order = 2, demname = cop_dem, save_remapped_sec = True)

#########MAKE#GIF#############################################################################################
postprocessing.make_video(matches, video_name = "group1_remapped.mp4", ext = "_remap", crop = 300)