#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 16:15:50 2023

@author: ariane
"""

import os, glob
import matching_functions as matf
import pandas as pd

files = glob.glob("/home/ariane/Downloads/DC_jujuy*/files/PSScene/*/basic_analytic_udm2/*AnalyticMS.tif")
tpath = "./Dove-C_Jujuy_all/L1B/"
for file in files:
    out_image = matf.isolateBand(file)
    _,fn = os.path.split(out_image)
    cmd = f"cp {out_image} {tpath}{fn}"
    os.system(cmd)
    
# refscene = tpath+"20220924_134300_71_2212_1B_AnalyticMS_b2.tif"

# refscene = tpath+"20200718_142603_1105_1B_Analytic_b2.tif"
# files = glob.glob(tpath+"*_b2.tif")
# files = [f for f in files if f != refscene]
# df = pd.DataFrame({"ref": refscene, "sec": files})
# df.to_csv(tpath+"Dove-C_matches.csv", index = False)

if __name__ == 'main':
    