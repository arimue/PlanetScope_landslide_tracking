#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 19:09:01 2023

@author: ariane
"""

import json

with open('config.json', 'r') as config_file:
    config = json.load(config_file)
    
print(config)