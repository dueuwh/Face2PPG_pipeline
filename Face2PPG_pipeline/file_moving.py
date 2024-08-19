# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 10:51:47 2024

@author: ys
"""

import os
import shutil

base_dir = "D:/home/rPPG/data/PURE_rPPG/"
folder_list = [name for name in os.listdir(base_dir) if '.json' not in name]

for folder in folder_list:
    json_file_flag = True
    contents_in_folder = os.listdir(base_dir+folder)
    for name in contents_in_folder:
        if '.json' in name:
            source_path = base_dir+folder+'/'+name
            destination_folder = base_dir
            shutil.move(source_path, destination_folder)
            print(f"{folder} json file moved")
            json_file_flag = False
            break
    if json_file_flag:
        print("f{folder} json file is not detected")
        