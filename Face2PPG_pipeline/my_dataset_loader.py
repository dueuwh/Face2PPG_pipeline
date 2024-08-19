# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 11:23:15 2024

@author: ys
"""

import os
import json

class pure_loader():
    """
    PURE Dataset

    .. PURE dataset structure:
    .. -----------------
    ..     datasetDIR/   
    ..     |
    ..     |-- 01-01/
    ..     |---- Image...1.png
    ..     |---- Image.....png
    ..     |---- Image...n.png
    ..     |-- 01-01.json
    ..     |...
    ..     |...
    ..     |-- nn-nn/
    ..     |---- Image...1.png
    ..     |---- Image.....png
    ..     |---- Image...n.png        
    ..     |-- nn-nn.json
    ..     |...
    
    @ Image folders, label.json files are in the same folder @
    
    """
    
    def __init__(self, base_path):
        self.bp = base_path
        self.folder_list = [name for name in os.listdir(self.bp) if 'json' not in name]
        self.label_list = [name for name in os.listdir(self.bp) if 'json' in name]
    
    def synchronization(self):
        a = 1
        

if __name__ == "__main__":
    pure_dataset = "D:/home/rPPG/data/PURE_rPPG/"
    folder_list = [name for name in os.listdir(pure_dataset) if 'json' not in name]
    label_list = [name for name in os.listdir(pure_dataset) if 'json' in name]
    
    with open(pure_dataset+label_list[0], 'r') as f:
        label_0101 = json.load(f)
    
    full = label_0101['/FullPackage']
    image_timestamp = label_0101['/Image']
    