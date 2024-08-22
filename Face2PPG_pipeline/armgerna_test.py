import os
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

base_dir = "D:/home/BCML/drax/PAPER/data/results/rppg_emma/"
label_dir = "D:/home/BCML/drax/PAPER/data/labels/synchro/"

folder_list = [name for name in os.listdir(base_dir) if "LHT_1" not in name]

for folder in folder_list:
    subject = folder.split('_')[0]+'_'+folder.split('_')[1]
    label = pd.read_csv(label_dir + subject+'_synchro.csv', index_col=0)
    if label.shape[1] > 1:
        
    bpm = []
    idx = []
    with open(base_dir + folder + '/omit_total.txt', 'r') as f:
        lines = f.readlines()
        
        for line in lines:
            bpm.append(int(line.split(' ')[0]))
            idx.append(int(line.split(' ')[1]))
    
    
        
