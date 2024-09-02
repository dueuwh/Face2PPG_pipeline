import os
import matplotlib.pyplot as plt
import pickle
import numpy as np
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import pandas as pd


def ploting_rppg():
    base_dir = "D:/home/BCML/IITP/data/DEAP/rppg/emma/"
    subject_list = os.listdir(base_dir)
    
    for subject in subject_list:
        folder_list = os.listdir(base_dir+subject)
        for folder in folder_list:
            sample = f"D:/home/BCML/IITP/data/DEAP/rppg/emma/{subject}/{folder}/ppg_omit.txt"
            
            data = []
            
            with open(sample, 'r') as f:
                lines = f.readlines()
                temp_step = []
                for line in lines:
                    
                    if '[' in line:
                        temp_step = []
                        temp = line.split('[')[1]
                        temp_split = temp.split(' ')[1:]
                        temp_split = [item for item in temp_split if item != '']
                        for value in temp_split:
                            temp_step.append(float(value))
                    elif ']' in line:
                        temp = line.split(']')[0]
                        temp_split = temp.split(' ')[1:]
                        temp_split = [item for item in temp_split if item != '']
                        for value in temp_split:
                            temp_step.append(float(value))
                        data.append(temp_step)
                    else:
                        temp_split = temp.split(' ')[1:]
                        temp_split = [item for item in temp_split if item != '']
                        for value in temp_split:
                            temp_step.append(float(value))
            crop = []
            for i, window in enumerate(data):
                crop.append(window[-1])
            
            plt.rcParams.update({'font.size': 20})
            plt.figure(figsize=(200,20))
            plt.plot(crop, marker='o', markersize=6, markerfacecolor='red', markeredgecolor='red')
            plt.title("crop")
            plt.show()


def zero1normal(inputseries):
    value_max = max(inputseries)
    value_min = min(inputseries)
    value_v = value_max - value_min
    
    return [(value - value_min)/value_v for value in inputseries]


def cube(ax, point1, point2, color):
    
    ax.plot_surface(np.array([[point1[0], point2[0]], [point1[0], point2[0]]]), 
                    np.array([[point1[1], point1[1]], [point2[1], point2[1]]]),
                    np.array([[point1[2], point1[2]], [point1[2], point1[2]]]),
                    alpha=0.2, color=color)
    
    ax.plot_surface(np.array([[point1[0], point2[0]], [point1[0], point2[0]]]), 
                    np.array([[point1[1], point1[1]], [point2[1], point2[1]]]),
                    np.array([[point2[2], point2[2]], [point2[2], point2[2]]]),
                    alpha=0.2, color=color)
    
    ax.plot_surface(np.array([[point1[0], point1[0]], [point1[0], point1[0]]]), 
                    np.array([[point1[1], point1[1]], [point2[1], point2[1]]]),
                    np.array([[point2[2], point1[2]], [point2[2], point1[2]]]),
                    alpha=0.2, color=color)
    
    ax.plot_surface(np.array([[point2[0], point2[0]], [point2[0], point2[0]]]), 
                    np.array([[point1[1], point1[1]], [point2[1], point2[1]]]),
                    np.array([[point2[2], point1[2]], [point2[2], point1[2]]]),
                    alpha=0.2, color=color)
    
    ax.plot_surface(np.array([[point2[0], point2[0]], [point1[0], point1[0]]]), 
                    np.array([[point1[1], point1[1]], [point1[1], point1[1]]]),
                    np.array([[point2[2], point1[2]], [point2[2], point1[2]]]),
                    alpha=0.2, color=color)
    
    ax.plot_surface(np.array([[point2[0], point2[0]], [point1[0], point1[0]]]), 
                    np.array([[point2[1], point2[1]], [point2[1], point2[1]]]),
                    np.array([[point2[2], point1[2]], [point2[2], point1[2]]]),
                    alpha=0.2, color=color)


def ppg_loader4deap(file_name):
    base_dir = "D:/home/BCML/IITP/data/DEAP/clear_emotion/"
    ppg = pd.read_csv(base_dir + "ppg/" + file_name, index_col=0)
    label = pd.read_csv(base_dir + "label/" + file_name, index_col=0)
    
    # The sampling rate of DEAP dataset is 128
    
    return ppg, label, 128


def clear_emotion():
    save_dir = "D:/home/BCML/IITP/data/DEAP/clear_emotion/"
    rppg_folder = save_dir + "rppg/"
    ppg_folder = save_dir + "ppg/"
    label_folder = save_dir + "label/"
    
    dataset_path = "D:/home/BCML/IITP/data/DEAP/data_preprocessed_python/"
    subject_list = os.listdir(dataset_path)
    
    total_label = {}
    total_ppg = {}
    
    total_arousal = []
    total_valence = []
    total_dominacne = []
    
    clear_label = []
    clear_ppg = []
    clear_label_original = []
    
    # happy = 0, calm = 1, anger = 2, sadness = 3
    
    criterion_small = 3.0
    criterion_big = 5.0
    
    for subject_no in subject_list:
        deap_dataset = pickle.load(open(dataset_path + subject_no, 'rb'), encoding='latin1')
        total_ppg[subject_no] = deap_dataset["data"]
        total_label[subject_no] = deap_dataset["labels"]
        
        for i in range(40):
            total_arousal.append(deap_dataset["labels"][i, 1])
            total_valence.append(deap_dataset["labels"][i, 0])
            total_dominacne.append(deap_dataset["labels"][i, 2])
            
            # happy
            if deap_dataset["labels"][i, 1] >= criterion_big and deap_dataset["labels"][i, 0] >= criterion_big:
                clear_label.append(0)
                clear_ppg.append(deap_dataset["data"][i, 38, :])
                clear_label_original.append(deap_dataset["labels"][i, :])
            
            # calm
            if deap_dataset["labels"][i, 1] <= criterion_small and deap_dataset["labels"][i, 0] >= criterion_big:
                clear_label.append(1)
                clear_ppg.append(deap_dataset["data"][i, 38, :])
                clear_label_original.append(deap_dataset["labels"][i, :])
                
            # anger
            if deap_dataset["labels"][i, 1] >= criterion_big and deap_dataset["labels"][i, 0] <= criterion_small:
                clear_label.append(2)
                clear_ppg.append(deap_dataset["data"][i, 38, :])
                clear_label_original.append(deap_dataset["labels"][i, :])
                
            # sadness
            if deap_dataset["labels"][i, 1] <= criterion_small and deap_dataset["labels"][i, 0] <= criterion_small:
                clear_label.append(3)
                clear_ppg.append(deap_dataset["data"][i, 38, :])
                clear_label_original.append(deap_dataset["labels"][i, :])
                    
    print("clear_label lenght: ", len(clear_label))
    print("clear_ppg length: ", len(clear_ppg))
    
    clear_label_df = pd.DataFrame(clear_label)
    clear_ppg_df = pd.DataFrame(clear_ppg)
    clear_label_original_df = pd.DataFrame(clear_label_original)
    
    clear_label_df.to_csv(label_folder + f"{int(criterion_small*10)}_{int(criterion_big*10)}.csv")
    clear_ppg_df.to_csv(ppg_folder + f"{int(criterion_small*10)}_{int(criterion_big*10)}.csv")
    clear_label_original_df.to_csv(label_folder + f"{int(criterion_small*10)}_{int(criterion_big*10)}_original.csv")
    
    plt.rcParams['font.size'] = 20

    fig = plt.figure(figsize=(30,30))
    ax = fig.add_subplot(111, projection='3d')
    
    colors = np.zeros((len(total_arousal), 3))
    colors[:, 0] = zero1normal(total_arousal)
    colors[:, 1] = zero1normal(total_dominacne)
    colors[:, 2] = zero1normal(total_valence)
    
    norm_x = Normalize(vmin=min(total_arousal), vmax=max(total_arousal))
    norm_y = Normalize(vmin=min(total_dominacne), vmax=max(total_dominacne))
    norm_z = Normalize(vmin=min(total_valence), vmax=max(total_valence))
    
    cmap_x = plt.cm.Reds
    cmap_y = plt.cm.Blues
    cmap_z = plt.cm.Greens
    
    axins_x = inset_axes(ax, width="5%", height="50%", loc='upper right', bbox_to_anchor=(0.60, 0.4, 0.3, 0.5), bbox_transform=ax.transAxes, borderpad=0)
    axins_y = inset_axes(ax, width="5%", height="50%", loc='upper right', bbox_to_anchor=(0.65, 0.4, 0.3, 0.5), bbox_transform=ax.transAxes, borderpad=0)
    axins_z = inset_axes(ax, width="5%", height="50%", loc='upper right', bbox_to_anchor=(0.70, 0.4, 0.3, 0.5), bbox_transform=ax.transAxes, borderpad=0)
    
    cbar_x = fig.colorbar(plt.cm.ScalarMappable(norm=norm_x, cmap=cmap_x), ax=ax, orientation='vertical', fraction=0.02, pad=0.1, cax=axins_x)
    cbar_y = fig.colorbar(plt.cm.ScalarMappable(norm=norm_y, cmap=cmap_y), ax=ax, orientation='vertical', fraction=0.02, pad=0.1, cax=axins_y)
    cbar_z = fig.colorbar(plt.cm.ScalarMappable(norm=norm_z, cmap=cmap_z), ax=ax, orientation='vertical', fraction=0.02, pad=0.1, cax=axins_z)
    
    cbar_x.set_label('Arousal (X)')
    cbar_y.set_label('Dominance (Y)')
    cbar_z.set_label('Valence (Z)')
    
    scatter = ax.scatter(total_arousal, total_dominacne, total_valence, s=50, c=colors, alpha=0.6)
    
    cube(ax, [0,0,6], [2,3,8], 'g')
    cube(ax, [6,0,6], [8,3,8], 'r')
    cube(ax, [0,0,0], [2,3,2], 'b')
    cube(ax, [6,0,0], [8,3,2], 'm')
    
    ax.view_init(elev=0, azim=45)
    
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_zlim(0, 10)
    
    ax.set_title("DEAP dataset label distribution")
    ax.set_xlabel("Arousal (X)")
    ax.set_ylabel("Dominance (Y)")
    ax.set_zlabel("Valence (Z)")
    
    plt.show()
    
    
if __name__ == "__main__":
    clear_emotion()
    
    