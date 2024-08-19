# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 10:18:53 2024

@author: ys
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import mediapipe as mp
import cv2
from utils import *
from copy import deepcopy
import queue
import seaborn as sns

import multiprocessing
from multiprocessing import Process, Queue, Value

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")

def coord_preprocessing(coords_series):
    num_row, num_col = coords_series[0].shape
    
    x_series = []
    y_series = []
    
    for coord_frame in coords_series:
        temp_summation = np.sum(coord_frame, axis=0)
        x_series.append(temp_summation[0])
        y_series.append(temp_summation[1])
    return x_series, y_series

class video_loader:
    def __init__(self, video_dir):
        self.cap = cv2.VideoCapture(video_dir)
        self.video_length = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.video_shape = (self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
                            self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    
    def get_fps(self):
        return self.video_fps
    
    def get_frame_shape(self):
        return self.video_shape[0], self.video_shape[1]
    
    def get_length(self):
        return self.video_length
    
    def get_current_frame(self):
        return int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
    
    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return "EoD"
        else:
            return frame
    
class frame_loader:
    def __init__(self, frame_dir, start_index=0):
        
        self.frame_dir = frame_dir
        self.frame_list = os.listdir(self.frame_dir)
        self.frame_length = len(self.frame_list)
        self.frame_count = 0
        self.frame_shape = np.load(self.frame_dir+self.frame_list[0]).shape[:2]
        self.capture_fps = int(frame_dir.split('/')[-2].split('_')[-1].split('f')[0])
        self.start_index = start_index
    
    def get_fps(self):
        return self.capture_fps
    
    def get_frame_shape(self):
        return self.frame_shape[0], self.frame_shape[1]
    
    def get_length(self):
        return self.frame_length
    
    def get_current_frame(self):
        return self.frame_count
    
    def get_frame(self):
        self.frame_count += 1
        if self.frame_count < self.frame_length:
            try:
                output = np.load(self.frame_dir+self.frame_list[self.frame_count + self.start_index])
                return output
            except ValueError:
                return None
        else:
            return "EoD"


# multiprocessing loader
# add redundant annotation
# for git commit test
# yeah

def frame_q_loader(frame_dir, start_index):
    try:
        frame_list = os.listdir(input_dir)
        input_loader = frame_loader(input_dir, start_index=frame_start_idx)
        dataset_type = True
        print("input is frames in folder")
    except:
        print("input is video")
        input_loader = video_loader(input_dir)
        dataset_type = False

    fps = input_loader.get_fps()
    height, width = input_loader.get_frame_shape()
    total_frame = input_loader.get_length()
    
    processed_frames_count = 0 + frame_start_idx

    print(f"input_dir: {input_dir}")
    print(f"input information: fps: {fps}, h: {height}, w: {width}, total: {total_frame}")
    
    while True:
        if processed_frames_count % 300 == 0:
            print(f"{ldmk_save_dir}\n{round(processed_frames_count / total_frame * 100, 2)} % {processed_frames_count}")
        
        image = input_loader.get_frame()
        if not isinstance(image, np.ndarray):
            print(f"{ldmk_save_dir} finished")
            if image == "EoD":
                break
            elif image == None:
                if save_flag:
                    with open(ldmk_save_dir, 'a') as f:
                        f.write(f"noframe noframe\n{processed_frames_count}")
                continue
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        

def  polygon_model_pipeline(ldmk_list, input_dir, object_state, cropped_skin_verbose=False,
                            segment_verbose=False, rejection_verbose=False, frame_start_idx=0,
                            ldmk_save_dir=None, kalman_type='stationary',
                            save_flag=False, rppg_save=True,
                            RGB_LOW_TH=55, RGB_HIGH_TH=200,
                            rgb_window=300, rppg_dir=None):
    
    """
    
        ldmk_list: landmarks list to extract
        input_dir: input frame folder(for one video) or input video dir
        object_state: "resting" or "dynamic" argument for kalman filter type
        cropped_skin_verbose: Show cropped skin image. Dafault is False
        
        futher implementation:
            2024.07.24.
            The skin extraction process is redundant in this pipeline.
            It can be combined to one process; polygon extraction process.
            So, replace skin extration + polygon extraction set with polygon
            extraction process.
    """
    
    # === skin extraction === #
    
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh()
    mp_drawing = mp.solutions.drawing_utils
    
    
    # === input frame loader initialization === #
    
    try:
        frame_list = os.listdir(input_dir)
        input_loader = frame_loader(input_dir, start_index=frame_start_idx)
        dataset_type = True
        print("input is frames in folder")
    except:
        print("input is video")
        input_loader = video_loader(input_dir)
        dataset_type = False
    
    # === Legacy test codes === #
    
    # face_detector = os.getcwd()+"/face_detection_opencv/"
    # detector = cv2.dnn.readNetFromCaffe(f"{face_detector}/deploy.prototxt" , f"{face_detector}res10_300x300_ssd_iter_140000.caffemodel")
    
    
    fps = input_loader.get_fps()
    height, width = input_loader.get_frame_shape()
    total_frame = input_loader.get_length()
    
    skin_ex = SkinExtractionConvexHull('CPU')
    polygon_ex = SkinExtractionConvexHull_Polygon('CPU')
    
    processed_frames_count = 0 + frame_start_idx
    
    coord_row = len(ldmk_list)
    coord_col = 2
    # coord_ch = len(frame_list)
    coord_arr = np.zeros((coord_row, coord_col))
    # print("coordinate array shape: ", coord_arr.shape)
    
    coord_list = []
    
    print(f"input_dir: {input_dir}")
    print(f"input information: fps: {fps}, h: {height}, w: {width}, total: {total_frame}")
    
    rgb_signal = []
    
    outputs = {}
    algorithms = ["lgi", "chrom", "pos", "omit"]

    # outputs["rppg_window"] = {}
    outputs["rppg_save"] = {}
    outputs["bpm_save"] = {}
    for algorithm in algorithms:
        outputs["rppg_save"][algorithm] = []
        # outputs["rppg_window"][algorithm] = []
        outputs["bpm_save"][algorithm] = []
    
    
    bvp_proceed = bvps()
    bpm_proceed = BPM()
    
    while True:
        if processed_frames_count % 300 == 0:
            print(f"{ldmk_save_dir}\n{round(processed_frames_count / total_frame * 100, 2)} % {processed_frames_count}")
        
        image = input_loader.get_frame()
        if not isinstance(image, np.ndarray):
            print(f"{ldmk_save_dir} finished")
            if image == "EoD":
                break
            elif image == None:
                if save_flag:
                    with open(ldmk_save_dir, 'a') as f:
                        f.write(f"noframe noframe\n{processed_frames_count}")
                continue
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        results = face_mesh.process(image)
        

        # skin region extraction
        if results.multi_face_landmarks:
            try:
                for face_landmarks in results.multi_face_landmarks:
                    ldmk_counter = 0
                    landmark_coords = np.zeros((468, 2), dtype=np.float32)
                    coord_row_counter = 0
                    for lm in face_landmarks.landmark:
                        coords = mp_drawing._normalized_to_pixel_coordinates(lm.x, lm.y, width, height)
                        
                        # kalman filter
                        
                        landmark_coords[ldmk_counter, 0] = coords[1]
                        landmark_coords[ldmk_counter, 1] = coords[0]
                        
                        if ldmk_counter in ldmk_list:
                            coord_arr[coord_row_counter, :] = np.array(coords)
                            coord_row_counter += 1
                        ldmk_counter += 1
                    coord_list.append(coord_arr)
                    cropped_skin_im, full_skin_im = skin_ex.extract_skin(image, landmark_coords)
                    # print("landmark_coords: ", landmark_coords.shape)
                if save_flag:
                    with open(ldmk_save_dir, 'a') as f:
                        for i in range(landmark_coords.shape[0]):
                            f.write(f"{landmark_coords[i, 0]} {landmark_coords[i, 1]},")
                        f.write(f"\n{processed_frames_count}\n")
            except TypeError:
                if save_flag:
                    with open(ldmk_save_dir, 'a') as f:
                        f.write(f"None None\n{processed_frames_count}")
        else:
            print(f"A face is not detected {processed_frames_count}")
            cropped_skin_im = np.zeros_like(image)
            full_skin_im = np.zeros_like(image)
            if save_flag:
                with open(ldmk_save_dir, 'a') as f:
                    f.write(f"None None\n{processed_frames_count}")
                
        
        
        if cropped_skin_verbose:
            x_sc = landmark_coords[:, 1] - min(landmark_coords[:, 1])
            y_sc = landmark_coords[:, 0] - min(landmark_coords[:, 0])
            fig = plt.figure()
            plt.imshow(cropped_skin_im)
            plt.scatter(x_sc, y_sc, s=1, c='r')  # drawing landmarks on raw frame
            plt.show()
        
        # polygon segmentation
        """
        xy_sc = np.concatenate((y_sc.reshape(-1, 1), x_sc.reshape(-1, 1)), axis=1)
        print("xy_sc shape: ", xy_sc.shape)
        mesh_set = faceldmk_utils.face_trimesh
        mesh_set_dict = {}
        
        for key in mesh_set.keys():
            mesh_set_dict[key] = polygon_ex.extract_polygon(cropped_skin_im, xy_sc[mesh_set[key], :])
            
        if segment_verbose:
            plt.imshow(mesh_set_dict[key])
            plt.scatter(x_sc[mesh_set[key]], y_sc[mesh_set[key]], s=1, c='r')
            plt.show()
        """
        
        if rppg_save:
            rgb_signal.append(holistic_mean(cropped_skin_im, RGB_LOW_TH, RGB_HIGH_TH))
            
            if len(rgb_signal) >= rgb_window:
                input_signal = np.array(rgb_signal)
                input_signal = np.swapaxes(input_signal, 1, 2)
                filtered_input_signal = np.zeros_like(input_signal)
                for i in range(3):
                    filtered_input_signal[:, i, 0] = bpf(input_signal[:, i, 0])
                # print("input_signal shape: ", input_signal.shape)
                # input_signal = pre_filter(input_signal)
                # "lgi", "chrom", "pos", "omit", "ssr"
                
                for algorithm in algorithms:
                    # print("algorithm: ", algorithm)
                    temp = bvp_proceed.bvp(algorithm, filtered_input_signal)
                    # outputs["rppg_window"][algorithm] = temp
                    
                    filtered_temp = np.zeros_like(temp)
                    filtered_temp[:, 0] = bpf(temp[:, 0])

                    bpm, snr, psnr, pfreqs, power = bpm_proceed.BVP_to_BPM(filtered_temp[:, 0])
                    if snr > 0.045:
                        outputs["bpm_save"][algorithm].append(bpm)
                    else:
                        outputs["bpm_save"][algorithm].append(-1)
                    
                    if processed_frames_count + 1 == total_frame:
                        outputs["rppg_save"][algorithm]+filtered_temp[:, 0].tolist()
                        outputs["bpm_save"][algorithm] + [0 for _ in range(len(outputs["rppg_save"][algorithm])-len(outputs["bpm_save"][algorithm]))]
                        rppg_df = pd.DataFrame(outputs["rppg_save"][algorithm])
                        bpm_df = pd.DataFrame(outputs["bpm_save"][algorithm])
                        save_csv = pd.concat([rppg_df, bpm_df], axis=1)
                        save_csv.to_csv(rppg_dir+algorithm+'.csv')
                        
                    else:
                        outputs["rppg_save"][algorithm].append(float(filtered_temp[0, 0]))
                rgb_signal.pop(0)
            
        processed_frames_count += 1

# =============================================================================
# # ldmk_test
# 
# if __name__ == "__main__":
#     frame_start = 0
#     folder_start = 0
#     kalman_type = 'stationary'
#     
#     base_save_dir = "D:/home/BCML/drax/PAPER/data/coordinates/stationary_kalman/"
#     landmark_list = [i for i in range(468)]
#     # input_dir = "D:/home/BCML/IITP/data/videos/multimodal_pilottest_2.mp4"
#     base_dir = "D:/home/BCML/drax/PAPER/data/treadmill_dataset/frames/"
#     folder_list = os.listdir(base_dir)
#     folder_list = folder_list[folder_start:]
#     for name in folder_list:
#         if folder_list.index(name) > folder_start:
#             frame_start = 0
#         split_name = name.split('_')
#         save_folder = split_name[0] + "_" + split_name[1] 
#         coord_save_dir = base_save_dir + save_folder + '.txt'
#         
#         input_dir = base_dir + name + '/'
#         polygon_model_pipeline(ldmk_list=landmark_list, input_dir=input_dir,
#                             object_state="resting", cropped_skin_verbose=False,
#                             segment_verbose=False, rejection_verbose=False, frame_start_idx=frame_start,
#                             ldmk_save_dir=coord_save_dir, kalman_type=kalman_type)
# =============================================================================

if __name__ == "__main__":
    
    multiprocessing.set_start_method('spawn', force=True)
    
    frame_q = Queue()
    p2 = Process(target)
    
    frame_start = 0
    folder_start = 0
    kalman_type = 'stationary'
    save_flag = False
    rppg_save = True
    RGB_LOW_TH = np.int32(55)
    RGB_HIGH_TH = np.int32(200)
    rgb_window = 300
    
    base_save_dir = "D:/home/BCML/drax/data/illuminance_2/"
    landmark_list = [i for i in range(468)]
    # input_dir = "D:/home/BCML/IITP/data/videos/multimodal_pilottest_2.mp4"
    base_dir = "D:/home/BCML/drax/data/illuminance_2/video/"
    folder_list = os.listdir(base_dir)
    folder_list = folder_list[folder_start:]
    for name in folder_list:
        
        video_list = [name for name in os.listdir(base_dir+name) if '.mov' in name]
        
        if folder_list.index(name) > folder_start:
            frame_start = 0
        split_name = name.split('_')
        save_folder = name[:-4]
        coord_save_dir = base_save_dir + save_folder + '.txt'
        
        for video in video_list:
            input_dir = base_dir + name + '/' + video
            rppg_dir = base_save_dir + "results/" + name + "_snr/"
            if name+"_snr" not in os.listdir(base_save_dir + "results/"):
                os.mkdir(rppg_dir)
            rppg_dir = rppg_dir + video.split('.')[0] + "_"
            polygon_model_pipeline(ldmk_list=landmark_list, input_dir=input_dir,
                                object_state="resting", cropped_skin_verbose=False,
                                segment_verbose=False, rejection_verbose=False, frame_start_idx=frame_start,
                                ldmk_save_dir=coord_save_dir, kalman_type=kalman_type,
                                save_flag=save_flag, rppg_save=rppg_save,
                                RGB_LOW_TH=RGB_LOW_TH, 
                                RGB_HIGH_TH=RGB_HIGH_TH,
                                rgb_window=rgb_window,
                                rppg_dir=rppg_dir)
    
    
    