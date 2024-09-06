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
import sys

import multiprocessing
from multiprocessing import Process, Queue, Value

import warnings

import time
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
        self.frame_list = sorted(os.listdir(self.frame_dir))
        print("frame_list: ", self.frame_list)
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
        if self.frame_count < self.frame_length - self.start_index:
            try:
                output = np.load(self.frame_dir+self.frame_list[self.frame_count + self.start_index])
                return output
            except ValueError:
                return None
        else:
            return "EoD"


# multiprocessing loader

def frame_q_loader(frame_queue, frame_dir, start_index):
    try:
        frame_list = os.listdir(frame_dir)
        input_loader = frame_loader(frame_dir, start_index=start_index)
        dataset_type = True
        print("input is frames in folder")
    except:
        print("input is video")
        input_loader = video_loader(frame_dir)
        dataset_type = False

    fps = input_loader.get_fps()
    height, width = input_loader.get_frame_shape()
    total_frame = input_loader.get_length()
    
    processed_frames_count = 0 + start_index

    print(f"input_dir: {frame_dir}")
    print(f"input information: fps: {fps}, h: {height}, w: {width}, total: {total_frame}")
    
    while True:
        if processed_frames_count % 300 == 0:
            print(f"{round(processed_frames_count / total_frame * 100, 2)} % {processed_frames_count}")
        
        image = input_loader.get_frame()
        if not isinstance(image, np.ndarray):
            print(f"frame loading is finished")
            if image == "EoD":
                break
            elif image == None:
                # if save_flag:
                #     with open(ldmk_save_dir, 'a') as f:
                #         f.write(f"noframe noframe\n{processed_frames_count}")
                continue
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        frame_queue.put(image)
        processed_frames_count += 1
        

def  polygon_model_pipeline(ldmk_list, frame_queue, object_state, cropped_skin_verbose=False,
                            segment_verbose=False, rejection_verbose=False, frame_start_idx=0,
                            ldmk_save_dir=None, kalman_type='stationary',
                            save_flag=False, rppg_save=True,
                            RGB_LOW_TH=55, RGB_HIGH_TH=200,
                            rgb_window=300, rppg_dir=None,
                            width=640, height=480,
                            fps=30, total=0,
                            polygon_keys=[]):
    
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
    
    skin_ex = SkinExtractionConvexHull('CPU')
    polygon_ex = SkinExtractionConvexHull_Polygon('CPU')
    
    processed_frames_count = 0 + frame_start_idx
    
    coord_row = len(ldmk_list)
    coord_col = 2
    # coord_ch = len(frame_list)
    coord_arr = np.zeros((coord_row, coord_col))
    # print("coordinate array shape: ", coord_arr.shape)
    
    coord_list = []
    
    rgb_signal = []
    
    outputs = {}
    algorithms = ["lgi", "chrom", "pos", "omit"]

    # outputs["rppg_window"] = {}
    outputs["rppg_save"] = {}
    outputs["bpm_save"] = {}
    outputs["snr_pass"] = {}
    for algorithm in algorithms:
        outputs["rppg_save"][algorithm] = []
        # outputs["rppg_window"][algorithm] = []
        outputs["bpm_save"][algorithm] = []
        outputs["snr_pass"][algorithm] = []
    
    
    bvp_proceed = bvps()
    bpm_proceed = BPM()

    old_ldmk = []

    while True:
        try:
            image = frame_queue.get(timeout=5)
        except queue.Empty:
            for algorithm in algorithms:
                outputs["rppg_save"][algorithm] + filtered_temp[:, 0].tolist()
                outputs["bpm_save"][algorithm] += [0 for _ in range(
                    len(outputs["rppg_save"][algorithm]) - len(outputs["bpm_save"][algorithm]))]
                rppg_df = pd.DataFrame(outputs["rppg_save"][algorithm])
                bpm_df = pd.DataFrame(outputs["bpm_save"][algorithm])
                snr_df = pd.DataFrame(outputs["snr_pass"][algorithm])
                save_csv = pd.concat([rppg_df, bpm_df, snr_df], axis=1)
                save_csv.to_csv(rppg_dir + algorithm + '.csv')
            break

        if processed_frames_count % 300 == 0:
            print(f"rPPG processing: {round(processed_frames_count/total*100, 2)}%")

        if not isinstance(image, np.ndarray):
            if image == "EoD":
                print("finished ")
                break
            elif image == None:
                if save_flag:
                    with open(ldmk_save_dir, 'a') as f:
                        f.write(f"noframe noframe\n{processed_frames_count}")
                continue
        
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        results = face_mesh.process(image)
        cropped_skin_im = np.zeros((3,3,3))

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
                    cropped_skin_im, full_skin_im, rminv, cminv = skin_ex.extract_skin(image, landmark_coords)
                    # print("landmark_coords:\n", landmark_coords)
                old_ldmk = landmark_coords
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
            cropped_skin_im, _, rminv, cminv = skin_ex.extract_skin(image, old_ldmk)
            if save_flag:
                with open(ldmk_save_dir, 'a') as f:
                    f.write(f"None None\n{processed_frames_count}")

        if cropped_skin_verbose:
            if processed_frames_count % 30 == 0:
                x_sc = landmark_coords[:, 1] - min(landmark_coords[:, 1])
                y_sc = landmark_coords[:, 0] - min(landmark_coords[:, 0])
                plt.imshow(cropped_skin_im)
                plt.title(f"{cropped_skin_im.shape}")
                # if len(outputs["bpm_save"]["omit"]) >= 1:
                #     plt.title(f"bpm: {outputs['bpm_save']['omit'][-1]}")
                plt.scatter(x_sc, y_sc, s=1, c='r')  # drawing landmarks on raw frame
                plt.show()

        # polygon segmentation

        polygon_dic = faceldmk_utils.face_trimesh

        total_x = landmark_coords[:, 1] - cminv
        total_y = landmark_coords[:, 0] - rminv

        for key in polygon_keys:
            vertices = polygon_dic[key]
            x = [landmark_coords[vertex, 1] for vertex in vertices] - cminv
            y = [landmark_coords[vertex, 0] for vertex in vertices] - rminv
            polygon_coords = np.array([(y[i], x[i]) for i in range(3)])
            polygon_coords = polygon_coords.astype(np.float32)
            # print("polygon_coords: \n", polygon_coords)
            cropped_polygon = polygon_ex.extract_polygon(cropped_skin_im, polygon_coords)
            
            if segment_verbose:
                plt.imshow(cropped_skin_im)
                plt.scatter(total_x, total_y, s=5)
                plt.scatter(x, y, c='r', s=15)
                plt.title(f"cropped polygon {key} {cropped_polygon.shape}")
                plt.draw()
                plt.pause(0.3)
                plt.clf()

        plt.close()


        # rppg extraction
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
                    outputs["bpm_save"][algorithm].append(bpm)
                    outputs["rppg_save"][algorithm].append(float(filtered_temp[0, 0]))
                    if snr > 0.045:
                        outputs["snr_pass"][algorithm].append(1)
                    else:
                        outputs["snr_pass"][algorithm].append(0)
                rgb_signal.pop(0)


            
        processed_frames_count += 1
    print("All process is finished")
    
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
    
    good_keys = []
    
    for key in faceldmk_utils.face_trimesh.keys():
        if set(faceldmk_utils.face_trimesh[key]).issubset(set(faceldmk_utils.left_eye)):
            pass
        elif set(faceldmk_utils.face_trimesh[key]).issubset(set(faceldmk_utils.right_eye)):
            pass
        elif set(faceldmk_utils.face_trimesh[key]).issubset(set(faceldmk_utils.mounth)):
            pass
        else:
             good_keys.append(key)
    
    multiprocessing.set_start_method('spawn', force=True)
    
    frame_q = Queue()

    frame_start = 1000
    folder_start = 0
    kalman_type = 'stationary'
    save_flag = False
    rppg_save = True
    RGB_LOW_TH = np.int32(55)
    RGB_HIGH_TH = np.int32(200)
    rgb_window = 300
    landmark_list = [i for i in range(468)]
    coord_save_dir = "D:/home/BCML/drax/PAPER/data/coordinates/"
    
    base_video_dir = "D:/home/BCML/drax/PAPER/data/frames/"
    frame_folder_list = os.listdir(base_video_dir)

    sel_folder = frame_folder_list[1]  # 0,

    sel_dir = base_video_dir + sel_folder + '/'
    total_length = len(os.listdir(sel_dir))

    rppg_dir = "D:/home/BCML/drax/PAPER/data/results/rppg/" + sel_folder + '/'

    if sel_folder not in os.listdir("D:/home/BCML/drax/PAPER/data/results/rppg/"):
        os.mkdir(rppg_dir)

    p2 = Process(target=frame_q_loader, args=(frame_q, sel_dir, frame_start))
    p2.start()
    
    polygon_model_pipeline(ldmk_list=landmark_list, frame_queue=frame_q,
                           object_state="resting", cropped_skin_verbose=True,
                           segment_verbose=True, rejection_verbose=False, frame_start_idx=frame_start,
                           ldmk_save_dir=coord_save_dir, kalman_type=kalman_type,
                           save_flag=save_flag, rppg_save=rppg_save,
                           RGB_LOW_TH=RGB_LOW_TH, 
                           RGB_HIGH_TH=RGB_HIGH_TH,
                           rgb_window=rgb_window,
                           rppg_dir=rppg_dir, total=total_length,
                           polygon_keys=good_keys)
    
    p2.join()
    
    # base_save_dir = "D:/home/BCML/drax/data/illuminance_2/"
    # landmark_list = [i for i in range(468)]
    # # input_dir = "D:/home/BCML/IITP/data/videos/multimodal_pilottest_2.mp4"
    # base_dir = "D:/home/BCML/drax/data/illuminance_2/video/"
    # folder_list = os.listdir(base_dir)
    # folder_list = folder_list[folder_start:]
    # for name in folder_list:
        
    #     video_list = [name for name in os.listdir(base_dir+name) if '.mov' in name]
        
    #     if folder_list.index(name) > folder_start:
    #         frame_start = 0
    #     split_name = name.split('_')
    #     save_folder = name[:-4]
    #     coord_save_dir = base_save_dir + save_folder + '.txt'
        
    #     for video in video_list:
    #         input_dir = base_dir + name + '/' + video
    #         rppg_dir = base_save_dir + "results/" + name + "_snr/"
    #         if name+"_snr" not in os.listdir(base_save_dir + "results/"):
    #             os.mkdir(rppg_dir)
    #         rppg_dir = rppg_dir + video.split('.')[0] + "_"
    #         polygon_model_pipeline(ldmk_list=landmark_list, frame_queue=frame_q,
    #                             object_state="resting", cropped_skin_verbose=False,
    #                             segment_verbose=False, rejection_verbose=False, frame_start_idx=frame_start,
    #                             ldmk_save_dir=coord_save_dir, kalman_type=kalman_type,
    #                             save_flag=save_flag, rppg_save=rppg_save,
    #                             RGB_LOW_TH=RGB_LOW_TH, 
    #                             RGB_HIGH_TH=RGB_HIGH_TH,
    #                             rgb_window=rgb_window,
    #                             rppg_dir=rppg_dir)
    
    
    