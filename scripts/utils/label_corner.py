import cv2, imutils
import numpy as np
import math

import time
import random
import os
import io
import glob

colors = [(255, 255, 255), (0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0)]

chess_piece_height = {"king": (0.081, 0.097), "queen": (0.07, 0.0762), "bishop": (0.058, 0.065), "knight": (0.054, 0.05715), "rook": (0.02845, 0.048), "pawn": (0.043, 0.045)}
chess_piece_diameter = {"king": (0.028, 0.0381), "queen": (0.028, 0.0362), "bishop": (0.026, 0.032), "knight": (0.026, 0.03255), "rook": (0.026, 0.03255), "pawn": (0.0191, 0.02825)}
# scan_box_height = max(chess_piece_height['king'])
scan_box_height = min(chess_piece_height['king'])

# cameraMatrix = np.array([[1438.4337197221366, 0.0, 934.4226787746103], [0.0, 1437.7513778197347, 557.7771398018671], [0.0, 0.0, 1.0]], np.float32) # Module
cameraMatrix = np.array([[1395.3709390074625, 0.0, 984.6248356317226], [0.0, 1396.2122002126725, 534.9517311724618], [0.0, 0.0, 1.0]], np.float32) # Humanoid
# cameraMatrix = np.array([[852.6434105992806, 0.0, 398.3286136737032], [0.0, 860.8765484709088, 302.00038413294385], [0.0, 0.0, 1.0]], np.float32) # ESP32
# cameraMatrix = np.array([[615.871040, 0.000000, 511.500000], [0.000000, 615.871040, 299.500000 ], [0.0, 0.0, 1.0]], np.float32) # Blender
# dist = np.array([[0.07229278436610362, -0.5836205675336522, 0.0003932499370206642, 0.0002754754987376089, 1.7293977700105942]]) # Module
dist = np.array([[0.1097213194870457, -0.1989645299789654, -0.002106454674127449, 0.004428959364733587, 0.06865838341764481]]) # Humanoid
# dist = np.array([[0.02220329099612066, 0.13530759611493004, -0.0041870520396677805, 0.007599954530058233, -0.4722284261198788]]) # ESP32
rpm = 2
fps = 30
frame_limit = fps/rpm*60
_save = True
export_size = 256
log = [] # [[frame0, angle, rvec0, tvec0, inner_point0], [frame1, angle1]]
current_rvec, start_rvec, start_tvec = None, None, None
current_angle = 0
# piece_symbols = "rnbqkpRNBQKP"
piece_symbols = "rnbqkp_"

if __name__ == '__main__':
    data_save = []
    label_save = []
    angle_save = []
    global_image_counter = 0
    FEN_name = "FEN.txt"
    FEN_label = ""
    FEN_label_extracted = []
    folders_path = "/mnt/HDD4TB/datasets/module89/V4_vdo+labels/"
    for folder_path_number in range(24):
        folder_path = os.path.join(folders_path, str(folder_path_number))
        FEN_path = os.path.join(folder_path, FEN_name)
        xml_list = []
        vdo_list = glob.glob(folder_path + '/*.avi')
        with open(FEN_path) as f:
            FEN_label = f.readline()
        FEN_buff = FEN_label.split('/')
        FEN_buff[-1] = FEN_buff[-1][:-1]  # remove \n
        for line in FEN_buff:
            line_extracted = []
            for letter in line:
                if letter.isalpha():
                    line_extracted.append(letter)
                else:
                    for i in range(int(letter)): line_extracted.append('_')
            FEN_label_extracted.append(line_extracted)
        print(FEN_label_extracted)
        for file_path in vdo_list:
            file_name = file_path[-6:-4]
            if not file_name[0].isnumeric(): file_name = file_name[1:]
            # print(file_name)
            if "0-" in file_path: continue

            save_path = file_path[:-4]
            if os.path.exists(save_path + ".npy"): continue # Skip labeled VDO

            four_points = []
            for i in range(4):  # for each corner
                cap = cv2.VideoCapture(file_path)
                last_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
                points = [] # Reset buffer
                def click_corner(event, x, y, flags, param):
                    global img, four_points, canvas
                    if event == cv2.EVENT_LBUTTONDOWN:
                        cv2.circle(canvas, (x, y), 5, (255, 0, 0), -1)
                        points.append((x, y))
                    if event == cv2.EVENT_MOUSEMOVE:
                        canvas = img.copy()
                        cv2.line(canvas, (x, 0), (x, 1080), (0, 255, 0), 1)
                        cv2.line(canvas, (0, y), (1920, y), (0, 255, 0), 1)
                        # for (ax, ay) in points: cv2.circle(canvas, (ax, ay), 5, (255, 0, 0), -1)
                for frame_id in range(last_frame):
                    img = cap.read()[1]
                    canvas = img.copy()
                    cv2.namedWindow('Assign Corner', cv2.WND_PROP_FULLSCREEN)
                    cv2.setWindowProperty('Assign Corner', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    cv2.setMouseCallback('Assign Corner', click_corner)
                    while True:  # Loop until 4 corners assigned
                        cv2.imshow('Assign Corner', canvas)
                        k = cv2.waitKey(1) & 0xFF
                        if len(points) > frame_id: break    # Already click
                four_points.append(points)
            np.save(save_path ,np.array(four_points))
