import imutils
import onnxruntime as ort
import os
import numpy as np
import cv2
import simplejson
import time

config = simplejson.load(open(os.path.join('../config/camera_config.json')))
# cameraMatrix = np.array(config['camera_matrix'], np.float32)
cameraMatrix = np.array([[1395.3709390074625, 0.0, 984.6248356317226], [0.0, 1396.2122002126725, 534.9517311724618], [0.0, 0.0, 1.0]], np.float32)
# dist = np.array(config['dist'])
dist = np.array([[0.1097213194870457, -0.1989645299789654, -0.002106454674127449, 0.004428959364733587, 0.06865838341764481]], np.float32)
based_obj_points = np.array([[-0.2, 0.23, 0], [-0.2, -0.23, 0], [0.2, 0.23, 0], [0.2, -0.23, 0]])

def FindMax(maps):
    # max_points = []
    # max_vals = []
    # for map in maps:
    #     local_max_mask = NonMaximaSuppression(map)
    #     local_max_indices = np.where(local_max_mask == [255])
    #     local_max_indices = np.array(local_max_indices).transpose()
    #     vals = []
    #     for i in range(len(local_max_indices)):
    #         vals.append((local_max_indices[i][1], local_max_indices[i][0]))
    #         max_points.append((local_max_indices[i][1], local_max_indices[i][0]))
    #     # max_val = max(vals)
    #     # max_point = local_max_indices[vals.index(max_val)]
    #     # max_points.append((max_point[1], max_point[0]))
    #     # max_vals.append(max_val)
    # return max_points, max_vals
    max_points, max_vals = [], []
    for m in maps:
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(cv2.GaussianBlur(m, (3, 3), 0))
        max_points.append(max_loc)
        max_vals.append(max_val)
    return max_points, max_vals
def NonMaximaSuppression(map): # Non-maxima suppression
    # Find pixel that are equal to the local neighborhood not maximum (including 'plateaus')
    map = map - cv2.GaussianBlur(map, (0, 0), sigmaX=3) + 127   # Highpass filter
    maxima = cv2.dilate(map, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=20)
    mask = cv2.compare(map, maxima, cv2.CMP_GE)
    # cv2.imshow("NonMaxSuppression", imutils.resize(canvas, height=480))
    return mask

# cap = cv2.VideoCapture("/mnt/HDD/dataset/module89/V4_vdo+labels/20/62.avi")
cap = cv2.VideoCapture(0)
# cap.set(3, 640)
# cap.set(4, 480)
cap.set(3, 1920)
cap.set(4, 1080)
cap.set(cv2.CAP_PROP_FPS, 30.0)
# cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
cap.set(cv2.CAP_PROP_AUTO_WB, 0)
cap.set(cv2.CAP_PROP_FOCUS, 2)
# vdo_length = int(cap. get(cv2. CAP_PROP_FRAME_COUNT))
model_path = '/media/teera/SSD250GB/model/belief/chessboard_mono_6_stage/net_epoch_51.onnx'
ort_sess = ort.InferenceSession(model_path, providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider'])  # TensorrtExecutionProvider having the higher priority.

# for f in range(vdo_length):
while True:
    stamp = time.time()
    image = cap.read()[1]
    # convert from NHWC to NCHW (batch N, channels C, height H, width W)
    image_480p = imutils.resize(image, height=480)[:, 106:106 + 640]
    x = image_480p.transpose([2, 0, 1])   # ([0, 3, 1, 2])
    x = np.float32(x)
    x /= 255
    x = np.expand_dims(x, 0)
    outputs = ort_sess.run(None, {'input': x})  # outputs.shape = (1, 4, 60, 80)
    outputs = outputs[0][0]
    overlay = np.zeros(outputs[0].shape, dtype=np.float32)
    for i in range(4): overlay += outputs[i]
    overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)
    overlay = np.array(overlay*255, dtype=np.uint8)
    overlay = imutils.resize(overlay, height=image_480p.shape[0])
    canvas = cv2.addWeighted(image_480p, 0.3, overlay, 0.7, 0)
    cv2.imshow("Belief", canvas)

    canvas = image_480p.copy()
    # points, vals = FindMax(outputs[0][0])
    # for i in range(len(points)): cv2.circle(canvas, (points[i][0]*8, points[i][1]*8), 3, (255, 0, 0), -1)
    points, vals = FindMax(outputs)
    # print(vals)
    confidences = [False if val < 0.03 else True for val in vals]
    for i in range(4):
        cv2.circle(canvas, (int(points[i][0] * 8), int(points[i][1] * 8)), 3, (255, 0, 0), -1)
        # if confidences[i] is True:
        #     cv2.circle(canvas, (points[i][0]*8, points[i][1]*8), 3, (255, 0, 0), -1)
    cv2.imshow("Detected points", canvas)


    if not False in confidences:
        canvas = image.copy()
        obj_points = np.array([[-0.2, 0.23, 0], [-0.2, -0.23, 0], [0.2, 0.23, 0], [0.2, -0.23, 0]])
        img_points = np.array([points[0], points[1], points[2], points[3]], dtype=np.double) * 8

        # map points 480p -> 1080p
        scale = 1080 / 480
        img_points = np.array([((point[0] + 106) * scale, point[1] * scale) for point in img_points], dtype=np.double)
        print(img_points)
        ret, rvec, tvec = cv2.solvePnP(objectPoints=obj_points,
                                       imagePoints=img_points,
                                       cameraMatrix=cameraMatrix,
                                       distCoeffs=dist,
                                       flags=0)
        cv2.aruco.drawAxis(image=canvas,
                           cameraMatrix=cameraMatrix,
                           distCoeffs=dist,
                           rvec=rvec,
                           tvec=tvec,
                           length=0.1)
        cv2.imshow('Estimated Pose', canvas)
    print("FPS: %d"%(int(1/(time.time() - stamp))))
    key = cv2.waitKey(1)
    if key == ord('q'): break

cv2.destroyAllWindows()