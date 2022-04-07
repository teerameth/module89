import imutils
import onnxruntime as ort
import os
import numpy as np
import cv2
import simplejson
import time

config = simplejson.load(open(os.path.join('../config/camera_config.json')))
cameraMatrix = np.array(config['camera_matrix'], np.float32)
dist = np.array(config['dist'])
based_obj_points = np.array([[-0.2, 0.23, 0], [-0.2, -0.23, 0], [0.2, 0.23, 0], [0.2, -0.23, 0], [0, 0, 0]])

def FindMax(maps):
    max_points, max_vals = [], []
    for m in maps:
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(cv2.GaussianBlur(m, (3, 3), 0))
        max_points.append(max_loc)
        max_vals.append(max_val)
    return max_points, max_vals

cap = cv2.VideoCapture("/mnt/HDD4TB/datasets/module89/V4_vdo+labels/20/62.avi")
# cap = cv2.VideoCapture(0)
cap.set(3, 1920)
cap.set(4, 1080)
vdo_length = int(cap. get(cv2. CAP_PROP_FRAME_COUNT))
ort_sess = ort.InferenceSession('../models/chessboard.onnx', providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider'])  # TensorrtExecutionProvider having the higher priority.

for f in range(vdo_length):
# while True:
    stamp = time.time()
    image = cap.read()[1]
    image = imutils.resize(image, height=480)[:, 106:106 + 640]
    # if image.shape != (480, 640, 3):
    #     if image.0..................................................shape[0] / image.shape[1] > 480/640:   # Use height as reference
    #         image = imutils.resize(image, height=480)
    #         delta = 640 - image.shape[1]
    #         image = cv2.copyMakeBorder(image, 0, 0, int(delta/2), int(delta/2), borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))
    #     else:   # Use width as reference
    #         image = imutils.resize(image, width=640)
    #         delta = 480 - image.shape[0]
    #         image = cv2.copyMakeBorder(image, int(delta/2), int(delta/2), 0, 0, borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))

    # convert from NHWC to NCHW (batch N, channels C, height H, width W)
    x = image.transpose([2, 0, 1])   # ([0, 3, 1, 2])
    x = np.float32(x)
    x /= 255
    x = np.expand_dims(x, 0)
    outputs = ort_sess.run(None, {'input': x})
    for i in range(5):
        cv2.imshow("Belief" + str(i), imutils.resize(outputs[0][0][i], height=480))

    canvas = image.copy()
    points, vals = FindMax(outputs[0][0])
    print(vals)
    confidences = [False if val < 0.05 else True for val in vals]
    for i in range(5):
        if confidences[i] is True:
            cv2.circle(canvas, (points[i][0]*8, points[i][1]*8), 3, (255, 0, 0), -1)
    # obj_points, img_points = [], []
    # for i in range(5):
    #     if confidences[i] is True:
    #         obj_points.append(based_obj_points[i])
    #         img_points.append(points[i])

    # if len(obj_points) >= 4:
    #     obj_points = np.array(obj_points)
    #     img_points = np.array(img_points, dtype=np.double) * 8
    #     ret, rvec, tvec = cv2.solvePnP(objectPoints=obj_points,
    #                                    imagePoints=img_points,
    #                                    cameraMatrix=cameraMatrix,
    #                                    distCoeffs=dist,
    #                                    flags=0)
    #     cv2.aruco.drawAxis(image=canvas,
    #                        cameraMatrix=cameraMatrix,
    #                        distCoeffs=dist,
    #                        rvec=rvec,
    #                        tvec=tvec,
    #                        length=0.1)
    if not False in confidences:
        obj_points = np.array([[-0.2, 0.23, 0], [-0.2, -0.23, 0], [0.2, 0.23, 0], [0.2, -0.23, 0]])
        img_points = np.array([points[0], points[1], points[2], points[3]], dtype=np.double) * 8

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
    # print(int(1/(time.time() - stamp)))
    cv2.imshow('A', canvas)
    key = cv2.waitKey(1)
    if key == ord('q'): break

cv2.destroyAllWindows()