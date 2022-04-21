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
cap.set(3, 640)
cap.set(4, 480)
cap.set(cv2.CAP_PROP_FPS, 30.0)
# cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
cap.set(cv2.CAP_PROP_AUTO_WB, 0)
cap.set(cv2.CAP_PROP_FOCUS, 2)
# vdo_length = int(cap. get(cv2. CAP_PROP_FRAME_COUNT))
model_path = '/media/teera/ROGESD/model/belief/chessboard_mono_6_stage/net_epoch_51.onnx'
ort_sess = ort.InferenceSession(model_path, providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider'])  # TensorrtExecutionProvider having the higher priority.

# for f in range(vdo_length):
while True:
    stamp = time.time()
    image = cap.read()[1]
    if image.shape == (1080, 1920): image = imutils.resize(image, height=480)[:, 106:106 + 640]
    # if image.shape != (480, 640, 3):
    #     if image.shape[0] / image.shape[1] > 480/640:   # Use height as reference
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
    outputs = ort_sess.run(None, {'input': x})  # outputs.shape = (1, 4, 60, 80)
    outputs = outputs[0][0]
    overlay = np.zeros(outputs[0].shape, dtype=np.float32)
    for i in range(4):
        overlay += outputs[0][0][i]
    overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)
    overlay = np.array(overlay*255, dtype=np.uint8)
    overlay = imutils.resize(overlay, height=image.shape[0])
    canvas = cv2.addWeighted(image, 0.3, overlay, 0.7, 0)
    cv2.imshow("Belief", canvas)

    canvas = image.copy()
    # points, vals = FindMax(outputs[0][0])
    # for i in range(len(points)): cv2.circle(canvas, (points[i][0]*8, points[i][1]*8), 3, (255, 0, 0), -1)
    points, vals = FindMax(outputs)
    # print(vals)
    confidences = [False if val < 0.05 else True for val in vals]
    for i in range(4):
        cv2.circle(canvas, (points[i][0] * 8, points[i][1] * 8), 3, (255, 0, 0), -1)
        # if confidences[i] is True:
        #     cv2.circle(canvas, (points[i][0]*8, points[i][1]*8), 3, (255, 0, 0), -1)

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
    print("FPS: %d"%(int(1/(time.time() - stamp))))
    cv2.imshow('A', canvas)
    key = cv2.waitKey(1)
    if key == ord('q'): break

cv2.destroyAllWindows()