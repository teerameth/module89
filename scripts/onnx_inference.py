import onnxruntime as ort
import os
import numpy as np
import cv2
import simplejson

config = simplejson.load(open(os.path.join('../config/camera_config.json')))
cameraMatrix = np.array(config['camera_matrix'], np.float32)
dist = np.array(config['dist'])

def FindMax(maps):
    max_points = []
    for m in maps:
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(cv2.GaussianBlur(m, (3, 3), 0))
        max_points.append(max_loc)
    return max_points

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

while True:
    image = cap.read()[1]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    x = np.float32(image)
    x /= 255
    x = x.reshape((3, 480, 640))
    x = np.expand_dims(x, 0)
    # x, y = test_data[0][0], test_data[0][1]
    ort_sess = ort.InferenceSession('../models/chessboard.onnx', providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider'])
    outputs = ort_sess.run(None, {'input': x})
    points = FindMax(outputs[0][0])
    # print(outputs[0][0].shape)
    print(points)
    obj_points = np.array([[-0.2, 0.23, 0], [-0.2, -0.23, 0], [0.2, 0.23, 0], [0.2, -0.23, 0]])
    img_points = np.array([points[0], points[1], points[2], points[3]], dtype=np.double) * 8

    ret, rvec, tvec = cv2.solvePnP(objectPoints=obj_points, imagePoints=img_points, cameraMatrix=cameraMatrix,
                                   distCoeffs=dist, flags=0)
    canvas = image.copy()
    cv2.aruco.drawAxis(image=canvas, cameraMatrix=cameraMatrix, distCoeffs=dist, rvec=rvec, tvec=tvec, length=0.1)
    cv2.imshow('A', canvas)
    cv2.waitKey(1)

    # cv2.imshow("A", outputs)
    # cv2.waitKey(0)
# Print Result
# predicted, actual = classes[outputs[0][0].argmax(0)], classes[y]
# print(f'Predicted: "{predicted}", Actual: "{actual}"')



# cv2.destroyAllWindows()