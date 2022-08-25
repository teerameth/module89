#!/usr/bin/env -S HOME=${HOME} ${HOME}/openvino_env/bin/python
import os.path

import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout
from PyQt5.QtGui import QPixmap

import rclpy
from rclpy.node import Node
from std_msgs.msg import UInt8MultiArray, Float32
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import cv2, imutils
import math
import numpy as np
import random
import mediapipe as mp
from camera import Camera
from openvino.runtime import Core, Layout, Type
from sklearn.cluster import KMeans

# obj_points = np.array([[-0.2, -0.23, 0], [0.2, -0.23, 0], [0.2, 0.23, 0], [-0.2, 0.23, 0]])
obj_points = np.array([[-0.2, -0.2, 0], [0.2, -0.2, 0], [0.2, 0.2, 0], [-0.2, 0.2, 0]])
# C922
# cameraMatrix = np.array([[1395.3709390074625, 0.0, 984.6248356317226], [0.0, 1396.2122002126725, 534.9517311724618], [0.0, 0.0, 1.0]], np.float32) # Humanoid
# dist = np.array([[0.1097213194870457, -0.1989645299789654, -0.002106454674127449, 0.004428959364733587, 0.06865838341764481]]) # Humanoid
# C930e
cameraMatrix = np.array([[1176.3318391347427, 0.0, 933.6507321953259], [0.0, 1173.600391970806, 544.8869859448852], [0.0, 0.0, 1.0]], np.float32)
dist = np.array([[0.08051354728224885, -0.17305343907163906, 0.0006471377956487211, 0.0002086472663196551, 0.06817438846344685]])
export_size = 100
chess_piece_height = {"king": (0.081, 0.097), "queen": (0.07, 0.0762), "bishop": (0.058, 0.065), "knight": (0.054, 0.05715), "rook": (0.02845, 0.048), "pawn": (0.043, 0.045)}
chess_piece_diameter = {"king": (0.028, 0.0381), "queen": (0.028, 0.0362), "bishop": (0.026, 0.032), "knight": (0.026, 0.03255), "rook": (0.026, 0.03255), "pawn": (0.0191, 0.02825)}
mask_contour_index_list = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
scan_box_height = min(chess_piece_height['king'])   # reference height for cuboid zero-padding
model_path_binary = "/home/fiborobotlab/models/classification/chess/chess_ir"

ie = Core()
model_binary = ie.read_model(model=model_path_binary+".xml", weights=model_path_binary+".bin")
model_binary.reshape([64, 100, 100, 3]) # set input as fixed batch
compiled_model_binary = ie.compile_model(model_binary, "GPU", {"PERFORMANCE_HINT": "THROUGHPUT"})
infer_request_binary = compiled_model_binary.create_infer_request()

# Load mediapipe hand detection model
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)
def draw_axis(img, corners, tvec, rvec):
    axis = np.float32([[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]]).reshape(-1, 3)
    # project 3D points to image plane
    imgpts, jac = cv2.projectPoints(axis, rvec, tvec, cameraMatrix, dist)
    # print(corners)
    # corner = np.mean(corners, axis=0, dtype=np.int)
    # print(corner)
    # corner = (int(corners[0][0]), int(corners[0][1]))
    imgpts = np.array(imgpts, dtype=np.int32).reshape((4, 2))
    img = cv2.line(img, imgpts[0], imgpts[1], (0, 0, 255), 5)
    img = cv2.line(img, imgpts[0], imgpts[2], (0, 255, 0), 5)
    img = cv2.line(img, imgpts[0], imgpts[3], (255, 0, 0), 5)
    # img = cv2.line(img, corner, (int(imgpts[0][0][0]), int(imgpts[0][0][1])), (255,0,0), 5)
    # img = cv2.line(img, corner, (int(imgpts[1][0][0]), int(imgpts[1][0][1])), (0,255,0), 5)
    # img = cv2.line(img, corner, (int(imgpts[2][0][0]), int(imgpts[2][0][1])), (0,0,255), 5)
    return img
def order_points(pts):
    # sort the points based on their x-coordinates
    xSorted = pts[np.argsort(pts[:, 0]), :]
    # grab the left-most and right-most points from the sorted
    # x-roodinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]
    # now, sort the left-most coordinates according to their
    # y-coordinates so we can grab the top-left and bottom-left
    # points, respectively
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost
    # if use Euclidean distance, it will run in error when the object
    # is trapezoid. So we should use the same simple y-coordinates order method.
    # now, sort the right-most coordinates according to their
    # y-coordinates so we can grab the top-right and bottom-right
    # points, respectively
    rightMost = rightMost[np.argsort(rightMost[:, 1]), :]
    (tr, br) = rightMost
    # return the coordinates in top-left, top-right,
    # bottom-right, and bottom-left order
    return np.array([tl, tr, br, bl], dtype="float32")
def get_bars(rvec, tvec, image=None, debug=False):
    Bar_up = [np.array([-0.2, 0.23, 0.0]), np.array([0.2, 0.23, 0.0]), np.array([0.2, 0.2, 0.0]), np.array([-0.2, 0.2, 0.0])]
    Bar_down = [np.array([-0.2, -0.2, 0.0]), np.array([0.2, -0.2, 0.0]), np.array([0.2, -0.23, 0.0]), np.array([-0.2, -0.23, 0.0])]
    bar_list = []
    # for bar in [Bar_up, Bar_left, Bar_right, Bar_down]:
    for bar in [Bar_up, Bar_down]:
        objpts = np.float32(bar).reshape(-1, 3)
        imgpts, jac = cv2.projectPoints(objpts, rvec, tvec, cameraMatrix, dist)
        imgpts = [(int(x[0][0]), int(x[0][1])) for x in imgpts]
        bar_list.append(np.array(imgpts))
    if debug:
        canvas = image.copy()
        cv2.polylines(canvas, [bar_list[0]], isClosed=True, color=(0, 0, 0), thickness=2)
        cv2.polylines(canvas, [bar_list[1]], isClosed=True, color=(255, 255, 255), thickness=2)
        cv2.imshow("Bar", canvas)
        # cv2.imwrite("Bar.png", canvas)
        cv2.waitKey(1)
    return bar_list
def correct_90(img, rvec, tvec):
    rotM = np.zeros(shape=(3, 3))
    cv2.Rodrigues(rvec, rotM, jacobian=0)
    if len(img.shape) != 2: canvas = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else: canvas = img.copy()
    black_pixel_list, white_pixel_list = [], []
    for y in range(-4, 4):
        for x in range(-4, 4):
            board_coordinate = np.array([x * 0.05, y * 0.05, 0.0])
            poly = getPoly2D(rvec, tvec + np.dot(board_coordinate, rotM.T).reshape((3, 1)), size = 0.05)
            tile_samples = sample_polygon(np.array(poly, dtype=np.int32).reshape(-1, 2), n=10)
            for tile_sample in tile_samples:
                intensity = canvas[tile_sample[1]][tile_sample[0]]
                if (y%2 + x%2) % 2 == 0:
                    white_pixel_list.append(intensity)
                    cv2.circle(img, tile_sample, 3, (255, 0, 0), -1)
                else:
                    black_pixel_list.append(intensity)
                    cv2.circle(img, tile_sample, 3, (0, 255, 0), -1)

    black_pixel = sorted(black_pixel_list)[int(len(black_pixel_list)/2)]
    white_pixel = sorted(white_pixel_list)[int(len(white_pixel_list)/2)]

    if black_pixel > white_pixel: # wrong side -> rotate 90 degree
        rotM, _ = cv2.Rodrigues(rvec, rotM, jacobian=0) # Convert from rotation vector -> rotation matrix (rvec=rot_vector, rotM=rot_matrix)
        rotM = np.dot(rotM, np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]))  # Rotate 180 degree about Z-axis
        rvec, _ = cv2.Rodrigues(rotM, rvec, jacobian=0) # Convert from rotation matrix -> rotation vector
    return rvec
def correct_180(img, rvec, tvec, n=10):
    rotM = np.zeros(shape=(3, 3))
    cv2.Rodrigues(rvec, rotM, jacobian=0)
    if len(img.shape) != 2: canvas = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else: canvas = img.copy()
    [poly_up, poly_down] = get_bars(rvec, tvec, image=img)
    up_sample, down_sample = sample_polygon(poly_up, n=n), sample_polygon(poly_down, n=n)
    up_intensity, down_intensity = 0, 0
    for up_point, down_point in zip(up_sample, down_sample):
        up_intensity += canvas[up_point[1]][up_point[0]]
        down_intensity += canvas[down_point[1]][down_point[0]]
    # print(up_intensity, down_intensity)
    if up_intensity > down_intensity: # wrong side (white nedeed to be downside)
        rotM, _ = cv2.Rodrigues(rvec, rotM, jacobian=0) # Convert from rotation vector -> rotation matrix (rvec=rot_vector, rotM=rot_matrix)
        rotM = np.dot(rotM, np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])) # Rotate 180 degree about Z-axis
        rvec, _ = cv2.Rodrigues(rotM, rvec, jacobian=0) # Convert from rotation matrix -> rotation vector
    return rvec
def getBox2D(rvec, tvec, size = 0.05, height = scan_box_height):
    objpts = np.float32([[0, 0, 0], [size, 0, 0], [size, size, 0], [0, size, 0], [0, 0, height], [size, 0, height], [size, size, height], [0, size, height]]).reshape(-1, 3)
    imgpts, jac = cv2.projectPoints(objpts, rvec, tvec, cameraMatrix, dist)
    min_x = int(min(imgpts, key=lambda x: x[0][0]).ravel()[0])
    max_x = int(max(imgpts, key=lambda x: x[0][0]).ravel()[0])
    min_y = int(min(imgpts, key=lambda x: x[0][1]).ravel()[1])
    max_y = int(max(imgpts, key=lambda x: x[0][1]).ravel()[1])
    return (min_x, min_y), (max_x, max_y)
def getValidContour2D(rvec, tvec, size = 0.05, height = scan_box_height):
    objpts = np.float32([[0, 0, 0], [size, 0, 0], [size, size, 0], [0, size, 0], [0, 0, height], [size, 0, height], [size, size, height], [0, size, height]]).reshape(-1, 3)
    imgpts, jac = cv2.projectPoints(objpts, rvec, tvec, cameraMatrix, dist)
    valid_contours = []
    for mask_contour_index in mask_contour_index_list:
        valid_contours.append([imgpts[mask_contour_index[i]] for i in range(len(mask_contour_index))])
    return valid_contours
def getPoly2D(rvec, tvec, size = 0.05):
    objpts = np.float32([[0, 0, 0], [size, 0, 0], [size, size, 0], [0, size, 0]]).reshape(-1, 3)
    imgpts, jac = cv2.projectPoints(objpts, rvec, tvec, cameraMatrix, dist)
    return imgpts
def sample_polygon(poly, n=10):
    x_list = np.transpose(poly)[0]
    y_list = np.transpose(poly)[1]
    counter = 0 # count sample points inside polygon
    sample_points = []
    while True:
        x, y = random.randint(min(x_list), max(x_list)), random.randint(min(y_list), max( y_list))  # randomly sample point from inside polygon
        if cv2.pointPolygonTest(poly, (x, y), False) == 1.0:
            sample_points.append((x, y))
            counter += 1
        if counter == n: break
    return sample_points
def drawPoly2D(frame, rvec, tvec, size = 0.05, color=(0, 0, 255), thickness = 2):
    imgpts = getPoly2D(rvec, tvec, size=size).astype(np.int32)
    cv2.line(frame, tuple(imgpts[0].ravel()), tuple(imgpts[1].ravel()), color, thickness)
    cv2.line(frame, tuple(imgpts[1].ravel()), tuple(imgpts[2].ravel()), color, thickness)
    cv2.line(frame, tuple(imgpts[2].ravel()), tuple(imgpts[3].ravel()), color, thickness)
    cv2.line(frame, tuple(imgpts[3].ravel()), tuple(imgpts[0].ravel()), color, thickness)
def poly2view_angle(poly):
    rvec_tile, tvec_tile, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners=np.asarray([poly]), markerLength=0.05, cameraMatrix=cameraMatrix, distCoeffs=dist)
    rotM_tile = np.zeros(shape=(3, 3))
    rotM_tile, _ = cv2.Rodrigues(rvec_tile, rotM_tile, jacobian=0)
    tvec_tile_final = np.dot(tvec_tile, rotM_tile.T).reshape(3)
    tile_x, tile_y, tile_z = tvec_tile_final[0], tvec_tile_final[1], tvec_tile_final[2]
    angle_rad = math.asin((math.sqrt(tile_x ** 2 + tile_y ** 2)) / (math.sqrt(tile_x ** 2 + tile_y ** 2 + tile_z ** 2)))
    return angle_rad
def llr_tile(rvec, tvec):
    rotM = np.zeros(shape=(3, 3))
    rotM, _ = cv2.Rodrigues(rvec, rotM, jacobian=0)
    ### Draw chess piece space ###
    counter = 0
    tile_volume_bbox_list, angle_list, valid_contours_list = [], [], []
    for y in range(3, -5, -1):
        for x in range(-4, 4):
            board_coordinate = np.array([x * 0.05, y * 0.05, 0.0])
            (min_x, min_y), (max_x, max_y) = getBox2D(rvec, tvec + np.dot(board_coordinate, rotM.T).reshape((3, 1)), size=0.05, height=scan_box_height)
            tile_volume_bbox_list.append([(min_x, min_y), (max_x, max_y)])

            translated_tvec = tvec + np.dot(board_coordinate, rotM.T).reshape((3, 1))
            valid_contours = getValidContour2D(rvec, translated_tvec, size=0.05, height=scan_box_height)
            valid_contours_list.append(valid_contours)

            counter += 1
    tile_volume_bbox_list_new = []
    for i in range(64):
        y, x = 7 - int(i / 8), i % 8
        tile_volume_bbox_list_new.append(tile_volume_bbox_list[8*y+x])
    return tile_volume_bbox_list, valid_contours_list
def getCNNinput(img, bbox_list, valid_contours_list):
    CNNinputs = []
    for i in range(len(bbox_list)):
        [(min_x, min_y), (max_x, max_y)] = bbox_list[i]
        if min_x < 0: min_x = 0
        if min_y < 0: min_y = 0
        if max_x >= img.shape[1]: max_x = img.shape[1]-1
        if max_y >= img.shape[0]: max_y = img.shape[0]-1
        cropped = img[min_y:max_y, min_x:max_x].copy()
        valid_contours = valid_contours_list[i]
        mask = np.zeros(cropped.shape[:2], dtype="uint8")
        for valid_contour in valid_contours:
            local_valid_contour = []
            for point in valid_contour:
                x = int(point[0][0] - min_x)
                y = int(point[0][1] - min_y)
                local_valid_contour.append([x, y])
            local_valid_contour = np.array(local_valid_contour).reshape((-1, 1, 2)).astype(np.int32)
            cv2.drawContours(mask, [local_valid_contour], -1, 255, -1)
        CNNinputs.append(cv2.bitwise_and(cropped, cropped, mask=mask))
    return CNNinputs
def resize_and_pad(img, size=300, padding_color=(0,0,0)):
    old_size = img.shape[:2]
    ratio = float(size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])
    img = cv2.resize(img, (new_size[1], new_size[0]))
    delta_w = size - new_size[1]
    delta_h = size - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=padding_color)
def get_tile(img, rvec, tvec):
    tile_volume_bbox_list, valid_contours_list = llr_tile(rvec, tvec)
    CNNinputs = getCNNinput(img, tile_volume_bbox_list, valid_contours_list)
    CNNinputs_padded = []
    for i in range(64):
        CNNinput_padded = resize_and_pad(CNNinputs[i], size=export_size)
        CNNinputs_padded.append(CNNinput_padded)
    return CNNinputs_padded
def llr_tile_top(rvec, tvec):
    rotM = np.zeros(shape=(3, 3))
    rotM, _ = cv2.Rodrigues(rvec, rotM, jacobian=0)
    ### Draw chess piece space ###
    counter = 0
    poly_tile_list = []
    for y in range(3, -5, -1):
        for x in range(-4, 4):
            board_coordinate = np.array([x * 0.05, y * 0.05, 0.0])
            # find angle of each tile
            translated_tvec = tvec + np.dot(board_coordinate, rotM.T).reshape(3, 1)
            poly_tile = getPoly2D(rvec, translated_tvec, size=0.05)
            poly_tile_list.append(poly_tile)
    return poly_tile_list
def get_tile_top(img, rvec, tvec):
    poly_tile_list = llr_tile_top(rvec, tvec)
    CNNinputs = []
    pts2 = np.float32([(0, export_size-1), (export_size-1, export_size-1), (export_size-1, 0), (0, 0)])
    for poly_tile in poly_tile_list:
        M = cv2.getAffineTransform(np.float32(poly_tile).reshape((4, 2))[:3], pts2[:3])
        CNNinputs.append(cv2.warpAffine(img, M, (export_size, export_size)))
    return CNNinputs
def combine_CNNinputs(CNNinputs_padded):
    vertical_images = []
    for x in range(8):
        image_list_vertical = []
        for y in range(7, -1, -1):
            canvas = resize_and_pad(CNNinputs_padded[8 * y + x].copy(), size=100)
            # image_list_vertical.append(canvas)
            image_list_vertical.append(cv2.copyMakeBorder(canvas, 1, 1, 1, 1, cv2.BORDER_CONSTANT, None, (0, 255, 0)))
        vertical_images.append(np.vstack(image_list_vertical))
    combined_images = np.hstack(vertical_images)
    return combined_images
class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.four_points = []   # chessboard corners input buffer
        self.chessboard_corners = None    # actual (used) chessboard corners
        self.rvec, self.tvec = None, None
        self.hand_in_frame, self.robot_in_frame = False, True
        self.board_result_binary = np.zeros((8, 8), dtype=np.uint8)
        self.board_result_binary_buffer = []  # store history of self.board_result_binary
        self.board_result_color = np.zeros((8, 8), dtype=np.uint8)
        self.board_result_color_buffer = []  # store history of self.board_result_color
        self.binary_filter_length = 10
        self.color_filter_length = 5
        self.clustering = None
        self.clustering_lock = False
        self.clustering_flip = False
        self.zoom_area = 50
        self.cursor = (-1, -1) # mouse tracking (for zoom)

        self.camera_resolution = (1920, 1080)
        self.cap = Camera(0, self.camera_resolution[0], self.camera_resolution[1])
        self.setWindowTitle('Corner Calibration')
        self.setGeometry(0, 0, 1280, 720)
        self.display_scale = 0.5
        self.disply_width = self.camera_resolution[0]*self.display_scale
        self.display_height = self.camera_resolution[1]*self.display_scale
        self.create_widgets()

        # ROS2 init
        rclpy.init(args=None)
        self.node = Node('chessboard_estimator')
        self.pub_img = self.node.create_publisher(Image, "/camera0/image", 10)
        self.pub_fen_binary = self.node.create_publisher(UInt8MultiArray, '/chessboard/fen_binary', 10)
        self.pub_fen_color = self.node.create_publisher(UInt8MultiArray, '/chessboard/fen_color', 10)
        self.pub_fen = self.node.create_publisher(UInt8MultiArray, '/chessboard/fen', 10)
        self.joint0_sub = self.node.create_subscription(Float32, '/chessboard/joint0', self.robot_joint0_callback, 10)

        self.bridge = CvBridge()        # Bridge between "CV (NumPy array)" <-> "ROS sensor_msgs/Image"
    def __del__(self): self.node.destroy_node()
    def robot_joint0_callback(self, joint0):
        angle = joint0.data
        if abs(angle) < 0.75: self.robot_in_frame = True
        else: self.robot_in_frame = False
    def create_widgets(self):
        # create label to display image
        self.image_frame = QLabel(self)
        self.image_frame.resize(self.disply_width, self.display_height)
        # create label to display CNN padded image
        self.image_frame_CNN_padded = QLabel(self)
        # create text label
        self.textLabel = QLabel('[]')


        self.button_clear = QPushButton('clear_points')
        self.button_clear.clicked.connect(self.clear_points)
        self.button_confirm = QPushButton('Confirm')
        self.button_confirm.clicked.connect(self.set_points)
        self.button_save = QPushButton('Save')
        self.button_save.clicked.connect(self.save_points)
        self.button_load = QPushButton('Load')
        self.button_load.clicked.connect(self.load_points)
        self.button_color_lock = QPushButton('LOCK Color clustering')
        self.button_color_lock.clicked.connect(self.lock_color)

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.image_frame)
        self.layout.addWidget(self.image_frame_CNN_padded)
        self.layout.addWidget(self.textLabel)
        self.layout.addWidget(self.button_clear)
        self.layout.addWidget(self.button_confirm)
        self.layout.addWidget(self.button_save)
        self.layout.addWidget(self.button_load)
        self.layout.addWidget(self.button_color_lock)
        self.setMouseTracking(True)  # enable mouse tracking event not pressing
        self.setLayout(self.layout)
        self.show()

        # create timer
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.timer_callback)
        self.timer.start(10)
    def timer_callback(self):
        rclpy.spin_once(self.node)
        self.frame = self.cap.read()
        image_msg = self.bridge.cv2_to_imgmsg(self.frame, "bgr8")
        image_msg.header.stamp = self.node.get_clock().now().to_msg()
        self.pub_img.publish(image_msg)         # Publish image
        # Display image
        canvas = self.frame.copy()
        if self.rvec is not None and self.tvec is not None:
            canvas = draw_axis(canvas, self.chessboard_corners, self.tvec, self.rvec)
            cv2.polylines(canvas, [np.array(self.chessboard_corners, np.int32).reshape((-1,1,2))], True, (0, 0, 255))
            # drawPoly2D(canvas, self.rvec, self.tvec+[[0.2], [0.2], [0]], size=0.4, color=(0, 0, 255), thickness=2)
            CNNinputs_padded = get_tile_top(self.frame, self.rvec, self.tvec)
            CNNinputs_padded_canvas = np.array(CNNinputs_padded, dtype=np.uint8)
            for y in range(8):
                for x in range(8):
                    overlay = np.zeros((export_size, export_size, 3), dtype=np.uint8)
                    overlay[:, :] = (0, 0, 255) if self.board_result_binary[y][x] == 0 else (0, 255, 0)
                    CNNinputs_padded_canvas[8 * y + x] = cv2.addWeighted(CNNinputs_padded_canvas[8 * y + x], 0.8, overlay, 0.2, 0.0)

            combined_image = combine_CNNinputs(CNNinputs_padded_canvas)
            # cv2.imwrite("/home/fiborobotlab/test_CNNinputs.png", combined_image)
            self.image_frame_CNN_padded.setPixmap(self.convert_cv_qt(combined_image, 200, 200))
            # Detect Hand
            results = hands.process(cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB))
            # Check if hand overlap the chessboard
            hand_in_frame = False
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw the hand annotations on the image.
                    mp_drawing.draw_landmarks(
                        canvas,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
                    for point in hand_landmarks.landmark:
                        x = point.x * self.frame.shape[1]
                        y = point.y * self.frame.shape[0]
                        chessboard_poly = getPoly2D(self.rvec, self.tvec+[[0.2], [0.2], [0]], size=0.4).astype(np.int32)
                        if cv2.pointPolygonTest(chessboard_poly, (x, y), False) == 1.0:
                            hand_in_frame = True
                            break
                    if hand_in_frame == True: break
            self.hand_in_frame = hand_in_frame
            if not (self.hand_in_frame or self.robot_in_frame):  # only continue to classify chess pieces if no hand&robot in frame
                infer_request_binary.infer([CNNinputs_padded])
                board_result = infer_request_binary.get_output_tensor().data  # shape = (64, 8)
                board_result_binary = np.argmax(board_result, axis=1).reshape(8, 8)
                board_result_binary = np.where(board_result_binary == 0, 0, 1)  # Interpreted prediction
                board_result_color = board_result[:, 1:]  # exclude first channel (empty tile) and keep only color features
                # color clustering
                tile_index_non_empty = np.argwhere(board_result_binary.ravel() != 0).reshape(-1)    # get indices of all non-empty tile
                board_result_color = board_result_color[tile_index_non_empty]
                if self.clustering_lock:  # Use saved clustering model
                    clustering = self.clustering
                    cluster_result = clustering.predict(board_result_color)
                else:
                    if len(board_result_color) < 2:  # not enough sample for clustering
                        clustering = None
                        self.node.get_logger().warn("Not enough sample for clustering (Minimum is 2)")
                    else:
                        clustering = KMeans(n_clusters=2, random_state=0).fit(board_result_color)
                        cluster_result = clustering.labels_
                        self.clustering = clustering  # update clustering model
                        ## fit to nearest side automatically ##
                        white_side, black_side = [], []
                        for j in range(len(tile_index_non_empty)):
                            row_index = int(tile_index_non_empty[j] / 8)
                            if row_index < 4:
                                black_side.append(cluster_result[j])
                            else:
                                white_side.append(cluster_result[j])
                        try:
                            if np.argmax(np.bincount(white_side)) == 0:  # Whites already labeled as '0'
                                self.clustering_flip = False
                            else:
                                self.clustering_flip = True
                        except:
                            pass
                if clustering is not None:
                    for j in range(len(cluster_result)):
                        cluster_label = cluster_result[j]
                        if self.clustering_flip == True:
                            cluster_label = abs(cluster_label - 1)  # flip 0 <-> 1
                    # self.board_result_color_buffer.append(self.board_result_color)
                    color_array = np.zeros((8, 8), dtype=np.uint8)
                    for i in range(len(tile_index_non_empty)):
                        non_empty_index = tile_index_non_empty[i]
                        color_array[int(non_empty_index / 8)][non_empty_index % 8] = cluster_result[i]
                    self.board_result_color_buffer.append(color_array)
                    # print(self.board_result_color_buffer)
                    if len(self.board_result_color_buffer) >= self.color_filter_length:  # fill buffer first
                        for y in range(8):
                            for x in range(8):
                                buffer = []
                                for i in range(self.color_filter_length): buffer.append(
                                    self.board_result_color_buffer[i][y][x])
                                # update value to most frequent in buffer
                                self.board_result_color[y][x] = np.argmax(np.bincount(buffer))
                                # if np.all(np.array(buffer) == buffer[0]): self.board_result_color[y][x] = buffer[0]
                        while len(self.board_result_color_buffer) >= self.color_filter_length:
                            self.board_result_color_buffer.pop(0)  # remove first element in buffer

                # print(board_result_binary)
                # print(np.bincount(board_result_binary.ravel()))

                ## Publish FEN binary ##
                fen_binary_msg = UInt8MultiArray()
                fen_binary_msg.data = [int(item) for item in self.board_result_binary.reshape(-1)]
                self.pub_fen_binary .publish(fen_binary_msg)
                fen_color_msg = UInt8MultiArray()
                fen_color_msg.data = [int(item) for item in self.board_result_color.reshape(-1)]
                self.pub_fen_color.publish(fen_color_msg)

                self.board_result_binary_buffer.append(board_result_binary)
                if len(self.board_result_binary_buffer) >= self.binary_filter_length:
                    for y in range(8):
                        for x in range(8):
                            buffer = []
                            for i in range(self.binary_filter_length): buffer.append(self.board_result_binary_buffer[i][y][x])
                            # update value to most frequent in buffer
                            self.board_result_binary[y][x] = np.argmax(np.bincount(buffer))
                            # if np.all(np.array(buffer) == buffer[0]): self.board_result_binary[y][x] = buffer[0]
                    while len(self.board_result_binary_buffer) >= self.binary_filter_length:
                        self.board_result_binary_buffer.pop(0)  # remove first element in buffer
                print(self.board_result_binary)
        if len(self.four_points) != 0:
            for center in self.four_points:
                scaled_center = (int(center[0]/self.display_scale), int(center[1]/self.display_scale))
                canvas = cv2.circle(canvas, scaled_center, 5, (0, 255, 0), -1)
        # Partial zoom when mouse moving inside camera frame
        # if self.cursor[0] >0 and self.cursor[0] < canvas.shape[1] and self.cursor[1] > 0 and self.cursor[1] < canvas.shape[0]:
        #     if self.cursor[0] - 2 * self.zoom_area > 0 and canvas.shape[1] - self.cursor[0] > 2 * self.zoom_area and self.cursor[ 1] - 2 * self.zoom_area > 0 and canvas.shape[0] - self.cursor[1] > 2 * self.zoom_area:
        #         zoom_source = canvas[self.cursor[1] - self.zoom_area:self.cursor[1] + self.zoom_area, self.cursor[0] - self.zoom_area:self.cursor[0] + self.zoom_area]
        #         canvas[self.cursor[1] - 2 * self.zoom_area:self.cursor[1] + 2 * self.zoom_area, self.cursor[0] - 2 * self.zoom_area:self.cursor[0] + 2 * self.zoom_area] = imutils.resize(zoom_source, height=self.zoom_area * 4)
        self.image_frame.setPixmap(self.convert_cv_qt(canvas, self.disply_width, self.display_height))
        self.show()
        # update after 1 second
        self.timer.start(10)
    def mouseReleaseEvent(self, event):
        x = event.x()                      # actual cursor position
        y = event.y()
        x_offset = int(self.image_frame.pos().x())   # image offset
        y_offset = int(self.image_frame.pos().y())
        self.four_points.append((x-x_offset, y-y_offset))
        self.update_text()
    def mouseMoveEvent(self, event) -> None:
        # print(event.x(), event.y())
        self.cursor = (int((event.x()-self.image_frame.pos().x())/self.display_scale), int((event.y()-self.image_frame.pos().y())/self.display_scale))

    def update_text(self):
        self.textLabel.setText(str(self.four_points))
    @QtCore.pyqtSlot()
    def clear_points(self):
        self.four_points = []
        self.update_text()

    @QtCore.pyqtSlot()
    def set_points(self):
        global obj_points, cameraMatrix, dist

        if len(self.four_points) == 4: self.chessboard_corners = order_points(np.array(self.four_points.copy()))/self.display_scale
        self.update_text()
        if self.chessboard_corners is not None:
            self.solvePnP()
    def save_points(self):
        if self.chessboard_corners is not None:
            np.savez("calibration.npz",
                     tvec=self.tvec,
                     rvec=self.rvec,
                     corners=self.chessboard_corners)
    def load_points(self):
        if os.path.exists("calibration.npz"):
            loaded = np.load("calibration.npz")
            self.tvec = loaded['tvec']
            self.rvec = loaded['rvec']
            self.chessboard_corners = loaded['corners']
            self.solvePnP()
    def lock_color(self):
        if self.clustering is not None: # if clustering model already exist
            if self.clustering_lock:
                self.clustering_lock = False
                self.button_color_lock.setText("LOCK Color clustering")
            else:
                self.clustering_lock = True
                self.button_color_lock.setText("UNLOCK Color clustering")
    def solvePnP(self):
        img_points = np.array(self.chessboard_corners, dtype=np.float32).reshape((-1, 2))
        ret, rvec, tvec = cv2.solvePnP(objectPoints=obj_points,
                                       imagePoints=img_points,
                                       cameraMatrix=cameraMatrix,
                                       distCoeffs=dist,
                                       flags=0)
        # Correct the chessboard side (white stript on lower Y-axis)
        rvec = correct_90(self.frame, rvec, tvec)  # check & correct board rotation
        rvec = correct_180(self.frame, rvec, tvec)

        self.rvec, self.tvec = rvec, tvec
        # self.four_points = []
    def convert_cv_qt(self, cv_img, width, height):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(width, height, QtCore.Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())