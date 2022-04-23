#!/usr/bin/env /home/teera/.virtualenvs/cv/bin/python

import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Float32, String
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Pose, Point, Quaternion
from sensor_msgs.msg import CameraInfo, Image
from module89.msg import ChessboardImgPose

import cv2
import math, random
import json
import pyquaternion as q
import tensorflow as tf

from transform import four_point_transform, order_points, poly2view_angle

import os
import numpy as np

## Avoid to use all GPU(s)VRAM
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)

colors = [(255, 255, 255), (0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0)]

camera_config = json.load(open(os.path.join(get_package_share_directory('module89'), 'config', 'camera_config.json')))
model_config = json.load(open(os.path.join(get_package_share_directory('module89'), 'config', 'model_config.json')))
cameraMatrix = np.array(camera_config['camera_matrix'], np.float32)
dist = np.array(camera_config['dist'])
export_size = 224
chess_piece_height = {"king": (0.081, 0.097), "queen": (0.07, 0.0762), "bishop": (0.058, 0.065), "knight": (0.054, 0.05715), "rook": (0.02845, 0.048), "pawn": (0.043, 0.045)}
chess_piece_diameter = {"king": (0.028, 0.0381), "queen": (0.028, 0.0362), "bishop": (0.026, 0.032), "knight": (0.026, 0.03255), "rook": (0.026, 0.03255), "pawn": (0.0191, 0.02825)}
mask_contour_index_list = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
scan_box_height = min(chess_piece_height['king'])

cv_bridge = CvBridge()
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
def llr_tile(rvec, tvec):
    rotM = np.zeros(shape=(3, 3))
    rotM, _ = cv2.Rodrigues(rvec, rotM, jacobian=0)
    ### Draw chess piece space ###
    counter = 0
    tile_volume_bbox_list, angle_list, valid_contours_list = [], [], []
    for y in range(3, -5, -1):
        for x in range(-4, 4):
            board_coordinate = np.array([x * 0.05, y * 0.05, 0.0])
            (min_x, min_y), (max_x, max_y) = getBox2D(rvec, tvec + np.dot(board_coordinate, rotM.T), size=0.05, height=scan_box_height)
            tile_volume_bbox_list.append([(min_x, min_y), (max_x, max_y)])

            # find angle of each tile
            translated_tvec = tvec + np.dot(board_coordinate, rotM.T)
            poly_tile = getPoly2D(rvec, translated_tvec, size=0.05)
            valid_contours = getValidContour2D(rvec, translated_tvec, size=0.05, height=scan_box_height)
            valid_contours_list.append(valid_contours)
            angle_rad = poly2view_angle(poly_tile)

            angle_deg = angle_rad / 3.14 * 180
            angle_list.append(angle_deg)
            counter += 1
    tile_volume_bbox_list_new, angle_list_new = [], []
    for i in range(64):
        y, x = 7 - int(i / 8), i % 8
        tile_volume_bbox_list_new.append(tile_volume_bbox_list[8*y+x])
        angle_list_new.append(angle_list[8*y+x])
    return tile_volume_bbox_list, angle_list, valid_contours_list
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
    tile_volume_bbox_list, angle_list, valid_contours_list = llr_tile(rvec, tvec)
    CNNinputs = getCNNinput(img, tile_volume_bbox_list, valid_contours_list)
    CNNinputs_padded = []
    for i in range(64):
        CNNinput_padded = resize_and_pad(CNNinputs[i], size=export_size)
        CNNinputs_padded.append(CNNinput_padded)
    return CNNinputs_padded, angle_list
def get_tile_ImgPose(img_pose: ChessboardImgPose):
    tvec = img_pose.pose.position
    tvec = np.array([tvec.x, tvec.y, tvec.z])
    rvec = img_pose.pose.orientation
    rvec = q.Quaternion(x=rvec.x, y=rvec.y, z=rvec.z, w=rvec.w)  # Convert to PyQuaternion object
    img = cv_bridge.imgmsg_to_cv2(img_pose.image, desired_encoding='passthrough')
    rvec, _ = cv2.Rodrigues(rvec.rotation_matrix, jacobian=0)
    # cv2.aruco.drawAxis(image=img, cameraMatrix=cameraMatrix, distCoeffs=dist, rvec=rvec, tvec=tvec, length=0.1)
    # cv2.imshow("BBB", img)
    # cv2.waitKey(1)
    return get_tile(img, rvec, tvec)

def to_FEN(board):
    symbol_dict = {1:'b', 2:'k', 3:'n', 4:'p', 5:'q', 6:'r'}
    FEN_string = ""
    for y in range(8):
        empty = 0
        for x in range(8):
            if board[y][x] == 0: empty+= 1
            else:
                if empty != 0:
                    FEN_string += str(empty)
                    empty = 0
                FEN_string += symbol_dict[board[y][x]]
        if empty != 0: FEN_string += str(empty)
        FEN_string += '/'
    return FEN_string[:-1]  # exclude lastest '/'
class ChessboardClassifier(Node):
    def __init__(self):
        super().__init__('chessboard_classifier')
        # Subscribe chessboard poses
        self.chessboard_pose_top_sub = self.create_subscription(ChessboardImgPose, '/chessboard/top/ImgPose', self.chessboard_pose_top_callback, 10)
        self.chessboard_pose_side_sub = self.create_subscription(ChessboardImgPose, '/chessboard/side/ImgPose', self.chessboard_pose_side_callback, 10)
        self.fen_pub = self.create_publisher(String, 'chessboard/fen', 10)

        self.bridge = CvBridge()  # Bridge between "CV (NumPy array)" <-> "ROS sensor_msgs/Image"

        self._valid = np.zeros((8, 8), dtype=np.uint8)
        self._chess = np.zeros((8, 8), dtype=np.uint8)

        self.top_model = tf.keras.models.load_model(model_config['top_classifier'])
        self.side_model = tf.keras.models.load_model(model_config['side_classifier'])
        self.top_model.summary()
        self.side_model.summary()

        self.board_result_binary = np.zeros((8, 8))
        self.board_result_binary_buffer = []   # store history of self.board_result_binary
        self.board_result = np.zeros((8, 8))
        self.board_result_buffer = []   # store history of self.board_result

        self.top_filter = True
        self.side_filter = True
        self.top_filter_length = 3
        self.side_filter_length = 3
    def chessboard_pose_top_callback(self, img_pose):
        frame = self.bridge.imgmsg_to_cv2(img_pose.image, desired_encoding='passthrough')
        # self.get_logger().info(str(get_tile(img_pose)))
        CNNinputs_padded, angle_list = get_tile_ImgPose(img_pose)
        Y = self.top_model.predict(np.array(CNNinputs_padded).reshape((-1, 224, 224, 3)))
        Y = np.array(Y).reshape((-1))
        Y = np.where(Y < 0, 0, 1)   # Interpreted prediction
        self.board_result_binary_buffer.append(Y.reshape((8, 8)))
        if len(self.board_result_binary_buffer) >= self.top_filter_length:  # fill buffer first
            for y in range(8):
                for x in range(8):
                    buffer = []
                    for i in range(self.top_filter_length): buffer.append(self.board_result_binary_buffer[i][y][x])
                    # update only value that pass filter
                    if np.all(np.array(buffer) == buffer[0]): self.board_result_binary[y][x] = buffer[0]
            while len(self.board_result_binary_buffer) >= self.top_filter_length:
                self.board_result_binary_buffer.pop(0)  # remove first element in buffer
        # self.get_logger().info(str(self.board_result_binary))

        # try:
        #     if True:
        #         vertical_images = []
        #         for x in range(8):
        #             image_list_vertical = []
        #             for y in range(7, -1, -1):
        #                 canvas = resize_and_pad(CNNinputs_padded[8 * y + x].copy(), size=100)
        #                 cv2.putText(canvas, str(round(angle_list[8 * y + x])), (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
        #                             color=(0, 0, 255))
        #                 image_list_vertical.append(
        #                     cv2.copyMakeBorder(canvas, 1, 1, 1, 1, cv2.BORDER_CONSTANT, None, (0, 255, 0)))
        #             vertical_images.append(np.vstack(image_list_vertical))
        #         combined_images = np.hstack(vertical_images)
        #         cv2.imshow("All CNN inputs", combined_images)
        # except: pass

        cv2.imshow("Top", frame)
        cv2.waitKey(1)
    def chessboard_pose_side_callback(self, img_pose):
        frame = self.bridge.imgmsg_to_cv2(img_pose.image, desired_encoding='passthrough')
        # self.get_logger().info(str(get_tile(img_pose)))
        board_not_empty = np.argwhere(self.board_result_binary.reshape(-1) != 0).reshape(-1)
        tile_index_non_empty = board_not_empty
        CNNinputs_padded, angle_list = get_tile_ImgPose(img_pose)
        CNNinputs_padded_non_empty = np.array(CNNinputs_padded)[tile_index_non_empty]
        board_result = np.zeros((8, 8))    # Reset to empty board
        if len(board_not_empty) > 0:
            Y = np.array(self.side_model.predict(np.array(CNNinputs_padded_non_empty)))
            Y = np.argmax(Y, axis=1)  # Use class with max score
            # Remap back to chessboard
            for i in range(len(tile_index_non_empty)):
                index = tile_index_non_empty[i]
                board_result[int(index / 8)][index % 8] = Y[i] + 1
        self.board_result_buffer.append(board_result)
        if len(self.board_result_buffer) >= self.side_filter_length:  # fill buffer first
            for y in range(8):
                for x in range(8):
                    buffer = []
                    for i in range(self.side_filter_length): buffer.append(self.board_result_buffer[i][y][x])
                    # update only value that pass filter
                    if np.all(np.array(buffer) == buffer[0]): self.board_result[y][x] = buffer[0]
            while len(self.board_result_buffer) >= self.side_filter_length:
                self.board_result_buffer.pop(0)  # remove first element in buffer
        fen_message = String()
        fen_message.data = to_FEN(self.board_result)
        self.fen_pub.publish(fen_message)

        # try:
        #     if True:
        #         vertical_images = []
        #         for x in range(8):
        #             image_list_vertical = []
        #             for y in range(7, -1, -1):
        #                 canvas = resize_and_pad(CNNinputs_padded[8 * y + x].copy(), size=100)
        #                 cv2.putText(canvas, str(round(angle_list[8 * y + x])), (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
        #                             color=(0, 0, 255))
        #                 image_list_vertical.append(
        #                     cv2.copyMakeBorder(canvas, 1, 1, 1, 1, cv2.BORDER_CONSTANT, None, (0, 255, 0)))
        #             vertical_images.append(np.vstack(image_list_vertical))
        #         combined_images = np.hstack(vertical_images)
        #         cv2.imshow("All CNN inputs", combined_images)
        # except: pass

        cv2.imshow("Side", frame)
        cv2.waitKey(1)

def main():
    rclpy.init()
    chessboard_classifier = ChessboardClassifier()
    rclpy.spin(chessboard_classifier)
    # chessboard_detector.destroy_subscription(chessboard_detector.camera_sub) # Not need camera after init pose
    rclpy.shutdown()

if __name__ == "__main__":
    main()