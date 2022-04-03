#!/usr/bin/env /home/teera/.virtualenvs/cv/bin/python

import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Float32
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Pose, Point, Quaternion
from sensor_msgs.msg import CameraInfo, Image
from module89.msg import ChessboardImgPose

import cv2
import math, random
import simplejson
import pyquaternion as q

from transform import four_point_transform, order_points, poly2view_angle

import os
import numpy as np

colors = [(255, 255, 255), (0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0)]

config = simplejson.load(open(os.path.join(get_package_share_directory('module89'), 'config', 'camera_config.json')))
cameraMatrix = np.array(config['camera_matrix'], np.float32)
dist = np.array(config['dist'])

chess_piece_height = {"king": (0.081, 0.097), "queen": (0.07, 0.0762), "bishop": (0.058, 0.065), "knight": (0.054, 0.05715), "rook": (0.02845, 0.048), "pawn": (0.043, 0.045)}
chess_piece_diameter = {"king": (0.028, 0.0381), "queen": (0.028, 0.0362), "bishop": (0.026, 0.032), "knight": (0.026, 0.03255), "rook": (0.026, 0.03255), "pawn": (0.0191, 0.02825)}
# scan_box_height = max(chess_piece_height['king'])
scan_box_height = min(chess_piece_height['king'])

cv_bridge = CvBridge()
def getPoly2D(rvec, tvec, size = 0.05):
    objpts = np.float32([[0, 0, 0], [size, 0, 0], [size, size, 0], [0, size, 0]]).reshape(-1, 3)
    imgpts, jac = cv2.projectPoints(objpts, rvec, tvec, cameraMatrix, dist)
    return imgpts
def getCNNinput(img, bbox_list):
    CNNinputs = []
    for [(min_x, min_y), (max_x, max_y)] in bbox_list:
        if min_x < 0: min_x = 0
        if min_y < 0: min_y = 0
        if max_x >= img.shape[1]: max_x = img.shape[1]-1
        if max_y >= img.shape[0]: max_y = img.shape[0]-1
        CNNinputs.append(img[min_y:max_y, min_x:max_x])
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
def getBox2D(rvec, tvec, size = 0.05, height = scan_box_height):
    objpts = np.float32([[0, 0, 0], [size, 0, 0], [size, size, 0], [0, size, 0], [0, 0, height], [size, 0, height], [size, size, height], [0, size, height]]).reshape(-1, 3)
    imgpts, jac = cv2.projectPoints(objpts, rvec, tvec, cameraMatrix, dist)
    min_x = int(min(imgpts, key=lambda x: x[0][0]).ravel()[0])
    max_x = int(max(imgpts, key=lambda x: x[0][0]).ravel()[0])
    min_y = int(min(imgpts, key=lambda x: x[0][1]).ravel()[1])
    max_y = int(max(imgpts, key=lambda x: x[0][1]).ravel()[1])
    return (min_x, min_y), (max_x, max_y)
def drawPoly2D(frame, rvec, tvec, size = 0.05, color=(0, 0, 255), thickness = 2):
    imgpts = getPoly2D(rvec, tvec)
    cv2.line(frame, tuple(imgpts[0].ravel()), tuple(imgpts[1].ravel()), color, thickness)
    cv2.line(frame, tuple(imgpts[1].ravel()), tuple(imgpts[2].ravel()), color, thickness)
    cv2.line(frame, tuple(imgpts[2].ravel()), tuple(imgpts[3].ravel()), color, thickness)
    cv2.line(frame, tuple(imgpts[3].ravel()), tuple(imgpts[0].ravel()), color, thickness)

def drawBox3D(frame, rvec, tvec, size = 0.05, height = scan_box_height, color=(0, 0, 255), thickness = 2):
    objpts = np.float32([[0,0,0], [size,0,0], [size,size,0], [0,size,0], [0,0,height], [size,0,height], [size,size,height], [0,size,height]]).reshape(-1,3)
    imgpts, jac = cv2.projectPoints(objpts, rvec, tvec, cameraMatrix, dist)

    cv2.line(frame, tuple(imgpts[0].ravel()), tuple(imgpts[1].ravel()), color, thickness)
    cv2.line(frame, tuple(imgpts[1].ravel()), tuple(imgpts[2].ravel()), color, thickness)
    cv2.line(frame, tuple(imgpts[2].ravel()), tuple(imgpts[3].ravel()), color, thickness)
    cv2.line(frame, tuple(imgpts[3].ravel()), tuple(imgpts[0].ravel()), color, thickness)

    cv2.line(frame, tuple(imgpts[0].ravel()), tuple(imgpts[0+4].ravel()), color, thickness)
    cv2.line(frame, tuple(imgpts[1].ravel()), tuple(imgpts[1+4].ravel()), color, thickness)
    cv2.line(frame, tuple(imgpts[2].ravel()), tuple(imgpts[2+4].ravel()), color, thickness)
    cv2.line(frame, tuple(imgpts[3].ravel()), tuple(imgpts[3+4].ravel()), color, thickness)

    cv2.line(frame, tuple(imgpts[0+4].ravel()), tuple(imgpts[1+4].ravel()), color, thickness)
    cv2.line(frame, tuple(imgpts[1+4].ravel()), tuple(imgpts[2+4].ravel()), color, thickness)
    cv2.line(frame, tuple(imgpts[2+4].ravel()), tuple(imgpts[3+4].ravel()), color, thickness)
    cv2.line(frame, tuple(imgpts[3+4].ravel()), tuple(imgpts[0+4].ravel()), color, thickness)
def drawBox2D(frame, rvec, tvec, size = 0.05, height = scan_box_height, color=(0, 0, 255), thickness=1):
    objpts = np.float32([[0, 0, 0], [size, 0, 0], [size, size, 0], [0, size, 0], [0, 0, height], [size, 0, height], [size, size, height], [0, size, height]]).reshape(-1, 3)
    imgpts, jac = cv2.projectPoints(objpts, rvec, tvec, cameraMatrix, dist)
    min_x = int(min(imgpts, key=lambda x: x[0][0]).ravel()[0])
    max_x = int(max(imgpts, key=lambda x: x[0][0]).ravel()[0])
    min_y = int(min(imgpts, key=lambda x: x[0][1]).ravel()[1])
    max_y = int(max(imgpts, key=lambda x: x[0][1]).ravel()[1])
    cv2.polylines(frame, [np.array([(min_x, min_y), (min_x, max_y), (max_x, max_y), (max_x, min_y)])], isClosed=True, color=color, thickness=thickness)
def llr_tile(img, rvec, tvec, debug=False):
    rotM = np.zeros(shape=(3, 3))
    rotM, _ = cv2.Rodrigues(rvec, rotM, jacobian=0)
    # rotM, _ = cv2.Rodrigues(rvec) # get new rotation matrix

    canvas = img.copy()
    cv2.aruco.drawAxis(image=canvas, cameraMatrix=cameraMatrix, distCoeffs=dist, rvec=rvec, tvec=tvec, length=0.1)
    cv2.imshow("Rotated", canvas)
    cv2.waitKey(100)
    ### Draw chess piece space ###
    counter = 0
    tile_volume_bbox_list, angle_list = [], []
    if debug: canvas1 = img.copy()
    for y in range(-4, 4):
        for x in range(-4, 4):
            board_coordinate = np.array([x * 0.05, y * 0.05, 0.0])
            (min_x, min_y), (max_x, max_y) = getBox2D(rvec, tvec + np.dot(board_coordinate, rotM.T), size=0.05,
                                                      height=scan_box_height)
            tile_volume_bbox_list.append([(min_x, min_y), (max_x, max_y)])

            # find angle of each tile
            poly_tile = getPoly2D(rvec, tvec + np.dot(board_coordinate, rotM.T), size=0.05)
            angle_rad = poly2view_angle(poly_tile)

            angle_deg = angle_rad / 3.14 * 180
            angle_list.append(angle_deg)

            if debug:
                canvas = img.copy()
                # canvas1 = img.copy()
                canvas2 = img.copy()
                drawBox3D(canvas, rvec, tvec + np.dot(board_coordinate, rotM.T), size=0.05, height=scan_box_height,
                          color=colors[counter % len(colors)], thickness=1)
                drawPoly2D(canvas1, rvec, tvec + np.dot(board_coordinate, rotM.T), size=0.05,
                           color=colors[counter % len(colors)], thickness=1)
                drawBox2D(canvas2, rvec, tvec + np.dot(board_coordinate, rotM.T), size=0.05, height=scan_box_height,
                          color=colors[counter % len(colors)], thickness=1)
                # print(tvec + np.dot(board_coordinate, rotM.T))
                cv2.imshow("3D&2D box tiles", np.vstack([canvas, canvas2]))
                cv2.imshow("Poly tile", canvas1)
                # cv2.imwrite("box_tile_" + str(counter) + ".png", np.vstack([canvas, canvas2]))
                cv2.waitKey(10)
            counter += 1
    return tile_volume_bbox_list, angle_list
def get_tile(img, rvec, tvec, debug = False):
    tile_volume_bbox_list, angle_list = llr_tile(img, rvec, tvec, debug=debug)
    CNNinputs = getCNNinput(img, tile_volume_bbox_list)
    if debug:
        vertical_images = []
        for x in range(8):
            image_list_vertical = []
            for y in range(7, -1, -1):
                canvas = resize_and_pad(CNNinputs[8*y+x].copy(), size=100)
                cv2.putText(canvas, str(round(angle_list[8*y+x])), (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color=(0, 0, 255))
                image_list_vertical.append(cv2.copyMakeBorder(canvas, 1, 1, 1, 1, cv2.BORDER_CONSTANT, None, (0, 255, 0)))
            vertical_images.append(np.vstack(image_list_vertical))
        combined_images = np.hstack(vertical_images)
        cv2.imshow("All CNN inputs", combined_images)
        # cv2.imwrite("All_CNN_inputs.png", combined_images)
        cv2.waitKey(0)
    CNNinputs_padded = []
    for y in range(7, -1, -1):
        for x in range(8):
            CNNinput_padded = resize_and_pad(CNNinputs[8*y+x], size=300)
            CNNinputs_padded.append(CNNinput_padded)
            # cv2.imshow("A", CNNinput_padded)
            # cv2.waitKey(0)
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
    return get_tile(img, rvec, tvec, debug=False)

class ChessboardClassifier(Node):
    def __init__(self):
        super().__init__('chessboard_classifier')
        # Subscribe chessboard poses
        self.chessboard_pose_top_sub = self.create_subscription(ChessboardImgPose, '/chessboard/top/ImgPose', self.chessboard_pose_top_callback, 10)
        self.chessboard_pose_side_sub = self.create_subscription(ChessboardImgPose, '/chessboard/side/ImgPose', self.chessboard_pose_side_callback, 10)

        self.bridge = CvBridge()  # Bridge between "CV (NumPy array)" <-> "ROS sensor_msgs/Image"

        self._valid = np.zeros((8, 8), dtype=np.uint8)
        self._chess = np.zeros((8, 8), dtype=np.uint8)
    def chessboard_pose_top_callback(self, img_pose):
        frame = self.bridge.imgmsg_to_cv2(img_pose.image, desired_encoding='passthrough')
        # self.get_logger().info(str(get_tile(img_pose)))
        try:
            CNNinputs_padded, angle_list = get_tile_ImgPose(img_pose)
            if True:
                vertical_images = []
                for x in range(8):
                    image_list_vertical = []
                    for y in range(7, -1, -1):
                        canvas = resize_and_pad(CNNinputs_padded[8 * y + x].copy(), size=100)
                        cv2.putText(canvas, str(round(angle_list[8 * y + x])), (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    color=(0, 0, 255))
                        image_list_vertical.append(
                            cv2.copyMakeBorder(canvas, 1, 1, 1, 1, cv2.BORDER_CONSTANT, None, (0, 255, 0)))
                    vertical_images.append(np.vstack(image_list_vertical))
                combined_images = np.hstack(vertical_images)
                cv2.imshow("All CNN inputs", combined_images)
        except: pass
        # print(time.time()-start)
        cv2.imshow("AAA", frame)
        cv2.waitKey(1)
    def chessboard_pose_side_callback(self, img_pose):
        pass

def main():
    rclpy.init()
    chessboard_classifier = ChessboardClassifier()
    rclpy.spin(chessboard_classifier)
    # chessboard_detector.destroy_subscription(chessboard_detector.camera_sub) # Not need camera after init pose
    rclpy.shutdown()

if __name__ == "__main__":
    main()