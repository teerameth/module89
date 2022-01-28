#!/usr/bin/env /home/teera/.virtualenvs/cv/bin/python

import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from ament_index_python.packages import get_package_share_directory
from module89.srv import FindChessboardPose

import cv2
import math
import simplejson

import os
import numpy as np

config = simplejson.load(open(os.path.join(get_package_share_directory('module89'), 'config', 'camera_config.json')))
cameraMatrix = np.array(config['camera_matrix'], np.float32)
dist = np.array(config['dist'])

class ChessboardDetect(Node):
    def __init__(self):
        super().__init__('chessboard_detector')
        self.camera_sub = self.create_subscription(Image, '/camera1/image', self.image_listener_callback, 10)
        self.chessboard_encoder = self.create_subscription(Float32, '/chessboard/encoder', self.chessboard_rotation, 10)
        self.bridge = CvBridge()
        self.chessboard_init_rot = None
        self.chessboard_init_pose = None
        self.chessboard_rot = None
        self.chessboard_pose = None
        self.frame = None
        self.publisher_chessboard = self.create_publisher(Image, '/chessboard/pose', 10)

    def image_listener_callback(self, image):
        self.frame = self.bridge.imgmsg_to_cv2(image, desired_encoding='passthrough')
        if self.frame is not None:
            canvas = self.frame.copy()
            if self.chessboard_rot is not None: cv2.putText(canvas, "%.2f"%self.chessboard_rot.data, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.imshow("A", canvas)
            key = cv2.waitKey(1)
            if key == ord(' '): # capture initial pose
                # self.chessboard_init_pose = find_chessboard(self.frame)
                canvas = self.frame.copy()
            elif key == ord('c'): # capture
                cv2.imwrite('/home/teera/data_test/capture.png', self.frame)

    def chessboard_rotation(self, encoder):
        self.chessboard_rot = encoder
        if self.chessboard_init_pose is not None and self.chessboard_init_rot is not None:
            rot = self.chessboard_rot - self.chessboard_init_rot
            if rot < 0: rot += math.pi
            # Rotate initial pose around Z-axis
            self.publisher_chessboard.publish(self.pose)
def main():
    rclpy.init()
    chessboard_detector = ChessboardDetect()
    rclpy.spin(chessboard_detector)

    # chessboard_detector.destroy_subscription(chessboard_detector.camera_sub) # Not need camera after init pose
    rclpy.shutdown()

if __name__ == "__main__":
    main()