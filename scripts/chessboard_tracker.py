#!/usr/bin/env /home/teera/.virtualenvs/cv/bin/python
import time

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
import math
import simplejson
import os
import numpy as np
import onnxruntime as ort
from scipy.spatial.transform import Rotation as R

def FindMax(maps):
    max_points = []
    for m in maps:
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(cv2.GaussianBlur(m, (3, 3), 0))
        max_points.append(max_loc)
    return max_points

config = simplejson.load(open(os.path.join(get_package_share_directory('module89'), 'config', 'camera_config.json')))
cameraMatrix = np.array(config['camera_matrix'], np.float32)
dist = np.array(config['dist'])
frame_buffer_length = 10

model_path = os.path.join(get_package_share_directory('module89'), 'models', 'chessboard.onnx')
# TensorrtExecutionProvider having the higher priority.
ort_sess = ort.InferenceSession(model_path, providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider'])

def NHWC2NCHW(img):
    # convert from NHWC to NCHW (batch N, channels C, height H, width W)
    x = img.copy()
    x = x.transpose([2, 0, 1])
    x = np.float32(x) / 255 # normalize
    x = np.expand_dims(x, 0)
    return x
def preparePose(rvec, tvec):
    pose_msg = Pose()
    pose_msg.position = Point(x=tvec[0][0],
                              y=tvec[1][0],
                              z=tvec[2][0])
    r = R.from_matrix(cv2.Rodrigues(rvec)[0])
    rvec = Quaternion()
    [rvec.x, rvec.y, rvec.z, rvec.w] = r.as_quat()
    pose_msg.orientation = rvec
    return pose_msg
class ChessboardTracker(Node):
    def __init__(self):
        super().__init__('chessboard_tracker')
        self.image_buffer = {'camera0':[], 'camera1':[]}  # image buffer
        # Create camera_image, chessboard_encoder subscriber
        self.camera0_sub = self.create_subscription(Image, '/camera0/image', self.camera0_listener_callback, 10)
        self.camera1_sub = self.create_subscription(Image, '/camera1/image', self.camera1_listener_callback, 10)
        self.encoder_sub = self.create_subscription(Float32, '/chessboard/encoder', self.chessboard_encoder_callback, 10)

        self.top_pose_pub = self.create_publisher(ChessboardImgPose, '/chessboard/top/ImgPose', 10)
        self.side_pose_pub = self.create_publisher(ChessboardImgPose, '/chessboard/side/ImgPose', 10)

        self.bridge = CvBridge()  # Bridge between "CV (NumPy array)" <-> "ROS sensor_msgs/Image"
        self.chessboard_init_encoder, self.chessboard_init_pose = None, None  # Pair of encoder & pose used for reference (have same timestamp)
        self.chessboard_encoder, self.chessboard_pose = None, None  # encoder & pose in real-time (independent)

        ## Create timer to handle pipeline feeding
        self.timer = self.create_timer(0.02, self.timer_callback)   # 50 Hz
        self._last_cam = 1  # to indicate lastest inference

    def camera0_listener_callback(self, image):
        self.top_frame = self.bridge.imgmsg_to_cv2(image, desired_encoding='passthrough')
        self.image_buffer['camera0'].append(self.bridge.imgmsg_to_cv2(image, desired_encoding='passthrough'))
        if len(self.image_buffer['camera0']) > frame_buffer_length: self.image_buffer['camera0'] = self.image_buffer['camera0'][-frame_buffer_length:]

    def camera1_listener_callback(self, image):
        self.side_frame = self.bridge.imgmsg_to_cv2(image, desired_encoding='passthrough')
        self.image_buffer['camera1'].append(self.bridge.imgmsg_to_cv2(image, desired_encoding='passthrough'))
        if len(self.image_buffer['camera1']) > frame_buffer_length: self.image_buffer['camera1'] = self.image_buffer['camera1'][-frame_buffer_length:]

    def chessboard_encoder_callback(self, encoder):
        pass

    def timer_callback(self):
        image_buffer = []
        # Switch camera if next camera is avaliable
        if self._last_cam == 0 and len(self.image_buffer['camera1']) > 0:
            self._last_cam = 1
            image_buffer = self.image_buffer['camera1']
        elif self._last_cam == 1 and len(self.image_buffer['camera0']) > 0:
            self._last_cam = 0
            image_buffer = self.image_buffer['camera0']

        if len(image_buffer) > 0:
            image = image_buffer[-1]
            image_buffer = []   # reset buffer
            x = NHWC2NCHW(image)
            outputs = ort_sess.run(None, {'input': x})
            points = FindMax(outputs[0][0])
#       ░░░ = Black, ███ = White
#     0 ░░░░░░░░░░░░░░░░░░░░░░░░ 2
#       ░░░███░░░███░░░███░░░███
#       ███░░░███░░░███░░░███░░░
#       ░░░███░░░███░░░███░░░███
#       ███░░░███░░4███░░░███░░░
#       ░░░███░░░███░░░███░░░███
#       ███░░░███░░░███░░░███░░░
#       ░░░███░░░███░░░███░░░███
#       ███░░░███░░░███░░░███░░░
#     1 ████████████████████████ 3
            obj_points = np.array([[-0.2, 0.23, 0], [-0.2, -0.23, 0], [0.2, 0.23, 0], [0.2, -0.23, 0]])
            img_points = np.array([points[0], points[1], points[2], points[3]], dtype=np.double) * 8
            ret, rvec, tvec = cv2.solvePnP(objectPoints=obj_points,
                                           imagePoints=img_points,
                                           cameraMatrix=cameraMatrix,
                                           distCoeffs=dist,
                                           flags=0)
            img_pose_msg = ChessboardImgPose()
            img_pose_msg.pose = preparePose(rvec, tvec)
            img_pose_msg.image = self.bridge.cv2_to_imgmsg(image, "bgr8")
            # select topic to publish by view angle
            if img_pose_msg.pose.orientation.w < 0.2:
                pose_pub = self.top_pose_pub
            else:
                pose_pub = self.side_pose_pub
            pose_pub.publish(img_pose_msg)

            # canvas = image.copy()
            # cv2.aruco.drawAxis(image=canvas,
            #                    cameraMatrix=cameraMatrix,
            #                    distCoeffs=dist,
            #                    rvec=rvec,
            #                    tvec=tvec,
            #                    length=0.1)
            # cv2.imshow('A', canvas)
            # cv2.waitKey(1)

        self._last_cam = 0 if self._last_cam == 1 else 1    # switch camera

def main():
    rclpy.init()
    chessboard_detector = ChessboardTracker()
    rclpy.spin(chessboard_detector)
    # chessboard_detector.destroy_subscription(chessboard_detector.camera_sub) # Not need camera after init pose
    rclpy.shutdown()

if __name__ == "__main__":
    main()