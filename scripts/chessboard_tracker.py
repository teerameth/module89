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
from module89.srv import PoseLock

import cv2
import imutils
import math
import simplejson
import os
import numpy as np
import onnxruntime as ort
from scipy.spatial.transform import Rotation as R

def FindMax(maps):
    max_points, max_vals = [], []
    for m in maps:
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(cv2.GaussianBlur(m, (3, 3), 0))
        max_points.append(max_loc)
        max_vals.append(max_val)
    return max_points, max_vals

config = simplejson.load(open(os.path.join(get_package_share_directory('module89'), 'config', 'camera_config.json')))
cameraMatrix = np.array(config['camera_matrix'], np.float32)
dist = np.array(config['dist'])
frame_buffer_length = 10

confidence_treshold = 0.03
obj_points = np.array([[-0.2, 0.23, 0], [-0.2, -0.23, 0], [0.2, 0.23, 0], [0.2, -0.23, 0]])

model_config = simplejson.load(open(os.path.join(get_package_share_directory('module89'), 'config', 'model_config.json')))
model_path = model_config['dope']
# model_path = os.path.join(get_package_share_directory('module89'), 'models', 'chessboard.onnx')
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
def pose2view_angle(rvec, tvec):
    rotM = np.zeros(shape=(3, 3))
    rotM, _ = cv2.Rodrigues(rvec, rotM, jacobian=0)
    tvec_tile_final = np.dot(tvec, rotM.T).reshape(3)
    tile_x, tile_y, tile_z = tvec_tile_final[0], tvec_tile_final[1], tvec_tile_final[2]
    angle_rad = math.asin((math.sqrt(tile_x ** 2 + tile_y ** 2)) / (math.sqrt(tile_x ** 2 + tile_y ** 2 + tile_z ** 2)))
    return angle_rad
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
        self.tracker_viz_pub = self.create_publisher(Image, '/viz/pose_estimation', 10)

        self.lock_srv = self.create_service(PoseLock, 'command_pose_lock', self.pose_lock_callback)

        self.bridge = CvBridge()  # Bridge between "CV (NumPy array)" <-> "ROS sensor_msgs/Image"
        self.chessboard_init_encoder, self.chessboard_init_pose = None, None  # Pair of encoder & pose used for reference (have same timestamp)
        self.chessboard_encoder, self.chessboard_pose = None, None  # encoder & pose in real-time (independent)

        ## Create timer to handle pipeline feeding
        self.timer = self.create_timer(0.05, self.timer_callback)   # 20 Hz

        self.camera_lock = [False, False]
        self.camera_lock_pose = [None, None]
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
    #       ░░░ = Black, ███ = White
    #     0 ░░░░░░░░░░░░░░░░░░░░░░░░ 2
    #       ░░░███░░░███░░░███░░░███
    #       ███░░░███░░░███░░░███░░░
    #       ░░░███░░░███░░░███░░░███
    #       ███░░░███░░░███░░░███░░░
    #       ░░░███░░░███░░░███░░░███
    #       ███░░░███░░░███░░░███░░░
    #       ░░░███░░░███░░░███░░░███
    #       ███░░░███░░░███░░░███░░░
    #     1 ████████████████████████ 3
    def timer_callback(self):
        global obj_points
        # Wait until both cameras ready
        # print(len(self.image_buffer['camera0']), len(self.image_buffer['camera1']))
        if len(self.image_buffer['camera0']) > 0 and len(self.image_buffer['camera1']) > 0:
            images = [self.image_buffer['camera0'][-1], self.image_buffer['camera1'][-1]]
            self.image_buffer['camera0'], self.image_buffer['camera1'] = [], [] # reset buffer
            canvas_image_list, canvas_belief_list, canvas_pose_list = [], [], []
            for i in range(2):
                image = images[i]
                canvas_pose = image.copy()
                canvas_pose_list.append(canvas_pose)
                canvas_image_list.append(image.copy())
                img_pose_msg = None
                if self.camera_lock[i]: # if camera pose is locked
                    (rvec, tvec) = self.camera_lock_pose[i] # Use stored pose
                    print(rvec.shape, tvec.shape)
                    img_pose_msg = ChessboardImgPose()
                    img_pose_msg.pose = preparePose(rvec, tvec)
                    angle = pose2view_angle(rvec.reshape((1, 3)), tvec.reshape((1, 3)))
                    img_pose_msg.image = self.bridge.cv2_to_imgmsg(image, "bgr8")
                    canvas_belief_list.append(np.zeros((480, 640, 3), dtype=np.uint8))
                else:   # Real-time detection (not locked)
                    x = NHWC2NCHW(image)
                    outputs = ort_sess.run(None, {'input': x})  # outputs.shape = (1, 4, 60, 80)
                    outputs = outputs[0][0]
                    overlay = np.zeros(outputs[0].shape, dtype=np.float32)
                    for j in range(4): overlay += outputs[i]
                    overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)
                    overlay = np.array(overlay * 255, dtype=np.uint8)
                    overlay = imutils.resize(overlay, height=image.shape[0])
                    canvas_belief = cv2.addWeighted(image, 0.3, overlay, 0.7, 0)
                    canvas_belief_list.append(canvas_belief)
                    points, vals = FindMax(outputs)
                    confidences = [False if val < confidence_treshold else True for val in vals]
                    for j in range(4): cv2.circle(canvas_pose, (points[i][0] * 8, points[i][1] * 8), 3, (255, 0, 0), -1)
                    if not False in confidences:    # Confident to estimate pose
                        img_points = np.array([points[0], points[1], points[2], points[3]], dtype=np.double) * 8
                        ret, rvec, tvec = cv2.solvePnP(objectPoints=obj_points,
                                                       imagePoints=img_points,
                                                       cameraMatrix=cameraMatrix,
                                                       distCoeffs=dist,
                                                       flags=0)
                        img_pose_msg = ChessboardImgPose()
                        img_pose_msg.pose = preparePose(rvec, tvec)
                        angle = pose2view_angle(rvec.reshape((1, 3)), tvec.reshape((1, 3)))
                        img_pose_msg.image = self.bridge.cv2_to_imgmsg(image, "bgr8")
                        # Store in buffer (for locking purpose)
                        self.camera_lock_pose[i] = (rvec, tvec)
                    else:   # Unable to estimate pose
                        pass
                # select topic to publish by view angle
                if img_pose_msg is not None:
                    cv2.rectangle(canvas_pose, (0, 0), (200, 80), (255, 255, 255), -1)
                    absolute_distance = sum([tvec[k]**2 for k in range(len(tvec))])
                    if i == 0:  # angle 0.00, tvec 1.26
                        color_angle = (0, 255, 0) if abs(angle) < 0.05 else (0, 0, 255)
                        color_distance = (0, 255, 0) if abs(absolute_distance-1.26) < 0.05 else (0, 0, 255)
                        cv2.putText(canvas_pose, "%.2f/0.00" % (angle), (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color_angle, 2, cv2.LINE_AA)
                        cv2.putText(canvas_pose, "%.2f/1.26" % (absolute_distance), (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, color_distance, 2, cv2.LINE_AA)
                    if i == 1:  # angle 0.35, tvec 0.70
                        color_angle = (0, 255, 0) if abs(angle - 0.35) < 0.05 else (0, 0, 255)
                        color_distance = (0, 255, 0) if abs(absolute_distance-0.7) < 0.05 else (0, 0, 255)
                        cv2.putText(canvas_pose, "%.2f/0.35" % (angle), (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color_angle, 2, cv2.LINE_AA)
                        cv2.putText(canvas_pose, "%.2f/0.70" % (absolute_distance), (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, color_distance, 2, cv2.LINE_AA)
                    pose_pub = self.top_pose_pub if angle < 0.2 else self.side_pose_pub
                    pose_pub.publish(img_pose_msg)
                    cv2.aruco.drawAxis(image=canvas_pose,
                                       cameraMatrix=cameraMatrix,
                                       distCoeffs=dist,
                                       rvec=rvec,
                                       tvec=tvec,
                                       length=0.1)
            canvas_image_list = cv2.hconcat(canvas_image_list)
            canvas_belief_list = cv2.hconcat(canvas_belief_list)
            canvas_pose_list = cv2.hconcat(canvas_pose_list)
            canvas = cv2.vconcat([canvas_image_list, canvas_belief_list, canvas_pose_list])
        # else:
        #     canvas_image_list = []
        #     if len(self.image_buffer['camera0']) > 0: canvas_image_list.append(self.image_buffer['camera0'][-1])
        #     else:canvas_image_list.append(np.zeros((480, 640, 3), dtype=np.uint8))
        #     if len(self.image_buffer['camera1']) > 0: canvas_image_list.append(self.image_buffer['camera1'][-1])
        #     else: canvas_image_list.append(np.zeros((480, 640, 3), dtype=np.uint8))
        #     canvas = cv2.hconcat(canvas_image_list)
        #     canvas = cv2.vconcat([canvas, np.zeros((480*2, 640*2, 3), dtype=np.uint8)])
            # Publish viz image
            canvas = imutils.resize(canvas, height=900)
            # canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)    # Unity
            image_msg = self.bridge.cv2_to_imgmsg(canvas, "bgr8")
            image_msg.header.stamp = self.get_clock().now().to_msg()
            self.tracker_viz_pub.publish(image_msg)  # Publish image
            # cv2.imshow("Pose Estimation", imutils.resize(canvas, height=900))
            # cv2.waitKey(1)

    def pose_lock_callback(self, request, response):
        mode = request.mode # 1/2 = init from 4 points, 3/4 = init from dope
        if mode == 0:
            self.camera_lock[0] = False
            self.camera_lock[1] = False
            response.acknowledge = 1
        if mode == 1 or mode == 2:
            img_points = request.corners.data
            ret, rvec, tvec = cv2.solvePnP(objectPoints=obj_points,
                                           imagePoints=img_points,
                                           cameraMatrix=cameraMatrix,
                                           distCoeffs=dist,
                                           flags=0)
            if ret == False:
                response.acknowledge = 0
                return response
            response.acknowledge = 1
            if mode == 1: self.camera_lock_pose[0] = (rvec, tvec)
            if mode == 2: self.camera_lock_pose[1] = (rvec, tvec)

        elif mode == 3 or mode == 4:
            if mode == 3:
                if self.camera_lock[0]: # Lock -> Unlock
                    self.camera_lock[0] = False
                else:   # Unlock -> Lock
                    if self.camera_lock_pose[0] is None:    # Unable to lock (No stored pose in buffer)
                        self.get_logger().warn("Unable to lock pose from /camera0 (No stored pose in buffer)")
                        response.acknowledge = 0
                        return response
                    else: self.camera_lock[0] = True
            if mode == 4:
                if self.camera_lock[1]: # Lock -> Unlock
                    self.camera_lock[1] = False
                else:
                    if self.camera_lock_pose[1] is None:    # Unable to lock (No stored pose in buffer)
                        self.get_logger().warn("Unable to lock pose from /camera1 (No stored pose in buffer)")
                        response.acknowledge = 0
                        return response
                    else: self.camera_lock[1] = True
            response.acknowledge = 1
        return response
def main():
    rclpy.init()
    chessboard_detector = ChessboardTracker()
    rclpy.spin(chessboard_detector)
    # chessboard_detector.destroy_subscription(chessboard_detector.camera_sub) # Not need camera after init pose
    rclpy.shutdown()

if __name__ == "__main__":
    main()