#!/usr/bin/env /home/teera/.virtualenvs/cv/bin/python

import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Pose, Point, Quaternion
from sensor_msgs.msg import CameraInfo, Image
from module89.srv import ChessboardPose

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
        # Create chessboard_locator client
        self.chessboard_locator_cli = self.create_client(ChessboardPose, 'chessboard_locator')
        while not self.chessboard_locator_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service \'ChessboardPose\' not available, waiting ...')
        self.get_logger().info('Service \'ChessboardPose\' founded')

        # Create camera_info, camera_image, chessboard_encoder subscriber
        self.camera_info_sub = self.create_subscription(CameraInfo, '/camera_info', self.camera_info_listener, 10)
        self.camera_sub = self.create_subscription(Image, '/image', self.image_listener_callback, 10)
        self.encoder_sub = self.create_subscription(Float32, '/chessboard/encoder', self.chessboard_rotation_callback, 10)

        self.bridge = CvBridge()    # Bridge between "CV (NumPy array)" <-> "ROS sensor_msgs/Image"
        self.chessboard_init_encoder, self.chessboard_init_pose = None, None    # Pair of encoder & pose used for reference (have same timestamp)
        self.chessboard_encoder, self.chessboard_pose = None, None  # encoder & pose in real-time (independent)
        self.frame = None
        self.camera_matrix = None
        self.publisher_chessboard = self.create_publisher(Pose, '/chessboard/pose', 10)

        self.req = ChessboardPose.Request()

        self.candidate = {'image': None, 'encoder': None}
        self.future = []    # To store service call async process
        self._track = False # Tracking mode indicator
    def image_listener_callback(self, image):
        self.frame = self.bridge.imgmsg_to_cv2(image, desired_encoding='passthrough')

        if self.frame is not None:
            canvas = self.frame.copy()
            if self.chessboard_encoder is not None: cv2.putText(canvas, "%.2f"%self.chessboard_encoder, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.imshow("A", canvas)
            key = cv2.waitKey(1)
            if key == ord(' '):     # Switch between track & capture mode (track mode is available only when chessboard was founded)
                if self.chessboard_init_pose is not None: self._track = not self._track
            if not self._track:     # If not tracking
                # Pair image & encoder with same timestamp together
                self.candidate['image'] = image # sensor_msgs/Image
                self.candidate['encoder'] = self.chessboard_encoder
                self.req.img = image

                if len(self.future) == 0:   # If no service call pending -< then call it
                    self.future.append(self.chessboard_locator_cli.call_async(self.req))    # Append to service call queue
                else:
                    if self.future[0].done():   # Check if result returned from service
                        try:
                            response = self.future.pop().result()   # take out finished process and get result
                            self.get_logger().info(str(response))
                        except Exception as e:
                            self.get_logger().info('Service call failed %r' % (e,))
                        else:  # After finishing service call without error
                            if response.valid:  # If chessboard founded -> update reference (pose, encoder) pair
                                self.chessboard_init_pose = response.init_pose
                                self.chessboard_init_encoder = self.candidate['encoder']

            elif key == ord('c'): # capture
                cv2.imwrite('/home/teera/data_test/capture.png', self.frame)

    def chessboard_rotation_callback(self, encoder):
        self.chessboard_encoder = encoder.data
        if self.chessboard_init_pose is not None and self.chessboard_init_encoder is not None:
            rot = self.chessboard_encoder - self.chessboard_init_encoder
            if rot < 0: rot += math.pi
            # Rotate initial pose around Z-axis
            self.pose = Pose()
            self.pose.position = Point()
            self.pose.orientation = Quaternion()
            self.publisher_chessboard.publish(self.pose)
    def camera_info_listener(self, camera_info):
        self.camera_matrix = camera_info.K

    # def send_request(self):
    #     self.future = self.chessboard_locator_cli.call_async(self.req)

def main():
    rclpy.init()
    chessboard_detector = ChessboardDetect()
    rclpy.spin(chessboard_detector)
    # chessboard_detector.destroy_subscription(chessboard_detector.camera_sub) # Not need camera after init pose
    rclpy.shutdown()

if __name__ == "__main__":
    main()