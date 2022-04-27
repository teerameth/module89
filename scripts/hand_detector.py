#!/usr/bin/env /home/teera/.virtualenvs/cv/bin/python
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool

import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

class HandDetector(Node):
    def __init__(self):
        super().__init__('hand_detector')
        self.hands = mp_hands.Hands(
                        model_complexity=0,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5)

        self.camera0_sub = self.create_subscription(Image, '/camera0/image', self.camera0_listener_callback, 10)
        self.camera1_sub = self.create_subscription(Image, '/camera1/image', self.camera1_listener_callback, 10)

        self.camera0_hand_pub = self.create_publisher(Bool, '/camera0/hand', 10)
        self.camera1_hand_pub = self.create_publisher(Bool, '/camera1/hand', 10)

        self.bridge = CvBridge()  # Bridge between "CV (NumPy array)" <-> "ROS sensor_msgs/Image"

    def camera0_listener_callback(self, image):
        frame = self.bridge.imgmsg_to_cv2(image, desired_encoding='passthrough')
        results = self.hands.process(frame)
        hand_in_frame = Bool()
        if results.multi_hand_landmarks: hand_in_frame.data = False
        else: hand_in_frame.data = True
        self.camera0_hand_pub.publish(hand_in_frame)
    def camera1_listener_callback(self, image):
        frame = self.bridge.imgmsg_to_cv2(image, desired_encoding='passthrough')
        results = self.hands.process(frame)
        hand_in_frame = Bool()
        if results.multi_hand_landmarks: hand_in_frame.data = False
        else: hand_in_frame.data = True
        self.camera1_hand_pub.publish(hand_in_frame)