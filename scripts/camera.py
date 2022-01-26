#!/usr/bin/env /home/teera/.virtualenvs/cv/bin/python

import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from ament_index_python.packages import get_package_share_directory

import cv2
import v4l2capture
import select
import os
import numpy as np
import subprocess
import simplejson

class Camera():
    def __init__(self, i=0, width=1920, height=1080):
        if os.name == 'nt':
            self.cap = cv2.VideoCapture(cv2.CAP_DSHOW+i)
            self.cap.set(cv2.CAP_PROP_SETTINGS, 1) # Open setup window
            # codec = 0x47504A4D  # MJPG
            self.cap.set(cv2.CAP_PROP_FPS, 30.0)
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
            self.cap.set(cv2.CAP_PROP_AUTO_WB, 0)
            self.cap.set(cv2.CAP_PROP_FOCUS, 2)
            self.cap.set(3, width)
            self.cap.set(4, height)
        else:
            self.cap = v4l2capture.Video_device("/dev/video" + str(i))
            size_x, size_y = self.cap.set_format(width, height, fourcc='MJPG')
            self.cap.set_fps(30)
            devnull = open(os.devnull, 'w')  # For no output
            subprocess.call(['v4l2-ctl', '--set-ctrl', 'power_line_frequency=1'], stdout=devnull, stderr=devnull)
            self.cap.set_focus_auto(0)
            self.cap.set_exposure_auto(3)
            # # self.cap.set_exposure_absolute(250)
            self.cap.set_auto_white_balance(0)
            # # self.cap.set_white_balance_temperature(2500)
            self.cap.create_buffers(1)  # Create a buffer to store image data before calling 'start'
            self.cap.queue_all_buffers()# Send the buffer to the device. Some devices require this to be done before calling 'start'.
            self.cap.start()

    def read(self):
        if os.name == 'nt':
            return self.cap.read()[1]
        else:
            # Wait for the device to fill buffer
            select.select((self.cap,), (), ())
            image_data = self.cap.read_and_queue()
            return cv2.imdecode(np.frombuffer(image_data, dtype=np.uint8), cv2.IMREAD_COLOR)    # Decode & return


class CameraPublisher(Node):
    def __init__(self):
        super().__init__('camera_publisher')
        self.publisher_ = self.create_publisher(Image, '/camera1/image', 10)
        timer_period = 1/30  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        config = simplejson.load(open(os.path.join(get_package_share_directory('module67'), 'config', 'camera_config.json')))
        print(config)
        self.cap = Camera(0, width=config['width'], height=config['height'])
        self.frame = self.cap.read()
        self.bridge = CvBridge()

    def timer_callback(self):
        self.frame = self.cap.read()
        image_msg = self.bridge.cv2_to_imgmsg(self.frame, "bgr8")
        self.publisher_.publish(image_msg)
        self.get_logger().info('Publishing image')

def main():
    rclpy.init()
    camera_service = CameraPublisher()
    rclpy.spin(camera_service)

    camera_service.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()