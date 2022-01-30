#!/usr/bin/env /home/teera/.virtualenvs/tf/bin/python
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from ament_index_python.packages import get_package_share_directory
from module89.srv import ChessboardDetection


import os
import simplejson
import numpy as np

class ChessboardDetectorService(Node):
    def __init__(self):
        super().__init__('chessboard_detector_fake')
        fake_data = simplejson.load(open(os.path.join(get_package_share_directory('module89'), 'models', '00001.json')))
        x2, y2 = fake_data['objects'][0]['bounding_box']['bottom_right']
        x1, y1 = fake_data['objects'][0]['bounding_box']['top_left']
        height = int(fake_data['camera_data']['height'])
        width = int(fake_data['camera_data']['width'])
        if x1 < 0: x1 = 0
        if x2 >= width: x2 = width-1
        if y1 < 0: y1 = 0
        if y2 >= height: y2 = height-1
        self.fake_data = [int(x1), int(x2), int(y1), int(y2)]
        print(self.fake_data)
        self.srv = self.create_service(ChessboardDetection, 'chessboard_detection', self.detect_callback)
    def detect_callback(self, request, response):
        print(type(response.bbox))
        response.bbox = np.array(self.fake_data, dtype=np.uint16)
        return response.bbox

def main():
    rclpy.init()
    detection_service = ChessboardDetectorService()
    rclpy.spin(detection_service)
    rclpy.shutdown()

if __name__ == '__main__':
    main()