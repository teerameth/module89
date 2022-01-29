#!/usr/bin/env /home/teera/.virtualenvs/tf/bin/python
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from ament_index_python.packages import get_package_share_directory
from module89.srv import ChessboardDetection


import os
import simplejson

class ChessboardDetectorService(Node):
    def __init__(self):
        super().__init__('fake_chessboard_detector')
        fake_data = simplejson.load(open(os.path.join(get_package_share_directory('module89'), 'models', '00001.json')))
        x2, y2 = fake_data['objects'][0]['bounding_box']['bottom_right']
        x1, y1 = fake_data['objects'][0]['bounding_box']['top_left']
        self.bbox = [x1, x2, y1, y2]
        self.srv = self.create_service(ChessboardDetection, 'chessboard_detection', self.detect)
    def detect(self, request, response):
        response.bbox = self.bbox
        return self.bbox

def main():
    rclpy.init()
    detection_service = ChessboardDetectorService()
    rclpy.spin(detection_service)
    rclpy.shutdown()

if __name__ == '__main__':
    main()