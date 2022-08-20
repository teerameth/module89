#!/usr/bin/env -S HOME=${HOME} ${HOME}/openvino_env/bin/python
from PyQt5 import QtGui, QtCore
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout, QPushButton
from PyQt5.QtGui import QPixmap
import sys
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread

import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Float32, Bool, Float32MultiArray
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Pose, Point, Quaternion
from sensor_msgs.msg import CameraInfo, Image
from module89.msg import ChessboardImgPose
from module89.srv import PoseLock

import cv2
import v4l2capture
import subprocess
import select
import imutils
import mediapipe as mp
import math
import simplejson
import os
import numpy as np
from scipy.spatial.transform import Rotation as R

class Camera():
    def __init__(self, i=0, width=1920, height=1080):
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

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    def run(self):
        cap = Camera(0)
        while True:
            cv_img = cap.read()
            self.change_pixmap_signal.emit(cv_img)

class App(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Corner calibration")
        self.disply_width = 640
        self.display_height = 480
        # create the label that holds the image
        self.image_label = QLabel(self)
        self.image_label.resize(self.disply_width, self.display_height)
        # create a text label
        self.textLabel = QLabel('camera0')
        # create buttons
        self.button_clear = QPushButton('Clear Points')
        self.button_clear.clicked.connect(self.clear_points)
        self.button_confirm = QPushButton('Confirm')
        self.button_confirm.clicked.connect(self.set_points)

        # create a vertical box layout and add the two labels
        vbox = QVBoxLayout()
        vbox.addWidget(self.image_label)
        vbox.addWidget(self.textLabel)
        vbox.addWidget(self.button_clear)
        vbox.addWidget(self.button_confirm)
        # set the vbox layout as the widgets layout
        self.setLayout(vbox)

        # create the video capture thread
        self.thread = VideoThread()
        # connect its signal to the update_image slot
        self.thread.change_pixmap_signal.connect(self.update_image)
        # start the thread
        self.thread.start()

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)
    @QtCore.pyqtSlot()
    def clear_points(self):
        self.four_points = []

    @QtCore.pyqtSlot()
    def set_points(self):
        print(self.four_points)
        self.four_points = []
    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.disply_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)


four_points = []
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

obj_points = np.array([[-0.2, -0.23, 0], [0.2, -0.23, 0], [0.2, 0.23, 0], [-0.2, 0.23, 0]])
model_config = simplejson.load(open(os.path.join(get_package_share_directory('module89'), 'config', 'model_config.json')))
model_path = os.path.join(model_config['base_path'], model_config['dope'])

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

def rotate(rvec, angle):
    rotM = np.zeros(shape=(3, 3))
    new_rvec = np.zeros(3)
    rotM, _ = cv2.Rodrigues(rvec, rotM, jacobian=0)  # Convert from rotation vector -> rotation matrix (rvec=rot_vector, rotM=rot_matrix)
    rotM = np.dot(rotM, np.array([[math.cos(angle), -math.sin(angle), 0], [math.sin(angle), math.cos(angle), 0], [0, 0, 1]]))  # Rotate zeta degree about Z-axis
    new_rvec, _ = cv2.Rodrigues(rotM, new_rvec, jacobian=0)  # Convert from rotation matrix -> rotation vector
    return new_rvec

class ChessboardEstimator(Node):
    def __init__(self):
        super().__init__('chessboard_estimator')
        # self.cap = Camera(0)
        self.robot_joint0_sub = self.create_subscription(Float32, '/chessboard/joint0', self.robot_joint0_callback, 10)
        self.hand_pub = self.create_publisher(Bool, '/camera0/hand', 10)
        self.bridge = CvBridge()  # Bridge between "CV (NumPy array)" <-> "ROS sensor_msgs/Image"
        self.timer = self.create_timer(1/30, self.timer_callback)   # 10 Hz
        self.hand_in_frame = False
        self.robot_in_frame = False
        # load calibration UI
        self.app = QApplication(sys.argv)
        self.a = App()
        self.a.show()
        self.app.exec_()

    def timer_callback(self):
        # frame = self.cap.read()
        print("AAA")
    def robot_joint0_callback(self, joint0):
        angle = joint0.data
        if abs(angle) < 0.75: self.robot_in_frame = True
        else: self.robot_in_frame = False
def main():
    rclpy.init()
    chessboard_estimator = ChessboardEstimator()
    rclpy.spin(chessboard_estimator)
    # sys.exit(ChessboardEstimator.app.exec_())
    rclpy.shutdown()

if __name__ == "__main__":
    main()