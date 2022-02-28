#!/usr/bin/env python3

import cv2
import numpy as np
import imutils

from geometry_msgs.msg import Pose
from std_msgs.msg import UInt16MultiArray
from sensor_msgs.msg import CameraInfo, Image
from cv_bridge import CvBridge

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from isaac_ros_nvengine_interfaces.msg import TensorList

kMapPeakThreshhold = 0.01
kInputMapsRow = 60
kInputMapsColumn = 80
kInputMapsChannels = 25
kGaussianSigma = 3.0
kMinimumWeightSum = 1e-6
kOffsetDueToUpsampling = 0.4395

colors = [(255, 255, 255), (0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0)]

# NumPy     return (row, col) = (y, x)
# OpenCV    return (x, y) = (col, row)

def IsolateMaxima(img):
    mask = cv2.dilate(img, kernel=np.ones((3, 3)))
    mask = cv2.compare(src1=img, src2=mask, cmpop=cv2.CMP_GE) # CMP_GE: src1 is greater than or equal to src2
    # non_plateau_mask = np.zeros((60, 40, 1), dtype=np.float32)
    non_plateau_mask = cv2.erode(src=img,kernel=np.ones((3, 3)))
    non_plateau_mask = cv2.compare(src1=img, src2=non_plateau_mask, cmpop=cv2.CMP_GE) # CMP_GE: src1 is greater than or equal to src2
    cv2.bitwise_and(mask, non_plateau_mask, mask=mask)
    return mask

def FindPeaks(img, threshold=kMapPeakThreshhold):
    mask = IsolateMaxima(img)
    maxima = cv2.findNonZero(mask)  # OpenCV return (col, row)
    peaks = []
    for i in range(len(maxima)):
        [x, y] = maxima[0][0]
        if img[y, x] > threshold:   # NumPy use [row, col]
            peaks.append(maxima[i])
    return peaks
def FindObjects(maps):  # maps: Array of belief map (80, 60) np.float32
    all_peaks = []      # Vector2(x, y): location of peak   z: belief map value
    channel_peaks = []  # int[kNumVertexChannel]
    for chan in range(len(maps)):
        channel_peaks_buffer = []
        image = maps[chan]  # size = (kInputMapsRow, kInputMapsColumn) dtype=np.float32
        blurred = cv2.GaussianBlur(src=image,
                                   ksize=(0, 0),
                                   sigmaX=kGaussianSigma,
                                   sigmaY=kGaussianSigma,
                                   borderType=cv2.BORDER_DEFAULT)
        peaks = FindPeaks(blurred)
        for pp in range(len(peaks)):
            peak = peaks[pp][0]    # Peak pixel
# Compute the weighted average for localizing the peak, using a 5x5 window
#           ███████████████
#           ███████████████
#           ██████░░░██████
#           ███████████████
#           ███████████████
            peak_sum = [0, 0]
            weight_sum = 0
            for xx in range(-2, 3):
                for yy in range(-2, 3):
                    row = peak[0] + xx
                    col = peak[1] + yy
                    if col < 0 or col >= image.shape[1] or row < 0 or row >= image.shape[0]: continue
                    weight = image[row, col]
                    weight_sum += weight
                    peak_sum[0] += row * weight
                    peak_sum[1] += col * weight
            if image[peak[1], peak[0]] >= kMapPeakThreshhold:
                # channel_peaks.append()
                if weight_sum < kMinimumWeightSum:
                    channel_peaks_buffer.append((peak[0] + kOffsetDueToUpsampling,
                                                 peak[1] + kOffsetDueToUpsampling))
                else:
                    channel_peaks_buffer.append((peak_sum[0]/weight_sum + kOffsetDueToUpsampling,
                                                 peak_sum[1]/weight_sum + kOffsetDueToUpsampling))
        channel_peaks.append(channel_peaks_buffer)
    return channel_peaks
def normalize8(I):
  mn = I.min()
  mx = I.max()

  mx -= mn

  I = ((I - mn)/mx) * 255
  return I.astype(np.uint8)
class ChessboardDecoder(Node):
    def __init__(self):
        super().__init__('chessboard_dope_decoder')
        self.tensor_sub = self.create_subscription(TensorList, '/tensor_sub', self.tensor_listener_callback, 10)
        self.frame_sub = self.create_subscription(Image, '/image', self.image_listener_callback, 10)
        self.frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.bridge = CvBridge()
        self.get_logger().info('Node started ...')
    def image_listener_callback(self, data):
        self.frame = self.bridge.imgmsg_to_cv2(data)
    def tensor_listener_callback(self, tensor_list):
        tensor = tensor_list.tensors[0]
        # tensor_name = tensor.name           # output
        # tensor_shape = tensor.shape         # "isaac_ros_nvengine_interfaces.msg.TensorShape(rank=4, dims=[1, 25, 60, 80])"
        # tensor_data_type = tensor.data_type # 9: float32
        # tensor_strides = tensor.strides     # uint64[]  "array('Q', [480000, 19200, 320, 4])"
        tensor_data = tensor.data           # uint8[480000]
        # self.get_logger().info('Tensor Name: "%s"' % tensor_name)
        # self.get_logger().info('Tensor Shape: "%s"' % tensor_shape)
        # self.get_logger().info('Data Type: "%s"' % tensor_data_type)
        # self.get_logger().info('Tensor Stride: "%s"' % tensor_strides)
        # self.get_logger().info('Tensor Data: "%s"' % tensor_data)
        # self.get_logger().info('Tensor Data: "%d"' % len(tensor_data))


        # Convert uint8[4] -> float32 [ALL]
        # tensor_np = np.array(tensor_data).view(dtype=np.float32).reshape((25, 60, 80, 1))

        # Convert uint8[4] -> float32 [4 CORNERS]
#       ░ = Black, █ = White
#     1 ░░░░░░░░░░░░░░░░░░░░░░░░ 5
#       ░░░███░░░███░░░███░░░███
#       ███░░░███░░░███░░░███░░░
#       ░░░███░░░███░░░███░░░███
#       ███░░░███░░░███░░░███░░░
#       ░░░███░░░███░░░███░░░███
#       ███░░░███░░░███░░░███░░░
#       ░░░███░░░███░░░███░░░███
#       ███░░░███░░░███░░░███░░░
#     2 ████████████████████████ 6
        maps = []
        # tensor_data = np.array(tensor_data).reshape((4, 25*60*80))
        # tensor_data = np.swapaxes(tensor_data, 0, 1)
        # tensor_data = tensor_data.flatten()
        # self.get_logger().info('Tensor shape: "%s"' % str(tensor_data.shape))
        for i in [1, 2, 5, 6]:
        # for i in range(9):
            stride = kInputMapsRow * kInputMapsColumn * 4   # 4 = sizeof(float)
            offset = stride * i
            # Slice tensor & convert uint8[4] -> float32
            maps.append(np.array(tensor_data[offset:offset+stride]).view('<f4').reshape((kInputMapsRow, kInputMapsColumn)))
        objs = FindObjects(maps)
        # self.get_logger().info('Object: "%s"' % str(objs))
        self.get_logger().info('Object: "%s"' % str([len(obj) for obj in objs]))
        canvas = self.frame.copy()
        self.get_logger().info('Canvas shape: "%s"' % str(canvas.shape))
        for i in range(4):
            peak_list = objs[i]
            for point in peak_list:
                cv2.circle(canvas, (int(point[0]*8), int(point[1]*8)), 3, colors[i%len(colors)], -1)
        cv2.imshow("A", canvas)
        canvas1 = np.hstack([maps[0], maps[1]])
        canvas2 = np.hstack([maps[2], maps[3]])
        canvas = np.vstack([canvas1, canvas2])
        cv2.imshow("B", imutils.resize(normalize8(canvas), height=480*2))
        cv2.waitKey(1)
        # peaks = []
        # for i in [1, 2, 5, 6]:
        #     peaks.append(FindPeaks(tensor_np[i]))
        #     self.get_logger().info('Peak %d: "%s"' % (i, str(peaks[-1])))

        # canvas = np.array(canvas, dtype=np.uint8)
        # self.get_logger().info('Canvas shape: "%s"' % str(canvas.shape))
        # self.get_logger().info('Canvas type: "%s"' % str(type(canvas[0][0][0])))
        # cv2.imshow("A", imutils.resize(canvas, height=480))
        # cv2.waitKey(1)

def main():
    rclpy.init()
    chessboard_decoder = ChessboardDecoder()
    rclpy.spin(chessboard_decoder)
    rclpy.shutdown()


if __name__ == '__main__':
    main()