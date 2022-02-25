#!/usr/bin/env /home/teera/.virtualenvs/cv/bin/python

import cv2
import numpy as np

from geometry_msgs.msg import Pose
from std_msgs.msg import UInt16MultiArray
from cv_bridge import CvBridge

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor

def IsolateMaxima(img):
    mask = cv2.dilate(img, kernel=np.one((3, 3)))
    cv2.compare(src1=img, src2=mask, dst=(3, 3), cmpop=cv2.CMP_GE) # CMP_GE: src1 is greater than or equal to src2
    non_plateau_mask = np.zeros((60, 40))
    cv2.erode(src=img, dst=non_plateau_mask, kernel=np.ones((3, 3)))
    cv2.compare(img, non_plateau_mask, cmpop=np.ones((3, 3)))
    cv2.bitwise_and(mask, non_plateau_mask, mask=mask)
    return mask

def FindPeaks(img, threshold):
    mask = IsolateMaxima(img)
    maxima = cv2.findNonZero(mask)
    peaks = []
    for i in range(len(maxima)):
        if img[maxima[i].y, maxima[i].x] > threshold:
            peaks.append(maxima[i])
    return peaks
