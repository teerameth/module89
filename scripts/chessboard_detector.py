#!/usr/bin/env /home/teera/.virtualenvs/tf/bin/python

import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from ament_index_python.packages import get_package_share_directory


import os
import numpy as np
import simplejson
import geometry
import collections
import scipy, scipy.cluster
from keras.models import model_from_json
from transform import four_point_transform, order_points, poly2view_angle
import itertools, sklearn.cluster, pyclipper, matplotlib.path

__laps_model = '../models/laps.model.json'
__laps_weights = '../models/laps.weights.h5'
config = simplejson.load(open(os.path.join(get_package_share_directory('module67'), 'config', 'camera_config.json')))
cameraMatrix = np.array(config['camera_matrix'], np.float32)
dist = np.array(config['dist'])
resolution_x, resolution_y = config['width'], config['height']