import os
import glob
import json
import numpy as np
import tensorflow as tf
import cv2

# Converting the values into features
def _int64_feature(value):  # _int64 is used for numeric values
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
def _bytes_feature(value):  # _bytes is used for string/char values
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

# Create a dictionary describing the features.
image_feature_description = {
    'height': tf.io.FixedLenFeature([], tf.int64),
    'width': tf.io.FixedLenFeature([], tf.int64),
    'depth': tf.io.FixedLenFeature([], tf.int64),
    'image': tf.io.FixedLenFeature([], tf.string),
    "rvec": tf.io.FixedLenFeature([3], tf.float32),
    "tvec": tf.io.FixedLenFeature([3], tf.float32),
    "camera_matrix": tf.io.FixedLenFeature([9], tf.float32),
    'dist': tf.io.FixedLenFeature([5], tf.float32)
}

def _parse_image_function(example_proto):
  # Parse the input tf.train.Example proto using the dictionary above.
  return tf.io.parse_single_example(example_proto, image_feature_description)

dataset_config = json.load(open(os.path.join('../../config/dataset_config.json')))
output_path = dataset_config['capture_path']
file_list = sorted(glob.glob(os.path.join(output_path, '*.tfrecords')))
for file_path in file_list:
    print(file_path)
    dataset = tf.data.TFRecordDataset(file_path)
    parsed_image_dataset = dataset.map(_parse_image_function)
    # print(parsed_image_dataset)
    for image_features in parsed_image_dataset:
        height = image_features['height'].numpy()
        width = image_features['width'].numpy()
        depth = image_features['depth'].numpy()
        image = tf.io.decode_png(image_features['image'])   # Auto detect image shape when decoded
        image = np.array(image, dtype=np.uint8)
        rvec = image_features['rvec'].numpy()
        tvec = image_features['tvec'].numpy()
        cameraMatrix = image_features['camera_matrix'].numpy().reshape((3, 3))
        dist = image_features['dist'].numpy()
        canvas = image.copy()
        cv2.aruco.drawAxis(image=canvas,
                           cameraMatrix=cameraMatrix,
                           distCoeffs=dist,
                           rvec=rvec,
                           tvec=tvec,
                           length=0.1)
        cv2.imshow("A", canvas)
        key = cv2.waitKey(0)
        if key == ord('n'): break