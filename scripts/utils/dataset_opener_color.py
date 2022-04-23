import os
import glob
import json

import cv2
import imutils
import numpy as np
import tensorflow as tf

# Converting the values into features
def _int64_feature(value):  # _int64 is used for numeric values
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
def _bytes_feature(value):  # _bytes is used for string/char values
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


# Create a dictionary describing the features.
image_feature_description = {
    'image': tf.io.FixedLenFeature([], tf.string),
    'color': tf.io.FixedLenFeature([], tf.int64)
}

def _parse_image_function(example_proto):
  # Parse the input tf.train.Example proto using the dictionary above.
  return tf.io.parse_single_example(example_proto, image_feature_description)

colorNames = ["black", "gold", "green", "pink", "silver", "wood", "yellow"]

dataset_config = json.load(open(os.path.join('../../config/dataset_config.json')))
output_path = dataset_config['capture_path']
color_file = os.path.join(output_path, 'color.tfrecords')

dataset = tf.data.TFRecordDataset(color_file)
parsed_image_dataset = dataset.map(_parse_image_function)
# print(parsed_image_dataset)
canvas_dict = {}
for image_features in parsed_image_dataset:
    color_index = image_features['color'].numpy()
    image = tf.io.decode_png(image_features['image'])   # Auto detect image shape when decoded
    image = np.array(image, dtype=np.uint8)
    color = colorNames[color_index]
    print(image.shape)
    cv2.putText(image, colorNames[color_index], (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color=(0, 0, 255))
    if color not in canvas_dict.keys(): canvas_dict[color] = [] # Register new color
    canvas_dict[color].append(image)
    if len(canvas_dict[color]) > 8: canvas_dict[color] = canvas_dict[color][-8:]
    while len(canvas_dict[color]) < 8: canvas_dict[color] = [np.zeros_like(image)] + canvas_dict[color]
    canvas_list = []
    for key in canvas_dict.keys(): canvas_list.append(cv2.hconcat(canvas_dict[key]))
    cv2.imshow("A", imutils.resize(cv2.vconcat(canvas_list), width=1000))
    cv2.waitKey(1)