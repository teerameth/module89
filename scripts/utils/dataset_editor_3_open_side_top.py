import os
import glob
import json
import numpy as np
import tensorflow as tf
import cv2
import math
from transform import order_points, poly2view_angle
from tqdm import tqdm

# Converting the values into features
def _int64_feature(value):  # _int64 is used for numeric values
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
def _bytes_feature(value):  # _bytes is used for string/char values
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))
labelNames = ["Bishop", "King", "Knight", "Pawn", "Queen", "Rook"]
labels = ['b', 'k', 'n', 'p', 'q', 'r']

tile_feature_description = {
    'image': tf.io.FixedLenFeature([], tf.string),
    'label': tf.io.FixedLenFeature([], tf.int64)
}
def _parse_image_function(example_proto):
  # Parse the input tf.train.Example proto using the dictionary above.
  return tf.io.parse_single_example(example_proto, tile_feature_description)

dataset_config = json.load(open(os.path.join('../../config/dataset_config.json')))
output_path = dataset_config['capture_path']
top_tfrecord = os.path.join(output_path, 'top.tfrecords')
side_tfrecord = os.path.join(output_path, 'side.tfrecords')

dataset = tf.data.TFRecordDataset(side_tfrecord)
# print(sum(1 for _ in dataset))

parsed_image_dataset = dataset.map(_parse_image_function)
for image_features in parsed_image_dataset:
    label = image_features['label'].numpy()
    image = tf.io.decode_png(image_features['image'])  # Auto detect image shape when decoded
    image = np.array(image, dtype=np.uint8)
    cv2.imshow(labels[label], image)
    key = cv2.waitKey(1)
    if key == ord('q'): break
