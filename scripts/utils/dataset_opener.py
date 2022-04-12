import os
import glob
import json
import numpy as np
import tensorflow as tf

# Converting the values into features
def _int64_feature(value):  # _int64 is used for numeric values
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
def _bytes_feature(value):  # _bytes is used for string/char values
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))
def decode_fn(record_bytes):
    return tf.io.parse_single_example(
        # Data
        record_bytes,
        # Schema
        {
            'height': tf.io.FixedLenFeature([1], tf.int64),
            'width': tf.io.FixedLenFeature([1], tf.int64),
            'depth': tf.io.FixedLenFeature([1], tf.int64),
            "image": tf.io.FixedLenFeature([480*640*3], dtype=tf.string),
            "corners": tf.io.FixedLenFeature([8], dtype=tf.int64),
            # "camera_matrix": tf.io.FixedLenFeature([9], dtype=tf.float32)
        }
    )

# Create a dictionary describing the features.
image_feature_description = {
    'height': tf.io.FixedLenFeature([], tf.int64),
    'width': tf.io.FixedLenFeature([], tf.int64),
    'depth': tf.io.FixedLenFeature([], tf.int64),
    'image': tf.io.FixedLenFeature([], tf.string),
    'corners': tf.io.FixedLenFeature([8], tf.int64)
}

def _parse_image_function(example_proto):
  # Parse the input tf.train.Example proto using the dictionary above.
  return tf.io.parse_single_example(example_proto, image_feature_description)

dataset_config = json.load(open(os.path.join('../../config/dataset_config.json')))
output_path = dataset_config['capture_path']
file_list = glob.glob(os.path.join(output_path, '*.tfrecords'))
for file_path in file_list:
    dataset = tf.data.TFRecordDataset(file_path)
    parsed_image_dataset = dataset.map(_parse_image_function)
    # print(parsed_image_dataset)
    for image_features in parsed_image_dataset:
        height = image_features['height'].numpy()
        width = image_features['width'].numpy()
        depth = image_features['depth'].numpy()
        image = tf.io.decode_png(image_features['image'])   # Auto detect image shape when decoded
        print(image.shape)