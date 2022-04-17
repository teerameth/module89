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
# def _parse_image_function(example_proto):
#   # Parse the input tf.train.Example proto using the dictionary above.
#   return tf.io.parse_single_example(example_proto, tile_feature_description)

def _parse_image_function(example_proto):
    data = tf.io.parse_single_example(example_proto, tile_feature_description) # Parse the input tf.train.Example proto using dictionary.
    label = data['label']
    image = tf.io.decode_png(data['image'])  # Auto detect image shape when decoded
    return image, label
dataset_config = json.load(open(os.path.join('../../config/dataset_config.json')))
output_path = dataset_config['capture_path']
top_tfrecord_pattern = os.path.join(output_path, 'top.tfrecords-????-of-????')
side_tfrecord_pattern = os.path.join(output_path, 'side.tfrecords-????-of-????')

shards = tf.data.TFRecordDataset.list_files(side_tfrecord_pattern)
shards = shards.shuffle(buffer_size=1000) # Make sure to fully shuffle the list of tfrecord files.

# Preprocesses 10 files concurrently and interleaves records from each file into a single, unified dataset.
dataset = shards.interleave(
    tf.data.TFRecordDataset,
    cycle_length=10,
    block_length=1)
# Here we convert raw protobufs into a structs.

## Batched ##
dataset = dataset.map(_parse_image_function, num_parallel_calls=10)
dataset = dataset.shuffle(2000)
batch_size = 1000
dataset = dataset.batch(batch_size, drop_remainder=True)
dataset = dataset.shuffle(buffer_size=20)
for image_batch, label_batch in dataset:
    print(len(label_batch))
    for i in range(batch_size):
        label = label_batch[i]
        # image = np.array(image_batch[i], dtype=np.uint8)
        # cv2.imshow(labels[label], image)
        # key = cv2.waitKey(1)

## Sequence ##
# dataset = dataset.map(_parse_image_function, num_parallel_calls=10)
# dataset = dataset.shuffle(2000)
# # dataset = dataset.shuffle(buffer_size=20)
# for image, label in dataset:
#     cv2.imshow(labels[label], np.array(image, dtype=np.uint8))
#     key = cv2.waitKey(1)