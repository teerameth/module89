import os
import glob
import json
import numpy as np
import tensorflow as tf
import cv2
import math
from transform import order_points, poly2view_angle
from tqdm import tqdm

dataset_config = json.load(open(os.path.join('../../config/dataset_config.json')))
output_path = dataset_config['capture_path']
top_tfrecord = os.path.join(output_path, 'top.tfrecords')
side_tfrecord = os.path.join(output_path, 'side.tfrecords')

def save_shard(dataset_name, output_path, num_shard=100):
    raw_dataset = tf.data.TFRecordDataset(os.path.join(output_path, dataset_name))
    for i in tqdm(range(num_shard)):
        writer = tf.data.experimental.TFRecordWriter(os.path.join(output_path, f'{dataset_name}-{i:04d}-of-{(num_shard-1):04d}'))
        writer.write(raw_dataset.shard(num_shard, i))

save_shard('top_random.tfrecords', output_path, 100)