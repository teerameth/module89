import os
import glob
import json
import numpy as np
import tensorflow as tf
import tempfile

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))): # if value is tensor
        value = value.numpy() # get value of tensor
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def image_example(image, corners):   # Create a dictionary with features that may be relevant.
    feature = {
        'height': _int64_feature(image.shape[0]),
        'width': _int64_feature(image.shape[1]),
        'depth': _int64_feature(image.shape[2]),
        'image': _bytes_feature(tf.io.encode_png(image)),
        # 'image': _bytes_feature(image),
        'corners': tf.train.Feature(int64_list=tf.train.Int64List(value=corners)),
        # 'camera_matrix': _float_feature(tf.io.serialize_tensor(cameraMatrix)),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))

camera_config = json.load(open(os.path.join('../../config/camera_config.json')))
cameraMatrix = np.array(camera_config['camera_matrix'], np.float32)
dist = np.array(camera_config['dist'])

dataset_config = json.load(open(os.path.join('../../config/dataset_config.json')))
output_path = dataset_config['capture_path']
file_list = glob.glob(os.path.join(output_path, '*'))
last_index = len(file_list)

image_size = (480, 640, 3)

### Define reference pose before first folder creation ###

for i in range(5):  # Each VDO
    example_path = os.path.join(tempfile.gettempdir(), "example.tfrecords")
    tfrecord_filename = os.path.join(output_path, f"{str(last_index).zfill(5)}.tfrecords")
    # Initiating the writer and creating the tfrecords file.
    writer = tf.io.TFRecordWriter(tfrecord_filename)
    for frame in range(10): # 360 degree capture
        image = np.random.randint(low=0, high=255, size=image_size, dtype=np.uint8)
        corners = [1, 2, 3, 4, 5, 6, 7, 8] # [serialized (x, y) x 4]
        tf_example = image_example(image, corners)
        writer.write(tf_example.SerializeToString())
    writer.close()
    last_index += 1