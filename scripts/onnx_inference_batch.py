import imutils
import onnxruntime as ort
import tensorflow as tf
import os
import glob
import numpy as np
import cv2
import math
from scipy.spatial.transform import Rotation as R
import simplejson
import time
from pyquaternion import Quaternion as q

## Avoid to use all GPU(s)VRAM
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)

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

def pose2view_angle(rvec, tvec):
    rotM = np.zeros(shape=(3, 3))
    rotM, _ = cv2.Rodrigues(rvec, rotM, jacobian=0)
    tvec_tile_final = np.dot(tvec, rotM.T).reshape(3)
    tile_x, tile_y, tile_z = tvec_tile_final[0], tvec_tile_final[1], tvec_tile_final[2]
    angle_rad = math.asin((math.sqrt(tile_x ** 2 + tile_y ** 2)) / (math.sqrt(tile_x ** 2 + tile_y ** 2 + tile_z ** 2)))
    return angle_rad

config = simplejson.load(open(os.path.join('../config/camera_config.json')))
cameraMatrix = np.array(config['camera_matrix'], np.float32)
dist = np.array(config['dist'])
based_obj_points = np.array([[-0.2, 0.23, 0], [-0.2, -0.23, 0], [0.2, 0.23, 0], [0.2, -0.23, 0]])

def FindMax(maps):
    # max_points = []
    # max_vals = []
    # for map in maps:
    #     local_max_mask = NonMaximaSuppression(map)
    #     local_max_indices = np.where(local_max_mask == [255])
    #     local_max_indices = np.array(local_max_indices).transpose()
    #     vals = []
    #     for i in range(len(local_max_indices)):
    #         vals.append((local_max_indices[i][1], local_max_indices[i][0]))
    #         max_points.append((local_max_indices[i][1], local_max_indices[i][0]))
    #     # max_val = max(vals)
    #     # max_point = local_max_indices[vals.index(max_val)]
    #     # max_points.append((max_point[1], max_point[0]))
    #     # max_vals.append(max_val)
    # return max_points, max_vals
    max_points, max_vals = [], []
    for m in maps:
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(cv2.GaussianBlur(m, (3, 3), 0))
        max_points.append(max_loc)
        max_vals.append(max_val)
    return max_points, max_vals
def NonMaximaSuppression(map): # Non-maxima suppression
    # Find pixel that are equal to the local neighborhood not maximum (including 'plateaus')
    map = map - cv2.GaussianBlur(map, (0, 0), sigmaX=3) + 127   # Highpass filter
    maxima = cv2.dilate(map, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=20)
    mask = cv2.compare(map, maxima, cv2.CMP_GE)
    # cv2.imshow("NonMaxSuppression", imutils.resize(canvas, height=480))
    return mask
def inversePerspective(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    R = np.matrix(R).T
    invTvec = np.dot(R, np.matrix(-tvec))
    invRvec, _ = cv2.Rodrigues(R)
    return invRvec, invTvec
def relativePosition(rvec1, tvec1, rvec2, tvec2):
    rvec1, tvec1 = rvec1.reshape((3, 1)), tvec1.reshape((3, 1))
    rvec2, tvec2 = rvec2.reshape((3, 1)), tvec2.reshape((3, 1))

    # Inverse the second marker, the right one in the image
    invRvec, invTvec = inversePerspective(rvec2, tvec2)

    info = cv2.composeRT(rvec1, tvec1, invRvec, invTvec)
    composedRvec, composedTvec = info[0], info[1]

    composedRvec = composedRvec.reshape((3, 1))
    composedTvec = composedTvec.reshape((3, 1))
    return composedRvec, composedTvec
model_folder = '/media/teera/ROGESD/model/belief/chessboard_blender_6_stage'
model_list = glob.glob(os.path.join(model_folder, '*.onnx'))
print("Model files: " + str(len(model_list)))
result_folder = os.path.join(model_folder, 'report')
if not os.path.exists(result_folder): os.mkdir(result_folder)
data_config = simplejson.load(open('../config/dataset_config.json'))
# Prepare *.tfrecord
output_path = data_config['capture_path']
file_list = sorted(glob.glob(os.path.join(output_path, '*.tfrecords')))
ckpt_interval = 5
for model_index in range(int(len(model_list)/ckpt_interval)):
    model_file = os.path.join(model_folder, 'net_epoch_%d.onnx'%((model_index*ckpt_interval)+1))
    diff_rvec_list, diff_tvec_list = [], []
    print(model_file)
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    output_path = os.path.join(result_folder, os.path.basename(model_file)[:-4] + "mp4")
    out = cv2.VideoWriter(output_path, fourcc, 10.0, (1280, 480))
    ort_sess = ort.InferenceSession(model_file, providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider'])  # TensorrtExecutionProvider having the higher priority.
    for file_path in file_list[:10]:
        dataset = tf.data.TFRecordDataset(file_path)
        parsed_image_dataset = dataset.map(_parse_image_function)
        image_top_buffer = []
        image_side_buffer = []
        pose_top_buffer = []
        pose_side_buffer = []
        try:
            for image_features in parsed_image_dataset:
                # height = image_features['height'].numpy()
                # width = image_features['width'].numpy()
                # depth = image_features['depth'].numpy()
                image = tf.io.decode_png(image_features['image'])  # Auto detect image shape when decoded
                image = np.array(image, dtype=np.uint8)
                rvec = image_features['rvec'].numpy()
                tvec = image_features['tvec'].numpy()
                angle = pose2view_angle(rvec, tvec)  # get view angle (radian)
                if angle > 0.2:  # side view
                    image_side_buffer.append(image)
                    pose_side_buffer.append((rvec, tvec))
                else:  # top view
                    image_top_buffer.append(image)
                    pose_top_buffer.append((rvec, tvec))
        except: pass
        image_buffer = image_side_buffer + image_top_buffer
        pose_buffer = pose_side_buffer + pose_top_buffer
        for frame in range(int(len(image_buffer)/10)):
            image = image_buffer[frame*10]
            pose = pose_buffer[frame*10]
            canvas_list = []
            # convert from NHWC to NCHW (batch N, channels C, height H, width W)
            x = image.transpose([2, 0, 1])  # ([0, 3, 1, 2])
            x = np.float32(x)
            x /= 255
            x = np.expand_dims(x, 0)
            outputs = ort_sess.run(None, {'input': x})
            for i in range(4):
                overlay = imutils.resize(outputs[0][0][i], height=240)
                overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)
                overlay = np.array(overlay * 255, dtype=np.uint8)
                canvas = cv2.addWeighted(imutils.resize(image, height=240), 0.3, overlay, 0.7, 0)
                # cv2.imshow("Belief" + str(i), canvas)
                canvas_list.append(canvas)

            canvas = image.copy()
            # points, vals = FindMax(outputs[0][0])
            # for i in range(len(points)): cv2.circle(canvas, (points[i][0]*8, points[i][1]*8), 3, (255, 0, 0), -1)
            points, vals = FindMax(outputs[0][0])
            # print(vals)
            confidences = [False if val < 0.05 else True for val in vals]
            for i in range(4):
                cv2.circle(canvas, (points[i][0] * 8, points[i][1] * 8), 3, (255, 0, 0), -1)
                # if confidences[i] is True:
                #     cv2.circle(canvas, (points[i][0]*8, points[i][1]*8), 3, (255, 0, 0), -1)
            # if not False in confidences:
            obj_points = np.array([[-0.2, 0.23, 0], [-0.2, -0.23, 0], [0.2, 0.23, 0], [0.2, -0.23, 0]])
            img_points = np.array([points[0], points[1], points[2], points[3]], dtype=np.double) * 8

            ret, rvec, tvec = cv2.solvePnP(objectPoints=obj_points,
                                           imagePoints=img_points,
                                           cameraMatrix=cameraMatrix,
                                           distCoeffs=dist,
                                           flags=0)
            cv2.aruco.drawAxis(image=canvas,
                               cameraMatrix=cameraMatrix,
                               distCoeffs=dist,
                               rvec=rvec,
                               tvec=tvec,
                               length=0.1)
            diff_rvec, diff_tvec = relativePosition(rvec, tvec, pose[0], pose[1])
            diff_rvec_list.append(diff_rvec)
            diff_tvec_list.append(diff_tvec)
            # cv2.imshow("A", canvas)
            canvas_list.append(canvas)

            # Save result
            row1 = cv2.hconcat([canvas_list[0], canvas_list[1]])
            row2 = cv2.hconcat([canvas_list[2], canvas_list[3]])
            full_canvas = cv2.vconcat([row1, row2])
            full_canvas = cv2.hconcat([full_canvas, canvas_list[4]])
            cv2.imshow("A", full_canvas)
            out.write(full_canvas)
            cv2.waitKey(1)
        np.save(os.path.join(result_folder, os.path.basename(model_file)[:-5] + "_rvec.npy"), np.array(diff_rvec_list))
        np.save(os.path.join(result_folder, os.path.basename(model_file)[:-5] + "_tvec.npy"), np.array(diff_tvec_list))
    out.release()
    del ort_sess