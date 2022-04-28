import os
import glob
import json
import random

import numpy as np
import tensorflow as tf
import cv2
import math
from transform import order_points, poly2view_angle
from tqdm import tqdm

chess_piece_height = {"king": (0.081, 0.097), "queen": (0.07, 0.0762), "bishop": (0.058, 0.065), "knight": (0.054, 0.05715), "rook": (0.02845, 0.048), "pawn": (0.043, 0.045)}
chess_piece_diameter = {"king": (0.028, 0.0381), "queen": (0.028, 0.0362), "bishop": (0.026, 0.032), "knight": (0.026, 0.03255), "rook": (0.026, 0.03255), "pawn": (0.0191, 0.02825)}
mask_contour_index_list = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
scan_box_height = min(chess_piece_height['king'])
tile_size = 0.05
export_size = 224

## Avoid to use all GPU(s)VRAM
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)

def getBox2D(rvec, tvec, size = 0.05, height = scan_box_height, only_base=False):
    objpts = np.float32([[0, 0, 0], [size, 0, 0], [size, size, 0], [0, size, 0], [0, 0, height], [size, 0, height], [size, size, height], [0, size, height]]).reshape(-1, 3)
    if only_base: objpts = objpts[mask_contour_index_list[0]]
    imgpts, jac = cv2.projectPoints(objpts, rvec, tvec, cameraMatrix, dist)
    min_x = int(min(imgpts, key=lambda x: x[0][0]).ravel()[0])
    max_x = int(max(imgpts, key=lambda x: x[0][0]).ravel()[0])
    min_y = int(min(imgpts, key=lambda x: x[0][1]).ravel()[1])
    max_y = int(max(imgpts, key=lambda x: x[0][1]).ravel()[1])
    return (min_x, min_y), (max_x, max_y)
def getValidContour2D(rvec, tvec, size = 0.05, height = scan_box_height, only_base=False):
    objpts = np.float32([[0, 0, 0], [size, 0, 0], [size, size, 0], [0, size, 0], [0, 0, height], [size, 0, height], [size, size, height], [0, size, height]]).reshape(-1, 3)
    imgpts, jac = cv2.projectPoints(objpts, rvec, tvec, cameraMatrix, dist)
    valid_contours = []
    if only_base: valid_contours.append([imgpts[mask_contour_index_list[0][i]] for i in range(len(mask_contour_index_list[0]))])
    else:
        for mask_contour_index in mask_contour_index_list:
            valid_contours.append([imgpts[mask_contour_index[i]] for i in range(len(mask_contour_index))])
    return valid_contours
def getPoly2D(rvec, tvec, size = 0.05):
    objpts = np.float32([[0, 0, 0], [size, 0, 0], [size, size, 0], [0, size, 0]]).reshape(-1, 3)
    imgpts, jac = cv2.projectPoints(objpts, rvec, tvec, cameraMatrix, dist)
    return imgpts
def llr_tile_top(rvec, tvec):
    rotM = np.zeros(shape=(3, 3))
    rotM, _ = cv2.Rodrigues(rvec, rotM, jacobian=0)
    ### Draw chess piece space ###
    counter = 0
    poly_tile_list = []
    for y in range(3, -5, -1):
        for x in range(-4, 4):
            board_coordinate = np.array([x * 0.05, y * 0.05, 0.0])
            # find angle of each tile
            translated_tvec = tvec + np.dot(board_coordinate, rotM.T)
            poly_tile = getPoly2D(rvec, translated_tvec, size=0.05)
            poly_tile_list.append(poly_tile)
    return poly_tile_list
def llr_tile(rvec, tvec, only_base = False):
    rotM = np.zeros(shape=(3, 3))
    rotM, _ = cv2.Rodrigues(rvec, rotM, jacobian=0)
    ### Draw chess piece space ###
    counter = 0
    tile_volume_bbox_list, angle_list, valid_contours_list = [], [], []
    for y in range(3, -5, -1):
        for x in range(-4, 4):
            board_coordinate = np.array([x * 0.05, y * 0.05, 0.0])
            (min_x, min_y), (max_x, max_y) = getBox2D(rvec, tvec + np.dot(board_coordinate, rotM.T), size=0.05, height=scan_box_height, only_base=only_base)
            tile_volume_bbox_list.append([(min_x, min_y), (max_x, max_y)])

            # find angle of each tile
            translated_tvec = tvec + np.dot(board_coordinate, rotM.T)
            poly_tile = getPoly2D(rvec, translated_tvec, size=0.05)
            valid_contours = getValidContour2D(rvec, translated_tvec, size=0.05, height=scan_box_height, only_base=only_base)
            valid_contours_list.append(valid_contours)
            angle_rad = pose2view_angle(rvec, tvec)
            # angle_rad = poly2view_angle(poly_tile)
            angle_deg = angle_rad / 3.14 * 180
            angle_list.append(angle_deg)
            counter += 1
    tile_volume_bbox_list_new, angle_list_new = [], []
    for i in range(64):
        y, x = 7 - int(i / 8), i % 8
        tile_volume_bbox_list_new.append(tile_volume_bbox_list[8*y+x])
        angle_list_new.append(angle_list[8*y+x])
    return tile_volume_bbox_list, angle_list, valid_contours_list
def getCNNinput(img, bbox_list, valid_contours_list):
    CNNinputs = []
    for i in range(len(bbox_list)):
        [(min_x, min_y), (max_x, max_y)] = bbox_list[i]
        if min_x < 0: min_x = 0
        if min_y < 0: min_y = 0
        if max_x >= img.shape[1]: max_x = img.shape[1]-1
        if max_y >= img.shape[0]: max_y = img.shape[0]-1
        cropped = img[min_y:max_y, min_x:max_x].copy()
        valid_contours = valid_contours_list[i]
        mask = np.zeros(cropped.shape[:2], dtype="uint8")
        for valid_contour in valid_contours:
            local_valid_contour = []
            for point in valid_contour:
                x = int(point[0][0] - min_x)
                y = int(point[0][1] - min_y)
                local_valid_contour.append([x, y])
            local_valid_contour = np.array(local_valid_contour).reshape((-1, 1, 2)).astype(np.int32)
            cv2.drawContours(mask, [local_valid_contour], -1, 255, -1)
        CNNinputs.append(cv2.bitwise_and(cropped, cropped, mask=mask))
    return CNNinputs
def resize_and_pad(img, size=300, padding_color=(0,0,0)):
    old_size = img.shape[:2]
    ratio = float(size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])
    img = cv2.resize(img, (new_size[1], new_size[0]))
    delta_w = size - new_size[1]
    delta_h = size - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=padding_color)
def get_tile(img, rvec, tvec, only_base = False):
    tile_volume_bbox_list, angle_list, valid_contours_list = llr_tile(rvec, tvec, only_base=only_base)
    CNNinputs = getCNNinput(img, tile_volume_bbox_list, valid_contours_list)
    CNNinputs_padded = []
    for i in range(64):
        CNNinput_padded = resize_and_pad(CNNinputs[i], size=export_size)
        CNNinputs_padded.append(CNNinput_padded)
    return CNNinputs_padded, angle_list
def get_tile_top(img, rvec, tvec):
    poly_tile_list = llr_tile_top(rvec, tvec)
    CNNinputs = []
    pts2 = np.float32([(0, export_size-1), (export_size-1, export_size-1), (export_size-1, 0), (0, 0)])
    for poly_tile in poly_tile_list:
        M = cv2.getAffineTransform(np.float32(poly_tile).reshape((4, 2))[:3], pts2[:3])
        CNNinputs.append(cv2.warpAffine(img, M, (export_size, export_size)))
    return CNNinputs

# Converting the values into features
def _int64_feature(value):  # _int64 is used for numeric values
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _bytes_feature(value):  # _bytes is used for string/char values
    if isinstance(value, type(tf.constant(0))): # if value is tensor
        value = value.numpy() # get value of tensor
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

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

tile_feature_description = {
    'image': tf.io.FixedLenFeature([], tf.string),
    'label': tf.io.FixedLenFeature([], tf.int64)
}

def image_example(image, label):
    feature = {
        # 'height': _int64_feature(image.shape[0]),
        # 'width': _int64_feature(image.shape[1]),
        # 'depth': _int64_feature(image.shape[2]),
        'image': _bytes_feature(tf.io.encode_png(image)),
        'label': _int64_feature(label),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))

def _parse_image_function(example_proto):
  # Parse the input tf.train.Example proto using the dictionary above.
  return tf.io.parse_single_example(example_proto, image_feature_description)

labelNames = ["Bishop", "King", "Knight", "Pawn", "Queen", "Rook"]
labels = ['b', 'k', 'n', 'p', 'q', 'r']
def fen2board(fen): #
    board = np.zeros((8, 8), dtype=np.uint8)
    lines = fen.split('/')
    for i in range(8):
        line = lines[i]
        index = 0
        for char in line:
            if char.isnumeric(): index += int(char)
            else:
                board[i][index] = labels.index(char.lower()) + 1
                index += 1
    return board
def pose2view_angle(rvec, tvec):
    rotM = np.zeros(shape=(3, 3))
    rotM, _ = cv2.Rodrigues(rvec, rotM, jacobian=0)
    tvec_tile_final = np.dot(tvec, rotM.T).reshape(3)
    tile_x, tile_y, tile_z = tvec_tile_final[0], tvec_tile_final[1], tvec_tile_final[2]
    angle_rad = math.asin((math.sqrt(tile_x ** 2 + tile_y ** 2)) / (math.sqrt(tile_x ** 2 + tile_y ** 2 + tile_z ** 2)))
    return angle_rad

dataset_config = json.load(open(os.path.join('../../config/dataset_config.json')))
output_path = dataset_config['capture_path']
file_list = sorted(glob.glob(os.path.join(output_path, '*.tfrecords')))
fen_list = sorted(glob.glob(os.path.join(output_path, '*.txt')))
top_writer = tf.io.TFRecordWriter(os.path.join(output_path, 'top_random.tfrecords'))

tvec_displacement = 0.2 * tile_size

for i in range(len(file_list)):
    file_path = file_list[i]
    if not os.path.basename(file_path)[0].isnumeric(): continue # select only raw dataset
    fen = open(fen_list[i], "r").readline()
    print(file_path, fen_list[i])
    if '\n' in fen: fen = fen[:-1]
    board = fen2board(fen)
    dataset = tf.data.TFRecordDataset(file_path)
    parsed_image_dataset = dataset.map(_parse_image_function, num_parallel_calls=tf.data.AUTOTUNE)
    with tqdm() as pbar:
        for image_features in parsed_image_dataset:
            pbar.update(1)
            height = image_features['height'].numpy()
            width = image_features['width'].numpy()
            depth = image_features['depth'].numpy()
            image = tf.io.decode_png(image_features['image'])   # Auto detect image shape when decoded
            image = np.array(image, dtype=np.uint8)
            rvec = image_features['rvec'].numpy()
            tvec = image_features['tvec'].numpy()
            cameraMatrix = image_features['camera_matrix'].numpy().reshape((3, 3))
            dist = image_features['dist'].numpy()

            angle = pose2view_angle(rvec, tvec) # get view angle (radian)
            if angle > 0.2: continue # skip side view

            displacement_x = random.uniform(-tvec_displacement, tvec_displacement)
            displacement_y = random.uniform(-tvec_displacement, tvec_displacement)
            rotM = np.zeros(shape=(3, 3))
            rotM, _ = cv2.Rodrigues(rvec, rotM, jacobian=0)
            tvec = tvec + np.dot((displacement_x, displacement_y, 0), rotM.T)

            CNNinputs = get_tile_top(image, rvec, tvec)

            # divide 2 camera view [top <0.05, side >0.3]
            for x in range(8):
                for y in range(8):
                    label = board[y][x]
                    tile_image = CNNinputs[8 * y + x]
                    if label != 0: label = 1    # binary label
                    if label == 0 and random.randint(0, 3)%4!=0: continue   # empty too much (4.33:1)
                    tf_example = image_example(tile_image, label)
                    top_writer.write(tf_example.SerializeToString())

            # canvas = image.copy()
            # cv2.aruco.drawAxis(image=canvas,
            #                    cameraMatrix=cameraMatrix,
            #                    distCoeffs=dist,
            #                    rvec=rvec,
            #                    tvec=tvec,
            #                    length=0.1)
            # cv2.imshow("A", canvas)
            #
            # vertical_images = []
            # for x in range(8):
            #     image_list_vertical = []
            #     for y in range(8):
            #         canvas = resize_and_pad(CNNinputs[8 * y + x].copy(), size=100)
            #         # cv2.putText(canvas, str(round(angle_list[8 * y + x])), (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color=(0, 0, 255))
            #         label = board[y][x]
            #         if label != 0:
            #             cv2.putText(canvas, labels[label - 1], (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            #                         color=(0, 0, 255))
            #         image_list_vertical.append(
            #             cv2.copyMakeBorder(canvas, 1, 1, 1, 1, cv2.BORDER_CONSTANT, None, (0, 255, 0)))
            #     vertical_images.append(np.vstack(image_list_vertical))
            # combined_images = np.hstack(vertical_images)
            # cv2.imshow("All CNN inputs", combined_images)
            # key = cv2.waitKey(1)
            # if key == ord('n'): break
top_writer.close()