import imutils
import onnxruntime as ort
import os
import numpy as np
import cv2
import v4l2capture
import select
import simplejson
import time
import subprocess
import tensorflow as tf # Added as colab instance often crash
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

config = simplejson.load(open(os.path.join('../config/camera_config.json')))
cameraMatrix = np.array(config['camera_matrix'], np.float32)
dist = np.array(config['dist'])

### Load Object Detection model ###
PATH_TO_LABELS = '/mnt/HDD4TB/model/object_detection/chessboard/label_map.pbtxt'    # Label Map path
PATH_TO_SAVED_MODEL = "/mnt/HDD4TB/model/object_detection/chessboard/saved_model"   # Saved model path

## on-demand VRAM usage (avoid eating up all VRAM)
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)

print('Loading object detection model...', end='')
start_time = time.time()
# Load saved model and build the detection function
detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)
end_time = time.time()
elapsed_time = end_time - start_time
print('Loaded! Took {} seconds'.format(elapsed_time))

# Set category index
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
import numpy as np
import warnings
warnings.filterwarnings('ignore')   # Suppress Matplotlib warnings

### Load DOPE model ###
# ort_sess = ort.InferenceSession('/mnt/HDD4TB/model/dope/chessboard/chessboard.onnx', providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider'])  # TensorrtExecutionProvider having the higher priority.
ort_sess = ort.InferenceSession('/mnt/HDD4TB/model/dope/chessboard/chessboard.onnx', providers=['CUDAExecutionProvider'])

based_obj_points = np.array([[-0.2, 0.23, 0], [-0.2, -0.23, 0], [0.2, 0.23, 0], [0.2, -0.23, 0], [0, 0, 0]])

class Camera():
    def __init__(self, i=0, width=1920, height=1080):
        self.cap = v4l2capture.Video_device("/dev/video" + str(i))
        size_x, size_y = self.cap.set_format(width, height, fourcc='MJPG')
        self.cap.set_fps(30)
        devnull = open(os.devnull, 'w')  # For no output
        subprocess.call(['v4l2-ctl', '--set-ctrl', 'power_line_frequency=1'], stdout=devnull, stderr=devnull)
        self.cap.set_focus_auto(0)
        self.cap.set_exposure_auto(3)
        # # self.cap.set_exposure_absolute(250)
        self.cap.set_auto_white_balance(0)
        # # self.cap.set_white_balance_temperature(2500)
        self.cap.create_buffers(1)  # Create a buffer to store image data before calling 'start'
        self.cap.queue_all_buffers()  # Send the buffer to the device. Some devices require this to be done before calling 'start'.
        self.cap.start()
    def read(self):
        # Wait for the device to fill buffer
        select.select((self.cap,), (), ())
        image_data = self.cap.read_and_queue()
        return cv2.imdecode(np.frombuffer(image_data, dtype=np.uint8), cv2.IMREAD_COLOR)  # Decode & return

def FindMax(maps):
    max_points, max_vals = [], []
    for m in maps:
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(cv2.GaussianBlur(m, (3, 3), 0))
        max_points.append(max_loc)
        max_vals.append(max_val)
    return max_points, max_vals

# cap = cv2.VideoCapture("/media/teera/WDPassport4TB/chess/0/0.avi")
cap = Camera(0, width=config['width'], height=config['height'])

while True:
    stamp = time.time()
    frame = cap.read()
    canvas = frame.copy()
    # scale image to (300, 300) for object detection input (SSDMobileNetV2)
    input_tensor = tf.convert_to_tensor(frame)      # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = input_tensor[tf.newaxis, ...]    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    detections = detect_fn(input_tensor)  # input_tensor = np.expand_dims(image_np, 0)
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
    detections['num_detections'] = num_detections
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
    # Visualize (bounding box)
    viz_utils.visualize_boxes_and_labels_on_image_array(
        canvas,
        detections['detection_boxes'],
        detections['detection_classes'],
        detections['detection_scores'],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=1,
        min_score_thresh=.70,
        agnostic_mode=False)
    if len(detections['detection_scores']) != 0:    # Something detected
        if detections['detection_scores'][0] > 0.7:  # minimum detection threshold
            bbox = detections['detection_boxes'][0] # [ymin, xmin, ymax, xmax] scaled in (0, 1)
            (height, width, _) = frame.shape
            ## Expand 10% on each side ##
            expand_scale = 0.2
            bbox_height = bbox[2] - bbox[0]
            bbox_width = bbox[3] - bbox[1]
            bbox[0] = bbox[0] - bbox_height * expand_scale
            bbox[1] = bbox[1] - bbox_width * expand_scale
            bbox[2] = bbox[2] + bbox_height * expand_scale
            bbox[3] = bbox[3] + bbox_width * expand_scale
            if bbox[0] < 0: bbox[0] = 0
            if bbox[1] < 0: bbox[1] = 0
            if bbox[2] > 1: bbox[2] = 1
            if bbox[3] > 1: bbox[3] = 1
            # Convert from (0, 1) to pixel coordinate
            bbox[0] = bbox[0] * height
            bbox[1] = bbox[1] * width
            bbox[2] = bbox[2] * height
            bbox[3] = bbox[3] * width
            bbox = [int(x) for x in bbox]
            cv2.rectangle(canvas, (bbox[1], bbox[0]), (bbox[3], bbox[2]), (0, 0, 255), 2)
            ## Fit with DOPE input shape (480, 640) ##
            origin = (bbox[0], bbox[1])
            if (bbox[2] - bbox[0])/(bbox[3] - bbox[1]) > (480/640): # Use height as reference
                dope_scale = (bbox[2] - bbox[0]) / 480
                center_x = (bbox[3] + bbox[1])/2
                scaled_width = dope_scale*640
                if center_x < scaled_width/2:
                    center_x = scaled_width/2                   # Shift right
                elif frame.shape[1] - center_x < scaled_width/2:
                    center_x = frame.shape[1] - scaled_width/2  # Shift left
                origin = (int(center_x-scaled_width/2), bbox[0])
                des = (int(center_x+scaled_width/2), bbox[2])
                dope_input = frame[origin[1]:des[1], origin[0]:des[0]]
                cv2.rectangle(canvas, origin, des, (255, 0, 0), 2)
            else:                               # Use width as reference
                dope_scale = (bbox[3] - bbox[1]) / 640
                center_y = (bbox[2] + bbox[0])/2
                scaled_height = dope_scale*480
                if center_y < scaled_height/2:
                    center_y = scaled_height/2                   # Shift down
                elif frame.shape[0] - center_y < scaled_height/2:
                    center_y = frame.shape[0] - scaled_height/2  # Shift up
                origin = (bbox[1], int(center_y - scaled_height / 2))
                des = (bbox[3], int(center_y + scaled_height / 2))
                dope_input = frame[origin[1]:des[1], origin[0]:des[0]]
                cv2.rectangle(canvas, origin, des, (255, 0, 0), 2)
            dope_input = imutils.resize(dope_input, height=480)
            if dope_input.shape[1] > 640: dope_input = dope_input[:, :640]
            if dope_input.shape[1] < 640:
                delta = 640 - dope_input.shape[1]
                if delta % 2 == 0: dope_input = cv2.copyMakeBorder(dope_input, 0, 0, int(delta/2), int(delta/2), borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))
                else: dope_input = cv2.copyMakeBorder(dope_input, 0, 0, int(delta/2), int(delta/2)+1, borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))
            cv2.imshow("DOPE input", dope_input)
            ## convert from NHWC to NCHW (batch N, channels C, height H, width W)
            x = dope_input.transpose([2, 0, 1])  # ([0, 3, 1, 2])
            x = np.float32(x)
            x /= 255
            x = np.expand_dims(x, 0)
            outputs = ort_sess.run(None, {'input': x})
            # for i in range(5): cv2.imshow("Belief" + str(i), imutils.resize(outputs[0][0][i], height=480))

            canvas = dope_input.copy()
            points, vals = FindMax(outputs[0][0])
            confidences = [False if val < 0.1 else True for val in vals]
            print(confidences)
            obj_points, img_points = [], []
            for i in range(5):
                if confidences[i] is True:
                    obj_points.append(based_obj_points[i])
                    img_points.append(points[i])

            if len(obj_points) >= 4:
                obj_points = np.array(obj_points)
                img_points = np.array(img_points, dtype=np.double) * 8
                print(obj_points)
                print(img_points)
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
            # if not False in confidences:
            #     obj_points = np.array([[-0.2, 0.23, 0], [-0.2, -0.23, 0], [0.2, 0.23, 0], [0.2, -0.23, 0]])
            #     img_points = np.array([points[0], points[1], points[2], points[3]], dtype=np.double) * 8

    cv2.imshow("A", canvas)
    key = cv2.waitKey(1)
    if key == ord('q'): break


    # # convert from NHWC to NCHW (batch N, channels C, height H, width W)
    # x = image.transpose([2, 0, 1])   # ([0, 3, 1, 2])
    # x = np.float32(x)
    # x /= 255
    # x = np.expand_dims(x, 0)
    # outputs = ort_sess.run(None, {'input': x})
    # for i in range(5):
    #     cv2.imshow("Belief" + str(i), imutils.resize(outputs[0][0][i], height=480))
    #
    # canvas = image.copy()
    # points, vals = FindMax(outputs[0][0])
    # confidences = [False if val < 0.1 else True for val in vals]
    # if not False in confidences:
    #     obj_points = np.array([[-0.2, 0.23, 0], [-0.2, -0.23, 0], [0.2, 0.23, 0], [0.2, -0.23, 0]])
    #     img_points = np.array([points[0], points[1], points[2], points[3]], dtype=np.double) * 8
    #
    #     ret, rvec, tvec = cv2.solvePnP(objectPoints=obj_points,
    #                                    imagePoints=img_points,
    #                                    cameraMatrix=cameraMatrix,
    #                                    distCoeffs=dist,
    #                                    flags=0)
    #     cv2.aruco.drawAxis(image=canvas,
    #                        cameraMatrix=cameraMatrix,
    #                        distCoeffs=dist,
    #                        rvec=rvec,
    #                        tvec=tvec,
    #                        length=0.1)
    # print(int(1/(time.time() - stamp)))
    # cv2.imshow('A', canvas)
    # cv2.waitKey(1)


# Print Result
# predicted, actual = classes[outputs[0][0].argmax(0)], classes[y]
# print(f'Predicted: "{predicted}", Actual: "{actual}"')

cv2.destroyAllWindows()