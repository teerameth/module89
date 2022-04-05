import os, time
import cv2
import v4l2capture
import select
import os
import numpy as np
import subprocess
import tensorflow as tf # Added as colab instance often crash
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import imutils
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

PATH_TO_LABELS = '/media/teera/HDD1TB/model/object_detection/chessboard/label_map.pbtxt'    # Label Map path
PATH_TO_SAVED_MODEL = "/media/teera/HDD1TB/model/object_detection/chessboard/saved_model"   # Saved model path

print('Loading model...', end='')
start_time = time.time()
# Load saved model and build the detection function
detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)
end_time = time.time()
elapsed_time = end_time - start_time
print('Done! Took {} seconds'.format(elapsed_time))

# Set category index
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
import numpy as np
import warnings
warnings.filterwarnings('ignore')   # Suppress Matplotlib warnings
# This is required to display the images.

# cap = cv2.VideoCapture('/home/teera/1.mp4')
# cap = cv2.VideoCapture(0)
cap = Camera(0, width=1920, height=1080)

while True:
    timestamp = time.time()
    img = cap.read()
    if img is None: break
    cv2.imshow("Original", img)
    img = imutils.resize(img, height=300)

    input_tensor = tf.convert_to_tensor(img)        # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = input_tensor[tf.newaxis, ...]    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    detections = detect_fn(input_tensor)            # input_tensor = np.expand_dims(image_np, 0)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                   for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    image_np_with_detections = img.copy()

    # viz_utils.visualize_boxes_and_labels_on_image_array(
    #       image_np_with_detections,
    #       detections['detection_boxes'],
    #       detections['detection_classes'],
    #       detections['detection_scores'],
    #       category_index,
    #       use_normalized_coordinates=True,
    #       max_boxes_to_draw=20,
    #       min_score_thresh=.30,
    #       agnostic_mode=False)

    # print('detection_multiclass_scores')
    # print(detections['detection_multiclass_scores'][0][0])
    # print('detection_classes')
    # print(detections['detection_classes'][0])
    # print('num_detections')
    # print(detections['num_detections'])
    # print('detection_anchor_indices')
    # print(detections['detection_anchor_indices'][0])
    # print('detection_scores')
    print(detections['detection_scores'][0])
    # print('detection_boxes')
    # print(detections['detection_boxes'][0])
    # print('raw_detection_scores')
    # print(detections['raw_detection_scores'][0])
    # print('raw_detection_boxes')
    # print(detections['raw_detection_boxes'][0])
    if len(detections['detection_scores']) != 0:    # Something detected
        if detections['detection_scores'][0] > 0.7:  # minimum detection threshold
            bbox = detections['detection_boxes'][0] # [ymin, xmin, ymax, xmax]

        (height, width, _) = img.shape
        bbox[0] = round(bbox[0] * height)
        bbox[1] = round(bbox[1] * width)
        bbox[2] = round(bbox[2] * height)
        bbox[3] = round(bbox[3] * width)
        bbox = [int(x) for x in bbox]
        # print(bbox)
        cv2.rectangle(image_np_with_detections, (bbox[1], bbox[0]), (bbox[3], bbox[2]), (0, 0, 255), 2)
        cv2.imshow("A", image_np_with_detections)
    key = cv2.waitKey(1)
    if key == ord('q'): break
    # print(1/(time.time() - timestamp))

cv2.destroyAllWindows()