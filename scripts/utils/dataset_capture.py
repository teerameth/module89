import os
import cv2
import time
import glob
import imutils

save_path = "/media/teera/HDD1TB/dataset/classification/chess"
rpm = 1.03
vdo_path_list = glob.glob(os.path.join(save_path, '*.avi'))
vdo_list = [int(os.path.basename(vdo_path_list[i])[:-4]) for i in range(len(vdo_path_list))]
if len(vdo_list) > 0: n = max(vdo_list) + 1
else: n = 0

camera_index = [2, 4, 6]

caps = [cv2.VideoCapture(i) for i in camera_index]
for cap in caps:
    cap.set(3, 1920)
    cap.set(4, 1080)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    cap.set(cv2.CAP_PROP_AUTO_WB, 0)
    cap.set(cv2.CAP_PROP_FOCUS, 2)

while True:
    imgs = [cap.read()[1] for cap in caps]

    canvas = cv2.vconcat(imgs)
    cv2.imshow("A", imutils.resize(canvas, height=960))
    key = cv2.waitKey(1)
    if key == ord(' '):
        results = [cv2.VideoWriter(os.path.join(save_path, str(n+i) + '.avi'),
                                 cv2.VideoWriter_fourcc(*'MJPG'),
                                 10, (1920, 1080)) for i in range(len(camera_index))]

        [print("begin " + str(n+i)) for i in range(len(camera_index))]
        timestamp = time.time()
        while time.time() - timestamp < 60/rpm:
            imgs = [cap.read()[1] for cap in caps]
            [results[i].write(imgs[i]) for i in range(len(camera_index))]
            canvas = cv2.vconcat(imgs)
            cv2.imshow("A", imutils.resize(canvas, height=960))
            cv2.waitKey(1)
        [results[i].release() for i in range(len(camera_index))]
        [print("save " + str(n+i)) for i in range(len(camera_index))]
        n += len(camera_index)
    elif key == 27: break

[cap.release() for cap in caps]