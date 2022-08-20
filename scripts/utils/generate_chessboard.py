import cv2
import numpy as np
tile_size = 100
image = np.zeros((9*tile_size, 8*tile_size, 3))
image[-tile_size:][:] = (255, 255, 255)
for i in range(8):
    for j in range(8):
        y_offset = int((i+0.5)*tile_size)
        x_offset = j*tile_size
        if (i%2 + j%2) % 2 == 0:
            image[y_offset:y_offset+tile_size, x_offset:x_offset+tile_size] = (255, 255, 255)
        else:
            image[y_offset:y_offset + tile_size, x_offset:x_offset + tile_size] = (0, 0, 0)
cv2.imshow("A", image)
cv2.waitKey(0)
cv2.imwrite("chessboard.png", image)