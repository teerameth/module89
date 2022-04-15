import numpy as np
import cv2
import math

cameraMatrix = np.array([[1395.3709390074625, 0.0, 984.6248356317226], [0.0, 1396.2122002126725, 534.9517311724618], [0.0, 0.0, 1.0]], np.float32) # Humanoid
dist = np.array([[0.1097213194870457, -0.1989645299789654, -0.002106454674127449, 0.004428959364733587, 0.06865838341764481]]) # Humanoid
rvec = np.array([0.0, 0.0, 0.0]) # float only
tvec = np.array([0.0, 0.0, 0.0]) # float only

##################################
### Perspective Transformation ###
##################################
def order_points(pts):
	# sort the points based on their x-coordinates
	xSorted = pts[np.argsort(pts[:, 0]), :]
	# grab the left-most and right-most points from the sorted
	# x-roodinate points
	leftMost = xSorted[:2, :]
	rightMost = xSorted[2:, :]
	# now, sort the left-most coordinates according to their
	# y-coordinates so we can grab the top-left and bottom-left
	# points, respectively
	leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
	(tl, bl) = leftMost
	# if use Euclidean distance, it will run in error when the object
	# is trapezoid. So we should use the same simple y-coordinates order method.
	# now, sort the right-most coordinates according to their
	# y-coordinates so we can grab the top-right and bottom-right
	# points, respectively
	rightMost = rightMost[np.argsort(rightMost[:, 1]), :]
	(tr, br) = rightMost
	# return the coordinates in top-left, top-right,
	# bottom-right, and bottom-left order
	return np.array([tl, tr, br, bl], dtype="float32")
def four_point_transform(image, pts):
	rect = order_points(pts)
	(tl, tr, br, bl) = rect
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
	M = cv2.getPerspectiveTransform(rect, dst) # compute the perspective transform matrix and then apply it
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
	warped = cv2.rotate(warped, cv2.ROTATE_90_COUNTERCLOCKWISE)
	return warped
################################
### Aruco Perspective Warped ###
################################
parameters =  cv2.aruco.DetectorParameters_create()
dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_250)
markerLength = 0.04
markerSeparation = 0.01
board = cv2.aruco.GridBoard_create(markersX=10, markersY=10, markerLength=markerLength, markerSeparation=markerSeparation, dictionary=dictionary)
marker_register = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [10, 20, 30, 40, 50, 60, 70, 80, 90], [9, 19, 29, 39, 49, 59, 69, 79, 89, 99], [90, 91, 92, 93, 94, 95, 96, 97, 98, 99]]

def aruco_crop(frame):
	markerCorners, markerIds, _ = cv2.aruco.detectMarkers(frame, dictionary, parameters=parameters)
	side_count = 0
	for register in marker_register:
		if markerIds is None: break
		for id in markerIds:
			if id in register:
				side_count += 1
				break
	if side_count < 3: return 0
	if markerIds is not None:
		ret, _, _ = cv2.aruco.estimatePoseBoard(corners=markerCorners, ids=markerIds, board=board, cameraMatrix=cameraMatrix, distCoeffs=dist, rvec=rvec, tvec=tvec)
		if ret:
			# cv2.aruco.drawAxis(image=frame, cameraMatrix=cameraMatrix, distCoeffs=dist, rvec=rvec, tvec=tvec, length=0.1) # origin
			T_marker = np.array([markerLength, markerLength, 0.0])
			A = np.array([0.0, 0.0, 0.0]) + T_marker
			B = np.array([0.4 + markerSeparation, 0.0, 0.0]) + T_marker
			C = np.array([0.4 + markerSeparation, 0.4 + markerSeparation, 0.0]) + T_marker
			D = np.array([0.0, 0.4 + markerSeparation, 0.0]) + T_marker
			### Find Transformatio Matrix ###
			rotM = np.zeros(shape=(3, 3))
			cv2.Rodrigues(rvec, rotM, jacobian=0)
			### Map to image coordinate ###
			pts, jac = cv2.projectPoints(np.float32([A, B, C, D]).reshape(-1, 3), rvec, tvec, cameraMatrix, dist)
			pts = np.array([tuple(pts[i].ravel()) for i in range(4)], dtype="float32")
			pts = order_points(pts)
		else: return 0
		## Perspective Crop ##
		warped = four_point_transform(frame, pts)
		warped = cv2.resize(warped, (800, 800))
		valid_mask = four_point_transform(np.ones(frame.shape[:2], dtype="uint8") * 255, pts)
		valid_mask = cv2.resize(valid_mask, (800, 800))
		# cv2.imshow("Warped", warped)
		# cv2.imshow("Valid", valid_mask)
		return warped, valid_mask
	else: return 0

def inversePerspective(rvec, tvec):
	""" Applies perspective transform for given rvec and tvec. """
	R, _ = cv2.Rodrigues(rvec)
	R = np.matrix(R).T
	invTvec = np.dot(R, np.matrix(-tvec))
	invRvec, _ = cv2.Rodrigues(R)
	return invRvec, invTvec

def relativePosition(rvec1, tvec1, rvec2, tvec2):
	""" Get relative position for rvec2 & tvec2. Compose the returned rvec & tvec to use composeRT with rvec2 & tvec2 """
	rvec1, tvec1 = rvec1.reshape((3, 1)), tvec1.reshape((3, 1))
	rvec2, tvec2 = rvec2.reshape((3, 1)), tvec2.reshape((3, 1))

	# Inverse the second marker, the right one in the image
	invRvec, invTvec = inversePerspective(rvec2, tvec2)

	info = cv2.composeRT(rvec1, tvec1, invRvec, invTvec)
	composedRvec, composedTvec = info[0], info[1]

	composedRvec = composedRvec.reshape((3, 1))
	composedTvec = composedTvec.reshape((3, 1))
	return composedRvec, composedTvec

def translatedPosition(rvec_main, tvec_main, rvec_tile_board_space, tvec_tile_board_space):
	rotM_main_inv = np.zeros(shape=(3, 3))
	rotM_tile_board_space = np.zeros(shape=(3, 3))
	homogeneous_main_inv = np.identity(4)
	homogeneous_tile_board_space = np.identity(4)
	rvec_tile_cam_space = np.zeros(shape=(3, 3))

	rvec_main_inv, tvec_main_inv = inversePerspective(rvec_main.reshape((3, 1)), tvec_main.reshape((3, 1)))
	cv2.Rodrigues(rvec_main_inv, rotM_main_inv, jacobian=0)
	cv2.Rodrigues(rvec_tile_board_space, rotM_tile_board_space, jacobian=0)
	# Convert to Homogeneous Matrix
	homogeneous_main_inv[:3, :3] = rotM_main_inv
	homogeneous_main_inv[:3, 3] = tvec_main_inv.reshape(3)
	homogeneous_tile_board_space[:3, :3] = rotM_tile_board_space
	homogeneous_tile_board_space[:3, 3] = tvec_tile_board_space.reshape(3)
	homogeneous_tile_cam_space = np.matmul(homogeneous_main_inv, homogeneous_tile_board_space) # Premultiply homogeneous matrix

	rvec_tile_cam_space, _ = cv2.Rodrigues(rvec_tile_cam_space, homogeneous_tile_cam_space[:3, 3].reshape((3, 1)), jacobian=0)
	tvec_tile_cam_space = homogeneous_tile_cam_space[:3, 3]
	info_inv = cv2.composeRT(rvec_main_inv, tvec_main_inv, rvec_tile_cam_space, tvec_tile_cam_space)
	rvec_new_inv_cam_space, tvec_tile_inv_cam_space = info_inv[0], info_inv[1]
	rvec_new_cam_space, tvec_new_cam_space = inversePerspective(rvec_new_inv_cam_space.reshape((3, 1)), tvec_tile_inv_cam_space.reshape((3, 1)))
	print(rvec_new_cam_space)
	return rvec_new_cam_space, tvec_new_cam_space

def poly2view_angle(poly):
	rvec_tile, tvec_tile, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners=np.asarray([poly]), markerLength=0.05, cameraMatrix=cameraMatrix, distCoeffs=dist)
	rotM_tile = np.zeros(shape=(3, 3))
	rotM_tile, _ = cv2.Rodrigues(rvec_tile, rotM_tile, jacobian=0)
	tvec_tile_final = np.dot(tvec_tile, rotM_tile.T).reshape(3)
	tile_x, tile_y, tile_z = tvec_tile_final[0], tvec_tile_final[1], tvec_tile_final[2]
	angle_rad = math.asin((math.sqrt(tile_x ** 2 + tile_y ** 2)) / (math.sqrt(tile_x ** 2 + tile_y ** 2 + tile_z ** 2)))
	return angle_rad