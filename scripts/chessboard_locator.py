#!/usr/bin/env /home/teera/.virtualenvs/tf/bin/python

import collections
import itertools
import matplotlib.path
import os
import pyclipper
import sklearn.cluster

import cv2
import math
import numpy as np
import random
import scipy
import scipy.cluster
import simplejson
from ament_index_python.packages import get_package_share_directory
from keras.models import model_from_json

import geometry
from transform import order_points, poly2view_angle

__laps_model = os.path.join(get_package_share_directory('module89'), 'models', 'laps.model.json')
__laps_weights = os.path.join(get_package_share_directory('module89'), 'models', 'laps.weights.h5')
NC_LAPS_MODEL = model_from_json(open(__laps_model, 'r').read())
NC_LAPS_MODEL.load_weights(__laps_weights)
colors = [(255, 255, 255), (0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0)]

config = simplejson.load(open(os.path.join(get_package_share_directory('module89'), 'config', 'camera_config.json')))
cameraMatrix = np.array(config['camera_matrix'], np.float32)
dist = np.array(config['dist'])
resolution_x, resolution_y = config['width'], config['height']

chess_piece_height = {"king": (0.081, 0.097), "queen": (0.07, 0.0762), "bishop": (0.058, 0.065), "knight": (0.054, 0.05715), "rook": (0.02845, 0.048), "pawn": (0.043, 0.045)}
chess_piece_diameter = {"king": (0.028, 0.0381), "queen": (0.028, 0.0362), "bishop": (0.026, 0.032), "knight": (0.026, 0.03255), "rook": (0.026, 0.03255), "pawn": (0.0191, 0.02825)}
scan_box_height = min(chess_piece_height['king'])

def slid_canny(img, sigma=0.25):
    """apply Canny edge detector (automatic thresh)"""
    v = np.median(img)
    img = cv2.medianBlur(img, 5)
    img = cv2.GaussianBlur(img, (7, 7), 2)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    return cv2.Canny(img, lower, upper)
def slid_detector(img, alpha=150, beta=2):
    """detect lines using Hough algorithm"""
    __lines, lines = [], cv2.HoughLinesP(img, rho=1, theta=np.pi / 360 * beta, threshold=40, minLineLength=50, maxLineGap=15)  # [40, 40, 10]
    if lines is None: return []
    for line in np.reshape(lines, (-1, 4)):
        __lines += [[[int(line[0]), int(line[1])],
                     [int(line[2]), int(line[3])]]]
    return __lines
def slid_clahe(img, limit=2, grid=(3, 3), iters=5):
    """repair using CLAHE algorithm (adaptive histogram equalization)"""
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for i in range(iters): img = cv2.createCLAHE(clipLimit=limit, tileGridSize=grid).apply(img)
    if limit != 0:
        kernel = np.ones((10, 10), np.uint8)
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    return img
def inside_bbox(bbox, line): # bbox = [xmin, ymin, xmax, ymax]
    [xmin, ymin, xmax, ymax] = bbox
    if line[0][0] < xmin or line[0][0] > xmax: return False
    if line[1][0] < xmin or line[1][0] > xmax: return False
    if line[0][1] < ymin or line[0][1] > ymax: return False
    if line[1][1] < ymin or line[1][1] > ymax: return False
    return True
def laps_intersections(lines):
    """find all intersections"""
    __lines = [[(a[0], a[1]), (b[0], b[1])] for a, b in lines]
    return geometry.isect_segments(__lines)

NC_SLID_CLAHE = [[3, (2, 6), 5],  # @1
                 [3, (6, 2), 5],  # @2
                 [5, (3, 3), 5],  # @3
                 [0, (0, 0), 0]]  # EE

def pSLID(img, thresh=100, debug=False): # find all lines using different settings
    segments = []
    i = 0
    for key, arr in enumerate(NC_SLID_CLAHE):
        tmp = slid_clahe(img, limit=arr[0], grid=arr[1], iters=arr[2])
        if debug:
            cv2.imshow("pSLID: Filter setting " + str(key) + " - " + str(arr), tmp)
            cv2.waitKey(0)
            cv2.destroyWindow("pSLID: Filter setting " + str(key) + " - " + str(arr))
        __segments = list(slid_detector(slid_canny(tmp), thresh))
        segments += __segments
        i += 1
    return segments
def slid_tendency(raw_lines, s=4):
    lines = []
    scale = lambda x, y, s: int(x * (1 + s) / 2 + y * (1 - s) / 2)
    for a, b in raw_lines:
        a[0] = scale(a[0], b[0], s)
        a[1] = scale(a[1], b[1], s)
        b[0] = scale(b[0], a[0], s)
        b[1] = scale(b[1], a[1], s)
        lines += [[a, b]]
    return lines
def SLID(segments):
    global all_points
    all_points = []
    pregroup, group, hashmap, raw_lines = [[], []], {}, {}, []
    __cache = {}
    def __dis(a, b):
        idx = hash("__dis" + str(a) + str(b))
        if idx in __cache: return __cache[idx]
        __cache[idx] = np.linalg.norm(np.array(a) - np.array(b))
        return __cache[idx]
    X = {}
    def __fi(x):
        if x not in X: X[x] = 0;
        if (X[x] == x or X[x] == 0):
            X[x] = x
        else:
            X[x] = __fi(X[x])
        return X[x]
    def __un(a, b):
        ia, ib = __fi(a), __fi(b)
        X[ia] = ib
        group[ib] |= group[ia]
    # shortest path // height
    nln = lambda l1, x, dx: \
        np.linalg.norm(np.cross(np.array(l1[1]) - np.array(l1[0]), np.array(l1[0]) - np.array(x))) / dx
    def __similar(l1, l2):
        da, db = __dis(l1[0], l1[1]), __dis(l2[0], l2[1])
        d1a, d2a = nln(l1, l2[0], da), nln(l1, l2[1], da)
        d1b, d2b = nln(l2, l1[0], db), nln(l2, l1[1], db)
        ds = 0.25 * (d1a + d1b + d2a + d2b) + 0.00001
        alpha = 0.0625 * (da + db)
        t1 = (da / ds > alpha and db / ds > alpha)
        if not t1: return False
        return True
    def __generate(a, b, n):
        points = []
        t = 1 / n
        for i in range(n):
            x = a[0] + (b[0] - a[0]) * (i * t)
            y = a[1] + (b[1] - a[1]) * (i * t)
            points += [[int(x), int(y)]]
        return points
    def __analyze(group):
        global all_points
        points = []
        for idx in group: points += __generate(*hashmap[idx], 10)
        _, radius = cv2.minEnclosingCircle(np.array(points))
        w = radius * (math.pi / 2)
        vx, vy, cx, cy = cv2.fitLine(np.array(points), cv2.DIST_L2, 0, 0.01, 0.01)
        all_points += points
        return [[int(cx - vx * w), int(cy - vy * w)], [int(cx + vx * w), int(cy + vy * w)]]
    for l in segments:
        h = hash(str(l))
        t1 = l[0][0] - l[1][0]
        t2 = l[0][1] - l[1][1]
        hashmap[h] = l
        group[h] = set([h])
        X[h] = h
        if abs(t1) < abs(t2): pregroup[0].append(l)
        else: pregroup[1].append(l)
    for lines in pregroup:
        for i in range(len(lines)):
            l1 = lines[i];
            h1 = hash(str(l1))
            if (X[h1] != h1): continue
            for j in range(i + 1, len(lines)):
                l2 = lines[j]
                h2 = hash(str(l2))
                if (X[h2] != h2): continue
                if not __similar(l1, l2): continue
                __un(h1, h2)  # union & find
    for i in group:
        if (X[i] != i): continue
        ls = [hashmap[h] for h in group[i]]
    for i in group:
        if (X[i] != i): continue
        raw_lines += [__analyze(group[i])]
    return raw_lines
def laps_cluster(points, max_dist=10):
    """cluster very similar points"""
    if len(points) == 0:
        print("No Cluster!")
        return []
    try:
        Y = scipy.spatial.distance.pdist(points)
        Z = scipy.cluster.hierarchy.single(Y)
        T = scipy.cluster.hierarchy.fcluster(Z, max_dist, 'distance')
        clusters = collections.defaultdict(list)
        for i in range(len(T)):
            clusters[T[i]].append(points[i])
        clusters = clusters.values()
        clusters = map(lambda arr: (np.mean(np.array(arr)[:, 0]),
                                    np.mean(np.array(arr)[:, 1])), clusters)
        return list(clusters)  # if two points are close, they become one mean point
    except:
        print("Problem in Cluster")
        print(points)
        return []
def laps_detector(img, debug=False):
    """determine if that shape is positive"""
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)[1]
    img = cv2.Canny(img, 0, 255)
    img = cv2.resize(img, (21, 21), interpolation=cv2.INTER_CUBIC)

    X = [np.where(img > int(255 / 2), 1, 0).ravel()]
    X = X[0].reshape([-1, 21, 21, 1])

    img = cv2.dilate(img, None)
    mask = cv2.copyMakeBorder(img, top=1, bottom=1, left=1, right=1, borderType=cv2.BORDER_CONSTANT, value=[255, 255, 255])
    mask = cv2.bitwise_not(mask)
    i = 0
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # geometric detector
    if debug: _c = np.zeros((23, 23, 3), np.uint8)
    for cnt in contours:
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        x, y = int(x), int(y)
        approx = cv2.approxPolyDP(cnt, 0.1 * cv2.arcLength(cnt, True), True)

        if len(approx) == 4 and radius < 14:
            if debug: cv2.drawContours(_c, [cnt], 0, (0, 255, 0), 1)
            i += 1
        else:
            if debug: cv2.drawContours(_c, [cnt], 0, (0, 0, 255), 1)
    if debug:
        cv2.imshow("laps_detector", _c)
        cv2.waitKey(0)
    if i == 4: return (True, 1)
    pred = NC_LAPS_MODEL.predict(X)
    a, b = pred[0][0], pred[0][1]
    t = a > b and b < 0.03 and a > 0.975
    # decision
    if t: return (True, pred[0])
    else: return (False, pred[0])
def LAPS(img, intersec_points, size=10):
    points = []
    for pt in intersec_points:
        # pixels are in integers
        pt = list(map(int, pt))
        # size of our analysis area
        lx1 = max(0, int(pt[0] - size - 1))
        lx2 = max(0, int(pt[0] + size))
        ly1 = max(0, int(pt[1] - size))
        ly2 = max(0, int(pt[1] + size + 1))
        # cropping for detector
        dimg = img[ly1:ly2, lx1:lx2]
        dimg_shape = np.shape(dimg)
        # not valid
        if dimg_shape[0] <= 0 or dimg_shape[1] <= 0: continue
        # use neural network
        re_laps = laps_detector(dimg) # determine if that shape is positive
        if not re_laps[0]: continue
        # add if okay
        if pt[0] < 0 or pt[1] < 0: continue
        points += [pt]
    points = laps_cluster(points)
    return points
def llr_normalize(points): return [[int(a), int(b)] for a, b in points]
def llr_correctness(points, shape):
    __points = []
    for pt in points:
        if pt[0] < 0 or pt[1] < 0 or  pt[0] > shape[1] or  pt[1] > shape[0]: continue
        __points += [pt]
    return __points
def llr_polysort(pts): #sort points clockwise
    mlat = sum(x[0] for x in pts) / len(pts)
    mlng = sum(x[1] for x in pts) / len(pts)
    def __sort(x):  return (math.atan2(x[0]-mlat, x[1]-mlng) + 2*math.pi)%(2*math.pi)
    pts.sort(key=__sort)
    return pts
def llr_polyscore(cnt, pts, cen, alpha=5, beta=2):
    four_points = [(cnt[x][0], cnt[x][1]) for x in range(len(cnt))]
    area = cv2.contourArea(cnt)
    t2 = area < (4 * alpha**2) * 5
    if t2: return 0
    gamma = alpha / 1.5
    pco = pyclipper.PyclipperOffset()
    pco.AddPath(cnt, pyclipper.JT_MITER, pyclipper.ET_CLOSEDPOLYGON)
    pcnt = matplotlib.path.Path(pco.Execute(gamma)[0])  # FIXME: alpha/1.5
    wtfs = pcnt.contains_points(pts)
    pts_in = min(np.count_nonzero(wtfs), 49)
    t1 = pts_in < min(len(pts), 49) - 2 * beta - 1
    if t1: return 0
    A = pts_in
    B = area
    nln = lambda l1, x, dx: np.linalg.norm(np.cross(np.array(l1[1]) - np.array(l1[0]), np.array(l1[0]) - np.array(x))) / dx
    pcnt_in = []
    i = 0
    for pt in wtfs:
        if pt: pcnt_in += [pts[i]]
        i += 1
    def __convex_approx(points, alpha=0.001):
        hull = scipy.spatial.ConvexHull(np.array(points)).vertices
        cnt = np.array([points[pt] for pt in hull])
        return cnt
    cnt_in = __convex_approx(np.array(pcnt_in))
    points = cnt_in
    x = [p[0] for p in points]  # we are looking for a point
    y = [p[1] for p in points]  # central cluster
    cen2 = (sum(x) / len(points), sum(y) / len(points))
    G = np.linalg.norm(np.array(cen) - np.array(cen2))
    a = [cnt[0], cnt[1]]
    b = [cnt[1], cnt[2]]
    c = [cnt[2], cnt[3]]
    d = [cnt[3], cnt[0]]
    lns = [a, b, c, d]
    E = 0
    F = 0
    for l in lns:
        d = np.linalg.norm(np.array(l[0]) - np.array(l[1]))
        for p in cnt_in:
            r = nln(l, p, d)
            if r < gamma:
                E += r
                F += 1
    if F == 0: return 0
    E /= F
    if B == 0 or A == 0: return 0
    C = 1 + (E / A) ** (1 / 3)  # equality
    D = 1 + (G / A) ** (1 / 5)  # centroid
    R = ((A**4)) / ((B ** 2) * C * D) ## ((A**4)) / ((B ** 2) * C * D)
    ## Tracking score ##
    # if len(tracking_rvec) > 0:
    #     # Project point from old frame to image coordinate
    #     pts_old, jac_old = cv2.projectPoints(np.float32([np.zeros(3, dtype=np.float32)]).reshape(-1, 3), tracking_rvec[0], tracking_tvec[0], cameraMatrix, dist)
    #     # Project new point to image coordinate
    #     points = order_points(np.asarray(four_points))
    #     rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners=np.asarray([points]), markerLength=0.3, cameraMatrix=cameraMatrix, distCoeffs=dist)
    #     pts_new, jac_new = cv2.projectPoints(np.float32([np.zeros(3, dtype=np.float32)]).reshape(-1, 3), rvec, tvec, cameraMatrix, dist)
    #     pts_old = pts_old.reshape(2)
    #     pts_new = pts_new.reshape(2)
    #     # print(abs(pts_old[0]-pts_new[0])/resolution_x, abs(pts_old[1]-pts_new[1])/resolution_y)
    #     R -= (abs(pts_old[0]-pts_new[0])/resolution_x + abs(pts_old[1]-pts_new[1])/resolution_y)/2*R
    #print(R * (10 ** 12), A, "|", B, C, D, "|", E, G)
    return R
def llr_unique(a):
    indices = sorted(range(len(a)), key=a.__getitem__)
    indices = set(next(it) for k, it in itertools.groupby(indices, key=a.__getitem__))
    return [x for i, x in enumerate(a) if i in indices]
def LLR(img, points, lines, debug=False):
    old = points
    # --- otoczka
    def __convex_approx(points, alpha=0.01):
        hull = scipy.spatial.ConvexHull(np.array(points)).vertices
        cnt = np.array([points[pt] for pt in hull])
        approx = cv2.approxPolyDP(cnt, alpha * cv2.arcLength(cnt, True), True)
        return llr_normalize(itertools.chain(*approx))
    # --- geometry
    __cache = {}
    def __dis(a, b):
        idx = hash("__dis" + str(a) + str(b))
        if idx in __cache: return __cache[idx]
        __cache[idx] = np.linalg.norm(np.array(a) - np.array(b))
        return __cache[idx]
    nln = lambda l1, x, dx:  np.linalg.norm(np.cross(np.array(l1[1]) - np.array(l1[0]), np.array(l1[0]) - np.array(x))) / dx
    pregroup = [[], []]  # division in 2 group (of th frame)
    S = {}  # frame ranking result
    points = llr_correctness(llr_normalize(points), img.shape)  # correct the point
    # --- Clustering
    __points = {}
    points = llr_polysort(points)
    __max, __points_max = 0, []
    alpha = math.sqrt(cv2.contourArea(np.array(points)) / 49)
    X = sklearn.cluster.DBSCAN(eps=alpha * 4).fit(points)
    for i in range(len(points)): __points[i] = []
    for i in range(len(points)):
        if X.labels_[i] != -1: __points[X.labels_[i]] += [points[i]]
    for i in range(len(points)):
        if len(__points[i]) > __max:
            __max = len(__points[i])
            __points_max = __points[i]
    if len(__points) > 0 and len(points) > 49 / 2: points = __points_max
    # we create the outer ring
    ring = __convex_approx(llr_polysort(points))
    n = len(points)
    beta = n * (5 / 100)  # beta=n*(100-(effectiveness LAPS))
    alpha = math.sqrt(cv2.contourArea(np.array(points)) / 49)  # medium sheath of the mesh

    x = [p[0] for p in points]  # we are looking for a point
    y = [p[1] for p in points]  # central cluster
    centroid = (sum(x) / len(points), sum(y) / len(points))

    def __v(l):
        y_0, x_0 = l[0][0], l[0][1]
        y_1, x_1 = l[1][0], l[1][1]
        x_2 = 0
        t = (x_0 - x_2) / (x_0 - x_1 + 0.0001)
        a = [int((1 - t) * x_0 + t * x_1), int((1 - t) * y_0 + t * y_1)][::-1]
        x_2 = img.shape[0]
        t = (x_0 - x_2) / (x_0 - x_1 + 0.0001)
        b = [int((1 - t) * x_0 + t * x_1), int((1 - t) * y_0 + t * y_1)][::-1]
        poly1 = llr_polysort([[0, 0], [0, img.shape[0]], a, b])
        s1 = llr_polyscore(np.array(poly1), points, centroid, beta=int(beta), alpha=int(alpha / 2))
        poly2 = llr_polysort([a, b, [img.shape[1], 0], [img.shape[1], img.shape[0]]])
        s2 = llr_polyscore(np.array(poly2), points, centroid, beta=int(beta), alpha=int(alpha / 2))
        return [a, b], s1, s2
    def __h(l):
        x_0, y_0 = l[0][0], l[0][1]
        x_1, y_1 = l[1][0], l[1][1]
        x_2 = 0
        t = (x_0 - x_2) / (x_0 - x_1 + 0.0001)
        a = [int((1 - t) * x_0 + t * x_1), int((1 - t) * y_0 + t * y_1)]
        x_2 = img.shape[1]
        t = (x_0 - x_2) / (x_0 - x_1 + 0.0001)
        b = [int((1 - t) * x_0 + t * x_1), int((1 - t) * y_0 + t * y_1)]
        poly1 = llr_polysort([[0, 0], [img.shape[1], 0], a, b])
        s1 = llr_polyscore(np.array(poly1), points, centroid, beta=int(beta), alpha=int(alpha / 2))
        poly2 = llr_polysort([a, b, [0, img.shape[0]], [img.shape[1], img.shape[0]]])
        s2 = llr_polyscore(np.array(poly2), points, centroid, beta=int(beta), alpha=int(alpha / 2))
        return [a, b], s1, s2
    for l in lines:
        for p in points:
            t1 = nln(l, p, __dis(*l)) < alpha # (1) the line passes close to a good point
            t2 = nln(l, centroid, __dis(*l)) > alpha * 2.5 # (2) the line passes through the center of the cluster
            if t1 and t2:
                tx, ty = l[0][0] - l[1][0], l[0][1] - l[1][1]
                if abs(tx) < abs(ty): ll, s1, s2 = __v(l); o = 0
                else: ll, s1, s2 = __h(l); o = 1
                if s1 == 0 and s2 == 0: continue
                pregroup[o] += [ll]
    pregroup[0] = llr_unique(pregroup[0])
    pregroup[1] = llr_unique(pregroup[1])
    for v in itertools.combinations(pregroup[0], 2):
        for h in itertools.combinations(pregroup[1], 2):
            poly = laps_intersections([v[0], v[1], h[0], h[1]])
            poly = llr_correctness(poly, img.shape)
            if len(poly) != 4: continue
            poly = np.array(llr_polysort(llr_normalize(poly)))
            if not cv2.isContourConvex(poly): continue
            S[-llr_polyscore(poly, points, centroid, beta=int(beta), alpha=int(alpha / 2))] = poly
    if debug:
        counter = 0
        canvas = img.copy()
        for p in points: cv2.circle(canvas, (p[0], p[1]), 5, (255, 255, 255), 2)
        scores = []
        for polyscore, corner in S.items():
            scores.append(polyscore)
            canvas = drawLines(canvas, lines=[[corner[0], corner[1]], [corner[1], corner[2]], [corner[2], corner[3]], [corner[3], corner[0]]], mode=1, color=colors[counter%len(colors)], copy=False)
            h = 15 + (counter*15)%(canvas.shape[0]-15)
            w = 10 + int((counter*15)/(canvas.shape[0]-15))*80
            cv2.putText(canvas, str(round(-polyscore*1000, 4)), (w, h), cv2.FONT_HERSHEY_SIMPLEX, 0.5,color=colors[counter%len(colors)])
            counter += 1
            cv2.imshow("Polyscore", canvas)
            # cv2.imwrite("Polyscore" + str(counter) + ".png", canvas)
            cv2.waitKey(0)
        max_index = scores.index(max(scores))
    S = collections.OrderedDict(sorted(S.items()))  # max
    K = next(iter(S))
    four_points = llr_normalize(S[K])  # score
    return four_points
def sample_from_poly(img, poly, n=10):
    points = []
    # Split poly(length=4) into 2 triangle https://mathworld.wolfram.com/TrianglePointPicking.html
    triangleA = np.array((poly[0], poly[1], poly[2]))
    triangleB = np.array((poly[1], poly[2], poly[3]))
    for i in range(int(n/2+1)):
        ## Unfinished ##
        pass
def correct_90(img, rvec, tvec, rotM, debug=False):
    if len(img.shape) != 2: canvas = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else: canvas = img.copy()
    counter = 0
    black_pixel_list, white_pixel_list = [], []
    for y in range(-4, 4):
        for x in range(-4, 4):
            board_coordinate = np.array([x * 0.05, y * 0.05, 0.0])
            poly = getPoly2D(rvec, tvec + np.dot(board_coordinate, rotM.T), size = 0.05)
            center = [sum(y) / len(y) for y in zip(*poly)] # Sample from center of tiles
            pixel_value = canvas[int(center[0][1])][int(center[0][0])]
            if int(counter/8)%2==0 and counter%2==0 or int(counter/8)%2!=0 and counter%2!=0: black_pixel_list.append(pixel_value)
            else: white_pixel_list.append(pixel_value)
            counter += 1
    black_pixel = sorted(black_pixel_list)[int(len(black_pixel_list)/2)]
    white_pixel = sorted(white_pixel_list)[int(len(white_pixel_list)/2)]
    if black_pixel > white_pixel: # wrong side -> rotate 90 degree
        rotM, _ = cv2.Rodrigues(rvec, rotM, jacobian=0) # Convert from rotation vector -> rotation matrix (rvec=rot_vector, rotM=rot_matrix)
        rotM = np.dot(rotM, np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]))  # Rotate 180 degree about Z-axis
        rvec, _ = cv2.Rodrigues(rotM, rvec, jacobian=0) # Convert from rotation matrix -> rotation vector
    return rvec
def correct_180(img, rvec, tvec, rotM, n=10, debug=False):
    if len(img.shape) != 2: canvas = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else: canvas = img.copy()
    [poly_up, poly_down] = get_bars(rvec, tvec, rotM, image=img, debug=debug)
    up_sample, down_sample = sample_polygon(poly_up, n=n), sample_polygon(poly_down, n=n)
    up_intensity, down_intensity = 0, 0
    for up_point, down_point in zip(up_sample, down_sample):
        up_intensity += canvas[up_point[1]][up_point[0]]
        down_intensity += canvas[down_point[1]][down_point[0]]
    if up_intensity > down_intensity: # wrong side (white nedeed to be downside)
        rotM, _ = cv2.Rodrigues(rvec, rotM, jacobian=0) # Convert from rotation vector -> rotation matrix (rvec=rot_vector, rotM=rot_matrix)
        rotM = np.dot(rotM, np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])) # Rotate 180 degree about Z-axis
        rvec, _ = cv2.Rodrigues(rotM, rvec, jacobian=0) # Convert from rotation matrix -> rotation vector
    return rvec
def sample_polygon(poly, n=10):
    x_list = np.transpose(poly)[0]
    y_list = np.transpose(poly)[1]
    counter = 0 # count sample points inside polygon
    sample_points = []
    while True:
        x, y = random.randint(min(x_list), max(x_list)), random.randint(min(y_list), max( y_list))  # randomly sample point from inside polygon
        if cv2.pointPolygonTest(poly, (x, y), False) == 1.0:
            sample_points.append((x, y))
            counter += 1
        if counter == n: break
    return sample_points
def llr_pad(four_points, img, debug=False):
    points = order_points(np.asarray(four_points))
    rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners=np.asarray([points]), markerLength=0.3, cameraMatrix=cameraMatrix, distCoeffs=dist)
    ## [0.0, 0.0, 0.0] is center of ChessBoard
    O = np.array([0.0, 0.0, 0.0])
    A = np.array([-0.2, -0.2, 0.0])
    B = np.array([0.2, -0.2, 0.0])
    C = np.array([0.2, 0.2, 0.0])
    D = np.array([-0.2, 0.2, 0.0])
    ### Find Transformation Matrix #four_points##
    rotM = np.zeros(shape=(3, 3))
    cv2.Rodrigues(rvec, rotM, jacobian=0)
    ### Map to image coordinate ###
    pts, jac = cv2.projectPoints(np.float32([A, B, C, D]).reshape(-1, 3), rvec, tvec, cameraMatrix, dist)
    pts = np.array([tuple(pts[i].ravel()) for i in range(4)], dtype="float32")
    pts = order_points(pts)
    pts = [[int(pts[0][0]), int(pts[0][1])], [int(pts[1][0]), int(pts[1][1])], [int(pts[2][0]), int(pts[2][1])], [int(pts[3][0]), int(pts[3][1])]]
    # print(pts)
    ### Draw axis ###
    canvas = img.copy()
    # for point in [O, A, B, C, D]: cv2.aruco.drawAxis(image=canvas, cameraMatrix=cameraMatrix, distCoeffs=dist, rvec=rvec, tvec=tvec + np.dot(point, rotM.T), length=0.1)
    # cv2.imwrite("Board Corner Axis.png", canvas)
    cv2.aruco.drawAxis(image=canvas, cameraMatrix=cameraMatrix, distCoeffs=dist, rvec=rvec, tvec=tvec, length=0.1)
    ## Update lastest tvec, rvec
    # tracking_rvec.append(rvec)
    # tracking_tvec.append(tvec)
    cv2.polylines(canvas, [np.array(four_points)], isClosed=True, color=(0, 0, 255), thickness=2)
    cv2.polylines(canvas, [np.array(pts)], isClosed=True, color=(100, 255, 100), thickness=2)
    cv2.imshow("Polyline", canvas)
    cv2.waitKey(1)
    # Correct the chessboard side (white on lower y axis)
    rvec = correct_90(img, rvec, tvec, rotM, debug=debug) # check & correct board rotation
    rvec = correct_180(img, rvec, tvec, rotM, debug=debug)
    return rvec, tvec

def llr_tile(img, rvec, tvec, debug=False):
    rotM = np.zeros(shape=(3, 3))
    rotM, _ = cv2.Rodrigues(rvec, rotM, jacobian=0)
    # rotM, _ = cv2.Rodrigues(rvec) # get new rotation matrix

    canvas = img.copy()
    cv2.aruco.drawAxis(image=canvas, cameraMatrix=cameraMatrix, distCoeffs=dist, rvec=rvec, tvec=tvec, length=0.1)
    cv2.imshow("Rotated", canvas)
    cv2.waitKey(100)
    ### Draw chess piece space ###
    counter = 0
    tile_volume_bbox_list, angle_list = [], []
    if debug: canvas1 = img.copy()
    for y in range(-4, 4):
        for x in range(-4, 4):
            board_coordinate = np.array([x * 0.05, y * 0.05, 0.0])
            (min_x, min_y), (max_x, max_y) = getBox2D(rvec, tvec + np.dot(board_coordinate, rotM.T), size=0.05,
                                                      height=scan_box_height)
            tile_volume_bbox_list.append([(min_x, min_y), (max_x, max_y)])

            # find angle of each tile
            poly_tile = getPoly2D(rvec, tvec + np.dot(board_coordinate, rotM.T), size=0.05)
            angle_rad = poly2view_angle(poly_tile)

            angle_deg = angle_rad / 3.14 * 180
            angle_list.append(angle_deg)

            if debug:
                canvas = img.copy()
                # canvas1 = img.copy()
                canvas2 = img.copy()
                drawBox3D(canvas, rvec, tvec + np.dot(board_coordinate, rotM.T), size=0.05, height=scan_box_height,
                          color=colors[counter % len(colors)], thickness=1)
                drawPoly2D(canvas1, rvec, tvec + np.dot(board_coordinate, rotM.T), size=0.05,
                           color=colors[counter % len(colors)], thickness=1)
                drawBox2D(canvas2, rvec, tvec + np.dot(board_coordinate, rotM.T), size=0.05, height=scan_box_height,
                          color=colors[counter % len(colors)], thickness=1)
                # print(tvec + np.dot(board_coordinate, rotM.T))
                cv2.imshow("3D&2D box tiles", np.vstack([canvas, canvas2]))
                cv2.imshow("Poly tile", canvas1)
                # cv2.imwrite("box_tile_" + str(counter) + ".png", np.vstack([canvas, canvas2]))
                cv2.waitKey(10)
            counter += 1
    return tile_volume_bbox_list, angle_list

def get_bars(rvec, tvec, rotM, size = 0.05, image=None, debug=False):
    Bar_up = [np.array([-0.2, 0.23, 0.0]), np.array([0.2, 0.23, 0.0]), np.array([0.2, 0.2, 0.0]), np.array([-0.2, 0.2, 0.0])]
    # Bar_left = [np.array([-0.23, 0.2, 0.0]), np.array([-0.20, 0.2, 0.0]), np.array([-0.2, -0.2, 0.0]), np.array([-0.23, -0.2, 0.0])]
    # Bar_right = [np.array([0.2, 0.2, 0.0]), np.array([0.23, 0.2, 0.0]), np.array([0.23, -0.2, 0.0]), np.array([0.2, -0.2, 0.0])]
    Bar_down = [np.array([-0.2, -0.2, 0.0]), np.array([0.2, -0.2, 0.0]), np.array([0.2, -0.23, 0.0]), np.array([-0.2, -0.23, 0.0])]
    bar_list = []
    # for bar in [Bar_up, Bar_left, Bar_right, Bar_down]:
    for bar in [Bar_up, Bar_down]:
        objpts = np.float32(bar).reshape(-1, 3)
        imgpts, jac = cv2.projectPoints(objpts, rvec, tvec, cameraMatrix, dist)
        imgpts = [(int(x[0][0]), int(x[0][1])) for x in imgpts]
        bar_list.append(np.array(imgpts))
    if debug:
        canvas = image.copy()
        cv2.polylines(canvas, bar_list, isClosed=True, color=(255, 255, 0), thickness=2)
        cv2.imshow("Bar", canvas)
        # cv2.imwrite("Bar.png", canvas)
        cv2.waitKey(0)
    return bar_list
def getBox2D(rvec, tvec, size = 0.05, height = scan_box_height):
    objpts = np.float32([[0, 0, 0], [size, 0, 0], [size, size, 0], [0, size, 0], [0, 0, height], [size, 0, height], [size, size, height], [0, size, height]]).reshape(-1, 3)
    imgpts, jac = cv2.projectPoints(objpts, rvec, tvec, cameraMatrix, dist)
    min_x = int(min(imgpts, key=lambda x: x[0][0]).ravel()[0])
    max_x = int(max(imgpts, key=lambda x: x[0][0]).ravel()[0])
    min_y = int(min(imgpts, key=lambda x: x[0][1]).ravel()[1])
    max_y = int(max(imgpts, key=lambda x: x[0][1]).ravel()[1])
    return (min_x, min_y), (max_x, max_y)
def getPoly2D(rvec, tvec, size = 0.05):
    objpts = np.float32([[0, 0, 0], [size, 0, 0], [size, size, 0], [0, size, 0]]).reshape(-1, 3)
    imgpts, jac = cv2.projectPoints(objpts, rvec, tvec, cameraMatrix, dist)
    return imgpts
def getCNNinput(img, bbox_list):
    CNNinputs = []
    for [(min_x, min_y), (max_x, max_y)] in bbox_list:
        if min_x < 0: min_x = 0
        if min_y < 0: min_y = 0
        if max_x >= img.shape[1]: max_x = img.shape[1]-1
        if max_y >= img.shape[0]: max_y = img.shape[0]-1
        CNNinputs.append(img[min_y:max_y, min_x:max_x])
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

def drawLines(frame, lines, thickness=2, mode=0, color=(0, 255, 0), transform=None, copy=True):
    if copy: canvas = frame.copy()
    else: canvas = frame
    counter = 0
    for line in lines:
        if mode == 0: color = colors[counter%len(colors)]
        if transform is None: cv2.line(canvas, tuple(line[0]), tuple(line[1]), color, thickness=thickness)
        else: cv2.line(canvas, (line[0][0] + transform[0], line[0][1] + transform[1]), (line[1][0] + transform[0], line[1][1] + transform[1]), color, thickness=thickness)
        counter += 1
    return canvas

# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(rvec):
    R = np.zeros(shape=(3, 3))
    R, _ = cv2.Rodrigues(rvec, R, jacobian=0)
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0
    return np.array([x, y, z])

def drawPoly2D(frame, rvec, tvec, size = 0.05, color=(0, 0, 255), thickness = 2):
    imgpts = getPoly2D(rvec, tvec)
    cv2.line(frame, tuple(imgpts[0].ravel()), tuple(imgpts[1].ravel()), color, thickness)
    cv2.line(frame, tuple(imgpts[1].ravel()), tuple(imgpts[2].ravel()), color, thickness)
    cv2.line(frame, tuple(imgpts[2].ravel()), tuple(imgpts[3].ravel()), color, thickness)
    cv2.line(frame, tuple(imgpts[3].ravel()), tuple(imgpts[0].ravel()), color, thickness)

def drawBox3D(frame, rvec, tvec, size = 0.05, height = scan_box_height, color=(0, 0, 255), thickness = 2):
    objpts = np.float32([[0,0,0], [size,0,0], [size,size,0], [0,size,0], [0,0,height], [size,0,height], [size,size,height], [0,size,height]]).reshape(-1,3)
    imgpts, jac = cv2.projectPoints(objpts, rvec, tvec, cameraMatrix, dist)

    cv2.line(frame, tuple(imgpts[0].ravel()), tuple(imgpts[1].ravel()), color, thickness)
    cv2.line(frame, tuple(imgpts[1].ravel()), tuple(imgpts[2].ravel()), color, thickness)
    cv2.line(frame, tuple(imgpts[2].ravel()), tuple(imgpts[3].ravel()), color, thickness)
    cv2.line(frame, tuple(imgpts[3].ravel()), tuple(imgpts[0].ravel()), color, thickness)

    cv2.line(frame, tuple(imgpts[0].ravel()), tuple(imgpts[0+4].ravel()), color, thickness)
    cv2.line(frame, tuple(imgpts[1].ravel()), tuple(imgpts[1+4].ravel()), color, thickness)
    cv2.line(frame, tuple(imgpts[2].ravel()), tuple(imgpts[2+4].ravel()), color, thickness)
    cv2.line(frame, tuple(imgpts[3].ravel()), tuple(imgpts[3+4].ravel()), color, thickness)

    cv2.line(frame, tuple(imgpts[0+4].ravel()), tuple(imgpts[1+4].ravel()), color, thickness)
    cv2.line(frame, tuple(imgpts[1+4].ravel()), tuple(imgpts[2+4].ravel()), color, thickness)
    cv2.line(frame, tuple(imgpts[2+4].ravel()), tuple(imgpts[3+4].ravel()), color, thickness)
    cv2.line(frame, tuple(imgpts[3+4].ravel()), tuple(imgpts[0+4].ravel()), color, thickness)

def drawBox2D(frame, rvec, tvec, size = 0.05, height = scan_box_height, color=(0, 0, 255), thickness=1):
    objpts = np.float32([[0, 0, 0], [size, 0, 0], [size, size, 0], [0, size, 0], [0, 0, height], [size, 0, height], [size, size, height], [0, size, height]]).reshape(-1, 3)
    imgpts, jac = cv2.projectPoints(objpts, rvec, tvec, cameraMatrix, dist)
    min_x = int(min(imgpts, key=lambda x: x[0][0]).ravel()[0])
    max_x = int(max(imgpts, key=lambda x: x[0][0]).ravel()[0])
    min_y = int(min(imgpts, key=lambda x: x[0][1]).ravel()[1])
    max_y = int(max(imgpts, key=lambda x: x[0][1]).ravel()[1])
    cv2.polylines(frame, [np.array([(min_x, min_y), (min_x, max_y), (max_x, max_y), (max_x, min_y)])], isClosed=True, color=color, thickness=thickness)

def find_chessboard(img, use_chessboard_bbox=False, chessboard_bbox = None):
    board_img = img[chessboard_bbox[1]:chessboard_bbox[3], chessboard_bbox[0]:chessboard_bbox[2]] if use_chessboard_bbox else img
    ### Step 1: Find all possible lines using SLID (Straight line detector) ###
    segments = pSLID(board_img)  # find all lines using different settings
    raw_lines = SLID(segments)
    slid_lines = slid_tendency(raw_lines)
    ### Step 2: Find all lines intersections ###
    intersec_points = laps_intersections(slid_lines)
    ### Step 3: Transformed points & lines (board -> original)
    if use_chessboard_bbox:
        points = [(int(point[0] + chessboard_bbox[0]), int(point[1]) + chessboard_bbox[1]) for point in intersec_points]
        lines = [[[line[0][0] + chessboard_bbox[0], line[0][1] + chessboard_bbox[1]], [line[1][0] + chessboard_bbox[0], line[1][1] + chessboard_bbox[1]]] for line in slid_lines]
    else:
        points = [(int(point[0]), int(point[1])) for point in intersec_points]
        lines = slid_lines
    # for point in intersec_points:
    #     if use_chessboard_bbox: point = (int(point[0] + chessboard_bbox[0]), int(point[1]) + chessboard_bbox[1])  # transfrom back from board_img -> img
    #     else: point = (int(point[0]), int(point[1]))
    #     points.append(point)
    #     lines = []
    #     for line in slid_lines:
    #         if use_chessboard_bbox: line = [[line[0][0] + chessboard_bbox[0], line[0][1] + chessboard_bbox[1]], [line[1][0] + chessboard_bbox[0], line[1][1] + chessboard_bbox[1]]]  # transfrom back from board_img -> img
    #         lines.append(line)

    ### Step 4: Filter for intersection of mesh grid ###
    points = [(int(point[0]), int(point[1])) for point in LAPS(img, points)]
    inner_points = LLR(img, points, lines)
    rvec, tvec = llr_pad(inner_points, img)
    return rvec, tvec

from module89.srv import ChessboardDetection, ChessboardPose
from geometry_msgs.msg import Pose
from std_msgs.msg import UInt16MultiArray
from cv_bridge import CvBridge

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor

import time
from threading import Thread

class FindChessboardPoseService(Node):
    def __init__(self):
        super().__init__('find_chessboard_pose_service')
        self.chessboard_locator_srv = self.create_service(ChessboardPose, 'chessboard_locator', self.findpose_callback)  # CHANGE
        self.chessboard_detection_cli = self.create_client(ChessboardDetection, 'chessboard_detection')
        while not self.chessboard_detection_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service \'ChessboardDectection\' not available, waiting ...')
        self.get_logger().info('Service \'ChessboardDetection\' founded')

        self.req_detection = ChessboardDetection.Request()

        self.bridge = CvBridge()

        self.executor = MultiThreadedExecutor(num_threads=4)

    def findpose_callback(self, request, response):
        image = self.bridge.imgmsg_to_cv2(request.img, 'bgr8')
        self.get_logger().info("AAAAAAAAAAAAAAAA")
        future = self.send_request(request.img)
        self.get_logger().info("CCCCCCCCCCCCCCCC")
        result = future.result()

        if len(result.bbox.data) == 0:
            response.valid = False
            return response
        else:
            rvec, tvec = find_chessboard(image, use_chessboard_bbox=True, chessboard_bbox=result.bbox.data)
            self.get_logger().info("EEEEEEEEEEEEEEEEEE")
            pose = Pose()
            pose.position.x, pose.position.y, pose.position.z = 0.0, 0.0, 0.0
            pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = 1.0, 0.0, 0.0, 0.0
            response.init_pose = pose
            response.valid = True
            return response
    def send_request(self, image):
        self.req_detection.img = image
        return self.chessboard_detection_cli.call_async(self.req_detection)


def main():
    rclpy.init()
    find_chessboard_service = FindChessboardPoseService()
    rclpy.spin(find_chessboard_service)
    rclpy.shutdown()


if __name__ == '__main__':
    main()