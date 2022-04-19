import json
import cv2
import os
import glob
target_folder = "/media/teera/ROGESD/dataset/dope/output/chessboard_mono_emptyboard"
# target_folder = "/media/teera/ROGESD/dataset/dope/output_blender/chessboard"
ndds_list = sorted(glob.glob(os.path.join(target_folder, "*.json")))
image_list = sorted(glob.glob(os.path.join(target_folder, "*.png")))

for i in range(len(ndds_list)):
    ndds_file = ndds_list[i]
    filename1 = os.path.basename(ndds_file)
    filename2 = os.path.basename(image_list[i])
    if filename1[:-5] != filename2[:-4]: print(filename1, filename2)
    # print(ndds_file)
    ndds = json.load(open(ndds_file))
    points = ndds['objects'][0]['projected_cuboid']
    image = cv2.imread(image_list[i])
    for point in points:
        cv2.circle(image, (int(point[0]), int(point[1])), 3, (0, 255, 0), -1)
        cv2.imshow("A", image)
        cv2.waitKey(0)
    # new_points = [points[1], points[2], points[5], points[6]]
    # ndds['objects'][0]['projected_cuboid'] = new_points
    # with open(ndds_file, 'w') as fp:
    #     json.dump(ndds, fp, indent=4, sort_keys=True)