import os
import glob
import shutil
save_path = "/media/teera/HDD1TB/dataset/classification/chess"
vdo_path_list = sorted(glob.glob(os.path.join(save_path, '*.avi')))
vdo_list = [os.path.basename(vdo_path_list[i]) for i in range(len(vdo_path_list))]
vdo_num_list = [int(os.path.basename(vdo_path_list[i])[:-4]) for i in range(len(vdo_path_list))]
vdo_path_list = [x for _, x in sorted(zip(vdo_num_list, vdo_path_list))]
vdo_list = [x for _, x in sorted(zip(vdo_num_list, vdo_list))]
for i in range(int(len(vdo_path_list)/3)):
    os.mkdir(os.path.join(save_path, str(i)))
    shutil.move(vdo_path_list[3 * i + 0], os.path.join(save_path, str(i), vdo_list[3 * i + 0]))
    shutil.move(vdo_path_list[3 * i + 1], os.path.join(save_path, str(i), vdo_list[3 * i + 1]))
    shutil.move(vdo_path_list[3 * i + 2], os.path.join(save_path, str(i), vdo_list[3 * i + 2]))
