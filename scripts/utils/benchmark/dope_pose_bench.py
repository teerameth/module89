import os.path

import numpy as np
import glob

report_path = "/media/teera/ROGESD/model/belief/chessboard_mono_6_stage/report"
# report_path = "/media/teera/ROGESD/model/belief/chessboard_blender_6_stage/report"
# report_path = "/media/teera/ROGESD/model/belief/chessboard_blender_3_stage/report"
npy_list = glob.glob(os.path.join(report_path, '*.npy'))
score_table = {}
for i in range(1, 60):
    rvec_file = os.path.join(report_path, 'net_epoch_%d_rvec.npy'%(i))
    tvec_file = os.path.join(report_path, 'net_epoch_%d_tvec.npy'%(i))
    if not os.path.exists(rvec_file): continue
    if not os.path.exists(tvec_file): continue
    npy_rvec = np.load(rvec_file)
    npy_tvec = np.load(tvec_file)
    npy_rvec = np.abs(npy_rvec)
    npy_tvec = np.abs(npy_tvec)
    print(i, np.sum(npy_rvec), np.sum(npy_tvec))
    score_table[i] = (np.sum(npy_rvec), np.sum(npy_tvec))
print("Minimum diff_rvec")
print(min(score_table.items(), key=lambda k: k[1][0]))
print("Minimum diff_tvec")
print(min(score_table.items(), key = lambda k : k[1][1]))