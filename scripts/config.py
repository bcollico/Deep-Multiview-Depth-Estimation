import torch
from numpy import int64, floor, array

ACC_THRESH = 0.05 # correct depth is within ACC_THESH percent of true depth

D_SCALE = 21.25 # scaling factor for depth interval to control distance between front planes
D_NUM = 20   # number of planes to warp images into

N_DEPTH_EST = torch.tensor(6) # how many depth estimates to fuse in initial depth map

# parameters for feature extraction net
DIM_REDUCE = 4
IN_H = 512
IN_W = 640

FEAT_H = int(IN_H / DIM_REDUCE)
FEAT_W = int(IN_W / DIM_REDUCE)

# parameters for 3D cost reg net
PAD = tuple(int64(floor(array([D_NUM, FEAT_H, FEAT_W])/2) + 1)) # dim/2 + 1
OUTPAD = tuple((array([D_NUM, FEAT_H, FEAT_W]) + 1) % 2) # 0 for odd, 1 for even  


DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')