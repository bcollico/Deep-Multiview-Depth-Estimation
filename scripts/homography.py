import numpy as np
import torch
from kornia.geometry.transform import warp_perspective
from utils import print_size, unsqueeze_n

def homography_warping(K_batch, R_batch, T_batch, d_batch,
                        feature_maps, batch_size, n_views):
    """Differentiable homography warping from source image to reference image.
    Computed according to MVSNet paper, implemented using differentiable
    perspective warping in Kornia python module. Returns warped feature maps."""

    # unique reference indicies
    ref_idx_0 = np.arange(0,batch_size*n_views,n_views)

    # repeated reference indices to warp auxilliary views
    ref_idx = np.sort(np.tile(ref_idx_0,n_views))

    # identity matrix for contructing homography
    I = unsqueeze_n(torch.eye(3), 1)

    # get camera parameters for reference views
    Kref = K_batch[ref_idx,:,:] # intrinsics
    Rref = R_batch[ref_idx,:,:] # rotation
    Tref = T_batch[ref_idx,:,:] # translation
    nref = torch.unsqueeze(Rref[:,:,-1], 1) # principle axis of the reference view

    # compute pt. {1} of homography eqn.
    RK = torch.matmul(K_batch, R_batch)

    # compute pt. {3} of homography eqn.
    RKref = torch.matmul(torch.transpose(Rref, 1, 2), torch.inverse(Kref))

    # compute pt. {2} of homography eqn.
    Tdiff = torch.sub(I, torch.div(torch.matmul(torch.sub(Tref, T_batch), nref), d_batch))

    ## HOMOGRAPHY = Ki*Ri*(I - (t-ti)/di)*inv(R)*inv(K) = {1} * {2} * {3} ##
    H_i = torch.matmul(RK, torch.matmul(Tdiff, RKref))

    H_i[ref_idx_0] = I # set ref view homography to identity

    # compute homography on feature maps
    w_and_h = tuple(feature_maps.size()[-2:])
    warped_feature_maps = warp_perspective(feature_maps, H_i, w_and_h)

    return warped_feature_maps
