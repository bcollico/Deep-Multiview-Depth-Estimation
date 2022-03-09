import numpy as np
import torch
# from kornia import warp_perspective
from utils import print_size, unsqueeze_n

def homography_warping(K_batch, R_batch, T_batch, d_batch,
                        feature_maps, batch_size, n_views):
    # you'll need to bilinear sample here after warping

    n_vals = batch_size*n_views

    ref_idx = np.sort(np.tile(np.arange(0,batch_size*n_views,n_views),n_views))

    I = unsqueeze_n(torch.eye(3), 1)

    Kref = K_batch[ref_idx,:,:]
    Rref = R_batch[ref_idx,:,:]
    Tref = T_batch[ref_idx,:,:]
    dref = d_batch[ref_idx,:,:]
    nref = torch.unsqueeze(Rref[:,:,-1], 1) # principle axis of the reference view

    print_size(Kref); print_size(Rref); print_size(Tref); print_size(dref); print_size(nref)

    RKref = torch.matmul(torch.transpose(Rref, 1, 2), torch.inverse(Kref))

    # K_i = torch.squeeze(K[i,:,:,:], dim=0)
    # R_i = torch.squeeze(R[i,:,:,:], dim=0)
    # T_i = torch.squeeze(T[i,:,:,:], dim=0)
    # d_i = torch.squeeze(d[i,:,:,:], dim=0)

    # print_size(K_i); print_size(R_i); print_size(T_i); print_size(d_i); print_size(n_i)

    # RK_i = torch.matmul(K_i, R_i)

    RK = torch.matmul(K_batch, R_batch)

    # print_size(RK)

    Tdiff = torch.sub(I, torch.div(torch.matmul(torch.sub(Tref, T_batch), nref), d_batch))


    H_i = torch.matmul(RK, torch.matmul(Tdiff, RKref))

    # Check that first homography is identity
    print(H_i[0,:,:])
    assert torch.all(abs(H_i[0,:,:] - I[0,:,:]) < 1e-3)

    return feature_maps

def get_pixel_grids(h,w):
    """Pixel grid for bilinear interpolation. Adopted from MVSNet code."""
    xrange = torch.linspace(0.5, w.type(np.float32)-0.5, w)
    yrange = torch.linspace(0.5, h.type(np.float32)-0.5, h)

    x, y = torch.meshgrid(xrange, yrange)

    x = x.reshape(-1,1)
    y = y.reshape(-1,1)
    o = torch.ones_like(x)

    grid = torch.cat((x,y,o))

    return grid
