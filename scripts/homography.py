import torch
from kornia.geometry.transform import warp_perspective
from utils import print_size, unsqueeze_n

def homography_warping(K_batch, R_batch, T_batch, d_min, d_int,
                        feature_maps, batch_size, n_views, d_num=5):
    """Differentiable homography warping from source image to reference image.
    Computed according to MVSNet paper, implemented using differentiable
    perspective warping in Kornia python module. Returns warped feature maps.
    Input  <batch_size * n_views, ch, w, h> size feature map
    Output <batch_size * n_views*depth_num, ch, w, h> warped feature maps """

    N = batch_size*n_views

    # create batch depth mapping for batch_size*n_views*d_num 
    d_batch = torch.tile(torch.linspace(d_min, d_min+d_int*d_num, steps=d_num), N)
    d_batch = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(d_batch, dim=1), dim=1))

    print(d_batch.size())

    # separate reference indices and view indices
    ref_idx_0 = torch.tile(torch.arange(0, N, d_num))
    # img_idx_0 = torch.tensor([i for i in torch.arange(0,N) if i not in ref_idx_0])

    # repeated indices to warp feature maps - there are <batch_size> reference 
    # views that need to be repeated <n_views>*<d_num> times 
    ref_idx = torch.sort(torch.tile(ref_idx_0, n_views))
    img_idx   = torch.tile(torch.arange(0, N), d_num) # lazy way -- might need to exclude ref views


    # identity matrix for contructing homography
    I = unsqueeze_n(torch.eye(3), 1)

    # get camera parameters for reference views
    K_ref = K_batch[ref_idx,:,:] # intrinsics
    R_ref = R_batch[ref_idx,:,:] # rotation
    T_ref = T_batch[ref_idx,:,:] # translation
    n_ref = torch.unsqueeze(R_ref[:,:,-1], 1) # principle axis of the reference view

    K_view = K_batch[img_idx,:,:] # intrinsics
    R_view = R_batch[img_idx,:,:] # rotation
    T_view = T_batch[img_idx,:,:] # translation

    print(K_ref.size())
    print(K_view.size())
    
    # compute pt. {1} of homography eqn.
    RK = torch.matmul(K_view, R_view)

    # compute pt. {3} of homography eqn.
    RK_ref = torch.matmul(torch.transpose(R_ref, 1, 2), torch.inverse(K_ref))

    # compute pt. {2} of homography eqn.
    Tdiff = torch.sub(I, torch.div(torch.matmul(torch.sub(T_ref, T_view), n_ref), d_batch))

    ## HOMOGRAPHY = Ki*Ri*(I - (t-ti)/di)*inv(R)*inv(K) = {1} * {2} * {3} ##
    H_i = torch.matmul(RK, torch.matmul(Tdiff, RK_ref))

    H_i[ref_idx_0] = I # set ref view homography to identity

    # compute homography on feature maps
    w_and_h = tuple(feature_maps.size()[-2:])
    warped_feature_maps = warp_perspective(feature_maps, H_i, w_and_h)

    return warped_feature_maps
