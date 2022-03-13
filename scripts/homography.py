import torch
from kornia.geometry.transform import warp_perspective
from utils import print_gpu_memory, print_size, unsqueeze_n
from config import D_SCALE, D_NUM, DEVICE
import warnings

def homography_warping(K_batch:torch.Tensor, 
                       R_batch:torch.Tensor, 
                       T_batch:torch.Tensor, 
                       d_min:torch.Tensor, 
                       d_int:torch.Tensor,
                       feature_maps:torch.Tensor, 
                       batch_size:int, 
                       n_views:int, 
                       d_num:int=D_NUM):
    """Differentiable homography warping from source image to reference image.
    Computed according to MVSNet paper, implemented using differentiable
    perspective warping in Kornia python module. Returns warped feature maps.
    Input  <batch_size * n_views, ch, w, h> size feature map
    Output <batch_size * n_views*depth_num, ch, w, h> warped feature maps """

    N = batch_size*n_views

    # create batch depth mapping of shape: (d_num, batch_size*n_views, 1, 1)
    d_num_tensor = torch.arange(d_num).reshape(1, d_num, 1, 1)
    d_batch_0 = (d_min + D_SCALE*d_int * d_num_tensor).to(DEVICE)
    d_batch = torch.tile(d_batch_0, (n_views-1,1,1,1))

    # separate reference indices and view indices
    ref_idx_0 = torch.arange(0, N, n_views) #(batch_size,)

    # repeated indices to warp feature maps - there are <batch_size> reference 
    # views that need to be repeated <n_views>*<d_num> times 
    ref_idx, _ = torch.sort(torch.tile(ref_idx_0, (1,n_views-1)))
    ref_idx = ref_idx.squeeze(0)
    # img_idx   = torch.tile(torch.arange(0, N), d_num) # lazy way -- might need to exclude ref views
    img_idx = torch.tensor([i for i in torch.arange(0,N) if i not in ref_idx_0])

    # print(img_idx, img_idx.size())


    # identity matrix for contructing homography: shape (1, d, 3, 3)
    I = torch.eye(3).to(DEVICE).unsqueeze(0).unsqueeze(1).expand((-1, d_num, -1, -1))

    # get camera parameters for reference views: shape (batch_size*(n_views-1), d, 3, 3)
    K_ref = K_batch[ref_idx,:,:].to(DEVICE).unsqueeze(1).expand((-1, d_num, -1, -1))  # intrinsic
    R_ref = R_batch[ref_idx,:,:].to(DEVICE).unsqueeze(1).expand((-1, d_num, -1, -1))  # rotation
    T_ref = T_batch[ref_idx,:,:].to(DEVICE).unsqueeze(1).expand((-1, d_num, -1, -1))  # translation
    n_ref = R_ref[:, :, :,-1].unsqueeze(2) # principle axis of the reference view (b*(n-1) d, 1, 3)

    # get camera parameters for other views: shape (batch_size*(n_views-1), d, 3, 3)
    K_view = K_batch[img_idx,:,:].to(DEVICE).unsqueeze(1).expand((-1, d_num, -1, -1)) # intrinsics
    R_view = R_batch[img_idx,:,:].to(DEVICE).unsqueeze(1).expand((-1, d_num, -1, -1)) # rotation
    T_view = T_batch[img_idx,:,:].to(DEVICE).unsqueeze(1).expand((-1, d_num, -1, -1)) # translation

    # print("Camera Parameters Moved to GPU with Shapes", K_ref.size(), K_view.size())
    # print_gpu_memory()
    
    # compute pt. {1} of homography eqn. -- on DEVICE
    RK = torch.matmul(K_view, R_view)

    # compute pt. {3} of homography eqn. -- on DEVICE
    RK_ref = torch.matmul(torch.transpose(R_ref, R_ref.size()[-2], R_ref.size()[-1]),
                          torch.inverse(K_ref))

    # print(RK_ref.size())
    # print(RK.size())
    # print(I.size())
    # print(T_ref.size())
    # print(T_view.size())
    # print(n_ref.size())
    # print(d_batch.size())

    # compute pt. {2} of homography eqn. -- on DEVICE
    Tdiff = torch.sub(I, torch.div(torch.matmul(torch.sub(T_ref, T_view), n_ref), d_batch))

    ## HOMOGRAPHY = Ki*Ri*(I - (t-ti)/di)*inv(R)*inv(K) = {1} * {2} * {3} ##
    # does not include identity mapping to save on compute
    H_i = torch.matmul(RK, torch.matmul(Tdiff, RK_ref))

    # print("Computed Homographies with Size ", H_i.size())
    # print_gpu_memory()

    # H_i[ref_idx_0] = I # set ref view homography to identity

    # compute homography on feature maps
    w_and_h = tuple(feature_maps.size()[-2:])
    warped_feature_maps = feature_maps.clone().unsqueeze(2).to(DEVICE).expand((-1,-1,d_num,-1,-1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for d in range(d_num):
            # print(feature_maps[img_idx].size())
            # print(warped_feature_maps[d, img_idx].size())
            # print(H_i[d,:,:,:].size())
            warped_feature_maps[img_idx, :, d] = \
                warp_perspective(feature_maps[img_idx], H_i[:,d,:,:], w_and_h)

    # warped_feature_maps_new_dim = warped_feature_maps.transpose(0,1).transpose(1,2)

    return warped_feature_maps, d_batch_0, ref_idx_0
