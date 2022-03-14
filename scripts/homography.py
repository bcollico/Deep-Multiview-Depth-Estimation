import torch
from kornia.geometry.transform import warp_perspective
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
    d_batch = torch.tile(d_batch_0, (n_views,1,1,1))

    # separate reference indices and view indices
    ref_idx_0 = torch.arange(0, N, n_views) #(batch_size,)

    # repeated indices to warp feature maps - there are <batch_size> reference 
    # views that need to be repeated <n_views>*<d_num> times 
    ref_idx, _ = torch.sort(torch.tile(ref_idx_0, (1,n_views)))
    ref_idx = ref_idx.squeeze(0)
    # img_idx = torch.tensor([i for i in torch.arange(0,N) if i not in ref_idx_0])
    img_idx = torch.arange(0,N) # lazy way -- might need to exclude ref views


    # identity matrix for contructing homography: shape (1, d, 3, 3)
    I = torch.eye(3).to(DEVICE).unsqueeze(0).unsqueeze(1).expand((-1, d_num, -1, -1))

    # get camera parameters for reference views: shape (batch_size*(n_views-1), d, 3, 3)
    K_ref   = K_batch[ref_idx,:,:].to(DEVICE).unsqueeze(1).expand((-1, d_num, -1, -1))  # intrinsic
    R_ref_0 = R_batch[ref_idx,:,:].to(DEVICE)
    T_ref_0 = T_batch[ref_idx,:,:].to(DEVICE)

    R_ref = R_ref_0.unsqueeze(1).expand((-1, d_num, -1, -1))  # rotation
    T_ref = -torch.matmul(R_ref_0.transpose(-2,-1),T_ref_0).unsqueeze(1).expand((-1, d_num, -1, -1))  # translation
    n_ref = R_ref[:, :, :,2].unsqueeze(2) # principle axis of the reference view (b*(n-1) d, 1, 3)

    # get camera parameters for other views: shape (batch_size*(n_views-1), d, 3, 3)
    K_view   = K_batch[img_idx,:,:].to(DEVICE).unsqueeze(1).expand((-1, d_num, -1, -1)) # intrinsics
    R_view_0 = R_batch[img_idx,:,:].to(DEVICE)
    T_view_0 = T_batch[img_idx,:,:].to(DEVICE)


    R_view = R_view_0.unsqueeze(1).expand((-1, d_num, -1, -1)) # rotation
    T_view = -torch.matmul(R_view_0.transpose(-2,-1), T_view_0).unsqueeze(1).expand((-1, d_num, -1, -1)) # translation
    
    # compute pt. {1} of homography eqn. -- on DEVICE
    RK = torch.matmul(K_view, R_view)

    # compute pt. {3} of homography eqn. -- on DEVICE
    RK_ref = torch.matmul(torch.transpose(R_ref, -2, -1),
                          torch.inverse(K_ref))

    # compute pt. {2} of homography eqn. -- on DEVICE
    t_diff_0      = torch.sub(T_view, T_ref)
    t_dot_n       = torch.matmul(t_diff_0, n_ref)
    t_dot_n_div_d = torch.div(t_dot_n, d_batch) 
    Tdiff         = torch.sub(I, t_dot_n_div_d) 

    ## HOMOGRAPHY = Ki*Ri*(I - (t-ti)/di)*inv(R)*inv(K) = {1} * {2} * {3} ##
    # does not include identity mapping to save on compute
    H_i = torch.matmul(RK, torch.matmul(Tdiff, RK_ref))

    # compute homography on feature maps
    w_and_h = tuple(feature_maps.size()[-2:])
    warped_feature_maps = feature_maps.detach().clone().unsqueeze(2).to(DEVICE).expand((-1,-1,d_num,-1,-1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Kornia.warp_perspective triggers a warning in a pytorch function
        # during interpolation
        for d in range(d_num):
            warped_feature_maps[img_idx, :, d] = \
                warp_perspective(feature_maps[img_idx], H_i[:,d,:,:], w_and_h)

    return warped_feature_maps, d_batch_0, ref_idx_0

if __name__ == '__main__':
    from data import DtuTrainDataset
    import matplotlib.pyplot as plt

    train_data_loader = torch.load('test_dataloader')
    for batch_idx, batch in enumerate(train_data_loader):
        
        print("Running Homography")
        
        batch_size, n_views, ch, h, w = batch['input_img'].size()

        input= batch['input_img'][0,:,:,:,:].to(DEVICE)

        K_batch = batch['K'][0,:,:,:]
        R_batch = batch['R'][0,:,:,:]
        T_batch = batch['T'][0,:,:,:]
        d_min   = batch['d'][0]
        d_int   = batch['d_int'][0]

        out, _, _ = homography_warping(K_batch, R_batch, T_batch, d_min, d_int, input, batch_size, n_views)

        fig = plt.figure()
        plt.axis('off')
        plt.subplots_adjust(hspace=0.25)
        # plt.title("Original")
        for i in range(n_views):
            # print(D_NUM*i + 1)
            img = input[i,:,:,:]
            img = img-img.min()
            img = img/img.max()
            fig.add_subplot(n_views, D_NUM+1, (D_NUM+1)*i+1)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                plt.imshow( img.permute(1, 2, 0).cpu() )
            if i == 0:
                plt.title('Original')
            # plt.title("Depth {:f}".format(float((d_min).cpu())))
            plt.axis('off')

        for d in range(D_NUM):
            for i in range(n_views):
                # print( d*n_views + 2 + i)
                img = out[i,:,d,:,:]
                img = img-img.min()
                img = img/img.max()
                fig.add_subplot(n_views, D_NUM+1, (D_NUM+1)*i+2 + d)
                plt.imshow( img.permute(1, 2, 0).cpu() )
                if i == 0:
                    plt.title("D = {:g}".format(float((d_min + D_SCALE*d_int*d).cpu())))
                # plt.title('View {:d}'.format(i))
                plt.axis('off')
        plt.show()
        del fig

