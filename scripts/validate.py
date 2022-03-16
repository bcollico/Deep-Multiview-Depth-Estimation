import numpy as np
import torch
from torch.utils.data import DataLoader
from loss import loss_fcn
from config import DEVICE

def validate(valid_data_loader:DataLoader,
          model:torch.nn.Module=None,
          optimizer=None):

    if isinstance(valid_data_loader, str):
        valid_data_loader = torch.load(valid_data_loader)

    num_trainloader = len(valid_data_loader)

    model.eval()

    torch.autograd.set_detect_anomaly(True)
    with torch.no_grad():

        batch_loss        = np.zeros(num_trainloader)
        batch_initial_acc = np.zeros(num_trainloader)
        batch_refined_acc = np.zeros(num_trainloader)

        for batch_idx, batch in enumerate(valid_data_loader):

            batch_size, n_views, ch, h, w = batch['input_img'].size()
            _         , _      , _ ,dh,dw = batch['depth_ref'].size()
                
            optimizer.zero_grad(set_to_none=True)
        
            nn_input= torch.reshape(batch['input_img'], (batch_size*n_views, ch, h, w)).to(DEVICE) # b*n, ch, h, w
            gt_depth= torch.reshape(batch['depth_ref'], (batch_size, 1, dh, dw)).to(DEVICE)       # b*n, ch, h, w
            K_batch = torch.reshape(batch['K'], (batch_size*n_views, 3, 3)) # b*n, 1, 3, 3
            R_batch = torch.reshape(batch['R'], (batch_size*n_views, 3, 3))
            T_batch = torch.reshape(batch['T'], (batch_size*n_views, 3, 1))
            d_min   = batch['d']
            d_int   = batch['d_int']

            d_min = torch.tensor(0)*d_min # error in dataset, use d_min = 0
            d_int = d_int.div(d_int) # set this to 1 so that we fully control interval from config

            initial_depth_map, refined_depth_map = model(nn_input, K_batch, 
                        R_batch, T_batch, d_min, d_int, batch_size, n_views)

            loss, initial_acc, refined_acc = loss_fcn(gt_depth, 
                                        initial_depth_map, refined_depth_map)

            # store statistics of current item in batch
            batch_loss[batch_idx] = float(loss.detach().mean())
            batch_initial_acc[batch_idx] = float(initial_acc.detach().mean())
            batch_refined_acc[batch_idx] = float(refined_acc.detach().mean())


    return batch_loss.mean(), batch_initial_acc.mean(), batch_refined_acc.mean()
