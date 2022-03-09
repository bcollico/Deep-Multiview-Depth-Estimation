import torch
import torch.nn.functional as f
from loss import loss_fcn
from utils import *
from model import *


def train(epochs, 
          train_data_loader,
          lr=0.001,
          save_path='/checkpoints',
          device=None,
          checkpoint=None,
          model=None,
          start_epoch=0):

    """This script is only used for checking the individual components of 
    the MVSNet. Each component has passed an in/out check for shape."""

    if isinstance(train_data_loader, str):
        train_data_loader = torch.load(train_data_loader)

    num_trainloader = len(train_data_loader)

    device = torch.device('cuda:0' if device == 'cuda' else 'cpu')
    
    feature_model = FeatureEncoder()

    cost_volume_model = CostVolumeReg()

    depth_refine_model = DepthRefinement()

    mvsnet_model = MVSNet()


    epoch_loss = []
    batch_loss = []
    sum_loss   = 0

    epoch_accuracy = []

    for epoch in range(start_epoch, epochs):

        feature_model.train()
        cost_volume_model.train()
        depth_refine_model.train()

        for batch_idx, batch in enumerate(train_data_loader):
            # camera = dict()
            # for sidx in range(batch['input_img'].size()[0]):
            # print(batch['input_img'][idx,:,:,:,:].size())
            # optimizer.zero_grad()

            # camera = batch['camera'] # dictionary: each component is [b, 1, 1, x, y]

            # nn_input  = torch.cat((image_ref, image_one, image_two), dim=0)
            # nn_input  = batch['input_img'][sidx,:,:,:,:]
            # gt_depth  = batch['depth_ref'][sidx,:,:,:,:]

            # camera['K'] = batch['K'][sidx,:,:,:,:]
            # camera['R'] = batch['R'][sidx,:,:,:,:]
            # camera['T'] = batch['T'][sidx,:,:,:,:]
            # camera['d'] = batch['d'][sidx,:,:,:,:]

            # image_ref = torch.unsqueeze(nn_input[0,:,:,:], dim=0)

            # print("Batch {:d}: Feature Map Input Size ".format(batch_idx), nn_input.size())

            # nn_output = feature_model(nn_input)
            # print("Batch {:d}: Feature Map Output Size ".format(batch_idx), nn_output.size())

            # cv_output = cost_volume_model(torch.unsqueeze(nn_output[0,:,:,:], dim=0))
            
            # print("Batch {:d}: Cost Volume Output Size ".format(batch_idx), cv_output.size())

            # resized_ref = f.interpolate(image_ref, size=(128,160))

            # depth_input = torch.cat((resized_ref, cv_output), dim=1)

            # print("Batch {:d}: Depth Network Input Size ".format(batch_idx), depth_input.size())

            # depth_output = depth_refine_model(depth_input)

            # print("Batch {:d}: Depth Network Output Size ".format(batch_idx), depth_output.size())

            batch_size, n_views, ch, h, w = batch['input_img'].size()
            _, _, _, h_gt, w_gt = batch['depth_ref'].size()

            print_size(batch['depth_ref'])
        
            nn_input= torch.reshape(batch['input_img'], (batch_size*n_views, ch, h, w)) # b*n, ch, h, w
            gt_depth= torch.reshape(batch['depth_ref'], (batch_size, 1, h_gt, w_gt))    # b*n, ch, h, w
            K_batch = torch.reshape(batch['K'], (batch_size*n_views, 3, 3)) # b*n, 1, 3, 3
            R_batch = torch.reshape(batch['R'], (batch_size*n_views, 3, 3))
            T_batch = torch.reshape(batch['T'], (batch_size*n_views, 3, 1))
            d_batch = torch.reshape(batch['d'], (batch_size*n_views, 1, 1))

            print("Batch {:d}:MVS Network Input Image Size ".format(batch_idx), nn_input.size())
            print("Batch {:d}:MVS Network Input K     Size ".format(batch_idx), K_batch.size())
            print("Batch {:d}:MVS Network Input R     Size ".format(batch_idx), R_batch.size())
            print("Batch {:d}:MVS Network Input T     Size ".format(batch_idx), T_batch.size())
            print("Batch {:d}:MVS Network Input d     Size ".format(batch_idx), d_batch.size())

            mvs_output = mvsnet_model(nn_input, K_batch, R_batch, T_batch, d_batch, batch_size, n_views)

            print("Batch {:d}:MVS Network Output Size ".format(batch_idx), mvs_output.size())
            

if __name__ =="__main__":
    # really strange behavior, need to include the dataset class
    # here in order to load the saved dataset properly
    from data import DtuTrainDataset
    train(epochs=1, train_data_loader="test_dataloader")
