from doctest import IGNORE_EXCEPTION_DETAIL
import time
from os.path import join
import numpy as np
import torch
import torch.nn.functional as f
from torch.utils.data import DataLoader
from loss import loss_fcn
from utils import *
from model import *
from tqdm import tqdm
from config import DEVICE, D_NUM, D_SCALE
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

VISUALIZE = True


def test(epochs:int, 
          test_data_loader:DataLoader,
          save_path:str=join('.','checkpoints'),
          checkpoint:str=None,
          model:torch.nn.Module=None,
          start_epoch:int=0):

    if isinstance(test_data_loader, str):
        test_data_loader = torch.load(test_data_loader)

    num_trainloader = len(test_data_loader)

    if model is not None:
        print("Using Passed In Model.")
        epoch_loss        = [np.zeros(num_trainloader) for i in range(epochs)]
        epoch_initial_acc = [np.zeros(num_trainloader) for i in range(epochs)]
        epoch_refined_acc = [np.zeros(num_trainloader) for i in range(epochs)]
    elif checkpoint is not None:
            print('Loading Checkpoint...')
            out = load_from_ckpt(checkpoint, epochs)
            model, start_epoch, \
                epoch_loss, epoch_initial_acc, epoch_refined_acc = out
    else:
        raise Exception("Test requires a model input")

    print('Start Training From Epoch #{:d}'.format(start_epoch))

    # print_gpu_memory()
    print('Model sent to device: ', DEVICE)
    model = model.to(DEVICE)
    # print_gpu_memory()

    id_str = "test_"+str(int(time.time()))

    model.eval()

    total = num_trainloader
    with tqdm(total=total) as pbar:
        for epoch in range(start_epoch, epochs):
            # with torch.no_grad():
            print("----- TESTING EPOCH #{:d} -----".format(epoch+1+start_epoch))

            model.train()
            epoch_start_time = time.time()
            current_time     = time.time()

            batch_loss        = np.zeros(num_trainloader)
            batch_initial_acc = np.zeros(num_trainloader)
            batch_refined_acc = np.zeros(num_trainloader)

            with torch.no_grad():
                for batch_idx, batch in enumerate(test_data_loader):

                    batch_size, n_views, ch, h, w = batch['input_img'].size()
                    _         , _      , _ ,dh,dw = batch['depth_ref'].size()
                
                    nn_input= torch.reshape(batch['input_img'], (batch_size*n_views, ch, h, w)).to(DEVICE) # b*n, ch, h, w
                    gt_depth= torch.reshape(batch['depth_ref'], (batch_size, 1, dh, dw)).to(DEVICE)       # b*n, ch, h, w
                    K_batch = torch.reshape(batch['K'], (batch_size*n_views, 3, 3)) # b*n, 1, 3, 3
                    R_batch = torch.reshape(batch['R'], (batch_size*n_views, 3, 3))
                    T_batch = torch.reshape(batch['T'], (batch_size*n_views, 3, 1))
                    d_min   = batch['d']
                    d_int   = batch['d_int']

                    # d_min = torch.tensor(0)*d_min # error in dataset, use d_min = 0
                    d_int = d_int.div(d_int) # set this to 1 so that we fully control interval from config

                    # print("NN Input sent to ", DEVICE, "with shape: ", nn_input.size())
                    # print_gpu_memory()

                    initial_depth_map, refined_depth_map = model(nn_input, K_batch, 
                                R_batch, T_batch, d_min, d_int, batch_size, n_views)

                    loss, initial_acc, refined_acc = loss_fcn(gt_depth, 
                                            initial_depth_map, refined_depth_map)


                    # store statistics of current item in batch
                    batch_loss[batch_idx] = float(loss.mean())
                    batch_initial_acc[batch_idx] = float(initial_acc.mean())
                    batch_refined_acc[batch_idx] = float(refined_acc.mean())
                    
                    current_time = time.time()
                    
                    if True or (batch_idx+1) % 14 == 0:
                        print(
                            "Epoch: #{:d} Batch: {:d}/{:d}  Time: {:g}\t"
                            "Loss(curr/avg) {:.4f}/{:.4f}\t"
                            "Acc 1(curr/avg) {:.4f}/{:.4f}\t"
                            "Acc 2(curr/avg) {:.4f}/{:.4f}\t"
                            .format(epoch, 
                                    batch_idx+1, 
                                    num_trainloader, 
                                    current_time - epoch_start_time,
                                    batch_loss[batch_idx],
                                    np.mean(batch_loss[:(batch_idx+1)]),
                                    batch_initial_acc[batch_idx], 
                                    np.mean(batch_initial_acc[:(batch_idx+1)]),
                                    batch_refined_acc[batch_idx], 
                                    np.mean(batch_refined_acc[:(batch_idx+1)])
                                    )
                            )

                        if VISUALIZE:
                            for b in range(batch_size):
                                visualize_depth(batch['input_img'][b,0].unsqueeze(0),
                                                gt_depth[b].unsqueeze(0),
                                                initial_depth_map[b].unsqueeze(0),
                                                refined_depth_map[b].unsqueeze(0))
                        pbar.update(1)

            # save stats for each epoch
            epoch_loss[epoch]        = batch_loss
            epoch_initial_acc[epoch] = batch_initial_acc
            epoch_refined_acc[epoch] = batch_refined_acc

            # if (epoch) % 1 == 0:
            #     torch.save({
            #                 'epoch': epoch,
            #                 'model_state_dict': model.state_dict(),
            #                 'loss': epoch_loss,
            #                 'acc_1': epoch_initial_acc,
            #                 'acc_2': epoch_refined_acc,
            #                 }, join(save_path, id_str+"_"+str(epoch)))

    return epoch_loss, epoch_initial_acc, epoch_refined_acc

def visualize_depth(rgb:torch.Tensor, gt:torch.Tensor, initial:torch.Tensor, refined:torch.Tensor):

    import matplotlib.pyplot as plt
    def plot_depth(img:torch.Tensor, fig, idx, label, maxmax):
        img = img.cpu()
        # img = img-img.min()
        # img = img/img.max()
        img = img/maxmax
        fig.add_subplot(2, 2, idx+1)
        plt.title(label)
        plt.axis('off')
        depth = plt.imshow( (img[0] * 255).type(torch.ByteTensor),  cmap='plasma', vmin=0, vmax=255)
        cbar = plt.colorbar(depth)
        cbar.set_label('Depth Value')
        return fig

    # ignore invalid points in the depth map
    for i in range(gt.size()[0]):

        mask = torch.eq(gt, torch.tensor(0.0).to(DEVICE)).float()
        p_valid = torch.neg(mask - torch.tensor(1.0)).sum((1,2,3))

        fig = plt.figure()
        plt.axis('off')
        plt.suptitle("Depth Comparison, {:d} Valid Points".format(int(p_valid)))
        plt.subplots_adjust(hspace=0.1)

        img = rgb[i*3]
        img = img-img.min()
        img = img/img.max()
        fig.add_subplot(2, 2, 1)
        plt.title("Original")
        plt.axis('off')
        plt.imshow( img.permute(1, 2, 0).cpu() )

        maxmax = torch.max(torch.max((gt)[i].max(), initial[i].max()), refined[i].max()).cpu()

        img = (gt)[i]
        fig = plot_depth(img, fig, 1, "Truth", maxmax)

        # print(img.min(), img.max())

        img = initial[i]
        fig = plot_depth(img, fig, 2, "Initial", maxmax)

        # print(torch.abs(initial[i]-gt[i]).min(), torch.abs(initial[i]-gt[i]).max())

        img = refined[i]
        fig = plot_depth(img, fig, 3, "Refined", maxmax)

        # print(torch.abs(refined[i]-gt[i]).min(), torch.abs(refined[i]-gt[i]).max())

        plt.show()

        del fig

def init_test_model(epochs=10):
    model = MVSNet()
    start_epoch = 0

    return model, start_epoch

def load_from_ckpt(ckpt, epochs):

    model, _ = init_test_model(epochs)

    checkpoint = torch.load(ckpt)

    model.load_state_dict(checkpoint["model_state_dict"])
    loss  = checkpoint['loss']
    acc_1 = checkpoint['acc_1']
    acc_2 = checkpoint['acc_2']
    
    return model,  0, loss, acc_1, acc_2
        

if __name__ =="__main__":
    # really strange behavior, need to include the dataset class
    # here in order to load the saved dataset properly
    from data import DtuTrainDataset
    model, _ = init_test_model()
    loss, acc_1, acc_2, = test(epochs=1, model=None, 
    checkpoint=join('.','checkpoints','train_1647468534_19_9'), test_data_loader="test_dataloader")

