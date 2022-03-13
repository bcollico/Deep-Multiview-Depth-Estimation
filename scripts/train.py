import time
import numpy as np
import torch
import torch.nn.functional as f
from loss import loss_fcn
from utils import *
from model import *
from tqdm import tqdm
from config import DEVICE


def train(epochs, 
          train_data_loader,
          lr=0.001,
          save_path='/checkpoints',
          checkpoint=None,
          model=None,
          start_epoch=0):

    if isinstance(train_data_loader, str):
        train_data_loader = torch.load(train_data_loader)

    num_trainloader = len(train_data_loader)

    if model is not None:
        print('Using Input Model...')
        optimizer = torch.optim.Adam(model.parameters)
    else:
        if checkpoint is not None:
            print('Loading Checkpoint...')
            model, optimizer, start_epoch = load_from_ckpt(checkpoint, epochs)
        else:
            print('Initializing Model...')
            model, optimizer, start_epoch = init_training_model(epochs, lr)

    print('Start Training From Epoch #{:d}'.format(start_epoch))

    # print_gpu_memory()
    print('Model sent to device: ', DEVICE)
    model = model.to(DEVICE)
    # print_gpu_memory()

    # average statistics for each epoch in range(epochs)
    epoch_loss        = np.array([])
    epoch_initial_acc = np.array([])
    epoch_refined_acc = np.array([])

    total = epochs
    with tqdm(total=total) as pbar:
        for epoch in range(start_epoch, epochs):
            # with torch.no_grad():
            print("----- TRAINING EPOCH #{:d} -----".format(epoch))

            append_zero(epoch_loss)
            append_zero(epoch_initial_acc)
            append_zero(epoch_refined_acc)

            model.train()
            epoch_start_time = time.time()
            current_time     = time.time()

            for batch_idx, batch in enumerate(train_data_loader):

                batch_size, n_views, ch, h, w = batch['input_img'].size()
                _         , _      , _ ,dh,dw = batch['depth_ref'].size()

                if batch_idx == 0:
                    # statistics for each element in this batch
                    batch_loss        = np.zeros(batch_size)
                    batch_initial_acc = np.zeros(batch_size)
                    batch_refined_acc = np.zeros(batch_size)

                append_zero(batch_loss)
                append_zero(batch_initial_acc)
                append_zero(batch_refined_acc)

                optimizer.zero_grad()
            
                nn_input= torch.reshape(batch['input_img'], (batch_size*n_views, ch, h, w)).to(DEVICE) # b*n, ch, h, w
                gt_depth= torch.reshape(batch['depth_ref'], (batch_size, 1, dh, dw)).to(DEVICE)       # b*n, ch, h, w
                K_batch = torch.reshape(batch['K'], (batch_size*n_views, 3, 3)) # b*n, 1, 3, 3
                R_batch = torch.reshape(batch['R'], (batch_size*n_views, 3, 3))
                T_batch = torch.reshape(batch['T'], (batch_size*n_views, 3, 1))
                d_min   = batch['d']
                d_int   = batch['d_int']

                # print("NN Input sent to ", DEVICE, "with shape: ", nn_input.size())
                # print_gpu_memory()

                initial_depth_map, refined_depth_map = model(nn_input, K_batch, 
                            R_batch, T_batch, d_min, d_int, batch_size, n_views)

                loss, initial_acc, refined_acc = loss_fcn(gt_depth, 
                                           initial_depth_map, refined_depth_map)

                if epoch > 0:
                    loss.backward()
                    optimizer.step()

                # store statistics of current item in batch
                batch_loss[batch_idx] = float(loss.mean())
                batch_initial_acc[batch_idx] = float(initial_acc.mean())
                batch_refined_acc[batch_idx] = float(refined_acc.mean())
                
                current_time = time.time()
                
                if batch_idx % 20 == 0:
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
                                np.mean(batch_loss[:batch_idx]),
                                batch_initial_acc[batch_idx], 
                                np.mean(batch_initial_acc[:batch_idx]),
                                batch_refined_acc[batch_idx], 
                                np.mean(batch_refined_acc[:batch_idx])
                                )
                        )

            # add average stats for the most recent batch
            epoch_loss[-1]        += np.mean(batch_loss)
            epoch_initial_acc[-1] += np.mean(batch_initial_acc)
            epoch_refined_acc[-1] += np.mean(batch_refined_acc)

            # compute average stats for epoch
            epoch_loss[-1]        /= num_trainloader
            epoch_initial_acc[-1] /= num_trainloader
            epoch_refined_acc[-1] /= num_trainloader
        pbar.update(1)


def init_training_model(epochs=10, lr=0.001):
    model = MVSNet()
    optimizer = torch.optim.Adam(model.parameters, lr=lr)
    start_epoch = 0

    return model, optimizer, start_epoch

def load_from_ckpt(ckpt, model, optimizer, epochs):

    model, optimizer, _ = init_training_model(epochs, 0)

    checkpoint = torch.load(ckpt)
    ckpt_epoch = epochs - (checkpoint["epoch"]+1)
    if ckpt_epoch <= 0:
        raise ValueError("Epochs provided: {}, epochs completed in ckpt: {}".format(
    epochs, checkpoint["epoch"]+1))

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optim_state_dict"])
    
    return model, optimizer, ckpt_epoch
        

if __name__ =="__main__":
    # really strange behavior, need to include the dataset class
    # here in order to load the saved dataset properly
    from data import DtuTrainDataset
    train(epochs=1, train_data_loader="test_dataloader")
