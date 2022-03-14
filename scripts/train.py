import time
from os.path import join
import numpy as np
import torch
from torch.utils.data import DataLoader
from loss import loss_fcn
from utils import print_gpu_memory
from model import MVSNet
from config import DEVICE
import gc
from sys import getsizeof

def train(epochs:int, 
          train_data_loader:DataLoader,
          lr:float=0.001,
          save_path:str=join('.','checkpoints'),
          checkpoint:str=None,
          model:torch.nn.Module=None,
          start_epoch:int=0):

    if isinstance(train_data_loader, str):
        train_data_loader = torch.load(train_data_loader)

    num_trainloader = len(train_data_loader)

    if model is not None:
        print('Using Input Model...')
        optimizer = torch.optim.Adam(model.parameters)
        batch_start_idx = 0
    else:
        if checkpoint is not None:
            print('Loading Checkpoint...')
            out = load_from_ckpt(checkpoint, epochs)
            model, optimizer, start_epoch, batch_start_idx, \
                epoch_loss, epoch_initial_acc, epoch_refined_acc = out
        else:
            print('Initializing Model...')
            model, optimizer, start_epoch = init_training_model(epochs, lr)
            batch_start_idx   = 0
            epoch_loss        = np.zeros(num_trainloader)
            epoch_initial_acc = np.zeros(num_trainloader)
            epoch_refined_acc = np.zeros(num_trainloader)

    print('Start Training From Epoch #{:d}'.format(start_epoch))

    # print_gpu_memory()
    print('Model sent to device: ', DEVICE)
    model = model.to(DEVICE)
    # print_gpu_memory()

    id_str = 'train_'+str(int(time.time()))

    model.train()
    for epoch in range(start_epoch, epochs):
        # with torch.no_grad():
        print("----- TRAINING EPOCH #{:d} -----".format(epoch+1+start_epoch))

        # epoch_start_time = time.time()
        # current_time     = time.time()

        batch_loss        = np.zeros(num_trainloader)
        batch_initial_acc = np.zeros(num_trainloader)
        batch_refined_acc = np.zeros(num_trainloader)

        batch_idx = -1
        for batch in train_data_loader:
            batch_idx += 1
            if batch_idx < batch_start_idx:
                continue
            else: 
                batch_start_idx = -1

            batch_size, n_views, ch, h, w = batch['input_img'].size()
            _         , _      , _ ,dh,dw = batch['depth_ref'].size()
                
            optimizer.zero_grad(set_to_none=True)

            # print_gpu_memory('before unpack : ')
        
            nn_input= torch.reshape(batch['input_img'], (batch_size*n_views, ch, h, w)).to(DEVICE) # b*n, ch, h, w
            gt_depth= torch.reshape(batch['depth_ref'], (batch_size, 1, dh, dw)).to(DEVICE)       # b*n, ch, h, w
            K_batch = torch.reshape(batch['K'], (batch_size*n_views, 3, 3)) # b*n, 1, 3, 3
            R_batch = torch.reshape(batch['R'], (batch_size*n_views, 3, 3))
            T_batch = torch.reshape(batch['T'], (batch_size*n_views, 3, 1))
            d_min   = batch['d']
            d_int   = batch['d_int']

            # print_gpu_memory('before model : ')

            initial_depth_map, refined_depth_map = model(nn_input, K_batch, 
                        R_batch, T_batch, d_min, d_int, batch_size, n_views)

            # print_gpu_memory('after model : ')

            loss, initial_acc, refined_acc = loss_fcn(gt_depth, 
                                        initial_depth_map, refined_depth_map)

            # print_gpu_memory('after loss : ')

            loss.backward()
            optimizer.step()

            # print_gpu_memory('after optim : ')

            # store statistics of current item in batch
            batch_loss[batch_idx] = float(loss.detach().mean())
            batch_initial_acc[batch_idx] = float(initial_acc.detach().mean())
            batch_refined_acc[batch_idx] = float(refined_acc.detach().mean())
            
            # print_gpu_memory('after store : ')

            if ((batch_idx+1)%10 == 0) or ((batch_idx+1) >= num_trainloader):
                torch.save({
                    'epoch':epoch,
                    'batch_idx':batch_idx,
                    'model_state_dict':model.state_dict(),
                    'optimizer_state_dict':optimizer.state_dict(),
                    'loss':epoch_loss,
                    'acc_1':epoch_initial_acc,
                    'acc_2':epoch_refined_acc,
                    }, join(save_path, id_str+'_'+str(batch_idx)))
            
            # print_gpu_memory('after print : ')

            if (batch_idx+1) % 1 == 0:
                print(
                    "Epoch: #{:d} Batch: {:d}/{:d}\tLoss(curr/avg) {:.4f}/{:.4f}\t"
                    "Acc 1(curr/avg) {:.4f}/{:.4f}  Acc 2(curr/avg) {:.4f}/{:.4f}\t"
                    .format(epoch, 
                            batch_idx+1, 
                            num_trainloader, 
                            batch_loss[batch_idx],
                            np.mean(batch_loss[:(batch_idx+1)]),
                            batch_initial_acc[batch_idx], 
                            np.mean(batch_initial_acc[:(batch_idx+1)]),
                            batch_refined_acc[batch_idx], 
                            np.mean(batch_refined_acc[:(batch_idx+1)]),
                            )
                    )

            # print_gpu_memory('end batch :\t ')
            
            # del batch
            # del nn_input; del gt_depth
            # del K_batch; del R_batch; del T_batch
            # del d_int;del d_min
            # del initial_acc; del refined_acc
            # del initial_depth_map; del refined_depth_map
            # del batch_size; del n_views; del ch; del h; del w; del dh; del dw

            # gc.collect()
            # torch.cuda.empty_cache()

            # print_gpu_memory('after garbage : ')

        # compute average stats for epoch
        epoch_loss[epoch]        = np.mean(batch_loss)
        epoch_initial_acc[epoch] = np.mean(batch_initial_acc)
        epoch_refined_acc[epoch] = np.mean(batch_refined_acc)

    return model, epoch_loss, epoch_initial_acc, epoch_refined_acc

def init_training_model(epochs=10, lr=0.001):
    model = MVSNet()
    optimizer = torch.optim.Adam(model.parameters, lr=lr)
    start_epoch = 0

    return model, optimizer, start_epoch

def load_from_ckpt(ckpt, epochs):

    model, optimizer, _ = init_training_model(epochs, 0)

    checkpoint = torch.load(ckpt)
    ckpt_epoch = epochs - (checkpoint["epoch"]+1)
    if ckpt_epoch <= 0:
        raise ValueError("Epochs provided: {:d}, epochs completed in ckpt: {:d}".format(
    epochs, checkpoint["epoch"]+1))

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    loss  = checkpoint['loss']
    acc_1 = checkpoint['acc_1']
    acc_2 = checkpoint['acc_2']
    b_idx = checkpoint['batch_idx']
    
    return model, optimizer, checkpoint["epoch"], b_idx, loss, acc_1, acc_2
        

if __name__ =="__main__":
    # really strange behavior, need to include the dataset class
    # here in order to load the saved dataset properly
    from data import DtuTrainDataset
    model, loss, acc_1, acc_2, = train(epochs=10, checkpoint=None, train_data_loader="test_dataloader")
