import time
from os.path import join
import numpy as np
import torch
from torch.utils.data import DataLoader
from validate import validate
from loss import loss_fcn
from utils import print_gpu_memory
from model import MVSNet
from config import DEVICE
import copy

def train(epochs:int, 
          train_data_loader:DataLoader,
          lr:float=0.005,
          save_path:str=join('.','checkpoints'),
          checkpoint:str=None,
          model:torch.nn.Module=None,
          optimizer=None,
          start_epoch:int=0):

    if isinstance(train_data_loader, str):
        train_data_loader = torch.load(train_data_loader)

    valid_data_loader = torch.load('validation_dataloader')

    num_trainloader = len(train_data_loader)

    if model is not None:
        print('Using Input Model...')
        if optimizer is None:
             torch.optim.Adam(model.parameters, lr=lr)
        batch_start_idx = 0
        check_batch_idx = False
    else:
        if checkpoint is not None:
            print('Loading Checkpoint...')
            out = load_from_ckpt(checkpoint, epochs, lr)
            model, optimizer, start_epoch, batch_start_idx, \
                epoch_loss, epoch_initial_acc, epoch_refined_acc = out
            check_batch_idx = True
        else:
            print('Initializing Model...')
            model, optimizer, scheduler, start_epoch = init_training_model(epochs, lr)
            batch_start_idx   = 0
            check_batch_idx   = False
            epoch_loss        = [np.zeros(num_trainloader) for _ in range(epochs-start_epoch)]
            epoch_initial_acc = [np.zeros(num_trainloader) for _ in range(epochs-start_epoch)]
            epoch_refined_acc = [np.zeros(num_trainloader) for _ in range(epochs-start_epoch)]

    print('Start Training From Epoch #{:d}'.format(start_epoch+1))

    # print_gpu_memory()
    print('Model sent to device: ', DEVICE)
    model = model.to(DEVICE)
    # print_gpu_memory()

    id_str = 'train_'+str(int(time.time()))
    start_time = time.time()

    model.train()
    torch.autograd.set_detect_anomaly(True)
    for epoch in range(start_epoch, epochs):
        # with torch.no_grad():
        print("----- TRAINING EPOCH #{:d} -----".format(1+epoch))

        # epoch_start_time = time.time()
        # current_time     = time.time()

        batch_loss        = np.zeros(num_trainloader)
        batch_initial_acc = np.zeros(num_trainloader)
        batch_refined_acc = np.zeros(num_trainloader)

        for batch_idx, batch in enumerate(train_data_loader):
            if check_batch_idx and batch_idx < batch_start_idx:
                continue
            else: 
                batch_start_idx = -1
                check_batch_idx = False

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

            # d_min = torch.tensor(100)*d_min # error in dataset, use d_min = 0
            d_int = d_int.div(d_int) # set this to 1 so that we fully control interval from config

            initial_depth_map, refined_depth_map = model(nn_input, K_batch, 
                        R_batch, T_batch, d_min, d_int, batch_size, n_views)

            loss, initial_acc, refined_acc = loss_fcn(gt_depth, 
                                        initial_depth_map, refined_depth_map)

            loss.backward()
            optimizer.step()

            # store statistics of current item in batch
            batch_loss[batch_idx] = float(loss.detach().mean())
            batch_initial_acc[batch_idx] = float(initial_acc.detach().mean())
            batch_refined_acc[batch_idx] = float(refined_acc.detach().mean())
            
            if ((batch_idx+1)%100 == 0) or ((batch_idx+1) >= num_trainloader):
                torch.save({
                    'epoch':epoch,
                    'batch_idx':batch_idx,
                    'model_state_dict':model.state_dict(),
                    'optimizer_state_dict':optimizer.state_dict(),
                    'scheduler_state_dict':scheduler.state_dict(),
                    'loss':epoch_loss,
                    'acc_1':epoch_initial_acc,
                    'acc_2':epoch_refined_acc,
                    }, join(save_path, id_str+'_'+str(epoch)+'_'+str(batch_idx)))
            
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

        # compute average stats for epoch
        epoch_loss[epoch]        = batch_loss
        epoch_initial_acc[epoch] = batch_initial_acc
        epoch_refined_acc[epoch] = batch_refined_acc

        if scheduler is not None:
            valid_loss, valid_initial_acc, valid_refined_acc = \
                        validate(model=copy.deepcopy(model), optimizer=optimizer, valid_data_loader=valid_data_loader)
            scheduler.step(valid_loss)
            print("Validation Results: Loss {:.4f}\t"
                "Acc 1 {:.4f}  Acc 2 {:.4f}\t".format(
                    valid_loss, valid_initial_acc, valid_refined_acc
                ))


    end_time = time.time()
    print("Total training time: ", end_time-start_time)
    return model, epoch_loss, epoch_initial_acc, epoch_refined_acc

def init_training_model(epochs=10, lr=0.001):
    model = MVSNet()
    optimizer = torch.optim.Adam(model.parameters, lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                factor=0.8, patience=2, cooldown=4, min_lr=0.0001, verbose=True)
    start_epoch = 0

    return model, optimizer, scheduler, start_epoch

def load_from_ckpt(ckpt, epochs, lr):

    model, optimizer, scheduler, _ = init_training_model(epochs, 0)

    model = model.to(DEVICE)

    checkpoint = torch.load(ckpt)
    ckpt_epoch = epochs - (checkpoint["epoch"]+1)
    if ckpt_epoch <= 0:
        raise ValueError("Epochs provided: {:d}, epochs completed in ckpt: {:d}".format(
    epochs, checkpoint["epoch"]+1))

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    loss  = checkpoint['loss']
    acc_1 = checkpoint['acc_1']
    acc_2 = checkpoint['acc_2']
    b_idx = checkpoint['batch_idx']
    
    return model, optimizer, checkpoint["epoch"]+1, b_idx, loss, acc_1, acc_2
        

if __name__ =="__main__":
    # really strange behavior, need to include the dataset class
    # here in order to load the saved dataset properly
    from data import DtuTrainDataset
    model, loss, acc_1, acc_2, = train(epochs=20, 
                                        checkpoint=None,#join('.','checkpoints','train_1647421268_2_19'), 
                                        train_data_loader="test_dataloader")
