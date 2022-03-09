import time
import numpy as np
import torch
import torch.nn.functional as f
from loss import loss_fcn
from utils import *
from model import *
from tqdm import tqdm


def train(epochs, 
          train_data_loader,
          lr=0.001,
          save_path='/checkpoints',
          device=None,
          checkpoint=None,
          model=None,
          start_epoch=0):

    if isinstance(train_data_loader, str):
        train_data_loader = torch.load(train_data_loader)

    num_trainloader = len(train_data_loader)

    device = torch.device('cuda:0' if device == 'cuda' else 'cpu')

    if model is not None:
        print('Using Input Model...')
        optimizer = torch.optim.Adam(model.parameters())
    else:
        if checkpoint is not None:
            print('Loading Checkpoint...')
            model, optimizer, start_epoch = load_from_ckpt(checkpoint, epochs)
        else:
            print('Initializing Model...')
            model, optimizer, start_epoch = init_training_model(epochs, lr)

    print('Start Training From Epoch #{:d}'.format(start_epoch))

    print('Model sent to device: ', device)
    model = model.to(device)

    # average statistics for each epoch in range(epochs)
    epoch_loss        = np.array([])
    epoch_initial_acc = np.array([])
    epoch_refined_acc = np.array([])


    for epoch in range(start_epoch, epochs):
        print("TRAINING EPOCH #", epoch)

        append_zero(epoch_loss)
        append_zero(epoch_initial_acc)
        append_zero(epoch_refined_acc)

        model.train()
        epoch_start_time = time.time()
        current_time     = time.time()

        for batch_idx, batch in tqdm(enumerate(train_data_loader)):

            batch_size = batch['input_img'].size()[0]

            # statistics for each element in this batch
            batch_loss        = np.zeros(batch_size)
            batch_initial_acc = np.zeros(batch_size)
            batch_refined_acc = np.zeros(batch_size)

            camera = dict()

            for sidx in range(batch_size):

                append_zero(batch_loss)
                append_zero(batch_initial_acc)
                append_zero(batch_refined_acc)

                optimizer.zero_grad()

                nn_input  = batch['input_img'][sidx,:,:,:,:]
                gt_depth  = batch['depth_ref'][sidx,:,:,:,:]

                camera['K'] = batch['K'][sidx,:,:,:,:]
                camera['R'] = batch['R'][sidx,:,:,:,:]
                camera['T'] = batch['T'][sidx,:,:,:,:]
                camera['d'] = batch['d'][sidx,:,:,:,:]

                initial_depth_map, refined_depth_map = model(nn_input, camera, sidx)

                refined_acc, initial_acc = compute_accuracy(refined_depth_map, 
                                                            initial_depth_map, 
                                                            gt_depth)

                loss = loss_fcn(refined_acc, initial_acc)

                if epoch > 0:
                    loss.backward()
                    optimizer.step()

                # store statistics of current item in batch
                batch_loss[-1] = loss
                batch_initial_acc[-1] = initial_acc
                batch_refined_acc[-1] = refined_acc
                
                current_time = time.time()
            
            if True:
                print(
                    "Epoch: #{:d} Batch: {:d}/{:d}  Batch Elapsed Time: {:g}\t"
                    "Batch Loss (final/mean) {:.4f}/{:.4f}\t"
                    "Batch Init. Acc (final/mean) {:.4f}/{:.4f}\t"
                    "Batch Refi. Acc (final/mean) {:.4f}/{:.4f}\t"
                    .format(epoch, 
                            batch_idx+1, 
                            num_trainloader, 
                            current_time - epoch_start_time,
                            batch_loss[-1], np.mean(batch_loss),
                            batch_initial_acc[-1], np.mean(batch_initial_acc),
                            batch_refined_acc[-1], np.mean(batch_refined_acc)))

            # add average stats for the most recent batch
            epoch_loss[-1]        += np.mean(batch_loss)
            epoch_initial_acc[-1] += np.mean(batch_initial_acc)
            epoch_refined_acc[-1] += np.mean(batch_refined_acc)

        # compute average stats for epoch
        epoch_loss[-1]        /= num_trainloader
        epoch_initial_acc[-1] /= num_trainloader
        epoch_refined_acc[-1] /= num_trainloader


def init_training_model(epochs=10, lr=0.001):
    model = MVSNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
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
