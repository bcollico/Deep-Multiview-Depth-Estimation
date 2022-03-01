import torch
from loss import loss_fcn
from utils import compute_accuracy


def train(epochs, 
          train_data_loader,
          lr=0.001,
          save_path='/output/checkpoint',
          device=None,
          checkpoint=None,
          model=None,
          start_epoch=0):

    num_trainloader = len(train_data_loader)

    device = torch.device('cuda:0' if device == 'cuda' else 'cpu')

    if model is not None:
        print('Loading Model...')
        optimizer = torch.optim.Adam(model.parameters())
    else:
        if checkpoint is not None:
            print('Loading Checkpoint...')
        else:
            print('Initializing Model...')
        
        
        # init model here
        model, optimizer = init_training(epochs=epochs, lr=lr,checkpoint=checkpoint,device=device)

        print('Start Training From Epoch #{:d}'.format(epoch))

        model = model.to(device)
    

    epoch_loss = []
    batch_loss = []
    sum_loss   = 0

    epoch_accuracy = []

    for epoch in range(start_epoch, epochs):

        model.train()

        for idx, sample in enumerate(train_data_loader):

            optimizer.zero_grad()

            camera = sample['camera']

            image_ref = sample['image_ref']
            image_one = sample['image_one']
            image_two = sample['image_two']

            nn_input  = torch.cat((image_ref, image_one, image_two), dim=0)
            gt_depth  = sample['depth_ref']

            initial_depth_map, refined_depth_map = model(nn_input, camera)

            loss = loss_fcn(initial_depth_map, refined_depth_map, gt_depth)

            accuracy = compute_accuracy

            if e > 0:
                loss.backward()
                optimizer.step()
            

