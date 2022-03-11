from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import Sampler
from torchvision import transforms
import torch

from cv2 import flip
from PIL import Image
from os.path import join
from itertools import product
from tqdm import tqdm
from utils import unsqueeze_n

import numpy as np  
import re
import random

class Cameras():

    def __init__(self, path, cam_list):

        self.cam_list   = cam_list
        self.base_path  = path
        self.class_path = join(self.base_path, 'Cameras')
        self.K          = []
        self.R          = []
        self.T          = []
        self.d          = []
        self.pairs      = []

        self.file_names  = []
        for idx in self.cam_list:
            num = '{:0>8}'.format(str(idx))+'_cam.txt'
            self.file_names.append(join(self.class_path, num))

        self.load()
        self.pair()

    def load(self):
        for file_ in self.file_names:
            with open(file_) as f:
                f.readline()                
                r1 = np.float64(f.readline().split())
                r2 = np.float64(f.readline().split())
                r3 = np.float64(f.readline().split())
                r4 = np.float64(f.readline().split())

                f.readline(); f.readline()
                r7 = np.float64(f.readline().split())
                r8 = np.float64(f.readline().split())
                r9 = np.float64(f.readline().split())

                f.readline()
                r11= np.float64(f.readline().split())

                K = np.vstack((r7,r8,r9))
                R = np.vstack((r1[0:3], r2[0:3], r3[0:3]))

                self.K.append(np.vstack((r7,r8,r9)))
                self.R.append(np.vstack((r1[0:3], r2[0:3], r3[0:3])))
                self.T.append(np.vstack((r1[-1], r2[-1], r3[-1])))                 
                self.d.append(np.array([r11[0]]).reshape(-1,1))

    def pair(self):

        with open(join(self.class_path, 'pair.txt')) as f:
            f.readline() # discard the header
            line = f.readline()
            while line:                
                if int(line[0]) in self.cam_list:
                    pair_line = f.readline().split()
                    self.pairs.append(np.int64(pair_line[1::2]))
                line = f.readline().split()



class Depths():

    def __init__(self, path, cam_list, scan_idx=1, event='train'):
        self.cam_list   = cam_list
        self.base_path  = path
        self.class_path = join(self.base_path, 'Depths')
        self.scan_path  = [join(self.class_path, 'scan'+str(scan)+'_'+event) for scan in scan_idx]
        self.img        = []

        self.file_names  = []
        for scan_path in self.scan_path:
            temp_file_names = []
            for idx in self.cam_list:
                num = 'depth_map_'+'{:0>4}'.format(str(idx))+'.pfm'
                temp_file_names.append(join(scan_path, num))
            self.file_names.append(temp_file_names)

        self.load()

    def load(self):
        for scan in self.file_names:
            img_temp = []
            for file_ in scan:
                with open(file_, 'rb') as f:
                    # This is largely taken from the source MVSNet repository
                    # included in this repo
                    header = f.readline().decode('UTF-8').rstrip()

                    if header == 'PF':
                        ch_dim = 3
                    elif header == 'Pf':
                        ch_dim = 1
                    else:
                        raise Exception("Invalid Header for PFM file.")
                    
                    dim_match = re.match(r'^(\d+)\s(\d+)\s$', f.readline().decode('UTF-8'))
                    if dim_match:
                        width, height = map(int, dim_match.groups())
                    else:
                        raise Exception("PFM header gives no dimensions.")
                    
                    scale = float((f.readline()).decode('UTF-8').rstrip())
                    if scale > 0:
                        data_type = '<f'
                    else:
                        data_type = '>f'
                    
                    data_string = f.read()
                    data = np.frombuffer(data_string, data_type)
                    data = np.reshape(data, (height, width, ch_dim))
                    data = flip(data,0)
                    img_temp.append(data)
            self.img.append(img_temp)


class Rectified():

    def __init__(self, path, cam_list, scan_idx=1, light_idx=np.arange(7), event='train'):
        self.cam_list   = cam_list
        self.base_path  = path
        self.class_path = join(self.base_path, 'Rectified')
        self.scan_path  = [join(self.class_path, 'scan'+str(scan)+'_'+event) for scan in scan_idx]

        self.file_names      = []
        # for i in range(7):
        #     self.file_names[str(i)] = []

        for scan_path in self.scan_path:
            temp_scan_file_names = []
            for i in light_idx:   
                temp_light_file_names = []
                for idx in self.cam_list:
                    # camera labels in image filename are offset by one
                    # e.g. cam 0 corresponds to img rect_001_<light>_r5000.png
                    num = 'rect_'+'{:0>3}'.format(str(idx+1))+'_'+str(i)+'_r5000.png'
                    temp_light_file_names.append(join(scan_path, num))
                temp_scan_file_names.append(temp_light_file_names)
            self.file_names.append(temp_scan_file_names)

class DtuReader():

    def __init__(self, folder_path, cam_idx, scan_idx, event):

        self.cam_idx   = cam_idx
        self.scan_idx  = scan_idx
        self.event     = event

        self.Cameras   = Cameras(folder_path, cam_idx)
        self.Depths    = Depths(folder_path, cam_idx, scan_idx=scan_idx, event=event)
        self.Images    = Rectified(folder_path, cam_idx, scan_idx=scan_idx, event=event)

        self.n_images  = len(cam_idx)*len(scan_idx)
        
    def __len__(self):
        return self.n_images

class DtuTrainDataset(Dataset):

    def __init__(self, DTU:DtuReader, scan_idx):

        self.samples = []
        self.tensor_transform_img = transforms.Compose([
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float)
        ])
        self.tensor_transform_depth = torch.from_numpy

        # for ref in ref_cam_idx:
        #     for light in light_idx:
        #         for pair in pair_idx:

        ref_cam_idx = np.arange(49)
        light_idx_choices = np.arange(7)

        sample_iter = product(scan_idx, ref_cam_idx)

        total = len(scan_idx)*len(ref_cam_idx)
        with tqdm(total=total) as pbar:
            for (scan, ref) in sample_iter:

                light_ref = np.random.randint(0,7)

                ref_pairs = DTU.Cameras.pairs[ref]
                pair_idx  = random.sample(set(ref_pairs), 2) # 2 random unique
                light_idx = random.choices(light_idx_choices, k=2) # 2 random
                pair_idx1 = pair_idx[0] # cam indices start at 0, don't subtract 1
                pair_idx2 = pair_idx[1]

                # print(ref, light_ref, light_idx[0], light_idx[1], pair_idx)

                # get reference image and two
                image_ref = DTU.Images.file_names[scan][light_ref][ref]
                image_one = DTU.Images.file_names[scan][light_idx[0]][pair_idx1]
                image_two = DTU.Images.file_names[scan][light_idx[1]][pair_idx2]

                sample = dict()
                sample['scan_idx']  = scan
                sample['ref_idx']   = ref
                sample['pair_idx']  = pair_idx
                sample['light_idx'] = light_idx

                # sample['view_ref']  = self.tensor_transform_img(Image.open(image_ref).convert('RGB'))
                # sample['view_one']  = self.tensor_transform_img(Image.open(image_one).convert('RGB'))
                # sample['view_two']  = self.tensor_transform_img(Image.open(image_two).convert('RGB'))
                view_ref  = torch.unsqueeze(self.tensor_transform_img(Image.open(image_ref).convert('RGB')), 0)
                view_one  = torch.unsqueeze(self.tensor_transform_img(Image.open(image_one).convert('RGB')), 0)
                view_two  = torch.unsqueeze(self.tensor_transform_img(Image.open(image_two).convert('RGB')), 0)
                    
                sample['input_img'] = torch.cat((view_ref, view_one, view_two), dim=0)
                sample['depth_ref'] = unsqueeze_n(self.tensor_transform_depth(DTU.Depths.img[scan][ref]), 2)
                
                Kref      = unsqueeze_n(self.tensor_transform_depth(DTU.Cameras.K[ref]), 1)
                Rref      = unsqueeze_n(self.tensor_transform_depth(DTU.Cameras.R[ref]), 1)
                Tref      = unsqueeze_n(self.tensor_transform_depth(DTU.Cameras.T[ref]), 1)
                dref      = unsqueeze_n(self.tensor_transform_depth(DTU.Cameras.d[ref]), 1)

                K1        = unsqueeze_n(self.tensor_transform_depth(DTU.Cameras.K[pair_idx1]), 1)
                R1        = unsqueeze_n(self.tensor_transform_depth(DTU.Cameras.R[pair_idx1]), 1)
                T1        = unsqueeze_n(self.tensor_transform_depth(DTU.Cameras.T[pair_idx1]), 1)
                d1        = unsqueeze_n(self.tensor_transform_depth(DTU.Cameras.d[pair_idx1]), 1)

                K2        = unsqueeze_n(self.tensor_transform_depth(DTU.Cameras.K[pair_idx2]), 1)
                R2        = unsqueeze_n(self.tensor_transform_depth(DTU.Cameras.R[pair_idx2]), 1)
                T2        = unsqueeze_n(self.tensor_transform_depth(DTU.Cameras.T[pair_idx2]), 1)
                d2        = unsqueeze_n(self.tensor_transform_depth(DTU.Cameras.d[pair_idx2]), 1)

                sample['K'] = torch.cat((Kref, K1, K2),dim=0)
                sample['R'] = torch.cat((Rref, R1, R2),dim=0)
                sample['T'] = torch.cat((Tref, T1, T2),dim=0)
                sample['d'] = torch.cat((dref, d1, d2),dim=0)

                self.samples.append(sample)

                pbar.update(1)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def get_dtu_loader(folder_path, cam_idx, scan_idx, event,
                    batch_size=14, i_start=0):

    if event=='train':
        shuffle=True
    else:
        shuffle=False
    print("Constructing Training Dataloader...")
    DTU             = DtuReader(folder_path, cam_idx, scan_idx, light_idx, event)
    dtu_dataset     = DtuTrainDataset(DTU, scan_idx-1)

    # sampler         = CustomSampler(dtu_dataset, i=i_start, batch_size=batch_size)
    dtu_dataloader  = DataLoader(dataset=dtu_dataset, 
                                 batch_size=batch_size, 
                                #  sampler=sampler, 
                                 shuffle=shuffle)

    print("\nDataloader Finished.")
    print("    Batch Size: {:d}".format(batch_size))
    print("    Samples:    {:d}".format(len(dtu_dataset)))

    return dtu_dataloader

class CustomSampler(Sampler):
    """Resumable sampler code adapted from 
    https://stackoverflow.com/questions/60993677/how-can-i-save-pytorchs-dataloader-instance
    to allow for pausing/resuming training with the same dataloader state.
    By knowing i_batch during training, we can load the dataloader from file and
    resume the sequence exactly where we left off."""
    def __init__(self, data, i=0, batch_size=14):
        random.shuffle(data)
        self.seq = list(range(len(data)))[i * batch_size:]

    def __iter__(self):
        return iter(self.seq)

    def __len__(self):
        return len(self.seq)


if __name__ == '__main__':

    case = 'training' # 'evaluation', 'validation'
    batch_size = 49

    random.seed(401)
    path = '../data/mvs_training/dtu'

    cam_idx=np.arange(49)
    
    if case == 'training':
        scan_idx=np.arange(2, 6, 7, 8, 14, 16, 18, 19, 20, 22, 30, 31, 36, 39, 41, 42, 44,
45, 46, 47, 50, 51, 52, 53, 55, 57, 58, 60, 61, 63, 64, 65, 68, 69, 70, 71, 72,
74, 76, 83, 84, 85, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100,
101, 102, 103, 104, 105, 107, 108, 109, 111, 112, 113, 115, 116, 119, 120,
121, 122, 123, 124, 125, 126, 127, 128)
        file_name = 'training_dataloader'
    elif case == 'evaluation':
        scan_idx=np.arange(1, 4, 9, 10, 11,
12, 13, 15, 23, 24, 29, 32, 33, 34, 48, 49, 62, 75, 77, 110, 114, 118)
        file_name = 'evaluation_dataloader'
    elif case == 'validation':
        scan_idx=np.arange(3, 5, 17, 21, 28, 35,
37, 38, 40, 43, 56, 59, 66, 67, 82, 86, 106, 117)
        file_name = 'validation_dataloader'

    light_idx=np.arange(7)

    DTU = DtuReader(path, cam_idx=np.arange(49), scan_idx=np.arange(1,2), light_idx=np.arange(7), event='train')

    dtu_train_dataset = DtuTrainDataset(DTU, np.arange(0, 1))

    dtu_train_dataloader = get_dtu_loader(path, cam_idx, scan_idx, light_idx, event='train')

    torch.save(dtu_train_dataloader, file_path)

    # dtu_train_dataloader = torch.load(file_path)

    # print(dtu_train_dataset.__getitem__(0))
    # print(dtu_train_dataset.__len__())
    # print(DTU.Cameras.pairs[0])
    print(len(list(dtu_train_dataloader)))
#    for idx, batch in enumerate(dtu_train_dataloader):
#        print(idx, batch['input_img'].size())
#        print(idx, batch['depth_ref'].size())
#        print(idx, batch['K'].size())
#        print(idx, batch['R'].size())
#        print(idx, batch['T'].size())
#        print(idx, batch['d'].size())
        # print(idx, batch['camera']['Rref'].size())
        # for sidx in range(batch['input_img'].size()[0]):
            # print(batch['input_img'][idx,:,:,:,:].size())
        # print(idx, batch['input_img'].view(-1, 3, 512, 640).size())

