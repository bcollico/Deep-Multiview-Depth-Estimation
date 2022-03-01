from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils
import torch

from cv2 import flip
from PIL import Image
from os.path import join

import numpy as np  
import re


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
                self.d.append(r11[0])

    def pair(self):

        with open(join(self.class_path, 'pair.txt')) as f:
            f.readline() # discard the header
            line = f.readline()
            while line:                
                if int(line[0]) in self.cam_list:
                    pair_line = f.readline().split()
                    self.pairs.append(np.array([np.float64(pair_line[1::2])]))
                line = f.readline().split()



class Depths():

    def __init__(self, path, cam_list, scan=1, event='train'):
        self.cam_list   = cam_list
        self.base_path  = path
        self.class_path = join(self.base_path, 'Depths')
        self.scan_path  = join(self.class_path, 'scan'+str(scan)+'_'+event)
        self.img        = []

        self.file_names  = []
        for idx in self.cam_list:
            num = 'depth_map_'+'{:0>4}'.format(str(idx))+'.pfm'
            self.file_names.append(join(self.scan_path, num))

        self.load()

    def load(self):
        for file_ in self.file_names:
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
                self.img.append(data)


class Rectified():

    def __init__(self, path, cam_list, scan=1, light_idx=0, event='train'):
        self.cam_list   = cam_list
        self.base_path  = path
        self.class_path = join(self.base_path, 'Rectified')
        self.scan_path  = join(self.class_path, 'scan'+str(scan)+'_'+event)

        self.file_names  = dict()
        for i in range(7):
            self.file_names[str(i)] = []

        for idx in self.cam_list:
            for i in light_idx:
                num = 'rect_'+'{:0>3}'.format(str(idx+1))+'_'+str(i)+'_r5000.png'
                self.file_names[str(i)].append(join(self.scan_path, num))

class DtuReader():

    def __init__(self, folder_path, cam_idx, scan_idx, light_idx, event):

        self.cam_idx   = cam_idx
        self.scan_idx  = scan_idx
        self.light_idx = light_idx
        self.event     = event

        self.Cameras   = Cameras(folder_path, cam_idx)
        self.Depths    = Depths(folder_path, cam_idx, scan=scan_idx, event=event)
        self.Images    = Rectified(folder_path, cam_idx, scan=scan_idx, light_idx=light_idx, event=event)

        self.keys      = self.Images.file_names.keys()
        self.n_images  = np.sum(np.array([len(self.Images.file_names[i]) for i in self.keys]))
        
        self.samples   = []

    def __len__(self):
        return self.n_images

class DtuTrainDataset(Dataset):

    def __init__(self, DTU:DtuReader, ref_cam_idx, light_idx, pair_idx):

        self.samples = []
        self.tensor_transform_img = transforms.Compose([
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float)
        ])
        self.tensor_transform_depth = torch.from_numpy

        for ref in ref_cam_idx:
            for light in light_idx:
                for pair in pair_idx:
                    print(ref, light, pair)

                    # get reference image and two
                    image_ref = (DTU.Images.file_names[str(light)])[ref]
                    image_one = DTU.Images.file_names[str(light)][pair]
                    image_two = DTU.Images.file_names[str(light)][pair+1]

                    sample = dict()
                    camera = dict()
                    sample['view_ref']  = self.tensor_transform_img(Image.open(image_ref).convert('RGB'))
                    sample['view_one']  = self.tensor_transform_img(Image.open(image_one).convert('RGB'))
                    sample['view_two']  = self.tensor_transform_img(Image.open(image_two).convert('RGB'))
                    sample['depth_ref'] = self.tensor_transform_depth(DTU.Depths.img[ref])
                    
                    camera['Kref']      = DTU.Cameras.K[ref]
                    camera['Rref']      = DTU.Cameras.R[ref]
                    camera['Tref']      = DTU.Cameras.T[ref]
                    camera['dref']      = DTU.Cameras.d[ref]

                    camera['K1']        = DTU.Cameras.K[pair]
                    camera['R1']        = DTU.Cameras.R[pair]
                    camera['T1']        = DTU.Cameras.T[pair]
                    camera['d1']        = DTU.Cameras.d[pair]

                    camera['K2']        = DTU.Cameras.K[pair+1]
                    camera['R2']        = DTU.Cameras.R[pair+1]
                    camera['T2']        = DTU.Cameras.T[pair+1]
                    camera['d2']        = DTU.Cameras.d[pair+1]

                    self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

path = '../data/mvs_training/dtu'
DTU = DtuReader(path, np.arange(49), 1, np.arange(7), 'train')
dtu_train_dataset = DtuTrainDataset(DTU, np.arange(2), np.arange(2), np.arange(3))
# print(dtu_train_dataset.__getitem__(0))
print(dtu_train_dataset.__len__())

