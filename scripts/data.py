from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils
import torch
import numpy as np  
from PIL import Image
from os.path import join
import re
from cv2 import flip


class Cameras():

    def __init__(self, path, idx_list):

        self.idx_list   = idx_list
        self.base_path  = path
        self.class_path = join(self.base_path, 'Cameras')
        self.K          = []
        self.R          = []
        self.T          = []
        self.d          = []

        self.file_names  = []
        for idx in self.idx_list:
            num = '{:0>8}'.format(str(idx))+'_cam.txt'
            self.file_names.append(join(self.class_path, num))

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


class Depths():

    def __init__(self, path, idx_list, scan_num=1, event='train'):
        self.idx_list   = idx_list
        self.base_path  = path
        self.class_path = join(self.base_path, 'Depths')
        self.scan_path  = join(self.class_path, 'scan'+str(scan_num)+'_'+event)
        self.img        = []

        self.file_names  = []
        for idx in self.idx_list:
            num = 'depth_map_'+'{:0>4}'.format(str(idx))+'.pfm'
            self.file_names.append(join(self.scan_path, num))

    def load(self):
        for file_ in self.file_names:
            with open(file_) as f:
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
                
                scale = float((f.readline()).decode('UTF-8').restrip())
                if scale > 0:
                    data_type = '<f'
                else:
                    data_type = '>f'
                
                data_string = f.read()
                data = np.fromstring(data_string, data_type)
                data = np.reshape(data, (height, width, dim))
                data = flip(data,0)
                self.img.append(data)


class Rectified():

    def __init__(self, path, idx_list, scan_num=1, event='train'):
        self.idx_list   = idx_list
        self.base_path  = path
        self.class_path = join(self.base_path, 'Depths')
        self.scan_path  = join(self.class_path, 'scan'+str(scan_num)+event)
        self.img       = []

        self.file_names  = dict()
        for i in range(7):
            self.file_names[str(i)] = []

        for idx in self.idx_list:
            for i in range(7):
                num = 'rect_'+'{:0>4}'.format(str(idx))+'_'+str(i)+'_r5000.png'
                self.file_names[str(i)].append(join(self.scan_path, num))

class DtuDataset(Dataset):

    def __init__(self, data):
        self.data = list(data)
        self.n_data = len(self.data)
        
        self.samples = []

        for idx in range(len(self.data)):
            sample = self.data[idx]

            
            self.samples.append(sample)

    def __getitem__(self, idx):
        return self.samples[idx]

    def __len__(self):
        return self.n_data

path = '../data/mvs_training/dtu'
a = Cameras(path=path, idx_list=np.arange(49))
b = Depths(path=path, idx_list=np.arange(49))
c = Rectified(path=path, idx_list=np.arange(49))

# a.load()
b.load()
# print(a.K[1])
# print(a.R[1])
# print(a.T[1])
# print(a.d[1])
