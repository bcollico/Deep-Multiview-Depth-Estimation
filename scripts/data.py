from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import Sampler
from torchvision import transforms
import torch

from cv2 import flip
from PIL import Image
from os.path import join
from itertools import product
from tqdm import tqdm
from utils import unsqueeze_n as unsqz

import numpy as np  
import re
import random

class Cameras():
    """Class for handling camera parameters."""
    def __init__(self, path, cam_list):

        self.cam_list   = cam_list  # list of camera indices
        self.base_path  = path      # path to data folder
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
        """Load camera intrinsics and extrinsics from file."""
        for file_ in self.file_names:
            with open(file_) as f:
                f.readline() # text line

                # extrinsics     
                r1 = np.float64(f.readline().split())
                r2 = np.float64(f.readline().split())
                r3 = np.float64(f.readline().split())
                r4 = np.float64(f.readline().split()) # r4 = [0, 0, 0, 1]

                f.readline(); f.readline() # empty lines

                # intrinsics
                r7 = np.float64(f.readline().split())
                r8 = np.float64(f.readline().split())
                r9 = np.float64(f.readline().split())

                f.readline()# empty line

                #depth
                r11= np.float64(f.readline().split())

                self.K.append(np.vstack((r7,r8,r9)))
                self.R.append(np.vstack((r1[0:3], r2[0:3], r3[0:3])))
                self.T.append(np.vstack((r1[-1], r2[-1], r3[-1])))                 
                self.d.append(np.array([r11[0]]).reshape(-1,1))

    def pair(self):
        """Read the pre-computed best pairs for each reference camera view."""
        with open(join(self.class_path, 'pair.txt')) as f:
            f.readline() # discard the header
            line = f.readline()
            while line:                
                if int(line[0]) in self.cam_list:
                    pair_line = f.readline().split()
                    self.pairs.append(np.int64(pair_line[1::2]))
                line = f.readline().split()


class Depths():
    """Class for handling ground truth depth values."""
    def __init__(self, path, cam_list, scan_idx=1, event='train'):
        self.cam_list   = cam_list # list of camera indices
        self.base_path  = path     # path to data folder
        self.class_path = join(self.base_path, 'Depths') # path to depths folder
        self.scan_path  = [join(self.class_path, 'scan'+str(scan)+'_'+event) \
                            for scan in scan_idx] # path to each scan in depths
        self.img        = [] # UNUSED | list for storing depth values

        self.file_names  = [] # list of depth filenames for dataloader
        for scan_path in self.scan_path:
            temp_file_names = []
            for idx in self.cam_list:
                num = 'depth_map_'+'{:0>4}'.format(str(idx))+'.pfm'
                temp_file_names.append(join(scan_path, num))
            self.file_names.append(temp_file_names)

        # self.load() # do not load depth maps until __getitem__

    def load(self):
        """UNUSED | Load the ground truth depth map values from file."""
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
                    
                    dim_match = re.match(r'^(\d+)\s(\d+)\s$', 
                                    f.readline().decode('UTF-8'))
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
    """Class for handling rectified camera images from modified DTU dataset."""
    def __init__(self, path, cam_list, scan_idx=1, event='train'):
        self.cam_list   = cam_list
        self.base_path  = path
        self.class_path = join(self.base_path, 'Rectified')
        self.scan_path  = [join(self.class_path, 
                            'scan'+str(scan)+'_train') for scan in scan_idx]

        self.file_names      = []

        light_idx=np.arange(7) # 7 lighting conditions

        for scan_path in self.scan_path:
            temp_scan_file_names = []
            for i in light_idx:   
                temp_light_file_names = []
                for idx in self.cam_list:
                    # camera labels in image filename are offset by one
                    # e.g. cam 0 corresponds to img rect_001_<light>_r5000.png
                    num = 'rect_'+'{:0>3}'.format(str(idx+1)) \
                            +'_'+str(i)+'_r5000.png'
                    temp_light_file_names.append(join(scan_path, num))
                temp_scan_file_names.append(temp_light_file_names)
            self.file_names.append(temp_scan_file_names)

class DtuReader():
    """Aggregate class for DTU camera parameters, depth maps, and source images.
    Used in creating DtuTrainDataset PyTorch Dataset."""
    def __init__(self, folder_path, cam_idx, scan_idx, event):

        self.cam_idx   = cam_idx   # cameras to include in dataset
        self.scan_idx  = scan_idx  # list of scan indices to include in dataset
        self.event     = event     # 'training', 'validation', 'evaluation'

        self.Cameras   = Cameras(folder_path, cam_idx)
        self.Depths    = Depths(folder_path, cam_idx, 
                                    scan_idx=scan_idx, event=event)
        self.Images    = Rectified(folder_path, cam_idx, 
                                    scan_idx=scan_idx, event=event)

        self.n_images  = len(cam_idx)*len(scan_idx)
        
    def __len__(self):
        return self.n_images

class DtuTrainDataset(Dataset):
    """PyTorch datset for modified DTU data."""
    def __init__(self, DTU:DtuReader):

        self.samples = [] # List of samples

        # Image transformations, TODO: Get mean and std for normalization
        self.img_xform = transforms.Compose([
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float)
        ])

        # Transformation to torch.float32
        self.npy_xform = torch.FloatTensor

        self.scan_idx = DTU.scan_idx # list of scans to include in dataset
        scan_idx      = self.scan_idx

        ref_cam_idx       = DTU.cam_idx
        light_idx_choices = np.arange(7) # choose from all lighting angles

        sample_iter = product(np.arange(len(scan_idx)),    # scan     indexing
                              light_idx_choices,           # lighting indexing
                              np.arange(len(ref_cam_idx))) # cam      indexing

        # total samples (for tqdm)
        total = len(scan_idx)*len(ref_cam_idx)*len(light_idx_choices) 
        with tqdm(total=total) as pbar:
            for (scan, light_ref, ref) in sample_iter:

                ref_pairs = DTU.Cameras.pairs[ref]
                pair_idx  = random.sample(set(ref_pairs), 2) # 2 random unique
                light_idx = random.choices(light_idx_choices, k=2) # 2 random
                pair_idx1 = pair_idx[0] # cam idx start at 0, don't subtract 1
                pair_idx2 = pair_idx[1]

                # get reference image and two auxilliary views
                image_ref = DTU.Images.file_names[scan][light_ref][ref]
                image_one = DTU.Images.file_names[scan][light_idx[0]][pair_idx1]
                image_two = DTU.Images.file_names[scan][light_idx[1]][pair_idx2]

                sample = dict()

                # store image and depth map file names
                sample['img_filenames']  = [image_ref, image_one, image_two]
                sample['depth_filename'] = DTU.Depths.file_names[scan][ref]

                # store pertinent sample information
                sample['scan_idx']  = self.scan_idx[scan]
                sample['ref_idx']   = ref_cam_idx[ref]
                sample['pair_idx']  = pair_idx
                sample['light_idx'] = np.concatenate((np.array([light_ref]), 
                                                        light_idx))
                
                Kref      = unsqz(self.npy_xform(DTU.Cameras.K[ref]), 1)
                Rref      = unsqz(self.npy_xform(DTU.Cameras.R[ref]), 1)
                Tref      = unsqz(self.npy_xform(DTU.Cameras.T[ref]), 1)
                dref      = unsqz(self.npy_xform(DTU.Cameras.d[ref]), 1)

                K1        = unsqz(self.npy_xform(DTU.Cameras.K[pair_idx1]), 1)
                R1        = unsqz(self.npy_xform(DTU.Cameras.R[pair_idx1]), 1)
                T1        = unsqz(self.npy_xform(DTU.Cameras.T[pair_idx1]), 1)
                d1        = unsqz(self.npy_xform(DTU.Cameras.d[pair_idx1]), 1)

                K2        = unsqz(self.npy_xform(DTU.Cameras.K[pair_idx2]), 1)
                R2        = unsqz(self.npy_xform(DTU.Cameras.R[pair_idx2]), 1)
                T2        = unsqz(self.npy_xform(DTU.Cameras.T[pair_idx2]), 1)
                d2        = unsqz(self.npy_xform(DTU.Cameras.d[pair_idx2]), 1)

                sample['K'] = torch.cat((Kref, K1, K2),dim=0)
                sample['R'] = torch.cat((Rref, R1, R2),dim=0)
                sample['T'] = torch.cat((Tref, T1, T2),dim=0)
                sample['d'] = torch.cat((dref, d1, d2),dim=0)

                self.samples.append(sample)

                pbar.update(1)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        sample = self.samples[idx]

        # reference view
        input_img = torch.unsqueeze(self.img_xform(Image.open(
                        sample['img_filenames'][0]).convert('RGB')), 0)

        for i in range(1,len(sample['img_filenames'])):
            next_view = torch.unsqueeze(self.img_xform(Image.open(
                        sample['img_filenames'][i]).convert('RGB')), 0)
            input_img = torch.cat((input_img, next_view), dim=0)
         
        sample['input_img'] = input_img
        sample['depth_ref'] = unsqz(self.npy_xform(
                                load_depth(sample['depth_filename'])), 2)

        return sample
    
def load_depth(depth_filename):
    with open(depth_filename, 'rb') as f:
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
    return data

def get_dtu_loader(folder_path, cam_idx, scan_idx, event,
                    batch_size=14, i_start=0):

    if event=='training' or event=='validation':
        shuffle=True
    else:
        shuffle=False
    print("Constructing Training Dataloader...")
    DTU             = DtuReader(folder_path, cam_idx, scan_idx, event)
    dtu_dataset     = DtuTrainDataset(DTU)

    # sampler         = CustomSampler(dtu_dataset, i=i_start, 
    #                                   batch_size=batch_size)
    dtu_dataloader  = DataLoader(dataset=dtu_dataset, 
                                 batch_size=batch_size, 
                                #  sampler=sampler, 
                                 shuffle=shuffle)

    print("\nDataloader Finished.")
    print("    Batch Size: {:d}".format(batch_size))
    print("    Samples:    {:d}".format(len(dtu_dataset)))

    return dtu_dataloader

def compute_dtu_mean_and_stddev(path):
    """Compute the mean and standard deviation of all images in the dataset."""

    scans = np.array([
        1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,
        14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  28,  29,
        30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,  41,  42,
        43,  44,  45,  46,  47,  48,  49,  50,  51,  52,  53,  55,  56,
        57,  58,  59,  60,  61,  62,  63,  64,  65,  66,  67,  68,  69,
        70,  71,  72,  74,  75,  76,  77,  82,  83,  84,  85,  86,  87,
        88,  89,  90,  91,  92,  93,  94,  95,  96,  97,  98,  99, 100,
       101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113,
       114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126,
       127, 128
       ])

    DTU=DtuReader(path, np.arange(49), scans, '')

    mean = torch.zeros(3)
    std  = torch.zeros(3)

    cam_idx   = DTU.cam_idx
    scan_idx  = DTU.scan_idx
    light_idx = np.arange(7)

    sample_iter = product(np.arange(len(scan_idx)),   # scan     indexing
                          light_idx,                  # lighting indexing
                          np.arange(len(cam_idx)))    # cam      indexing

    img_xform = transforms.Compose([
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float)
        ])

    total = len(scan_idx)*len(light_idx)*len(cam_idx)
    with tqdm(total=total) as pbar:
        for (scan, light_ref, ref) in sample_iter:
            img = img_xform(Image.open(
                    DTU.Images.file_names[scan][light_ref][ref]).convert('RGB'))

            mean += torch.sum(img, (1,2))
            pbar.update(1)
    mean = mean / total

    with tqdm(total=total) as pbar:
        for (scan, light_ref, ref) in sample_iter:
            img = img_xform(Image.open(
                    DTU.Images.file_names[scan][light_ref][ref]).convert('RGB'))

            std += ((img - mean.unsqueeze(1).unsqueeze(1))**2).sum([1,2])
            pbar.update(1)
    std = torch.sqrt(std / (total*512*640))
    

    print(mean, std)
    return mean, std

class CustomSampler(Sampler):
    """Resumable sampler code adapted from 
    https://stackoverflow.com/questions/60993677/how-can-i-save-pytorchs-
    dataloader-instance to allow for pausing/resuming training with the same 
    dataloader state. By knowing i_batch during training, we can load the 
    dataloader from file and resume the sequence exactly where we left off."""
    def __init__(self, data, i=0, batch_size=14):
        random.shuffle(data)
        self.seq = list(range(len(data)))[i * batch_size:]

    def __iter__(self):
        return iter(self.seq)

    def __len__(self):
        return len(self.seq)


if __name__ == '__main__':
    """Create dataloaders using training/validation/evaluation split from
    MVS paper."""

    # 'test', 'training', 'evaluation', 'validation'
    cases = ['test', 'training', 'validation', 'evaluation']

    random.seed(401)
    path = '../data/mvs_training/dtu'

    mean, std = compute_dtu_mean_and_stddev(path)

    # cam_idx=np.arange(49)
    
    # for case in cases:
    #     if case == 'training':
    #         scan_idx=np.array([2, 6, 7, 8, 14, 16, 18, 19, 20, 22, 30, 31, 36,
    #                 39, 41, 42, 44, 45, 46, 47, 50, 51, 52, 53, 55, 57, 58,
    #                 60, 61, 63, 64, 65, 68, 69, 70, 71, 72, 74, 76, 83,
    #                 84, 85, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97,
    #                 98, 99, 100, 101, 102, 103, 104, 105, 107, 108, 109,
    #                 111, 112, 113, 115, 116, 119, 120, 121, 122, 123,
    #                 124, 125, 126, 127, 128, 3, 5, 17, 21, 28, 35, 37, 38, 40, 43, 56, 59,
    #                 66, 67, 82, 86, 106, 117, 1, 4, 9, 10, 11, 12, 13, 15, 23, 24, 29, 32, 33,
    #                 34, 48, 49, 62, 75, 77, 110, 114, 118])
    #         file_name = 'training_dataloader'
    #         batch_size = 49
    #     elif case == 'evaluation':
    #         scan_idx= np.array([1, 4, 9, 10, 11, 12, 13, 15, 23, 24, 29, 32, 33,
    #                              34, 48, 49, 62, 75, 77, 110, 114, 118])
    #         file_name = 'evaluation_dataloader'
    #         batch_size = 49
    #     elif case == 'validation':
    #         scan_idx= np.array([3, 5, 17, 21, 28, 35, 37, 38, 40, 43, 56, 59,
    #                             66, 67, 82, 86, 106, 117])
    #         file_name = 'validation_dataloader'
    #         batch_size = 49
    #     elif case == 'test':
    #         scan_idx = np.array([1,2])
    #         file_name = 'test_dataloader'
    #         batch_size = 14

    #     dtu_train_dataloader = get_dtu_loader(path, 
    #                                           cam_idx, 
    #                                           scan_idx, 
    #                                           batch_size=batch_size, 
    #                                           event=case)

    #     # save dataloader
    #     torch.save(dtu_train_dataloader, file_name)

