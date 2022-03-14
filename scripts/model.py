import torch
import torch.nn as nn
import torch.nn.functional as f

from homography import homography_warping
from costvolume import assemble_cost_volume
from depthmap import extract_depth_map
from utils import print_gpu_memory

from config import D_NUM, DEVICE, PAD, OUTPAD, DIM_REDUCE

# from torch.autograd import Variable
# import numpy as np
# from PIL import Image
# from torch.nn.modules.activation import ReLU
# from torch.nn.modules.conv import Conv2d
# from torch.nn.modules.flatten import Flatten
# from torch.nn.modules.pooling import MaxPool2d


class FeatureEncoder(nn.Module):

    def __init__(self, in_ch=3, base_filt=8, device=DEVICE):
        super(FeatureEncoder, self).__init__()

        out_1 = base_filt
        out_2 = int(base_filt * DIM_REDUCE/2)
        out_3 = int(base_filt * DIM_REDUCE  )

        self.model = nn.Sequential(

            ConvLayer2D(in_ch, out_1, kernel=3, stride=1, padding=1, device=device),
            BNLayer(out_1), ReLULayer(),

            ConvLayer2D(out_1, out_1, kernel=3, stride=1, padding=1, device=device),
            BNLayer(out_1), ReLULayer(),

            ConvLayer2D(out_1, out_2, kernel=5, stride=2, padding=2, device=device),
            BNLayer(out_2), ReLULayer(),

            ConvLayer2D(out_2, out_2, kernel=3, stride=1, padding=1, device=device),
            BNLayer(out_2), ReLULayer(),

            ConvLayer2D(out_2, out_2, kernel=3, stride=1, padding=1, device=device),
            BNLayer(out_2), ReLULayer(),

            ConvLayer2D(out_2, out_3, kernel=5, stride=2, padding=2, device=device),
            BNLayer(out_3), ReLULayer(),

            ConvLayer2D(out_3, out_3, kernel=3, stride=1, padding=1, device=device),
            BNLayer(out_3), ReLULayer(),

            ConvLayer2D(out_3, out_3, kernel=3, stride=1, padding=1, bias=False, device=device)
        )

    def forward(self, input):
        """Input is a batch set of 3-channel RGB images."""
        return self.model(input)


class CostVolumeReg(nn.Module):

    def __init__(self, in_ch=32, base_filt=8, device=DEVICE):
        super(CostVolumeReg, self).__init__()

        self.conv_0_0 = ConvLayer3D(in_ch, base_filt*1, kernel=3, stride=1, padding=1, device=device) # input data
        self.conv_1_0 = ConvLayer3D(in_ch, base_filt*2, kernel=3, stride=2, padding=PAD, device=device)
        self.conv_2_0 = ConvLayer3D(in_ch, base_filt*4, kernel=3, stride=2, padding=PAD, device=device)
        self.conv_3_0 = ConvLayer3D(in_ch, base_filt*8, kernel=3, stride=2, padding=PAD, device=device)

        self.conv_1_1 = ConvLayer3D(base_filt*2, base_filt*2, kernel=3, stride=1, padding=1, device=device) # input conv 1_0
        self.conv_2_1 = ConvLayer3D(base_filt*4, base_filt*4, kernel=3, stride=1, padding=1, device=device) # input conv 2_0
        self.conv_3_1 = ConvLayer3D(base_filt*8, base_filt*8, kernel=3, stride=1, padding=1, device=device) # input conv 3_0

        self.deconv_3_0 = DeConvLayer3D(base_filt*8, base_filt*4, kernel=3, stride=2, padding=PAD, output_padding=OUTPAD, device=device) # input conv 3_1
        self.deconv_2_0 = DeConvLayer3D(base_filt*4, base_filt*2, kernel=3, stride=2, padding=PAD, output_padding=OUTPAD, device=device) # input deconv 3_0, conv 2_1
        self.deconv_1_0 = DeConvLayer3D(base_filt*2, base_filt*1, kernel=3, stride=2, padding=PAD, output_padding=OUTPAD, device=device) # input deconv 2_0, conv 1_1

        self.conv_out = ConvLayer3D(base_filt*1, 1, kernel=3, stride=1, padding=1, bias=False, device=device) # input deconv 1_0, conv 0_1

        self.ReLU = ReLULayer()
        self.BN_0 = BNLayer3D(base_filt*1)
        self.BN_1 = BNLayer3D(base_filt*2)
        self.BN_2 = BNLayer3D(base_filt*4)
        self.BN_3 = BNLayer3D(base_filt*8)
        self.Norm = torch.nn.Softmax(2) # normalize depth dimension

    def forward(self, cv):
        y0 = self.ReLU(self.BN_0(self.conv_0_0(cv))) # in 32, out 8
        # print("y0: ",y0.size())
        y1 = self.ReLU(self.BN_1(self.conv_1_0(cv))) # in 32, out 16
        # print("y1: ",y1.size())
        y2 = self.ReLU(self.BN_2(self.conv_2_0(cv))) # in 32, out 32
        # print("y2: ",y2.size())
        y3 = self.ReLU(self.BN_3(self.conv_3_0(cv))) # in 32, out 64
        # print("y3: ",y3.size())

        y1 = self.ReLU(self.BN_1(self.conv_1_1(y1))) # in 16, out 16
        # print("y1: ",y1.size())
        y2 = self.ReLU(self.BN_2(self.conv_2_1(y2))) # in 32, out 32
        # print("y2: ",y2.size())
        y3 = self.ReLU(self.BN_3(self.conv_3_1(y3))) # in 64, out 64
        # print("y3: ",y3.size())

        y3 = self.ReLU(self.BN_2(self.deconv_3_0(y3))) # in 64, out 32
        # print("y3: ",y3.size())
        y2 = self.ReLU(self.BN_1(self.deconv_2_0(torch.add(y3, y2)))) # in 32, out 16
        # print("y2: ",y2.size())
        y1 = self.ReLU(self.BN_0(self.deconv_1_0(torch.add(y2, y1)))) # in 16, out 8
        # print("y1: ",y1.size())
        y  = self.Norm(self.conv_out(torch.add(y1, y0))) # in 8, out 1
        # print("y : ",y.size())

        return y


class DepthRefinement(nn.Module):

    def __init__(self, in_ch=4, base_filt=32, device=DEVICE):
        super(DepthRefinement, self).__init__()

        # .conv_bn(3, 32, 1, name='refine_conv0')
        # .conv_bn(3, 32, 1, name='refine_conv1')
        # .conv_bn(3, 32, 1, name='refine_conv2')
        # .conv(3, 1, 1, relu=False, name='refine_conv3'))
        # .conv_bn(kernel, filters, stride, center=True, scale=True, name='conv0_0')

        self.model = nn.Sequential(
            ConvLayer2D(in_ch    , base_filt, kernel=3, stride=1, padding=1, device=device),
            BNLayer(base_filt), ReLULayer(),

            ConvLayer2D(base_filt, base_filt, kernel=3, stride=1, padding=1, device=device),
            BNLayer(base_filt), ReLULayer(),

            ConvLayer2D(base_filt, base_filt, kernel=3, stride=1, padding=1, device=device),
            BNLayer(base_filt), ReLULayer(),

            ConvLayer2D(base_filt, 1        , kernel=3, stride=1, padding=1, device=device)
        )

    def forward(self, depth_and_input):
        """Input is 4-channel: depth map + input image"""
        return self.model(depth_and_input)


class MVSNet(nn.Module):

    def __init__(self):
        super(MVSNet, self).__init__()

        self.feature_encoder = FeatureEncoder()
        self.cost_volume_reg = CostVolumeReg()
        self.depthmap_refine = DepthRefinement()

        self.parameters = list(self.feature_encoder.parameters()) + \
                          list(self.cost_volume_reg.parameters()) + \
                          list(self.depthmap_refine.parameters())

    def forward(self, nn_input, K_batch, R_batch, T_batch, d_min, 
                    d_int, batch_size, n_views):

        # USE ONLY TORCH OPERATIONS TO ENSURE DIFFERENTIABILITY

        feature_maps = self.feature_encoder(nn_input) # out: b, 32 , h , w

        # print("Feature Map Computed with Shape ", feature_maps.size())
        # print_gpu_memory()

        warped_feature_maps, d_batch, ref_views = homography_warping(K_batch, R_batch, T_batch, 
                                d_min, d_int, feature_maps, batch_size, n_views)

        # print("Feature Volumes Computed with Shape ", warped_feature_maps.size())
        # print_gpu_memory()

        cost_volume = assemble_cost_volume(warped_feature_maps, n_views)

        # print("Cost Volumes Computed with Shape ", cost_volume.size())
        # print_gpu_memory()

        prob_volume = self.cost_volume_reg(cost_volume)

        # print("Probability Volumes Computed with Shape ", prob_volume.size())
        # print_gpu_memory()

        initial_depth_map = extract_depth_map(prob_volume, d_batch)

        # print("Initial Depth Map Computed with Shape ", initial_depth_map.size())
        # print_gpu_memory()

        # Don't forget to scale the depth values before refining
        d_trans = d_min.to(DEVICE)
        d_scale = d_int.to(DEVICE).mul(D_NUM)
        tran_depth_map = torch.subtract(initial_depth_map, d_trans)
        norm_depth_map = torch.div(tran_depth_map, d_scale)

        _, _, h, w = norm_depth_map.size()

        # reshape input images and concatenate with depth map
        refine_input = torch.cat((norm_depth_map, f.interpolate(nn_input[ref_views], (h, w),mode='bilinear')), dim=1)

        norm_refined_depth_map = self.depthmap_refine(refine_input) + norm_depth_map

        refined_depth_map = norm_refined_depth_map.mul(d_scale).add(d_trans)

        # print("Refined Depth Map Computed with Shape ", refined_depth_map.size())
        # print_gpu_memory()

        return initial_depth_map, refined_depth_map


def ConvLayer2D(in_ch, out_ch, stride=1, kernel=3, device=None, 
                padding=0, padmode='zeros', bias=False):

    return nn.Conv2d(in_ch, out_ch, kernel, stride=stride,
        padding=padding, padding_mode=padmode, bias=False, device=device)

def DeConvLayer2D(in_ch, out_ch, stride=1, kernel=3, device=None, 
                padding=0, output_padding=0, padmode='zeros', bias=False):

    return nn.ConvTranspose2d(in_ch, out_ch, kernel, stride=stride,
        padding=padding, output_padding=output_padding, padding_mode=padmode, 
        bias=bias, device=device)

def ConvLayer3D(in_ch, out_ch, stride=1, kernel=3, device=None, 
                padding=0, padmode='zeros', bias=False):

    return nn.Conv3d(in_ch, out_ch, kernel, stride=stride,
        padding=padding, padding_mode=padmode, bias=bias, device=device)

def DeConvLayer3D(in_ch, out_ch, stride=1, kernel=3, device=None, 
                padding=0, output_padding=0, padmode='zeros', bias=False):

    return nn.ConvTranspose3d(in_ch, out_ch, kernel, stride=stride,
        padding=padding, output_padding=output_padding, padding_mode=padmode, 
        bias=bias, device=device)
    
def BNLayer(in_ch, eps=1e-05, momentum=0.1, device=None):

    return nn.BatchNorm2d(in_ch, eps=eps, momentum=momentum, 
                            track_running_stats=True, device=device)

def BNLayer3D(in_ch, eps=1e-05, momentum=0.1, device=None):

    return torch.nn.BatchNorm3d(in_ch, eps=eps, momentum=momentum, 
                                track_running_stats=True, device=device)

def ReLULayer():
    return nn.ReLU()
