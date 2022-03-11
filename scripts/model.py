from numpy import extract
import torch
import torch.nn as nn

from homography import homography_warping
from costvolume import assemble_cost_volume
from depthmap import extract_depth_map

# from torch.autograd import Variable
# import numpy as np
# from PIL import Image
# from torch.nn.modules.activation import ReLU
# from torch.nn.modules.conv import Conv2d
# from torch.nn.modules.flatten import Flatten
# from torch.nn.modules.pooling import MaxPool2d


class FeatureEncoder(nn.Module):

    def __init__(self, in_ch=3, base_filt=8, device=None):
        super(FeatureEncoder, self).__init__()

        # example convolutional network pytorch vs tensorflow:
        # filters = out_channels
        # third dimension of the input shape is the in_channels
        # n.Conv2d(1, 6, kernel_size=5, stride=1, padding=2)
        # keras.layers.Conv2D(input_shape=(28,28,1), filters=6, kernel_size=5, strides=1, padding="same", activation=tf.nn.relu)
        #
        # MVSNet
        # conv = self.conv(input_tensor, kernel_size, filters, strides, name, relu=False,
        #                  dilation_rate=dilation_rate, padding=padding,
        #                  biased=biased, reuse=reuse, separable=separable)
        # conv_bn = self.batch_normalization(conv, name + '/bn',
        #                                    center=center, scale=scale, relu=relu, reuse=reuse)
        # 
        # .conv_bn(3, base_filter, 1, center=True, scale=True, name='conv0_0')
        # .conv_bn(3, base_filter, 1, center=True, scale=True, name='conv0_1')
        # .conv_bn(5, base_filter * 2, 2, center=True, scale=True, name='conv1_0')
        # .conv_bn(3, base_filter * 2, 1, center=True, scale=True, name='conv1_1')
        # .conv_bn(3, base_filter * 2, 1, center=True, scale=True, name='conv1_2')
        # .conv_bn(5, base_filter * 4, 2, center=True, scale=True, name='conv2_0')
        # .conv_bn(3, base_filter * 4, 1, center=True, scale=True, name='conv2_1')
        # .conv(3, base_filter * 4, 1, biased=False, relu=False, name='conv2_2'))
        # .conv_bn(kernel, filters, stride, center=True, scale=True, name='conv0_0')

        self.model = nn.Sequential(
            ConvLayer2D(in_ch, base_filt, kernel=3, stride=1, padding=1, device=device),
            BNLayer(base_filt), ReLULayer(),

            ConvLayer2D(base_filt, base_filt, kernel=3, stride=1, padding=1, device=device),
            BNLayer(base_filt), ReLULayer(),

            ConvLayer2D(base_filt, base_filt*2, kernel=5, stride=2, padding=2, device=device),
            BNLayer(base_filt*2), ReLULayer(),

            ConvLayer2D(base_filt*2, base_filt*2, kernel=3, stride=1, padding=1, device=device),
            BNLayer(base_filt*2), ReLULayer(),

            ConvLayer2D(base_filt*2, base_filt*2, kernel=3, stride=1, padding=1, device=device),
            BNLayer(base_filt*2), ReLULayer(),

            ConvLayer2D(base_filt*2, base_filt*4, kernel=5, stride=2, padding=2, device=device),
            BNLayer(base_filt*4), ReLULayer(),

            ConvLayer2D(base_filt*4, base_filt*4, kernel=3, stride=1, padding=1, device=device),
            BNLayer(base_filt*4), ReLULayer(),

            ConvLayer2D(base_filt*4, base_filt*4, kernel=3, stride=1, padding=1, bias=False, device=device)
        )

    def forward(self, input):
        """Input is a batch set of 3-channel RGB images."""
        return self.model(input)


class CostVolumeReg(nn.Module):

    def __init__(self, in_ch=32, base_filt=8, device=None):
        super(CostVolumeReg, self).__init__()

    #  (self.feed('data')
    # .conv_bn(3, base_filter * 2, 2, center=True, scale=True, name='3dconv1_0') #32 in channels, 16 out channels, stride 2
    # .conv_bn(3, base_filter * 4, 2, center=True, scale=True, name='3dconv2_0') #32 in channels, 32 out channels, stride 2
    # .conv_bn(3, base_filter * 8, 2, center=True, scale=True, name='3dconv3_0'))#32 in channels, 64 out channels, stride 2

    # (self.feed('data')
    # .conv_bn(3, base_filter, 1, center=True, scale=True, name='3dconv0_1'))    #32 in channels, 8 out channels

    # (self.feed('3dconv1_0')
    # .conv_bn(3, base_filter * 2, 1, center=True, scale=True, name='3dconv1_1'))#16 in channels 16 out channels

    # (self.feed('3dconv2_0')
    # .conv_bn(3, base_filter * 4, 1, center=True, scale=True, name='3dconv2_1'))#32 in channels, 32 out channels

    # (self.feed('3dconv3_0')
    # .conv_bn(3, base_filter * 8, 1, center=True, scale=True, name='3dconv3_1') #64 in channels, 64 out channels
    # .deconv_bn(3, base_filter * 4, 2, center=True, scale=True, name='3dconv4_0'))#64 in channels, 32 out channels, stride 2

    # (self.feed('3dconv4_0', '3dconv2_1')
    # .add(name='3dconv4_1')
    # .deconv_bn(3, base_filter * 2, 2, center=True, scale=True, name='3dconv5_0'))#32 in channels, 16 out channels, stride 2

    # (self.feed('3dconv5_0', '3dconv1_1')
    # .add(name='3dconv5_1')
    # .deconv_bn(3, base_filter, 2, center=True, scale=True, name='3dconv6_0'))  #16 in channels, 8 out channels, stride 2

    # (self.feed('3dconv6_0', '3dconv0_1')
    # .add(name='3dconv6_1')
    # .conv(3, 1, 1, biased=False, relu=False, name='3dconv6_2')) #8 in channels, 1 out channel, stride 1

        # self.conv_0_0 = ConvLayer3D(in_ch, base_filt*1, kernel=3, stride=1) # input data
        # self.conv_1_0 = ConvLayer3D(in_ch, base_filt*2, kernel=3, stride=2)
        # self.conv_2_0 = ConvLayer3D(in_ch, base_filt*4, kernel=3, stride=2)
        # self.conv_3_0 = ConvLayer3D(in_ch, base_filt*8, kernel=3, stride=2)

        # self.conv_1_1 = ConvLayer3D(base_filt*2, base_filt*2, kernel=3, stride=1) # input conv 1_0
        # self.conv_2_1 = ConvLayer3D(base_filt*4, base_filt*4, kernel=3, stride=1) # input conv 2_0
        # self.conv_3_1 = ConvLayer3D(base_filt*8, base_filt*8, kernel=3, stride=1) # input conv 3_0

        # self.deconv_3_0 = DeConvLayer3D(base_filt*8, base_filt*4, kernel=3, stride=2) # input conv 3_0
        # self.deconv_2_0 = DeConvLayer3D(base_filt*4, base_filt*2, kernel=3, stride=2) # input deconv 3_0, conv 2_1
        # self.deconv_1_0 = DeConvLayer3D(base_filt*2, base_filt*1, kernel=3, stride=2) # input deconv 2_0, conv 1_1

        # self.conv_out = ConvLayer3D(base_filt*1, 1, kernel=3, stride=1) # input deconv 1_0, conv 0_1

        # self.ReLU = ReLULayer()
        # self.BN_0 = BNLayer(base_filt*1)
        # self.BN_1 = BNLayer(base_filt*2)
        # self.BN_2 = BNLayer(base_filt*4)
        # self.BN_3 = BNLayer(base_filt*8)

        self.conv_0_0 = ConvLayer2D(in_ch, base_filt*1, kernel=3, stride=1, padding=1      , device=device) # input data
        self.conv_1_0 = ConvLayer2D(in_ch, base_filt*2, kernel=3, stride=2, padding=(65,81), device=device)
        self.conv_2_0 = ConvLayer2D(in_ch, base_filt*4, kernel=3, stride=2, padding=(65,81), device=device)
        self.conv_3_0 = ConvLayer2D(in_ch, base_filt*8, kernel=3, stride=2, padding=(65,81), device=device)

        self.conv_1_1 = ConvLayer2D(base_filt*2, base_filt*2, kernel=3, stride=1, padding=1, device=device) # input conv 1_0
        self.conv_2_1 = ConvLayer2D(base_filt*4, base_filt*4, kernel=3, stride=1, padding=1, device=device) # input conv 2_0
        self.conv_3_1 = ConvLayer2D(base_filt*8, base_filt*8, kernel=3, stride=1, padding=1, device=device) # input conv 3_0

        self.deconv_3_0 = DeConvLayer2D(base_filt*8, base_filt*4, kernel=3, stride=2, padding=(65,81), output_padding=1, device=device) # input conv 3_1
        self.deconv_2_0 = DeConvLayer2D(base_filt*4, base_filt*2, kernel=3, stride=2, padding=(65,81), output_padding=1, device=device) # input deconv 3_0, conv 2_1
        self.deconv_1_0 = DeConvLayer2D(base_filt*2, base_filt*1, kernel=3, stride=2, padding=(65,81), output_padding=1, device=device) # input deconv 2_0, conv 1_1

        self.conv_out = ConvLayer2D(base_filt*1, 1, kernel=3, stride=1, padding=1, device=device) # input deconv 1_0, conv 0_1

        self.ReLU = ReLULayer()
        self.BN_0 = BNLayer(base_filt*1)
        self.BN_1 = BNLayer(base_filt*2)
        self.BN_2 = BNLayer(base_filt*4)
        self.BN_3 = BNLayer(base_filt*8)

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
        y  = self.conv_out(torch.add(y1, y0)) # in 8, out 1
        # print("y : ",y.size())

        return y


class DepthRefinement(nn.Module):

    def __init__(self, in_ch=4, base_filt=32, device=None):
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

        # TODO: BE SURE TO USE ONLY TORCH OPERATIONS TO ENSURE DIFFERENTIABILITY

        feature_maps = self.feature_encoder(nn_input) # out: b, 32 , h , w

        print("Batch ##:Feature Network Output Size ", feature_maps.size())

        warped_feature_maps = homography_warping(K_batch, R_batch, T_batch, 
                                d_min, d_int, feature_maps, batch_size, n_views)

        raise Exception("Stop Here")

        cost_volume = assemble_cost_volume(warped_feature_maps)

        prob_volume = self.cost_volume_reg(cost_volume)

        initial_depth_map = extract_depth_map(prob_volume)

        # Don't forget to scale the depth values before refining

        refine_input = torch.cat((initial_depth_map, nn_input[0,:,:,:]),dim=1)

        refined_depth_map = self.depthmap_refine(refine_input)

        return initial_depth_map, refined_depth_map


def ConvLayer2D(in_ch, out_ch, stride=1, kernel=3, device=None, 
                padding=0, padmode='zeros', bias=True):

    return nn.Conv2d(in_ch, out_ch, kernel, stride=stride,
        padding=padding, padding_mode=padmode, bias=False, device=device)

def DeConvLayer2D(in_ch, out_ch, stride=1, kernel=3, device=None, 
                padding=0, output_padding=0, padmode='zeros', bias=True):

    return nn.ConvTranspose2d(in_ch, out_ch, kernel, stride=stride,
        padding=padding, output_padding=output_padding, padding_mode=padmode, 
        bias=bias, device=device)

def ConvLayer3D(in_ch, out_ch, stride=1, kernel=3, device=None, 
                padding=0, padmode='zeros', bias=True):

    return nn.Conv3d(in_ch, out_ch, kernel, stride=stride,
        padding=padding, padding_mode=padmode, bias=bias, device=device)

def DeConvLayer3D(in_ch, out_ch, stride=1, kernel=3, device=None, 
                padding=0, output_padding=0, padmode='zeros', bias=True):

    return nn.ConvTranspose3d(in_ch, out_ch, kernel, stride=stride,
        padding=padding, output_padding=output_padding, padding_mode=padmode, 
        bias=bias, device=device)
    
def BNLayer(in_ch, eps=1e-05, momentum=0.1, device=None):

    return nn.BatchNorm2d(in_ch, eps=eps, momentum=momentum, device=device)

def ReLULayer():
    return nn.ReLU()
