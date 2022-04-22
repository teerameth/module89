# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import argparse
import os
import glob
import time

import onnx
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.onnx
import torchvision.models as models

"""
This code is taken from the official DOPE Github repository:
# https://github.com/NVlabs/Deep_Object_Pose/blob/master/src/dope/inference/detector.py
# For the DopeNetwork model

This script converts pre-trained DOPE model to given format for TensorRT or Triton
infernce. It works with any pre-trained model provided on the official DOPE Github repository,
or trained using the training script
https://github.com/NVlabs/Deep_Object_Pose/blob/master/scripts/train.py in the repository.
"""

n_belief = 4
class DopeNetwork(nn.Module):
    def __init__(
            self,
            pretrained=False,
            numBeliefMap=n_belief,
            stop_at_stage=6  # number of stages to process (if less than total number of stages)
    ):
        super(DopeNetwork, self).__init__()

        self.stop_at_stage = stop_at_stage

        if pretrained is False:
            print("Training network without imagenet weights.")
        else:
            print("Training network pretrained on imagenet.")

        print(torchvision.__version__)
        # vgg_full = models.vgg19(pretrained=pretrained).features
        # mobilenetV2_full = models.mobilenet_v2(pretrained=True).features
        efficientnet_b2 = models.efficientnet_b2(pretrained=True).features
        print(efficientnet_b2)
        exit(0)
        # print(mobilenetV2_full)
        # print(vgg_full)
        # self.vgg = nn.Sequential()
        # self.mobile = nn.Sequential()
        self.efficient = nn.Sequential()
        # for i_layer in range(24): self.vgg.add_module(str(i_layer), vgg_full[i_layer])
        # for i_layer in range(7): self.mobile.add_module(str(i_layer), mobilenetV2_full[i_layer])
        for i_layer in range(4): self.efficient.add_module(str(i_layer), efficientnet_b2[i_layer])

        # Add some layers
        # i_layer = 23
        # self.vgg.add_module(str(i_layer), nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1))
        # self.vgg.add_module(str(i_layer+1), nn.ReLU(inplace=True))
        # self.vgg.add_module(str(i_layer+2), nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1))
        # self.vgg.add_module(str(i_layer+3), nn.ReLU(inplace=True))
        # i_layer = 7
        # self.mobile.add_module(str(i_layer), nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1))
        # self.mobile.add_module(str(i_layer+1), nn.ReLU(inplace=True))
        i_layer = 4
        self.efficient.add_module(str(i_layer), nn.Conv2d(48, 24, kernel_size=3, stride=1, padding=1))
        self.efficient.add_module(str(i_layer + 1), nn.ReLU(inplace=True))

        # print('---Belief------------------------------------------------')
        # _2 are the belief map stages
        # self.m1_2 = DopeNetwork.create_stage(128, numBeliefMap, True)
        # self.m2_2 = DopeNetwork.create_stage(128 + numBeliefMap, numBeliefMap, False)
        # self.m3_2 = DopeNetwork.create_stage(128 + numBeliefMap, numBeliefMap, False)
        # self.m4_2 = DopeNetwork.create_stage(128 + numBeliefMap, numBeliefMap, False)
        # self.m5_2 = DopeNetwork.create_stage(128 + numBeliefMap, numBeliefMap, False)
        # self.m6_2 = DopeNetwork.create_stage(128 + numBeliefMap, numBeliefMap, False)
        self.m1_2 = DopeNetwork.create_stage(24, numBeliefMap, True)
        self.m2_2 = DopeNetwork.create_stage(24 + numBeliefMap, numBeliefMap, False)
        self.m3_2 = DopeNetwork.create_stage(24 + numBeliefMap, numBeliefMap, False)
        self.m4_2 = DopeNetwork.create_stage(24 + numBeliefMap, numBeliefMap, False)
        self.m5_2 = DopeNetwork.create_stage(24 + numBeliefMap, numBeliefMap, False)
        self.m6_2 = DopeNetwork.create_stage(24 + numBeliefMap, numBeliefMap, False)

    def forward(self, x):
        '''Runs inference on the neural network'''
        # out1 = self.vgg(x)
        # out1 = self.mobile(x)
        out1 = self.efficient(x)
        out1_2 = self.m1_2(out1)
        if self.stop_at_stage == 1: return [out1_2]
        out2 = torch.cat([out1_2, out1], 1)
        out2_2 = self.m2_2(out2)
        if self.stop_at_stage == 2: return [out1_2, out2_2]
        out3 = torch.cat([out2_2, out1], 1)
        out3_2 = self.m3_2(out3)
        if self.stop_at_stage == 3: return [out1_2, out2_2, out3_2]
        out4 = torch.cat([out3_2, out1], 1)
        out4_2 = self.m4_2(out4)
        if self.stop_at_stage == 4: return [out1_2, out2_2, out3_2, out4_2]
        out5 = torch.cat([out4_2, out1], 1)
        out5_2 = self.m5_2(out5)
        if self.stop_at_stage == 5: return [out1_2, out2_2, out3_2, out4_2, out5_2]
        out6 = torch.cat([out5_2, out1], 1)
        out6_2 = self.m6_2(out6)
        return [out1_2, out2_2, out3_2, out4_2, out5_2, out6_2]

    @staticmethod
    def create_stage(in_channels, out_channels, first=False):  # Create the neural network layers for a single stage.
        model = nn.Sequential()
        mid_channels = 128
        if first:
            padding = 1
            kernel = 3
            count = 6
            # count = 4
            final_channels = 512
        else:
            padding = 3
            kernel = 7
            count = 10
            # count = 6
            final_channels = mid_channels

        # First convolution
        model.add_module("0",
                         nn.Conv2d(
                             in_channels,
                             mid_channels,
                             kernel_size=kernel,
                             stride=1,
                             padding=padding)
                         )

        # Middle convolutions
        i = 1
        while i < count - 1:
            model.add_module(str(i), nn.ReLU(inplace=True))
            i += 1
            model.add_module(str(i),
                             nn.Conv2d(
                                 mid_channels,
                                 mid_channels,
                                 kernel_size=kernel,
                                 stride=1,
                                 padding=padding))
            i += 1

        # Penultimate convolution
        model.add_module(str(i), nn.ReLU(inplace=True))
        i += 1
        model.add_module(str(i), nn.Conv2d(mid_channels, final_channels, kernel_size=1, stride=1))
        i += 1

        # Last convolution
        model.add_module(str(i), nn.ReLU(inplace=True))
        i += 1
        model.add_module(str(i), nn.Conv2d(final_channels, out_channels, kernel_size=1, stride=1))
        i += 1

        return model


def get_node_output_shape(node):
    return [x.dim_value for x in node.type.tensor_type.shape.dim]


def save_pytorch(net, x, output_file):
    jit_net = torch.jit.script(net.module, (x, ))
    torch.jit.save(jit_net, output_file)


def save_onnx(net, x, output_file, input_name, output_name):
    torch.onnx.export(
        net.module, x, output_file, input_names=[input_name], output_names=[output_name])

    # Validate and log onnx model information
    model = onnx.load(output_file)
    net_output = [(node.name, get_node_output_shape(node)) for node in model.graph.output]

    input_initializer = [node.name for node in model.graph.initializer]
    net_feed_input = [(node.name, get_node_output_shape(node)) for node in model.graph.input
                      if node.name not in input_initializer]

    print('\n=== onnx model info ===')
    print('Inputs:')
    print('\n'.join(map(str, net_feed_input)))
    print('Outputs:')
    print('\n'.join(map(str, net_output)))

    onnx.checker.check_model(model)


def convert(pth_file):
    output_file = pth_file[:-3] + "onnx"
    model_loading_start_time = time.time()
    print('Loading torch model {}'.format(pth_file))
    net = torch.nn.DataParallel(DopeNetwork(), [0]).cuda()
    net.load_state_dict(torch.load(pth_file), strict=True)
    print('Model loaded in {0:.2f} seconds'.format(time.time() - model_loading_start_time))

    x = Variable(torch.randn(1, 3, 480, 640)).cuda()
    # Export the model
    save_onnx(net, x, output_file, 'input', 'output')
    print('Saved output model to {}'.format(output_file))


if __name__ == '__main__':
    pth_list = glob.glob("/media/teera/ROGESD/model/belief/chessboard_mono_6_stage_lr_0.00003/*.pth")
    for pth_file in pth_list:
        convert(pth_file)
