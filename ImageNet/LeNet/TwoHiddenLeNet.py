#-*- coding: utf-8 -*-

r"""
    @Copyright 2022 
    The Intelligence Data Analysis, in DKU. All Rights Reserved.    
"""

import torch
import torch.nn as nn
import torchsummary
import collections

r"""
    [Examples]
    for load model and print model structure
        model = LeNet(in_channels=1, num_classes=1000).cuda()
        torchsummary.summary(model, input_size=(1,28,28))
"""
class LeNet(nn.Module):
    r"""
        Return the torch modules, containing LeNet v1 model.

        Args:
            in_channels : the number of channels included in input data.
            num_classes : the number of classes.
    """
    def __init__(self, in_channels=1, num_classes=1000):
        super(LeNet, self).__init__()

        self.ConvBlock_0 = nn.Sequential(
                collections.OrderedDict([
                    ("Conv2d_0", nn.Conv2d(
                        in_channels = in_channels,
                        out_channels = 20,
                        kernel_size = 5,
                        stride = 1,
                        padding = 2)),
                    ("Activation_0", nn.Tanh()),
                    ("Pooling2d_0", nn.AvgPool2d(kernel_size=2, stride=2))
                    ]))

        self.ConvBlock_1 = nn.Sequential(
                collections.OrderedDict([
                    ("Conv2d_1", nn.Conv2d(
                        in_channels = 20,
                        out_channels = 50,
                        kernel_size = 5,
                        stride = 1,
                        padding = 2)),
                    ("Activation_1", nn.Tanh()),
                    ("Pooling2d_1", nn.AvgPool2d(kernel_size=2, stride=2))
                    ]))

        self.MlpLayer_0 = nn.Sequential(
                collections.OrderedDict([
                    ("Flatten", nn.Flatten()),
                    ("FC_0", nn.Linear(in_features=7*7*50, out_features=500)),
                    ("Activation_2", nn.Tanh()),
                    ("FC_1", nn.Linear(in_features=500, out_features=10)),
                    ("Activation_3", nn.Softmax(dim=1))
                    ]))

    def forward(self, x):
        
        x = self.ConvBlock_0(x)
        x = self.ConvBlock_1(x)
        out = self.MlpLayer_0(x)
        
        return x
