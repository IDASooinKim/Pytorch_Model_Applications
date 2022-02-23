#-- coding: utf-8 -*-

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
        model = LeNet(in_channels=3, num_classes=1000).cuda()
        torchsummary.summary(model, input_size=(3,224,224))
"""

class VGG11(nn.Module):

        r"""
            Return the torch modules, containing VGG11 model.
            Args:
                in_channels : the number of channels included in input data.
                num_classes : the number of classes.
        """
        
        def __init__(self, in_channels=3, num_classes=1000):
            super(VGG11, self).__init__()

            self.ConvBlock_0 = nn.Sequential(
                    collections.OrderedDict([
                        ("Conv2d_00", nn.Conv2d(
                            in_channels = in_channels,
                            out_channels = 64,
                            kernel_size = (3,3),
                            stride = (1,1),
                            padding = 1)),
                        ("Activation_00", nn.LeakyReLU()),
                        ("BatchNormal_00", nn.BatchNorm2d(64)),
                        ("Pooling2d_0", nn.MaxPool2d(kernel_size = (2,2), stride = (2,2)))
                        ]))


            self.ConvBlock_1 = nn.Sequential(
                    collections.OrderedDict([
                        ("Conv2d_10", nn.Conv2d(
                            in_channels = 64,
                            out_channels = 128,
                            kernel_size = (3,3),
                            stride = (1,1),
                            padding = 1)),
                        ("Activation_10", nn.LeakyReLU()),
                        ("BatchNormal_10", nn.BatchNorm2d(128)),
                        ("Pooling2d_1", nn.MaxPool2d(kernel_size = (2,2), stride = (2,2)))
                        ]))

            self.ConvBlock_2 = nn.Sequential(
                    collections.OrderedDict([
                        ("Conv2d_20", nn.Conv2d(
                            in_channels = 128,
                            out_channels = 256,
                            kernel_size = (3,3),
                            stride = (1,1),
                            padding = 1)),
                        ("Activation_20", nn.LeakyReLU()),
                        ("BatchNormal_20", nn.BatchNorm2d(256)),
                        
                        ("Conv2d_21", nn.Conv2d(
                            in_channels = 256,
                            out_channels = 256,
                            kernel_size = (3,3),
                            stride = (1,1),
                            padding = 1)),
                        ("Activation_21", nn.LeakyReLU()),
                        ("BatchNormal_21", nn.BatchNorm2d(256)),
                        ("Pooling2d_2", nn.MaxPool2d(kernel_size = (2,2), stride = (2,2)))
                        ]))

            self.ConvBlock_3 = nn.Sequential(
                    collections.OrderedDict([
                        ("Conv2d_30", nn.Conv2d(
                            in_channels = 256,
                            out_channels = 512,
                            kernel_size = (3,3),
                            stride = (1,1),
                            padding = 1)),
                        ("Activation_30", nn.LeakyReLU()),
                        ("BatchNormal_30", nn.BatchNorm2d(512)),
                        
                        ("Conv2d_31", nn.Conv2d(
                            in_channels = 512,
                            out_channels = 512,
                            kernel_size = (3,3),
                            stride = (1,1),
                            padding = 1)),
                        ("Activation_31", nn.LeakyReLU()),
                        ("BatchNormal_31", nn.BatchNorm2d(512)),
                        ("Pooling2d_3", nn.MaxPool2d(kernel_size = (2,2), stride = (2,2)))
                        ]))

            self.ConvBlock_4 = nn.Sequential(
                    collections.OrderedDict([
                        ("Conv2d_40", nn.Conv2d(
                            in_channels = 512,
                            out_channels = 512,
                            kernel_size = (3,3),
                            stride = (1,1),
                            padding = 1)),
                        ("Activation_40", nn.LeakyReLU()),
                        ("BatchNormal_40", nn.BatchNorm2d(512)),
                        
                        ("Conv2d_41", nn.Conv2d(
                            in_channels = 512,
                            out_channels = 512,
                            kernel_size = (3,3),
                            stride = (1,1),
                            padding = 1)),
                        ("Activation_41", nn.LeakyReLU()),
                        ("BatchNormal_41", nn.BatchNorm2d(512)),
                        ("Pooling2d_4", nn.MaxPool2d(kernel_size = (2,2), stride = (2,2)))
                        ]))

            self.MlpLayer_0 = nn.Sequential(
                    collections.OrderedDict([
                        ("Flatten", nn.Flatten()),
                        ("FC_0", nn.Linear(25088,4096)),
                        ("Activation_50", nn.LeakyReLU()),
                        ("FC_1", nn.Linear(4096,4096)),
                        ("Activation_51", nn.LeakyReLU()),
                        ("FC_2", nn.Linear(4096,1000)),
                        ("Activation_52", nn.Softmax(dim=1))
                        ]))
            r"""
                Default : Initialize weights by kaiming uniform
            """
            for m in self.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                    
                    if m.bias is not None:
                        m.bias.detach().zero_()

        def forward(self, x):

            x = self.ConvBlock_0(x)
            x = self.ConvBlock_1(x)
            x = self.ConvBlock_2(x)
            x = self.ConvBlock_3(x)
            x = self.ConvBlock_4(x)
            x = self.MlpLayer_0(x)
            
            return x

class VGG13(nn.Module):
 
        r"""
            Return the torch modules, containing VGG13 model.
            Args:
                in_channels : the number of channels included in input data.
                num_classes : the number of classes.
        """
    
        def __init__(self, in_channels=3, num_classes=1000):
            super(VGG13, self).__init__()

            self.ConvBlock_0 = nn.Sequential(
                    collections.OrderedDict([
                        ("Conv2d_00", nn.Conv2d(
                            in_channels = in_channels,
                            out_channels = 64,
                            kernel_size = (3,3),
                            stride = (1,1),
                            padding = 1)),
                        ("Activation_00", nn.LeakyReLU()),
                        ("BatchNormal_00", nn.BatchNorm2d(64)),
                        ("Pooling2d_0", nn.MaxPool2d(kernel_size = (2,2), stride = (2,2))),
                        
                        ("Conv2d_01", nn.Conv2d(
                            in_channels = 64,
                            out_channels = 64,
                            kernel_size = (3,3),
                            stride = (1,1),
                            padding = 1)),
                        ("Activation_00", nn.LeakyReLU()),
                        ("BatchNormal_00", nn.BatchNorm2d(64)),
                        ("Pooling2d_0", nn.MaxPool2d(kernel_size = (2,2), stride = (2,2)))
                        ]))


            self.ConvBlock_1 = nn.Sequential(
                    collections.OrderedDict([
                        ("Conv2d_10", nn.Conv2d(
                            in_channels = 64,
                            out_channels = 128,
                            kernel_size = (3,3),
                            stride = (1,1),
                            padding = 1)),
                        ("Activation_10", nn.LeakyReLU()),
                        ("BatchNormal_10", nn.BatchNorm2d(128)),
                        ("Pooling2d_1", nn.MaxPool2d(kernel_size = (2,2), stride = (2,2))),
                        
                        ("Conv2d_11", nn.Conv2d(
                            in_channels = 128,
                            out_channels = 128,
                            kernel_size = (3,3),
                            stride = (1,1),
                            padding = 1)),
                        ("Activation_10", nn.LeakyReLU()),
                        ("BatchNormal_10", nn.BatchNorm2d(128)),
                        ("Pooling2d_1", nn.MaxPool2d(kernel_size = (2,2), stride = (2,2)))
                        ]))

            self.ConvBlock_2 = nn.Sequential(
                    collections.OrderedDict([
                        ("Conv2d_20", nn.Conv2d(
                            in_channels = 128,
                            out_channels = 256,
                            kernel_size = (3,3),
                            stride = (1,1),
                            padding = 1)),
                        ("Activation_20", nn.LeakyReLU()),
                        ("BatchNormal_20", nn.BatchNorm2d(256)),
                        
                        ("Conv2d_21", nn.Conv2d(
                            in_channels = 256,
                            out_channels = 256,
                            kernel_size = (3,3),
                            stride = (1,1),
                            padding = 1)),
                        ("Activation_21", nn.LeakyReLU()),
                        ("BatchNormal_21", nn.BatchNorm2d(256)),
                        ("Pooling2d_2", nn.MaxPool2d(kernel_size = (2,2), stride = (2,2)))
                        ]))

            self.ConvBlock_3 = nn.Sequential(
                    collections.OrderedDict([
                        ("Conv2d_30", nn.Conv2d(
                            in_channels = 256,
                            out_channels = 512,
                            kernel_size = (3,3),
                            stride = (1,1),
                            padding = 1)),
                        ("Activation_30", nn.LeakyReLU()),
                        ("BatchNormal_30", nn.BatchNorm2d(512)),
                        
                        ("Conv2d_31", nn.Conv2d(
                            in_channels = 512,
                            out_channels = 512,
                            kernel_size = (3,3),
                            stride = (1,1),
                            padding = 1)),
                        ("Activation_31", nn.LeakyReLU()),
                        ("BatchNormal_31", nn.BatchNorm2d(512)),
                        ("Pooling2d_3", nn.MaxPool2d(kernel_size = (2,2), stride = (2,2)))
                        ]))

            self.ConvBlock_4 = nn.Sequential(
                    collections.OrderedDict([
                        ("Conv2d_40", nn.Conv2d(
                            in_channels = 512,
                            out_channels = 512,
                            kernel_size = (3,3),
                            stride = (1,1),
                            padding = 1)),
                        ("Activation_40", nn.LeakyReLU()),
                        ("BatchNormal_40", nn.BatchNorm2d(512)),
                        
                        ("Conv2d_41", nn.Conv2d(
                            in_channels = 512,
                            out_channels = 512,
                            kernel_size = (3,3),
                            stride = (1,1),
                            padding = 1)),
                        ("Activation_41", nn.LeakyReLU()),
                        ("BatchNormal_41", nn.BatchNorm2d(512)),
                        ("Pooling2d_4", nn.MaxPool2d(kernel_size = (2,2), stride = (2,2)))
                        ]))

            self.MlpLayer_0 = nn.Sequential(
                    collections.OrderedDict([
                        ("Flatten", nn.Flatten()),
                        ("FC_0", nn.Linear(25088,4096)),
                        ("Activation_50", nn.LeakyReLU()),
                        ("FC_1", nn.Linear(4096,4096)),
                        ("Activation_51", nn.LeakyReLU()),
                        ("FC_2", nn.Linear(4096,1000)),
                        ("Activation_52", nn.Softmax(dim=1))
                        ]))
            r"""
                Default : Initialize weights by kaiming uniform
            """
            for m in self.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                    
                    if m.bias is not None:
                        m.bias.detach().zero_()

        def forward(self, x):

            x = self.ConvBlock_0(x)
            x = self.ConvBlock_1(x)
            x = self.ConvBlock_2(x)
            x = self.ConvBlock_3(x)
            x = self.ConvBlock_4(x)
            x = self.MlpLayer_0(x)
            
            return x
        
class VGG16(nn.Module):
    
        r"""
            Return the torch modules, containing LeNet v1 model.
            Args:
                in_channels : the number of channels included in input data.
                num_classes : the number of classes.
        """

        def __init__(self, in_channels=3, num_classes=1000):
            super(VGG16, self).__init__()

            self.ConvBlock_0 = nn.Sequential(
                    collections.OrderedDict([
                        ("Conv2d_00", nn.Conv2d(
                            in_channels = in_channels,
                            out_channels = 64,
                            kernel_size = (3,3),
                            stride = (1,1),
                            padding = 1)),
                        ("Activation_00", nn.LeakyReLU()),
                        ("BatchNormal_00", nn.BatchNorm2d(64)),
                        
                        ("Conv2d_01", nn.Conv2d(
                            in_channels = 64,
                            out_channels = 64,
                            kernel_size = (3,3),
                            stride = (1,1),
                            padding = 1)),
                        ("Activation_01", nn.LeakyReLU()),
                        ("BatchNormal_01", nn.BatchNorm2d(64)),
                        ("Pooling2d_0", nn.MaxPool2d(kernel_size = (2,2), stride = (2,2)))
                        ]))


            self.ConvBlock_1 = nn.Sequential(
                    collections.OrderedDict([
                        ("Conv2d_10", nn.Conv2d(
                            in_channels = 64,
                            out_channels = 128,
                            kernel_size = (3,3),
                            stride = (1,1),
                            padding = 1)),
                        ("Activation_10", nn.LeakyReLU()),
                        ("BatchNormal_10", nn.BatchNorm2d(128)),
                        
                        ("Conv2d_11", nn.Conv2d(
                            in_channels = 128,
                            out_channels = 128,
                            kernel_size = (3,3),
                            stride = (1,1),
                            padding = 1)),
                        ("Activation_11", nn.LeakyReLU()),
                        ("BatchNormal_11", nn.BatchNorm2d(128)),
                        ("Pooling2d_1", nn.MaxPool2d(kernel_size = (2,2), stride = (2,2)))
                        ]))

            self.ConvBlock_2 = nn.Sequential(
                    collections.OrderedDict([
                        ("Conv2d_20", nn.Conv2d(
                            in_channels = 128,
                            out_channels = 256,
                            kernel_size = (3,3),
                            stride = (1,1),
                            padding = 1)),
                        ("Activation_20", nn.LeakyReLU()),
                        ("BatchNormal_20", nn.BatchNorm2d(256)),
                        
                        ("Conv2d_21", nn.Conv2d(
                            in_channels = 256,
                            out_channels = 256,
                            kernel_size = (3,3),
                            stride = (1,1),
                            padding = 1)),
                        ("Activation_21", nn.LeakyReLU()),
                        ("BatchNormal_21", nn.BatchNorm2d(256)),

                        ("Conv2d_22", nn.Conv2d(
                            in_channels = 256,
                            out_channels = 256,
                            kernel_size = (3,3),
                            stride = (1,1),
                            padding = 1)),
                        ("Activation_22", nn.LeakyReLU()),
                        ("BatchNormal_22", nn.BatchNorm2d(256)),

                        ("Pooling2d_2", nn.MaxPool2d(kernel_size = (2,2), stride = (2,2)))
                        ]))

            self.ConvBlock_3 = nn.Sequential(
                    collections.OrderedDict([
                        ("Conv2d_30", nn.Conv2d(
                            in_channels = 256,
                            out_channels = 512,
                            kernel_size = (3,3),
                            stride = (1,1),
                            padding = 1)),
                        ("Activation_30", nn.LeakyReLU()),
                        ("BatchNormal_30", nn.BatchNorm2d(512)),
                        
                        ("Conv2d_31", nn.Conv2d(
                            in_channels = 512,
                            out_channels = 512,
                            kernel_size = (3,3),
                            stride = (1,1),
                            padding = 1)),
                        ("Activation_31", nn.LeakyReLU()),
                        ("BatchNormal_31", nn.BatchNorm2d(512)),

                        ("Conv2d_32", nn.Conv2d(
                            in_channels = 512,
                            out_channels = 512,
                            kernel_size = (3,3),
                            stride = (1,1),
                            padding = 1)),
                        ("Activation_32", nn.LeakyReLU()),
                        ("BatchNormal_32", nn.BatchNorm2d(512)),

                        ("Pooling2d_3", nn.MaxPool2d(kernel_size = (2,2), stride = (2,2)))
                        ]))

            self.ConvBlock_4 = nn.Sequential(
                    collections.OrderedDict([
                        ("Conv2d_40", nn.Conv2d(
                            in_channels = 512,
                            out_channels = 512,
                            kernel_size = (3,3),
                            stride = (1,1),
                            padding = 1)),
                        ("Activation_40", nn.LeakyReLU()),
                        ("BatchNormal_40", nn.BatchNorm2d(512)),
                        
                        ("Conv2d_41", nn.Conv2d(
                            in_channels = 512,
                            out_channels = 512,
                            kernel_size = (3,3),
                            stride = (1,1),
                            padding = 1)),
                        ("Activation_41", nn.LeakyReLU()),
                        ("BatchNormal_41", nn.BatchNorm2d(512)),

                        ("Conv2d_42", nn.Conv2d(
                            in_channels = 512,
                            out_channels = 512,
                            kernel_size = (3,3),
                            stride = (1,1),
                            padding = 1)),
                        ("Activation_42", nn.LeakyReLU()),
                        ("BatchNormal_42", nn.BatchNorm2d(512)),

                        ("Pooling2d_4", nn.MaxPool2d(kernel_size = (2,2), stride = (2,2)))
                        ]))

            self.MlpLayer_0 = nn.Sequential(
                    collections.OrderedDict([
                        ("Flatten", nn.Flatten()),
                        ("FC_0", nn.Linear(25088,4096)),
                        ("Activation_50", nn.LeakyReLU()),
                        ("FC_1", nn.Linear(4096,4096)),
                        ("Activation_51", nn.LeakyReLU()),
                        ("FC_2", nn.Linear(4096,num_classes)),
                        ("Activation_52", nn.Softmax(dim=1))
                        ]))
            r"""
                Default : Initialize weights by kaiming uniform
            """
            for m in self.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                    
                    if m.bias is not None:
                        m.bias.detach().zero_()

        def forward(self, x):

            x = self.ConvBlock_0(x)
            x = self.ConvBlock_1(x)
            x = self.ConvBlock_2(x)
            x = self.ConvBlock_3(x)
            x = self.ConvBlock_4(x)
            x = self.MlpLayer_0(x)
            
            return x
