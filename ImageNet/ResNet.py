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
        model = LeNet(in_channels=1, num_classes=1000).cuda()
        torchsummary.summary(model, input_size=(1,28,28))
    [Note]
    Default model is ResNet50. you can extend ResNet manually below code.
    e.g. resnet = ResNet(repeats=[3,4,23,3], num_classes = 10) # 101 Layer
         resnet = ResNet(repeats=[3,8,36,3], num_classes = 10) # 152 Layer
"""

def conv_start():
    return nn.Sequential(
            collections.OrderedDict([
                ("Conv2d_00", nn.Conv2d(
                    in_channels = 3,
                    out_channels = 64,
                    kernel_size = (7,7),
                    stride = (2,2),
                    padding = 4)),
                ("BatchNormal_00", nn.BatchNorm2d(64)),
                ("Activation_00", nn.LeakyReLU()),
                ("Pooling2d_00", nn.MaxPool2d(kernel_size = (3,3), stride = (2,2)))
                ]))

def bottleneck_block(in_channels, mid_channels, out_channels, down=False):
    r"""
        Return BottleNeck layer.
        Args:
            in_channels : Depth of previous extracted features.
            mid_channels : Depth of second extracted features in bottle-neck.
            out_channels : Depth of last extracted or out features in bottle-neck.
            dowm : if False, middle layer are used for feature size reduction.
            
        Term:
            BottleNeck
                A typical convolution layer has a kernel size of 2 or more, allowing spatial information to be extracted.
                In contrast, BottleNeck layer has single kernel size only for calculation volumn reduction.
    """
    layers = []
    
    if down:
        layers.append(nn.Conv2d(in_channels, mid_channels, kernel_size=(1,1), stride=(2,2), padding=0))
    else:
        layers.append(nn.Conv2d(in_channels, mid_channels, kernel_size=(1,1), stride=(1,1), padding=0))
    
    layers.extend([
        nn.BatchNorm2d(mid_channels),
        nn.LeakyReLU(),
        nn.Conv2d(mid_channels, mid_channels, kernel_size=(3,3), stride=(1,1), padding=1),
        nn.BatchNorm2d(mid_channels),
        nn.LeakyReLU(),
        nn.Conv2d(mid_channels, out_channels, kernel_size=(1,1), stride=(1,1), padding=0),
        nn.BatchNorm2d(out_channels),
    ])
    
    return nn.Sequential(*layers)

class Bottleneck(nn.Module):
    
    def __init__(self, in_channels, mid_channels, out_channels, down:bool = False, starting:bool=False) -> None:
        super(Bottleneck, self).__init__()
        
        if starting:
            down = False
        self.block = bottleneck_block(in_channels, mid_channels, out_channels, down=down)
        self.relu = nn.LeakyReLU()
        
        if down:
            conn_layer = nn.Conv2d(in_channels, out_channels, kernel_size=(1,1), stride=(2,2), padding=0)
        else:
            conn_layer = nn.Conv2d(in_channels, out_channels, kernel_size=(1,1), stride=(1,1), padding=0)

        self.changedim = nn.Sequential(conn_layer, nn.BatchNorm2d(out_channels))

    def forward(self, x):
        
        identity = self.changedim(x)
        x = self.block(x)
        x += identity
        x = self.relu(x)
        
        return x

def make_layer(in_channels, mid_channels, out_channels, repeats, starting=False):
        
        layers = []
        layers.append(Bottleneck(in_channels, mid_channels, out_channels, down=True, starting=starting))
        
        for _ in range(1, repeats):
            layers.append(Bottleneck(out_channels, mid_channels, out_channels, down=False))
        
        return nn.Sequential(*layers)

class ResNet(nn.Module):
   r"""
        Return the torch modules, containing VGG11 model.
        Args:
            repeats:list : repeat skip-connection
            num_classes : the number of classes.
    """
    def __init__(self, repeats:list = [3,4,6,3], num_classes=1000):
        super(ResNet, self).__init__()
        self.num_classes = num_classes
        
        self.conv1 = conv_start()
                
        base_dim = 64
        self.conv2 = make_layer(base_dim, base_dim, base_dim*4, repeats[0], starting=True)
        self.conv3 = make_layer(base_dim*4, base_dim*2, base_dim*8, repeats[1])
        self.conv4 = make_layer(base_dim*8, base_dim*4, base_dim*16, repeats[2])
        self.conv5 = make_layer(base_dim*16, base_dim*8, base_dim*32, repeats[3])
                
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.classifer = nn.Linear(2048, self.num_classes)

    def forward(self, x):
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifer(x)
        
        return x
