# -*- coding: utf-8 -*- 
from __future__ import print_function, division

"""
Encoder models
"""

import torch
import torch.nn as nn
import torchvision.models as models


class TripletAlexNet(nn.Module):
    def __init__(self, out_size=300, pretrained=True, attention=True):
        super(TripletAlexNet, self).__init__()        
        # Embedd images in a 1000 space
        alex_aux = models.alexnet(pretrained)
        self.cnn_features = alex_aux.features
        self.map = alex_aux.classifier
        del self.map._modules['6']
        del self.map._modules['5']

    def forward(self, im):
        x = self.cnn_features(im) # Size (BS x 512 x 7 x 7)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.map(x)


        return x, None

