# -*- coding: utf-8 -*- 
from __future__ import print_function, division

"""
Encoder models
"""

import torch
import torch.nn as nn
import torchvision.models as models

class AttentionModel(nn.Module):
    def __init__(self, hidden_layer=380):
        super(AttentionModel, self).__init__()        
        
        self.attn_hidden_layer = hidden_layer
        self.net = nn.Sequential(nn.Conv2d(512, self.attn_hidden_layer, kernel_size=1), nn.Conv2d(self.attn_hidden_layer, 1, kernel_size=1))

    def forward(self, x):
        attn_mask = self.net(x)
        attn_mask = attn_mask.view(attn_mask.size(0), -1)
        attn_mask = nn.Softmax(dim=1)(attn_mask)
        attn_mask = attn_mask.view(attn_mask.size(0), 1, x.size(2), x.size(3)) 
        x_attn = x * attn_mask
        x = x + x_attn
        return x, attn_mask

class EncoderCNN(nn.Module):
    def __init__(self, out_size=300, pretrained=True, attention=True):
        super(EncoderCNN, self).__init__()        
        # Embedd images in a 1000 space
        vgg_aux = models.vgg16_bn(pretrained)
        self.cnn_features = vgg_aux.features

        # Attention model
        self.attention = attention

        self.attn = AttentionModel()
        
        self.map = vgg_aux.classifier
        self.map._modules['6'] = nn.Linear(4096, out_size)

    def forward(self, im):
        x = self.cnn_features(im) # Size (BS x 512 x 7 x 7)

        if self.attention:
            x, attn_mask = self.attn(x) # Size (BS x 512)
        
        x = x.view(x.size(0), -1)

        x = self.map(x)
        if not self.attention:
            attn_mask = torch.zeros(7,7)
        return x, attn_mask


def normalize(x):
    return x / x.norm(dim=1, keepdim=True)
