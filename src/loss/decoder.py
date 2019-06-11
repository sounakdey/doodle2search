# -*- coding: utf-8 -*- 
from __future__ import print_function, division

"""
Metric models
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from .layers import Gconv, Wcompute 


class MetricNN(nn.Module):
    """
    Based on https://github.com/vgsatorras/few-shot-gnn/
    """
    def __init__(self, input_size=300, hidden_size=512, out_size=1000, nlayers=3):
        super(MetricNN, self).__init__()

        self.input_features = input_size
        self.out_size = out_size
        self.nlayers = nlayers
        self.hidden_size = hidden_size

        self.wcomp_start = Wcompute(self.input_features, self.input_features)
        self.gconv_start = Gconv(self.input_features, self.hidden_size)
        for i in range(1, self.nlayers-1): # Last layer will be controlled separatedly
            # Compute W
            module_w = Wcompute(self.hidden_size, self.hidden_size)
            self.add_module('wcomp_l{}'.format(i), module_w)
            # Graph Convolution
            module_l = Gconv(self.hidden_size, self.hidden_size) # Hidden size should change
            self.add_module('gconv_l{}'.format(i), module_l)
        # Last W
        self.wlayer_last = Wcompute(self.hidden_size, self.hidden_size, activation='softmax')
        self.gconv_last = Gconv(self.hidden_size, out_size, bn_bool=False)
    
    def forward(self, sk, im):
        x = torch.cat((im, sk.unsqueeze(1)), dim=1) 
        # Initialize W as identity matrix
        W_init = Variable(torch.eye(x.size(1)).unsqueeze(0).repeat(x.size(0), 1, 1).unsqueeze(3))
        if x.is_cuda:
            W_init = W_init.cuda()

	# Compute W 
        Wi = self.wcomp_start(x, W_init)
	# Graph Convolution
        x = F.leaky_relu(self.gconv_start(Wi, x))
        for i in range(1, self.nlayers-1):
            # Compute W
            Wi = self._modules['wcomp_l{}'.format(i)](x, W_init)
            # Graph Convolution
            x_new = F.leaky_relu(self._modules['gconv_l{}'.format(i)](Wi, x))
            # Combine x and x_new
            x = x + x_new # Bruna propose to concat
        # Compute last W
        W = self.wlayer_last(x, W_init)
        # Last Graph Convolution
        out = self.gconv_last(W, x)
        out_im = out[:,:-1,:]
        out_sk = out[:,-1,:]
        return out_sk, out_im


class LinearNN(nn.Module):
    """
    Similar structure than LinearNN but without taking into account other images 
    """
    def __init__(self, input_size=300, hidden_size=512, out_size=1000, nlayers=2):
        super(LinearNN, self).__init__()

        self.input_features = input_size
        self.out_size = out_size
        self.nlayers = nlayers
        self.hidden_size = hidden_size

        self.l_start = nn.Linear(self.input_features, self.hidden_size, bias = False)
        self.bn_start = nn.BatchNorm1d(self.hidden_size)
        for i in range(1, self.nlayers-1): # Last layer will be controlled separatedly
            # Linear Layer 
            module_l = nn.Linear(self.hidden_size, self.hidden_size, bias=False) # Hidden size should change
            self.add_module('linear_l{}'.format(i), module_l)
            
            module_bn = nn.BatchNorm1d(self.hidden_size)
            self.add_module('bn_l{}'.format(i), module_bn)
        # Last W
        self.l_last = nn.Linear(self.hidden_size, out_size)
    
    def forward(self, sk, im):
        x = torch.cat((im, sk.unsqueeze(1)), dim=1)
        x_size = x.shape
        x = x.view(-1, x.size(-1))
        x = self.l_start(x)
        x = F.leaky_relu(self.bn_start(x))
        for i in range(1, self.nlayers-1):
            # Linear Layer 
            x_new = self._modules['linear_l{}'.format(i)](x)
            x_new = F.leaky_relu(self._modules['bn_l{}'.format(i)](x_new))
            # Combine x and x_new
            x = x + x_new # Bruna propose to concat
        # Linear Layer
        out = self.l_last(x)
        # out = out.view(x_size[:-1], -1)
        out = out.view(x_size[0], x_size[1], out.size(-1))
        out_im = out[:,:-1,:]
        out_sk = out[:,-1,:]
        return out_sk, out_im


class GraphNN(nn.Module):
    """
    Based on https://github.com/vgsatorras/few-shot-gnn/
    """
    def __init__(self, input_size=300, hidden_size=512, out_size=1000, nlayers=3):
        super(GraphNN, self).__init__()

        self.input_features = input_size
        self.out_size = out_size
        self.nlayers = nlayers
        self.hidden_size = hidden_size

        self.gconv_start = Gconv(self.input_features, self.hidden_size)
        for i in range(1, self.nlayers-1): # Last layer will be controlled separatedly
            # Graph Convolution
            module_l = Gconv(self.hidden_size, self.hidden_size) # Hidden size should change
            self.add_module('gconv_l{}'.format(i), module_l)
        self.gconv_last = Gconv(self.hidden_size, out_size, bn_bool=False)
    
    def forward(self, sk, im, Wi):
        x = torch.cat((im, sk.unsqueeze(1)), dim=1) 
        # Initialize W as identity matrix
        W_init = Variable(torch.eye(x.size(1)).unsqueeze(0).repeat(x.size(0), 1, 1).unsqueeze(3))
        if x.is_cuda:
            W_init = W_init.cuda()
        Wi = torch.cat([W_init, Wi.unsqueeze(-1)], 3) # size: bs x N x N x 2 (identity and features)

	# Graph Convolution
        x = F.leaky_relu(self.gconv_start(Wi, x))
        for i in range(1, self.nlayers-1):
            # Graph Convolution
            x_new = F.leaky_relu(self._modules['gconv_l{}'.format(i)](Wi, x))
            # Combine x and x_new
            x = x + x_new # Bruna propose to concat
        # Last Graph Convolution
        out = self.gconv_last(Wi, x)
        out_im = out[:,:-1,:]
        out_sk = out[:,-1,:]
        return out_sk, out_im
