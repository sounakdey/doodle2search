# -*- coding: utf-8 -*- 
from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd=0.5):
        # ctx is a context object that can be used to stash information
        # for backward computation
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # We return as many input gradients as there were arguments.
        # Gradients of non-Tensor arguments to forward must be None.
        return ctx.lambd * grad_output.neg(), None


def grad_reverse(x, lambd=0.5):
    return GradReverse.apply(x, lambd)


def gmul(W, x):
    # x is a tensor of size (bs, N, num_features). Node signal. |V|=N
    # W is a tensor of size (bs, N, N, J). Local graph operators (J powers of adjacency matrix)
    x_size = x.size()
    W_size = W.size()
    N = W_size[-2] 
    W = W.split(1, 3) # View will not join the desired dimensions
    W = torch.cat(W, 1).squeeze(3) # W is now a tensor of size (bs, J*N, N) 
    output = torch.bmm(W, x) # output has size (bs, J*N, num_features)  
    output = output.split(N, 1)      
    output = torch.cat(output, 2) # output has size (bs, N, J*num_features)
    output = output.contiguous()   
    return output


class Gconv(nn.Module):
    def __init__(self, nf_input, nf_output, bn_bool=True, J=2):
        super(Gconv, self).__init__() 
        self.num_inputs = J*nf_input # After gmul (W*x)
        self.num_outputs = nf_output

        self.bn_bool = bn_bool
        if self.bn_bool:
            self.fc = nn.Linear(self.num_inputs, self.num_outputs, bias=False)
            self.bn = nn.BatchNorm1d(self.num_outputs)
        else:
            self.fc = nn.Linear(self.num_inputs, self.num_outputs) 

    def forward(self, W, x):
        x = gmul(W, x) # out has size (bs, N, num_inputs) 
        x_size = x.size()   

        x = x.view(-1, self.num_inputs)
        x = self.fc(x) # has size (bs*N, num_outputs) 

        if self.bn_bool:
            x = self.bn(x)
        x = x.view(x_size[0], x_size[1], self.num_outputs) # has size (bs, N, num_outputs)

        return x 


class Wcompute(nn.Module):
    def __init__(self, input_features, nf, operator='J2', activation='softmax', ratio=[2,2,1,1], num_operators=1, drop=False):
        super(Wcompute, self).__init__()
        self.num_features = nf
        self.operator = operator
        self.conv2d_1 = nn.Conv2d(input_features, int(nf * ratio[0]), 1, stride=1)
        self.bn_1 = nn.BatchNorm2d(int(nf * ratio[0]))
        self.drop = drop
        if self.drop:
            self.dropout = nn.Dropout(0.3)
        self.conv2d_2 = nn.Conv2d(int(nf * ratio[0]), int(nf * ratio[1]), 1, stride=1)
        self.bn_2 = nn.BatchNorm2d(int(nf * ratio[1]))
        self.conv2d_3 = nn.Conv2d(int(nf * ratio[1]), nf*ratio[2], 1, stride=1)
        self.bn_3 = nn.BatchNorm2d(nf*ratio[2])
        self.conv2d_4 = nn.Conv2d(nf*ratio[2], nf*ratio[3], 1, stride=1)
        self.bn_4 = nn.BatchNorm2d(nf*ratio[3])
        self.conv2d_last = nn.Conv2d(nf, num_operators, 1, stride=1)
        self.activation = activation

    def forward(self, x, W_id):
        W1 = x.unsqueeze(2)
        W2 = torch.transpose(W1, 1, 2) #size: bs x N x N x num_features
        W_new = torch.abs(W1 - W2) #size: bs x N x N x num_features
        W_new = torch.transpose(W_new, 1, 3) #size: bs x num_features x N x N

        W_new = self.conv2d_1(W_new)
        W_new = self.bn_1(W_new)
        W_new = F.leaky_relu(W_new)
        if self.drop:
            W_new = self.dropout(W_new)

        W_new = self.conv2d_2(W_new)
        W_new = self.bn_2(W_new)
        W_new = F.leaky_relu(W_new)

        W_new = self.conv2d_3(W_new)
        W_new = self.bn_3(W_new)
        W_new = F.leaky_relu(W_new)

        W_new = self.conv2d_4(W_new)
        W_new = self.bn_4(W_new)
        W_new = F.leaky_relu(W_new)

        W_new = self.conv2d_last(W_new)
        W_new = torch.transpose(W_new, 1, 3) #size: bs x N x N x 1

        if self.activation == 'softmax':
            W_new = W_new - W_id.expand_as(W_new) * 1e8
            W_new = torch.transpose(W_new, 2, 3)
            # Applying Softmax
            W_new = W_new.contiguous()
            W_new_size = W_new.size()
            W_new = W_new.view(-1, W_new.size(3))
            W_new = F.softmax(W_new, dim=1)
            W_new = W_new.view(W_new_size)
            # Softmax applied
            W_new = torch.transpose(W_new, 2, 3)

        elif self.activation == 'sigmoid':
            W_new = F.sigmoid(W_new)
            W_new *= (1 - W_id)
        elif self.activation == 'none':
            W_new *= (1 - W_id)
        else:
            raise (NotImplementedError)

        if self.operator == 'laplace':
            W_new = W_id - W_new
        elif self.operator == 'J2':
            W_new = torch.cat([W_id, W_new], 3) # size: bs x N x N x 2 (identity and features)
        else:
            raise(NotImplementedError)

        return W_new

