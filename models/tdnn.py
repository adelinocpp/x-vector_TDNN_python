#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: cvqluu
repo: https://github.com/cvqluu/TDNN
"""

import torch.nn as nn
import torch.nn.functional as F

class TDNN(nn.Module):
    
    def __init__(
                    self, 
                    input_dim=23, 
                    output_dim=512,
                    context_size=5,
                    stride=1,
                    dilation=1,
                    batch_norm=True,
                    dropout_p=0.2
                ):
        '''
        TDNN as defined by https://www.danielpovey.com/files/2015_interspeech_multisplice.pdf

        Affine transformation not applied globally to all frames but smaller windows with local context

        batch_norm: True to include batch normalisation after the non linearity
        
        Context size and dilation determine the frames selected
        (although context size is not really defined in the traditional sense)
        For example:
            context size 5 and dilation 1 is equivalent to [-2,-1,0,1,2]
            context size 3 and dilation 2 is equivalent to [-2, 0, 2]
            context size 1 and dilation 1 is equivalent to [0]
        '''
        super(TDNN, self).__init__()
        self.context_size = context_size
        self.stride = stride
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dilation = dilation
        self.dropout_p = dropout_p
        self.batch_norm = batch_norm
        self.kernel = nn.Linear(input_dim*context_size, output_dim)
        self.nonlinearity = nn.ReLU()
        if self.batch_norm:
            self.bn = nn.BatchNorm1d(output_dim)
        if self.dropout_p:
            self.drop = nn.Dropout(p=self.dropout_p)
        
    def forward(self, x):
        
        '''
        input: size (batch, seq_len, input_features)
        outpu: size (batch, new_seq_len, output_features)
        '''
        # _, _, d = x.shape # <-LINHA ORIGINAL
        # print("x.shape {}".format(x.shape))
        useContextOne = False
        if (len(x.shape) == 4):
            # _, _, d, _ = x.shape
            a, b, d, e = x.shape
            if ((a ==1) and (b == 1) and (d == 1)):
                x = x
                useContextOne = True
            else:
                # print("X: len: {:}, shape: {:}".format(len(x.shape),x.shape))
                assert (d == self.input_dim), 'Input dimension was wrong. Expected ({:}), got ({:})'.format(self.input_dim, d)
                # x = x.unsqueeze(1) # <-LINHA ORIGINAL
                # Unfold input into smaller temporal contexts
                x = x.transpose(2,3)
        else:
            a, b, d = x.shape
            if ((a == 1) and (b == 1)):
                x = x
                useContextOne = True
            else:
                assert (d == self.input_dim), 'Input dimension was wrong. Expected ({:}), got ({:})'.format(self.input_dim, d)
            x = x.unsqueeze(1)

        # print("forward: len: {:}, shape {:}".format(len(x.shape),x.shape))
        if (useContextOne):
            x = F.unfold(x, (1, self.input_dim), stride=(1,self.input_dim), 
                        dilation=(self.dilation,1))
            x = x.transpose(1,2)
            KF = nn.Linear(self.input_dim, self.output_dim)
            x = KF(x.float())
            x = self.nonlinearity(x)
            
            if self.dropout_p:
                x = self.drop(x)
    
            if self.batch_norm:
                x = x.transpose(1,2)
                x = self.bn(x)
                x = x.transpose(1,2)
            
        else:
            x = F.unfold(x, (self.context_size, self.input_dim), 
                        stride=(1,self.input_dim), dilation=(self.dilation,1))
            # N, output_dim*context_size, new_t = x.shape
            x = x.transpose(1,2)
            x = self.kernel(x.float())
            x = self.nonlinearity(x)
            
            if self.dropout_p:
                x = self.drop(x)
    
            if self.batch_norm:
                x = x.transpose(1,2)
                x = self.bn(x)
                x = x.transpose(1,2)

        return x
