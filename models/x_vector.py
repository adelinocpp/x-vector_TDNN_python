#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 30 19:59:45 2020

@author: krishna

"""


import torch.nn as nn
from models.tdnn import TDNN
import torch
import torch.nn.functional as F

class X_vector(nn.Module):
    def __init__(self, input_dim = 40, num_classes=8):
        super(X_vector, self).__init__()
        self.tdnn1 = TDNN(input_dim=input_dim, output_dim=512, context_size=5, dilation=1,dropout_p=0.5)
        self.tdnn2 = TDNN(input_dim=512, output_dim=512, context_size=3, dilation=1,dropout_p=0.5)
        self.tdnn3 = TDNN(input_dim=512, output_dim=512, context_size=2, dilation=2,dropout_p=0.5)
        self.tdnn4 = TDNN(input_dim=512, output_dim=512, context_size=1, dilation=1,dropout_p=0.5)
        self.tdnn5 = TDNN(input_dim=512, output_dim=512, context_size=1, dilation=3,dropout_p=0.5)
        #### Frame levelPooling
        self.segment6 = nn.Linear(1024, 512)
        self.segment7 = nn.Linear(512, 512)
        self.output = nn.Linear(512, num_classes)# <-LINHA ORIGINAL
        # self.output = nn.Linear(512, 1)
        self.softmax = nn.Softmax(dim=1)
    def forward(self, inputs):
        tdnn1_out = self.tdnn1(inputs)
        tdnn2_out = self.tdnn2(tdnn1_out)
        tdnn3_out = self.tdnn3(tdnn2_out)
        tdnn4_out = self.tdnn4(tdnn3_out)
        tdnn5_out = self.tdnn5(tdnn4_out)
        ### Stat Pool
        a,b,c = tdnn5_out.shape
        
        if (b > 1):
            mean = torch.mean(tdnn5_out,1)
            std = torch.var(tdnn5_out,1)
            stat_pooling = torch.cat((mean,std),1)
        else:
            base = tdnn5_out.squeeze(1)
            stat_pooling = torch.cat((base,torch.zeros(base.shape)),1)
        
        segment6_out = self.segment6(stat_pooling)
        x_vec = self.segment7(segment6_out)
        predictions = self.output(x_vec)
        # return predictions, x_vec  # <-LINHA ORIGINAL
        return x_vec, predictions