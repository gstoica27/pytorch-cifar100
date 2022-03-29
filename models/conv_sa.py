from __future__ import print_function
import argparse
from calendar import c
import random
import pdb
import math
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import numpy as np
from tqdm import tqdm
from positional_encodings import PositionalEncoding2D
import torchvision

class SummarizedConvolutionalSelfAttention(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, instructions=None, baseplate=None):
        super().__init__()

        self.C = in_channels
        self.out_C = out_channels
        self.filter_H = self.filter_W = kernel_size
        self.instructions = instructions

        self.padding = padding
        self.stride = stride
        self.bias = bias
        
        self.add_shortcut = instructions['add_residual']
        self.value_dim = instructions['value_dim']
        self.num_heads = instructions['num_heads']
        self.dropout = instructions['dropout']
        self.add_kv_bias = instructions['add_kv_bias']

        if instructions['add_positional_encodings']:
            self.get_positional_encodings = PositionalEncoding2D(self.C)

        if baseplate is not None:
            self.baseplate = baseplate
        else:
            self.baseplate = nn.Parameter(
                torch.rand((self.filter_H, self.filter_W, self.C), requires_grad=True)
            )
        self.msa = nn.MultiheadAttention(
            embed_dim=self.C, num_heads=self.num_heads, 
            dropout=self.dropout, add_bias_kv=self.add_kv_bias,
            batch_first=True
        )
        self.value_reducer = nn.Linear(self.C, self.value_dim)
        self.filter_generator = nn.Linear(self.value_dim, self.C * self.out_C)
        if self.bias:
            self.bias_generator = nn.Linear(self.value_dim, self.out_C)
    
    def forward(self, batch):
        B, C, H, W = batch.shape
        # if B > 1: pdb.set_trace()
        position_encodings = self.get_positional_encodings(batch.permute(0, 2, 3, 1)).flatten(1,2)
        flat_batch = batch.flatten(2).permute(0, 2, 1)
        flat_batch_pe = flat_batch + position_encodings

        batch_summaries = self.msa(
            query=self.baseplate.flatten(0,1).unsqueeze(0).repeat(B, 1, 1), 
            key=flat_batch_pe,
            value=flat_batch_pe
        )[0]
        
        batch_summaries = batch_summaries.reshape(
            B, self.filter_H, self.filter_W, self.C
        )
        
        reduced_summaries = self.value_reducer(batch_summaries)

        filter_weights = self.filter_generator(reduced_summaries).\
            reshape(B, self.filter_H, self.filter_W, self.out_C, self.C).\
                permute(0, 3, 4, 1, 2).\
                    flatten(0, 1)
        if self.bias:
            filter_bias = self.bias_generator(reduced_summaries).\
                reshape(B, self.filter_H, self.filter_W,self.out_C).\
                    flatten(1,2).\
                        mean(1).flatten(0)
        else:
            filter_bias = None
        # pdb.set_trace()
        convolution_output = F.conv2d(
            input=batch.unsqueeze(0).flatten(1,2),
            weight=filter_weights,
            bias=filter_bias,
            stride=self.stride,
            padding=self.padding,
            groups=B
        )
        _, BC, oH, oW = convolution_output.shape
        batched_output = convolution_output.reshape(B, self.out_C, oH, oW)
        return batched_output
        

