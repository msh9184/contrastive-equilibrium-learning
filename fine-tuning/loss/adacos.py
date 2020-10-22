#! /usr/bin/python
# -*- encoding: utf-8 -*-
# Adapted from https://github.com/wujiyang/Face_Pytorch (Apache License)

import torch
import torch.nn as nn
import torch.nn.functional as F
import time, pdb, numpy, math
from accuracy import accuracy
from torch.nn import Parameter
import math

class ADASoftmax(nn.Module):
    def __init__(self, 
                 in_feats, 
                 n_classes, 
                 m=0.50):
        super(ADASoftmax, self).__init__()
        self.in_feats = in_feats
        self.n_classes = n_classes
        self.s = math.sqrt(2) * math.log(n_classes - 1)
        self.m = m
        self.W = Parameter(torch.FloatTensor(n_classes, in_feats))
        self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_uniform_(self.W)

    def forward(self, input, label=None):
        # normalize features
        x = F.normalize(input)
        # normalize weights
        W = F.normalize(self.W)
        # dot product
        logits = F.linear(x, W)
        if label is None:
            return logits
        # feature re-scale
        theta = torch.acos(torch.clamp(logits, -1.0 + 1e-7, 1.0 - 1e-7))
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        with torch.no_grad():
            B_avg = torch.where(one_hot < 1, torch.exp(self.s * logits), torch.zeros_like(logits))
            B_avg = torch.sum(B_avg) / input.size(0)
            # print(B_avg)
            theta_med = torch.median(theta[one_hot == 1])
            self.s = torch.log(B_avg) / torch.cos(torch.min(math.pi/4 * torch.ones_like(theta_med), theta_med))
        output = self.s * logits

        loss = self.ce(output, label)
        prec1, _    = accuracy(output.detach().cpu(), label.detach().cpu(), topk=(1, 5))
        return loss, prec1

