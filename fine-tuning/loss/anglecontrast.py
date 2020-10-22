#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import time, pdb, numpy
from accuracy import accuracy

class AngleContrastiveLoss(nn.Module):

    def __init__(self, init_w=10.0, init_b=-5.0):
        super(AngleContrastiveLoss, self).__init__()
        self.w = nn.Parameter(torch.tensor(init_w))
        self.b = nn.Parameter(torch.tensor(init_b))
        self.criterion  = torch.nn.CrossEntropyLoss()

        print('Initialised AngleContrastive')

    def forward(self, x, label=None):

        seg1      = x[:,1,:]
        seg2      = x[:,0,:]
        K  = seg1.size()[0]

        cos_sim_matrix  = F.cosine_similarity(seg1.unsqueeze(-1), seg2.unsqueeze(-1).transpose(0,2))
        torch.clamp(self.w, 1e-6)
        cos_sim_matrix1 = cos_sim_matrix * self.w + self.b
        cos_sim_matrix2 = cos_sim_matrix.transpose(0,1) * self.w + self.b

        label   = torch.from_numpy(numpy.asarray(range(0, K))).cuda()
        nloss1  = self.criterion(cos_sim_matrix1, label)
        nloss2  = self.criterion(cos_sim_matrix2, label)
        nloss   = (nloss1 + nloss2)/2

        prec1, _    = accuracy(cos_sim_matrix1.detach().cpu(), label.detach().cpu(), topk=(1, 5))
        prec2, _    = accuracy(cos_sim_matrix2.detach().cpu(), label.detach().cpu(), topk=(1, 5))
        prec = (prec1 + prec2)/2

        return nloss, prec

