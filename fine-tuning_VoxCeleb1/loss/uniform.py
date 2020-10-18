#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import time, pdb, numpy
from accuracy import accuracy

class Uniformity(nn.Module):

    def __init__(self, uniform_t=2, sample_type='PoN'):
        super(Uniformity, self).__init__()
        self.t = uniform_t
        self.sample_type = sample_type
        print('Initialised Uniformity Loss, t=%.1f'%self.t)

    def forward(self, x, label=None):

        # Part of netgative pairs
        if self.sample_type == 'PoN':
            x1 = x[:,0,:]
            x2 = x[:,1,:]
            nloss1 = torch.pdist(x1, p=2).pow(2).mul(-self.t).exp().mean().log()
            nloss2 = torch.pdist(x2, p=2).pow(2).mul(-self.t).exp().mean().log()
            nloss = (nloss1 + nloss2) / 2

        # All positive and negtive pairs
        elif self.sample_type == 'APN':
            x = torch.reshape(x, (-1,x.size()[-1]))
            nloss = torch.pdist(x, p=2).pow(2).mul(-self.t).exp().mean().log()

        # All negative pairs
        elif self.sample_type == 'AN':
            x1 = x[:,0,:]
            x2 = x[:,1,:]
            K  = x.size()[0]

            nloss1 = torch.pdist(x1, p=2).pow(2).mul(-self.t).exp().sum()
            nloss2 = torch.pdist(x2, p=2).pow(2).mul(-self.t).exp().sum()
            nloss3 = F.pairwise_distance(x1.unsqueeze(-1), x2.unsqueeze(-1).transpose(0,2)).pow(2).mul(-self.t).exp()
            nloss3 = nloss3 * (1-torch.eye(K).cuda())
            nloss3 = nloss3.sum()
            nloss = torch.log(torch.div(nloss1 + nloss2 + nloss3,K*(K-1)/2 + K*(K-1)/2 + K*(K-1)))

        return nloss, 0
