#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from models.ResNetBlocks import *

class TDNN(nn.Module):
    def __init__(self, block, layers, num_filters, nOut, encoder_type='SAP', **kwargs):

        print('Embedding size is %d, encoder %s.'%(nOut, encoder_type))
        
        self.inplanes = num_filters[0]
        self.encoder_type = encoder_type
        super(ResNetSE, self).__init__()

        self.conv1 = nn.Conv2d(1, num_filters[0] , kernel_size=7, stride=(2, 1), padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters[0])
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, num_filters[0], layers[0])
        self.layer2 = self._make_layer(block, num_filters[1], layers[1], stride=(2, 2))
        self.layer3 = self._make_layer(block, num_filters[2], layers[2], stride=(2, 2))
        self.layer4 = self._make_layer(block, num_filters[3], layers[3], stride=(1, 1))

        self.avgpool = nn.AvgPool2d((5, 1), stride=1)

        self.num_clust=8

        self.instancenorm   = nn.InstanceNorm1d(40)
        self.torchfb        = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, f_min=0.0, f_max=8000, pad=0, n_mels=40)

        if self.encoder_type == "SAP":
            self.sap_linear = nn.Linear(num_filters[3] * block.expansion, num_filters[3] * block.expansion)
            self.attention = self.new_parameter(num_filters[3] * block.expansion, 1)
            out_dim = num_filters[3] * block.expansion
        elif self.encoder_type=="VLAD":
            out_dim = num_filters[3] * block.expansion
            self.mu=torch.nn.Parameter(torch.randn(num_filters[3]*block.expansion, self.num_clust))
            self.sig=torch.nn.Parameter(torch.ones(1, num_filters[3]*block.expansion))
            self.w=torch.nn.Parameter(torch.ones(1, self.num_clust)/self.num_clust)
        elif self.encoder_type=="LDE":
            out_dim = num_filters[3] * block.expansion
            self.mu=torch.nn.Parameter(torch.randn(num_filters[3]*block.expansion, self.num_clust))
            self.sig=torch.nn.Parameter(torch.ones(1, self.num_clust))
            self.beta=torch.nn.Parameter(torch.randn(1, self.num_clust))
        elif self.encoder_type=="UDB":
            out_dim = num_filters[3] * block.expansion
            self.mu=torch.nn.Parameter(torch.randn(num_filters[3]*block.expansion, self.num_clust))
            self.sig=torch.nn.Parameter(torch.ones(1, self.num_clust))
            self.beta=torch.nn.Parameter(torch.randn(1, self.num_clust))
            self.udb_layer_1=nn.Linear(num_filters[3]*block.expansion*self.num_clust, 1024)
            self.udb_layer_2=nn.Linear(1024, 1024)
            self.udb_layer_3=nn.Linear(1024, num_filters[3]*block.expansion*self.num_clust)
        else:
            raise ValueError('Undefined encoder')

        if self.encoder_type=="VLAD" or self.encoder_type=="LDE" or self.encoder_type=="UDB":
            self.fc=nn.Linear(out_dim*self.num_clust, nOut)
        else:
            self.fc = nn.Linear(out_dim, nOut)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def new_parameter(self, *size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out

    def lde_response(self, x, mu, sig, beta):
        x_cent=x.unsqueeze(3)-mu.unsqueeze(0).unsqueeze(0)
        x_cent_norm=torch.norm(x_cent, 2)
        logits=-0.5*torch.mul(sig.unsqueeze(0), torch.pow(x_cent_norm, 2))+beta.unsqueeze(0)
        output=F.softmax(logits, dim=-1)
        return output

    def vlad_response(self, x, mu, sig, w):
        omega=torch.div(mu.unsqueeze(0), sig.unsqueeze(-1)+1e-8)
        phi=torch.log(F.relu(w)+1e-8)-0.5*torch.sum(torch.mul(omega, mu.unsqueeze(0)), 1)
        logits=torch.sum(torch.mul(omega.unsqueeze(1), x.unsqueeze(-1)), 2)+phi.unsqueeze(0)
        output=F.softmax(logits, dim=-1)
        return output

    def forward(self, x):

        x = self.torchfb(x)+1e-6
        x = self.instancenorm(x.log()).unsqueeze(1).detach()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)

        if self.encoder_type == "SAP":
            x = x.permute(0, 2, 1, 3)
            x = x.squeeze(dim=1).permute(0, 2, 1)  # batch * L * D
            h = torch.tanh(self.sap_linear(x))
            w = torch.matmul(h, self.attention).squeeze(dim=2)
            w = F.softmax(w, dim=1).view(x.size(0), x.size(1), 1)
            x = torch.sum(x * w, dim=1)
        elif self.encoder_type=="VLAD":
            x = x.permute(0, 2, 1, 3)
            x = x.squeeze(dim=1).permute(0, 2, 1)  # batch * L * D
            gamma_x=self.vlad_response(x, self.mu, self.sig, self.w)
            x=torch.sum(torch.mul(x.unsqueeze(3)-self.mu.unsqueeze(0).unsqueeze(0), gamma_x.unsqueeze(2)), 1)
            x=F.normalize(x, p=2, dim=1)
        elif self.encoder_type=="LDE":
            x = x.permute(0, 2, 1, 3)
            x = x.squeeze(dim=1).permute(0, 2, 1)  # batch * L * D
            gamma_x=self.lde_response(x, self.mu, self.sig, self.beta)
            n=torch.sum(gamma_x, 1)
            x=torch.div(torch.sum(torch.mul(x.unsqueeze(3), gamma_x.unsqueeze(2)), 1), n.unsqueeze(1)+1e-8)
            x=self.mu.unsqueeze(0)-x
        elif self.encoder_type=="UDB":
            x = x.permute(0, 2, 1, 3)
            x = x.squeeze(dim=1).permute(0, 2, 1)  # batch * L * D
            gamma_x=self.lde_response(x, self.mu, self.sig, self.beta) # [B, D, C]
            n=torch.sum(gamma_x, 1)
            x=torch.div(torch.sum(torch.mul(x.unsqueeze(3), gamma_x.unsqueeze(2)), 1), n.unsqueeze(1)+1e-8)
            x_udb=self.udb_layer_1(x.view(x.size()[0], -1))
            x_udb=self.relu(x_udb)
            x_udb=self.udb_layer_2(x_udb)
            x_udb=self.relu(x_udb)
            x_udb=self.udb_layer_3(x_udb).view(x.size()[0], -1, self.num_clust)
            alpha=torch.div(n, n+16).unsqueeze(1)
            x=torch.mul(alpha, x)-torch.mul(alpha, x_udb)-self.mu.unsqueeze(0)
#            x=x_udb
            
        x = x.view(x.size()[0], -1)
        x = self.fc(x)

        return x


def ResNetSE34L(nOut=256, **kwargs):
    # Number of filters
    num_filters = [16, 32, 64, 128]
    model = ResNetSE(SEBasicBlock, [3, 4, 6, 3], num_filters, nOut, **kwargs)
    return model
