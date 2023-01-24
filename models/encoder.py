#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: model.py
@Time: 2018/10/13 6:35 PM
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)  # (batch_size, num_points, k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2,
                    1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.z_size = config['z_size']
        self.use_bias = config['model']['E']['use_bias']
        self.relu_slope = config['model']['E']['relu_slope']
        self.dropout = config['model']['E']['dropout']
        self.embed = config['model']['E']['embedding']
        self.k = config['model']['E']['knn_k']

        # accordingly to original paper ~16 channels per group yields best results
        self.gn1 = nn.GroupNorm(4,64)
        self.gn2 = nn.GroupNorm(4,64)
        self.gn3 = nn.GroupNorm(8,128)
        self.gn4 = nn.GroupNorm(8,256)
        self.gn5 = nn.GroupNorm(self.embed//16,self.embed) # makes sure self.embed is divisible by 16

        self.conv1 = nn.Sequential(nn.Conv2d(12, 64, kernel_size=1, bias = self.use_bias),
                                   self.gn1,
                                   nn.LeakyReLU(negative_slope=self.relu_slope))
        self.conv2 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias = self.use_bias),
                                   self.gn2,
                                   nn.LeakyReLU(negative_slope=self.relu_slope))
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias = self.use_bias),
                                   self.gn3,
                                   nn.LeakyReLU(negative_slope=self.relu_slope))
        self.conv4 = nn.Sequential(nn.Conv2d(128 * 2, 256, kernel_size=1, bias = self.use_bias),
                                   self.gn4,
                                   nn.LeakyReLU(negative_slope=self.relu_slope))
        self.conv5 = nn.Sequential(nn.Conv1d(512, self.embed, kernel_size=1, bias = self.use_bias),
                                   self.gn5,
                                   nn.LeakyReLU(negative_slope=self.relu_slope))
        self.linear1 = nn.Linear(self.embed * 2, 1024, bias = self.use_bias)
        self.gn6 = nn.GroupNorm(32,1024)
        self.dp1 = nn.Dropout(p=self.dropout)
        self.linear2 = nn.Linear(1024, 1024)
        self.gn7 = nn.GroupNorm(32,1024)
        self.dp2 = nn.Dropout(p=self.dropout)
        self.linear3 = nn.Linear(1024, self.z_size)

    def forward(self, x):
        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv5(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        x = F.leaky_relu(self.gn6(self.linear1(x)), negative_slope=self.relu_slope)
        x = self.dp1(x)
        x = F.leaky_relu(self.gn7(self.linear2(x)), negative_slope=self.relu_slope)
        x = self.dp2(x)
        x = self.linear3(x)

        return x
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
def knn(x, k):

    batch_size = x.size(0)
    num_points = x.size(2)

    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    # print('idx 1',idx.shape,torch.isnan(torch.flatten(idx)).any())

    if idx.get_device() == -1:
        idx_base = torch.arange(0, batch_size).view(-1, 1, 1) * num_points
    else:
        idx_base = torch.arange(0, batch_size, device=idx.get_device()).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)
    # print('idx 2',idx.shape,torch.isnan(torch.flatten(idx)).any())

    return idx


def get_graph_feature(x, k=8, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)  # (batch_size, num_dims, num_points)
    if idx is None:
        idx = knn(x, k=k)  # (batch_size, num_points, k)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()  # (batch_size, num_points, num_dims)
    feature = x.view(batch_size * num_points, -1)[idx, :]  # (batch_size*n, num_dims) -> (batch_size*n*k, num_dims)
    feature = feature.view(batch_size, num_points, k, num_dims)  # (batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)  # (batch_size, num_points, k, num_dims)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1,
                                                         2).contiguous()  # (batch_size, num_points, k, 2*num_dims) -> (batch_size, 2*num_dims, num_points, k)

    return feature  # (batch_size, 2*num_dims, num_points, k)


class Encoder(nn.Module):
    def __init__(self,config,  k=8):
        super(Encoder, self).__init__()

        self.z_size = config['z_size']
        self.k = config['model']['E']['knn_k']
        self.use_bias = config['model']['E']['use_bias']
        self.relu_slope = config['model']['E']['relu_slope']

        feat_dim = 1024 # TODO: for now, try to reactivate conv6/7 (out after one more cat [1,512,2048] and [1,z_size,2048])

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(feat_dim)
        self.bnfc = nn.BatchNorm1d((feat_dim + self.z_size)//2)
        #self.bn6 = nn.BatchNorm1d(512)
        #self.bn7 = nn.BatchNorm1d(self.z_size)

        self.conv1 = nn.Sequential(nn.Conv2d(6 * 2, 64, kernel_size=1, bias = self.use_bias),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope = self.relu_slope))
        self.conv2 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias = self.use_bias),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope = self.relu_slope))
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias = self.use_bias),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope = self.relu_slope))
        self.conv4 = nn.Sequential(nn.Conv2d(128 * 2, 256, kernel_size=1, bias = self.use_bias),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope = self.relu_slope))
        self.conv5 = nn.Sequential(nn.Conv1d(512, feat_dim, kernel_size=1, bias = self.use_bias),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope = self.relu_slope))
        self.fc = nn.Sequential(
            nn.Linear(feat_dim, (feat_dim + self.z_size)//2, bias=True),
            nn.LeakyReLU(negative_slope=self.relu_slope),
            nn.Linear((feat_dim + self.z_size)//2, self.z_size, bias=True),
        )
        # self.conv6 = nn.Sequential(nn.Conv1d(feat_dim + 512,(feat_dim + 512) , kernel_size=1, bias = self.use_bias),
        #                            self.bn6,
        #                            nn.LeakyReLU(negative_slope = self.relu_slope))
        # self.conv7 = nn.Sequential(nn.Conv1d(512, self.z_size, kernel_size=1, bias = self.use_bias),
        #                            self.bn7,
        #                            nn.LeakyReLU(negative_slope = self.relu_slope))

    def forward(self, x):
        # x = x.transpose(2, 1).contiguous()

        batch_size, _, num_points = x.size()
        # print('batch_size {}, num_points {}'.format(batch_size,num_points))

        x = get_graph_feature(x, k=self.k)  # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        # print('ggf {} : x {} NaNs {}'.format(1,x.shape,torch.isnan(torch.flatten(x)).any()))
        x = self.conv1(x)  # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        # print('conv {} : x {} NaNs {}'.format(1,x.shape,torch.isnan(torch.flatten(x)).any()))
        x1 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        # print('max {} : x {} NaNs {}'.format(1,x1.shape,torch.isnan(torch.flatten(x1)).any()))

        x = get_graph_feature(x1, k=self.k)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        # print('ggf {} : x {} NaNs {}'.format(2,x.shape,torch.isnan(torch.flatten(x)).any()))
        x = self.conv2(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        # print('conv {} : x {} NaNs {}'.format(2,x.shape,torch.isnan(torch.flatten(x)).any()))
        x2 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        # print('max {} : x {} NaNs {}'.format(2,x2.shape,torch.isnan(torch.flatten(x2)).any()))

        x = get_graph_feature(x2, k=self.k)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        # print('ggf {} : x {} NaNs {}'.format(3,x.shape,torch.isnan(torch.flatten(x)).any()))
        x = self.conv3(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 128, num_points, k)
        # print('conv {} : x {} NaNs {}'.format(3,x.shape,torch.isnan(torch.flatten(x)).any()))
        x3 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)
        # print('max {} : x {} NaNs {}'.format(3,x3.shape,torch.isnan(torch.flatten(x3)).any()))

        x = get_graph_feature(x3, k=self.k)  # (batch_size, 128, num_points) -> (batch_size, 128*2, num_points, k)
        # print('ggf {} : x {} NaNs {}'.format(4,x.shape,torch.isnan(torch.flatten(x)).any()))
        x = self.conv4(x)  # (batch_size, 128*2, num_points, k) -> (batch_size, 256, num_points, k)
        # print('conv {} : x {} NaNs {}'.format(4,x.shape,torch.isnan(torch.flatten(x)).any()))
        x4 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 256, num_points, k) -> (batch_size, 256, num_points)
        # print('max {} : x {} NaNs {}'.format(4,x4.shape,torch.isnan(torch.flatten(x4)).any()))

        x = torch.cat((x1, x2, x3, x4), dim=1)  # (batch_size, 512, num_points)
        # print('cat {} : x {} NaNs {}'.format(1,x.shape,torch.isnan(torch.flatten(x)).any()))

        x = self.conv5(x)  # (batch_size, 512, num_points) -> (batch_size, feat_dims, num_points)
        # print('conv {} : x {} NaNs {}'.format(5,x.shape,torch.isnan(torch.flatten(x)).any()))
        x = x.max(dim=-1, keepdim=True)[0]  # (batch_size, feat_dims, num_points) -> (batch_size, feat_dims)
        # print('max {} : x {} NaNs {}'.format(5,x.shape,torch.isnan(torch.flatten(x)).any()))

        x = torch.squeeze(x,dim=-1)
        # # print('squeeze {} : x {} NaNs {}'.format(1,x.shape,torch.isnan(torch.flatten(x)).any()))
        x = self.fc(x)
        # print('fc {} : x {} NaNs {}'.format(1,x.shape,torch.isnan(torch.flatten(x)).any()))
        # x = x.repeat(1, 1, num_points)
        # # print('repeat {} : x {} NaNs {}'.format(1,x1.shape))
        # x = torch.cat((x, x1, x2, x3, x4), dim=1)
        # # print('cat {} : x {} NaNs {}'.format(2,x.shape))
        
        # x = self.conv6(x)
        # # print('conv {} : x {} NaNs {}'.format(6,x.shape))
        # x = self.conv7(x)
        # # print('conv {} : x {} NaNs {}'.format(7,x.shape))

        return x # .permute(0, 2, 1).contiguous()
        
"""
