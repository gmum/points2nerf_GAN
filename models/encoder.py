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
