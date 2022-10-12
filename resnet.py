import os
from turtle import down, forward
import numpy as np
import pandas as pd
import torch
from torch import nn
from torchvision.io import read_image

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels//4, kernel_size=1, stride=1)
        if downsample:
            self.conv2 = nn.Conv2d(out_channels//4, out_channels//4, kernel_size=3, stride=2, padding=1)
        else:
            self.conv2 = nn.Conv2d(out_channels//4, out_channels//4, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(out_channels//4, out_channels, kernel_size=1, stride=1)
        self.skip = nn.Sequential()

        if downsample or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2 if downsample else 1),
                nn.BatchNorm2d(out_channels)
            )

        self.batch_norm1 = nn.BatchNorm2d(out_channels//4)
        self.batch_norm2 = nn.BatchNorm2d(out_channels//4)
        self.batch_norm3 = nn.BatchNorm2d(out_channels)

    def forward(self, X):
        skip = self.skip(X)
        relu = nn.ReLU()
        X = relu(self.batch_norm1(self.conv1(X)))
        X = relu(self.batch_norm2(self.conv2(X)))
        X = relu(self.batch_norm3(self.conv3(X)))
        X += skip
        return relu(X)

class ResNet(nn.Module):
    def __init__(self, version, in_channels, output_size=9):
        super().__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        assert version in [50, 101, 152]

        repeats_by_version = {
            50: [3, 4, 6, 3],
            101: [3, 4, 23, 3],
            152: [3, 8, 36, 3],
        }

        repeat = repeats_by_version[version]

        filters = [64, 256, 512, 1024, 2048]
        self.layer1 = nn.Sequential()
        self.layer1.add_module('conv2_1', ResidualBlock(filters[0], filters[1], downsample=False))
        for i in range(1, repeat[0]):
                self.layer1.add_module('conv2_%d'%(i+1,), ResidualBlock(filters[1], filters[1], downsample=False))

        self.layer2 = nn.Sequential()
        self.layer2.add_module('conv3_1', ResidualBlock(filters[1], filters[2], downsample=True))
        for i in range(1, repeat[1]):
                self.layer2.add_module('conv3_%d' % (i+1,), ResidualBlock(filters[2], filters[2], downsample=False))

        self.layer3 = nn.Sequential()
        self.layer3.add_module('conv4_1', ResidualBlock(filters[2], filters[3], downsample=True))
        for i in range(1, repeat[2]):
            self.layer3.add_module('conv2_%d' % (i+1,), ResidualBlock(filters[3], filters[3], downsample=False))

        self.layer4 = nn.Sequential()
        self.layer4.add_module('conv5_1', ResidualBlock(filters[3], filters[4], downsample=True))
        for i in range(1, repeat[3]):
            self.layer4.add_module('conv3_%d'%(i+1,), ResidualBlock(filters[4], filters[4], downsample=False))

        self.gap = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Linear(filters[4], output_size)

    def forward(self, X):
        X = self.layer0(X)
        X = self.layer1(X)
        X = self.layer2(X)
        X = self.layer3(X)
        X = self.layer4(X)
        X = self.gap(X)
        X = torch.flatten(X, start_dim=1)
        X = self.fc(X)
        return X