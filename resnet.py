import os
import numpy as np
import pandas as pd
import torch
from torch import nn
from torchvision.io import read_image

from utils import map_labels_to_int

class ResNetWrapper(nn.Module):
    def __init__(self, resnet, num_classes=2, device='cuda'):
        super().__init__()
        self.resnet = resnet
        self.clf = nn.Sequential(
            self.resnet,
            nn.Linear(1000, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
            nn.Softmax(dim=1)
        )

        self.device = device

    def forward(self, X):
        return self.clf(X)

    def finetune(self, train_loader, val_loader, learning_rate=1e-3, num_epochs=30):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.clf.parameters(), lr=learning_rate)

        # Arrays and variables for plotting metrics
        train_acc, val_acc = 0, 0
        epochs_arr = []
        train_losses = []
        train_accs = []
        val_losses = []
        val_accs = []

        for epoch in range(1+num_epochs):
            self.clf.train()
            for batch in train_loader:
                X, y = batch
                y = map_labels_to_int(y, dtype='long')
                X = X.to(self.device)
                y = y.to(self.device)
                optimizer.zero_grad()
                z = self.clf(X)
                train_loss = criterion(z, y)
                train_loss.backward()
                optimizer.step()

                train_total = train_correct = 0
                for idx, i in enumerate(z):
                    if torch.argmax(i) == y[idx]:
                        train_correct +=1
                    train_total +=1
                train_acc = train_correct / train_total

            train_losses.append(train_loss.to('cpu').detach().numpy())
            train_accs.append(train_acc)

            self.clf.eval()
            with torch.no_grad():
                for batch in val_loader:
                    X, y = batch
                    y = map_labels_to_int(y, dtype='long')
                    X = X.to(self.device)
                    y = y.to(self.device)
                    z = self.clf(X)
                    val_loss = criterion(z, y)
                    
                    val_total = val_correct = 0
                    for idx, i in enumerate(z):
                        if torch.argmax(i) == y[idx]:
                            val_correct +=1
                        val_total +=1
                    val_acc = val_correct / val_total

                val_losses.append(val_loss.to('cpu').to('cpu').detach().numpy())
                val_accs.append(val_acc)

            epochs_arr.append(epoch)

            print('[Epoch {0}] [train loss={1}] [train acc={2}] [val loss={3}] [val acc={4}]'
                .format(epoch, train_loss, round(train_acc, 3), val_loss, round(val_acc, 3)))

    def test(self, test_loader):
        self.clf.eval()
        with torch.no_grad():
            total = correct = 0
            for batch in test_loader:
                X, y = batch
                y = map_labels_to_int(y, dtype='long')
                X = X.to(self.device)
                y = y.to(self.device)
                z = self.clf(X)
                print(torch.argmax(z, axis=1))
                print(y)
                for idx, i in enumerate(z):
                    if torch.argmax(i) == y[idx]:
                        correct +=1
                    total +=1
            test_acc = correct / total

        return test_acc

        

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
    
class SupConResNet(nn.Module):
    """backbone + projection head"""
    def __init__(self, name='resnet50', head='mlp', feat_dim=128):
        super(SupConResNet, self).__init__()
        model_fun, dim_in = model_dict[name]
        self.encoder = model_fun()
        if head == 'linear':
            self.head = nn.Linear(dim_in, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))

    def forward(self, x):
        feat = self.encoder(x)
        feat = F.normalize(self.head(feat), dim=1)
        return feat
