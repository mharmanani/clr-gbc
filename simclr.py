import os
from re import M
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.io import read_image
from torchvision.models.resnet import resnet50, resnet101
from torch.cuda.amp import GradScaler, autocast

class SimCLR():
    def __init__(self, model_backbone, batch_size, num_epochs, num_views=2, temperature=0.07, device='cuda'):
        self.model = model_backbone
        self.optim = torch.optim.Adam(self.model.parameters(), )
        self.criterion = nn.CrossEntropyLoss()
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.views = num_views # number of augmented view of one sample
        self.temperature = temperature
        self.device = device 

    def infoNCELoss(self, X):
        labels = torch.eye(self.batch_size * self.views) # give a label to each view of each image
        X = F.normalize(X, dim=1) # apply normalization to the extracted features
        similarity_matrix = X@X.T

        mask = torch.eye(labels.shape[0], dtype=torch.bool)#.to(self.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long)#.to(self.device)

        logits = logits / self.temperature
        return logits, labels

    def train(self, train_loader, val_loader):
        i = 0

        for epoch in range(self.num_epochs):
            for X, _ in train_loader:
                print("X.shape=", X.shape)
                X = torch.cat((X, ) * self.views, dim=0)
                print("torch.cat(X).shape=", X.shape)
                #X = X.to(self.device)
                Y = self.model(X)
                print("Y.shape", Y.shape)
                Z, T = self.infoNCELoss(Y)
                
                loss = self.criterion(Z, T)
                self.optim.zero_grad()
                loss.backward()

                if i % 100 == 0:
                    # TODO: Add + log performance metrics
                    print('[iter {0}] Train loss={1}'.format(i, loss))
                    #print('[iter {0}] Validation loss={2}'.format(i, loss)) # TODO: update this

                i += 1
            
            checkpt_name = 'simclr_checkpt_{0}.pth'.format()
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.model.state_dict(),
                'loss': self.loss
            }, checkpt_name)

    def test(self, test_loader):
        pass






        