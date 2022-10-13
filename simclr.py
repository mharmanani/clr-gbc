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
    def __init__(self, model_backbone, batch_size, num_epochs, num_views=2, device='cuda'):
        self.model = model_backbone
        self.optim = torch.optim.Adam(self.model.parameters(), )
        self.criterion = nn.CrossEntropyLoss()
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.views = num_views # number of augmented view of one sample
        self.device = device 

    def infoNCELoss(self, X):
        labels = torch.cat([torch.arange(self.batch_size) for _ in range(self.views)])
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()

        X = F.normalize(X, dim=1)
        similarity_matrix = X@X.T

        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)

        logits = logits / self.args.temperature
        return logits, labels

    def train(self, train_loader, val_loader):
        i = 0

        for epoch in range(self.num_epochs):
            for X, _ in train_loader:
                X = torch.cat(X, dim=0)
                X = X.to(self.device)
                Y = self.model(X)
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






        