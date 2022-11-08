from operator import concat
import os
from re import M
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch import nn
from torchvision.io import read_image
from torchvision.models.resnet import resnet50, resnet101
from torchvision import transforms
from torch.cuda.amp import GradScaler, autocast

import time

import metrics

class SimCLR():
    def __init__(self, model_backbone, batch_size, num_epochs, num_views=2, project_dim=64,
                 temperature=0.5, device='cuda'):
        self.device = device 
        self.model = model_backbone.to(self.device)
        self.optim = torch.optim.Adam(self.model.parameters(), )
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.views = num_views # number of augmented view of one sample
        self.temperature = temperature
        self.projection_head = nn.Sequential(
            nn.Linear(1000, 1000, bias=False),
            nn.ReLU(),
            nn.Linear(1000, project_dim, bias=False),
        ).to(self.device)
    
    def forward(self, x_i, x_j):
        h_i = self.model(x_i)
        h_j = self.model(x_j)

        z_i = self.projection_head(h_i)
        z_j = self.projection_head(h_j)
        return h_i, h_j, z_i, z_j

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def NT_Xent(self, z_i, z_j):
        N = 2 * self.batch_size
        mask = self.mask_correlated_samples(self.batch_size)

        z = torch.cat((z_i, z_j), dim=0)
        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature
        sim_i_j = torch.diag(sim.long(), self.batch_size)
        sim_j_i = torch.diag(sim.long(), -self.batch_size)

        concated = torch.cat((sim_i_j, sim_j_i), dim=0)
        if concated.shape[0] < N:
            mask = self.mask_correlated_samples(sim.shape[0] // 2)
            N = concated.shape[0]

        positive_samples = concated.reshape(N, 1)
        negative_samples = sim[mask].reshape(N, -1)    
        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        
        return loss

    def infoNCELoss(self, X):
        X = F.normalize(X, dim=1) # apply normalization to the extracted features
        similarity_matrix = X@X.T

        #labels = torch.eye(self.batch_size * self.views) # give a label to each view of each image
        labels = torch.eye(X.shape[0]) # give a label to each view of each image

        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        #print("X.shape", X.shape)
        #print("SimM.shape", similarity_matrix.shape)
        #print("labels.shape", labels.shape)
        #print("mask.shape", mask.shape)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)

        logits = logits / self.temperature
        return logits, labels

    def train(self, train_loader, val_loader):
        for epoch in range(self.num_epochs):
            epoch_start_time = time.time()
            self.model.train()
            for (x_i, x_j), _ in train_loader:
                x_i, x_j = x_i.to(self.device), x_j.to(self.device)
                self.optim.zero_grad()
                h_i, h_j, z_i, z_j = self.forward(x_i, x_j)
                loss = self.NT_Xent(z_i, z_j)
                loss.backward()
                self.optim.step()

            self.model.eval()
            with torch.no_grad():
                for (x_vi, x_vj), _ in val_loader:
                    x_vi, x_vj = x_vi.to(self.device), x_vj.to(self.device)
                    h_vi, h_vj, z_vi, z_vj = self.forward(x_vi, x_vj)
                    #valid_loss = self.NT_Xent(z_vi, z_vj)

            epoch_end_time = time.time()
            epoch_dur = roud(epoch_end_time - epoch_start_time, 2)
            print('[epoch {0}] [train loss={1}] [valid loss={2}] [time elapsed={3}]'.format(epoch, loss, -42, epoch_dur))

            checkpt_name = 'simclr_checkpt_{0}.pth'.format(epoch)
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.model.state_dict(),
                'loss': loss
            }, checkpt_name)

    def test(self, test_loader):
        pass






        