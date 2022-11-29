from operator import concat
import os
from re import M
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
import time
from torch import nn
from torchvision.io import read_image
from torchvision.models.resnet import resnet18, resnet50, resnet101
from torchvision import transforms
from torch.cuda.amp import GradScaler, autocast
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
import torchmetrics

from utils import map_labels_to_int
from simclr import SimCLR

class SupCon(SimCLR):
    def __init__(self, model_backbone, batch_size, num_epochs, num_views=2, project_dim=64,
                 num_classes=2, learning_rate=3e-4, temperature=0.07, base_temperature=0.07, device='cuda'):
        self.device = device 
        self.backbone = model_backbone.to(self.device)
        
        self.projection_head = nn.Sequential(
            nn.Linear(1000, 128),
            nn.ReLU(),
            nn.Linear(128, project_dim),
            nn.ReLU()
        ).to(self.device)

        self.model = nn.Sequential(
            self.backbone,
            self.projection_head
        ).to(self.device)

        self.clf_head = nn.Sequential(
            nn.Linear(1000, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
            nn.Softmax(dim=1)
        ).to(self.device)
        
        self.learning_rate = learning_rate
        self.optim = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.views = num_views # number of augmented view of one sample
        self.temperature = temperature
        self.base_temperature = base_temperature
    
    def forward(self, x_i, x_j):
        h_i = self.backbone(x_i)
        h_j = self.backbone(x_j)

        z_i = self.projection_head(h_i)
        z_j = self.projection_head(h_j)
        return h_i, h_j, z_i, z_j

    def supcon_loss(self, x_i, x_j, y):
        X = torch.cat([x_i.unsqueeze(1), x_j.unsqueeze(1)], dim=1)
        y = y.contiguous().view(-1, 1)
        batch_size = self.batch_size

        if y.shape[0] != self.batch_size:
            #raise ValueError('Num of labels does not match num of features')
            batch_size = y.shape[0]
        
        mask = torch.eq(y, y.T).float().to(self.device)

        contrast_count = X.shape[1]
        contrast_feature = torch.cat(torch.unbind(X, dim=1), dim=0)
        anchor_feature = contrast_feature
        anchor_count = contrast_count

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(self.device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

    def train(self, train_loader, val_loader):
        for epoch in range(self.num_epochs):
            epoch_start_time = time.time()
            self.model.train()
            num_errs = 0
            train_losses = []
            val_losses = []
            for (x_i, x_j), y in train_loader:
                try:
                    x_i, x_j = x_i.to(self.device), x_j.to(self.device)
                    y = map_labels_to_int(y)
                    y = y.to(self.device)
                    self.optim.zero_grad()
                    h_i, h_j, z_i, z_j = self.forward(x_i, x_j)
                    loss = self.supcon_loss(z_i, z_j, y)
                    print(loss)
                    train_losses.append(loss.cpu().detach().numpy())
                    loss.backward()
                    self.optim.step()
                except RuntimeError:
                    num_errs += 1
                    print("error")
                    pass
            
            self.model.eval()
            with torch.no_grad():
                for (xv_i, xv_j), yv in val_loader:
                    try:
                        xv_i, xv_j = xv_i.to(self.device), xv_j.to(self.device)
                        yv = map_labels_to_int(yv)
                        yv = yv.to(self.device)
                        self.optim.zero_grad()
                        hv_i, hv_j, zv_i, zv_j = self.forward(xv_i, xv_j)
                        val_loss = self.supcon_loss(zv_i, zv_j, yv)
                        val_losses.append(val_loss.cpu().detach().numpy())
                    except RuntimeError:
                        num_errs += 1
                        pass

            epoch_end_time = time.time()
            epoch_dur = round(epoch_end_time - epoch_start_time, 2)
            avg_train_loss = sum(train_losses) / len(train_losses)
            avg_val_loss = sum(val_losses) / len(val_losses)
            print('[epoch {0}] [train loss={1}] [val loss={2}] [time elapsed={3}s] [errors:{4}]'.format(epoch, avg_train_loss, avg_val_loss, epoch_dur, num_errs))

            try: os.mkdir('checkpoints')
            except: pass

            try: os.mkdir('checkpoints/simclr')
            except: pass

            checkpt_name = 'checkpoints/simclr/{0}.pth'.format(epoch)
            torch.save(self.model.state_dict(), checkpt_name)

    def test(self, test_loader, batch_size=64, recompute=True):
        if recompute:
            ft_test_loader  = self.create_data_loaders_from_arrays(test_loader, batch_size=batch_size)
            torch.save(ft_test_loader, "data/ft_test.pt")
        else:
            ft_test_loader = torch.load("data/ft_test.pt")
        
        self.model.eval()
        preds = []
        ts = []
        auc = torchmetrics.AUROC(num_classes=2)
        with torch.no_grad():
            total = correct = 0
            for X, t in ft_test_loader:
                X = X.to(self.device)
                z = self.clf_head(X)
                auc.update(z.cpu(), t.cpu())
                for idx, i in enumerate(z.cpu()):
                    preds.append(torch.argmax(i))
                    ts.append(t.cpu()[idx])
        
        CM = confusion_matrix(ts, preds)
        sensitivity = CM[0,0] / (CM[0,0] + CM[0,1])
        specificity = CM[1,1] / (CM[1,1] + CM[1, 0])
        print(classification_report(ts, preds))
        return auc.compute().cpu(), accuracy_score(ts, preds), sensitivity, specificity
