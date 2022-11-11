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
from sklearn.metrics import accuracy_score

from utils import map_labels_to_int

class SimCLR():
    def __init__(self, model_backbone, batch_size, num_epochs, num_views=2, project_dim=64,
                 num_classes=2, temperature=0.5, device='cuda'):
        self.device = device 
        self.model = model_backbone.to(self.device)
        self.optim = torch.optim.Adam(self.model.parameters())
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

        self.clf_head = nn.Sequential(
            nn.Linear(1000, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
            nn.Softmax(dim=1)
        ).to(self.device)
    
    def forward(self, x_i, x_j):
        h_i = self.model(x_i)
        h_j = self.model(x_j)

        z_i = self.projection_head(h_i)
        z_j = self.projection_head(h_j)
        return h_i, h_j, z_i, z_j

    def build_feature_label_arrays(self, loader):
        feature_vector = []
        labels_vector = []
        for i, (X, y) in enumerate(loader):
            X = X.to(self.device)
            y = map_labels_to_int(y, dtype='long')

            # get encoding
            with torch.no_grad():
                h, _, z, _ = self.forward(X, X)

            h = h.detach()

            feature_vector.extend(h.cpu().detach().numpy())
            labels_vector.extend(y.detach().numpy())

            if i % 20 == 0:
                print(f"Step [{i}/{len(loader)}]\t Computing features...")
        
        feature_vector = np.array(feature_vector)
        labels_vector = np.array(labels_vector)
        return feature_vector, labels_vector

    def create_data_loaders_from_arrays(self, loader, batch_size):
        X, y = self.build_feature_label_arrays(loader)
        dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(X), torch.from_numpy(y)
        )
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False
        )
        return data_loader

    def train_clf_head(self, train_loader, num_epochs=30):
        ft_train_loader  = self.create_data_loaders_from_arrays(train_loader, 64)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.clf_head.parameters())

        for epoch in range(num_epochs):
            losses = []
            accs = []
            for (H, t) in ft_train_loader:
                optimizer.zero_grad()
                H = H.to(self.device)
                t = t.to(self.device)
                y = self.clf_head(H)
                loss = criterion(y, t)
                
                train_acc = accuracy_score(torch.argmax(y, axis=1).to('cpu').detach(), t.to('cpu').detach())

                loss.backward()
                optimizer.step()

                losses.append(loss.cpu().detach().numpy())
                accs.append(train_acc)

            print(f"Epoch [{epoch}/{num_epochs}]\t Loss: {sum(losses) / len(losses)}\t Accuracy: {sum(accs) / len(accs)}")

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

    def train(self, train_loader, val_loader):
        for epoch in range(self.num_epochs):
            epoch_start_time = time.time()
            self.model.train()
            num_errs = 0
            for (x_i, x_j), _ in train_loader:
                try:
                    x_i, x_j = x_i.to(self.device), x_j.to(self.device)
                    self.optim.zero_grad()
                    h_i, h_j, z_i, z_j = self.forward(x_i, x_j)
                    loss = self.NT_Xent(z_i, z_j)
                    loss.backward()
                    self.optim.step()
                except RuntimeError:
                    num_errs += 1
                    pass

            epoch_end_time = time.time()
            epoch_dur = round(epoch_end_time - epoch_start_time, 2)
            print('[epoch {0}] [train loss={1}] [time elapsed={2}s] [errors:{3}]'.format(epoch, loss, epoch_dur, num_errs))

            try: os.mkdir('checkpoints')
            except: pass

            try: os.mkdir('checkpoints/simclr')
            except: pass

            checkpt_name = 'checkpoints/simclr/{0}.pth'.format(epoch)
            torch.save(self.model.state_dict(), checkpt_name)

    def test(self, test_loader, batch_size=64):
        ft_test_loader  = self.create_data_loaders_from_arrays(test_loader, batch_size=batch_size)
        self.model.eval()
        with torch.no_grad():
            total = correct = 0
            for X, t in ft_test_loader:
                X = X.to(self.device)
                z = self.clf_head(X)
                y = torch.argmax(z, axis=1)
                for i in range(y.shape[0]):
                    if y.cpu()[i] == t.cpu()[i]:
                        correct +=1
                    total +=1
        return round(correct / total, 3)
                
