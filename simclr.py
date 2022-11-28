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

class SimCLR():
    """ SimCLR Model class """
    def __init__(self, model_backbone, batch_size, num_epochs, num_views=2, project_dim=128,
                 num_classes=2, learning_rate=3e-2, temperature=0.5, device='cuda'):
        self.device = device 
        self.backbone = model_backbone.to(self.device) # initialize model backbone
        self.backbone.fc = nn.Identity() # remove the backbone's classification head
        
        # initialize projection head g()
        self.projection_head = nn.Sequential(
            nn.Linear(512, 1000, bias=False),
            nn.ReLU(),
            nn.Linear(1000, project_dim, bias=False)
        ).to(self.device) 

        # create the SimCLR model g(f())
        self.model = nn.Sequential(
            self.backbone,
            self.projection_head
        ).to(self.device)

        # add a classification head for downstream tasks
        self.clf_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLu(),
            nn.Linear(128, num_classes),
            nn.Softmax(dim=1)
        ).to(self.device)
        
        # define hyperparameters
        self.learning_rate = learning_rate
        self.optim = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=0.00001)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.views = num_views # number of augmented view of one sample
        self.temperature = temperature
    
    def forward(self, x_i, x_j):
        """
        Compute the forward pass of SimCLR.
        
        :x_i, x_j:
            The augmented views of some input X
        """
        h_i = self.backbone(x_i)
        h_j = self.backbone(x_j)

        z_i = self.projection_head(h_i)
        z_j = self.projection_head(h_j)
        return h_i, h_j, z_i, z_j

    def build_feature_label_arrays(self, loader):
        """
        Compute the learned representations of the backbone model.

        :loader:
            The data loader containing the images to extract representations from.
        """
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
        """
        Helper function to convert NumPy arrays containing SimCLR features
        into PyTorch data loaders.

        :loader:
            The data loader containing the images to extract representations from.
        :batch_size:
            Parameter to control the newly created data loader's batch size.
        """
        X, y = self.build_feature_label_arrays(loader)
        dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(X), torch.from_numpy(y)
        )
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False
        )
        return data_loader

    def train_clf_head(self, train_loader, val_loader, num_epochs=30):
        """
        Train a shallow classifier on the learned SimCLR representations computed by the
        backbone.

        :train_loader:
            Training dataset without stochastic augmentation.
        :val_loader:
            Validation dataset without stochastic augmentation.
        """
        ft_train_loader  = self.create_data_loaders_from_arrays(train_loader, 64)
        ft_val_loader = self.create_data_loaders_from_arrays(val_loader, 64)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.clf_head.parameters(), lr=3e-3, weight_decay=1e-5)

        self.clf_head.train()
        for epoch in range(num_epochs):
            losses = []
            val_losses = []
            accs = []
            val_accs = []
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

            self.clf_head.eval()
            with torch.no_grad():
                for (Hv, tv) in ft_val_loader:
                    Hv = Hv.to(self.device)
                    tv = tv.to(self.device)
                    yv = self.clf_head(Hv)
                    val_loss = criterion(yv, tv)

                    val_acc = accuracy_score(torch.argmax(yv, axis=1).cpu().detach(), tv.cpu().detach())

                    val_losses.append(val_loss.cpu().detach().numpy())
                    val_accs.append(val_acc)

            if epoch % 10 == 0:
                print("[epoch {0}] [loss={1}] [val loss={2}] [acc={3}] [val acc={4}]".format(epoch,
                    sum(losses) / len(losses), sum(val_losses) / len(val_losses), sum(accs) / len(accs), sum(val_accs) / len(val_accs)
                ))

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def NT_Xent(self, z_i, z_j):
        """
        The normalized temperature-scaled cross entropy function used to train
        the SimCLR model's backbone. 

        :z_i:
            The projection output of the first augmented view of the training image.
        :z_i:
            The projection output of the first augmented view of the training image.
        """
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
        """
        Train the backbone model and projection head using the contrastive framework.

        :train_loader:
            The training dataset containing stochastically augmented images.
        :val_loader:
            The validation dataset containing stochastically augmented images.
        """
        for epoch in range(self.num_epochs):
            epoch_start_time = time.time()
            self.model.train()
            num_errs = 0
            train_losses = []
            val_losses = []
            for (x_i, x_j), _ in train_loader:
                try:
                    x_i, x_j = x_i.to(self.device), x_j.to(self.device)
                    self.optim.zero_grad()
                    h_i, h_j, z_i, z_j = self.forward(x_i, x_j)
                    loss = self.NT_Xent(z_i, z_j)
                    train_losses.append(loss.cpu().detach().numpy())
                    loss.backward()
                    self.optim.step()
                except RuntimeError:
                    # problematic images cause PyTorch numel to overflow
                    num_errs += 1 
                    pass
            
            self.model.eval()
            with torch.no_grad():
                for (xv_i, xv_j), _ in val_loader:
                    try:
                        xv_i, xv_j = xv_i.to(self.device), xv_j.to(self.device)
                        self.optim.zero_grad()
                        hv_i, hv_j, zv_i, zv_j = self.forward(xv_i, xv_j)
                        val_loss = self.NT_Xent(zv_i, zv_j)
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

    def test(self, test_loader, batch_size=64):
        """
        Test the performance of the classification head on the representations
        computed by the SimCLR backbone.

        :test_loader:
            A test dataset containing the computed representations of SimCLR when
            run on the original test set.
        """
        ft_test_loader  = self.create_data_loaders_from_arrays(test_loader, batch_size=batch_size)
        
        self.model.eval()
        self.clf_head.eval()
        preds = []
        ts = []
        auc = torchmetrics.AUROC(num_classes=2)
        with torch.no_grad():
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
        auc_value = auc.compute().cpu()
        acc = accuracy_score(ts, preds)
        print(classification_report(ts, preds))
        print("AUC: {0}".format(auc_value))
        print("Accuracy: {0}".format(acc))
        print("Sensitivity: {0}".format(sensitivity))
        print("Specificity: {0}".format(specificity))
        return auc_value, acc, sensitivity, specificity