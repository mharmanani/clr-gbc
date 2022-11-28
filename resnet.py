import os
import numpy as np
import pandas as pd
import torch
from torch import nn
from torchvision.io import read_image
from torchmetrics import AUROC
from sklearn.metrics import accuracy_score, confusion_matrix

from utils import map_labels_to_int

class ResNetWrapper(nn.Module):
    def __init__(self, resnet, num_classes=2, device='cuda', num_epochs=30):
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
        self.num_epochs = num_epochs

    def forward(self, X):
        return self.clf(X)

    def finetune(self, train_loader, val_loader, learning_rate=1e-3):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.clf.parameters(), lr=learning_rate)

        # Arrays and variables for plotting metrics
        train_acc, val_acc = 0, 0
        epochs_arr = []
        train_losses = []
        train_accs = []
        val_losses = []
        val_accs = []

        for epoch in range(1+self.num_epochs):
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

            checkpt_name = 'checkpoints/resnet/{0}.pth'.format(epoch)
            torch.save(self.clf.state_dict(), checkpt_name)

            print('[Epoch {0}] [train loss={1}] [train acc={2}] [val loss={3}] [val acc={4}]'
                .format(epoch, train_loss, round(train_acc, 3), val_loss, round(val_acc, 3)))

    def test(self, test_loader):
        self.clf.eval()
        with torch.no_grad():
            total = correct = 0
            preds = []
            pred_logits = []
            ys = []
            auc = AUROC(pos_label=1, num_classes=2)
            for batch in test_loader:
                X, y = batch
                y = map_labels_to_int(y, dtype='long')
                X = X.to(self.device)
                y = y.to(self.device)
                z = self.clf(X)
                auc.update(z, y)
                for idx, i in enumerate(z.cpu()):
                    preds.append(torch.argmax(i))
                    pred_logits.append(i)
                    ys.append(y.cpu()[idx])

        CM = confusion_matrix(ys, preds)
        sensitivity = CM[0,0] / (CM[0,0] + CM[0,1])
        specificity = CM[1,1] / (CM[1,1] + CM[1, 0])
        auc_value = auc.compute().cpu()
        acc = accuracy_score(ys, preds)
        print("AUC: {0}".format(auc_value))
        print("Accuracy: {0}".format(acc))
        print("Sensitivity: {0}".format(sensitivity))
        print("Specificity: {0}".format(specificity))
        return auc_value, acc, sensitivity, specificity
        