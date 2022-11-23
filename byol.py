import torch
import time
import os

import numpy as np
import torch.optim as optim
import torch.nn as nn

from byol_pytorch import BYOL
from torchvision.models.resnet import resnet50, resnet101, ResNet50_Weights
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from datasets import HistologyDataset, build_datasets
from utils import map_labels_to_int
from sklearn.metrics import accuracy_score

def train_byol_classifier(backbone, train_loader, val_loader, num_epochs=100, load_weights=False, from_epoch=0, device='cuda'):
    model = BYOL(
        backbone,
        image_size= 224,
        hidden_layer='avgpool'
    ).to(device)

    classifier = nn.Sequential(
        nn.Linear(2048, 1000),
        nn.ReLU(),
        nn.Linear(1000, 2),
        nn.Softmax(dim=1)
    ).to(device)

    if load_weights:
        print('load weights from epoch {0}'.format(from_epoch))
        model.load_state_dict(torch.load('checkpoints/byol/{0}.ckpt'.format(from_epoch)))
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters())

    for epoch in range(num_epochs):
        losses = []
        accs = []
        for (X, t) in train_loader:
            X = X.to(device)
            t = map_labels_to_int(t).to(device)
            h, z = model(X, return_embedding=True)
            optimizer.zero_grad()
            
            y = classifier(z)
            loss = criterion(y, t)
            
            train_acc = accuracy_score(torch.argmax(y, axis=1).to('cpu').detach(), t.to('cpu').detach())

            loss.backward()
            optimizer.step()

            losses.append(loss.cpu().detach().numpy())
            accs.append(train_acc)

        print(f"Epoch [{epoch}/{num_epochs}]\t Loss: {sum(losses) / len(losses)}\t Accuracy: {sum(accs) / len(accs)}")

    return classifier


def train_byol_model(model_backbone, epochs=10, batch_size=8, validation_split=0.2, random_seed=42, shuffle_dataset=True):
    # preparing the dataset
    train_path = 'data/train'
    train_annot = 'data/train_annot.csv'

    # Creating data indices for training and validation splits:
    dataset_size = len(os.listdir(train_path))
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    # Create the train data loader
    train_dataset = HistologyDataset(annotations_file=train_annot, img_dir=train_path, transform=transforms.ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)

    # Create the validation data loader
    val_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=val_sampler)

    # use GPU if available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f' Running on {device}')

    # defining the model
    model = BYOL(
        model_backbone,
        image_size= 224,
        hidden_layer='avgpool'
    ).to(device)

    # defnining the optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # training the model
    for epoch in range(epochs):
        start = time.time()
        avg_train_loss = 0.
        for image, _ in train_loader:
            image = image.to(device)

            optimizer.zero_grad()
            
            loss = model(image)
            
            loss.backward()
            optimizer.step()
            
            model.update_moving_average()
            
            avg_train_loss += loss.item()
            
        avg_train_loss = avg_train_loss / len(train_loader)
        avg_val_loss = 0

        # validating the model
        with torch.inference_mode():
            for sample in val_loader:
                image = sample[0].to(device)
                
                loss = model(image)
                
                avg_val_loss += loss.item()
                
            avg_val_loss = avg_val_loss / len(val_loader)
        end = time.time()

        torch.save(model.state_dict(), 'checkpoints/byol/{0}.ckpt'.format(epoch))
        print(f'[*] Epoch: {epoch} - Avg Train Loss: {avg_train_loss:.3f} - Avg Val Loss: {avg_val_loss:.3f} - Elapsed: {end - start:.2f}')

    return model

def test_byol_model(model_backbone, batch_size=128, load_weights=False, from_epoch=0, device='cuda'):
    train_loader, val_loader, test_loader = build_datasets(batch_size=batch_size, augment_views=True)
    classifier = train_byol_classifier(backbone=model_backbone, train_loader=train_loader, val_loader=val_loader, load_weights=load_weights, 
                                      from_epoch=from_epoch)
    model = BYOL(
        model_backbone,
        image_size= 224,
        hidden_layer='avgpool'
    ).to(device)

    if load_weights:
        model.load_state_dict(torch.load('checkpoints/byol/{0}.ckpt'.format(from_epoch)))

    model.eval()
    with torch.no_grad():
        total = correct = 0
        for batch in test_loader:
            X, y = batch
            y = map_labels_to_int(y, dtype='long')
            X = X.to(device)
            y = y.to(device)
            z = torch.argmax(classifier(X), dim=1)
            print(z.shape)
            print(y)
            for idx, i in enumerate(z):
                if torch.argmax(i) == y[idx]:
                    correct +=1
                total +=1
        test_acc = correct / total

    return test_acc
