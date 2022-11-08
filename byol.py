from pyexpat import model
import torch
import time
import os

import numpy as np
import torch.optim as optim

from byol_pytorch import BYOL
from torchvision.models.resnet import resnet50, resnet101, ResNet50_Weights
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from datasets import HistologyDataset


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
        print(f'[*] Epoch: {epoch} - Avg Train Loss: {avg_train_loss:.3f} - Avg Val Loss: {avg_val_loss:.3f} - Elapsed: {end - start:.2f}')

#resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
#train_byol_model(model_backbone=resnet)