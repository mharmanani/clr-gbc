from __future__ import annotations
import os
import numpy as np
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


from utils import generate_labels

TRAIN_PATH = 'data/train'
TEST_PATH = 'data/test'
TRAIN_ANNOT = 'data/test_annot.csv'
TEST_ANNOT = 'data/train_annot.csv'

class HistologyDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

# Create the labels file
generate_labels(TRAIN_PATH, TEST_PATH)

# Hyperparameters for batch GD and train/val split
batch_size = 64
validation_split = .2
shuffle_dataset = True
random_seed= 42

# Creating data indices for training and validation splits:
dataset_size = len(os.listdir(TRAIN_PATH))
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
train_dataset = HistologyDataset(annotations_file=TRAIN_ANNOT, img_dir=TRAIN_PATH)
train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)

# Create the validation data loader
val_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=val_sampler)

# Create the test data loader
test_dataset = HistologyDataset(annotations_file=TEST_ANNOT, img_dir=TEST_PATH)
test_loader = DataLoader(test_dataset)