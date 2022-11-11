import os
import pandas as pd
from torchvision import transforms
import torch
import glob
import random

class StochasticAugmentation:
    def __init__(self, size=(224, 224)):
        self.size = size
        self.aug = transforms.Compose([
            transforms.RandomResizedCrop(size=size),
            transforms.RandomHorizontalFlip(),  # with 0.5 probability
            transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor()
        ])

    def __call__(self, X):
        return self.aug(X), self.aug(X)

def generate_labels(train_path, test_path):
    val_annot = open('data/test_annot.csv', 'w')
    val_annot.write('img,label\n')
    train_annot = open('data/train_annot.csv', 'w')
    train_annot.write('img,label\n')
    for img in os.listdir(test_path):
        label = img[0:3]
        val_annot.write('{0},{1}\n'.format(img, label))
    for img in os.listdir(train_path):
        label = img[0:3]
        train_annot.write('{0},{1}\n'.format(img, label))
        
def map_labels_to_int(y, dtype='long', cancer_clf=True):
    encoding = {
        "ADI": 0,
        "BACK": 1,
        "BAC": 1,
        "DEB": 2,
        "LYM": 3,
        "MUC": 4,
        "MUS": 5,
        "NORM": 6, 
        "NOR": 6,
        "STR": 7,
        "TUM": 8
    }

    cancer = {
        "ADI": 0, "BAC": 0, "DEB": 0, "LYM": 0, 
        "MUC": 0, "MUS": 0, "NOR": 0, "STR": 1, "TUM": 1
    }

    if cancer_clf:
        return torch.LongTensor([cancer[yy] for yy in y])

    if dtype == 'long':
        return torch.LongTensor([encoding[yy] for yy in y])
    else:
        return torch.Tensor([encoding[yy] for yy in y])

def subset_data(data_dir_from, data_dir_to, subset_size=4000):
    for label in ['ADI', 'BACK', 'DEB', 'LYM', 'MUC', 'MUS', 'NORM', 'STR', 'TUM']:
        items = glob.glob(data_dir_from + '/{0}-*.tif'.format(label))
        items = [item.split('/')[-1] for item in items]
        i = 0
        while i < subset_size:
            item = random.choice(items)
            try:
                os.rename("{0}/{1}".format(data_dir_from, item), 
                        "{0}/{1}".format(data_dir_to, item))
                i += 1
            except:
                continue
    return




def generate_annotation_files():
    """
    This function generates the annotation files for the
    NCT-CRC-HE-100K-NONORM dataset. The annotation of each image
    in this dataset is the same as its parent directory.
    Furthermore, it is assumed that the dataset lies in the
    data/NCT-CRC-HE-100K-NONORM directory.
    The function saves the annotation files in the 'data/NCT-CRC-HE-100K-NONORM'
    directory as annotations.csv
    """
    
    annotations = {'name': [],
                   'class': []}
    
    # getting the directory names (annotations)
    dir_names = os.listdir('data/NCT-CRC-HE-100K-NONORM')
    
    for dir_name in dir_names:
        image_files = os.listdir(f'data/NCT-CRC-HE-100K-NONORM/{dir_name}')
        
        annotations['name'].extend(image_files)
        annotations['class'].extend([dir_name] * len(image_files))
        
    annotations = pd.DataFrame(annotations)
    annotations.to_csv('data/NCT-CRC-HE-100K-NONORM/annotations.csv')
    
    

if __name__ == '__main__':
    generate_annotation_files()