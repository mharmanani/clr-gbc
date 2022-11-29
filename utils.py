import os
import pandas as pd
from torchvision import transforms
import torch
import glob
import random

class StochasticAugmentation:
    def __init__(self, size=(96, 96)):
        self.size = size
        self.aug = transforms.Compose([
            transforms.RandomResizedCrop(size=size),
            transforms.RandomHorizontalFlip(),  # with 0.5 probability
            transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
        ])

    def __call__(self, X):
        return self.aug(X.convert("RGB")), self.aug(X.convert("RGB"))

class TestTransform:
    def __init__(self, size=(96, 96)):
        self.size = size
        self.tr = transforms.Compose([
            transforms.Resize(size=size),
            transforms.ToTensor()
        ])

    def __call__(self, X):
        return self.tr(X.convert("RGB"))

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
        class_size = subset_size if label not in ['STR', 'TUM'] else 3*subset_size
        while i < class_size:
            item = random.choice(items)
            print(item)
            try:
                os.rename("{0}/{1}".format(data_dir_from, item), 
                        "{0}/{1}".format(data_dir_to, item))
                i += 1
            except:
                print('error')
                continue
    return



"""
@author: Sana Arastehfar
"""
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
    
    print('[*] Creating the backup directories (if does not already exist)...')
    os.makedirs('data/NCT-CRC-HE-100K-NONORM-MODIFIED', exist_ok=True)
    os.makedirs('data/NCT-CRC-HE-100K-NONORM-MODIFIED/Train', exist_ok=True)
    
    for i in tqdm(range(len(dir_names)), desc='Preparing the Training Files'):
        dir_name = dir_names[i]
        image_files = os.listdir(f'data/NCT-CRC-HE-100K-NONORM/{dir_name}')
        
        annotations['name'].extend(image_files)
        annotations['class'].extend([dir_name] * len(image_files))
        
        for image_file in image_files:
            shutil.copy(f'data/NCT-CRC-HE-100K-NONORM/{dir_name}/{image_file}', 
                        f'data/NCT-CRC-HE-100K-NONORM-MODIFIED/Train/{image_file}')
        
    annotations = pd.DataFrame(annotations)
    annotations.to_csv('data/NCT-CRC-HE-100K-NONORM-MODIFIED/annotations.csv', index=False)
    

def sift_training_files():
    # loading the training samples from the file
    with open('train_files.txt', 'r') as handle:
        training_samples = handle.readlines()
    
    training_samples = list(map(lambda x: x.strip(), training_samples))
    
    # cleanining the annotation file
    annotations = pd.read_csv('data/NCT-CRC-HE-100K-NONORM-MODIFIED/annotations.csv')
    train_annotations = annotations[annotations['name'].isin(training_samples)]
    test_annotations = annotations[~annotations['name'].isin(training_samples)]
    train_annotations.to_csv('data/NCT-CRC-HE-100K-NONORM-MODIFIED/train_annotations.csv', index=False)
    test_annotations.to_csv('data/NCT-CRC-HE-100K-NONORM-MODIFIED/test_annotations.csv', index=False)
    
    
    # removing the redundant files
    existing_samples = os.listdir('data/NCT-CRC-HE-100K-NONORM-MODIFIED/Train/')
    os.makedirs('data//NCT-CRC-HE-100K-NONORM-MODIFIED/Test/', exist_ok=True)
    for file_name in existing_samples:
        if file_name not in training_samples:
            shutil.move(f'data/NCT-CRC-HE-100K-NONORM-MODIFIED/Train/{file_name}', f'data/NCT-CRC-HE-100K-NONORM-MODIFIED/Test/{file_name}')


def make_annotations_binary():
    warnings.filterwarnings('ignore')
    # loading the annotation file
    train_annotations = pd.read_csv('data/NCT-CRC-HE-100K-NONORM-MODIFIED/train_annotations.csv')
    test_annotations = pd.read_csv('data/NCT-CRC-HE-100K-NONORM-MODIFIED/test_annotations.csv')
    new_annotations = []
    for index in range(len(train_annotations)):
        name = train_annotations['name'].iloc[index]
        if name.startswith('STR') or name.startswith('TUM'):
            new_annotations.append(1)
        else:
            new_annotations.append(0)
            
    train_annotations = train_annotations.drop(columns=['class'])
    train_annotations['class'] = new_annotations
    
    train_annotations.to_csv('data/NCT-CRC-HE-100K-NONORM-MODIFIED/train_annotations_binary.csv', index=False)
    
    new_annotations = []
    for index in range(len(test_annotations)):
        name = test_annotations['name'].iloc[index]
        if name.startswith('STR') or name.startswith('TUM'):
            new_annotations.append(1)
        else:
            new_annotations.append(0)
            
    test_annotations = test_annotations.drop(columns=['class'])
    test_annotations['class'] = new_annotations
    
    test_annotations.to_csv('data/NCT-CRC-HE-100K-NONORM-MODIFIED/test_annotations_binary.csv', index=False)
    print('[*] Binary annotations generated')

if __name__ == '__main__':
    generate_annotation_files()
    sift_training_files()
    make_annotations_binary()
