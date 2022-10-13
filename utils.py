import os
import pandas as pd

def generate_labels(train_path, test_path):
    val_annot = open('data/test_annot.csv', 'w')
    val_annot.write('img,label\n')
    train_annot = open('data/train_annot.csv', 'w')
    train_annot.write('img,label\n')
    for img in os.listdir(test_path):
        label = img[0:3]
        val_annot.write('{0}/{1},{2}\n'.format(test_path, img, label))
    for img in os.listdir(train_path):
        label = img[0:3]
        train_annot.write('{0}/{1},{2}\n'.format(test_path, img, label))
        

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