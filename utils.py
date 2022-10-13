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