import os
from utils import subset_data

def main():
    subset_data('/mnt/c/Users/moham/OneDrive/Documents/clr-gbc/data/all_train_imgs', 
                '/mnt/c/Users/moham/OneDrive/Documents/clr-gbc/data/train')

def make_subset_from_list(list_dir, list_file, dir_from, dir_to):
    """
    :list_dir:
        The location of the file containing the img names.
    :list_file:
        The file containing the img names.
    :dir_from:
        The original folder containing ALL the training images.
    :dir_to:
        The folder to contain the eventual subsetted data 
    """
    fhand = open(list_dir+list_file) # read the file with img names
    for imgname in fhand.readlines(): # get each image name
        imgname = imgname.strip()
        os.rename(dir_from+imgname, dir_to+imgname)
    

if __name__ == '__main__':
    # TODO 1 - Make sure all directories exist
    # TODO 2 - Move train_files (list of images) into 'data/' directory
    # TODO 3 - Make sure the subfolders (data/train and data/all_train_imgs) are named or renamed appropriately
    make_subset_from_list('data/', 'train_files.txt', 'data/all_train_imgs/', 'data/train/')