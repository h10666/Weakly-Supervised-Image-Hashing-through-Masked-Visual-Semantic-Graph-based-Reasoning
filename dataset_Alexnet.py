from __future__ import print_function
from PIL import Image, PILLOW_VERSION
import numpy as np
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle
import torch
import torch.utils.data as data


class dataset(data.Dataset):
    def __init__(self, dir_path,img_lab,transform_pre=None, transform=None, target_transform=None, matrix_transform=None):
        
        self.transform_pre = transform_pre
        self.transform = transform
        self.target_transform = target_transform
        self.list = img_lab
        self.dir_path = dir_path
        

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        # dir_path = '/home/lujin/dataset/mirflickr25k/mirflickr/'
        fn, lb = self.list[index]
        # print(fn)
        img = Image.open(self.dir_path+fn).convert('RGB')
        if self.transform_pre is not None:
            img = self.transform_pre(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            lb = self.target_transform(lb)
        return img, lb


    def __len__(self):
        return len(self.list)
    
                        