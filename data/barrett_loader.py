import os, sys

import random
import torch
import matplotlib.pyplot as plt

import numpy  as np
import PIL
from PIL import Image
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch.utils.data as data
from torchvision import datasets
import json

class BarrettDataTrain(data.Dataset):
    def __init__(self, img_dir, mask_dir=None, train=True, image_transform=False):
        self.img_dir = img_dir
        self.data = os.listdir(img_dir)
        
        self.data.sort()  
        # train on the first 580 wsis, keep 197 for test
        # don't forget we also test on bolero
        #self.data = self.data[200:]
        
        # training size is number of WSIs in training set
        print("training set length: ", len(self.data))
        self.image_transform = image_transform
        self.train = train

    def sequential_sample(self, ind):
        if ind >= self.__len__() - 1:
            return self.__getitem__(0)
        return self.__getitem__(ind + 1)

    def skip_sample(self, ind):
        return self.sequential_sample(ind=ind)

    def __getitem__(self, index):
        path = os.path.join(self.img_dir, self.data[index])
        imgs_patches_path = os.path.join(path, "patches")
        masks_patches_path = os.path.join(path, "masks")
        patches = os.listdir(imgs_patches_path)
       
        try:
            chosen_patch = random.choice(patches)
        except:
            # some folders have 0 extracted patches
            return self.skip_sample(index)
            
        img_path = os.path.join(imgs_patches_path, chosen_patch)
        mask_path = os.path.join(masks_patches_path, chosen_patch)
        
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
                
        if self.image_transform:
            img, mask = self.transform(img, mask)
            
        mask = torch.tensor(np.array(mask))
        # ImageNet Encoder Normalization
        img = TF.to_tensor(img)
        img = TF.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        return img, mask

    def transform(self, img, mask):
        """
        Data augmentation pipeline, we could experiment
        with colour and the distortions.
        At the moment, we are not performing colour
        augmentation.
        """
        i, j, h, w = transforms.RandomCrop.get_params(img, output_size=(384,384))
        if random.random() > 0.5:
            img = TF.hflip(img)
            mask = TF.hflip(mask)
        if random.random() > 0.5:
            img = TF.vflip(img)
            mask = TF.vflip(mask)
        if random.random() > 0.5:
            rotation = random.choice([90,180,270])
            img = TF.rotate(img, rotation)
            mask = TF.rotate(mask, rotation)        
        return img, mask

    def __len__(self):
        return len(self.data)
    

class BarrettDataVal(data.Dataset):
    def __init__(self, img_dir, mask_dir=None):
        self.datalist = img_dir
        self.mask_dir = mask_dir
        self.data = os.listdir(self.datalist)
        # training size is number of WSIs in training set
        print("training set length: ", len(self.data))

    def __getitem__(self, index):
        img_path = os.path.join(self.datalist, self.data[index])
        mask_path = os.path.join(self.mask_dir, self.data[index])
        img = Image.open(img_path)
        mask = Image.open(mask_path)
                        
        # ImageNet Encoder Normalization
        img = TF.to_tensor(img)
        img = TF.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        mask = torch.tensor(np.array(mask))

        return img, mask

    def __len__(self):
        return 1600
    
