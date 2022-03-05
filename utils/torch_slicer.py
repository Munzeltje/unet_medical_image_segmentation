from pathlib import Path
from random import randint, choice

from PIL import Image
import argparse
import torch
import json
import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path

from torch.nn import functional as F
import time

# Missing masking patch selection -> choose patches according to mask annotation
# should be a simple if statement, though
def patch_wsi(wsi_id, wsi_path, mask_path, patch_size, stride, save_dir):

    slide = Image.open(wsi_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")
    
    mask = torch.tensor(np.array(mask)).float().unsqueeze(0).unsqueeze(0)
    mask = F.unfold(mask, (patch_size, patch_size), stride=stride)
    mask = mask.view(patch_size, patch_size, -1)
    
    #mask = mask.sum(dim=1)
    #positive_masks = torch.where(mask>5000)

    rgb_img = torch.tensor(np.array(slide)).float().permute(2,0,1).unsqueeze(0)
    rgb_img = F.unfold(rgb_img, (patch_size,patch_size), stride=stride)
    rgb_img = rgb_img.view(3,patch_size,patch_size,-1)
    
    # get the patches where there is something positive
    #rgb_img = rgb_img[0,:,:,:,positive_masks[1]]

    # Get rid of any black squares patches
    valid_patches_mean = rgb_img.view(3,patch_size*patch_size,-1).mean(dim=0)
    valid_patches_mean = torch.any(valid_patches_mean==0, dim=0)
    valid_patches_non_zero = torch.where(torch.logical_not(valid_patches_mean))
    
    valid_patches_non_black = rgb_img[:,:,:,valid_patches_non_zero[0]]
    valid_masks_non_black = mask[:,:,valid_patches_non_zero[0]]

    # get the patches from the positives that have at least some tissue in them
    unfolded_mean = valid_patches_non_black.view(-1,valid_patches_non_black.shape[-1]).mean(dim=0)
    # values chosen empirically, keep patches between 150, 222
    # patches with a mean lower than 150 are too dark, patches with a mean above 222 are too white
    unfolded_threshold = torch.where((unfolded_mean>150) & (unfolded_mean<222))
    # final valid patches we want to save
    valid_patches = valid_patches_non_black[:,:,:,unfolded_threshold[0]]
    valid_masks = valid_masks_non_black[:,:,unfolded_threshold[0]]
        
    for i in range(valid_patches.shape[-1]):
        img = Image.fromarray(valid_patches[:,:,:,i].permute(1,2,0).numpy().astype(np.uint8))
        mask = Image.fromarray(valid_masks[:,:,i].numpy().astype(np.uint8))
        img.save(os.path.join(save_dir, wsi_id, "patches", str(i)+".png"))
        mask.save(os.path.join(save_dir, wsi_id, "masks", str(i)+ ".png"))
       
def main():
    parser = argparse.ArgumentParser(description='Configurations of the parameters for the extraction')
    parser.add_argument('--patch_size', help='desired patch size',type=int, default=512)
    parser.add_argument("--stride", default=384, type=int, help="Specify the stride to overlap")
    parser.add_argument("--root_dir", type=str, help="Specify the directory where the WSIs are stored")
    parser.add_argument("--save_dir", type=str, help="Specify the directory where the patches will be saved along with their mask")
    args = parser.parse_args()
    patch_size = args.patch_size
    save_dir = args.save_dir
    path = args.root_dir
    stride = args.stride
    
    Path(save_dir).mkdir(parents=True, exist_ok=True)
   
    files = os.listdir(path)
    # sort for consistency
    files.sort()
    
    # get the paths
    wsis = [(os.path.join(path, file, "img.png"), os.path.join(path, file, "mask3_5_7.png")) for file in files]

    start_time = time.time()
    for i, wsi in enumerate(wsis):
               
        wsi_path = wsi[0]
        mask_path = wsi[1]
        wsi_id = files[i]
        print(i, wsi_id)
        
        # check if saving directories exist, else make them
        Path(save_dir+"/{}".format(wsi_id)).mkdir(parents=True, exist_ok=True)
        Path(save_dir+"/{}/patches".format(wsi_id)).mkdir(parents=True, exist_ok=True)
        Path(save_dir+"/{}/masks".format(wsi_id)).mkdir(parents=True, exist_ok=True)

        try:
            patch_wsi(wsi_id, wsi_path, mask_path, patch_size, stride, save_dir)
        except:
            print("excepted: ", wsi, i)
            continue
        

    print("Run Time: ", time.time() - start_time)

if __name__ == "__main__":
	main()