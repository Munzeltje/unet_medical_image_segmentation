import torchvision.transforms as transforms
from PIL import Image
from .barrett_loader import *
import pathlib

def get_barrett_data(config=None):
    print(pathlib.Path().absolute())
    training_data = BarrettDataTrain(img_dir="../Full_Data_Patched/",
                                image_transform=True, train=False
                               )
    return training_data

def get_barrett_val_data(config=None):
    val_data = BarrettDataVal(img_dir="../Bolero/test_20x/" ,
                                mask_dir="../Bolero/three_five_seven/test_20x_masks/"
                               )
    return val_data
