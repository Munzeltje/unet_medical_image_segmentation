Simple (pretrained) unet training code with histopathology

Two pretrained architectures were mainly used: DenseNet-161 encoding Unet and Efficientnets

WSI slicer is in /utils

example run:

 python train.py --batch_size 24 --model_type dense --n_eval 1000
 
 check the utils/config.py for configs
 
 data loader assumes patches for loading
 
 validation is done using DICE(f1)-score
 
 