import numpy as np
# Torch dependencies
import torch
import torch.nn as nn
from torch import optim
# Our own written dependencies
from metrics import Metric
#from efficientunet import *
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from DenseUnet import DenseUnet
from data.load_data import *
from scheduler import CycleScheduler
from utils.config import unet_config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_net(net, training_loader, val_loader, config, save_cp=True, model_name="model"):
    print("Training: ", model_name)

    # The direcotry where we would like to save our trained models
    dir_checkpoint = '../Models/'
    #scheduler = CycleScheduler(optimizer, lr, n_iter=N_train, momentum=Non)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=config.lr)

    # The weights are based on the proportion of each class
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1,1,2,2]).float().cuda())
    metric = Metric(device=device)
    seg_l = []
    for i in range(config.iterations):
        imgs, true_masks = next(iter(training_loader))
        imgs = imgs.to(device)
        true_masks = true_masks.to(device).long().squeeze(1)

        masks_probs = net(imgs)
        loss = criterion(masks_probs, true_masks)
        seg_l.append(loss.item())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        #print('{} --- loss: {}'.format(i, np.mean(total_l[-100:])))
        if (i+1) % config.n_eval==0:
            print("[==========================]")
            print("Current Iteration: {}".format(i+1))
            print('Segmentation Loss: %.4f' % np.mean(seg_l[-100:]))
            print("Learning Rate: ", optimizer.param_groups[0]['lr'])
            if config.validate:
                dice = test_net(net, val_loader)
                print("Dice Score: ", dice)
                save_model(net,dir_checkpoint + model_name + str(dice) + '_{}.pth'.format(i), save_model=save_cp)
                print("checkpoint saved")

            net.train()

    save_model(net,dir_checkpoint + model_name, save_model=save_cp)

def save_model(model, save_path, save_model=False):
    if save_model:
        torch.save(model.state_dict(), save_path)
        print('Checkpoint saved in {}!'.format(save_path))

def test_net(net,val_loader):
    print("Starting Validation")
    # The device we are using, naturally, the gpu by default
    metric = Metric(device=device)
    # Keep track of the training loss, we need to save this later on
    val_accuracy, val_dice = metric.evaluate(net,val_loader)
    print('Validation Dice: {0:.4g}  [===] Validation Accuracy: {1:.4g}'.format(val_dice, val_accuracy))
    return round(val_dice, 3)

if __name__ == '__main__':
    # Get the arguments given through the terminal
    config = unet_config()

    if "efficientnet" in config.model_type:
        net = smp.Unet(config.model_type, classes=config.n_classes, encoder_weights="imagenet")
    else:
        net = DenseUnet(num_filters=config.n_filters,
                                  num_classes=config.n_classes,
                                  is_deconv=False, pretrained=True)
        
    # If we would like to load a trained model
    print("GPU devices: ", torch.cuda.device_count())

    training_data = get_barrett_data(config)

    training_loader = DataLoader(training_data,
                             batch_size=config.batch_size,
                             shuffle=True,
                             drop_last=False)

    val_data = get_barrett_val_data(config)

    val_loader = DataLoader(val_data,
                             batch_size=16,
                             shuffle=True,
                             drop_last=False)

    net.train()
    net = nn.DataParallel(net)
    if config.model_checkpoint is not None:
        net.load_state_dict(torch.load(config.model_checkpoint))
        print("Model Loaded")

    net.to(device)
    #cudnn.benchmark = True # faster convolutions, but more memory
    train_net(net, training_loader, val_loader, config, save_cp=True, model_name=config.model_name)

    print("Done Training")

